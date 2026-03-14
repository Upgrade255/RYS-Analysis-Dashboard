"""
adaptive_trainer.py
===================
Advanced training systems built on top of the RYS neuroanatomy framework.

Modules
-------
1. GroundUpTrainer
   Train a model from scratch given an architecture config.
   Supports the same dataset sources as ModelTrainer.

2. LayerAwareTrainer  (Idea 1)
   Categorises each training sample by cognitive type (math, language, code,
   reasoning, etc.) and routes gradients selectively:
   - Only layers relevant to the current sample type receive gradient updates.
   - Periodically runs RYS-style probes to detect emerging specialisation.
   - When a layer range crosses a specialisation threshold, it triggers the
     LayerSurgeon to duplicate the region and reassign routing.
   Expected effect: more defined functional circuits, cleaner pruning targets.

3. RYSAdaptiveLR  (Idea 3)
   Uses the RYS sweep heatmap as a per-layer importance signal.
   Maps delta scores → per-layer learning rates via a configurable curve:
     high positive delta  →  higher LR  (load-bearing reasoning circuits)
     flat / negative      →  lower LR   (translation layers, already stable)
   Can be recomputed periodically during training.

4. StretchDistillTrainer  (Idea 2)
   Phase 1 — Stretch:  insert blank layers into a small model via LayerSurgeon.
   Phase 2 — Compartmentalise:  train with LayerAwareTrainer to force
             functional specialisation into named regions.
   Phase 3 — Distill:  train a fresh small model against the stretched teacher
             using both output KD (KL on logits) AND layer-targeted activation
             matching on identified functional regions.
   Difference from standard KD: the teacher has *structurally organised*
   internals, so the student learns where to put reasoning, not just what
   the answer is.

All trainers share the same PairDataset / DataLoader infrastructure from
model_trainer.py and emit log_callback strings compatible with the Gradio UI.
"""

from __future__ import annotations

import os
import json
import math
import copy
import time
import random
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Re-use shared utilities from model_trainer
from model_trainer import (
    PairDataset, _collate, load_dataset_from_source,
    _get_layer_stack, freeze_all, unfreeze_layer, count_trainable,
    inject_lora, merge_lora_weights, LoRALinear,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sample type classifier
# ─────────────────────────────────────────────────────────────────────────────

# Keyword sets for each cognitive category.
# These are deliberately broad; a trained classifier would do better, but
# this is sufficient for bootstrapping and is zero-overhead.
SAMPLE_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "math": [
        "calculate", "compute", "solve", "equation", "integral", "derivative",
        "probability", "algebra", "geometry", "arithmetic", "sum", "product",
        "prime", "factor", "matrix", "vector", "sqrt", "cube root", "modulo",
        "percentage", "ratio", "formula", "theorem", "proof", "digit",
        "multiply", "divide", "add", "subtract", "fraction", "decimal",
    ],
    "code": [
        "function", "def ", "class ", "import ", "return", "variable", "loop",
        "array", "algorithm", "python", "javascript", "c++", "rust", "sql",
        "debug", "error", "compile", "runtime", "recursion", "sort", "search",
        "data structure", "api", "json", "xml", "html", "css", "regex",
        "list comprehension", "lambda", "async", "thread",
    ],
    "reasoning": [
        "therefore", "because", "implies", "conclude", "deduce", "infer",
        "hypothesis", "argument", "premise", "logical", "if then", "cause",
        "effect", "evidence", "contradiction", "assume", "given that",
        "it follows", "counterexample", "fallacy", "valid", "sound",
    ],
    "language": [
        "translate", "summarise", "summarize", "paraphrase", "grammar",
        "sentence", "vocabulary", "synonym", "antonym", "definition",
        "meaning", "word", "phrase", "story", "poem", "essay", "write",
        "describe", "explain", "language",
    ],
    "factual": [
        "who is", "what is", "when did", "where is", "history", "biography",
        "capital of", "founded", "invented", "discovered", "year", "century",
        "country", "president", "king", "born", "died",
    ],
}

ALL_TYPES = list(SAMPLE_TYPE_KEYWORDS.keys())


def classify_sample(text: str) -> str:
    """
    Return the cognitive category of a text sample via keyword matching.
    Returns the category with the most keyword hits, or "general" if tied/empty.
    """
    text_lower = text.lower()
    scores = {t: 0 for t in ALL_TYPES}
    for stype, keywords in SAMPLE_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[stype] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ─────────────────────────────────────────────────────────────────────────────
# Tagged dataset
# ─────────────────────────────────────────────────────────────────────────────

class TaggedPairDataset(Dataset):
    """
    PairDataset extended with per-sample cognitive type tags.
    Used by LayerAwareTrainer for gradient routing.
    """

    def __init__(self, pairs: list, tokenizer, max_length: int = 512):
        self.samples = []
        type_counts: Dict[str, int] = {}
        for prompt, response in pairs:
            text = prompt + " " + response
            enc  = tokenizer(text, truncation=True, max_length=max_length,
                             return_tensors="pt")
            prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
            labels = enc.input_ids.clone()
            labels[0, :prompt_len] = -100
            sample_type = classify_sample(prompt)
            type_counts[sample_type] = type_counts.get(sample_type, 0) + 1
            self.samples.append({
                "input_ids":      enc.input_ids[0],
                "attention_mask": enc.attention_mask[0],
                "labels":         labels[0],
                "sample_type":    sample_type,
            })
        print(f"[TaggedDataset] {len(self.samples)} samples: {type_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _tagged_collate(batch):
    types = [b["sample_type"] for b in batch]
    tensor_batch = {k: v for k, v in _collate(
        [{k: v for k, v in b.items() if k != "sample_type"} for b in batch]
    ).items()}
    tensor_batch["sample_types"] = types
    return tensor_batch


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer gradient hooks
# ─────────────────────────────────────────────────────────────────────────────

class GradientRouter:
    """
    Registers hooks on model layers to zero out gradients for layers that
    should NOT be trained for the current sample type.

    Usage:
        router = GradientRouter(model)
        router.set_active_layers({0,1,2,7,8,9})   # call before each backward
        loss.backward()
        # layers not in active set have their grads zeroed automatically
        router.clear()   # remove hooks when done
    """

    def __init__(self, model):
        self.model  = model
        self.layers = _get_layer_stack(model)
        self._hooks: list = []
        self._active: set = set(range(len(self.layers)))  # all active by default

    def set_active_layers(self, active_set: set):
        self._active = active_set

    def activate_all(self):
        self._active = set(range(len(self.layers)))

    def install(self):
        """Register backward hooks that zero gradients for inactive layers."""
        self.clear()
        for i, layer in enumerate(self.layers):
            idx = i  # capture
            for param in layer.parameters():
                if param.requires_grad:
                    h = param.register_hook(
                        lambda grad, i=idx: grad if i in self._active else torch.zeros_like(grad)
                    )
                    self._hooks.append(h)

    def clear(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ─────────────────────────────────────────────────────────────────────────────
# Layer specialisation probe
# ─────────────────────────────────────────────────────────────────────────────

class SpecialisationProbe:
    """
    Runs lightweight RYS-style ablation probes during training to measure
    how much each layer contributes to each cognitive type.

    For each layer i, temporarily disable it (skip its forward pass) and
    measure the loss change on a small typed probe set.
    A large loss increase when layer i is disabled → layer i matters a lot
    for that type → it's specialising.

    Returns a matrix: [num_layers × num_types] of importance scores.
    """

    def __init__(self, model, tokenizer, device, probe_samples_per_type: int = 4):
        self.model    = model
        self.tokenizer = tokenizer
        self.device   = device
        self.n        = probe_samples_per_type
        self.layers   = _get_layer_stack(model)
        self._orig_forwards = [l.forward for l in self.layers]

    def _disable_layer(self, idx: int):
        def skip(hidden_states, *a, **kw):
            return (hidden_states,)
        self.layers[idx].forward = skip

    def _restore_layer(self, idx: int):
        self.layers[idx].forward = self._orig_forwards[idx]

    def _restore_all(self):
        for i, layer in enumerate(self.layers):
            layer.forward = self._orig_forwards[i]

    def _eval_loss(self, pairs: List[Tuple[str, str]]) -> float:
        """Compute average log-prob loss on a list of (prompt, response) pairs."""
        if not pairs:
            return 0.0
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for prompt, response in pairs:
                text = prompt + " " + response
                enc  = self.tokenizer(text, return_tensors="pt",
                                      truncation=True, max_length=256).to(self.device)
                p_len = len(self.tokenizer(prompt, return_tensors="pt").input_ids[0])
                labels = enc.input_ids.clone()
                labels[0, :p_len] = -100
                out  = self.model(**enc, labels=labels)
                total += out.loss.item()
        return total / len(pairs)

    def run(self, typed_probe_bank: Dict[str, List[Tuple[str, str]]]) -> Dict[str, List[float]]:
        """
        typed_probe_bank: {type_name: [(prompt, response), ...]}
        Returns: {type_name: [importance_score_per_layer]}
          score = baseline_loss_delta when layer is disabled
                  (positive = layer matters, negative = layer hurts)
        """
        results: Dict[str, List[float]] = {t: [] for t in typed_probe_bank}

        for stype, pairs in typed_probe_bank.items():
            if not pairs:
                results[stype] = [0.0] * len(self.layers)
                continue

            probe_pairs = pairs[:self.n]
            baseline = self._eval_loss(probe_pairs)
            scores = []
            for i in range(len(self.layers)):
                self._disable_layer(i)
                ablated = self._eval_loss(probe_pairs)
                self._restore_layer(i)
                # Positive = disabling hurts = layer is important
                scores.append(ablated - baseline)
            results[stype] = scores

        self._restore_all()
        self.model.train()
        return results


# ─────────────────────────────────────────────────────────────────────────────
# RYS Adaptive Learning Rate
# ─────────────────────────────────────────────────────────────────────────────

class RYSAdaptiveLR:
    """
    Maps RYS sweep delta scores to per-layer learning rate multipliers.

    Theory (Idea 3):
      Translation layers (early/late) show flat or negative RYS delta.
      They are already well-organised — high LR would destabilise them.
      Reasoning circuits (middle) show high positive delta.
      They benefit from more capacity AND more learning signal.

    The mapping function:
      multiplier = base + scale * tanh(delta / temperature)
      base=0.1  → translation layers get 10% of nominal LR
      base+scale=1.9 → reasoning circuits get 190% of nominal LR (capped at max_mult)

    After a sweep, call .apply_to_optimizer(optimizer, layer_param_groups)
    to update per-layer learning rates live during training.
    """

    def __init__(self,
                 base:        float = 0.1,   # LR multiplier for flat/negative layers
                 scale:       float = 1.8,   # range: [base, base+scale]
                 temperature: float = 0.05,  # sensitivity to delta magnitude
                 max_mult:    float = 3.0,   # hard cap on multiplier
                 log: Callable = print):
        self.base        = base
        self.scale       = scale
        self.temperature = temperature
        self.max_mult    = max_mult
        self.log         = log
        self.multipliers: List[float] = []
        self.delta_scores: List[float] = []

    def compute_from_matrix(self, delta_matrix, combine: str = "max"):
        """
        delta_matrix: np.ndarray shape [N_layers, N_end_layers] from RYSEngine.sweep()
                      OR a flat list of per-layer scores.
        combine: how to reduce the matrix to per-layer scores.
          "max"  — take the max delta in each row (best duplication of this start layer)
          "mean" — mean of all valid (i<j) deltas for this layer
        """
        import numpy as np

        if isinstance(delta_matrix, list):
            scores = [float(x) for x in delta_matrix]
        else:
            N = delta_matrix.shape[0]
            scores = []
            for i in range(N):
                row_vals = [delta_matrix[i, j] for j in range(i + 1, N)
                            if delta_matrix[i, j] != 0.0]
                if not row_vals:
                    scores.append(0.0)
                elif combine == "max":
                    scores.append(float(np.max(row_vals)))
                else:
                    scores.append(float(np.mean(row_vals)))

        self.delta_scores = scores
        self.multipliers  = [
            min(self.max_mult,
                self.base + self.scale * math.tanh(d / self.temperature))
            for d in scores
        ]
        self.log(f"[RYSAdaptiveLR] Computed {len(self.multipliers)} multipliers. "
                 f"Range: [{min(self.multipliers):.3f}, {max(self.multipliers):.3f}]")
        return self.multipliers

    def apply_to_optimizer(self, optimizer, layer_param_groups: List[dict],
                           base_lr: float):
        """
        layer_param_groups: list of optimizer param_groups, one per layer.
        Updates each group's lr to base_lr * multiplier[i].
        """
        if not self.multipliers:
            return
        for i, pg in enumerate(layer_param_groups):
            if i < len(self.multipliers):
                pg["lr"] = base_lr * self.multipliers[i]
        self.log(f"[RYSAdaptiveLR] Applied LR multipliers to {len(layer_param_groups)} groups")

    def plot(self):
        """Return a matplotlib figure of the LR multipliers."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        if self.delta_scores:
            axes[0].bar(range(len(self.delta_scores)), self.delta_scores,
                        color=["red" if d > 0 else "blue" for d in self.delta_scores],
                        alpha=0.7)
            axes[0].axhline(0, color="black", lw=0.8)
            axes[0].set_title("RYS Delta Scores (per layer)")
            axes[0].set_xlabel("Layer")
            axes[0].set_ylabel("Δ score")
        if self.multipliers:
            axes[1].bar(range(len(self.multipliers)), self.multipliers,
                        color="steelblue", alpha=0.7)
            axes[1].axhline(1.0, color="black", lw=0.8, linestyle="--",
                            label="baseline LR (1×)")
            axes[1].set_title("LR Multipliers (per layer)")
            axes[1].set_xlabel("Layer")
            axes[1].set_ylabel("Multiplier")
            axes[1].legend()
        plt.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ground-Up Trainer
# ─────────────────────────────────────────────────────────────────────────────

class GroundUpTrainer:
    """
    Train a transformer from random initialisation.

    model_source:
      - Path to an existing checkpoint whose *architecture* (config) is reused
        but whose weights are randomly re-initialised.
      - OR a dict of architecture kwargs passed to AutoConfig.for_model().

    This gives you a clean slate with a known architecture, which is the
    prerequisite for the stretch-distill and layer-aware workflows.
    """

    def __init__(self,
                 model_source: str,
                 device: str = "cuda",
                 log: Callable = print):
        self.model_source = model_source
        self.device = torch.device(
            device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.log = log
        self._load()

    def _load(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

        self.log(f"[GroundUp] Loading architecture from {self.model_source} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_source)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(self.model_source)
        config.use_cache = False
        self.log(f"[GroundUp] Architecture: {config.model_type}, "
                 f"{config.num_hidden_layers} layers, "
                 f"hidden={config.hidden_size}")

        # from_config initialises random weights — no pretrained data loaded
        self.model = AutoModelForCausalLM.from_config(
            config, torch_dtype=torch.float32
        ).to(self.device)

        total = sum(p.numel() for p in self.model.parameters())
        self.log(f"[GroundUp] Random-init model: {total/1e6:.1f}M parameters")

    def train(self,
              dataset_source,
              output_dir: str,
              epochs:       int   = 5,
              batch_size:   int   = 2,
              grad_accum:   int   = 8,
              learning_rate: float = 1e-3,
              max_length:   int   = 256,
              warmup_ratio: float = 0.05,
              save_every_n_steps: int = 500,
              ) -> dict:
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        os.makedirs(output_dir, exist_ok=True)
        self.log("[GroundUp] Preparing dataset ...")
        ds = load_dataset_from_source(dataset_source, self.tokenizer, max_length)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            collate_fn=_collate)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate,
                          weight_decay=0.01, betas=(0.9, 0.95))
        total_steps    = math.ceil(len(loader) / grad_accum) * epochs
        warmup_steps   = int(total_steps * warmup_ratio)
        scheduler      = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1),
                                           eta_min=learning_rate * 0.1)

        loss_history = []
        global_step  = 0
        best_loss    = float("inf")
        best_ckpt    = None
        self.log(f"[GroundUp] Training: {epochs} epochs, {len(ds)} samples, "
                 f"lr={learning_rate}, warmup={warmup_steps} steps")

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out   = self.model(**batch)
                loss  = out.loss / grad_accum
                loss.backward()
                epoch_loss += loss.item() * grad_accum

                if (step + 1) % grad_accum == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    # Linear warmup override
                    if global_step < warmup_steps:
                        for pg in optimizer.param_groups:
                            pg["lr"] = learning_rate * (global_step + 1) / max(warmup_steps, 1)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if save_every_n_steps and global_step % save_every_n_steps == 0:
                        ckpt = os.path.join(output_dir, f"step_{global_step}")
                        self._save(ckpt)

            avg = epoch_loss / max(len(loader), 1)
            loss_history.append({"epoch": epoch, "loss": round(avg, 6)})
            self.log(f"[GroundUp] Epoch {epoch}/{epochs} — loss={avg:.6f}")
            ckpt = os.path.join(output_dir, f"epoch_{epoch}")
            self._save(ckpt)
            if avg < best_loss:
                best_loss  = avg
                best_ckpt  = ckpt

        final = os.path.join(output_dir, "final")
        self._save(final)
        log_data = {
            "mode": "ground_up",
            "model_source": self.model_source,
            "epochs": epochs, "loss_history": loss_history,
            "best_checkpoint": best_ckpt,
        }
        with open(os.path.join(output_dir, "training_log.json"), "w") as f:
            json.dump(log_data, f, indent=2)
        return log_data

    def _save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        self.log(f"[GroundUp] Saved → {directory}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Layer-Aware Trainer  (Idea 1)
# ─────────────────────────────────────────────────────────────────────────────

class LayerAwareTrainer:
    """
    Trains a model with per-sample gradient routing based on cognitive type.

    Workflow per training step:
      1. Classify the batch's sample type (math / code / reasoning / etc.)
      2. Look up which layer range is assigned to that type (routing_map)
      3. Zero gradients for all other layers via GradientRouter hooks
      4. Normal forward/backward/step

    Routing map is initialised uniformly. Periodically, SpecialisationProbe
    runs to detect emerging specialisation and updates the routing map to
    match observed layer importance — reinforcing specialisation via a
    positive feedback loop.

    probe_bank: {type: [(prompt, response)]} — small typed eval set used by
    the probe. If None, probing is disabled and routing stays static.
    """

    def __init__(self,
                 model_path:    str,
                 routing_map:   Optional[Dict[str, List[int]]] = None,
                 probe_bank:    Optional[Dict[str, List[Tuple[str, str]]]] = None,
                 probe_every_n_steps: int = 200,
                 specialisation_threshold: float = 0.02,
                 device:        str = "cuda",
                 log:           Callable = print):
        """
        routing_map: {sample_type: [list of layer indices to train]}
          If None, all layers are trained for all types initially.
        specialisation_threshold: minimum per-layer delta to count as specialised.
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.model_path  = model_path
        self.probe_bank  = probe_bank or {}
        self.probe_every = probe_every_n_steps
        self.threshold   = specialisation_threshold
        self.device      = torch.device(
            device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.log         = log

        self.log(f"[LayerAware] Loading {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, device_map=None
        ).to(self.device)
        self.model.config.use_cache = False

        self.layers  = _get_layer_stack(self.model)
        self.N       = len(self.layers)

        # Initialise routing: each type gets all layers
        self.routing_map: Dict[str, set] = {}
        for stype in ALL_TYPES + ["general"]:
            if routing_map and stype in routing_map:
                self.routing_map[stype] = set(routing_map[stype])
            else:
                self.routing_map[stype] = set(range(self.N))

        self.router = GradientRouter(self.model)
        self.router.install()

        self.probe       = SpecialisationProbe(self.model, self.tokenizer,
                                               self.device) if probe_bank else None
        self.spec_history: List[dict] = []   # log of probe results over time

        total, _ = count_trainable(self.model)
        self.log(f"[LayerAware] Ready — {self.N} layers, {total/1e6:.1f}M params")

    def _update_routing_from_probe(self, step: int):
        """Run probe and update routing map based on specialisation scores."""
        if not self.probe or not self.probe_bank:
            return

        self.log(f"[LayerAware] Running specialisation probe at step {step} ...")
        scores = self.probe.run(self.probe_bank)  # {type: [score_per_layer]}

        for stype, layer_scores in scores.items():
            # Find layers where this type's score is above threshold
            specialised = {i for i, s in enumerate(layer_scores)
                           if s > self.threshold}
            if specialised:
                self.routing_map[stype] = specialised
                self.log(f"[LayerAware] {stype}: routing → layers "
                         f"{sorted(specialised)}")

        self.spec_history.append({
            "step": step,
            "scores": {t: [round(s, 5) for s in v] for t, v in scores.items()},
        })

    def train(self,
              dataset_source,
              output_dir:    str,
              epochs:        int   = 3,
              batch_size:    int   = 1,
              grad_accum:    int   = 4,
              learning_rate: float = 2e-4,
              max_length:    int   = 512,
              ) -> dict:
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        os.makedirs(output_dir, exist_ok=True)
        self.log("[LayerAware] Preparing tagged dataset ...")
        ds = TaggedPairDataset(
            self._source_to_pairs(dataset_source), self.tokenizer, max_length
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            collate_fn=_tagged_collate)

        optimizer    = AdamW(self.model.parameters(), lr=learning_rate,
                             weight_decay=0.01)
        total_steps  = math.ceil(len(loader) / grad_accum) * epochs
        scheduler    = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
        loss_history = []
        global_step  = 0

        self.log(f"[LayerAware] Training {epochs} epochs, {len(ds)} samples")

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(loader):
                # Determine dominant sample type in this batch
                types   = batch.pop("sample_types")
                dominant = max(set(types), key=types.count)
                active  = self.routing_map.get(dominant, set(range(self.N)))
                self.router.set_active_layers(active)

                batch  = {k: v.to(self.device) for k, v in batch.items()}
                out    = self.model(**batch)
                loss   = out.loss / grad_accum
                loss.backward()   # hooks zero inactive layer gradients
                epoch_loss += loss.item() * grad_accum

                if (step + 1) % grad_accum == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if (self.probe_every > 0
                            and global_step % self.probe_every == 0):
                        self._update_routing_from_probe(global_step)
                        self.model.train()

            avg = epoch_loss / max(len(loader), 1)
            loss_history.append({"epoch": epoch, "loss": round(avg, 6)})
            self.log(f"[LayerAware] Epoch {epoch}/{epochs} — loss={avg:.6f}")
            self._save(os.path.join(output_dir, f"epoch_{epoch}"))

        self.router.clear()
        log_data = {
            "mode": "layer_aware",
            "epochs": epochs, "loss_history": loss_history,
            "specialisation_history": self.spec_history,
            "final_routing": {k: sorted(v)
                              for k, v in self.routing_map.items()},
        }
        with open(os.path.join(output_dir, "training_log.json"), "w") as f:
            json.dump(log_data, f, indent=2)
        self._save(os.path.join(output_dir, "final"))
        return log_data

    def _source_to_pairs(self, source):
        if isinstance(source, list):
            return source
        if isinstance(source, str) and source.endswith(".json"):
            with open(source) as f:
                data = json.load(f)
            return [(d["prompt"], d["response"]) for d in data]
        from datasets import load_dataset as hf_load
        ds = hf_load(source, split="train")
        return [(r["prompt"], r["response"]) for r in ds]

    def _save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        self.log(f"[LayerAware] Saved → {directory}")

    def get_routing_plot(self):
        """Return a matplotlib figure showing the current routing map."""
        import matplotlib.pyplot as plt
        import numpy as np
        types = sorted(self.routing_map.keys())
        mat   = np.zeros((len(types), self.N))
        for r, t in enumerate(types):
            for c in self.routing_map[t]:
                if c < self.N:
                    mat[r, c] = 1.0
        fig, ax = plt.subplots(figsize=(max(8, self.N // 3), len(types) + 1))
        ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_yticks(range(len(types)))
        ax.set_yticklabels(types)
        ax.set_xlabel("Layer Index")
        ax.set_title("Layer Routing Map (blue = gradient active)")
        plt.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Stretch–Distill Pipeline  (Idea 2)
# ─────────────────────────────────────────────────────────────────────────────

class StretchDistillPipeline:
    """
    Phase 1 — Stretch:
      Insert blank layers into the small model via LayerSurgeon.
      Export stretched model as a checkpoint.

    Phase 2 — Compartmentalise:
      Train the stretched model with LayerAwareTrainer so functional
      regions form in the blank/new layers.

    Phase 3 — Distill:
      Train a fresh small model (from the original architecture) against
      the stretched+compartmentalised teacher using:
        - Output KD loss:  KL(student_logits || teacher_logits)
        - Activation loss: MSE between student and teacher hidden states
          at identified functional layer positions.

    The activation matching is what distinguishes this from standard KD —
    the student is guided to reproduce specific functional regions, not just
    match output distributions.
    """

    def __init__(self,
                 small_model_path: str,
                 blank_layers_to_insert: int = 4,
                 insert_after_layer: int = -1,   # -1 = middle
                 kd_alpha: float = 0.5,           # weight: output KD vs CE loss
                 act_alpha: float = 0.3,          # weight: activation matching
                 device: str = "cuda",
                 log: Callable = print):
        self.small_model_path = small_model_path
        self.n_blanks         = blank_layers_to_insert
        self.insert_after     = insert_after_layer
        self.kd_alpha         = kd_alpha
        self.act_alpha        = act_alpha
        self.device           = torch.device(
            device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.log = log
        self.stretched_path: Optional[str] = None
        self.teacher_model  = None
        self.student_model  = None

    def phase1_stretch(self, output_dir: str) -> str:
        """Insert blank layers and export. Returns path to stretched model."""
        from layer_surgeon import LayerSurgeon
        self.log(f"[Stretch] Phase 1 — stretching {self.small_model_path} ...")
        surgeon = LayerSurgeon(self.small_model_path, device=str(self.device))
        N = len(surgeon.layers)
        after = self.insert_after if self.insert_after >= 0 else N // 2

        for i in range(self.n_blanks):
            slot = after + i
            surgeon.insert_blank_layer(slot)
            self.log(f"[Stretch] Inserted blank layer at slot {slot + 1}")

        os.makedirs(output_dir, exist_ok=True)
        self.stretched_path = surgeon.export(output_dir)
        self.log(f"[Stretch] Phase 1 complete → {self.stretched_path}")
        return self.stretched_path

    def phase2_compartmentalise(self,
                                dataset_source,
                                output_dir: str,
                                probe_bank: Optional[Dict] = None,
                                epochs: int = 3,
                                learning_rate: float = 2e-4,
                                **kwargs) -> dict:
        """Train stretched model with layer-aware routing."""
        if not self.stretched_path:
            raise RuntimeError("Run phase1_stretch() first")
        self.log("[Stretch] Phase 2 — compartmentalising ...")
        trainer = LayerAwareTrainer(
            model_path=self.stretched_path,
            probe_bank=probe_bank,
            device=str(self.device),
            log=self.log,
        )
        log = trainer.train(dataset_source, output_dir,
                            epochs=epochs, learning_rate=learning_rate, **kwargs)
        self.stretched_path = os.path.join(output_dir, "final")
        self.log(f"[Stretch] Phase 2 complete → {self.stretched_path}")
        return log

    def phase3_distill(self,
                       dataset_source,
                       output_dir: str,
                       teacher_path: Optional[str] = None,
                       teacher_layer_indices: Optional[List[int]] = None,
                       epochs: int = 5,
                       batch_size: int = 1,
                       grad_accum: int = 4,
                       learning_rate: float = 2e-4,
                       max_length: int = 256,
                       temperature: float = 2.0,
                       ) -> dict:
        """
        Distil the compartmentalised teacher into a fresh small model.
        teacher_layer_indices: teacher layer positions to match activations from.
          If None, uses the middle third of teacher layers.
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        teacher_path = teacher_path or self.stretched_path
        if not teacher_path:
            raise RuntimeError("No teacher path — run phase2_compartmentalise() first")

        os.makedirs(output_dir, exist_ok=True)
        self.log("[Stretch] Phase 3 — distilling ...")

        # Load teacher
        self.log("[Stretch] Loading teacher ...")
        t_tok = AutoTokenizer.from_pretrained(teacher_path)
        if t_tok.pad_token is None:
            t_tok.pad_token = t_tok.eos_token
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_path, torch_dtype=torch.float32
        ).to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # Build fresh student from original small model architecture
        self.log("[Stretch] Initialising student (random weights, small arch) ...")
        s_config = AutoConfig.from_pretrained(self.small_model_path)
        s_config.use_cache = False
        student = AutoModelForCausalLM.from_config(
            s_config, torch_dtype=torch.float32
        ).to(self.device)
        s_tok = AutoTokenizer.from_pretrained(self.small_model_path)
        if s_tok.pad_token is None:
            s_tok.pad_token = s_tok.eos_token

        # Identify teacher layers to match
        t_layers = _get_layer_stack(teacher)
        s_layers = _get_layer_stack(student)
        T_N = len(t_layers)
        S_N = len(s_layers)
        if teacher_layer_indices is None:
            third = T_N // 3
            teacher_layer_indices = list(range(third, 2 * third))
        self.log(f"[Stretch] Activation matching: teacher layers "
                 f"{teacher_layer_indices} → student layers (proportional)")

        # Map teacher layer indices to proportional student layer indices
        def teacher_to_student_idx(ti):
            return min(int(ti / T_N * S_N), S_N - 1)

        activation_pairs = [
            (ti, teacher_to_student_idx(ti)) for ti in teacher_layer_indices
        ]

        # Activation capture hooks
        t_acts: Dict[int, torch.Tensor] = {}
        s_acts: Dict[int, torch.Tensor] = {}

        def make_capture(storage, idx):
            def hook(module, inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                storage[idx] = hs
            return hook

        t_hooks = [t_layers[ti].register_forward_hook(make_capture(t_acts, ti))
                   for ti, _ in activation_pairs]
        s_hooks = [s_layers[si].register_forward_hook(make_capture(s_acts, si))
                   for _, si in activation_pairs]

        # Dataset (use student tokenizer)
        ds = load_dataset_from_source(dataset_source, s_tok, max_length)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            collate_fn=_collate)

        optimizer   = AdamW(student.parameters(), lr=learning_rate,
                            weight_decay=0.01)
        total_steps = math.ceil(len(loader) / grad_accum) * epochs
        scheduler   = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
        loss_history = []
        global_step  = 0

        for epoch in range(1, epochs + 1):
            student.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Teacher forward (no grad)
                with torch.no_grad():
                    t_out = teacher(**{k: v for k, v in batch.items()
                                       if k != "labels"})
                    t_logits = t_out.logits

                # Student forward
                s_out    = student(**batch)
                ce_loss  = s_out.loss   # standard causal LM loss

                # Output KD loss: KL(soft_student || soft_teacher)
                T  = temperature
                s_soft = F.log_softmax(s_out.logits / T, dim=-1)
                t_soft = F.softmax(t_logits / T, dim=-1)
                # Align vocab sizes if they differ
                v_min = min(s_soft.shape[-1], t_soft.shape[-1])
                kd_loss = F.kl_div(s_soft[..., :v_min],
                                   t_soft[..., :v_min],
                                   reduction="batchmean") * (T ** 2)

                # Activation matching loss
                act_loss = torch.tensor(0.0, device=self.device)
                for (ti, si) in activation_pairs:
                    if ti in t_acts and si in s_acts:
                        ta = t_acts[ti]
                        sa = s_acts[si]
                        # Match spatial dims; project if hidden sizes differ
                        seq = min(ta.shape[1], sa.shape[1])
                        ta  = ta[:, :seq, :]
                        sa  = sa[:, :seq, :]
                        h_min = min(ta.shape[-1], sa.shape[-1])
                        act_loss = act_loss + F.mse_loss(
                            sa[..., :h_min], ta[..., :h_min].detach()
                        )

                # Combined loss
                total_loss = ((1 - self.kd_alpha - self.act_alpha) * ce_loss
                              + self.kd_alpha  * kd_loss
                              + self.act_alpha * act_loss)

                (total_loss / grad_accum).backward()
                epoch_loss += total_loss.item()

                if (step + 1) % grad_accum == 0:
                    nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            avg = epoch_loss / max(len(loader), 1)
            loss_history.append({"epoch": epoch, "loss": round(avg, 6)})
            self.log(f"[Stretch] Distil epoch {epoch}/{epochs} — loss={avg:.6f}")
            ckpt = os.path.join(output_dir, f"epoch_{epoch}")
            os.makedirs(ckpt, exist_ok=True)
            student.save_pretrained(ckpt)
            s_tok.save_pretrained(ckpt)

        # Cleanup hooks
        for h in t_hooks + s_hooks:
            h.remove()

        final = os.path.join(output_dir, "final")
        os.makedirs(final, exist_ok=True)
        student.save_pretrained(final)
        s_tok.save_pretrained(final)

        log_data = {
            "mode": "stretch_distill",
            "teacher": teacher_path,
            "student_arch": self.small_model_path,
            "epochs": epochs, "loss_history": loss_history,
        }
        with open(os.path.join(output_dir, "training_log.json"), "w") as f:
            json.dump(log_data, f, indent=2)
        self.log(f"[Stretch] Distillation complete → {final}")
        return log_data