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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple, Any

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

class RYSProbe:
    """
    RYS sweep probe — finds which layer range is ALREADY GOOD at a task.

    Core measurement (identical to RYSEngine.sweep)
    ────────────────────────────────────────────────
    score(i, j) = run_test_with_rys(i, j, loops=1) − baseline

    Positive delta → repeating that range IMPROVES performance on the probe
    questions → those layers are already doing useful task-specific work.
    These are the layers we want to finetune further, not the ones that
    hurt when disabled.

    Scoring metric: mean log-prob per answer token (higher = better, always
    negative).  Matches RYSEngine.run_test() exactly.

    Phase 2 — Minimal subset shrink
    ────────────────────────────────
    Starting from the best (start, end), greedily shrink from both ends.
    At each step, try applying RYS on [start+1, end] and [start, end-1].
    Keep whichever sub-range still scores >= min_subset_ratio * best_score.
    Stop when neither end can be dropped without losing too much signal.

    This finds the tightest range that still demonstrates math competence,
    giving a precise finetune target rather than a padded bounding box.
    """

    def __init__(self, model, tokenizer, device,
                 max_pairs: int = 4,
                 log: Callable = print):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
        self.max_pairs = max_pairs
        self.log       = log
        self.layers    = _get_layer_stack(model)
        self.N         = len(self.layers)
        self._orig_forwards = [l.forward for l in self.layers]

    # ── RYS application — mirrors RYSEngine.apply_rys exactly ────────────────

    def _apply_rys(self, start: int, end: int, loops: int = 1):
        """Patch layers[end].forward to repeat [start..end] `loops` extra times."""
        origs = self._orig_forwards
        s, e, L = start, end, loops

        def end_layer_forward(hidden_states, *args, **kwargs):
            outputs = origs[e](hidden_states, *args, **kwargs)
            if L <= 0:
                return outputs
            if isinstance(outputs, (tuple, list)):
                hs, tail, is_tuple = outputs[0], list(outputs[1:]), True
            else:
                hs, tail, is_tuple = outputs, [], False
            for _ in range(L):
                for k in range(s, e + 1):
                    out_k = origs[k](hs, *args, **kwargs)
                    if isinstance(out_k, (tuple, list)):
                        hs, tail, is_tuple = out_k[0], list(out_k[1:]), True
                    else:
                        hs, tail, is_tuple = out_k, [], False
            return (hs, *tail) if is_tuple else hs

        self.layers[e].forward = end_layer_forward

    def _restore_all(self):
        for i in range(self.N):
            self.layers[i].forward = self._orig_forwards[i]

    # ── Scoring — mirrors RYSEngine.run_test exactly ──────────────────────────

    def _run_test(self, pairs: List[Tuple[str, str]]) -> float:
        """Mean log-prob per answer token.  Higher = better.  Always negative."""
        if not pairs:
            return 0.0
        self.model.eval()
        prompts    = [q + " " for q, _ in pairs]
        answers    = [a       for _, a in pairs]
        full_texts = [p + a   for p, a in zip(prompts, answers)]

        full_inputs = self.tokenizer(
            full_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        ).to(self.device)

        prompt_lengths = [
            len(self.tokenizer(p, add_special_tokens=True).input_ids)
            for p in prompts
        ]
        answer_ids = [
            self.tokenizer(a, add_special_tokens=False).input_ids
            for a in answers
        ]

        with torch.no_grad():
            logits = self.model(**full_inputs).logits

        scores = []
        for i in range(len(pairs)):
            ans_ids = answer_ids[i]
            if not ans_ids:
                continue
            logprob = 0.0
            for j, token in enumerate(ans_ids):
                pos = prompt_lengths[i] + j - 1
                if pos < 0 or pos >= logits.shape[1]:
                    continue
                logprob += torch.log_softmax(
                    logits[i, pos], dim=-1)[token].item()
            scores.append(logprob / len(ans_ids))

        return float(sum(scores) / len(scores)) if scores else 0.0

    # ── Phase 1: full RYS sweep ───────────────────────────────────────────────

    def sweep(self, pairs: List[Tuple[str, str]],
              ) -> Tuple[float, List[List[float]]]:
        """
        Run RYS(i, j, loops=1) for all valid (i < j) pairs.
        Returns (baseline, matrix) where matrix[i][j] = delta above baseline.
        Positive delta = repeating that range improves math performance.
        """
        probe_pairs = pairs[:self.max_pairs]
        baseline    = self._run_test(probe_pairs)
        matrix      = [[0.0] * self.N for _ in range(self.N)]

        total = self.N * (self.N - 1) // 2
        done  = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                self._restore_all()
                self._apply_rys(i, j, loops=1)
                score        = self._run_test(probe_pairs)
                matrix[i][j] = score - baseline
                done += 1
                if done % 10 == 0:
                    self.log(f"[RYSProbe] Sweep {done}/{total} — "
                             f"RYS({i},{j}) delta={matrix[i][j]:+.5f}")

        self._restore_all()
        self.model.train()
        return baseline, matrix

    # ── Phase 2: minimal subset shrink ───────────────────────────────────────

    def shrink_range(self, pairs: List[Tuple[str, str]],
                     start: int, end: int,
                     best_score: float,
                     min_ratio:  float = 0.75,
                     ) -> Tuple[int, int, float]:
        """
        Greedily shrink [start, end] from both ends using RYS scoring.
        At each step, try [start+1, end] and [start, end-1].
        Keep the sub-range that retains the most signal above the floor.
        Floor = min_ratio * best_score.

        Returns (final_start, final_end, final_score).
        """
        probe_pairs = pairs[:self.max_pairs]
        baseline    = self._run_test(probe_pairs)
        threshold   = min_ratio * best_score
        cur_s, cur_e = start, end

        self.log(f"[RYSProbe] Shrink: [{start},{end}] "
                 f"best_delta={best_score:+.5f}  "
                 f"floor={threshold:+.5f} ({min_ratio*100:.0f}%)")

        while cur_s < cur_e:
            # Score [cur_s+1, cur_e]
            if cur_s + 1 <= cur_e:
                self._restore_all()
                self._apply_rys(cur_s + 1, cur_e)
                s_drop_left = self._run_test(probe_pairs) - baseline
            else:
                s_drop_left = float("-inf")

            # Score [cur_s, cur_e-1]
            if cur_s <= cur_e - 1:
                self._restore_all()
                self._apply_rys(cur_s, cur_e - 1)
                s_drop_right = self._run_test(probe_pairs) - baseline
            else:
                s_drop_right = float("-inf")

            self._restore_all()

            can_left  = s_drop_left  >= threshold
            can_right = s_drop_right >= threshold

            if not can_left and not can_right:
                self.log(f"[RYSProbe] Cannot shrink: "
                         f"drop_left={s_drop_left:+.5f}  "
                         f"drop_right={s_drop_right:+.5f}  "
                         f"floor={threshold:+.5f}")
                break

            if can_left and (not can_right or s_drop_left >= s_drop_right):
                self.log(f"[RYSProbe] ← drop L{cur_s}  "
                         f"(delta {s_drop_left:+.5f} ≥ floor {threshold:+.5f})")
                cur_s += 1
            else:
                self.log(f"[RYSProbe] → drop L{cur_e}  "
                         f"(delta {s_drop_right:+.5f} ≥ floor {threshold:+.5f})")
                cur_e -= 1

        # Final score for the shrunk range
        self._restore_all()
        if cur_s < cur_e:
            self._apply_rys(cur_s, cur_e)
        final_score = self._run_test(probe_pairs) - baseline
        self._restore_all()
        self.model.train()

        self.log(f"[RYSProbe] Minimal range: [{cur_s},{cur_e}]  "
                 f"delta={final_score:+.5f}  "
                 f"(original [{start},{end}] delta={best_score:+.5f})")
        return cur_s, cur_e, final_score

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, typed_probe_bank: Dict[str, List[Tuple[str, str]]],
            find_minimal_subset: bool = True,
            min_subset_ratio:    float = 0.75,
            ) -> Dict[str, dict]:
        """
        Run a full RYS sweep for each type in the probe bank.

        Returns per-type:
          {
            "baseline":     float,   # unmodified log-prob score
            "best_start":   int,     # range with highest positive RYS delta
            "best_end":     int,
            "best_score":   float,   # delta above baseline (positive = good at task)
            "final_start":  int,     # after minimal-subset shrink
            "final_end":    int,
            "final_score":  float,
            "matrix":       [[float]]
          }
        """
        results = {}

        for stype, pairs in typed_probe_bank.items():
            if not pairs:
                results[stype] = {
                    "baseline": 0.0,
                    "best_start": 0, "best_end": 0, "best_score": 0.0,
                    "final_start": 0, "final_end": 0, "final_score": 0.0,
                    "matrix": [],
                }
                continue

            n_pairs = self.N * (self.N - 1) // 2
            self.log(f"[RYSProbe] Sweeping type='{stype}'  "
                     f"{self.N} layers → {n_pairs} RYS pairs ...")

            baseline, matrix = self.sweep(pairs)

            best_start, best_end, best_score = 0, 1, float("-inf")
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    if matrix[i][j] > best_score:
                        best_score         = matrix[i][j]
                        best_start, best_end = i, j

            self.log(f"[RYSProbe] Best RYS range for '{stype}': "
                     f"[{best_start},{best_end}]  "
                     f"delta={best_score:+.5f}  "
                     f"(baseline={baseline:.5f})")

            if find_minimal_subset and best_score > 0 and best_start < best_end:
                fs, fe, fscore = self.shrink_range(
                    pairs, best_start, best_end,
                    best_score, min_ratio=min_subset_ratio,
                )
            else:
                fs, fe, fscore = best_start, best_end, best_score

            results[stype] = {
                "baseline":   baseline,
                "best_start": best_start,
                "best_end":   best_end,
                "best_score": best_score,
                "final_start": fs,
                "final_end":   fe,
                "final_score": fscore,
                "matrix":      matrix,
            }

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
              epochs:             int   = 5,
              batch_size:         int   = 2,
              grad_accum:         int   = 8,
              learning_rate:      float = 1e-3,
              max_length:         int   = 256,
              warmup_ratio:       float = 0.05,
              save_every_n_steps: int   = 500,
              # ── Probe-driven adaptive LR ──────────────────────────────────
              probe_bank:         Optional[Dict[str, List[Tuple[str, str]]]] = None,
              probe_every_n_steps: int  = 200,
              alr_base:           float = 0.1,
              alr_scale:          float = 1.8,
              alr_temperature:    float = 0.05,
              alr_max_mult:       float = 3.0,
              ) -> dict:
        """
        Train a model from random initialisation.

        probe_bank: optional {type: [(prompt, response), ...]} eval pairs.
          When provided, a SpecialisationProbe runs every probe_every_n_steps
          optimizer steps.  The per-layer importance scores are fed into
          RYSAdaptiveLR to recompute per-layer learning rates on the fly:
            - Layers the probe identifies as load-bearing → higher LR
            - Flat / unhelpful layers                    → lower LR
          This replaces the flat cosine schedule with a probe-informed one.
        """
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        os.makedirs(output_dir, exist_ok=True)
        self.log("[GroundUp] Preparing dataset ...")
        ds = load_dataset_from_source(dataset_source, self.tokenizer, max_length)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            collate_fn=_collate)

        layers       = _get_layer_stack(self.model)
        N            = len(layers)
        total_steps  = math.ceil(len(loader) / grad_accum) * epochs
        warmup_steps = int(total_steps * warmup_ratio)

        # Build one param-group per layer so RYSAdaptiveLR can set lr per layer.
        # Non-layer parameters (embeddings, lm_head, norm) share a single group.
        layer_param_sets = [set(id(p) for p in layer.parameters()) for layer in layers]
        all_layer_ids    = set().union(*layer_param_sets)

        other_params = [p for p in self.model.parameters()
                        if id(p) not in all_layer_ids]
        layer_groups = [
            {"params": list(layer.parameters()), "lr": learning_rate}
            for layer in layers
        ]
        param_groups = [{"params": other_params, "lr": learning_rate}] + layer_groups

        optimizer = AdamW(param_groups, weight_decay=0.01, betas=(0.9, 0.95))
        scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1),
                                      eta_min=learning_rate * 0.1)

        # Adaptive LR controller (probe-driven)
        adaptive_lr = RYSAdaptiveLR(
            base=alr_base, scale=alr_scale,
            temperature=alr_temperature, max_mult=alr_max_mult,
            log=self.log,
        ) if probe_bank else None

        # Specialisation probe (reuses SpecialisationProbe from this module)
        probe = SpecialisationProbe(
            self.model, self.tokenizer, self.device
        ) if probe_bank else None

        alr_history: List[dict] = []
        loss_history = []
        global_step  = 0
        best_loss    = float("inf")
        best_ckpt    = None

        self.log(f"[GroundUp] Training: {epochs} epochs, {len(ds)} samples, "
                 f"lr={learning_rate}, warmup={warmup_steps} steps"
                 + (f", probe every {probe_every_n_steps} steps" if probe else ""))

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

                    # Linear warmup: apply to ALL param groups uniformly
                    if global_step < warmup_steps:
                        warmup_factor = (global_step + 1) / max(warmup_steps, 1)
                        for pg in optimizer.param_groups:
                            pg["lr"] = learning_rate * warmup_factor
                    elif adaptive_lr and adaptive_lr.multipliers:
                        # After warmup: apply probe-derived per-layer LR multipliers.
                        # layer_groups are param_groups[1:] (index 0 = other params)
                        adaptive_lr.apply_to_optimizer(
                            optimizer,
                            layer_param_groups=optimizer.param_groups[1:],
                            base_lr=scheduler.get_last_lr()[0]
                            if hasattr(scheduler, "get_last_lr")
                            else learning_rate,
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # ── Probe-driven LR recalculation ─────────────────────
                    if (probe is not None
                            and probe_every_n_steps > 0
                            and global_step % probe_every_n_steps == 0):
                        self.log(f"[GroundUp] Running specialisation probe "
                                 f"at step {global_step} ...")
                        scores = probe.run(probe_bank)
                        # Reduce: per-layer max importance across all cognitive types
                        per_layer = [
                            max(scores[t][i] for t in scores)
                            for i in range(N)
                        ]
                        mults = adaptive_lr.compute_from_matrix(per_layer)
                        alr_history.append({
                            "step": global_step,
                            "per_layer_scores": [round(s, 5) for s in per_layer],
                            "lr_multipliers":   [round(m, 5) for m in mults],
                        })
                        self.model.train()   # probe sets eval mode; restore

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
            "adaptive_lr_history": alr_history,
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

# ─────────────────────────────────────────────────────────────────────────────
# TrainingModule — defines a cognitive domain + its emergence response
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingModule:
    """
    Defines a single cognitive domain that the ModularGroundUpTrainer monitors.

    Fields
    ------
    name            : Human-readable label, e.g. "math", "code", "reasoning"
    probe_bank      : {type: [(prompt, response), ...]} — keyed by cognitive type
    finetune_dataset: Dataset to use when emergence is triggered.
                      Accepts list of (prompt,response), .json path, or HF name.
    finetune_method : What to do when emergence is detected:
                        "lora"       — inject LoRA into the minimal emerged range
                        "junction"   — train only the two boundary layers
                        "full_layer" — fully unfreeze emerged range, finetune,
                                       then re-freeze
                        "stretch"    — insert blank layers around the emerged
                                       range via LayerSurgeon, reload model
    emergence_threshold   : Minimum score for the best contiguous range to
                            trigger (score = loss_delta when range is disabled)
    min_subset_ratio      : When shrinking to minimal subset, keep sub-ranges
                            whose score is >= this fraction of the best range
                            score.  0.75 = accept up to 25% signal loss for
                            a smaller target range.
    finetune_epochs       : Training epochs when triggered
    finetune_lr           : Learning rate for the triggered finetune pass
    lora_rank / lora_alpha: LoRA hyperparams (finetune_method="lora" only)
    n_blank_layers        : Blank layers to insert (finetune_method="stretch" only)
    retrigger_after_steps : Minimum steps between re-triggers for this module
    """

    name:                   str
    probe_bank:             Dict[str, List[Tuple[str, str]]]
    finetune_dataset:       Any
    finetune_method:        str   = "lora"
    emergence_threshold:    float = 0.02
    min_subset_ratio:       float = 0.75
    finetune_epochs:        int   = 1
    finetune_lr:            float = 2e-4
    lora_rank:              int   = 8
    lora_alpha:             float = 16.0
    n_blank_layers:         int   = 2
    retrigger_after_steps:  int   = 500

    # ── Runtime state (not user-configured) ──────────────────────────────────
    last_triggered_step:    int         = field(default=-1,             repr=False)
    trigger_history:        List[dict]  = field(default_factory=list,   repr=False)
    emergent_layers:        List[int]   = field(default_factory=list,   repr=False)
    lora_injected:          bool        = field(default=False,          repr=False)
    stretch_layer_offset:   int         = field(default=0,              repr=False)

    VALID_METHODS = ("lora", "junction", "full_layer", "stretch")

    def __post_init__(self):
        if self.finetune_method not in self.VALID_METHODS:
            raise ValueError(f"finetune_method must be one of {self.VALID_METHODS}")

    def cooldown_active(self, global_step: int) -> bool:
        return (self.last_triggered_step >= 0
                and global_step - self.last_triggered_step < self.retrigger_after_steps)

    def to_dict(self) -> dict:
        return {
            "name":               self.name,
            "finetune_method":    self.finetune_method,
            "trigger_history":    self.trigger_history,
            "emergent_layers":    self.emergent_layers,
            "last_triggered_step": self.last_triggered_step,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ModularGroundUpTrainer
# ─────────────────────────────────────────────────────────────────────────────

class ModularGroundUpTrainer:
    """
    Ground-up trainer with pluggable cognitive modules.

    Architecture
    ────────────
    Main loop  — trains on a general-purpose dataset, building broad language
                 capability from random initialisation.

    Modules    — each module watches for a specific cognitive type to emerge
                 naturally from the general training signal.  Every
                 `probe_interval` optimizer steps all module probes run.

    Emergence  — when a module's SpecialisationProbe sees >= min_emergent_layers
                 layers cross the emergence_threshold, that module fires:
                   1. Logs which layers emerged and their scores
                   2. Runs the module's chosen finetune_method on those layers
                      using the module's own finetune_dataset
                   3. Resumes main training with the now-specialised layers

    Per-module finetune methods
    ───────────────────────────
    lora        LoRA adapters are injected once into emerged layers and kept
                live for all subsequent main-training steps too.  Re-triggers
                add new adapters only to layers not already covered.

    junction    Freeze everything, train only the two boundary layers flanking
                the emerged region.  Clean, minimal, matches dnhkng's hypothesis.

    full_layer  Fully unfreeze emerged layers, run finetune epochs, then
                re-freeze them so main training doesn't overwrite gains.

    stretch     Export the current model, insert `n_blank_layers` blank layers
                immediately around the emerged region via LayerSurgeon, reload
                the expanded model, and continue training.  The blank layers
                give the emerging circuit room to grow.
                ⚠ Resets optimizer state and changes layer indices.

    Usage
    ─────
        modules = [
            TrainingModule(
                name="math",
                probe_bank={"math": [("What is 3*3?", "9"), ...]},
                finetune_dataset="path/to/math_pairs.json",
                finetune_method="lora",
                emergence_threshold=0.02,
                min_emergent_layers=2,
            ),
            TrainingModule(
                name="code",
                probe_bank={"code": [("Write hello world", "print('hello')"), ...]},
                finetune_dataset="codeparrot/github-code",
                finetune_method="full_layer",
            ),
        ]
        trainer = ModularGroundUpTrainer(
            model_source="Qwen/Qwen2-0.5B",
            modules=modules,
            probe_interval=200,
        )
        trainer.train(general_dataset, output_dir="./run1", epochs=20)
    """

    def __init__(self,
                 model_source:    str,
                 modules:         List[TrainingModule],
                 probe_interval:  int      = 200,
                 device:          str      = "cuda",
                 log:             Callable = print):
        self.model_source   = model_source
        self.modules        = modules
        self.probe_interval = probe_interval
        self.device         = torch.device(
            device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.log = log

        self._load_model(model_source)
        self.log(f"[Modular] Registered {len(modules)} modules: "
                 f"{[m.name for m in modules]}")

    # ── Model loading ────────────────────────────────────────────────────────

    def _load_model(self, source: str, from_checkpoint: bool = False):
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

        self.log(f"[Modular] Loading model from {source} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if from_checkpoint:
            self.model = AutoModelForCausalLM.from_pretrained(
                source, torch_dtype=torch.float32, device_map=None
            ).to(self.device)
        else:
            config = AutoConfig.from_pretrained(source)
            config.use_cache = False
            self.model = AutoModelForCausalLM.from_config(
                config, torch_dtype=torch.float32
            ).to(self.device)

        self.model.config.use_cache = False
        self.layers = _get_layer_stack(self.model)
        self.N = len(self.layers)
        total = sum(p.numel() for p in self.model.parameters())
        self.log(f"[Modular] Model ready — {self.N} layers, {total/1e6:.1f}M params")

    # ── Probe runner ─────────────────────────────────────────────────────────

    def _run_all_probes(self, global_step: int, output_dir: str,
                        main_optimizer, base_lr: float, max_length: int):
        """
        Probe every module that is not in cooldown using RYS sweep.

        Emergence condition: the best (i,j) RYS delta on the module's probe
        questions exceeds emergence_threshold.  A positive delta means
        repeating that range IMPROVES math performance — those layers are
        already good at the task and are the right target for finetuning.

        Returns True if a stretch reload occurred.
        """
        probe    = RYSProbe(self.model, self.tokenizer, self.device,
                            log=self.log)
        reloaded = False

        for module in self.modules:
            if module.cooldown_active(global_step):
                self.log(f"[Modular] Module '{module.name}' in cooldown "
                         f"({global_step - module.last_triggered_step}/"
                         f"{module.retrigger_after_steps} steps)")
                continue

            self.log(f"[Modular] RYS probing module '{module.name}' "
                     f"at step {global_step} ...")
            try:
                results = probe.run(
                    module.probe_bank,
                    find_minimal_subset=True,
                    min_subset_ratio=module.min_subset_ratio,
                )
            except Exception as e:
                import traceback
                self.log(f"[Modular] ✗ Probe error for '{module.name}': {e}\n"
                         f"{traceback.format_exc()}")
                probe._restore_all()
                self.model.train()
                continue

            # Take the type with the highest positive RYS delta
            best_type  = max(results, key=lambda t: results[t]["best_score"])
            r          = results[best_type]
            best_score = r["best_score"]   # positive = already good at task
            fs, fe     = r["final_start"], r["final_end"]
            fscore     = r["final_score"]

            self.log(
                f"[Modular] '{module.name}' RYS result: "
                f"best_type='{best_type}'  "
                f"best=[{r['best_start']},{r['best_end']}] "
                f"delta={best_score:+.5f}  "
                f"minimal=[{fs},{fe}] delta={fscore:+.5f}  "
                f"threshold={module.emergence_threshold:+.5f}"
            )

            if best_score >= module.emergence_threshold:
                self.log(
                    f"[Modular] ✦ EMERGENCE: '{module.name}' layers [{fs},{fe}] "
                    f"are already good at {best_type} "
                    f"(RYS delta={best_score:+.5f}) — finetuning further."
                )
                module.emergent_layers     = list(range(fs, fe + 1))
                module.last_triggered_step = global_step

                event = {
                    "step":          global_step,
                    "best_range":    [r["best_start"], r["best_end"]],
                    "best_delta":    round(best_score, 5),
                    "minimal_range": [fs, fe],
                    "minimal_delta": round(fscore, 5),
                    "probe_type":    best_type,
                    "acting_layers": module.emergent_layers,
                }

                try:
                    did_reload = self._finetune_module(
                        module, module.emergent_layers,
                        output_dir, global_step, max_length)
                    event["finetune_complete"] = True
                except Exception as e:
                    import traceback
                    self.log(f"[Modular] ✗ Finetune error for '{module.name}': {e}\n"
                             f"{traceback.format_exc()}")
                    event["finetune_complete"] = False
                    did_reload = False

                module.trigger_history.append(event)

                if did_reload:
                    reloaded = True
                    self.layers = _get_layer_stack(self.model)
                    self.N      = len(self.layers)
            else:
                self.log(
                    f"[Modular] '{module.name}' best delta={best_score:+.5f} "
                    f"below threshold {module.emergence_threshold:+.5f} — "
                    f"math capability not yet emerged"
                )

        self.model.train()
        return reloaded

    # ── Per-module finetune dispatch ─────────────────────────────────────────

    def _finetune_module(self, module: TrainingModule, emergent: List[int],
                         output_dir: str, global_step: int,
                         max_length: int) -> bool:
        """
        Run the module's configured finetune method on the exact emergent layers.

        emergent is the full list of layer indices that crossed the threshold —
        e.g. [3, 9, 14].  Each finetune method operates on these specific layers
        rather than the bounding-box range, so non-emergent layers in between
        are not touched.

        Returns True if a model reload occurred (stretch method).
        """
        method = module.finetune_method
        self.log(f"[Modular] Finetuning '{module.name}' via {method} "
                 f"on emergent layers {emergent} ...")

        if method == "lora":
            return self._ft_lora(module, emergent,
                                 output_dir, global_step, max_length)
        elif method == "junction":
            return self._ft_junction(module, emergent,
                                     output_dir, global_step, max_length)
        elif method == "full_layer":
            return self._ft_full_layer(module, emergent,
                                       output_dir, global_step, max_length)
        elif method == "stretch":
            return self._ft_stretch(module, emergent,
                                    output_dir, global_step, max_length)
        return False

    def _make_finetune_loader(self, module: TrainingModule,
                              max_length: int) -> DataLoader:
        ds = load_dataset_from_source(
            module.finetune_dataset, self.tokenizer, max_length)
        return DataLoader(ds, batch_size=1, shuffle=True, collate_fn=_collate)

    def _run_finetune_loop(self, module: TrainingModule, optimizer,
                           loader: DataLoader, epochs: int, label: str):
        """Shared inner training loop for all non-stretch finetune methods."""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer, T_max=max(len(loader) * epochs, 1))
        for ep in range(1, epochs + 1):
            self.model.train()
            ep_loss = 0.0
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out   = self.model(**batch)
                out.loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                ep_loss += out.loss.item()
            avg = ep_loss / max(len(loader), 1)
            self.log(f"[Modular] {label} ep {ep}/{epochs} loss={avg:.5f}")

    # ── lora ─────────────────────────────────────────────────────────────────

    def _ft_lora(self, module, emergent: List[int],
                 output_dir, step, max_length) -> bool:
        from torch.optim import AdamW
        if not module.lora_injected:
            # Inject LoRA into each emergent layer individually (not the full range)
            total_injected = []
            for idx in emergent:
                injected = inject_lora(
                    self.model, idx, idx,
                    rank=module.lora_rank, alpha=module.lora_alpha)
                total_injected.extend(injected)
            module.lora_injected = True
            self.log(f"[Modular] Injected LoRA into {len(total_injected)} modules "
                     f"across layers {emergent}")
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=module.finetune_lr)
        loader    = self._make_finetune_loader(module, max_length)
        self._run_finetune_loop(module, optimizer, loader,
                                module.finetune_epochs,
                                f"LoRA[{module.name}]")
        ckpt = os.path.join(output_dir, f"module_{module.name}_step{step}")
        self._save(ckpt)
        return False

    # ── junction ─────────────────────────────────────────────────────────────

    def _ft_junction(self, module, emergent: List[int],
                     output_dir, step, max_length) -> bool:
        from torch.optim import AdamW
        freeze_all(self.model)
        # For each emergent layer, train its immediate neighbours (the seam layers)
        junctions = set()
        for idx in emergent:
            if idx > 0:
                junctions.add(idx - 1)
            junctions.add(idx)
            if idx + 1 < self.N:
                junctions.add(idx + 1)
        for idx in junctions:
            unfreeze_layer(self.layers[idx])
        self.log(f"[Modular] Junction layers around {emergent}: {sorted(junctions)}")
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=module.finetune_lr)
        loader    = self._make_finetune_loader(module, max_length)
        self._run_finetune_loop(module, optimizer, loader,
                                module.finetune_epochs,
                                f"Junction[{module.name}]")
        for p in self.model.parameters():
            p.requires_grad = True
        ckpt = os.path.join(output_dir, f"module_{module.name}_step{step}")
        self._save(ckpt)
        return False

    # ── full_layer ───────────────────────────────────────────────────────────

    def _ft_full_layer(self, module, emergent: List[int],
                       output_dir, step, max_length) -> bool:
        from torch.optim import AdamW
        freeze_all(self.model)
        for idx in emergent:
            unfreeze_layer(self.layers[idx])
        self.log(f"[Modular] Full-layer finetune on exact emergent layers {emergent}")
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=module.finetune_lr)
        loader    = self._make_finetune_loader(module, max_length)
        self._run_finetune_loop(module, optimizer, loader,
                                module.finetune_epochs,
                                f"FullLayer[{module.name}]")
        for p in self.model.parameters():
            p.requires_grad = True
        ckpt = os.path.join(output_dir, f"module_{module.name}_step{step}")
        self._save(ckpt)
        return False

    # ── stretch ──────────────────────────────────────────────────────────────

    def _ft_stretch(self, module, emergent: List[int],
                    output_dir, step, max_length) -> bool:
        """
        Export → insert blank layers immediately after each emergent layer → reload.

        Inserts n_blank_layers after the highest-scored emergent layer only,
        to give the most active circuit room to grow without fragmenting the
        model with multiple insertion points in one pass.

        ⚠ Resets optimizer state and shifts all layer indices above the
          insertion point.
        """
        from layer_surgeon import LayerSurgeon
        interim_dir = os.path.join(output_dir, f"stretch_{module.name}_step{step}")
        self.log(f"[Modular] Stretch: exporting current model to {interim_dir} ...")
        self._save(interim_dir)

        surgeon = LayerSurgeon(interim_dir, device=str(self.device))

        # Insert after the last emergent layer — highest in the stack,
        # so earlier index shifts don't invalidate the insertion point.
        insert_after = max(emergent)
        for i in range(module.n_blank_layers):
            slot = surgeon.insert_blank_layer(insert_after + i)
            self.log(f"[Modular] Stretch: inserted blank layer at slot {slot} "
                     f"(after emergent layer {insert_after})")

        stretched_dir = interim_dir + "_stretched"
        surgeon.export(stretched_dir)
        self.log(f"[Modular] Stretch: reloading expanded model ({stretched_dir}) ...")

        self._load_model(stretched_dir, from_checkpoint=True)
        module.stretch_layer_offset += module.n_blank_layers
        self.log(f"[Modular] Stretch: model reloaded — now {self.N} layers. "
                 f"Emergent layers were {emergent}. "
                 f"⚠ Optimizer state reset.")
        return True

    # ── Save helper ──────────────────────────────────────────────────────────

    def _save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        self.log(f"[Modular] Saved → {directory}")

    # ── Main training loop ───────────────────────────────────────────────────

    def train(self,
              dataset_source,
              output_dir:    str,
              epochs:        int   = 20,
              batch_size:    int   = 2,
              grad_accum:    int   = 8,
              learning_rate: float = 1e-3,
              max_length:    int   = 256,
              warmup_ratio:  float = 0.05,
              save_every_n_steps: int = 500,
              ) -> dict:
        """
        Main training loop.

        Every `probe_interval` optimizer steps:
          1. All non-cooldown modules are probed.
          2. Any module that crosses its emergence threshold fires its
             configured finetune method.
          3. If a stretch reload occurred the optimizer is rebuilt from scratch
             (layer indices changed).  All other methods resume seamlessly.
        """
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        os.makedirs(output_dir, exist_ok=True)
        self.log("[Modular] Preparing general training dataset ...")
        ds     = load_dataset_from_source(dataset_source, self.tokenizer, max_length)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            collate_fn=_collate)

        total_steps  = math.ceil(len(ds) / (batch_size * grad_accum)) * epochs
        warmup_steps = int(total_steps * warmup_ratio)

        def _build_optimizer():
            """Build per-layer param groups for adaptive LR support."""
            layers      = _get_layer_stack(self.model)
            layer_ids   = set().union(*(set(id(p) for p in l.parameters())
                                        for l in layers))
            other       = [p for p in self.model.parameters()
                           if id(p) not in layer_ids]
            groups      = [{"params": other, "lr": learning_rate}]
            groups     += [{"params": list(l.parameters()), "lr": learning_rate}
                           for l in layers]
            opt = AdamW(groups, weight_decay=0.01, betas=(0.9, 0.95))
            sched = CosineAnnealingLR(opt, T_max=max(total_steps, 1),
                                      eta_min=learning_rate * 0.1)
            return opt, sched

        optimizer, scheduler = _build_optimizer()

        loss_history     = []
        global_step      = 0
        best_loss        = float("inf")
        best_ckpt        = None
        module_log: list = []

        self.log(f"[Modular] Starting: {epochs} epochs, {len(ds)} samples, "
                 f"probe every {self.probe_interval} steps, "
                 f"{len(self.modules)} modules active")

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

                    # Warmup
                    if global_step < warmup_steps:
                        factor = (global_step + 1) / max(warmup_steps, 1)
                        for pg in optimizer.param_groups:
                            pg["lr"] = learning_rate * factor

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # ── Module probe cycle ────────────────────────────────
                    if (self.probe_interval > 0
                            and global_step % self.probe_interval == 0):
                        reloaded = self._run_all_probes(
                            global_step, output_dir,
                            optimizer,
                            scheduler.get_last_lr()[0]
                                if hasattr(scheduler, "get_last_lr")
                                else learning_rate,
                            max_length,
                        )
                        # Stretch reload changes layer count — rebuild optimizer
                        if reloaded:
                            self.log("[Modular] Rebuilding optimizer after stretch reload")
                            optimizer, scheduler = _build_optimizer()

                        module_log.append({
                            "step": global_step,
                            "module_states": [m.to_dict() for m in self.modules],
                        })
                        self.model.train()

                    if save_every_n_steps and global_step % save_every_n_steps == 0:
                        self._save(os.path.join(output_dir, f"step_{global_step}"))

            avg = epoch_loss / max(len(loader), 1)
            loss_history.append({"epoch": epoch, "loss": round(avg, 6)})
            self.log(f"[Modular] Epoch {epoch}/{epochs} — loss={avg:.6f}")
            ckpt = os.path.join(output_dir, f"epoch_{epoch}")
            self._save(ckpt)
            if avg < best_loss:
                best_loss = avg
                best_ckpt = ckpt

        final = os.path.join(output_dir, "final")
        self._save(final)

        log_data = {
            "mode":            "modular_ground_up",
            "model_source":    self.model_source,
            "epochs":          epochs,
            "loss_history":    loss_history,
            "best_checkpoint": best_ckpt,
            "module_log":      module_log,
            "final_modules":   [m.to_dict() for m in self.modules],
        }
        with open(os.path.join(output_dir, "training_log.json"), "w") as f:
            json.dump(log_data, f, indent=2)

        self.log(f"[Modular] Done. Best checkpoint: {best_ckpt}")
        return log_data