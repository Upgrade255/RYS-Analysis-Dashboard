"""
model_trainer.py
================
Targeted fine-tuning for surgically modified transformer models.

Training modes (in priority order per project design):
  1. Junction fine-tune   — freeze everything, train only the two boundary layers
                            at the RYS duplication seam (dnhkng's hypothesis)
  2. LoRA on layer range  — low-rank adaptation on a selected span of layers
  3. Blank-layer growth   — train only newly inserted blank layers (neuro organ seeding)
  4. Full fine-tune       — all parameters unfrozen

All modes share the same DataLoader / training loop infrastructure.
Datasets: accepts (a) list of (prompt, response) tuples, (b) JSON file path,
          (c) HuggingFace dataset name.

Checkpointing: saves best-loss checkpoint + training log after each epoch.
"""

import os
import json
import math
import time
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """Simple prompt/response dataset."""

    def __init__(self, pairs: list, tokenizer, max_length: int = 512):
        self.samples = []
        for prompt, response in pairs:
            text = prompt + " " + response
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            # labels = input_ids (causal LM), prompt tokens masked to -100
            prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
            labels = enc.input_ids.clone()
            labels[0, :prompt_len] = -100          # mask prompt
            self.samples.append({
                "input_ids":      enc.input_ids[0],
                "attention_mask": enc.attention_mask[0],
                "labels":         labels[0],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        out[k] = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True,
            padding_value=(-100 if k == "labels" else 0)
        )
    return out


# Column name aliases: maps common HF dataset column names → (prompt_key, response_key)
_COLUMN_ALIASES = [
    ("prompt",      "response"),       # Dahoas/full-hh-rlhf, Dahoas/rm-static
    ("instruction", "output"),         # tatsu-lab/alpaca, garage-bAInd/Open-Platypus
    ("instruction", "response"),       # some instruction datasets
    ("question",    "answer"),         # openai/gsm8k, microsoft/orca-math-word-problems-200k
    ("question",    "response"),       # Open-Orca/OpenOrca
    ("query",       "response"),       # meta-math/MetaMathQA
    ("input",       "output"),         # various
    ("text",        "response"),       # fallback
]

def _detect_columns(row: dict):
    """Return (prompt_key, response_key) from a sample row, or raise."""
    for pk, rk in _COLUMN_ALIASES:
        if pk in row and rk in row:
            return pk, rk
    raise KeyError(
        f"Could not find prompt/response columns in dataset. "
        f"Found columns: {list(row.keys())}. "
        f"Supported pairs: {_COLUMN_ALIASES}"
    )


def load_dataset_from_source(source, tokenizer, max_length=512):
    """
    source: list of (prompt, response) tuples
          | path to JSON file containing [{"prompt":..,"response":..}]
          | HuggingFace dataset name string

    Automatically detects common column name variants across HF datasets:
      prompt/response, instruction/output, instruction/response,
      question/response, input/output.
    """
    if isinstance(source, list):
        pairs = source
    elif isinstance(source, str) and source.endswith(".json"):
        with open(source) as f:
            data = json.load(f)
        # Support both {prompt, response} and {instruction, output} in JSON too
        if data:
            pk, rk = _detect_columns(data[0])
        pairs = [(d[pk], d[rk]) for d in data]
    else:
        from datasets import load_dataset as hf_load
        ds = hf_load(source, split="train")
        first = dict(ds[0])
        pk, rk = _detect_columns(first)
        print(f"[Dataset] Using columns: prompt='{pk}', response='{rk}'")
        pairs = [(row[pk], row[rk]) for row in ds]
    return PairDataset(pairs, tokenizer, max_length=max_length)


# ──────────────────────────────────────────────────────────────────────────────
# Layer utilities
# ──────────────────────────────────────────────────────────────────────────────

def _get_layer_stack(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if (hasattr(model, "model") and hasattr(model.model, "decoder")
            and hasattr(model.model.decoder, "layers")):
        return model.model.decoder.layers
    raise RuntimeError("Could not locate transformer layers in model architecture")


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_layer(layer_module):
    for p in layer_module.parameters():
        p.requires_grad = True


def unfreeze_range(model, start: int, end: int):
    """Unfreeze layers[start:end+1] (inclusive)."""
    layers = _get_layer_stack(model)
    for i in range(start, min(end + 1, len(layers))):
        unfreeze_layer(layers[i])


def count_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# ──────────────────────────────────────────────────────────────────────────────
# LoRA injection
# ──────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Minimal LoRA adapter wrapping an nn.Linear."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_f = linear.in_features
        out_f = linear.out_features
        # Initialise on the same device as the wrapped layer
        device = next(linear.parameters()).device
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=device))
        # Freeze original weights
        for p in linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = self.linear(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora


def inject_lora(model, start: int, end: int, rank: int = 8, alpha: float = 16.0,
                target_modules=("q_proj", "v_proj", "k_proj", "o_proj")):
    """
    Replace target Linear layers in layers[start:end+1] with LoRALinear wrappers.
    Returns list of injected module paths for reporting.
    """
    layers = _get_layer_stack(model)
    injected = []
    for i in range(start, min(end + 1, len(layers))):
        layer = layers[i]
        for name, module in list(layer.named_modules()):
            if isinstance(module, nn.Linear):
                # Match by module name suffix
                if any(name.endswith(t) or t in name for t in target_modules):
                    # Navigate to parent and swap
                    parts = name.split(".")
                    parent = layer
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], LoRALinear(module, rank=rank, alpha=alpha))
                    injected.append(f"layer[{i}].{name}")
    return injected


def merge_lora_weights(model, start: int, end: int):
    """
    Merge LoRA adapters back into base weights and replace with plain nn.Linear.
    Call before export if you want a clean checkpoint.
    """
    layers = _get_layer_stack(model)
    for i in range(start, min(end + 1, len(layers))):
        layer = layers[i]
        for name, module in list(layer.named_modules()):
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    delta = (module.lora_B @ module.lora_A) * module.scaling
                    module.linear.weight += delta
                parts = name.split(".")
                parent = layer
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], module.linear)


# ──────────────────────────────────────────────────────────────────────────────
# Training engine
# ──────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Targeted fine-tuning for surgically modified (or plain) transformer models.

    Parameters
    ----------
    model_path : str
        Path to a HF checkpoint (including surgically exported models).
    mode : str
        "junction"   — freeze all, train only the two layers at start/end of RYS block
        "lora"       — inject LoRA into layer range [lora_start, lora_end]
        "blank"      — freeze all, train only layers whose source is "blank"
                       (reads surgery_manifest.json from model_path)
        "full"       — train all parameters
    """

    MODES = ("junction", "lora", "blank", "full")

    def __init__(
        self,
        model_path: str,
        mode: str = "junction",
        # Junction / range params
        junction_start: int = 0,
        junction_end: int = 0,
        # LoRA params
        lora_start: int = 0,
        lora_end: int = 0,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_targets: tuple = ("q_proj", "v_proj"),
        # General
        device: str = "cuda",
        log_callback=None,      # callable(str) for UI progress streaming
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")

        self.model_path = model_path
        self.mode = mode
        self.junction_start = junction_start
        self.junction_end = junction_end
        self.lora_start = lora_start
        self.lora_end = lora_end
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_targets = lora_targets
        self.device = torch.device(
            device if torch.cuda.is_available() and "cuda" in device else "cpu"
        )
        self.log = log_callback or print
        self.lora_injected = []
        self._load_model()

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.log(f"[Trainer] Loading {self.model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,   # fp32 for stable training
            device_map=None,
        ).to(self.device)
        self.model.config.use_cache = False
        self._apply_freeze_strategy()

    def _apply_freeze_strategy(self):
        layers = _get_layer_stack(self.model)
        N = len(layers)

        if self.mode == "full":
            self.log("[Trainer] Mode=full — all parameters trainable")
            return

        if self.mode == "junction":
            freeze_all(self.model)
            # The two "junction" layers: the layer just before the RYS block starts
            # and the layer just after it ends (dnhkng's hypothesis)
            junctions = set()
            if self.junction_start > 0:
                junctions.add(self.junction_start - 1)
            junctions.add(self.junction_start)
            junctions.add(self.junction_end)
            if self.junction_end + 1 < N:
                junctions.add(self.junction_end + 1)
            for idx in junctions:
                if idx < N:
                    unfreeze_layer(layers[idx])
            self.log(f"[Trainer] Mode=junction — training layers {sorted(junctions)}")

        elif self.mode == "lora":
            freeze_all(self.model)
            self.lora_injected = inject_lora(
                self.model, self.lora_start, self.lora_end,
                rank=self.lora_rank, alpha=self.lora_alpha,
                target_modules=self.lora_targets,
            )
            # LoRALinear adapters have requires_grad=True by construction
            self.log(f"[Trainer] Mode=lora — injected LoRA into {len(self.lora_injected)} modules "
                     f"(layers {self.lora_start}–{self.lora_end}, rank={self.lora_rank})")

        elif self.mode == "blank":
            freeze_all(self.model)
            manifest_path = os.path.join(self.model_path, "surgery_manifest.json")
            if not os.path.exists(manifest_path):
                self.log("[Trainer] Warning: no surgery_manifest.json — nothing to unfreeze")
                return
            with open(manifest_path) as f:
                manifest = json.load(f)
            blank_slots = [e["slot"] for e in manifest["plan"] if e["type"] == "blank"]
            for slot in blank_slots:
                if slot < len(layers):
                    unfreeze_layer(layers[slot])
            self.log(f"[Trainer] Mode=blank — training blank layers at slots {blank_slots}")

        trainable, total = count_trainable(self.model)
        pct = 100.0 * trainable / max(total, 1)
        self.log(f"[Trainer] Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")

    def train(
        self,
        dataset_source,         # list of (prompt,response) | .json path | HF dataset name
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 1,
        grad_accum: int = 4,
        learning_rate: float = 2e-4,
        max_length: int = 512,
        warmup_steps: int = 10,
        save_every_epoch: bool = True,
        merge_lora_on_save: bool = False,
    ) -> dict:
        """
        Run the training loop.
        Returns a log dict with loss history and final checkpoint path.
        """
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        os.makedirs(output_dir, exist_ok=True)

        self.log("[Trainer] Preparing dataset ...")
        ds = load_dataset_from_source(dataset_source, self.tokenizer, max_length)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            collate_fn=_collate)
        steps_per_epoch = math.ceil(len(loader) / grad_accum)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        total_steps = steps_per_epoch * epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

        loss_history = []
        best_loss = float("inf")
        best_ckpt = None
        global_step = 0

        self.log(f"[Trainer] Starting training: {epochs} epochs, "
                 f"{len(ds)} samples, lr={learning_rate}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss / grad_accum
                loss.backward()
                epoch_loss += loss.item() * grad_accum

                if (step + 1) % grad_accum == 0:
                    nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    # Linear warmup
                    if global_step < warmup_steps:
                        for pg in optimizer.param_groups:
                            pg["lr"] = learning_rate * (global_step + 1) / warmup_steps
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            avg_loss = epoch_loss / max(len(loader), 1)
            loss_history.append({"epoch": epoch, "loss": round(avg_loss, 6)})
            self.log(f"[Trainer] Epoch {epoch}/{epochs} — loss={avg_loss:.6f}")

            if save_every_epoch:
                ckpt_dir = os.path.join(output_dir, f"epoch_{epoch}")
                self._save(ckpt_dir, merge_lora_on_save)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_ckpt = ckpt_dir

        # Final save
        final_dir = os.path.join(output_dir, "final")
        self._save(final_dir, merge_lora_on_save)
        if best_ckpt is None or best_loss >= avg_loss:
            best_ckpt = final_dir

        log_path = os.path.join(output_dir, "training_log.json")
        log_data = {
            "mode": self.mode,
            "model_path": self.model_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss_history": loss_history,
            "best_checkpoint": best_ckpt,
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        self.log(f"[Trainer] Done. Best checkpoint: {best_ckpt} (loss={best_loss:.6f})")
        return log_data

    def _save(self, directory: str, merge_lora: bool = False):
        os.makedirs(directory, exist_ok=True)
        if merge_lora and self.mode == "lora" and self.lora_injected:
            self.log(f"[Trainer] Merging LoRA weights before save ...")
            merge_lora_weights(self.model, self.lora_start, self.lora_end)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        self.log(f"[Trainer] Saved checkpoint → {directory}")
