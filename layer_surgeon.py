"""
layer_surgeon.py
================
Physical layer manipulation and model export.
Operates on saved checkpoints — never touches the live RYS engine.

Supported operations:
  - Reorder layers (shuffle)
  - Duplicate layers (real weight copies)
  - Delete layers
  - Insert new randomly-initialized layers
  - Mask attention heads / MLP sublayers within a layer
  - Export modified model as HuggingFace checkpoint

Quantization loading support:
  - bitsandbytes 4-bit / 8-bit  (load_in_4bit / load_in_8bit)
  - AWQ / GPTQ                   (AutoAWQ / auto-gptq via HF)
  - GGUF / llama.cpp             (llama-cpp-python, read-only — export not supported)
  - ExLlamaV2                    (exllamav2, read-only — export not supported)

Note: GGUF and ExLlamaV2 models load for RYS sweep/inference testing only.
Surgery + export requires a HF-format model (including bnb/AWQ/GPTQ quantized).

Export strategy (why we don't deepcopy the live model):
  Mutating a live HuggingFace model object (swap ModuleList, then save_pretrained)
  corrupts HF's internal auto-class metadata — model_type, architectures, auto_map
  — causing "model of type X to instantiate model of type ''" on reload.

  Instead:
    1. Build a new state_dict by remapping layer-index keys per the plan
    2. deepcopy the *config only* and patch all layer-count fields
    3. AutoModelForCausalLM.from_config() — fresh, cleanly-registered model
    4. load_state_dict() into the fresh model
    5. save_pretrained() — metadata intact, no corruption
"""

import copy
import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Quantization format detection
# ──────────────────────────────────────────────────────────────────────────────

def _detect_quant_format(model_path: str) -> str:
    p = Path(model_path)
    if not p.exists():
        return "hf"
    names = [f.name.lower() for f in p.iterdir()]
    if any(n.endswith(".gguf") for n in names):
        return "gguf"
    if any("exl2" in n or "exllamav2" in n for n in names):
        return "exl2"
    config_path = p / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        qcfg = cfg.get("quantization_config", {})
        if "awq" in qcfg.get("quant_type", "").lower():
            return "awq"
        if "gptq" in qcfg.get("quant_type", "").lower() or qcfg.get("bits"):
            return "gptq"
    return "hf"


# ──────────────────────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load_hf_model(model_name: str, quant_mode: Optional[str], device: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {}
    if quant_mode == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quant_mode == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant_mode in ("awq", "gptq"):
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    else:
        use_cuda = torch.cuda.is_available() and "cuda" in device
        kwargs["torch_dtype"] = torch.float16 if use_cuda else torch.float32
        kwargs["device_map"] = "auto" if use_cuda else None

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def _load_gguf_model(model_path: str):
    try:
        from llama_cpp import Llama
    except ImportError:
        raise RuntimeError("llama-cpp-python not installed. "
                           "pip install llama-cpp-python --break-system-packages")
    gguf_files = list(Path(model_path).glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf file in {model_path}")
    return Llama(model_path=str(gguf_files[0]), n_ctx=2048, verbose=False)


def _load_exl2_model(model_path: str):
    try:
        from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
        from exllamav2.tokenizer import ExLlamaV2Tokenizer
    except ImportError:
        raise RuntimeError("exllamav2 not installed. "
                           "See https://github.com/turboderp/exllamav2")
    cfg = ExLlamaV2Config()
    cfg.model_dir = model_path
    cfg.prepare()
    model = ExLlamaV2(cfg)
    cache = ExLlamaV2Cache(model, lazy=True)
    model.load_autosplit(cache)
    return model, ExLlamaV2Tokenizer(cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Layer stack + key-prefix helpers
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


def _detect_layer_prefix(model, state_dict: dict) -> str:
    """
    Infer the state-dict key prefix for indexed transformer layers.
    E.g. "model.layers." for Qwen/Llama/Mistral, "transformer.h." for GPT-2.

    Strategy: take the first named parameter of layer[0], find its full key
    in the state dict (it will look like "{prefix}0.{param_name}"), and strip
    the "0.{param_name}" suffix.
    """
    layers = _get_layer_stack(model)
    if not layers:
        raise RuntimeError("Empty layer stack")

    for param_name, _ in layers[0].named_parameters():
        suffix = f"0.{param_name}"
        for k in state_dict.keys():
            if k.endswith(suffix):
                prefix = k[: -len(suffix)]
                # Sanity check: layer 1 should also exist with this prefix
                if any(kk.startswith(f"{prefix}1.") for kk in state_dict.keys()):
                    return prefix
    raise RuntimeError(
        "Could not determine layer key prefix. "
        "First layer parameters not found in state dict."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Config patching
# ──────────────────────────────────────────────────────────────────────────────

def _patch_layer_count(config, n: int):
    """
    Update every field that encodes the layer count.
    Handles the standard field, all known aliases, and nested sub-configs
    (e.g. text_config in Gemma / PaliGemma multimodal models).
    """
    config.num_hidden_layers = n
    for field in ("num_layers", "n_layer", "num_decoder_layers"):
        if hasattr(config, field) and getattr(config, field) is not None:
            setattr(config, field, n)
    if hasattr(config, "text_config") and config.text_config is not None:
        _patch_layer_count(config.text_config, n)


# ──────────────────────────────────────────────────────────────────────────────
# LayerSurgeon
# ──────────────────────────────────────────────────────────────────────────────

class LayerSurgeon:
    """
    Performs physical (weight-copying) surgery on a transformer model.

    All operations are staged on an internal plan list. Nothing is written
    to disk or committed to a live model until .export() is called.

    Typical workflow:
        surgeon = LayerSurgeon("Qwen/Qwen2-7B")
        surgeon.duplicate_layer(20)
        surgeon.insert_blank_layer(21)
        surgeon.export("./my_frankenstein")
        # → clean HF checkpoint, loadable by AutoModelForCausalLM / RYSEngine
    """

    def __init__(self, model_name: str, quant_mode: Optional[str] = None,
                 device: str = "cuda"):
        self.model_name = model_name
        self.quant_mode = quant_mode
        self.device = device
        self._export_capable = True

        fmt = _detect_quant_format(model_name)

        if fmt == "gguf":
            print("[Surgeon] Detected GGUF — inference only (no export)")
            self._gguf_model = _load_gguf_model(model_name)
            self._export_capable = False
            self.model = self.tokenizer = self.layers = None
            return

        if fmt == "exl2":
            print("[Surgeon] Detected ExLlamaV2 — inference only (no export)")
            self.model, self.tokenizer = _load_exl2_model(model_name)
            self._export_capable = False
            self.layers = None
            return

        effective_quant = quant_mode or (fmt if fmt in ("awq", "gptq") else None)
        print(f"[Surgeon] Loading {model_name} (quant={effective_quant or 'none'}) ...")
        self.model, self.tokenizer = _load_hf_model(model_name, effective_quant, device)
        self.layers = _get_layer_stack(self.model)

        # Snapshot full state dict on CPU — this is our immutable source of truth
        self._orig_state_dict = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}

        # Detect the layer key prefix (e.g. "model.layers.")
        self._layer_prefix = _detect_layer_prefix(self.model, self._orig_state_dict)
        print(f"[Surgeon] Layer key prefix: '{self._layer_prefix}'")

        # Per-layer state dicts (CPU tensors) — source for remapping on export
        self._layer_state_dicts = []
        for layer in self.layers:
            self._layer_state_dicts.append(
                {k: v.cpu().clone() for k, v in layer.state_dict().items()}
            )

        # Build initial plan
        self._layer_plan = [
            {"type": "original", "src": i, "sd": self._layer_state_dicts[i]}
            for i in range(len(self.layers))
        ]

        # Pending mask operations (applied at export time to state dict copies)
        self._masked_heads: dict = {}   # slot -> set of head indices
        self._masked_mlp: set   = set() # set of slots

        print(f"[Surgeon] Ready — {len(self._layer_plan)} layers")

    # ── Plan inspection ──────────────────────────────────────────────────────

    def get_plan(self):
        """Return list of dicts describing the current layer order."""
        rows = []
        for i, entry in enumerate(self._layer_plan):
            label = {
                "original":  f"L{entry['src']}",
                "duplicate": f"L{entry['src']} (copy)",
                "blank":     "NEW (blank)",
            }[entry["type"]]
            rows.append({
                "slot": i,
                "label": label,
                "type": entry["type"],
                "src": entry.get("src", -1),
            })
        return rows

    # ── Layer operations ─────────────────────────────────────────────────────

    def duplicate_layer(self, src_slot: int) -> int:
        """
        Deep-copy the layer at src_slot and insert it immediately after.
        Returns the slot index of the new copy.
        """
        self._require_hf()
        src = self._layer_plan[src_slot]
        new_sd = {k: v.clone() for k, v in src["sd"].items()}
        self._layer_plan.insert(src_slot + 1, {
            "type": "duplicate",
            "src":  src["src"],
            "sd":   new_sd,
        })
        print(f"[Surgeon] Duplicated slot {src_slot} → new copy at slot {src_slot + 1}")
        return src_slot + 1

    def delete_layer(self, slot: int):
        """Remove the layer at slot from the plan."""
        self._require_hf()
        if len(self._layer_plan) <= 1:
            raise ValueError("Cannot delete the last layer")
        removed = self._layer_plan.pop(slot)
        print(f"[Surgeon] Deleted slot {slot} ({removed['type']}, src={removed.get('src', -1)})")

    def move_layer(self, from_slot: int, to_slot: int):
        """Move the layer at from_slot to to_slot (reorder)."""
        self._require_hf()
        n = len(self._layer_plan)
        if not (0 <= from_slot < n and 0 <= to_slot < n):
            raise IndexError(f"Slot out of range [0, {n - 1}]")
        entry = self._layer_plan.pop(from_slot)
        self._layer_plan.insert(to_slot, entry)
        print(f"[Surgeon] Moved slot {from_slot} → {to_slot}")

    def insert_blank_layer(self, after_slot: int) -> int:
        """
        Insert a randomly-initialized layer after after_slot.
        Architecture mirrors layer 0; all weights are freshly initialized.
        Returns the slot index of the new layer.
        """
        self._require_hf()
        ref_sd = self._layer_state_dicts[0]
        blank_sd = {}
        for k, v in ref_sd.items():
            t = torch.empty_like(v)
            if v.dim() >= 2:
                nn.init.kaiming_uniform_(t)
            else:
                nn.init.zeros_(t)
            blank_sd[k] = t
        insert_at = after_slot + 1
        self._layer_plan.insert(insert_at, {
            "type": "blank",
            "src":  -1,
            "sd":   blank_sd,
        })
        print(f"[Surgeon] Inserted blank layer at slot {insert_at}")
        return insert_at

    def mask_attention_heads(self, slot: int, head_indices: list):
        """Schedule zeroing of specific attention heads at export time."""
        self._require_hf()
        self._masked_heads[slot] = set(head_indices)
        print(f"[Surgeon] Scheduled head mask: slot {slot}, heads {head_indices}")

    def mask_mlp(self, slot: int):
        """Schedule zeroing of the MLP sublayer at export time."""
        self._require_hf()
        self._masked_mlp.add(slot)
        print(f"[Surgeon] Scheduled MLP zero: slot {slot}")

    def reset_plan(self):
        """Restore plan to original order and clear all masks."""
        self._require_hf()
        self._layer_plan = [
            {"type": "original", "src": i, "sd": self._layer_state_dicts[i]}
            for i in range(len(self.layers))
        ]
        self._masked_heads.clear()
        self._masked_mlp.clear()
        print("[Surgeon] Plan reset to original")

    # ── Export ───────────────────────────────────────────────────────────────

    def export(self, output_dir: str, save_tokenizer: bool = True) -> str:
        """
        Build and save a clean HF checkpoint from the current layer plan.

        We write files directly rather than using a model object because:
          - from_config() triggers architecture __init__ that breaks on layer
            count changes (RoPE cache, sliding window attention, etc.)
          - deepcopy(live_model).save_pretrained() corrupts auto-class metadata
        A valid HF checkpoint is just files: config.json + weights + tokenizer.
        """
        import traceback as _tb

        self._require_hf()
        os.makedirs(output_dir, exist_ok=True)
        n_new  = len(self._layer_plan)
        prefix = self._layer_prefix

        # ── Step 1: Build remapped state dict ────────────────────────────────
        try:
            print(f"[Surgeon] Step 1/5 — building state dict "
                  f"({n_new} layers, prefix='{prefix}') ...")
            new_sd: dict = {}
            for k, v in self._orig_state_dict.items():
                if not k.startswith(prefix):
                    new_sd[k] = v.clone()
            for new_idx, entry in enumerate(self._layer_plan):
                layer_sd = {k: v.clone() for k, v in entry["sd"].items()}
                if new_idx in self._masked_heads:
                    layer_sd = self._apply_head_mask(
                        layer_sd, self._masked_heads[new_idx])
                if new_idx in self._masked_mlp:
                    layer_sd = self._apply_mlp_mask(layer_sd)
                for sub_key, val in layer_sd.items():
                    new_sd[f"{prefix}{new_idx}.{sub_key}"] = val
            print(f"[Surgeon] Step 1 done — {len(new_sd)} tensors total")
        except Exception:
            raise RuntimeError(
                f"Step 1 (build state dict) failed:\n{_tb.format_exc()}")

        # ── Step 2: Write config.json ─────────────────────────────────────────────
        # We serialise via model.config.to_dict() (plain dict, no class deps).
        # We also remap any per-layer LIST fields (e.g. Qwen2's layer_types)
        # whose length equals n_orig — same remapping logic as the state dict.
        try:
            print("[Surgeon] Step 2/5 — writing config.json ...")
            cfg_dict = self.model.config.to_dict()
            n_orig = len(self.layers)

            # Patch scalar layer-count fields
            for field in ("num_hidden_layers", "num_layers", "n_layer",
                          "num_decoder_layers"):
                if field in cfg_dict:
                    cfg_dict[field] = n_new

            # Remap per-layer LIST fields (length == n_orig) through the plan.
            # Handles layer_types (Qwen2/3), per-layer sliding_window, etc.
            # Generic: any list of the right length gets remapped automatically.
            for field, val in list(cfg_dict.items()):
                if isinstance(val, list) and len(val) == n_orig:
                    new_list = []
                    for entry in self._layer_plan:
                        src = entry["src"]
                        new_list.append(val[src] if 0 <= src < len(val) else val[0])
                    cfg_dict[field] = new_list
                    print(f"[Surgeon] Remapped per-layer config field '{field}' "
                          f"({n_orig} -> {n_new} entries)")

            # Recurse into nested text_config (Gemma, PaliGemma, etc.)
            if "text_config" in cfg_dict and isinstance(cfg_dict["text_config"], dict):
                tc = cfg_dict["text_config"]
                for field in ("num_hidden_layers", "num_layers", "n_layer",
                              "num_decoder_layers"):
                    if field in tc:
                        tc[field] = n_new
                for field, val in list(tc.items()):
                    if isinstance(val, list) and len(val) == n_orig:
                        new_list = []
                        for entry in self._layer_plan:
                            src = entry["src"]
                            new_list.append(val[src] if 0 <= src < len(val) else val[0])
                        tc[field] = new_list

            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(cfg_dict, f, indent=2)
            print("[Surgeon] Step 2 done")
        except Exception:
            raise RuntimeError(
                f"Step 2 (write config) failed:\n{_tb.format_exc()}")

        # ── Step 3: Write weights ─────────────────────────────────────────────
        try:
            print("[Surgeon] Step 3/5 — writing weights ...")
            self._save_state_dict(new_sd, output_dir)
            print("[Surgeon] Step 3 done")
        except Exception:
            raise RuntimeError(
                f"Step 3 (write weights) failed:\n{_tb.format_exc()}")

        # ── Step 4: Copy generation_config.json if it exists ─────────────────
        try:
            print("[Surgeon] Step 4/5 — copying generation config ...")
            import shutil
            src_dir = Path(self.model_name)
            if src_dir.exists():
                gc_src = src_dir / "generation_config.json"
                if gc_src.exists():
                    shutil.copy(gc_src,
                                os.path.join(output_dir, "generation_config.json"))
            else:
                # HF hub model — try fetching via transformers
                try:
                    from transformers import GenerationConfig
                    gen = GenerationConfig.from_pretrained(self.model_name)
                    gen.save_pretrained(output_dir)
                except Exception:
                    pass  # non-fatal — model will use default generation config
            print("[Surgeon] Step 4 done")
        except Exception:
            print(f"[Surgeon] Step 4 warning (non-fatal): {_tb.format_exc()}")

        # ── Step 5: Tokenizer ─────────────────────────────────────────────────
        try:
            print("[Surgeon] Step 5/5 — saving tokenizer ...")
            if save_tokenizer and self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            print("[Surgeon] Step 5 done")
        except Exception:
            raise RuntimeError(
                f"Step 5 (save tokenizer) failed:\n{_tb.format_exc()}")

        # ── Manifest ──────────────────────────────────────────────────────────
        manifest = {
            "source_model":        self.model_name,
            "num_layers_original": len(self.layers),
            "num_layers_exported": n_new,
            "layer_prefix":        prefix,
            "plan":                self.get_plan(),
            "masked_heads":        {str(k): list(v)
                                    for k, v in self._masked_heads.items()},
            "masked_mlp_slots":    list(self._masked_mlp),
        }
        with open(os.path.join(output_dir, "surgery_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[Surgeon] Export complete → {output_dir}")
        return output_dir

    # ── Weight serialisation ──────────────────────────────────────────────────

    @staticmethod
    def _save_state_dict(state_dict: dict, output_dir: str):
        """
        Save state_dict to output_dir as safetensors (preferred) or pytorch .bin.
        For large models (>10 GB) automatically shards into 5 GB chunks.
        """
        # Estimate total size in bytes
        total_bytes = sum(v.nelement() * v.element_size() for v in state_dict.values())
        shard_bytes = 5 * 1024 ** 3   # 5 GB per shard

        try:
            from safetensors.torch import save_file as st_save
            _use_safetensors = True
        except ImportError:
            _use_safetensors = False

        if total_bytes <= shard_bytes:
            # Single file
            if _use_safetensors:
                path = os.path.join(output_dir, "model.safetensors")
                # safetensors requires contiguous float tensors
                contiguous = {k: v.contiguous() for k, v in state_dict.items()}
                st_save(contiguous, path)
                print(f"[Surgeon] Saved model.safetensors "
                      f"({total_bytes / 1024**3:.2f} GB)")
            else:
                path = os.path.join(output_dir, "pytorch_model.bin")
                torch.save(state_dict, path)
                print(f"[Surgeon] Saved pytorch_model.bin "
                      f"({total_bytes / 1024**3:.2f} GB)")
        else:
            # Sharded save
            keys    = list(state_dict.keys())
            shards  = []
            current_shard: dict = {}
            current_size = 0
            for k in keys:
                v    = state_dict[k]
                size = v.nelement() * v.element_size()
                if current_size + size > shard_bytes and current_shard:
                    shards.append(current_shard)
                    current_shard = {}
                    current_size  = 0
                current_shard[k] = v
                current_size += size
            if current_shard:
                shards.append(current_shard)

            index = {"metadata": {"total_size": total_bytes}, "weight_map": {}}
            ext   = "safetensors" if _use_safetensors else "bin"
            for i, shard in enumerate(shards):
                fname = f"model-{i+1:05d}-of-{len(shards):05d}.{ext}"
                fpath = os.path.join(output_dir, fname)
                if _use_safetensors:
                    st_save({k: v.contiguous() for k, v in shard.items()}, fpath)
                else:
                    torch.save(shard, fpath)
                for k in shard:
                    index["weight_map"][k] = fname

            idx_name = (f"model.safetensors.index.json" if _use_safetensors
                        else "pytorch_model.bin.index.json")
            with open(os.path.join(output_dir, idx_name), "w") as f:
                json.dump(index, f, indent=2)
            print(f"[Surgeon] Saved {len(shards)} shards "
                  f"({total_bytes / 1024**3:.2f} GB total)")

    # ── State-dict mask helpers ───────────────────────────────────────────────

    @staticmethod
    def _apply_head_mask(layer_sd: dict, head_indices: set) -> dict:
        """
        Zero columns of the attention output-projection matrix corresponding
        to the specified head indices.

        Searches for keys ending in o_proj.weight / out_proj.weight / c_proj.weight.
        head_dim is inferred from weight column count divided by (max_head + 1).
        """
        out_key = None
        for k in layer_sd:
            if k.endswith(".weight") and any(
                n in k for n in ("o_proj", "out_proj", "c_proj")
            ):
                out_key = k
                break
        if out_key is None:
            print("[Surgeon] Warning: output-projection key not found — head mask skipped")
            return layer_sd

        w = layer_sd[out_key].clone()          # shape: [out_dim, in_dim]
        n_heads_inferred = max(head_indices) + 1
        head_dim = w.shape[1] // n_heads_inferred
        for h in head_indices:
            w[:, h * head_dim : (h + 1) * head_dim] = 0.0
        layer_sd = dict(layer_sd)
        layer_sd[out_key] = w
        return layer_sd

    @staticmethod
    def _apply_mlp_mask(layer_sd: dict) -> dict:
        """Zero all weight/bias tensors whose key contains 'mlp' or 'ffn'."""
        return {
            k: (torch.zeros_like(v) if ("mlp" in k.lower() or "ffn" in k.lower()) else v)
            for k, v in layer_sd.items()
        }

    # ── Guard ─────────────────────────────────────────────────────────────────

    def _require_hf(self):
        if not self._export_capable:
            raise RuntimeError(
                "This model (GGUF/ExLlamaV2) is inference-only. "
                "Surgery and export require a HuggingFace-format model."
            )
