"""
rys_engine.py — RYS sweep engine
https://dnhkng.github.io/posts/rys/  (CC BY 4.0)

Index convention: (start, end) where end is the LAST layer included in the loop.
Article uses (i, j) where j is the first layer AFTER the block → code.end = article.j - 1
Optimal Qwen2-72B config from the article (45, 52) → apply_rys(start=45, end=51).

run_test() returns mean log-prob per answer token (always negative; higher = better).
sweep() delta matrix = score(i,j) - baseline; positive = improvement over unmodified model.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


class RYSEngine:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(
            device if torch.cuda.is_available() and "cuda" in device else "cpu"
        )

        print(f"Loading model {model_name} on {self.device} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"  # required for correct answer-token offsets

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
            device_map="auto" if "cuda" in str(self.device) else None,
        )

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layer_stack = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layer_stack = self.model.transformer.h
        elif (hasattr(self.model, "model")
              and hasattr(self.model.model, "decoder")
              and hasattr(self.model.model.decoder, "layers")):
            layer_stack = self.model.model.decoder.layers
        else:
            raise RuntimeError("Could not locate transformer layers in model architecture")

        self.layers = layer_stack
        self.N = len(self.layers)
        self.original_forwards = [layer.forward for layer in self.layers]

    def restore_layers(self):
        """Remove patched instance-level forwards so class methods take over cleanly."""
        for layer in self.layers:
            try:
                del layer.forward
            except AttributeError:
                pass

    def _detect_layer_output_format(self) -> bool:
        """Return True if layers return tuples, False if they return plain tensors."""
        model_type = getattr(self.model.config, "model_type", "").lower()
        if model_type in {"qwen2", "qwen3", "qwen3_5", "qwen3.5"}:
            return False
        return True

    def disable_layer(self, layer_idx: int):
        """Replace a layer's forward with an identity pass-through for ablation."""
        if not hasattr(self, "_layer_returns_tuple") or self._layer_returns_tuple is None:
            self._layer_returns_tuple = self._detect_layer_output_format()

        engine_self = self

        def skip_forward(hidden_states, *args, **kwargs):
            if engine_self._layer_returns_tuple is None:
                orig = engine_self.original_forwards[layer_idx]
                with torch.no_grad():
                    sample = orig(hidden_states, *args, **kwargs)
                engine_self._layer_returns_tuple = isinstance(sample, (tuple, list))
                if engine_self._layer_returns_tuple:
                    return (hidden_states,) + tuple(sample[1:])
                return hidden_states
            if engine_self._layer_returns_tuple:
                return (hidden_states,)
            return hidden_states

        self.layers[layer_idx].forward = skip_forward

    def apply_rys(self, start: int, end: int, loops: int):
        """
        Repeat layers [start..end] an extra `loops` times after normal execution.
        Execution path (loops=1): [0..end, start..end, end+1..N-1]
        Only the end layer is patched; all others run their original class methods.
        """
        origs = self.original_forwards
        s, e, L = int(start), int(end), int(loops)

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

    def run_test(self, tests):
        """Return mean log-prob per answer token across all (question, answer) pairs."""
        prompts    = [q + " " for q, _ in tests]
        answers    = [a       for _, a in tests]
        full_texts = [p + a   for p, a in zip(prompts, answers)]

        full_inputs = self.tokenizer(
            full_texts, return_tensors="pt", padding=True,
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
        for i in range(len(tests)):
            ans_ids = answer_ids[i]
            if not ans_ids:
                continue
            logprob = 0.0
            for j, token in enumerate(ans_ids):
                pos = prompt_lengths[i] + j - 1
                if pos < 0 or pos >= logits.shape[1]:
                    continue
                logprob += torch.log_softmax(logits[i, pos], dim=-1)[token].item()
            scores.append(logprob / len(ans_ids))

        return float(sum(scores) / len(scores)) if scores else 0.0

    def sweep(self, tests, verbose=True):
        """
        Run all valid (start, end) pairs with loops=1.
        Returns (baseline, matrix) where matrix[i,j] = score(i,j) - baseline.
        """
        baseline = self.run_test(tests)
        matrix   = np.zeros((self.N, self.N), dtype=float)

        for i in range(self.N):
            for j in range(i + 1, self.N):
                self.restore_layers()
                self.apply_rys(i, j, loops=1)
                score        = self.run_test(tests)
                matrix[i, j] = score - baseline
                if verbose:
                    print(f"RYS ({i},{j}): score={score:.6f}  delta={matrix[i,j]:+.6f}")

        self.restore_layers()
        return baseline, matrix