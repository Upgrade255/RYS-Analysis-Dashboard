import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


class RYSEngine:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")

        print(f"Loading model {model_name} on {self.device} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
            device_map="auto" if "cuda" in str(self.device) else None
        )

        # disable KV cache if supported
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        layer_stack = None

        # common HF patterns
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layer_stack = self.model.model.layers

        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layer_stack = self.model.transformer.h

        elif hasattr(self.model, "model") and hasattr(self.model.model, "decoder") and hasattr(self.model.model.decoder,
                                                                                               "layers"):
            layer_stack = self.model.model.decoder.layers

        else:
            raise RuntimeError("Could not locate transformer layers in model architecture")

        self.layers = layer_stack
        self.N = len(self.layers)

        self.original_forwards = [layer.forward for layer in self.layers]
    # restore layers
    def restore_layers(self):
        for layer, orig in zip(self.layers, self.original_forwards):
            layer.forward = orig

    # disable layer
    def disable_layer(self, layer_idx: int):

        def skip_forward(hidden_states, *args, **kwargs):
            return (hidden_states,)

        self.layers[layer_idx].forward = skip_forward

    # apply RYS
    def apply_rys(self, start: int, end: int, loops: int):

        layers = self.layers
        origs = self.original_forwards
        s = int(start)
        e = int(end)
        L = int(loops)

        def make_new_forward(layer_idx):

            orig_fn = origs[layer_idx]

            def new_forward(hidden_states, *args, **kwargs):

                outputs = orig_fn(hidden_states, *args, **kwargs)

                # Determine structure
                if isinstance(outputs, (tuple, list)):
                    hidden_states_local = outputs[0]
                    tail = outputs[1:]
                    is_tuple = True
                else:
                    hidden_states_local = outputs
                    tail = None
                    is_tuple = False

                # only modify END layer
                if layer_idx == e and L > 0:

                    for _ in range(L):
                        for k in range(s, e + 1):

                            out_k = origs[k](hidden_states_local, *args, **kwargs)

                            if isinstance(out_k, (tuple, list)):
                                hidden_states_local = out_k[0]
                                last_tail = out_k[1:]
                                last_is_tuple = True
                            else:
                                hidden_states_local = out_k
                                last_tail = None
                                last_is_tuple = False

                    # return in correct structure
                    if last_is_tuple:
                        return (hidden_states_local,) + tuple(last_tail)
                    else:
                        return hidden_states_local

                return outputs

            return new_forward

        for idx in range(len(layers)):
            layers[idx].forward = make_new_forward(idx)

    # scoring
    def run_test(self, tests):

        prompts = [q + " " for q, _ in tests]
        answers = [a for _, a in tests]
        full_texts = [p + a for p, a in zip(prompts, answers)]

        prompt_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        full_inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True).to(self.device)

        prompt_lengths = [len(self.tokenizer(p).input_ids) for p in prompts]
        answer_ids = [self.tokenizer(a, add_special_tokens=False).input_ids for a in answers]

        with torch.no_grad():
            outputs = self.model(**full_inputs)
            logits = outputs.logits

        scores = []

        for i in range(len(tests)):

            start_idx = prompt_lengths[i]
            ans_ids = answer_ids[i]

            logprob = 0.0

            for j, token in enumerate(ans_ids):

                token_logits = logits[i, start_idx + j - 1]
                probs = torch.log_softmax(token_logits, dim=-1)

                logprob += probs[token].item()

            scores.append(logprob / len(ans_ids))

        return float(sum(scores) / len(scores))

    # sweep
    def sweep(self, tests, verbose=True):

        baseline = self.run_test(tests)

        matrix = np.zeros((self.N, self.N), dtype=float)

        for i in range(self.N):
            for j in range(i + 1, self.N):

                self.restore_layers()

                self.apply_rys(i, j, loops=1)

                score = self.run_test(tests)

                matrix[i, j] = score - baseline

                if verbose:
                    print(f"RYS {i}-{j}: score={score:.6f} delta={matrix[i,j]:.6f}")

        self.restore_layers()

        return baseline, matrix


# RYS sweep logic based on the technique described by David Noel Ng
# https://dnhkng.github.io/posts/rys/
# Licensed CC BY 4.0