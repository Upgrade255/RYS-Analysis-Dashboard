\# RYS Neuroanatomy Dashboard



> \*\*Repeat Yourself\*\* — A GUI for scanning, surgically modifying, and training transformer models based on their functional anatomy.



Based on the research by David Noel Ng:  

\[\*LLM Neuroanatomy: How I Topped the LLM Leaderboard Without Changing a Single Weight\*](https://dnhkng.github.io/posts/rys/) (2026)



---



\## What is this?



This dashboard lets you treat a transformer model the way a neuroscientist treats a brain — scan it, identify functional regions, cut it apart, stitch it back together differently, and observe what happens.



The core discovery behind RYS: large language models develop a \*\*functional anatomy\*\* during training. Early layers translate input tokens into an abstract representation. Late layers translate back to tokens. The middle layers perform reasoning — organised into discrete \*\*circuits\*\*, multi-layer pipelines that perform complete cognitive operations. Duplicating an entire reasoning circuit (without changing any weights) can measurably improve performance.



This tool makes that workflow accessible through a Gradio UI, and extends it with physical layer surgery, targeted fine-tuning, and experimental training paradigms.



---



\## Features



| Tab | What you can do |

|---|---|

| \*\*RYS Analysis\*\* | Load any HF model, run a full sweep of all `(start, end)` layer pairs, visualise the functional heatmap |

| \*\*Layer Surgery\*\* | Duplicate, delete, reorder, insert blank layers; zero attention heads or MLPs; export as a clean HF checkpoint |

| \*\*Targeted Training\*\* | Fine-tune only junction layers, a LoRA range, blank layers, or all parameters |

| \*\*Advanced Training\*\* | Ground-up training, layer-aware gradient routing, stretch→distill pipeline, RYS adaptive learning rates |



---



\## Quick Start



```bash

git clone https://github.com/YOUR\_USERNAME/rys-dashboard.git

cd rys-dashboard

pip install -r requirements.txt

python app.py

```



Then open `http://localhost:7860` in your browser.



---



\## Installation



\### Core (required)



```bash

pip install -r requirements.txt

```



\### Optional quantization backends



```bash

\# 4-bit / 8-bit (bitsandbytes)

pip install bitsandbytes



\# AWQ

pip install autoawq



\# GPTQ

pip install auto-gptq



\# GGUF (llama.cpp) — CPU

pip install llama-cpp-python



\# GGUF — CUDA

CMAKE\_ARGS="-DGGML\_CUDA=on" pip install llama-cpp-python



\# ExLlamaV2 — see https://github.com/turboderp/exllamav2

```



---



\## File Structure



```

rys-dashboard/

├── app.py                  # Gradio UI — four tabs

├── rys\_engine.py           # RYS sweep engine

├── layer\_surgeon.py        # Physical layer manipulation + HF export

├── model\_trainer.py        # Targeted fine-tuning

├── adaptive\_trainer.py     # Advanced training systems

├── question\_sets/          # Saved Q\&A probe sets

│   └── example\_probes.json

├── requirements.txt

├── DOCUMENTATION.md        # Full documentation

└── README.md

```



---



\## The RYS Sweep



The sweep tests every valid `(start, end)` layer pair. For each pair, it duplicates the layer range at inference time (no weight changes) and measures the score change on your probe dataset:



```

path: \[0 → 1 → ... → start → ... → end → start → ... → end → end+1 → ... → N-1]

```



The resulting heatmap is a \*\*functional MRI of the transformer\*\*:



\- 🔴 \*\*Red\*\* — duplicating this range improves performance (reasoning circuit)  

\- 🔵 \*\*Blue\*\* — duplicating this range hurts performance (translation/decoding region)  

\- \*\*Diagonal bands\*\* in the middle = reasoning circuits  

\- \*\*Blue wall\*\* on the right = decoding layers (don't duplicate)



---



\## Layer Surgery



All operations are \*\*staged\*\* — nothing is written to disk until you click Export. The export writes files directly (no live model mutation) to avoid HuggingFace metadata corruption issues:



1\. Remaps state dict keys per the operation plan

2\. Patches `config.json` via `to\_dict()` — including per-layer list fields like `layer\_types`

3\. Writes `model.safetensors` (auto-sharded for large models)

4\. Saves tokenizer and a `surgery\_manifest.json` for reproducibility



Exported models are standard HF checkpoints loadable with `AutoModelForCausalLM.from\_pretrained()`.



---



\## Advanced Training



Three experimental paradigms:



\*\*Layer-Aware Training\*\* — classifies each training sample by cognitive type (math, code, reasoning, etc.) and routes gradients only to layers relevant to that type. A `SpecialisationProbe` runs periodically to detect emerging functional specialisation and update routing to reinforce it.



\*\*Stretch → Distill\*\* — inserts blank layers into a small model, trains the stretched model to develop specialised functional regions, then distils back to the original architecture using output KD + activation matching. The activation matching teaches the student \*where\* to put reasoning, not just \*what\* the answer is.



\*\*RYS Adaptive LR\*\* — maps RYS sweep delta scores to per-layer learning rate multipliers. Reasoning circuit layers get higher LR; stable translation layers get lower LR.



---



\## Quantization Support



| Format | Surgery | Export | Notes |

|---|---|---|---|

| fp16 / fp32 | ✅ | ✅ | Full support |

| bitsandbytes 4-bit / 8-bit | ✅ | ✅ | |

| AWQ / GPTQ | ✅ | ✅ | |

| GGUF | ❌ | ❌ | Inference / RYS scan only |

| ExLlamaV2 | ❌ | ❌ | Inference / RYS scan only |



---



## Credits & Attribution

The RYS (Repeat Yourself) technique, the neuroanatomy framing, and the
original sweep engine design are the work of **David Noel Ng**:

> *LLM Neuroanatomy: How I Topped the LLM Leaderboard Without Changing
> a Single Weight* — https://dnhkng.github.io/posts/rys/
> Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

This dashboard is an independent implementation and extension built on
top of those concepts. The RYS engine, layer surgery, training systems,
and UI are original work released under the MIT License.



---



\## License



MIT

