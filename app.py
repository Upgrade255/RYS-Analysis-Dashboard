"""
app.py — RYS Analysis Dashboard + Layer Surgery + Targeted Training

Tabs
────
1. RYS Analysis   — original sweep / custom experiment (unchanged)
2. Layer Surgery  — physical layer manipulation + export to HF checkpoint
3. Training       — targeted fine-tuning (junction / LoRA / blank / full)

New files required alongside this one:
  layer_surgeon.py   — LayerSurgeon class
  model_trainer.py   — ModelTrainer class
  rys_engine.py      — unchanged (RYSEngine)
"""

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import threading
from transformers import AutoConfig

from rys_engine import RYSEngine
from layer_surgeon import LayerSurgeon
from model_trainer import ModelTrainer

# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────

engine        = None     # RYSEngine (for RYS tab)
surgeon       = None     # LayerSurgeon (for Surgery tab)
trainer       = None     # ModelTrainer (for Training tab)
disabled_layers = set()
layer_buttons   = []

QUANT_OPTIONS = ["none", "4bit", "8bit", "awq", "gptq", "gguf", "exl2"]

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_local_models():
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    models = []
    if os.path.exists(hf_cache):
        for folder in os.listdir(hf_cache):
            if folder.startswith("models--"):
                name = folder.replace("models--", "").replace("--", "/")
                models.append(name)
    # Also include any local directories with a config.json
    for d in os.listdir("."):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")):
            if d not in models:
                models.append(d)
    return sorted(models)


def get_question_sets():
    folder = "question_sets"
    if not os.path.exists(folder):
        os.makedirs(folder)
    return sorted([f for f in os.listdir(folder) if f.endswith(".json")])


def load_questions(filename):
    path = os.path.join("question_sets", filename)
    with open(path, "r") as f:
        return json.load(f)


def save_questions(data, filename):
    if not filename.endswith(".json"):
        filename += ".json"
    path = os.path.join("question_sets", filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return f"Saved {filename}", gr.update(choices=get_question_sets())


def parse_tests(table):
    tests = []
    for row in table:
        if row and len(row) >= 2:
            q = str(row[0]).strip()
            a = str(row[1]).strip()
            if q and a:
                tests.append((q, a))
    return tests


# ─────────────────────────────────────────────────────────────────────────────
# ██████╗ ██╗   ██╗███████╗    ████████╗ █████╗ ██████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝    ╚══██╔══╝██╔══██╗██╔══██╗
# ██████╔╝ ╚████╔╝ ███████╗       ██║   ███████║██████╔╝
# ██╔══██╗  ╚██╔╝  ╚════██║       ██║   ██╔══██║██╔══██╗
# ██║  ██║   ██║   ███████║       ██║   ██║  ██║██████╔╝
# ╚═╝  ╚═╝   ╚═╝   ╚══════╝       ╚═╝   ╚═╝  ╚═╝╚═════╝
# ─────────────────────────────────────────────────────────────────────────────

def toggle_layer(layer_id):
    global disabled_layers
    if layer_id in disabled_layers:
        disabled_layers.remove(layer_id)
    else:
        disabled_layers.add(layer_id)
    return [
        gr.update(variant="secondary" if i in disabled_layers else "primary")
        for i in range(len(layer_buttons))
    ]


def load_rys_model(dropdown_model, typed_model):
    global engine, disabled_layers
    disabled_layers = set()
    model_name = typed_model if typed_model else dropdown_model
    if not model_name:
        return "Select a model", None, None, *[gr.update() for _ in layer_buttons]
    engine = RYSEngine(model_name)
    config = AutoConfig.from_pretrained(model_name)

    def _cfg(attr, fallback="?"):
        return getattr(config, attr, None) or getattr(config, "text_config", object()).__class__.__dict__.get(attr,
                                                                                                              fallback)

    num_layers = (getattr(config, "num_hidden_layers", None)
                  or getattr(config, "num_layers", None)
                  or getattr(config, "n_layer", None)
                  or engine.N)  # fall back to what the engine actually found

    hidden = getattr(config, "hidden_size", getattr(config, "d_model", "?"))
    n_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", "?"))

    info = (f"Loaded {model_name}\n"
            f"Layers: {num_layers}\n"
            f"Hidden Size: {hidden}\n"
            f"Attention Heads: {n_heads}")
    layer_max = engine.N - 1
    updates = []
    for i in range(len(layer_buttons)):
        if i < engine.N:
            updates.append(gr.update(value=str(i), visible=True, variant="primary"))
        else:
            updates.append(gr.update(visible=False))
    return info, gr.update(maximum=layer_max), gr.update(maximum=layer_max), *updates


def run_custom(start, end, loops, table):
    global engine
    if engine is None:
        return "Load a model first.", None
    tests = parse_tests(table)
    if not tests:
        return "Provide valid Q/A pairs", None
    if start > end:
        return "Start must be <= End", None
    engine.restore_layers()
    engine.apply_rys(int(start), int(end), int(loops))
    for d in disabled_layers:
        engine.disable_layer(d)
    score = engine.run_test(tests)
    exec_plan = list(range(0, end + 1))
    for _ in range(int(loops)):
        exec_plan += list(range(start, end + 1))
    exec_plan += list(range(end + 1, engine.N))
    fig, ax = plt.subplots(figsize=(max(6, len(exec_plan) / 2), 2))
    colors = []
    for idx in exec_plan:
        if idx in disabled_layers:
            colors.append("gray")
        elif idx < start:
            colors.append("lightblue")
        elif start <= idx <= end:
            colors.append("orange")
        else:
            colors.append("lightgreen")
    ax.bar(range(len(exec_plan)), [1] * len(exec_plan), color=colors)
    ax.set_xticks(range(len(exec_plan)))
    ax.set_xticklabels(exec_plan, rotation=90)
    ax.set_yticks([])
    ax.set_title(f"Execution Plan — score {score:.6f}")
    plt.tight_layout()
    return f"Score: {score:.6f}", fig


def run_sweep(table):
    global engine
    if engine is None:
        return "Load a model first.", None
    tests = parse_tests(table)
    if not tests:
        return "Provide valid Q/A pairs", None
    engine.restore_layers()
    for d in disabled_layers:
        engine.disable_layer(d)
    baseline, matrix = engine.sweep(tests)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, origin="lower", cmap="RdBu")
    ax.set_title(f"RYS Sweep (baseline={baseline:.6f})")
    ax.set_xlabel("End Layer")
    ax.set_ylabel("Start Layer")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return f"Sweep complete — baseline {baseline:.6f}", fig


# ─────────────────────────────────────────────────────────────────────────────
# ███████╗██╗   ██╗██████╗  ██████╗ ███████╗██████╗ ██╗   ██╗
# ██╔════╝██║   ██║██╔══██╗██╔════╝ ██╔════╝██╔══██╗╚██╗ ██╔╝
# ███████╗██║   ██║██████╔╝██║  ███╗█████╗  ██████╔╝ ╚████╔╝
# ╚════██║██║   ██║██╔══██╗██║   ██║██╔══╝  ██╔══██╗  ╚██╔╝
# ███████║╚██████╔╝██║  ██║╚██████╔╝███████╗██║  ██║   ██║
# ╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝
# ─────────────────────────────────────────────────────────────────────────────

def load_surgeon_model(model_path, quant_mode):
    global surgeon
    if not model_path:
        return "Enter a model path or HF ID", "", []
    quant = None if quant_mode == "none" else quant_mode
    try:
        surgeon = LayerSurgeon(model_path, quant_mode=quant)
        info = (f"Loaded: {model_path}\n"
                f"Quant: {quant_mode}\n"
                f"Layers: {len(surgeon._layer_plan)}\n"
                f"Export capable: {surgeon._export_capable}")
        return info, "", _plan_to_table()
    except Exception as e:
        return f"Error: {e}", "", []


def _plan_to_table():
    if surgeon is None:
        return []
    return [[r["slot"], r["label"], r["type"]] for r in surgeon.get_plan()]


def surgeon_duplicate(slot):
    if surgeon is None:
        return "Load a model first", []
    try:
        new_slot = surgeon.duplicate_layer(int(slot))
        return f"Duplicated slot {slot} → new copy at {new_slot}", _plan_to_table()
    except Exception as e:
        return f"Error: {e}", _plan_to_table()


def surgeon_delete(slot):
    if surgeon is None:
        return "Load a model first", []
    try:
        surgeon.delete_layer(int(slot))
        return f"Deleted slot {slot}", _plan_to_table()
    except Exception as e:
        return f"Error: {e}", _plan_to_table()


def surgeon_move(from_slot, to_slot):
    if surgeon is None:
        return "Load a model first", []
    try:
        surgeon.move_layer(int(from_slot), int(to_slot))
        return f"Moved slot {from_slot} → {to_slot}", _plan_to_table()
    except Exception as e:
        return f"Error: {e}", _plan_to_table()


def surgeon_insert_blank(after_slot):
    if surgeon is None:
        return "Load a model first", []
    try:
        new_slot = surgeon.insert_blank_layer(int(after_slot))
        return f"Inserted blank layer at slot {new_slot}", _plan_to_table()
    except Exception as e:
        return f"Error: {e}", _plan_to_table()


def surgeon_mask_heads(slot, heads_str):
    if surgeon is None:
        return "Load a model first", []
    try:
        heads = [int(h.strip()) for h in heads_str.split(",") if h.strip()]
        surgeon.mask_attention_heads(int(slot), heads)
        return f"Marked heads {heads} in slot {slot} for masking", _plan_to_table()
    except Exception as e:
        return f"Error: {e}", _plan_to_table()


def surgeon_mask_mlp(slot):
    if surgeon is None:
        return "Load a model first", []
    try:
        surgeon.mask_mlp(int(slot))
        return f"Marked MLP in slot {slot} for zeroing", _plan_to_table()
    except Exception as e:
        return f"Error: {e}", _plan_to_table()


def surgeon_reset():
    if surgeon is None:
        return "Load a model first", []
    surgeon.reset_plan()
    return "Plan reset to original", _plan_to_table()


_export_running = False

def surgeon_export(output_dir):
    global _export_running
    if surgeon is None:
        return "Load a model first"
    if not output_dir:
        return "Enter an output directory path"
    if not surgeon._export_capable:
        return "This model format (GGUF/ExLlamaV2) cannot be exported as HF checkpoint"
    if _export_running:
        return "Export already in progress — please wait"
    _export_running = True
    try:
        path = surgeon.export(output_dir)
        return f"✓ Exported to: {path}\n\nYou can now load this in the RYS tab or Training tab."
    except Exception as e:
        import traceback
        return f"Export error: {e}\n\n{traceback.format_exc()}"
    finally:
        _export_running = False


def surgeon_load_to_rys(output_dir):
    """Load an exported model directly into the RYS engine."""
    global engine, disabled_layers
    if not output_dir or not os.path.exists(output_dir):
        return "Directory not found. Export first.", None, None, *[gr.update() for _ in layer_buttons]
    disabled_layers = set()
    try:
        engine = RYSEngine(output_dir)
        layer_max = engine.N - 1
        updates = []
        for i in range(len(layer_buttons)):
            if i < engine.N:
                updates.append(gr.update(value=str(i), visible=True, variant="primary"))
            else:
                updates.append(gr.update(visible=False))
        info = f"Loaded exported model from {output_dir}\nLayers: {engine.N}"
        return info, gr.update(maximum=layer_max), gr.update(maximum=layer_max), *updates
    except Exception as e:
        return f"Error: {e}", None, None, *[gr.update() for _ in layer_buttons]


# ─────────────────────────────────────────────────────────────────────────────
# ████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗
# ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝
#    ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
#    ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
#    ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
#    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝
# ─────────────────────────────────────────────────────────────────────────────

_train_log_buffer = []
_train_loss_history = []


def _trainer_log(msg: str):
    _train_log_buffer.append(msg)


def run_training(
    model_path, mode,
    junction_start, junction_end,
    lora_start, lora_end, lora_rank, lora_alpha,
    dataset_source_type, dataset_text, dataset_file, dataset_hf,
    output_dir, epochs, batch_size, grad_accum, lr,
    max_length, merge_lora,
):
    global trainer, _train_log_buffer, _train_loss_history
    _train_log_buffer = []
    _train_loss_history = []

    if not model_path:
        return "Enter a model path", "", None

    # Resolve dataset
    if dataset_source_type == "Text pairs (JSON in textbox)":
        try:
            pairs_raw = json.loads(dataset_text)
            dataset = [(d["prompt"], d["response"]) for d in pairs_raw]
        except Exception as e:
            return f"Invalid JSON in text box: {e}", "", None
    elif dataset_source_type == "JSON file path":
        dataset = dataset_file
    elif dataset_source_type == "HuggingFace dataset name":
        dataset = dataset_hf
    else:
        return "Select a dataset source type", "", None

    try:
        trainer = ModelTrainer(
            model_path=model_path,
            mode=mode,
            junction_start=int(junction_start),
            junction_end=int(junction_end),
            lora_start=int(lora_start),
            lora_end=int(lora_end),
            lora_rank=int(lora_rank),
            lora_alpha=float(lora_alpha),
            log_callback=_trainer_log,
        )
        log_data = trainer.train(
            dataset_source=dataset,
            output_dir=output_dir,
            epochs=int(epochs),
            batch_size=int(batch_size),
            grad_accum=int(grad_accum),
            learning_rate=float(lr),
            max_length=int(max_length),
            merge_lora_on_save=merge_lora,
        )
        _train_loss_history = log_data["loss_history"]

        # Loss curve
        fig, ax = plt.subplots(figsize=(7, 3))
        xs = [e["epoch"] for e in _train_loss_history]
        ys = [e["loss"] for e in _train_loss_history]
        ax.plot(xs, ys, marker="o", color="steelblue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss — {mode} mode")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        status = (f"✓ Training complete\n"
                  f"Best checkpoint: {log_data['best_checkpoint']}\n"
                  f"Final loss: {ys[-1]:.6f}")
        return status, "\n".join(_train_log_buffer), fig

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"Training error: {e}", tb, None


def refresh_train_log():
    return "\n".join(_train_log_buffer)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

MAX_LAYERS = 128
ROW_WIDTH  = 32

with gr.Blocks(title="RYS + Surgery + Training") as demo:

    gr.Markdown("# 🧠 RYS Layer Analysis Dashboard")

    # ── TAB 1: RYS (original, unchanged) ─────────────────────────────────────
    with gr.Tab("RYS Analysis"):
        gr.Markdown("### Model Selection")
        with gr.Row():
            model_dropdown = gr.Dropdown(choices=get_local_models(), label="Local Models")
            model_text     = gr.Textbox(label="HuggingFace Model ID", placeholder="microsoft/phi-2")
        load_btn    = gr.Button("Load Model")
        load_status = gr.Textbox(label="Status")
        start_slider = gr.Slider(0, 32, step=1, label="Start Layer")
        end_slider   = gr.Slider(0, 32, step=1, label="End Layer")

        gr.Markdown("### Layer Controls (click to disable)")
        for r in range(MAX_LAYERS // ROW_WIDTH):
            with gr.Row():
                for c in range(ROW_WIDTH):
                    idx = r * ROW_WIDTH + c
                    btn = gr.Button(str(idx), visible=False, size="sm", min_width=35)
                    layer_buttons.append(btn)
        for i, btn in enumerate(layer_buttons):
            btn.click(lambda i=i: toggle_layer(i), None, layer_buttons)

        gr.Markdown("### Question Sets")
        with gr.Row():
            question_dropdown = gr.Dropdown(choices=get_question_sets(), label="Load Question Set")
            load_q_btn = gr.Button("Load")
            save_name  = gr.Textbox(label="Save As")
            save_btn   = gr.Button("Save")
        qa_table = gr.Dataframe(
            headers=["Question", "Answer"],
            value=[["What is 2+2", "4"]],
            interactive=True, row_count=5
        )
        load_q_btn.click(load_questions, question_dropdown, qa_table)
        save_btn.click(save_questions, [qa_table, save_name], [load_status, question_dropdown])

        gr.Markdown("### Custom Experiment")
        loops_slider  = gr.Slider(0, 10, step=1, label="Repeat Loops")
        run_btn       = gr.Button("Run Custom")
        custom_status = gr.Textbox()
        custom_plot   = gr.Plot()
        run_btn.click(run_custom,
                      [start_slider, end_slider, loops_slider, qa_table],
                      [custom_status, custom_plot])

        gr.Markdown("### RYS Sweep")
        sweep_btn    = gr.Button("Run Sweep")
        sweep_status = gr.Textbox()
        sweep_plot   = gr.Plot()
        sweep_btn.click(run_sweep, qa_table, [sweep_status, sweep_plot])

        load_btn.click(
            load_rys_model,
            [model_dropdown, model_text],
            [load_status, start_slider, end_slider] + layer_buttons
        )

    # ── TAB 2: Layer Surgery ──────────────────────────────────────────────────
    with gr.Tab("🔬 Layer Surgery"):
        gr.Markdown(
            "**Physically manipulate layers** — duplicate, delete, reorder, "
            "insert blank layers, mask heads/MLP. Export as a real HF checkpoint "
            "ready for inference or training.\n\n"
            "_GGUF / ExLlamaV2 models load for inference testing only (no export)._"
        )

        with gr.Row():
            surg_model_input = gr.Textbox(
                label="Model Path / HF ID",
                placeholder="Qwen/Qwen2-7B  or  ./my_exported_model"
            )
            surg_quant = gr.Dropdown(choices=QUANT_OPTIONS, value="none", label="Quantization")
        surg_load_btn    = gr.Button("Load Model for Surgery")
        surg_load_status = gr.Textbox(label="Status", lines=4)

        gr.Markdown("### Layer Plan")
        gr.Markdown("_Shows current layer order. Slot = execution position, Label = source._")
        plan_table = gr.Dataframe(
            headers=["Slot", "Label", "Type"],
            interactive=False,
            label="Current Plan"
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Duplicate Layer")
                dup_slot = gr.Number(label="Source Slot", value=0, precision=0)
                dup_btn  = gr.Button("Duplicate")

            with gr.Column():
                gr.Markdown("#### Delete Layer")
                del_slot = gr.Number(label="Slot to Delete", value=0, precision=0)
                del_btn  = gr.Button("Delete")

            with gr.Column():
                gr.Markdown("#### Move Layer")
                mv_from = gr.Number(label="From Slot", value=0, precision=0)
                mv_to   = gr.Number(label="To Slot",   value=1, precision=0)
                mv_btn  = gr.Button("Move")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Insert Blank Layer")
                blank_after = gr.Number(label="Insert After Slot", value=0, precision=0)
                blank_btn   = gr.Button("Insert Blank")

            with gr.Column():
                gr.Markdown("#### Mask Attention Heads")
                mask_slot   = gr.Number(label="Slot", value=0, precision=0)
                mask_heads  = gr.Textbox(label="Head indices (comma-sep)", placeholder="0,1,2")
                mask_h_btn  = gr.Button("Mark Heads")

            with gr.Column():
                gr.Markdown("#### Zero MLP")
                mlp_slot = gr.Number(label="Slot", value=0, precision=0)
                mlp_btn  = gr.Button("Zero MLP")

        surg_op_status = gr.Textbox(label="Operation Status")

        reset_btn = gr.Button("↩ Reset Plan to Original", variant="secondary")

        gr.Markdown("### Export")
        with gr.Row():
            export_dir = gr.Textbox(label="Output Directory", placeholder="./my_frankenstein_model")
            export_btn = gr.Button("Export Model", variant="primary")
        export_status = gr.Textbox(label="Export Status", lines=3)

        gr.Markdown("### Quick Load into RYS Tab")
        gr.Markdown("_After exporting, click below to immediately test in the RYS Analysis tab._")
        load_to_rys_btn    = gr.Button("→ Load Exported Model into RYS Engine")
        load_to_rys_status = gr.Textbox(label="RYS Load Status")

        # Wiring
        surg_load_btn.click(
            load_surgeon_model,
            [surg_model_input, surg_quant],
            [surg_load_status, surg_op_status, plan_table]
        )
        dup_btn.click(surgeon_duplicate,   [dup_slot],            [surg_op_status, plan_table])
        del_btn.click(surgeon_delete,      [del_slot],            [surg_op_status, plan_table])
        mv_btn.click(surgeon_move,         [mv_from, mv_to],      [surg_op_status, plan_table])
        blank_btn.click(surgeon_insert_blank, [blank_after],      [surg_op_status, plan_table])
        mask_h_btn.click(surgeon_mask_heads, [mask_slot, mask_heads], [surg_op_status, plan_table])
        mlp_btn.click(surgeon_mask_mlp,    [mlp_slot],            [surg_op_status, plan_table])
        reset_btn.click(surgeon_reset,     [],                    [surg_op_status, plan_table])
        export_btn.click(surgeon_export,   [export_dir],          [export_status])
        load_to_rys_btn.click(
            surgeon_load_to_rys,
            [export_dir],
            [load_to_rys_status, start_slider, end_slider] + layer_buttons
        )

    # ── TAB 3: Training ───────────────────────────────────────────────────────
    with gr.Tab("⚡ Targeted Training"):
        gr.Markdown(
            "Fine-tune surgically modified (or plain) models.\n\n"
            "**Mode priority (per project design):**\n"
            "1. `junction` — freeze all, train only the boundary layers at the RYS seam _(dnhkng's hypothesis)_\n"
            "2. `lora` — inject LoRA adapters into a selected layer range\n"
            "3. `blank` — train only newly inserted blank layers (neuro organ growth)\n"
            "4. `full` — train all parameters"
        )

        with gr.Row():
            train_model_path = gr.Textbox(
                label="Model Path (HF checkpoint or exported surgery model)",
                placeholder="./my_frankenstein_model"
            )
            train_mode = gr.Dropdown(
                choices=["junction", "lora", "blank", "full"],
                value="junction",
                label="Training Mode"
            )

        gr.Markdown("#### Junction / Layer Range Settings")
        with gr.Row():
            junc_start = gr.Number(label="Junction / RYS Start",  value=45, precision=0)
            junc_end   = gr.Number(label="Junction / RYS End",    value=52, precision=0)

        gr.Markdown("#### LoRA Settings _(mode=lora only)_")
        with gr.Row():
            lora_start_n = gr.Number(label="LoRA Start Layer", value=40, precision=0)
            lora_end_n   = gr.Number(label="LoRA End Layer",   value=55, precision=0)
            lora_rank_n  = gr.Number(label="Rank",             value=8,  precision=0)
            lora_alpha_n = gr.Number(label="Alpha",            value=16, precision=0)

        gr.Markdown("#### Dataset")
        ds_type = gr.Radio(
            choices=["Text pairs (JSON in textbox)", "JSON file path", "HuggingFace dataset name"],
            value="Text pairs (JSON in textbox)",
            label="Dataset Source"
        )
        ds_text = gr.Textbox(
            label='Pairs JSON (e.g. [{"prompt":"...","response":"..."}])',
            lines=4,
            value='[{"prompt": "What is 2+2?", "response": "4"}]'
        )
        ds_file = gr.Textbox(label="JSON File Path", placeholder="./data/train.json")
        ds_hf   = gr.Textbox(label="HuggingFace Dataset Name", placeholder="tatsu-lab/alpaca")

        gr.Markdown("#### Training Hyperparameters")
        with gr.Row():
            train_output_dir  = gr.Textbox(label="Output Directory", value="./training_output")
            train_epochs      = gr.Slider(1, 20, step=1, value=3,   label="Epochs")
            train_batch       = gr.Slider(1, 8,  step=1, value=1,   label="Batch Size")
        with gr.Row():
            train_grad_accum  = gr.Slider(1, 32, step=1, value=4,   label="Gradient Accumulation")
            train_lr          = gr.Number(value=2e-4,               label="Learning Rate")
            train_max_len     = gr.Slider(64, 2048, step=64, value=512, label="Max Sequence Length")
        merge_lora_cb = gr.Checkbox(label="Merge LoRA weights into base on save", value=False)

        train_btn = gr.Button("▶ Start Training", variant="primary")

        train_status   = gr.Textbox(label="Status", lines=3)
        train_log_box  = gr.Textbox(label="Training Log", lines=10)
        train_plot     = gr.Plot(label="Loss Curve")
        refresh_btn    = gr.Button("↻ Refresh Log")

        train_btn.click(
            run_training,
            inputs=[
                train_model_path, train_mode,
                junc_start, junc_end,
                lora_start_n, lora_end_n, lora_rank_n, lora_alpha_n,
                ds_type, ds_text, ds_file, ds_hf,
                train_output_dir, train_epochs, train_batch, train_grad_accum, train_lr,
                train_max_len, merge_lora_cb,
            ],
            outputs=[train_status, train_log_box, train_plot]
        )
        refresh_btn.click(refresh_train_log, [], [train_log_box])

    # ── TAB 4: Advanced Training ──────────────────────────────────────────────
    with gr.Tab("🧬 Advanced Training"):
        gr.Markdown(
            "**Three experimental training paradigms** built on the RYS neuroanatomy framework.\n\n"
            "| Mode | What it does |\n"
            "|---|---|\n"
            "| **Ground-Up** | Reuse an existing model's *architecture* but re-init all weights randomly. Train from scratch. |\n"
            "| **Layer-Aware** | Tag each sample by cognitive type (math/code/reasoning/etc.), route gradients to relevant layers only. Probes detect emerging specialisation and reinforce it. |\n"
            "| **Stretch → Distill** | Stretch a small model by inserting blank layers, compartmentalise via layer-aware training, then distil back to original size using activation-matched KD. |"
        )

        adv_mode = gr.Radio(
            choices=["ground_up", "layer_aware", "stretch_distill"],
            value="ground_up",
            label="Advanced Training Mode"
        )

        # ── Shared dataset ────────────────────────────────────────────────────
        gr.Markdown("#### Dataset (shared across all modes)")
        adv_ds_type = gr.Radio(
            choices=["Text pairs (JSON in textbox)", "JSON file path", "HuggingFace dataset name"],
            value="Text pairs (JSON in textbox)",
            label="Dataset Source"
        )
        adv_ds_text = gr.Textbox(
            label='Pairs JSON',
            lines=3,
            value='[{"prompt": "What is 2+2?", "response": "4"}, '
                  '{"prompt": "Write a loop in Python", "response": "for i in range(10): print(i)"}]'
        )
        adv_ds_file = gr.Textbox(label="JSON File Path", placeholder="./data/train.json")
        adv_ds_hf   = gr.Textbox(label="HuggingFace Dataset Name", placeholder="tatsu-lab/alpaca")

        # ── Ground-up settings ─────────────────────────────────────────────────
        with gr.Accordion("Ground-Up Settings", open=True):
            gr.Markdown(
                "Loads the **architecture** (config) of the specified model and re-initialises "
                "all weights randomly. Train an entirely fresh model from that blueprint.\n\n"
                "**Probe bank** (optional): provide typed eval pairs and the trainer will run "
                "a `SpecialisationProbe` every N steps, converting per-layer importance scores "
                "into per-layer learning rate multipliers via `RYSAdaptiveLR`. "
                "Load-bearing layers get boosted LR; flat/unhelpful layers get reduced LR."
            )
            gu_model    = gr.Textbox(label="Architecture Source (model path or HF ID)",
                                     placeholder="Qwen/Qwen2-0.5B")
            gu_out      = gr.Textbox(label="Output Directory", value="./groundup_output")
            with gr.Row():
                gu_epochs   = gr.Slider(1, 50, step=1, value=5,    label="Epochs")
                gu_batch    = gr.Slider(1, 8,  step=1, value=2,    label="Batch Size")
                gu_accum    = gr.Slider(1, 32, step=1, value=8,    label="Grad Accumulation")
            with gr.Row():
                gu_lr       = gr.Number(value=1e-3,                label="Learning Rate")
                gu_maxlen   = gr.Slider(64, 1024, step=64, value=256, label="Max Length")
                gu_save_n   = gr.Number(value=500, precision=0,   label="Save every N steps")
            gr.Markdown("#### Probe-Driven Adaptive LR _(optional)_")
            gu_probe_bank = gr.Textbox(
                label="Probe bank JSON (same format as Layer-Aware)",
                lines=4,
                placeholder='{"math": [["What is 3*3?", "9"]], "code": [["Write hello world", "print(\'hello\')"]]}',
            )
            with gr.Row():
                gu_probe_every = gr.Slider(0, 1000, step=50, value=200,
                                           label="Probe every N steps (0=disabled)")
                gu_alr_base   = gr.Number(value=0.1,  label="ALR base multiplier")
                gu_alr_scale  = gr.Number(value=1.8,  label="ALR scale")
                gu_alr_temp   = gr.Number(value=0.05, label="ALR temperature")
                gu_alr_cap    = gr.Number(value=3.0,  label="ALR max multiplier")

        # ── Layer-Aware settings ───────────────────────────────────────────────
        with gr.Accordion("Layer-Aware Settings", open=False):
            gr.Markdown(
                "Samples are tagged by cognitive type (math, code, reasoning, language, factual). "
                "Gradients are routed only to layers assigned to the current sample type. "
                "A **SpecialisationProbe** runs periodically and updates routing as layers specialise.\n\n"
                "**Probe bank**: small eval pairs used to detect specialisation. "
                "Format: `{\"math\": [[\"prompt\",\"answer\"], ...], \"code\": [...], ...}`"
            )
            la_model      = gr.Textbox(label="Model Path / HF ID",
                                       placeholder="./my_model or Qwen/Qwen2-0.5B")
            la_out        = gr.Textbox(label="Output Directory", value="./layer_aware_output")
            with gr.Row():
                la_epochs   = gr.Slider(1, 20, step=1, value=3,   label="Epochs")
                la_batch    = gr.Slider(1, 4,  step=1, value=1,   label="Batch Size")
                la_accum    = gr.Slider(1, 32, step=1, value=4,   label="Grad Accumulation")
                la_lr       = gr.Number(value=2e-4,               label="Learning Rate")
            la_probe_every = gr.Slider(0, 1000, step=50, value=200,
                                       label="Run probe every N steps (0=disabled)")
            la_spec_thresh = gr.Number(value=0.02,
                                       label="Specialisation threshold (min delta to reroute)")
            la_probe_bank  = gr.Textbox(
                label='Probe bank JSON (optional)',
                lines=4,
                placeholder='{"math": [["What is 3*3?", "9"]], "code": [["Write hello world", "print(\'hello\')"]]}',
            )

        # ── Stretch-Distill settings ───────────────────────────────────────────
        with gr.Accordion("Stretch → Distill Settings", open=False):
            gr.Markdown(
                "**Phase 1**: Stretch a small model by inserting blank layers.\n\n"
                "**Phase 2**: Train the stretched model with layer-aware routing to "
                "force functional specialisation into the new layers.\n\n"
                "**Phase 3**: Distil back to the original small architecture using "
                "output KD (KL divergence) **plus** activation matching at "
                "the identified functional regions — so the student learns *where* "
                "to put reasoning, not just *what* the answer is."
            )
            sd_model        = gr.Textbox(label="Small Model Path / HF ID",
                                         placeholder="Qwen/Qwen2-0.5B")
            sd_out          = gr.Textbox(label="Output Base Directory",
                                         value="./stretch_distill_output")
            with gr.Row():
                sd_n_blanks = gr.Slider(1, 16, step=1, value=4,
                                        label="Blank layers to insert (Phase 1)")
                sd_after    = gr.Number(value=-1, precision=0,
                                        label="Insert after layer (-1 = middle)")
            with gr.Row():
                sd_p2_epochs = gr.Slider(1, 20, step=1, value=3, label="Phase 2 Epochs")
                sd_p3_epochs = gr.Slider(1, 20, step=1, value=5, label="Phase 3 Epochs")
                sd_kd_alpha  = gr.Number(value=0.5, label="KD loss weight (α)")
                sd_act_alpha = gr.Number(value=0.3, label="Activation loss weight (β)")
            with gr.Row():
                sd_lr        = gr.Number(value=2e-4, label="Learning Rate")
                sd_temp      = gr.Number(value=2.0,  label="KD Temperature")
                sd_maxlen    = gr.Slider(64, 512, step=64, value=256, label="Max Length")
            gr.Markdown("_Run phases independently or all at once:_")
            with gr.Row():
                sd_phase1_btn = gr.Button("▶ Phase 1: Stretch only")
                sd_phase2_btn = gr.Button("▶ Phase 2: Compartmentalise only")
                sd_phase3_btn = gr.Button("▶ Phase 3: Distil only")
                sd_all_btn    = gr.Button("▶▶ Run All Phases", variant="primary")

        # ── RYS Adaptive LR ───────────────────────────────────────────────────
        with gr.Accordion("RYS Adaptive LR — Compute from Sweep", open=False):
            gr.Markdown(
                "Load a saved RYS sweep matrix (numpy `.npy` file from a previous sweep) "
                "and compute per-layer LR multipliers from it.\n\n"
                "**High-positive delta layers** (reasoning circuits) → higher LR.\n"
                "**Flat/negative delta layers** (translation layers) → lower LR.\n\n"
                "The multiplier chart can guide manual LR choices, or the values are "
                "applied automatically when used inside LayerAwareTrainer."
            )
            alr_matrix_path = gr.Textbox(label="Sweep matrix path (.npy)",
                                          placeholder="./sweep_matrix.npy")
            with gr.Row():
                alr_base   = gr.Number(value=0.1, label="Base multiplier (flat layers)")
                alr_scale  = gr.Number(value=1.8, label="Scale (range width)")
                alr_temp   = gr.Number(value=0.05, label="Temperature (sensitivity)")
                alr_cap    = gr.Number(value=3.0, label="Max multiplier cap")
            alr_btn     = gr.Button("Compute LR Multipliers")
            alr_plot    = gr.Plot(label="LR Multiplier Chart")
            alr_status  = gr.Textbox(label="Status")

        # ── Shared outputs ────────────────────────────────────────────────────
        adv_run_btn    = gr.Button("▶ Run Selected Mode", variant="primary")
        adv_status     = gr.Textbox(label="Status", lines=3)
        adv_log        = gr.Textbox(label="Log", lines=12)
        adv_plot       = gr.Plot(label="Loss / Routing Chart")
        adv_refresh    = gr.Button("↻ Refresh Log")

        # ── Backend functions ─────────────────────────────────────────────────

        _adv_log_buf: list = []

        def _adv_log(msg):
            _adv_log_buf.append(msg)

        def _resolve_adv_dataset(ds_type, ds_text, ds_file, ds_hf):
            if ds_type == "Text pairs (JSON in textbox)":
                pairs = json.loads(ds_text)
                return [(d["prompt"], d["response"]) for d in pairs]
            elif ds_type == "JSON file path":
                return ds_file
            return ds_hf

        def run_advanced(mode, ds_type, ds_text, ds_file, ds_hf,
                         gu_model, gu_out, gu_epochs, gu_batch, gu_accum,
                         gu_lr, gu_maxlen, gu_save_n,
                         gu_probe_bank, gu_probe_every,
                         gu_alr_base, gu_alr_scale, gu_alr_temp, gu_alr_cap,
                         la_model, la_out, la_epochs, la_batch, la_accum,
                         la_lr, la_probe_every, la_spec_thresh, la_probe_bank,
                         sd_model, sd_out, sd_n_blanks, sd_after,
                         sd_p2_epochs, sd_p3_epochs, sd_kd_alpha, sd_act_alpha,
                         sd_lr, sd_temp, sd_maxlen):
            import matplotlib.pyplot as plt
            _adv_log_buf.clear()
            dataset = None
            try:
                dataset = _resolve_adv_dataset(ds_type, ds_text, ds_file, ds_hf)
            except Exception as e:
                return f"Dataset error: {e}", "", None

            try:
                if mode == "ground_up":
                    from adaptive_trainer import GroundUpTrainer
                    probe_bank = None
                    if gu_probe_bank and gu_probe_bank.strip():
                        raw = json.loads(gu_probe_bank)
                        probe_bank = {k: [tuple(p) for p in v] for k, v in raw.items()}
                    t = GroundUpTrainer(gu_model, log=_adv_log)
                    log = t.train(
                        dataset, gu_out,
                        epochs=int(gu_epochs),
                        batch_size=int(gu_batch),
                        grad_accum=int(gu_accum),
                        learning_rate=float(gu_lr),
                        max_length=int(gu_maxlen),
                        save_every_n_steps=int(gu_save_n),
                        probe_bank=probe_bank,
                        probe_every_n_steps=int(gu_probe_every),
                        alr_base=float(gu_alr_base),
                        alr_scale=float(gu_alr_scale),
                        alr_temperature=float(gu_alr_temp),
                        alr_max_mult=float(gu_alr_cap),
                    )

                elif mode == "layer_aware":
                    from adaptive_trainer import LayerAwareTrainer
                    probe_bank = None
                    if la_probe_bank.strip():
                        raw = json.loads(la_probe_bank)
                        probe_bank = {k: [tuple(p) for p in v]
                                      for k, v in raw.items()}
                    t = LayerAwareTrainer(
                        la_model,
                        probe_bank=probe_bank,
                        probe_every_n_steps=int(la_probe_every),
                        specialisation_threshold=float(la_spec_thresh),
                        log=_adv_log,
                    )
                    log = t.train(dataset, la_out, epochs=int(la_epochs),
                                  batch_size=int(la_batch),
                                  grad_accum=int(la_accum),
                                  learning_rate=float(la_lr))
                    fig = t.get_routing_plot()
                    return (f"✓ Layer-aware training complete\n"
                            f"Output: {la_out}"),\
                           "\n".join(_adv_log_buf), fig

                elif mode == "stretch_distill":
                    from adaptive_trainer import StretchDistillPipeline
                    pipe = StretchDistillPipeline(
                        sd_model,
                        blank_layers_to_insert=int(sd_n_blanks),
                        insert_after_layer=int(sd_after),
                        kd_alpha=float(sd_kd_alpha),
                        act_alpha=float(sd_act_alpha),
                        log=_adv_log,
                    )
                    pipe.phase1_stretch(os.path.join(sd_out, "stretched"))
                    pipe.phase2_compartmentalise(
                        dataset, os.path.join(sd_out, "compartmentalised"),
                        epochs=int(sd_p2_epochs), learning_rate=float(sd_lr))
                    log = pipe.phase3_distill(
                        dataset, os.path.join(sd_out, "distilled"),
                        epochs=int(sd_p3_epochs), learning_rate=float(sd_lr),
                        max_length=int(sd_maxlen), temperature=float(sd_temp))

                # Generic loss curve
                xs = [e["epoch"] for e in log["loss_history"]]
                ys = [e["loss"]  for e in log["loss_history"]]
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(xs, ys, marker="o", color="steelblue")
                ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
                ax.set_title(f"{mode} — training loss")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                return f"✓ Done", "\n".join(_adv_log_buf), fig

            except Exception as e:
                import traceback
                return f"Error: {e}", traceback.format_exc(), None

        def run_sd_phase(phase_num,
                         ds_type, ds_text, ds_file, ds_hf,
                         sd_model, sd_out, sd_n_blanks, sd_after,
                         sd_p2_epochs, sd_p3_epochs,
                         sd_kd_alpha, sd_act_alpha, sd_lr, sd_temp, sd_maxlen):
            import matplotlib.pyplot as plt
            _adv_log_buf.clear()
            try:
                dataset = _resolve_adv_dataset(ds_type, ds_text, ds_file, ds_hf)
                from adaptive_trainer import StretchDistillPipeline
                pipe = StretchDistillPipeline(
                    sd_model, blank_layers_to_insert=int(sd_n_blanks),
                    insert_after_layer=int(sd_after),
                    kd_alpha=float(sd_kd_alpha), act_alpha=float(sd_act_alpha),
                    log=_adv_log,
                )
                # Restore stretched path if phase 2/3 only
                stretched = os.path.join(sd_out, "stretched")
                compartmentalised = os.path.join(sd_out, "compartmentalised", "final")
                if phase_num >= 2 and os.path.exists(stretched):
                    pipe.stretched_path = stretched
                if phase_num >= 3 and os.path.exists(compartmentalised):
                    pipe.stretched_path = compartmentalised

                if phase_num == 1:
                    pipe.phase1_stretch(stretched)
                    return "✓ Phase 1 complete", "\n".join(_adv_log_buf), None
                elif phase_num == 2:
                    pipe.phase2_compartmentalise(
                        dataset, os.path.join(sd_out, "compartmentalised"),
                        epochs=int(sd_p2_epochs), learning_rate=float(sd_lr))
                    return "✓ Phase 2 complete", "\n".join(_adv_log_buf), None
                elif phase_num == 3:
                    log = pipe.phase3_distill(
                        dataset, os.path.join(sd_out, "distilled"),
                        epochs=int(sd_p3_epochs), learning_rate=float(sd_lr),
                        max_length=int(sd_maxlen), temperature=float(sd_temp))
                    xs = [e["epoch"] for e in log["loss_history"]]
                    ys = [e["loss"]  for e in log["loss_history"]]
                    fig, ax = plt.subplots(figsize=(7, 3))
                    ax.plot(xs, ys, marker="o", color="coral")
                    ax.set_title("Distillation loss"); ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    return "✓ Phase 3 complete", "\n".join(_adv_log_buf), fig
            except Exception as e:
                import traceback
                return f"Error: {e}", traceback.format_exc(), None

        def compute_alr(path, base, scale, temp, cap):
            try:
                import numpy as np
                from adaptive_trainer import RYSAdaptiveLR
                matrix = np.load(path)
                alr = RYSAdaptiveLR(base=float(base), scale=float(scale),
                                    temperature=float(temp), max_mult=float(cap))
                mults = alr.compute_from_matrix(matrix)
                fig = alr.plot()
                summary = (f"Layers: {len(mults)}\n"
                           f"Min multiplier: {min(mults):.3f}\n"
                           f"Max multiplier: {max(mults):.3f}\n"
                           f"Mean: {sum(mults)/len(mults):.3f}")
                return summary, fig
            except Exception as e:
                import traceback
                return f"Error: {e}\n{traceback.format_exc()}", None

        # Wiring
        _sd_phase_inputs = [
            adv_ds_type, adv_ds_text, adv_ds_file, adv_ds_hf,
            sd_model, sd_out, sd_n_blanks, sd_after,
            sd_p2_epochs, sd_p3_epochs, sd_kd_alpha, sd_act_alpha,
            sd_lr, sd_temp, sd_maxlen,
        ]
        sd_phase1_btn.click(lambda *a: run_sd_phase(1, *a),
                            _sd_phase_inputs, [adv_status, adv_log, adv_plot])
        sd_phase2_btn.click(lambda *a: run_sd_phase(2, *a),
                            _sd_phase_inputs, [adv_status, adv_log, adv_plot])
        sd_phase3_btn.click(lambda *a: run_sd_phase(3, *a),
                            _sd_phase_inputs, [adv_status, adv_log, adv_plot])
        sd_all_btn.click(
            lambda *a: run_advanced("stretch_distill", *a),
            [adv_ds_type, adv_ds_text, adv_ds_file, adv_ds_hf,
             gu_model, gu_out, gu_epochs, gu_batch, gu_accum, gu_lr, gu_maxlen, gu_save_n,
             la_model, la_out, la_epochs, la_batch, la_accum, la_lr,
             la_probe_every, la_spec_thresh, la_probe_bank,
             sd_model, sd_out, sd_n_blanks, sd_after,
             sd_p2_epochs, sd_p3_epochs, sd_kd_alpha, sd_act_alpha,
             sd_lr, sd_temp, sd_maxlen],
            [adv_status, adv_log, adv_plot]
        )
        adv_run_btn.click(
            run_advanced,
            [adv_mode,
             adv_ds_type, adv_ds_text, adv_ds_file, adv_ds_hf,
             gu_model, gu_out, gu_epochs, gu_batch, gu_accum, gu_lr, gu_maxlen, gu_save_n,
             gu_probe_bank, gu_probe_every,
             gu_alr_base, gu_alr_scale, gu_alr_temp, gu_alr_cap,
             la_model, la_out, la_epochs, la_batch, la_accum, la_lr,
             la_probe_every, la_spec_thresh, la_probe_bank,
             sd_model, sd_out, sd_n_blanks, sd_after,
             sd_p2_epochs, sd_p3_epochs, sd_kd_alpha, sd_act_alpha,
             sd_lr, sd_temp, sd_maxlen],
            [adv_status, adv_log, adv_plot]
        )
        adv_refresh.click(lambda: "\n".join(_adv_log_buf), [], [adv_log])
        alr_btn.click(compute_alr,
                      [alr_matrix_path, alr_base, alr_scale, alr_temp, alr_cap],
                      [alr_status, alr_plot])

    # ── TAB 5: Modular Training ───────────────────────────────────────────────
    with gr.Tab("🧩 Modular Training"):
        gr.Markdown(
            "**Modular Ground-Up Training** — trains from scratch on a general dataset "
            "while watching for cognitive capabilities to emerge naturally.\n\n"
            "Each **module** defines one capability domain (e.g. math, code, reasoning). "
            "Every `probe interval` steps, all modules are probed. When a module's layers "
            "cross the emergence threshold, the module's finetuning method fires automatically "
            "on those specific layers using the module's own dataset.\n\n"
            "| Method | What happens at emergence |\n"
            "|---|---|\n"
            "| `lora` | LoRA adapters injected into emerged layers, kept live for rest of training |\n"
            "| `junction` | Freeze all, train only the two boundary layers flanking the emerged region |\n"
            "| `full_layer` | Fully unfreeze and finetune emerged layers, then re-freeze |\n"
            "| `stretch` | Export → insert blank layers around emerged region → reload ⚠ resets optimizer |"
        )

        with gr.Row():
            mod_model = gr.Textbox(
                label="Architecture Source (model path or HF ID)",
                placeholder="Qwen/Qwen2-0.5B"
            )
            mod_out = gr.Textbox(label="Output Directory", value="./modular_output")

        gr.Markdown("#### General Training Dataset")
        mod_ds_type = gr.Radio(
            choices=["Text pairs (JSON in textbox)", "JSON file path", "HuggingFace dataset name"],
            value="HuggingFace dataset name",
            label="Dataset Source"
        )
        mod_ds_text = gr.Textbox(label="Pairs JSON", lines=2,
                                  value='[{"prompt":"Hello","response":"Hi there"}]')
        mod_ds_file = gr.Textbox(label="JSON File Path", placeholder="./data/train.json")
        mod_ds_hf   = gr.Textbox(label="HuggingFace Dataset Name",
                                  value="Dahoas/rm-static",
                                  placeholder="Dahoas/rm-static")

        gr.Markdown("#### Training Hyperparameters")
        with gr.Row():
            mod_epochs   = gr.Slider(1, 50, step=1, value=20,  label="Epochs")
            mod_batch    = gr.Slider(1, 8,  step=1, value=2,   label="Batch Size")
            mod_accum    = gr.Slider(1, 32, step=1, value=8,   label="Grad Accumulation")
        with gr.Row():
            mod_lr       = gr.Number(value=1e-3,               label="Learning Rate")
            mod_maxlen   = gr.Slider(64, 512, step=64, value=128, label="Max Length")
            mod_save_n   = gr.Number(value=500, precision=0,   label="Save every N steps")
        mod_probe_interval = gr.Slider(50, 1000, step=50, value=200,
                                        label="Probe interval (steps between all-module sweeps)")

        gr.Markdown("#### Modules")
        gr.Markdown(
            "Define each module as a JSON list. Each entry:\n"
            "```\n"
            "{\n"
            '  "name": "math",\n'
            '  "probe_bank": {"math": [["Q", "A"], ...]},\n'
            '  "finetune_dataset": "path/or/HF-name",\n'
            '  "finetune_method": "lora",          // lora | junction | full_layer | stretch\n'
            '  "emergence_threshold": 0.02,\n'
            '  "min_emergent_layers": 2,\n'
            '  "finetune_epochs": 1,\n'
            '  "finetune_lr": 0.0002,\n'
            '  "lora_rank": 8,\n'
            '  "lora_alpha": 16.0,\n'
            '  "n_blank_layers": 2,\n'
            '  "retrigger_after_steps": 500\n'
            "}\n"
            "```"
        )
        mod_modules_json = gr.Textbox(
            label="Modules JSON",
            lines=20,
            value=json.dumps([
                {
                    "name": "math",
                    "probe_bank": {
                        "math": [
                            ["What is 14x3", "42"],
                            ["What is 9473 squared?", "89737729"],
                            ["What is 78313 multiplied by 88537?", "6933598081"],
                            ["What is the cube root of 74088000000?", "4200"],
                            ["What is the cube root of 844444553408?", "9452"]
                        ]
                    },
                    "finetune_dataset": "Dahoas/rm-static",
                    "finetune_method": "lora",
                    "emergence_threshold": 0.02,
                    "min_subset_ratio": 0.75,
                    "finetune_epochs": 1,
                    "finetune_lr": 0.0002,
                    "lora_rank": 8,
                    "lora_alpha": 16.0,
                    "n_blank_layers": 2,
                    "retrigger_after_steps": 500
                }
            ], indent=2)
        )

        mod_run_btn  = gr.Button("▶ Start Modular Training", variant="primary")
        mod_status   = gr.Textbox(label="Status", lines=3)
        mod_log      = gr.Textbox(label="Log", lines=14)
        mod_plot     = gr.Plot(label="Loss Curve")
        mod_refresh  = gr.Button("↻ Refresh Log")

        _mod_log_buf: list = []

        def _mod_log_fn(msg):
            _mod_log_buf.append(msg)

        def run_modular(model_src, out_dir,
                        ds_type, ds_text, ds_file, ds_hf,
                        epochs, batch, accum, lr, maxlen, save_n,
                        probe_interval, modules_json):
            import matplotlib.pyplot as plt
            _mod_log_buf.clear()
            try:
                # Resolve dataset
                if ds_type == "Text pairs (JSON in textbox)":
                    dataset = [(d["prompt"], d["response"])
                               for d in json.loads(ds_text)]
                elif ds_type == "JSON file path":
                    dataset = ds_file
                else:
                    dataset = ds_hf

                # Parse modules
                raw_modules = json.loads(modules_json)
                from adaptive_trainer import TrainingModule, ModularGroundUpTrainer
                modules = []
                for m in raw_modules:
                    probe_bank = {
                        k: [tuple(p) for p in v]
                        for k, v in m["probe_bank"].items()
                    }
                    modules.append(TrainingModule(
                        name                 = m["name"],
                        probe_bank           = probe_bank,
                        finetune_dataset     = m["finetune_dataset"],
                        finetune_method      = m.get("finetune_method", "lora"),
                        emergence_threshold  = float(m.get("emergence_threshold", 0.02)),
                        min_subset_ratio     = float(m.get("min_subset_ratio", 0.75)),
                        finetune_epochs      = int(m.get("finetune_epochs", 1)),
                        finetune_lr          = float(m.get("finetune_lr", 2e-4)),
                        lora_rank            = int(m.get("lora_rank", 8)),
                        lora_alpha           = float(m.get("lora_alpha", 16.0)),
                        n_blank_layers       = int(m.get("n_blank_layers", 2)),
                        retrigger_after_steps= int(m.get("retrigger_after_steps", 500)),
                    ))

                trainer = ModularGroundUpTrainer(
                    model_source   = model_src,
                    modules        = modules,
                    probe_interval = int(probe_interval),
                    log            = _mod_log_fn,
                )
                log_data = trainer.train(
                    dataset_source     = dataset,
                    output_dir         = out_dir,
                    epochs             = int(epochs),
                    batch_size         = int(batch),
                    grad_accum         = int(accum),
                    learning_rate      = float(lr),
                    max_length         = int(maxlen),
                    save_every_n_steps = int(save_n),
                )

                xs  = [e["epoch"] for e in log_data["loss_history"]]
                ys  = [e["loss"]  for e in log_data["loss_history"]]
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(xs, ys, marker="o", color="steelblue")
                ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
                ax.set_title("Modular Ground-Up Training Loss")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                # Annotate emergence events on the loss curve
                for m in modules:
                    for event in m.trigger_history:
                        trigger_epoch = next(
                            (e["epoch"] for e in log_data["loss_history"]
                             if e.get("step", 0) >= event["step"]),
                            None)
                        if trigger_epoch:
                            ax.axvline(trigger_epoch, color="orange",
                                       linestyle="--", alpha=0.6,
                                       label=f"{m.name} emerged")
                ax.legend(fontsize=7)

                n_triggers = sum(len(m.trigger_history) for m in modules)
                status = (f"✓ Modular training complete\n"
                          f"Best checkpoint: {log_data['best_checkpoint']}\n"
                          f"Total module triggers: {n_triggers}")
                return status, "\n".join(_mod_log_buf), fig

            except Exception as e:
                import traceback
                return f"Error: {e}", traceback.format_exc(), None

        mod_run_btn.click(
            run_modular,
            inputs=[
                mod_model, mod_out,
                mod_ds_type, mod_ds_text, mod_ds_file, mod_ds_hf,
                mod_epochs, mod_batch, mod_accum, mod_lr, mod_maxlen, mod_save_n,
                mod_probe_interval, mod_modules_json,
            ],
            outputs=[mod_status, mod_log, mod_plot]
        )
        mod_refresh.click(lambda: "\n".join(_mod_log_buf), [], [mod_log])

demo.launch()
