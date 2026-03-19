# Quickstart Guide

Get your first LLM fine-tuning run up and running, from installation to a trained model.

---

## System Requirements

| Component | Minimum | Recommended | Ideal |
|-----------|---------|-------------|-------|
| Python | 3.10 | 3.12 | 3.12 |
| GPU VRAM | None (CPU works) | 24 GB (LoRA) | 80 GB (Full fine-tune) |
| System RAM | 16 GB | 32 GB | 64 GB+ |
| Disk Space | 20 GB | 50 GB | 100 GB+ |

Python 3.13 is **not supported** due to torch/ML wheel compatibility issues. Use Python 3.10, 3.11, or 3.12.

A GPU is optional. The CPU quickstart below uses a 135M-parameter model and completes in under 5 minutes on a modern laptop.

### GPU Compatibility Quick Reference

| GPU | VRAM | Best Training Mode | Notes |
|-----|------|-------------------|-------|
| No GPU / Apple Silicon | -- | LoRA (CPU/MPS) | Works for tiny models (<500M params) |
| RTX 3060 | 12 GB | QLoRA | 1B-3B models |
| RTX 3090 / 4090 | 24 GB | LoRA / QLoRA | Up to 7B models with QLoRA |
| A100 40GB | 40 GB | LoRA / Full (small) | 7B LoRA, 1-3B full fine-tune |
| A100 80GB | 80 GB | Full fine-tune | Up to 7B full fine-tune |
| H100 80GB | 80 GB | Full fine-tune + FP8 | FP8 support for 2x throughput |

---

## Installation

### Option A: Install from source (recommended for development)

```bash
git clone https://github.com/Nagavenkatasai7/llm-forge.git
cd llm-forge
pip install -e .
```

### Option B: Install from PyPI

```bash
pip install llm-forge
```

### Install optional extras

llm-forge uses optional dependency groups for different capabilities:

```bash
# All extras (everything)
pip install llm-forge[all]

# Individual extras
pip install llm-forge[rag]          # RAG pipeline (ChromaDB, LlamaIndex, LangChain)
pip install llm-forge[serve]        # Serving (Gradio, FastAPI, vLLM)
pip install llm-forge[eval]         # Evaluation (lm-eval-harness, ROUGE, NLTK)
pip install llm-forge[cleaning]     # Data cleaning (ftfy, presidio, detoxify, spacy)
pip install llm-forge[distributed]  # Distributed training (DeepSpeed, Transformer Engine)
pip install llm-forge[desktop]      # Desktop UI (pywebview + Gradio)
pip install llm-forge[dev]          # Development tools (pytest, ruff, mypy)
```

### Verify installation

```bash
llm-forge --version
```

Expected output:

```
llm-forge 0.1.0
```

---

## HuggingFace Login

Many models (Llama, Mistral, Gemma) are gated and require you to accept their license on HuggingFace before downloading. Even for non-gated models, logging in avoids rate limits.

```bash
pip install huggingface_hub
huggingface-cli login
```

You will be prompted for an access token. Create one at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with `read` permissions.

For gated models like Llama, you must also visit the model page (e.g., [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)) and accept the license agreement.

---

## CPU Quickstart (No GPU Required)

This uses `quickstart_tiny.yaml`, which fine-tunes a 135M-parameter model on 22 included sample records. It completes in under 5 minutes on CPU.

### 1. Examine the included config

The repository ships with a ready-to-run config at `configs/quickstart_tiny.yaml`:

```yaml
model:
  name: "HuggingFaceTB/SmolLM2-135M"    # Tiny model, downloads in seconds
  max_seq_length: 512
  attn_implementation: "sdpa"             # Works everywhere, no flash-attn needed
  torch_dtype: "bf16"

lora:
  r: 8                                    # Small rank for fast training
  alpha: 16
  dropout: 0.05
  target_modules:
    - q_proj
    - v_proj

data:
  train_path: "./examples/data/sample_train.jsonl"   # 22 samples included in repo
  format: "alpaca"
  test_size: 0.2

training:
  mode: "lora"
  output_dir: "./outputs/quickstart-tiny/"
  num_epochs: 2
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 1
  learning_rate: 3.0e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  bf16: true
  save_steps: 100
  logging_steps: 1
  report_to:
    - "none"                              # No wandb needed for testing

evaluation:
  enabled: false                          # Skip benchmarks for quick test

serving:
  export_format: "safetensors"
  merge_adapter: true
```

### 2. Validate the config

```bash
llm-forge validate configs/quickstart_tiny.yaml
```

This parses the YAML, checks all field types and ranges, estimates memory requirements, and reports any warnings. If the config is valid, you will see:

```
Config is valid.
```

### 3. Run training

```bash
llm-forge train --config configs/quickstart_tiny.yaml --verbose
```

The `--verbose` flag enables detailed logging. You will see progress bars, loss values, and a summary at the end.

### 4. Inspect the output

After training completes, the output directory contains:

```
outputs/quickstart-tiny/
  checkpoint-*/          # Intermediate checkpoints
  merged/                # Final merged model (base + LoRA adapter)
  adapter_model/         # LoRA adapter weights (before merging)
  training_args.bin      # Saved training arguments
```

---

## GPU Training: Llama-3.2-1B on Alpaca

This example fine-tunes Meta's Llama-3.2-1B on the Alpaca instruction-following dataset using LoRA. Requires a GPU with at least 16 GB VRAM.

### 1. Accept the Llama license

Visit [https://huggingface.co/meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and accept the license. Make sure you are logged in via `huggingface-cli login`.

### 2. Review the config

The repository includes `configs/demo_lora_llama.yaml`:

```yaml
model:
  name: "meta-llama/Llama-3.2-1B"
  max_seq_length: 2048
  attn_implementation: "flash_attention_2"
  torch_dtype: "bf16"

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

data:
  train_path: "tatsu-lab/alpaca"           # Auto-downloaded from HuggingFace Hub
  format: "alpaca"
  test_size: 0.05
  cleaning:
    enabled: true
    quality_preset: "balanced"
    unicode_fix: true
    dedup_enabled: true

training:
  mode: "lora"
  output_dir: "./outputs/demo-lora-llama3.2-1b/"
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4            # Effective batch size = 16
  learning_rate: 2.0e-4
  gradient_checkpointing: true
  bf16: true
  neftune_noise_alpha: 5.0
  report_to:
    - "none"

evaluation:
  enabled: true
  benchmarks:
    - hellaswag
    - arc_easy
  generate_report: true
```

### 3. Preview without training (dry run)

```bash
llm-forge train --config configs/demo_lora_llama.yaml --dry-run
```

This prints the full pipeline plan, config summary, and memory estimates without starting any training.

### 4. Start training

```bash
llm-forge train --config configs/demo_lora_llama.yaml --verbose
```

On a single A100 80GB, this takes approximately 30 minutes for 3 epochs on ~52K Alpaca samples.

### 5. Evaluate the trained model

```bash
llm-forge eval --config configs/demo_lora_llama.yaml \
  --model-path ./outputs/demo-lora-llama3.2-1b
```

This runs the configured benchmarks (HellaSwag, ARC Easy) and prints a results table.

---

## Validating Configs

Always validate before training. The validator catches type errors, missing required fields, conflicting settings (e.g., both `bf16` and `fp16` enabled), and potential issues like dangerously high learning rates.

```bash
llm-forge validate path/to/config.yaml
```

The validator also estimates VRAM requirements and compares them against your available GPU memory.

---

## Viewing Results

### Training output structure

```
outputs/your-run/
  checkpoint-100/         # Intermediate checkpoints (kept up to save_total_limit)
  checkpoint-200/
  merged/                 # Final model with LoRA adapter merged into base weights
  adapter_model/          # Standalone LoRA adapter (small, portable)
  training_args.bin       # Serialized training arguments
  trainer_state.json      # Training history (loss, learning rate per step)
```

### Evaluation reports

When `evaluation.generate_report: true`, an HTML report is generated after benchmarks complete, summarizing per-task scores with comparisons to the base model.

---

## Serving Your Model

### Gradio chat interface

```bash
llm-forge serve --config config.yaml --model-path ./outputs/my-model/merged
```

Open `http://localhost:7860` in your browser for an interactive chat interface.

### Desktop mode (native window)

```bash
llm-forge ui --desktop
```

Wraps the Gradio interface in a native window using pywebview. Requires `pip install llm-forge[desktop]`.

### Export to GGUF for Ollama

To serve your model through Ollama, configure GGUF export in your YAML:

```yaml
serving:
  export_format: "gguf"
  gguf_quantization: "Q4_K_M"
  merge_adapter: true
  generate_modelfile: true
  inference_temperature: 0.1
  inference_num_ctx: 2048
```

Then run the export:

```bash
llm-forge export --config config.yaml --model-path ./outputs/my-model/merged
```

This produces a quantized GGUF file and an Ollama Modelfile. Import into Ollama with:

```bash
ollama create my-model -f outputs/my-model/Modelfile
ollama run my-model
```

---

## Initializing a New Project

If you want to start from scratch rather than using an included config:

```bash
llm-forge init --template lora --output my-project.yaml
```

Available templates: `lora`, `qlora`, `pretrain`, `rag`, `full`

This creates a pre-filled YAML config that you can edit. Then validate and train:

```bash
llm-forge validate my-project.yaml
llm-forge train --config my-project.yaml
```

---

## Other CLI Commands

```bash
# Detect your hardware profile
llm-forge hardware

# Clean your training data (runs the 7-stage pipeline)
llm-forge clean --config config.yaml

# Run only specific pipeline stages
llm-forge train --config config.yaml --stages data_loading,training

# Skip specific pipeline stages
llm-forge train --config config.yaml --skip-stages evaluation,iti_baking

# Disable auto hardware optimization (use config values exactly as written)
llm-forge train --config config.yaml --no-auto-optimize

# RAG operations
llm-forge rag build --config config.yaml    # Build the vector index
llm-forge rag query --config config.yaml    # Interactive RAG query
```

---

## Troubleshooting Quick Fixes

| Issue | Solution |
|-------|----------|
| `OutOfMemoryError` | Switch to `mode: "qlora"`, reduce `per_device_train_batch_size`, or enable `gradient_checkpointing: true` |
| `flash_attn not installed` | Change `attn_implementation` to `"sdpa"` (works everywhere) |
| `BF16 not supported` | Set `bf16: false` and `fp16: true` for older GPUs |
| Model download fails | Run `huggingface-cli login` and accept the model license on HuggingFace |
| Config validation error | Run `llm-forge validate config.yaml` for detailed error messages with suggestions |
| Python 3.13 errors | Use Python 3.10-3.12; 3.13 is not supported |
| `auto_optimize` overrides your settings | Pass `--no-auto-optimize` to use your config values exactly |

---

## Next Steps

- [Configuration Reference](configuration.md) -- every YAML field explained with types, defaults, and ranges
- [Data Preparation Guide](data_preparation.md) -- format, clean, and augment your training data
- [Training Guide](training_guide.md) -- deep dive into training modes (LoRA, QLoRA, full, pretrain, DPO, ORPO, GRPO)
- [Evaluation Guide](evaluation_guide.md) -- benchmark your models with lm-eval-harness
- [Deployment Guide](deployment.md) -- serve models via Gradio, FastAPI, vLLM, or Ollama
- [Distributed Training](distributed_training.md) -- multi-GPU training with FSDP, DeepSpeed, and Megatron
