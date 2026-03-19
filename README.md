# llm-forge

**Build your own AI model. No ML expertise required.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/Nagavenkatasai7/llm-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/Nagavenkatasai7/llm-forge/actions)

llm-forge is a conversational AI assistant that helps you build custom language models. Just type `llm-forge` and tell it what you want — it handles everything: data cleaning, training, evaluation, and deployment.

## Install (One Command)

```bash
curl -fsSL https://raw.githubusercontent.com/Nagavenkatasai7/llm-forge/main/install.sh | bash
```

Then open a new terminal (or run `source ~/.zshrc`) and type:

```bash
llm-forge
```

That's it. LLM Forge will guide you through building your AI model.

**Already have Python 3.10+?** You can also install directly:

```bash
pip install llm-forge-new
llm-forge
```

## What Happens When You Run It

```
$ llm-forge

╭─ LLM Forge — Build your own AI model ─╮
│  Just tell me what you want to build.  │
╰────────────────────────────────────────╯

You: I want to build a customer support chatbot

Forge: I'll set that up. Let me check your hardware...
       Found: NVIDIA RTX 4090 (24 GB VRAM)

       I recommend Llama-3.2-1B with LoRA fine-tuning.
       Where is your training data?

You: ~/data/support_tickets.jsonl

Forge: Found 2,400 records in Alpaca format.
       Config saved. Starting training...
       ████████████░░░░░░  Step 142/300  loss: 1.42  ETA: 8 min
```

LLM Forge works as a conversational manager — you talk to it, it takes action. No YAML editing, no CLI commands to memorize.

**No API key?** LLM Forge includes a free guided wizard that works offline — no API needed.

**Want the full AI-powered experience?** Set a Claude API key:

```bash
export ANTHROPIC_API_KEY=your-key-here
llm-forge
```

## Quick Demo (5 Minutes, No Config)

```bash
llm-forge demo
```

Trains a tiny 135M model on bundled sample data. Works on CPU. Zero configuration.

## Prerequisites

- **Python 3.10+** (the installer handles this)
- **NVIDIA GPU with 8+ GB VRAM** (recommended) or CPU for testing
- **20 GB free disk space**

No GPU? You can test on CPU with small models, or use [Google Colab](https://colab.research.google.com) for free GPU access.

## Installation Options

```bash
# Option 1: One-line installer (recommended — handles everything)
curl -fsSL https://raw.githubusercontent.com/Nagavenkatasai7/llm-forge/main/install.sh | bash

# Option 2: pip install (if you have Python 3.10+)
pip install llm-forge-new

# Option 3: From source (for development)
git clone https://github.com/Nagavenkatasai7/llm-forge.git
cd llm-forge
pip install -e ".[all]"
```

## How It Works

```
You edit this:                  llm-forge handles the rest:
+-----------------+
|   config.yaml   |  ------>   Data Cleaning -> Training -> Evaluation -> Serving
+-----------------+
```

**1. Generate a starter config:**
```bash
llm-forge init --template lora
```

**2. Edit `config.yaml`** - set your model, data path, and hyperparameters. Every field has inline documentation.

**3. Validate before training** (catches errors early):
```bash
llm-forge validate config.yaml
```

**4. Train:**
```bash
llm-forge train --config config.yaml
```

llm-forge auto-detects your hardware and optimizes accordingly.

## Example: Train on Your Own Data

Create a JSONL file with instruction-output pairs:

```json
{"instruction": "Summarize this article", "input": "The economy grew...", "output": "Economic growth..."}
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}
```

Then create a config (`config.yaml`):

```yaml
model:
  name: "meta-llama/Llama-3.2-1B"
  max_seq_length: 2048

data:
  train_path: "./data/train.jsonl"    # Your data file
  format: "alpaca"

training:
  mode: "lora"                        # Memory-efficient fine-tuning
  output_dir: "./outputs/my-model"
  num_epochs: 3
  learning_rate: 2.0e-4
```

Run `llm-forge train --config config.yaml` and you're done.

## Features

### Getting Started
| Feature | Description |
|---------|-------------|
| **YAML-First Config** | Single config file controls the entire pipeline |
| **Hardware Auto-Detection** | Works seamlessly from RTX 3060 to H100 clusters |
| **Smart Validation** | Catches config errors with actionable suggestions |

### Training
| Feature | Description |
|---------|-------------|
| **LoRA / QLoRA** | Memory-efficient fine-tuning (train 7B models on 24GB GPUs) |
| **Full Fine-Tuning** | Unrestricted parameter updates when you have the VRAM |
| **Pre-Training** | Train language models from scratch |
| **DPO Alignment** | Direct Preference Optimization for RLHF |
| **NEFTune** | Noise-based regularization for better generalization |

### Data & Evaluation
| Feature | Description |
|---------|-------------|
| **Data Cleaning** | Unicode fixing, deduplication, PII redaction, toxicity filtering |
| **Evaluation** | lm-evaluation-harness benchmarks with HTML reports |
| **RAG Pipeline** | Chunking, embeddings, hybrid retrieval, reranking |

### Production
| Feature | Description |
|---------|-------------|
| **Distributed Training** | FSDP, DeepSpeed ZeRO, Megatron-Core |
| **Serving** | Gradio UI, FastAPI REST API, vLLM high-throughput |
| **Model Export** | safetensors, GGUF, ONNX formats |
| **HPC Support** | SLURM scripts, Singularity containers |

## Recommended Starting Models

| Model | Size | Best For | GPU Needed |
|-------|------|----------|------------|
| [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | 135M | Quick testing, learning | Any (even CPU) |
| [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) | 1B | General fine-tuning (recommended) | 8+ GB VRAM |
| [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B) | 1.5B | Multilingual, code | 12+ GB VRAM |
| [`meta-llama/Llama-3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B) | 3B | Higher quality results | 16+ GB VRAM |
| [`microsoft/phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/phi-3-mini-4k-instruct) | 3.8B | Instruction tuning | 24+ GB VRAM |

## Hardware Compatibility

| GPU | VRAM | Recommended Mode | Max Model Size |
|-----|------|-----------------|---------------|
| RTX 3090 | 24 GB | QLoRA (4-bit) | 7B |
| RTX 4090 | 24 GB | LoRA / QLoRA | 7B |
| A100 40GB | 40 GB | LoRA | 13B |
| A100 80GB | 80 GB | Full fine-tune | 7B |
| H100 80GB | 80 GB | Full fine-tune + FP8 | 13B |
| Multi-GPU | Varies | FSDP / DeepSpeed | 70B+ |
| CPU only | N/A | QLoRA (slow) | 1B |

## CLI Reference

```bash
llm-forge init [--template lora|qlora|pretrain|rag|full]   # Generate starter config
llm-forge validate config.yaml                              # Validate + hardware check
llm-forge train --config config.yaml [--verbose] [--dry-run]  # Train model
llm-forge eval --config config.yaml                         # Run benchmarks
llm-forge serve --config config.yaml                        # Launch chat UI
llm-forge export --config config.yaml --format gguf         # Export model
llm-forge clean --config config.yaml                        # Data cleaning only
llm-forge rag build --config config.yaml                    # Build RAG index
llm-forge rag query "question" --config config.yaml         # Query RAG
llm-forge info                                              # System + GPU info
llm-forge hardware                                          # Hardware summary
```

## Example Configs

Ready-to-use configs in `configs/`:

| Config | Use Case |
|--------|----------|
| `quickstart_tiny.yaml` | 5-min test: SmolLM2-135M on sample data (works on CPU) |
| `demo_lora_llama.yaml` | Demo: Llama-3.2-1B on Alpaca (copy-paste ready) |
| `benchmark_smollm_135m.yaml` | Benchmark: SmolLM2-135M (<5 min on A100) |
| `benchmark_smollm_360m.yaml` | Benchmark: SmolLM2-360M (~10 min on A100) |
| `benchmark_tinyllama_1b.yaml` | Benchmark: TinyLlama-1.1B (~15 min on A100) |
| `benchmark_qlora_phi2.yaml` | Benchmark: Phi-2 QLoRA 4-bit (~20 min on A100) |
| `benchmark_llama_1b_full.yaml` | Benchmark: Llama-3.2-1B full LoRA (~30 min on A100) |
| `example_lora.yaml` | LoRA fine-tuning template |
| `example_qlora.yaml` | QLoRA for memory-constrained GPUs |
| `example_pretrain.yaml` | Pre-training from scratch |
| `example_rag.yaml` | RAG pipeline |
| `example_medical_domain.yaml` | Medical domain fine-tuning |
| `example_legal_domain.yaml` | Legal domain fine-tuning |
| `example_code_domain.yaml` | Code generation fine-tuning |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `OutOfMemoryError` | Use `mode: "qlora"` in your config |
| `401 Unauthorized` from HuggingFace | Run `huggingface-cli login` |
| `flash_attention_2 not found` | OK - llm-forge auto-falls back to SDPA |
| Training interrupted | Rerun same command - auto-resumes from last checkpoint |
| `BF16 not supported` | llm-forge auto-falls back to FP16 on older GPUs |
| Model download fails | Rerun - downloads auto-resume |

## Docker

```bash
# GPU training
docker build -t llm-forge:gpu -f docker/Dockerfile.gpu .
docker run --gpus all -v $(pwd)/outputs:/app/outputs \
  llm-forge:gpu train --config configs/demo_lora_llama.yaml

# Docker Compose (training + serving + vector DB)
docker compose -f docker/docker-compose.yml up
```

## HPC / SLURM

```bash
# Single GPU on Hopper cluster
sbatch scripts/slurm/train_demo.sbatch

# Multi-node distributed
sbatch scripts/slurm/train_multi_node.sbatch
```

See [Hopper Cluster Guide](docs/hopper_cluster_guide.md) for detailed setup.

## Documentation

| Doc | For |
|-----|-----|
| [Quickstart Guide](docs/quickstart.md) | First-time setup walkthrough |
| [Configuration Reference](docs/configuration.md) | All YAML fields explained |
| [Data Preparation](docs/data_preparation.md) | Preparing your dataset |
| [Training Guide](docs/training_guide.md) | LoRA vs QLoRA vs full fine-tuning |
| [Evaluation Guide](docs/evaluation_guide.md) | Benchmarking your model |
| [Deployment Guide](docs/deployment.md) | Serving in production |
| [Distributed Training](docs/distributed_training.md) | Multi-GPU scaling |
| [API Reference](docs/api_reference.md) | Python API docs |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Citation

```bibtex
@software{llm_forge,
  author = {Chennu, Naga Venkata Sai},
  title = {llm-forge: Config-Driven LLM Training Platform},
  year = {2026},
  url = {https://github.com/Nagavenkatasai7/llm-forge}
}
```

## License

[Apache License 2.0](LICENSE)

## Author

**Naga Venkata Sai Chennu** - George Mason University
- GitHub: [@Nagavenkatasai7](https://github.com/Nagavenkatasai7)
- Email: nchennu@gmu.edu
