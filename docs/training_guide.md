# Training Guide

Deep dive into all training modes supported by llm-forge: LoRA, QLoRA, full fine-tuning, pre-training from scratch, and DPO alignment.

---

## Training Modes Overview

| Mode | Config Value | Description | VRAM Requirement | Trainable Params |
|------|-------------|-------------|-----------------|------------------|
| LoRA | `lora` | Low-Rank Adaptation of attention/MLP layers | 16-24 GB | ~0.1-0.5% |
| QLoRA | `qlora` | LoRA with 4-bit quantized base model | 8-16 GB | ~0.1-0.5% |
| Full Fine-Tune | `full` | Update all model parameters | 40-80+ GB | 100% |
| Pre-Training | `pretrain` | Train a model from random initialization | Varies | 100% |
| DPO | `dpo` | Direct Preference Optimization alignment | 24-80 GB | ~0.1-0.5% (LoRA) |

---

## LoRA Fine-Tuning

LoRA (Low-Rank Adaptation) freezes the base model weights and adds small trainable rank-decomposition matrices to selected layers. This dramatically reduces memory usage and training time while achieving performance comparable to full fine-tuning.

### How LoRA Works

For a pre-trained weight matrix `W` of dimension `d x k`, LoRA adds:

```
W' = W + (alpha/r) * B * A
```

Where `A` is `r x k`, `B` is `d x r`, and `r << min(d, k)`.

### Configuration

```yaml
model:
  name: "meta-llama/Llama-3.2-1B"
  max_seq_length: 2048
  torch_dtype: "bf16"

lora:
  r: 16                    # Rank (higher = more capacity, more memory)
  alpha: 32                # Scaling factor (typically 2x rank)
  dropout: 0.05            # Regularization dropout
  target_modules:          # Which layers to adapt
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  bias: "none"             # "none", "all", or "lora_only"
  task_type: "CAUSAL_LM"
  use_rslora: false        # Rank-Stabilized LoRA
  use_dora: false          # Weight-Decomposed LoRA (DoRA)

training:
  mode: "lora"
  output_dir: "./outputs/my-lora"
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  bf16: true

data:
  train_path: "tatsu-lab/alpaca"
  format: "alpaca"
```

### LoRA Rank Selection Guide

| Rank (`r`) | Use Case | Trainable Params (7B model) |
|-----------|----------|---------------------------|
| 8 | Quick experiments, simple tasks | ~1.7M |
| 16 | General instruction following (default) | ~3.4M |
| 32 | Complex domain adaptation | ~6.8M |
| 64 | Near full fine-tune expressiveness | ~13.6M |
| 128+ | Maximum adapter capacity | ~27M+ |

### Target Module Selection

| Module Pattern | Layer Type | When to Include |
|---------------|-----------|-----------------|
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | Attention | Always (core of LoRA) |
| `gate_proj`, `up_proj`, `down_proj` | MLP / FFN | Recommended for better results |
| `embed_tokens` | Input embedding | Only for vocabulary expansion |
| `lm_head` | Output head | Only for vocabulary expansion |

### Advanced LoRA Features

**RSLoRA (Rank-Stabilized LoRA):**

Scales the LoRA output by `1/sqrt(r)` instead of `1/r`, providing more stable training at higher ranks:

```yaml
lora:
  r: 64
  use_rslora: true
```

**DoRA (Weight-Decomposed LoRA):**

Decomposes weights into magnitude and direction components, updating them independently for improved learning:

```yaml
lora:
  use_dora: true
```

---

## QLoRA Fine-Tuning

QLoRA combines LoRA with 4-bit quantization of the base model weights, reducing VRAM requirements by roughly 4x compared to standard LoRA. The base model is loaded in NF4 (4-bit NormalFloat) precision while LoRA adapter weights remain in full precision.

### Configuration

```yaml
model:
  name: "meta-llama/Llama-3.2-3B"
  max_seq_length: 2048
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

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bf16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true

training:
  mode: "qlora"
  output_dir: "./outputs/my-qlora"
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  optim: "paged_adamw_32bit"    # Auto-configured for QLoRA
  bf16: true
  gradient_checkpointing: true   # Recommended for QLoRA

data:
  train_path: "tatsu-lab/alpaca"
  format: "alpaca"
```

### QLoRA Auto-Configuration

When you set `mode: "qlora"`, llm-forge automatically:

1. Enables `load_in_4bit: true` if quantization is not already configured
2. Sets `bnb_4bit_quant_type: "nf4"` and `bnb_4bit_use_double_quant: true`
3. Switches the optimizer from `adamw_torch` to `paged_adamw_32bit`
4. Prepares the model for k-bit training (handles gradient requirements)

### QLoRA vs LoRA Trade-offs

| Aspect | LoRA | QLoRA |
|--------|------|-------|
| Base model precision | FP16/BF16 (16-bit) | NF4 (4-bit) |
| VRAM for 7B model | ~16-18 GB | ~6-8 GB |
| Training speed | Faster | Slightly slower |
| Quality | Slightly higher | Very close to LoRA |
| Optimizer | `adamw_torch` | `paged_adamw_32bit` |

---

## Full Fine-Tuning

Full fine-tuning updates all model parameters. This provides maximum flexibility but requires significantly more VRAM and compute.

### Configuration

```yaml
model:
  name: "meta-llama/Llama-3.2-1B"
  max_seq_length: 2048
  torch_dtype: "bf16"

training:
  mode: "full"
  output_dir: "./outputs/my-full-ft"
  num_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5           # Lower LR than LoRA
  bf16: true
  gradient_checkpointing: true     # Essential for memory savings
  weight_decay: 0.1               # Higher for full fine-tuning
  warmup_ratio: 0.1

data:
  train_path: "tatsu-lab/alpaca"
  format: "alpaca"
```

### When to Use Full Fine-Tuning

- Significant domain shift (e.g., English model to code, medical, legal)
- Large amounts of training data (100K+ examples)
- Maximum quality is more important than efficiency
- Hardware budget supports it (A100 80GB or better)

### Key Differences from LoRA

| Setting | LoRA | Full Fine-Tune |
|---------|------|----------------|
| Learning rate | `2e-4` | `2e-5` (10x lower) |
| Weight decay | `0.01` | `0.1` (higher) |
| Warmup ratio | `0.03` | `0.1` (longer warmup) |
| Gradient checkpointing | Optional | Strongly recommended |
| Batch size | 4 | 2 (limited by VRAM) |
| Grad accum steps | 4 | 8+ (to compensate for smaller batch) |

---

## Pre-Training from Scratch

llm-forge supports training transformer language models from random initialization using the Llama architecture.

### Model Size Presets

| Preset | Hidden Size | Layers | Heads | Intermediate | KV Heads | Approx Params |
|--------|------------|--------|-------|-------------|----------|---------------|
| `125M` | 768 | 12 | 12 | 3072 | 12 | ~125M |
| `350M` | 1024 | 24 | 16 | 4096 | 16 | ~350M |
| `760M` | 1536 | 24 | 16 | 6144 | 16 | ~760M |
| `1B` | 2048 | 24 | 16 | 8192 | 8 | ~1B |

### Configuration

```yaml
model:
  name: "scratch-125M"          # Descriptive name (not a HF model ID)
  max_seq_length: 2048
  torch_dtype: "bf16"

training:
  mode: "pretrain"
  output_dir: "./outputs/my-pretrained-model"
  num_epochs: 1
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 8
  learning_rate: 6.0e-4          # Higher LR for pre-training
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.05
  weight_decay: 0.1
  bf16: true
  gradient_checkpointing: true

data:
  train_path: "./data/corpus/"   # Directory of text files
  format: "completion"
```

### Pre-Training Pipeline

The pre-training pipeline follows these steps:

1. **Train a BPE tokenizer** from your text corpus (vocab_size=32,000)
2. **Build the model** from scratch using the Llama architecture with specified dimensions
3. **Tokenize the dataset** and group into fixed-length blocks for causal language modelling
4. **Train** with linear warmup and cosine decay scheduling
5. **Save** the final model and tokenizer

### Tokenizer Training

The BPE tokenizer is trained on your corpus with the following settings:
- NFC unicode normalization
- Byte-level pre-tokenization (GPT-2 style)
- Special tokens: `<pad>`, `<s>`, `</s>`, `<unk>`

### Weight Initialization

All model weights are initialized from a normal distribution with `std=0.02`, following the original Llama paper.

---

## DPO Alignment

Direct Preference Optimization (DPO) aligns a model to human preferences without requiring a separate reward model. It trains directly on preference pairs (chosen vs. rejected responses).

### Preference Dataset Format

The dataset must contain three fields: `prompt`, `chosen`, and `rejected`.

```json
{
  "prompt": "Explain quantum computing in simple terms.",
  "chosen": "Quantum computing uses qubits that can be in multiple states simultaneously...",
  "rejected": "Quantum computing is very complicated and hard to understand..."
}
```

### Configuration

```yaml
model:
  name: "meta-llama/Llama-3.2-1B-Instruct"
  max_seq_length: 1024
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

training:
  mode: "dpo"
  output_dir: "./outputs/my-dpo"
  num_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 5.0e-7           # Very low LR for alignment
  bf16: true

data:
  train_path: "your-org/preference-dataset"
  format: "custom"
  test_size: 0.05
```

### DPO Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 0.1 | KL divergence regularization. Higher = closer to reference model. Range: 0.05-0.5. |
| `loss_type` | `"sigmoid"` | Loss variant: `"sigmoid"`, `"hinge"`, `"ipo"`, `"kto_pair"` |
| `max_length` | 1024 | Maximum combined length of prompt + response |
| `max_prompt_length` | 512 | Maximum length of the prompt portion |

### Reference Model Handling

- **With LoRA/PEFT:** DPOTrainer shares weights between the policy and reference model automatically. No separate reference model is loaded.
- **Without PEFT:** A separate frozen copy of the base model is loaded as the reference model.

### RLHF (PPO) Training

llm-forge also supports PPO-based RLHF with an explicit reward model. This is a more complex pipeline:

1. Load the policy model and a separate reward model
2. Generate responses from the policy model
3. Score responses with the reward model
4. Update the policy using Proximal Policy Optimization

```python
from llm_forge.training.alignment import AlignmentTrainer

aligner = AlignmentTrainer(config)
model, ref_model, tokenizer = aligner.setup_dpo()
reward_model = aligner.setup_reward_model("OpenAssistant/reward-model-deberta-v3-large-v2")

results = aligner.train_rlhf(
    model=model,
    dataset=dataset,
    reward_model=reward_model,
    init_kl_coef=0.2,
    target_kl=6.0,
)
```

---

## Hyperparameter Tuning Tips

### Learning Rate

The learning rate is the most important hyperparameter. Recommended starting points:

| Training Mode | Recommended LR | Range |
|--------------|----------------|-------|
| LoRA | `2e-4` | `1e-4` to `5e-4` |
| QLoRA | `2e-4` | `1e-4` to `5e-4` |
| Full fine-tune | `2e-5` | `5e-6` to `5e-5` |
| Pre-training | `6e-4` | `3e-4` to `1e-3` |
| DPO | `5e-7` | `1e-7` to `5e-6` |

### Batch Size and Gradient Accumulation

The effective batch size is: `per_device_batch_size * gradient_accumulation_steps * num_gpus`

Recommended effective batch sizes:

| Dataset Size | Effective Batch Size |
|-------------|---------------------|
| < 10K samples | 16-32 |
| 10K - 100K | 32-64 |
| 100K+ | 64-128 |

### Epochs

| Dataset Size | Recommended Epochs |
|-------------|-------------------|
| < 1K samples | 5-10 |
| 1K - 10K | 3-5 |
| 10K - 100K | 2-3 |
| 100K+ | 1-2 |

### Learning Rate Scheduler

```yaml
training:
  lr_scheduler_type: "cosine"    # Recommended for most cases
  warmup_ratio: 0.03             # 3% of total steps for warmup
```

Available schedulers: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`, `inverse_sqrt`, `reduce_on_plateau`.

### NEFTune Regularization

NEFTune adds noise to the embedding layer during training, which has been shown to improve instruction-following performance:

```yaml
training:
  neftune_noise_alpha: 5.0       # Typical range: 1.0-15.0
```

---

## Training Callbacks

llm-forge includes several built-in callbacks that activate automatically based on your configuration.

### Automatic Callbacks

| Callback | Trigger | Behavior |
|----------|---------|----------|
| `RichProgressCallback` | Rich library installed | Live progress bar with loss, LR, throughput |
| `GPUMonitorCallback` | CUDA available | Logs GPU memory utilization every 50 steps |
| `WandBCallback` | `report_to: ["wandb"]` | Logs all metrics to Weights & Biases |
| `EarlyStoppingCallback` | `eval_strategy != "no"` | Stops training after 3 evals with no improvement |
| `CheckpointCallback` | Always | Saves timed checkpoints every 30 minutes |

### WandB Integration

```yaml
training:
  report_to: ["wandb"]
```

Set your WandB API key before training:

```bash
export WANDB_API_KEY="your-key-here"
# Or
wandb login
```

### Disabling WandB

```yaml
training:
  report_to: []                   # No experiment tracking
```

---

## Unsloth Acceleration

llm-forge optionally integrates with [Unsloth](https://github.com/unslothai/unsloth) for accelerated LoRA training with custom CUDA kernels.

```yaml
training:
  use_unsloth: true
```

When enabled, Unsloth provides:
- 2x faster training speed
- Up to 60% less memory usage
- Custom fused kernels for attention and MLP layers
- Automatic LoRA application with optimized gradient checkpointing

**Requirements:** Unsloth must be installed separately: `pip install unsloth`

---

## Resuming from Checkpoints

Resume a training run from a saved checkpoint:

```yaml
training:
  resume_from_checkpoint: "./outputs/my-lora/checkpoint-1000"
```

Or via CLI:

```bash
llm-forge train --config config.yaml --resume ./outputs/my-lora/checkpoint-1000
```

---

## Merging LoRA Adapters

After LoRA/QLoRA training, merge the adapter weights back into the base model for deployment:

```yaml
serving:
  merge_adapter: true
```

This creates a `merged/` subdirectory in the output directory containing the full model with adapter weights baked in.

To merge manually via Python:

```python
from llm_forge.training.finetuner import FineTuner

finetuner = FineTuner(config)
model, tokenizer = finetuner.setup_model()
model = finetuner.apply_lora(model)

# ... train ...

merged_path = finetuner.merge_and_save(output_dir="./outputs/merged")
```

---

## Troubleshooting

### OutOfMemoryError

1. Switch to QLoRA: `mode: "qlora"`
2. Enable gradient checkpointing: `gradient_checkpointing: true`
3. Reduce batch size: `per_device_train_batch_size: 1`
4. Increase gradient accumulation: `gradient_accumulation_steps: 16`
5. Reduce sequence length: `max_seq_length: 1024`
6. Reduce LoRA rank: `r: 8`

### Loss Not Decreasing

1. Check your data format matches the actual data structure
2. Increase learning rate by 2-5x
3. Verify the dataset is properly formatted with `llm-forge validate config.yaml`
4. Try a different LR scheduler (e.g., `cosine`)
5. Increase warmup ratio to `0.1`

### Training is Very Slow

1. Install flash attention: `pip install flash-attn`
2. Enable `group_by_length: true` to reduce padding waste
3. Use Unsloth: `use_unsloth: true`
4. Reduce logging frequency: `logging_steps: 50`
5. Use a larger batch size if VRAM allows

### BF16 Not Supported

llm-forge automatically falls back to FP16 on older GPUs (pre-Ampere). If you want to force FP16:

```yaml
model:
  torch_dtype: "fp16"
training:
  bf16: false
  fp16: true
```

### Flash Attention Not Available

llm-forge automatically falls back to SDPA (Scaled Dot Product Attention) when `flash-attn` is not installed. To install it:

```bash
pip install flash-attn --no-build-isolation
```

---

## Next Steps

- [Data Preparation Guide](data_preparation.md) -- prepare your training data
- [Evaluation Guide](evaluation_guide.md) -- benchmark your trained model
- [Distributed Training](distributed_training.md) -- scale across multiple GPUs
- [Deployment Guide](deployment.md) -- serve your trained model
