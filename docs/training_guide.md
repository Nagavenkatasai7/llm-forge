# Training Guide

Deep dive into all training modes supported by llm-forge, with practical lessons learned from building a production finance-specialist model across 7 iterations.

---

## Table of Contents

1. [Training Modes Overview](#training-modes-overview)
2. [Training Mode Decision Matrix](#training-mode-decision-matrix)
3. [LoRA Fine-Tuning](#lora-fine-tuning)
4. [QLoRA Fine-Tuning](#qlora-fine-tuning)
5. [Full Fine-Tuning](#full-fine-tuning)
6. [Pre-Training from Scratch](#pre-training-from-scratch)
7. [DPO Alignment](#dpo-alignment)
8. [Token Masking: completion_only_loss vs assistant_only_loss](#token-masking-completion_only_loss-vs-assistant_only_loss)
9. [Chat Templates and Generation Markers](#chat-templates-and-generation-markers)
10. [Avoiding Catastrophic Forgetting](#avoiding-catastrophic-forgetting)
11. [Training Callbacks](#training-callbacks)
12. [Monitoring Training: Interpreting Loss Curves](#monitoring-training-interpreting-loss-curves)
13. [Resuming from Checkpoints](#resuming-from-checkpoints)
14. [Lessons Learned: Finance Specialist Model (v1-v7)](#lessons-learned-finance-specialist-model-v1-v7)

---

## Training Modes Overview

| Mode | Config Value | Description | VRAM Requirement | Trainable Params |
|------|-------------|-------------|-----------------|------------------|
| LoRA | `lora` | Low-Rank Adaptation of attention/MLP layers | 16-24 GB | ~0.1-0.5% |
| QLoRA | `qlora` | LoRA with 4-bit quantized base model | 8-16 GB | ~0.1-0.5% |
| Full Fine-Tune | `full` | Update all model parameters | 40-80+ GB | 100% |
| Pre-Training | `pretrain` | Train a model from random initialization | Varies | 100% |
| DPO | `dpo` | Direct Preference Optimization alignment | 24-80 GB | ~0.1-0.5% (LoRA) |
| ORPO | `orpo` | Odds Ratio Preference Optimization | 24-80 GB | ~0.1-0.5% (LoRA) |
| GRPO | `grpo` | Group Relative Policy Optimization | 24-80 GB | ~0.1-0.5% (LoRA) |

---

## Training Mode Decision Matrix

Use this table to select the right training mode for your situation:

| Scenario | GPU VRAM | Dataset Size | Recommended Mode | Why |
|----------|---------|-------------|-----------------|-----|
| Quick domain adaptation | 8-16 GB | 1K-50K samples | QLoRA | Fits on consumer GPUs, good quality |
| Domain adaptation (data center) | 24-80 GB | 1K-50K samples | LoRA | Faster than QLoRA, slightly higher quality |
| Heavy domain shift | 40-80+ GB | 100K+ samples | Full fine-tune | Maximum expressiveness for large domain gaps |
| Build a new model | 40-80+ GB | 1M+ tokens | Pre-training | No suitable base model exists |
| Align to human preferences | 24-80 GB | 5K-50K pairs | DPO | No reward model needed, stable training |
| Instruction following | 16-24 GB | 5K-20K samples | LoRA (attention-only) | Preserves general knowledge, fast |
| Consumer GPU (RTX 3060/4060) | 8-12 GB | Any | QLoRA | Only viable option at this VRAM level |
| Apple Silicon (M1/M2/M3) | 16-64 GB unified | Any | LoRA | MPS backend, use BF16 on M2+, FP32 on M1 |

---

## LoRA Fine-Tuning

LoRA (Low-Rank Adaptation) freezes the base model weights and adds small trainable rank-decomposition matrices to selected layers. For a pre-trained weight matrix `W` of dimension `d x k`, LoRA adds:

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

| Rank (`r`) | Use Case | Trainable Params (7B model) | Memory Overhead |
|-----------|----------|---------------------------|-----------------|
| 8 | Quick experiments, simple tasks, knowledge preservation | ~1.7M | Minimal |
| 16 | General instruction following (default) | ~3.4M | Low |
| 32 | Complex domain adaptation | ~6.8M | Moderate |
| 64 | Near full fine-tune expressiveness | ~13.6M | Higher |
| 128+ | Maximum adapter capacity | ~27M+ | Significant |

**Key insight from v6/v7**: Higher rank does not always mean better results. The finance-specialist v6 used r=32, alpha=64 and suffered catastrophic forgetting (GSM8K -27.5%, IFEval -17.8%). Dropping to r=8, alpha=16 in v7 eliminated forgetting while preserving model quality.

### Target Module Selection

| Module Pattern | Layer Type | When to Include |
|---------------|-----------|-----------------|
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | Attention | Always -- core of LoRA. Use attention-only for knowledge preservation. |
| `gate_proj`, `up_proj`, `down_proj` | MLP / FFN | Include for better domain adaptation. Increases forgetting risk. |
| `embed_tokens` | Input embedding | Only for vocabulary expansion |
| `lm_head` | Output head | Only for vocabulary expansion |

**Attention-only vs all-linear**: When your goal is to add a skill (like domain-specific question answering) without destroying existing capabilities, target only the attention modules (`q_proj`, `v_proj`, `k_proj`, `o_proj`). Including MLP layers (`gate_proj`, `up_proj`, `down_proj`) gives more expressiveness but increases the risk of catastrophic forgetting.

### LoRA Alpha Tuning

The alpha parameter controls the effective learning rate of the LoRA adapter through the scaling factor `alpha/r`. Common patterns:

| Alpha:Rank Ratio | Behavior | When to Use |
|------------------|----------|-------------|
| 2:1 (e.g., alpha=32, r=16) | Standard scaling (default) | Most use cases |
| 1:1 (e.g., alpha=16, r=16) | Conservative update magnitude | Knowledge preservation |
| 4:1 (e.g., alpha=64, r=16) | Aggressive update magnitude | Heavy domain shift |

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
| Merge complexity | Simple merge_and_unload | Must reload base in FP16 for clean merge |

**Important QLoRA merge note**: When merging QLoRA adapters, the finetuner automatically reloads the base model in float16 on CPU and applies the adapter there. This avoids bitsandbytes `.absmax` artifacts in the merged checkpoint that would break GGUF conversion.

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

1. **Train a BPE tokenizer** from your text corpus (vocab_size=32,000)
2. **Build the model** from scratch using the Llama architecture
3. **Tokenize the dataset** and group into fixed-length blocks for causal language modelling
4. **Train** with linear warmup and cosine decay scheduling
5. **Save** the final model and tokenizer

All weights are initialized from a normal distribution with `std=0.02`.

---

## DPO Alignment

Direct Preference Optimization (DPO) aligns a model to human preferences without requiring a separate reward model. It trains directly on preference pairs (chosen vs. rejected responses).

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

The dataset must contain `prompt`, `chosen`, and `rejected` fields. With LoRA/PEFT, DPOTrainer shares weights between policy and reference models automatically (no separate reference model loaded).

---

## Token Masking: completion_only_loss vs assistant_only_loss

This is the single most important training setting for instruction tuning. Getting it wrong is the difference between a model that generates coherent responses and one that produces gibberish.

### The Problem

In a conversational dataset, each sample contains system prompts, user messages, and assistant responses. If the model trains on ALL tokens equally, it learns to predict system prompts and user messages -- leading to:

- System prompt regurgitation in output
- Loss values of 20-30 (expected: 1-3)
- Incoherent generation

### completion_only_loss

For prompt-completion datasets (flat text with a delimiter):

```yaml
training:
  completion_only_loss: true
```

This masks everything before the delimiter so the model only learns to predict the completion portion.

### assistant_only_loss (Recommended for Conversational Data)

For conversational datasets with a `messages` column (ShareGPT, ChatML format):

```yaml
training:
  assistant_only_loss: true
```

This uses TRL's chat template pipeline to create a binary mask over assistant turns. Non-assistant tokens (system prompts, user messages) get `label=-100`, so the model only learns to predict assistant responses.

**Expected masking percentages:**

| Masking % | Interpretation |
|-----------|---------------|
| <30% | Something is wrong -- model is training on system/user tokens |
| 30-50% | Acceptable for datasets with short system prompts |
| 50-70% | Good -- most non-assistant content is masked |
| 70-97% | Excellent -- only assistant tokens contribute to loss |

llm-forge logs the masking percentage before training starts:

```
PRE-TRAINING DIAGNOSTIC: Masked tokens: 4891/5012 (97.6%) -- expected >50% for conversational data
```

### Which to Use

| Dataset Format | Setting | Dataset Column |
|---------------|---------|----------------|
| Flat text with delimiter | `completion_only_loss: true` | `text` |
| ShareGPT / ChatML conversations | `assistant_only_loss: true` | `messages` |
| Alpaca (instruction/input/output) | `assistant_only_loss: true` | Preprocessed to `messages` |

---

## Chat Templates and Generation Markers

### Why Chat Templates Matter

TRL's `assistant_only_loss` requires the tokenizer to have a chat template with `{% generation %}` markers. These Jinja2 tags tell TRL exactly which tokens are assistant-generated and should be included in the loss calculation.

### The Generation Marker Problem

- **Base Llama 3.x models** have NO chat template at all
- **Instruct variants** have a template but WITHOUT `{% generation %}` markers
- TRL v0.20+ requires these markers for `assistant_only_loss` to work

### How llm-forge Handles This

The finetuner automatically injects the correct template when:
1. The model uses Llama 3 tokens (`<|start_header_id|>` in vocabulary)
2. `assistant_only_loss: true` is set
3. The existing template is missing or lacks generation markers

The injected template wraps assistant content with markers:

```jinja2
{% if message['role'] == 'assistant' %}
  {% generation %}{{ message['content'] | trim }}<|eot_id|>{% endgeneration %}
{% else %}
  {{ message['content'] | trim }}<|eot_id|>
{% endif %}
```

### Export Cleanup

The `{% generation %}` markers are HuggingFace-only extensions that are **not** valid Jinja2. They would break llama.cpp, Ollama, and vLLM. The merge-and-save step automatically strips them from the exported model.

---

## Avoiding Catastrophic Forgetting

Catastrophic forgetting is when fine-tuning destroys the model's pre-existing knowledge. This was the primary challenge in the finance-specialist project.

### v6 Results (Severe Forgetting)

| Benchmark | Base | v6 Fine-Tuned | Delta |
|-----------|------|---------------|-------|
| GSM8K | 33.59% | 6.07% | **-27.5%** |
| IFEval | 43.07% | 25.26% | **-17.8%** |
| MMLU | 46.05% | 38.59% | **-7.4%** |
| Business Ethics | 49% | 28% | **-21%** |

### v7 Results (Forgetting Eliminated)

| Benchmark | Base | v7 Fine-Tuned | Delta |
|-----------|------|---------------|-------|
| GSM8K | 33.59% | 31.99% | -1.60% |
| IFEval | 43.07% | 41.04% | -2.03% |
| MMLU | 46.05% | 45.86% | -0.19% |
| Business Ethics | 49% | 49% | 0.00% |

### The v7 Anti-Forgetting Recipe

Apply ALL of the following together -- each one alone is insufficient:

**1. Attention-only LoRA modules**

```yaml
lora:
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
  # Do NOT include gate_proj, up_proj, down_proj
```

**2. Low rank and conservative alpha**

```yaml
lora:
  r: 8           # Lower than typical r=16 or r=32
  alpha: 16      # 2:1 ratio, not 4:1
```

**3. Very low learning rate**

```yaml
training:
  learning_rate: 1.0e-5   # NOT 2e-4 (typical LoRA default)
```

**4. Single epoch**

```yaml
training:
  num_epochs: 1    # Multiple epochs on small data causes overfitting
```

**5. No NEFTune on small datasets**

```yaml
training:
  neftune_noise_alpha: null    # Disabled -- harmful on <20K samples
```

NEFTune adds noise to embeddings during training, which regularizes on large datasets but destabilizes training on small ones.

**6. Data cleaning**

```yaml
data:
  cleaning:
    enabled: true
    quality_preset: "permissive"
    heuristic_filter: true
    dedup_enabled: true
```

The v7 dataset was reduced from 20K to 5,675 samples after cleaning (72% removed as noise/duplicates). Cleaner data means the model learns signal rather than noise.

### The Trade-off

Conservative settings that eliminate forgetting may not add strong domain expertise. The v7 model preserved general knowledge perfectly but scored identically to the base model on finance benchmarks. For strong domain adaptation with minimal forgetting, consider:

- Larger, higher-quality domain datasets (50K+ clean samples)
- Moderate LoRA settings (r=16, alpha=32)
- Learning rate of 5e-5 (between conservative 1e-5 and aggressive 2e-4)
- 1-2 epochs with early stopping

---

## Training Callbacks

llm-forge includes several built-in callbacks that integrate with the HuggingFace Trainer callback system.

### Available Callbacks

| Callback | Trigger | Behavior |
|----------|---------|----------|
| `RichProgressCallback` | Rich library installed | Live progress bar with loss sparkline, LR, memory, throughput |
| `GPUMonitorCallback` | CUDA available | Logs GPU memory utilization every 50 steps to trainer logs and W&B |
| `WandBCallback` | `report_to: ["wandb"]` | Logs all metrics to Weights & Biases with model artifact upload |
| `EarlyStoppingCallback` | `eval_strategy != "no"` | Stops training after N evals with no improvement |
| `CheckpointCallback` | Always | Saves timed checkpoints every 30 minutes (configurable) |
| `StopTrainingCallback` | UI dashboard | Allows users to cancel training gracefully from the Gradio UI |
| `MacMonitorCallback` | macOS detected | Monitors thermal state, memory pressure, battery; auto-pauses if throttling |

### RichProgressCallback Details

The Rich progress bar displays real-time training information:

```
Training  [##########---------] 50%  500/1000  00:12:34  00:12:30
  loss: 1.4523↓ ▇▆▅▄▃▃▂▂ | lr: 1.95e-04 | ep: 0.50 | bs: 16 | 0.7 steps/s | mps:4.2G | ram:12/16G
```

Features include:
- Unicode sparkline showing loss trend over recent steps
- Trend indicator: `↓` (improving), `↑` (worsening), `→` (stable)
- GPU/MPS memory usage and system RAM
- macOS thermal state and battery level (when on battery)
- Summary panel at training end with loss curve visualization

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

The WandB callback logs:
- Training loss and learning rate at every logging step
- Evaluation metrics after each eval round
- GPU memory statistics (if GPUMonitorCallback is also active)
- Total and trainable parameter counts
- Final model as a W&B artifact (when `log_model=True`)

### EarlyStoppingCallback

```python
EarlyStoppingCallback(
    patience=3,          # Stop after 3 evals with no improvement
    min_delta=0.001,     # Minimum decrease to qualify as improvement
    metric_name="eval_loss",  # Metric to monitor
)
```

Requires `eval_strategy` to be set to `"steps"` or `"epoch"` in the training config.

### MacMonitorCallback

Specific to Apple Silicon training, this callback:
- Warns when memory pressure exceeds 85% and suggests smaller batch sizes
- Pauses training for 30 seconds when thermal throttling is detected
- Stops training when battery drops below 20% (when unplugged)

---

## Monitoring Training: Interpreting Loss Curves

### What Good Loss Values Look Like

| Model Size | Task | Expected Initial Loss | Expected Final Loss | Red Flag |
|-----------|------|----------------------|--------------------|---------|
| 1B | Instruction tuning (LoRA) | 1.5-2.5 | 0.8-1.5 | >5.0 initial, >2.5 final |
| 1B | Instruction tuning (full) | 1.5-2.5 | 0.5-1.2 | >5.0 initial, >2.0 final |
| 1B | Pre-training | 8-12 | 3-6 | Not decreasing after 1000 steps |
| 3B-7B | Instruction tuning (LoRA) | 1.0-2.0 | 0.5-1.2 | >3.0 initial |

### Loss Curve Shapes and Diagnosis

**Healthy loss curve:**
```
Loss
 3.0 |*
     | **
 2.0 |   ***
     |      ****
 1.5 |          ********
     |                  *********
 1.0 +----------------------------> Step
```
Steep initial drop, gradual convergence. Final loss stable.

**Loss not decreasing (flat line):**
- Learning rate too low -- increase by 5-10x
- Data format mismatch -- verify dataset columns match the configured format
- Token masking broken -- check the PRE-TRAINING DIAGNOSTIC log line

**Loss exploding (shooting up):**
- Learning rate too high -- decrease by 2-5x
- Data quality issue -- enable `data.cleaning.enabled: true`
- Gradient accumulation bug -- check `max_grad_norm: 1.0` is set

**Loss very high (20-30 range):**
- Model is training on ALL tokens (system prompts, user messages)
- Fix: enable `assistant_only_loss: true` or `completion_only_loss: true`
- This was root cause #1 in the finance-specialist v1 failure

**Loss oscillating wildly:**
- Batch size too small -- increase `gradient_accumulation_steps`
- Data has high variance -- enable cleaning and deduplication
- Learning rate too high for the effective batch size

### Train vs Eval Loss Divergence

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Train loss = Eval loss | Healthy training | Continue |
| Train loss << Eval loss (gap widening) | Overfitting | Reduce epochs, increase dropout, add regularization |
| Train loss >> Eval loss | Underfitting or data split issue | Increase LR or epochs |
| Eval loss spikes | Data quality issue in eval set | Check eval data for anomalies |

---

## Resuming from Checkpoints

Resume a training run from a saved checkpoint:

### Via YAML Config

```yaml
training:
  resume_from_checkpoint: "./outputs/my-lora/checkpoint-1000"
```

### Via CLI

```bash
llm-forge train --config config.yaml --resume ./outputs/my-lora/checkpoint-1000
```

Checkpoints are saved at:
- Every `save_steps` steps (configured in `training.save_steps`)
- Every 30 minutes by the `CheckpointCallback` (timed checkpoints)
- At the end of training

The `save_total_limit` setting controls how many checkpoints are retained (oldest are pruned automatically).

---

## Lessons Learned: Finance Specialist Model (v1-v7)

These are real production failures encountered while building a finance domain specialist using Llama 3.2 1B. Each root cause maps to a specific configuration fix in llm-forge.

### Root Cause #1: Training on all tokens (v1)

**Symptom**: Loss of 23-30 (expected 1-3). Output was gibberish with random dates and system prompt regurgitation.

**Cause**: `completion_only_loss: false` -- model trained on ALL tokens including system prompts and user messages.

**Fix**:
```yaml
training:
  completion_only_loss: true
  assistant_only_loss: true
```

### Root Cause #2: Flat text pipeline bypasses masking (v1)

**Symptom**: Even with completion_only_loss enabled, masking was ineffective.

**Cause**: The preprocessor output a flat `text` column ("Human:...Assistant:..."). TRL could not identify assistant boundaries.

**Fix**: Preprocessor now outputs a `messages` column in ChatML format. TRL's chat-template pipeline applies the model's template and creates proper masks.

### Root Cause #3: Hardcoded dataset_text_field (v1)

**Symptom**: TRL ignored the chat template pipeline.

**Cause**: `dataset_text_field="text"` was hardcoded in the finetuner, bypassing TRL's automatic `messages` column detection.

**Fix**: The finetuner now detects whether the dataset has `messages` or `text` columns and configures SFTConfig accordingly. When `messages` is present, `dataset_text_field` is NOT set.

### Root Cause #4: Aggressive hyperparameters (v1)

**Symptom**: Overfitting and catastrophic forgetting.

**Cause**: `learning_rate: 2e-4` too high for LoRA r=64 on 500K samples. `num_epochs: 3` excessive.

**Fix**: Conservative hyperparameters for domain adaptation:
```yaml
training:
  learning_rate: 1.0e-5
  num_epochs: 1
```

### Root Cause #5: pad_token = eos_token (v1)

**Symptom**: Model generated padding tokens in output.

**Cause**: Setting `pad_token = eos_token` taught the model to predict padding.

**Fix**: The finetuner now uses Llama 3's dedicated `<|finetune_right_pad_id|>` token when available, falling back to `eos_token` only on older architectures.

### Root Cause #6: Missing generation markers (v3)

**Symptom**: Token masking was only 39% (expected >50%). Special token leaks in output.

**Cause**: TRL v0.20's `assistant_only_loss` requires `{% generation %}...{% endgeneration %}` markers in the Jinja chat template. Without them, TRL could not create binary masks.

**Fix**: The finetuner automatically injects a Llama 3 chat template with generation markers when `assistant_only_loss: true` is set.

### Root Cause #7: Base model vs Instruct model (v4)

**Symptom**: Model could not follow instructions despite correct training configuration.

**Cause**: Fine-tuning a base model (Llama-3.2-1B) with only 5K samples is insufficient to teach instruction-following from scratch.

**Fix**: Start from an Instruct model:
```yaml
model:
  name: "unsloth/Llama-3.2-1B-Instruct"
```

### Root Causes #8-11: Inference configuration (v5)

**Symptoms**: Single-turn responses were good but multi-turn broke (topic repetition, could not answer meta-questions).

**Causes and fixes**:
- Temperature too high (0.3) -- set to 0.1
- Repeat penalty too aggressive (1.3) -- set to 1.1
- Modelfile used `.Prompt`/`.Response` (single-turn) -- switched to `range .Messages` loop
- `max_seq_length: 1024` too short for multi-turn -- increased to 2048
- Missing `num_ctx` in Modelfile -- added `num_ctx: 2048`

### Root Cause #12: Python executable in subprocess (v6)

**Symptom**: GGUF export failed on macOS.

**Cause**: `export.py` called `"python"` instead of `sys.executable` in subprocess calls. macOS does not have a `python` command by default.

**Fix**: All subprocess calls now use `sys.executable`.

### Root Cause #13: auto_optimize_config() overrides (v7)

**Symptom**: OOM on A100 80GB despite conservative config.

**Cause**: The `auto_optimize_config()` function detected A100 80GB and overrode batch_size to 16 and gradient_checkpointing to False, exceeding VRAM.

**Fix**: Added `--no-auto-optimize` CLI flag to disable automatic hardware optimization.

---

## Hyperparameter Quick Reference

### Learning Rate by Mode

| Training Mode | Recommended LR | Range |
|--------------|----------------|-------|
| LoRA (general) | `2e-4` | `1e-4` to `5e-4` |
| LoRA (knowledge preservation) | `1e-5` | `5e-6` to `5e-5` |
| QLoRA | `2e-4` | `1e-4` to `5e-4` |
| Full fine-tune | `2e-5` | `5e-6` to `5e-5` |
| Pre-training | `6e-4` | `3e-4` to `1e-3` |
| DPO | `5e-7` | `1e-7` to `5e-6` |

### Epochs by Dataset Size

| Dataset Size | Recommended Epochs |
|-------------|-------------------|
| < 1K samples | 5-10 |
| 1K - 10K | 1-3 |
| 10K - 100K | 1-2 |
| 100K+ | 1 |

### Effective Batch Size

The effective batch size is: `per_device_batch_size * gradient_accumulation_steps * num_gpus`

| Dataset Size | Effective Batch Size |
|-------------|---------------------|
| < 10K samples | 16-32 |
| 10K - 100K | 32-64 |
| 100K+ | 64-128 |

---

## Unsloth Acceleration

llm-forge optionally integrates with Unsloth for accelerated LoRA training:

```yaml
training:
  use_unsloth: true
```

Benefits:
- 2x faster training speed
- Up to 60% less memory usage
- Custom fused kernels for attention and MLP layers
- Automatic LoRA application with optimized gradient checkpointing

**Requirements:** `pip install unsloth`

---

## Merging LoRA Adapters

After LoRA/QLoRA training, merge adapter weights back into the base model:

```yaml
serving:
  merge_adapter: true
```

This creates a `merged/` subdirectory containing the full model. The merge step also:
- Strips `{% generation %}` markers from the chat template
- Handles QLoRA by reloading the base model in float16 for a clean merge
- Saves in safetensors format by default

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

1. Check token masking -- look for the PRE-TRAINING DIAGNOSTIC log line
2. Verify dataset format matches the configured `data.format`
3. Increase learning rate by 2-5x
4. Run `llm-forge validate config.yaml` to check configuration
5. Try a different LR scheduler (e.g., `cosine`)

### TRL Version Compatibility

llm-forge supports both TRL v0.20 and v0.29+. The key difference is `max_seq_length` (v0.20) vs `max_length` (v0.29) in SFTConfig. The finetuner uses introspection to detect the current version and sets the correct parameter automatically.

---

## Next Steps

- [Data Preparation Guide](data_preparation.md) -- prepare your training data
- [Evaluation Guide](evaluation_guide.md) -- benchmark your trained model
- [Distributed Training](distributed_training.md) -- scale across multiple GPUs
- [Deployment Guide](deployment.md) -- serve your trained model
