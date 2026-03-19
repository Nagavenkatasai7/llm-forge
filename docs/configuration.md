# Complete YAML Configuration Reference

llm-forge is entirely config-driven. A single YAML file controls the full pipeline: data cleaning, training, evaluation, and serving. This document describes every field, its type, default value, and valid range.

---

## Configuration Structure Overview

```yaml
model:          # Which model to load and how
lora:           # LoRA / QLoRA adapter settings
quantization:   # BitsAndBytes quantization options
data:           # Dataset paths, format, and cleaning
  cleaning:     # Data cleaning sub-configuration
training:       # Training hyperparameters
distributed:    # Multi-GPU / multi-node settings
evaluation:     # Post-training benchmark configuration
rag:            # Retrieval-Augmented Generation settings
serving:        # Serving backend and export options
```

Required top-level sections: `model` and `data` (specifically `model.name` and `data.train_path`). All other sections have sensible defaults.

---

## `model` -- Model Configuration

Controls which pretrained model to load and how it is configured.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `name` | `string` | **required** | Any HuggingFace model ID or local path | The model to load, e.g. `"meta-llama/Llama-3.2-1B"` |
| `revision` | `string` | `null` | Any git ref | Git revision (branch, tag, or commit SHA) of the model repo |
| `trust_remote_code` | `bool` | `false` | | Whether to execute code shipped inside the model repo |
| `torch_dtype` | `string` | `"bf16"` | `fp32`, `fp16`, `bf16`, `fp8`, `int8`, `int4` | Dtype for loading model weights |
| `max_seq_length` | `int` | `2048` | 128 -- 131072 | Maximum sequence length (context window) |
| `attn_implementation` | `string` | `"flash_attention_2"` | `eager`, `sdpa`, `flash_attention_2` | Attention kernel implementation |
| `rope_scaling` | `dict` | `null` | | RoPE scaling configuration, e.g. `{type: dynamic, factor: 2.0}` |

### Example

```yaml
model:
  name: "meta-llama/Llama-3.2-1B"
  revision: "main"
  trust_remote_code: false
  torch_dtype: "bf16"
  max_seq_length: 4096
  attn_implementation: "flash_attention_2"
  rope_scaling:
    type: "dynamic"
    factor: 2.0
```

**Tip:** llm-forge automatically falls back to `sdpa` if `flash_attention_2` is not installed, and to `fp16` if `bf16` is not supported by your GPU.

---

## `lora` -- LoRA Configuration

Low-Rank Adaptation hyperparameters. Used when `training.mode` is `lora`, `qlora`, or `dpo`.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `r` | `int` | `16` | 1 -- 256 | LoRA rank (number of low-rank dimensions) |
| `alpha` | `int` | `32` | Any positive int | Scaling factor. Effective scale = `alpha / r` |
| `dropout` | `float` | `0.05` | 0.0 -- 0.5 | Dropout applied to LoRA layers |
| `target_modules` | `list[string]` | `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` | Module name patterns | Which layers to apply LoRA to |
| `bias` | `string` | `"none"` | `none`, `all`, `lora_only` | Which biases to train alongside LoRA weights |
| `task_type` | `string` | `"CAUSAL_LM"` | `CAUSAL_LM`, `SEQ_2_SEQ_LM`, `TOKEN_CLS`, `SEQ_CLS` | PEFT task type |
| `use_rslora` | `bool` | `false` | | Enable Rank-Stabilized LoRA (RSLoRA) |
| `use_dora` | `bool` | `false` | | Enable Weight-Decomposed Low-Rank Adaptation (DoRA) |

### Example

```yaml
lora:
  r: 32
  alpha: 64
  dropout: 0.1
  target_modules:
    - q_proj
    - v_proj
  bias: "none"
  use_rslora: true
```

**Tip:** Higher `r` values capture more complex adaptations but use more memory. For domain adaptation, `r: 32` or `r: 64` often works well. The default `r: 16` is a good starting point for instruction tuning.

---

## `quantization` -- Quantization Configuration

BitsAndBytes quantization settings for memory-efficient loading. Automatically enabled when `training.mode` is `qlora`.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `load_in_4bit` | `bool` | `false` | | Load model in 4-bit precision |
| `load_in_8bit` | `bool` | `false` | | Load model in 8-bit precision |
| `bnb_4bit_compute_dtype` | `string` | `"bf16"` | `fp32`, `fp16`, `bf16` | Compute dtype for 4-bit quantized weights |
| `bnb_4bit_quant_type` | `string` | `"nf4"` | `nf4`, `fp4` | Quantization data type (NF4 is preferred) |
| `bnb_4bit_use_double_quant` | `bool` | `true` | | Nested/double quantization saves ~0.4 bits/param |

**Constraint:** `load_in_4bit` and `load_in_8bit` cannot both be `true`.

### Example

```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bf16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
```

---

## `data` -- Data Configuration

Dataset loading, formatting, and cleaning settings.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `train_path` | `string` | **required** | Local path or HuggingFace ID | Path to training data (e.g. `"data/train.jsonl"` or `"tatsu-lab/alpaca"`) |
| `eval_path` | `string` | `null` | Local path or HuggingFace ID | Evaluation data path. If omitted, splits from `train_path` |
| `format` | `string` | `"alpaca"` | `alpaca`, `sharegpt`, `completion`, `custom` | Dataset conversation format |
| `input_field` | `string` | `"instruction"` | Any column name | Column containing the user instruction |
| `output_field` | `string` | `"output"` | Any column name | Column containing the expected model output |
| `context_field` | `string` | `"input"` | Any column name or `null` | Column for optional context supplement |
| `system_prompt` | `string` | `null` | Any string | System prompt prepended to every sample |
| `max_samples` | `int` | `null` | >= 1 | Cap the number of training samples (useful for debugging) |
| `test_size` | `float` | `0.05` | 0.0 -- 1.0 (exclusive) | Fraction held out for evaluation when `eval_path` is null |
| `seed` | `int` | `42` | Any integer | Random seed for reproducible splitting/shuffling |
| `streaming` | `bool` | `false` | | Stream data instead of loading into RAM |
| `num_workers` | `int` | `4` | >= 0 | Number of data-loader worker processes |
| `cleaning` | `object` | See below | | Data-cleaning sub-configuration |

### Data Format Reference

| Format | Description | Required Columns |
|--------|-------------|-----------------|
| `alpaca` | Instruction-response format | `instruction`, `output`, optionally `input` |
| `sharegpt` | Multi-turn conversation format | `conversations` (list of `{from, value}` dicts) |
| `completion` | Plain text completion | `text` |
| `custom` | Custom field mapping | Uses `input_field` and `output_field` settings |

### Example

```yaml
data:
  train_path: "tatsu-lab/alpaca"
  eval_path: null
  format: "alpaca"
  input_field: "instruction"
  output_field: "output"
  context_field: "input"
  system_prompt: "You are a helpful assistant."
  max_samples: 10000
  test_size: 0.05
  seed: 42
  streaming: false
  num_workers: 4
```

---

## `data.cleaning` -- Data Cleaning Configuration

Controls the 7-stage data cleaning pipeline. See [Data Preparation Guide](data_preparation.md) for detailed usage.

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `enabled` | `bool` | `true` | | Master switch for data cleaning |
| `quality_preset` | `string` | `"balanced"` | `permissive`, `balanced`, `strict` | Overall quality-filtering strictness |
| **Unicode** | | | | |
| `unicode_fix` | `bool` | `true` | | Apply ftfy unicode fixing (mojibake repair) |
| **Language** | | | | |
| `language_filter` | `list[string]` | `null` | ISO-639 codes | Keep only texts in these languages, e.g. `["en"]` |
| `language_confidence_threshold` | `float` | `0.65` | 0.0 -- 1.0 | Minimum fasttext confidence for language detection |
| **Heuristic Filtering** | | | | |
| `heuristic_filter` | `bool` | `true` | | Enable rule-based quality heuristics |
| `min_word_count` | `int` | `5` | >= 0 | Minimum words per sample |
| `max_word_count` | `int` | `100000` | >= 1 | Maximum words per sample |
| `min_char_count` | `int` | `20` | >= 0 | Minimum characters per sample |
| `max_char_count` | `int` | `5000000` | >= 1 | Maximum characters per sample |
| `alpha_ratio_threshold` | `float` | `0.6` | 0.0 -- 1.0 | Minimum fraction of alphabetic characters |
| `symbol_to_word_ratio` | `float` | `0.1` | 0.0 -- 1.0 | Maximum ratio of symbols to total words |
| `max_duplicate_line_fraction` | `float` | `0.3` | 0.0 -- 1.0 | Maximum fraction of duplicate lines within one sample |
| `max_duplicate_para_fraction` | `float` | `0.3` | 0.0 -- 1.0 | Maximum fraction of duplicate paragraphs within one sample |
| **Toxicity** | | | | |
| `toxicity_filter` | `bool` | `false` | | Enable toxicity scoring (requires `detoxify`) |
| `toxicity_threshold` | `float` | `0.8` | 0.0 -- 1.0 | Drop samples above this toxicity score |
| **PII** | | | | |
| `pii_redaction` | `bool` | `false` | | Enable PII detection/redaction (requires `presidio`) |
| `pii_entities` | `list[string]` | `[PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN, IP_ADDRESS]` | Entity types | Named entity types to redact |
| **Deduplication** | | | | |
| `dedup_enabled` | `bool` | `true` | | Enable cross-document deduplication |
| `dedup_tiers` | `list[string]` | `["exact", "fuzzy"]` | `exact`, `fuzzy`, `semantic` | Deduplication strategies to apply in order |
| `dedup_jaccard_threshold` | `float` | `0.85` | 0.0 -- 1.0 | Jaccard similarity threshold for fuzzy dedup |
| `dedup_num_perm` | `int` | `128` | >= 16 | Number of permutations for MinHash |
| `dedup_shingle_size` | `int` | `5` | >= 1 | Shingle (n-gram) size for MinHash |
| `semantic_dedup_enabled` | `bool` | `false` | | Enable embedding-based semantic deduplication |
| `semantic_dedup_threshold` | `float` | `0.95` | 0.0 -- 1.0 | Cosine similarity threshold for semantic dedup |
| `semantic_dedup_model` | `string` | `"sentence-transformers/all-MiniLM-L6-v2"` | Any sentence-transformers model | Embedding model for semantic dedup |

---

## `training` -- Training Configuration

Core training hyperparameters. Maps closely to HuggingFace `TrainingArguments`.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `mode` | `string` | `"lora"` | `lora`, `qlora`, `full`, `pretrain`, `dpo` | Training strategy |
| `output_dir` | `string` | `"outputs"` | Any path | Directory for checkpoints, logs, and final artefacts |
| `num_epochs` | `int` | `3` | >= 1 | Total number of training epochs |
| `per_device_train_batch_size` | `int` | `4` | >= 1 | Micro-batch size per GPU for training |
| `per_device_eval_batch_size` | `int` | `4` | >= 1 | Micro-batch size per GPU for evaluation |
| `gradient_accumulation_steps` | `int` | `4` | >= 1 | Micro-batches accumulated before a gradient step |
| `learning_rate` | `float` | `2e-4` | > 0.0 | Peak learning rate |
| `weight_decay` | `float` | `0.01` | >= 0.0 | L2 weight decay coefficient |
| `warmup_ratio` | `float` | `0.03` | 0.0 -- 1.0 | Fraction of total steps for linear warmup |
| `warmup_steps` | `int` | `null` | >= 0 | Exact warmup steps (overrides `warmup_ratio`) |
| `lr_scheduler_type` | `string` | `"cosine"` | `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`, `inverse_sqrt`, `reduce_on_plateau` | LR scheduler type |
| `max_grad_norm` | `float` | `1.0` | >= 0.0 | Maximum gradient norm for clipping |
| `logging_steps` | `int` | `10` | >= 1 | Log metrics every N steps |
| `eval_steps` | `int` | `null` | >= 1 | Run evaluation every N steps. `null` = every epoch |
| `eval_strategy` | `string` | `"epoch"` | `no`, `steps`, `epoch` | When to run evaluation |
| `save_steps` | `int` | `500` | >= 1 | Save a checkpoint every N steps |
| `save_total_limit` | `int` | `3` | >= 1 | Maximum number of checkpoints to keep |
| `bf16` | `bool` | `true` | | Enable bfloat16 mixed precision |
| `fp16` | `bool` | `false` | | Enable float16 mixed precision |
| `gradient_checkpointing` | `bool` | `false` | | Trade compute for memory by re-computing activations |
| `optim` | `string` | `"adamw_torch"` | `adamw_torch`, `adamw_8bit`, `paged_adamw_32bit`, `paged_adamw_8bit`, etc. | Optimizer name |
| `group_by_length` | `bool` | `true` | | Group similar-length samples to reduce padding |
| `report_to` | `list[string]` | `["wandb"]` | `wandb`, `tensorboard`, `none` | Experiment trackers |
| `resume_from_checkpoint` | `string` | `null` | Path to checkpoint dir | Resume training from a checkpoint |
| `neftune_noise_alpha` | `float` | `5.0` | >= 0.0 or null | NEFTune noise alpha for regularisation. Set to `null` to disable. |
| `label_smoothing_factor` | `float` | `0.1` | 0.0 – 1.0 | Label-smoothing coefficient. Set to `0.0` to disable. |
| `average_tokens_across_devices` | `bool` | `true` | | Sync token counts across GPUs for correct grad-accum loss scaling. |
| `use_unsloth` | `bool` | `false` | | Use Unsloth accelerated kernels when available |

**Constraint:** `bf16` and `fp16` cannot both be `true`.

### Effective Batch Size

The effective (global) batch size is:

```
effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
```

---

## `distributed` -- Distributed Training Configuration

Multi-GPU and multi-node training settings.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `enabled` | `bool` | `false` | | Enable distributed training |
| `framework` | `string` | `"auto"` | `auto`, `fsdp`, `deepspeed`, `megatron` | Distributed framework |
| `num_gpus` | `int` | `1` | >= 1 | Number of GPUs |
| `num_nodes` | `int` | `1` | >= 1 | Number of cluster nodes |
| `fsdp_sharding_strategy` | `string` | `"FULL_SHARD"` | `FULL_SHARD`, `SHARD_GRAD_OP`, `NO_SHARD`, `HYBRID_SHARD` | FSDP sharding strategy |
| `deepspeed_stage` | `int` | `2` | 0, 1, 2, 3 | DeepSpeed ZeRO stage |
| `deepspeed_offload` | `bool` | `false` | | Offload optimizer/parameters to CPU |
| `tensor_parallel_degree` | `int` | `1` | >= 1 | Tensor-parallelism degree (Megatron-LM) |
| `pipeline_parallel_degree` | `int` | `1` | >= 1 | Pipeline-parallelism degree (Megatron-LM) |
| `fp8_enabled` | `bool` | `false` | | Enable FP8 compute via Transformer Engine |
| `fp8_format` | `string` | `"HYBRID"` | `E4M3`, `HYBRID` | FP8 format (HYBRID = E4M3 forward, E5M2 backward) |
| `auto_micro_batch` | `bool` | `false` | | Automatically find the largest micro-batch that fits |

---

## `evaluation` -- Evaluation Configuration

Post-training benchmark and evaluation settings.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `enabled` | `bool` | `true` | | Run benchmarks after training |
| `benchmarks` | `list[string]` | `["hellaswag", "arc_easy", "mmlu"]` | See benchmark list below | lm-eval benchmark task names |
| `custom_eval_path` | `string` | `null` | Path to JSONL/JSON | Path to a custom evaluation dataset |
| `num_fewshot` | `int` | `0` | >= 0 | Number of few-shot examples |
| `batch_size` | `int` | `8` | >= 1 | Batch size for evaluation inference |
| `generate_report` | `bool` | `true` | | Generate an HTML/Markdown evaluation report |

### Supported Benchmarks

| Benchmark | Description | Default Few-shot | Metric |
|-----------|-------------|-----------------|--------|
| `mmlu` | Massive Multitask Language Understanding | 5 | `acc` |
| `hellaswag` | Commonsense NLI | 10 | `acc_norm` |
| `arc_easy` | AI2 Reasoning Challenge (Easy) | 25 | `acc_norm` |
| `arc_challenge` | AI2 Reasoning Challenge (Challenge) | 25 | `acc_norm` |
| `winogrande` | Winograd Schema Challenge | 5 | `acc` |
| `truthfulqa_mc2` | TruthfulQA Multiple Choice | 0 | `acc` |
| `gsm8k` | Grade School Math | 5 | `exact_match` |

---

## `rag` -- RAG Configuration

Retrieval-Augmented Generation pipeline settings.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `enabled` | `bool` | `false` | | Enable the RAG pipeline |
| `knowledge_base_path` | `string` | `null` | Directory path | Path to knowledge-base documents |
| `chunk_strategy` | `string` | `"recursive"` | `fixed`, `recursive`, `semantic`, `sentence` | Text chunking strategy |
| `chunk_size` | `int` | `512` | 64 -- 8192 | Target chunk size in tokens |
| `chunk_overlap` | `int` | `64` | >= 0 | Overlap between consecutive chunks |
| `embedding_model` | `string` | `"sentence-transformers/all-MiniLM-L6-v2"` | Any sentence-transformers model | Embedding model for vectorisation |
| `vectorstore` | `string` | `"chromadb"` | `chromadb`, `faiss`, `qdrant`, `weaviate` | Vector store backend |
| `top_k` | `int` | `5` | >= 1 | Number of chunks to retrieve per query |
| `reranker_model` | `string` | `null` | Cross-encoder model | Model for re-ranking retrieved chunks |
| `hybrid_search` | `bool` | `false` | | Combine dense retrieval with BM25 sparse retrieval |
| `similarity_threshold` | `float` | `0.7` | 0.0 -- 1.0 | Minimum similarity score to keep a retrieved chunk |

**Constraint:** `chunk_overlap` must be less than `chunk_size`.

---

## `serving` -- Serving Configuration

Model serving and export settings.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `backend` | `string` | `"gradio"` | `gradio`, `fastapi`, `vllm` | Serving backend |
| `host` | `string` | `"0.0.0.0"` | Any valid hostname/IP | Host to bind the server to |
| `port` | `int` | `7860` | 1 -- 65535 | Port number |
| `export_format` | `string` | `null` | `gguf`, `onnx`, `safetensors`, `awq`, `gptq` | Export format after training |
| `gguf_quantization` | `string` | `null` | e.g. `Q4_K_M`, `Q5_K_S` | GGUF quantization level |
| `merge_adapter` | `bool` | `true` | | Merge LoRA adapter into base model before serving |

---

## Common Configuration Patterns

### Minimal LoRA fine-tuning

```yaml
model:
  name: "meta-llama/Llama-3.2-1B"

data:
  train_path: "tatsu-lab/alpaca"

training:
  mode: "lora"
```

### QLoRA on a memory-constrained GPU

```yaml
model:
  name: "meta-llama/Llama-3.2-1B"
  torch_dtype: "bf16"

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true

data:
  train_path: "./data/train.jsonl"

training:
  mode: "qlora"
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  optim: "paged_adamw_8bit"
```

### Domain-specific fine-tuning with strict cleaning

```yaml
model:
  name: "meta-llama/Llama-3.2-1B"
  max_seq_length: 4096

lora:
  r: 32
  alpha: 64

data:
  train_path: "./data/medical_train.jsonl"
  system_prompt: "You are a medical knowledge assistant."
  cleaning:
    quality_preset: "strict"
    language_filter: ["en"]
    pii_redaction: true
    toxicity_filter: false

training:
  mode: "lora"
  learning_rate: 1.0e-4
  num_epochs: 5
```

### Multi-GPU training with DeepSpeed

```yaml
model:
  name: "meta-llama/Llama-3.1-8B"

data:
  train_path: "tatsu-lab/alpaca"

training:
  mode: "lora"
  per_device_train_batch_size: 8

distributed:
  enabled: true
  framework: "deepspeed"
  num_gpus: 4
  deepspeed_stage: 2
```

---

## Built-in Presets

llm-forge ships with five built-in presets used by `llm-forge init --template`:

| Preset | Mode | Model | Description |
|--------|------|-------|-------------|
| `lora_default` | LoRA | Llama-3.2-1B | Standard LoRA on consumer GPUs (>= 16 GB) |
| `qlora_default` | QLoRA | Llama-3.2-1B | Memory-efficient 4-bit LoRA (>= 8 GB) |
| `full_finetune` | Full | SmolLM2-135M | All-parameter fine-tuning of a small model |
| `pretrain_small` | Pretrain | GPT-2 scaffold | Train a 125M model from scratch |
| `rag_default` | RAG | Llama-3.2-1B | RAG pipeline with ChromaDB |

Load a preset programmatically:

```python
from llm_forge.config.validator import load_preset

config = load_preset("lora_default")
```

---

## Validation

llm-forge validates your configuration before training and provides actionable error messages with suggestions:

```bash
llm-forge validate config.yaml
```

Common validation messages:

| Error | Suggestion |
|-------|-----------|
| Unknown field `model_name` | Did you mean `model.name`? |
| Unknown field `epochs` | Did you mean `training.num_epochs`? |
| Unknown field `lr` | Did you mean `training.learning_rate`? |
| Unknown field `batch_size` | Did you mean `training.per_device_train_batch_size`? |
| Both `bf16` and `fp16` are `true` | Cannot enable both simultaneously |
| Both `load_in_4bit` and `load_in_8bit` are `true` | Cannot enable both simultaneously |
