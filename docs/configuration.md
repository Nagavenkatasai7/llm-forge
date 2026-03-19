# Complete YAML Configuration Reference

llm-forge is entirely config-driven. A single YAML file controls the full pipeline: data loading, cleaning, preprocessing, training, evaluation, export, and serving. This document describes every field, its type, default value, and valid range.

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
serving:        # Serving backend, export, and inference params
mac:            # Apple Silicon training optimisations
iti:            # Inference-Time Intervention (anti-hallucination)
refusal:        # Refusal-aware training (R-Tuning)
ifd:            # IFD data scoring and filtering
alignment:      # Preference-based alignment (DPO/ORPO/GRPO)
merge:          # Model merging (TIES/SLERP/linear)
mlx:            # MLX-based training on Apple Silicon
compute:        # Compute backend (local, SLURM, cloud)
```

**Required top-level sections:** `model` and `data` (specifically `model.name` and `data.train_path`). All other sections have sensible defaults and can be omitted entirely.

---

## `model` -- Model Configuration

Controls which pretrained model to load and how it is configured.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `name` | `string` | **required** | Any HuggingFace model ID or local path | The model to load, e.g. `"meta-llama/Llama-3.2-1B"` |
| `revision` | `string` | `null` | Any git ref | Git revision (branch, tag, or commit SHA) of the model repo |
| `trust_remote_code` | `bool` | `false` | | Whether to execute code shipped inside the model repo |
| `torch_dtype` | `string` | `"bf16"` | `fp32`, `fp16`, `bf16`, `fp8`, `int8`, `int4` | Dtype for loading model weights |
| `max_seq_length` | `int` | `2048` | 128 -- 131072 | Maximum sequence length (context window) for training |
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

**Notes:**
- Use `sdpa` if `flash_attention_2` is not installed. SDPA works on all platforms including Apple Silicon.
- `max_seq_length` directly affects VRAM usage. Each doubling roughly doubles activation memory. For multi-turn conversations, 2048 supports 4-6 turns.

---

## `lora` -- LoRA Configuration

Low-Rank Adaptation hyperparameters. Used when `training.mode` is `lora`, `qlora`, `dpo`, `orpo`, or `grpo`.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `r` | `int` | `16` | 1 -- 256 | LoRA rank (number of low-rank dimensions) |
| `alpha` | `int` | `32` | Any positive int | Scaling factor. Effective scale = `alpha / r` |
| `dropout` | `float` | `0.05` | 0.0 -- 0.5 | Dropout applied to LoRA layers |
| `target_modules` | `list[string]` or `string` | `["q_proj", "v_proj", "k_proj", "o_proj"]` | Module name patterns or `"all-linear"` | Which layers to apply LoRA to |
| `bias` | `string` | `"none"` | `none`, `all`, `lora_only` | Which biases to train alongside LoRA weights |
| `task_type` | `string` | `"CAUSAL_LM"` | `CAUSAL_LM`, `SEQ_2_SEQ_LM`, `TOKEN_CLS`, `SEQ_CLS` | PEFT task type |
| `use_rslora` | `bool` | `false` | | Enable Rank-Stabilized LoRA (RSLoRA) -- scales by `alpha / sqrt(r)` |
| `use_dora` | `bool` | `false` | | Enable Weight-Decomposed Low-Rank Adaptation (DoRA) |

### Example

```yaml
lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
  bias: "none"
  use_rslora: false
  use_dora: false
```

**Guidance on `target_modules`:**
- **Attention-only** (`["q_proj", "v_proj", "k_proj", "o_proj"]`): Safest for knowledge preservation. Best when you want to add domain knowledge without forgetting general capabilities.
- **All-linear** (`"all-linear"`): Includes MLP layers (`gate_proj`, `up_proj`, `down_proj`, `lm_head`). Maximum adaptability but higher risk of catastrophic forgetting on small datasets (<20K samples).
- **Explicit list with MLP**: A middle ground, e.g. `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.

**Guidance on rank:**
- `r: 8` -- Conservative, preserves base model knowledge (recommended for instruction tuning on Instruct models)
- `r: 16` -- Good default for general fine-tuning
- `r: 32-64` -- High capacity, useful for domain adaptation with large datasets

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

**Auto-configuration:** When `training.mode` is `qlora` and no quantization settings are specified, llm-forge automatically enables 4-bit quantization with NF4 and double quantization.

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
| `train_path` | `string` | **required** | Local path, directory, URL, or HuggingFace ID | Path to training data |
| `eval_path` | `string` | `null` | Same as `train_path` | Evaluation data path. If omitted, splits from `train_path` |
| `format` | `string` | `"alpaca"` | `alpaca`, `sharegpt`, `completion`, `custom` | Dataset conversation format |
| `input_field` | `string` | `"instruction"` | Any column name | Column containing the user instruction |
| `output_field` | `string` | `"output"` | Any column name | Column containing the expected model output |
| `context_field` | `string` | `"input"` | Any column name or `null` | Column for optional context supplement |
| `system_prompt` | `string` | `null` | Any string | System prompt prepended to every sample |
| `max_samples` | `int` | `null` | >= 1 | Cap the number of training samples |
| `test_size` | `float` | `0.05` | 0.0 -- 1.0 (exclusive) | Fraction held out for evaluation when `eval_path` is null |
| `seed` | `int` | `42` | Any integer | Random seed for reproducible splitting/shuffling |
| `streaming` | `bool` | `false` | | Stream data instead of loading into RAM |
| `num_workers` | `int` | `4` | >= 0 | Number of data-loader worker processes |
| `cleaning` | `object` | See below | | Data-cleaning sub-configuration |

### Data Format Reference

| Format | Description | Required Columns |
|--------|-------------|-----------------|
| `alpaca` | Instruction-response format | `instruction`, `output`, optionally `input` |
| `sharegpt` | Multi-turn conversation format | `conversations` (list of role/value dicts), or `messages`, or flat `user`/`assistant` columns |
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
  cleaning:
    enabled: true
    quality_preset: "balanced"
```

---

## `data.cleaning` -- Data Cleaning Configuration

Controls the 7-stage data cleaning pipeline. See [Data Preparation Guide](data_preparation.md) for detailed usage.

### Unicode and Language

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `enabled` | `bool` | `true` | | Master switch for data cleaning |
| `quality_preset` | `string` | `"balanced"` | `permissive`, `balanced`, `strict` | Overall quality-filtering strictness |
| `unicode_fix` | `bool` | `true` | | Apply ftfy unicode fixing (mojibake repair) |
| `language_filter` | `list[string]` | `null` | ISO-639 codes | Keep only texts in these languages, e.g. `["en"]` |
| `language_confidence_threshold` | `float` | `0.65` | 0.0 -- 1.0 | Minimum fasttext confidence for language detection |

### Heuristic Filtering

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `heuristic_filter` | `bool` | `true` | | Enable rule-based quality heuristics |
| `min_word_count` | `int` | `5` | >= 0 | Minimum words per sample |
| `max_word_count` | `int` | `100000` | >= 1 | Maximum words per sample |
| `min_char_count` | `int` | `20` | >= 0 | Minimum characters per sample |
| `max_char_count` | `int` | `5000000` | >= 1 | Maximum characters per sample |
| `alpha_ratio_threshold` | `float` | `0.6` | 0.0 -- 1.0 | Minimum fraction of alphabetic characters |
| `symbol_to_word_ratio` | `float` | `0.1` | 0.0 -- 1.0 | Maximum ratio of symbols to total words |
| `max_duplicate_line_fraction` | `float` | `0.3` | 0.0 -- 1.0 | Maximum fraction of duplicate lines within one sample |
| `max_duplicate_para_fraction` | `float` | `0.3` | 0.0 -- 1.0 | Maximum fraction of duplicate paragraphs within one sample |

### Deduplication

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `dedup_enabled` | `bool` | `true` | | Enable cross-document deduplication |
| `dedup_tiers` | `list[string]` | `["exact", "fuzzy"]` | `exact`, `fuzzy`, `semantic` | Deduplication strategies to apply in order |
| `dedup_jaccard_threshold` | `float` | `0.85` | 0.0 -- 1.0 | Jaccard similarity threshold for fuzzy dedup |
| `dedup_num_perm` | `int` | `128` | >= 16 | Number of permutations for MinHash |
| `dedup_shingle_size` | `int` | `5` | >= 1 | Shingle (n-gram) size for MinHash |
| `semantic_dedup_enabled` | `bool` | `false` | | Enable embedding-based semantic deduplication |
| `semantic_dedup_threshold` | `float` | `0.95` | 0.0 -- 1.0 | Cosine similarity threshold for semantic dedup |
| `semantic_dedup_model` | `string` | `"sentence-transformers/all-MiniLM-L6-v2"` | Any sentence-transformers model | Embedding model for semantic dedup |

### PII and Toxicity

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `pii_redaction` | `bool` | `false` | | Enable PII detection/redaction (requires `presidio`) |
| `pii_entities` | `list[string]` | `[PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN, IP_ADDRESS]` | Entity types | Named entity types to redact |
| `toxicity_filter` | `bool` | `false` | | Enable toxicity scoring (requires `detoxify`) |
| `toxicity_threshold` | `float` | `0.8` | 0.0 -- 1.0 | Drop samples above this toxicity score |

### Example

```yaml
data:
  cleaning:
    enabled: true
    quality_preset: "strict"
    unicode_fix: true
    language_filter: ["en"]
    language_confidence_threshold: 0.65
    heuristic_filter: true
    min_word_count: 10
    dedup_enabled: true
    dedup_tiers:
      - exact
      - fuzzy
    dedup_jaccard_threshold: 0.85
    pii_redaction: true
    pii_entities:
      - PERSON
      - EMAIL_ADDRESS
      - PHONE_NUMBER
    toxicity_filter: false
```

---

## `training` -- Training Configuration

Core training hyperparameters. Maps closely to HuggingFace `TrainingArguments` and TRL `SFTConfig`.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `mode` | `string` | `"lora"` | `lora`, `qlora`, `full`, `pretrain`, `dpo`, `orpo`, `grpo` | Training strategy |
| `output_dir` | `string` | `"outputs"` | Any path (supports `~` and `$HOME`) | Directory for checkpoints, logs, and final artefacts |
| `num_epochs` | `int` | `1` | >= 1 | Total number of training epochs |
| `per_device_train_batch_size` | `int` | `4` | >= 1 | Micro-batch size per GPU for training |
| `per_device_eval_batch_size` | `int` | `4` | >= 1 | Micro-batch size per GPU for evaluation |
| `gradient_accumulation_steps` | `int` | `4` | >= 1 | Micro-batches accumulated before a gradient step |
| `learning_rate` | `float` | `2e-5` | > 0.0 | Peak learning rate |
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
| `neftune_noise_alpha` | `float` | `null` | >= 0.0 or null | NEFTune noise alpha. `null` disables. Use 5.0 for datasets >50K samples. |
| `label_smoothing_factor` | `float` | `0.0` | 0.0 -- 1.0 | Label-smoothing coefficient. 0.0 disables. |
| `completion_only_loss` | `bool` | `true` | | Compute loss on completion tokens only (mask prompt tokens) |
| `assistant_only_loss` | `bool` | `true` | | Compute loss on assistant response tokens only (for chat-template datasets). Critical for instruction tuning. |
| `average_tokens_across_devices` | `bool` | `true` | | Sync token counts across GPUs for correct grad-accum loss scaling |
| `use_unsloth` | `bool` | `false` | | Use Unsloth accelerated kernels when available |
| `pack_sequences` | `bool` | `false` | | Pack multiple short sequences into max_seq_length to reduce padding. 2-4x throughput improvement. Not recommended for chat data. |

**Constraints:**
- `bf16` and `fp16` cannot both be `true`.
- `eval_steps` requires `eval_strategy: "steps"` (setting `eval_steps` with `eval_strategy: "no"` raises an error).
- `learning_rate` above `1e-3` triggers a warning about catastrophic forgetting risk.

### Effective Batch Size

```
effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
```

### Example

```yaml
training:
  mode: "lora"
  output_dir: "./outputs/my-run/"
  num_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  bf16: true
  gradient_checkpointing: true
  logging_steps: 10
  eval_strategy: "steps"
  eval_steps: 50
  save_steps: 100
  save_total_limit: 2
  completion_only_loss: true
  assistant_only_loss: true
  neftune_noise_alpha: null
  label_smoothing_factor: 0.0
  pack_sequences: false
  report_to:
    - "none"
```

---

## `evaluation` -- Evaluation Configuration

Post-training benchmark and evaluation settings. Uses the `lm-eval-harness` library.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `enabled` | `bool` | `true` | | Run benchmarks after training |
| `benchmarks` | `list[string]` | `["hellaswag", "arc_easy", "mmlu", "truthfulqa_mc2", "ifeval"]` | See list below | lm-eval benchmark task names |
| `custom_eval_path` | `string` | `null` | Path to custom eval script/dataset | Custom evaluation path |
| `num_fewshot` | `int` | `0` | >= 0 | Number of few-shot examples for benchmarks |
| `batch_size` | `int` | `8` | >= 1 | Batch size for evaluation inference |
| `generate_report` | `bool` | `true` | | Generate an HTML/Markdown evaluation report |
| `regression_check` | `bool` | `true` | | Compare against base model and warn on degradation |
| `regression_threshold` | `float` | `-0.02` | <= 0.0 | Max acceptable per-benchmark score drop |
| `llm_judge` | `bool` | `false` | | Run LLM-as-Judge evaluation |
| `judge_model` | `string` | `null` | Model path or HuggingFace ID | Judge model. `null` = use the trained model |
| `judge_criteria` | `list[string]` | `["helpfulness", "coherence"]` | | Criteria for LLM-as-Judge |
| `judge_samples` | `int` | `50` | >= 1 | Number of samples for LLM-as-Judge |
| `retention_probes` | `bool` | `false` | | Run 100 factual questions to detect catastrophic forgetting |
| `retention_threshold` | `float` | `0.80` | 0.0 -- 1.0 | Minimum acceptable retention rate |

### Supported Benchmarks

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| `mmlu` | Massive Multitask Language Understanding (57 subjects) | `acc` |
| `hellaswag` | Commonsense NLI | `acc_norm` |
| `arc_easy` | AI2 Reasoning Challenge (Easy) | `acc_norm` |
| `arc_challenge` | AI2 Reasoning Challenge (Challenge) | `acc_norm` |
| `winogrande` | Winograd Schema Challenge | `acc` |
| `truthfulqa_mc2` | TruthfulQA Multiple Choice | `acc` |
| `gsm8k` | Grade School Math | `exact_match` |
| `ifeval` | Instruction Following Eval | `prompt_strict_acc` |

### Example

```yaml
evaluation:
  enabled: true
  benchmarks:
    - hellaswag
    - arc_easy
    - mmlu
    - gsm8k
  num_fewshot: 0
  batch_size: 8
  generate_report: true
  regression_check: true
  regression_threshold: -0.02
  retention_probes: true
  retention_threshold: 0.80
```

---

## `serving` -- Serving Configuration

Model serving backend, export settings, and inference parameters.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `backend` | `string` | `"gradio"` | `gradio`, `fastapi`, `vllm` | Serving backend |
| `host` | `string` | `"0.0.0.0"` | Any valid hostname/IP | Host to bind the server to |
| `port` | `int` | `7860` | 1 -- 65535 | Port number |
| `export_format` | `string` | `null` | `gguf`, `onnx`, `safetensors`, `awq`, `gptq` | Export format after training |
| `gguf_quantization` | `string` | `null` | e.g. `Q4_K_M`, `Q5_K_S`, `Q8_0` | GGUF quantization level |
| `merge_adapter` | `bool` | `true` | | Merge LoRA adapter into base model before serving/export |
| `generate_modelfile` | `bool` | `true` | | Auto-generate an Ollama Modelfile alongside GGUF export |
| `ollama_system_prompt` | `string` | `null` | | System prompt for the Modelfile. If null, uses `data.system_prompt` |
| `inference_temperature` | `float` | `0.1` | 0.0 -- 2.0 | Sampling temperature. Small models (<3B) need low values (0.1-0.3). |
| `inference_top_p` | `float` | `0.9` | 0.0 -- 1.0 | Top-p (nucleus) sampling threshold |
| `inference_top_k` | `int` | `40` | >= 0 | Top-k sampling limit. 0 = disabled. |
| `inference_repeat_penalty` | `float` | `1.1` | 1.0 -- 2.0 | Repetition penalty. Values above 1.2 can degrade small models. |
| `inference_num_predict` | `int` | `256` | 32 -- 4096 | Maximum tokens to generate per response |
| `inference_num_ctx` | `int` | `2048` | 512 -- 131072 | KV-cache context window. Should match training max_seq_length. |

### Example

```yaml
serving:
  export_format: "gguf"
  gguf_quantization: "Q4_K_M"
  merge_adapter: true
  generate_modelfile: true
  inference_temperature: 0.1
  inference_top_p: 0.9
  inference_top_k: 40
  inference_repeat_penalty: 1.1
  inference_num_predict: 256
  inference_num_ctx: 2048
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
| `deepspeed_offload` | `bool` | `false` | | Offload optimizer/parameters to CPU (ZeRO-Offload) |
| `tensor_parallel_degree` | `int` | `1` | >= 1 | Tensor-parallelism degree (Megatron-LM) |
| `pipeline_parallel_degree` | `int` | `1` | >= 1 | Pipeline-parallelism degree (Megatron-LM) |
| `fp8_enabled` | `bool` | `false` | | Enable FP8 compute via Transformer Engine |
| `fp8_format` | `string` | `"HYBRID"` | `E4M3`, `HYBRID` | FP8 format (HYBRID = E4M3 forward, E5M2 backward) |
| `auto_micro_batch` | `bool` | `false` | | Automatically find the largest micro-batch that fits |

### Example

```yaml
distributed:
  enabled: true
  framework: "deepspeed"
  num_gpus: 4
  deepspeed_stage: 2
  deepspeed_offload: false
```

---

## `iti` -- Inference-Time Intervention Configuration

ITI finds "truthfulness directions" in attention heads and bakes them into model weights as o_proj biases, reducing hallucination at zero inference cost.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `enabled` | `bool` | `false` | | Enable ITI probing and baking pipeline stages |
| `probing_dataset` | `string` | `"truthful_qa"` | HuggingFace dataset ID | Dataset for probing truthfulness directions |
| `num_probing_samples` | `int` | `500` | >= 10 | Number of probing samples |
| `num_heads` | `int` | `48` | >= 1 | Top-K attention heads to intervene on |
| `alpha` | `float` | `15.0` | > 0.0 | Intervention strength |
| `method` | `string` | `"center_of_mass"` | `center_of_mass`, `linear_probe` | Method for computing truthfulness directions |
| `bake_in` | `bool` | `true` | | Bake directions into model weights (Ollama/GGUF compatible) |

### Example

```yaml
iti:
  enabled: true
  method: "center_of_mass"
  alpha: 15.0
  num_heads: 48
  bake_in: true
```

---

## `refusal` -- Refusal-Aware Training Configuration

Mixes "I don't know" refusal examples into training data so the model learns to refuse rather than hallucinate on questions beyond its knowledge (R-Tuning).

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `enabled` | `bool` | `false` | | Enable refusal data augmentation |
| `refusal_ratio` | `float` | `0.15` | 0.0 -- 1.0 (exclusive) | Fraction of training data to replace with refusal examples |
| `refusal_responses` | `list[string]` | 3 default templates | | Pool of refusal response templates |

### Example

```yaml
refusal:
  enabled: true
  refusal_ratio: 0.15
  refusal_responses:
    - "I don't have enough information to answer that accurately."
    - "I'm not confident in my knowledge about this topic."
    - "I don't know the answer to that question."
```

---

## `ifd` -- IFD Data Scoring Configuration

Instruction-Following Difficulty scoring filters training data by computing `IFD(Q, A) = s(A|Q) / s(A)`. High-IFD samples (where the instruction didn't help) are often more valuable for training.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `enabled` | `bool` | `false` | | Enable IFD scoring and filtering |
| `select_ratio` | `float` | `0.5` | 0.0 -- 1.0 | Fraction of data to keep (top-k by IFD score) |
| `batch_size` | `int` | `4` | >= 1 | Batch size for scoring forward passes |
| `max_length` | `int` | `512` | >= 64 | Maximum sequence length for scoring |

### Example

```yaml
ifd:
  enabled: true
  select_ratio: 0.5
  batch_size: 4
  max_length: 512
```

---

## `merge` -- Model Merging Configuration

Post-training model merging using TIES, SLERP, or linear averaging.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `enabled` | `bool` | `false` | | Enable model merging as a post-training step |
| `method` | `string` | `"linear"` | `linear`, `slerp`, `ties` | Merge method |
| `models` | `list[string]` | `[]` | Model paths or HuggingFace IDs | Models to merge |
| `weights` | `list[float]` | `[]` | | Per-model weights. Empty = equal weights. |
| `base_model` | `string` | `null` | | Base model for TIES (task vectors are computed relative to this) |
| `slerp_t` | `float` | `0.5` | 0.0 -- 1.0 | SLERP interpolation parameter |
| `ties_density` | `float` | `0.5` | 0.0 -- 1.0 | TIES trimming density. Lower = more aggressive. |
| `output_path` | `string` | `null` | | Where to save. Default: `output_dir/merged` |

### Example

```yaml
merge:
  enabled: true
  method: "slerp"
  models:
    - "./outputs/model-a/merged"
    - "./outputs/model-b/merged"
  slerp_t: 0.5
  output_path: "./outputs/merged-model"
```

---

## `alignment` -- Preference Alignment Configuration

Preference-based alignment training for DPO, ORPO, and GRPO modes.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `preference_dataset` | `string` | `null` | Path or HuggingFace ID | Preference dataset with prompt/chosen/rejected columns |
| `prompt_field` | `string` | `"prompt"` | | Column name for prompts |
| `chosen_field` | `string` | `"chosen"` | | Column name for preferred responses |
| `rejected_field` | `string` | `"rejected"` | | Column name for dispreferred responses |
| `beta` | `float` | `0.1` | > 0.0 | KL divergence penalty coefficient |
| `max_prompt_length` | `int` | `512` | >= 64 | Maximum prompt length in tokens |
| `max_length` | `int` | `1024` | >= 128 | Maximum total sequence length |
| `loss_type` | `string` | `"sigmoid"` | `sigmoid`, `hinge`, `ipo`, `kto_pair` | DPO loss variant |
| `num_generations` | `int` | `4` | >= 2 | Completions per prompt for GRPO group scoring |
| `max_completion_length` | `int` | `256` | >= 16 | Maximum completion length for GRPO |

### Example

```yaml
training:
  mode: "dpo"

alignment:
  preference_dataset: "Anthropic/hh-rlhf"
  beta: 0.1
  loss_type: "sigmoid"
  max_prompt_length: 512
  max_length: 1024
```

---

## `compute` -- Compute Backend Configuration

Controls whether training runs locally, on a SLURM cluster, or on a cloud GPU provider.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `backend` | `string` | `"local"` | `local`, `slurm`, `aws`, `gcp`, `azure`, `lambda`, `runpod`, `ssh` | Where to run training |
| `sync_code` | `bool` | `true` | | Rsync local code to remote before training |
| `stream_logs` | `bool` | `true` | | Stream training logs back to local terminal |
| `pull_outputs` | `bool` | `true` | | Pull output artifacts back to local after training |
| `local_output_dir` | `string` | `null` | | Local directory to pull remote outputs to |
| `ssh` | `object` | `null` | | SSH connection settings (required for `slurm` and `ssh` backends) |
| `slurm` | `object` | `null` | | SLURM job settings (required for `slurm` backend) |
| `cloud` | `object` | `null` | | Cloud GPU settings (for `aws`/`gcp`/`azure`/`lambda`/`runpod`) |

### `compute.ssh` -- SSH Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | `string` | **required** | SSH hostname or alias from `~/.ssh/config` |
| `user` | `string` | `null` | SSH username |
| `key_path` | `string` | `null` | Path to SSH private key |
| `remote_dir` | `string` | `"~/llm-forge"` | Working directory on the remote machine |
| `sync_exclude` | `list[string]` | `[".venv", "__pycache__", "*.pyc", "outputs/", ...]` | Patterns to exclude from rsync |

### `compute.slurm` -- SLURM Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `partition` | `string` | `"gpuq"` | SLURM partition to submit to |
| `qos` | `string` | `"gpu"` | Quality-of-service tier |
| `gres` | `string` | `"gpu:A100.80gb:1"` | Generic resource specification |
| `cpus_per_task` | `int` | `8` | CPU cores per task |
| `mem_gb` | `int` | `64` | Memory in GB |
| `time_limit` | `string` | `"2:00:00"` | Wall time limit (HH:MM:SS) |
| `exclude_nodes` | `string` | `null` | Comma-separated nodes to exclude |
| `modules` | `list[string]` | `[]` | Environment modules to load |
| `conda_env` | `string` | `null` | Conda environment name to activate |
| `conda_prefix` | `string` | `null` | Path to conda/miniforge installation |
| `extra_env` | `dict[string, string]` | `{}` | Extra environment variables |
| `extra_sbatch_flags` | `dict[string, string]` | `{}` | Additional SBATCH directives |

### `compute.cloud` -- Cloud GPU Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `instance_type` | `string` | `null` | Instance type (e.g. `p4d.24xlarge`) |
| `region` | `string` | `null` | Cloud region |
| `gpu_type` | `string` | `null` | GPU type (e.g. `A100`, `H100`) |
| `num_gpus` | `int` | `1` | Number of GPUs to request |
| `disk_gb` | `int` | `200` | Disk size in GB |
| `spot` | `bool` | `false` | Use spot/preemptible instances |
| `max_price_per_hour` | `float` | `null` | Maximum hourly price for spot (USD) |
| `auto_shutdown` | `bool` | `true` | Auto-stop instance after training |
| `docker_image` | `string` | `null` | Docker image to use |
| `setup_commands` | `list[string]` | `[]` | Shell commands to run before training |

### Example: SLURM Cluster

```yaml
compute:
  backend: "slurm"
  sync_code: true
  stream_logs: true
  pull_outputs: true
  ssh:
    host: "hopper"
    remote_dir: "~/llm-forge"
  slurm:
    partition: "gpuq"
    qos: "gpu"
    gres: "gpu:A100.80gb:1"
    cpus_per_task: 8
    mem_gb: 64
    time_limit: "2:00:00"
    exclude_nodes: "gpu032,dgx003"
    conda_env: "llm-forge"
    conda_prefix: "~/miniforge"
    extra_env:
      HF_HOME: "/scratch/user/hf_cache"
```

### Example: AWS Cloud

```yaml
compute:
  backend: "aws"
  cloud:
    instance_type: "p4d.24xlarge"
    region: "us-east-1"
    gpu_type: "A100"
    num_gpus: 8
    spot: true
    max_price_per_hour: 15.0
    auto_shutdown: true
```

---

## `mac` -- Apple Silicon Configuration

MacOS / Apple Silicon training optimisations.

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `smart_memory` | `bool` | `true` | | Auto-reduce batch size on memory pressure, OOM recovery |
| `memory_pressure_threshold` | `float` | `0.85` | 0.5 -- 0.99 | RAM fraction that triggers batch-size reduction |
| `thermal_aware` | `bool` | `true` | | Detect thermal throttling and pause training |
| `thermal_pause_seconds` | `int` | `30` | 5 -- 300 | Seconds to pause when throttling detected |
| `battery_aware` | `bool` | `true` | | Pause training on low battery |
| `min_battery_pct` | `int` | `20` | 5 -- 50 | Minimum battery percentage |
| `mps_high_watermark_ratio` | `float` | `0.0` | 0.0 -- 1.0 | MPS memory watermark. 0.0 = no limit. |

---

## `mlx` -- MLX Training Configuration

MLX-based training on Apple Silicon as an alternative to PyTorch.

| Field | Type | Default | Range/Options | Description |
|-------|------|---------|---------------|-------------|
| `enabled` | `bool` | `false` | | Use MLX backend instead of PyTorch |
| `fine_tune_type` | `string` | `"lora"` | `lora`, `dora`, `full` | Fine-tuning strategy |
| `num_layers` | `int` | `16` | -1 = all | Number of layers to adapt from the end |
| `lora_rank` | `int` | `8` | >= 1 | LoRA rank |
| `lora_scale` | `float` | `20.0` | > 0.0 | LoRA scaling factor |
| `lora_dropout` | `float` | `0.0` | 0.0 -- 1.0 | LoRA dropout |
| `iters` | `int` | `1000` | >= 1 | Total training iterations |
| `batch_size` | `int` | `4` | >= 1 | Batch size |
| `learning_rate` | `float` | `1e-5` | > 0.0 | Peak learning rate |
| `optimizer` | `string` | `"adam"` | `adam`, `adamw`, `sgd`, `adafactor` | Optimizer |
| `max_seq_length` | `int` | `2048` | >= 64 | Maximum sequence length |
| `grad_checkpoint` | `bool` | `false` | | Gradient checkpointing |
| `mask_prompt` | `bool` | `true` | | Train only on assistant/completion tokens |
| `lr_schedule` | `string` | `"cosine_decay"` | `cosine_decay`, `linear_decay`, `null` | LR schedule |
| `fuse_after_training` | `bool` | `true` | | Fuse LoRA adapters into base weights after training |

### Example

```yaml
mlx:
  enabled: true
  fine_tune_type: "lora"
  lora_rank: 8
  iters: 1000
  learning_rate: 1.0e-5
  max_seq_length: 2048
  mask_prompt: true
  fuse_after_training: true
```

---

## Common Configuration Patterns

### Minimal LoRA fine-tuning (only required fields)

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

### Knowledge-preserving domain specialization

```yaml
model:
  name: "unsloth/Llama-3.2-1B-Instruct"
  max_seq_length: 2048

lora:
  r: 8
  alpha: 16
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj

data:
  train_path: "Josephgflowers/Finance-Instruct-500k"
  format: "sharegpt"
  system_prompt: "You are a finance specialist."
  max_samples: 20000
  cleaning:
    enabled: true
    quality_preset: "permissive"
    dedup_enabled: true

training:
  mode: "lora"
  num_epochs: 1
  learning_rate: 1.0e-5
  assistant_only_loss: true
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

### Full pipeline with GGUF export

```yaml
model:
  name: "meta-llama/Llama-3.2-1B"
  max_seq_length: 2048

data:
  train_path: "./data/domain_train.jsonl"
  format: "sharegpt"
  cleaning:
    enabled: true
    quality_preset: "strict"
    language_filter: ["en"]
    pii_redaction: true

training:
  mode: "lora"
  num_epochs: 1
  learning_rate: 2.0e-5

evaluation:
  enabled: true
  benchmarks:
    - mmlu
    - hellaswag
  generate_report: true
  regression_check: true

serving:
  export_format: "gguf"
  gguf_quantization: "Q4_K_M"
  merge_adapter: true
  generate_modelfile: true
  inference_temperature: 0.1
  inference_num_ctx: 2048
```

---

## Validation

llm-forge validates your configuration before training and provides actionable error messages:

```bash
llm-forge validate config.yaml
```

The validator checks:
- All field types and value ranges
- Cross-field constraints (e.g., `bf16` + `fp16` conflict)
- Hardware compatibility and memory estimates
- Forgetting risk warnings (high learning rate, all-linear targets on small datasets, NEFTune on small datasets)
- Overfitting risk warnings (multiple epochs on small datasets)

---

## Built-in Presets

Generate a starter config with `llm-forge init --template <name>`:

| Template | Mode | Description |
|----------|------|-------------|
| `lora` | LoRA | Standard LoRA fine-tuning |
| `qlora` | QLoRA | Memory-efficient 4-bit LoRA |
| `pretrain` | Pretrain | Training from scratch with FSDP |
| `rag` | RAG + LoRA | RAG pipeline with ChromaDB |
| `full` | Full | All-parameter fine-tuning |
