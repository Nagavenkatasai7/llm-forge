# API Reference

Complete Python API reference for the llm-forge platform. All import paths, class signatures, key methods, and usage examples are derived from the actual source code.

---

## Table of Contents

1. [llm_forge.config](#llm_forgeconfig) -- Configuration, Validation, Hardware Detection
2. [llm_forge.data](#llm_forgedata) -- Data Loading, Preprocessing, Cleaning, IFD Scoring, Refusal Augmentation
3. [llm_forge.training](#llm_forgetraining) -- Fine-tuning, Pre-training, Alignment, MLX Training
4. [llm_forge.pipeline](#llm_forgepipeline) -- Pipeline Runner, DAG Builder
5. [llm_forge.evaluation](#llm_forgeevaluation) -- Benchmarks, Domain Evaluation, LLM Judge, Metrics
6. [llm_forge.serving](#llm_forgeserving) -- Model Export, FastAPI Server, Gradio App
7. [llm_forge.rag](#llm_forgerag) -- RAG Pipeline
8. [llm_forge.utils](#llm_forgeutils) -- Logging, Error Recovery, GPU Utilities
9. [Enums](#enums)
10. [CLI Commands](#cli-commands)
11. [Dependencies](#dependencies)

---

## llm_forge.config

The configuration system is the backbone of llm-forge. A single YAML file is parsed into a Pydantic v2 `LLMForgeConfig` model that drives every pipeline stage.

### LLMForgeConfig

Top-level configuration model. Every subsystem reads its parameters from this object.

```python
from llm_forge.config import LLMForgeConfig
```

| Attribute       | Type                 | Description                                          |
|-----------------|----------------------|------------------------------------------------------|
| `model`         | `ModelConfig`        | Pretrained model selection and loading. **Required.** |
| `lora`          | `LoRAConfig`         | LoRA / QLoRA adapter hyperparameters.                |
| `quantization`  | `QuantizationConfig` | BitsAndBytes quantization options.                   |
| `data`          | `DataConfig`         | Dataset paths, format, cleaning, and splitting.      |
| `training`      | `TrainingConfig`     | Core training hyperparameters.                       |
| `distributed`   | `DistributedConfig`  | Multi-GPU / multi-node settings.                     |
| `evaluation`    | `EvalConfig`         | Post-training evaluation.                            |
| `rag`           | `RAGConfig`          | Retrieval-Augmented Generation pipeline.             |
| `serving`       | `ServingConfig`      | Serving backend and model export.                    |

**Usage -- from YAML:**

```python
from llm_forge.config import validate_config_file

config = validate_config_file("configs/my_config.yaml")
# config is a fully validated LLMForgeConfig instance
```

**Usage -- from dict:**

```python
from llm_forge.config import validate_config_dict

config = validate_config_dict({
    "model": {"name": "meta-llama/Llama-3.2-1B"},
    "data": {"train_path": "tatsu-lab/alpaca"},
})
```

**Usage -- from Python:**

```python
from llm_forge.config.schema import LLMForgeConfig, ModelConfig, DataConfig

config = LLMForgeConfig(
    model=ModelConfig(name="meta-llama/Llama-3.2-1B"),
    data=DataConfig(train_path="tatsu-lab/alpaca"),
)
```

**Auto-configuration:** When `training.mode` is `"qlora"`, the schema validator automatically enables 4-bit quantization (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`).

---

### ModelConfig

```python
from llm_forge.config.schema import ModelConfig
```

| Field                  | Type                    | Default                | Description                                               |
|------------------------|-------------------------|------------------------|-----------------------------------------------------------|
| `name`                 | `str`                   | **Required**           | HuggingFace model ID or local path                        |
| `revision`             | `Optional[str]`         | `None`                 | Git revision (branch, tag, commit SHA)                    |
| `trust_remote_code`    | `bool`                  | `False`                | Trust code shipped inside the model repo                  |
| `torch_dtype`          | `PrecisionMode`         | `"bf16"`               | Dtype for loading weights                                 |
| `max_seq_length`       | `int`                   | `2048`                 | Maximum sequence length (128 - 131072)                    |
| `attn_implementation`  | `Literal`               | `"flash_attention_2"`  | Attention kernel: `"eager"`, `"sdpa"`, `"flash_attention_2"` |
| `rope_scaling`         | `Optional[dict]`        | `None`                 | RoPE scaling, e.g. `{"type": "dynamic", "factor": 2.0}`  |

---

### LoRAConfig

```python
from llm_forge.config.schema import LoRAConfig
```

| Field            | Type          | Default                                                                            | Description                      |
|------------------|---------------|------------------------------------------------------------------------------------|----------------------------------|
| `r`              | `int`         | `16`                                                                               | LoRA rank (1 - 256)              |
| `alpha`          | `int`         | `32`                                                                               | LoRA scaling factor              |
| `dropout`        | `float`       | `0.05`                                                                             | LoRA dropout (0.0 - 0.5)        |
| `target_modules` | `list[str]`   | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`   | Module patterns to apply LoRA to |
| `bias`           | `Literal`     | `"none"`                                                                           | Bias training: `"none"`, `"all"`, `"lora_only"` |
| `task_type`      | `Literal`     | `"CAUSAL_LM"`                                                                     | PEFT task type                   |
| `use_rslora`     | `bool`        | `False`                                                                            | Rank-Stabilized LoRA             |
| `use_dora`       | `bool`        | `False`                                                                            | Weight-Decomposed LoRA (DoRA)    |

---

### QuantizationConfig

```python
from llm_forge.config.schema import QuantizationConfig
```

| Field                       | Type            | Default  | Description                               |
|-----------------------------|-----------------|----------|-------------------------------------------|
| `load_in_4bit`              | `bool`          | `False`  | Load model in 4-bit precision             |
| `load_in_8bit`              | `bool`          | `False`  | Load model in 8-bit precision             |
| `bnb_4bit_compute_dtype`    | `PrecisionMode` | `"bf16"` | Compute dtype for 4-bit inference         |
| `bnb_4bit_quant_type`       | `Literal`       | `"nf4"`  | Quantization type: `"nf4"` or `"fp4"`    |
| `bnb_4bit_use_double_quant` | `bool`          | `True`   | Nested/double quantization                |

**Validation:** Cannot enable both `load_in_4bit` and `load_in_8bit`.

---

### DataConfig

```python
from llm_forge.config.schema import DataConfig
```

| Field             | Type                    | Default        | Description                                              |
|-------------------|-------------------------|----------------|----------------------------------------------------------|
| `train_path`      | `str`                   | **Required**   | Path or HuggingFace dataset ID                           |
| `eval_path`       | `Optional[str]`         | `None`         | Evaluation data path                                     |
| `format`          | `DataFormat`            | `"alpaca"`     | `"alpaca"`, `"sharegpt"`, `"completion"`, `"custom"`     |
| `input_field`     | `str`                   | `"instruction"`| Column for user instruction                              |
| `output_field`    | `str`                   | `"output"`     | Column for expected output                               |
| `context_field`   | `Optional[str]`         | `"input"`      | Column for optional context                              |
| `system_prompt`   | `Optional[str]`         | `None`         | System prompt prepended to every sample                  |
| `max_samples`     | `Optional[int]`         | `None`         | Cap training samples (for debugging)                     |
| `test_size`       | `float`                 | `0.05`         | Eval split fraction (0.0 - 1.0)                         |
| `seed`            | `int`                   | `42`           | Random seed                                              |
| `streaming`       | `bool`                  | `False`        | Stream instead of loading into RAM                       |
| `num_workers`     | `int`                   | `4`            | Data-loader worker count                                 |
| `cleaning`        | `DataCleaningConfig`    | Default        | Data cleaning sub-config                                 |

---

### DataCleaningConfig

```python
from llm_forge.config.schema import DataCleaningConfig
```

| Field                           | Type                     | Default                                     | Description                          |
|---------------------------------|--------------------------|---------------------------------------------|--------------------------------------|
| `enabled`                       | `bool`                   | `True`                                      | Master switch for data cleaning      |
| `quality_preset`                | `QualityPreset`          | `"balanced"`                                | `"permissive"`, `"balanced"`, `"strict"` |
| `unicode_fix`                   | `bool`                   | `True`                                      | Fix encoding via ftfy                |
| `language_filter`               | `Optional[list[str]]`    | `None`                                      | ISO-639 codes to keep, e.g. `["en"]`|
| `language_confidence_threshold` | `float`                  | `0.65`                                      | FastText confidence threshold        |
| `heuristic_filter`              | `bool`                   | `True`                                      | Rule-based quality heuristics        |
| `min_word_count`                | `int`                    | `5`                                         | Minimum words per document           |
| `max_word_count`                | `int`                    | `100000`                                    | Maximum words per document           |
| `dedup_enabled`                 | `bool`                   | `True`                                      | Enable deduplication                 |
| `dedup_tiers`                   | `list[DeduplicationTier]`| `["exact", "fuzzy"]`                        | Dedup strategies                     |
| `dedup_jaccard_threshold`       | `float`                  | `0.85`                                      | Jaccard threshold for fuzzy dedup    |
| `semantic_dedup_enabled`        | `bool`                   | `False`                                     | Enable semantic deduplication        |
| `toxicity_filter`               | `bool`                   | `False`                                     | Enable toxicity scoring              |
| `toxicity_threshold`            | `float`                  | `0.8`                                       | Toxicity score threshold             |
| `pii_redaction`                 | `bool`                   | `False`                                     | Enable PII redaction (Presidio)      |

---

### TrainingConfig

```python
from llm_forge.config.schema import TrainingConfig
```

| Field                             | Type               | Default         | Description                                          |
|-----------------------------------|--------------------|-----------------|------------------------------------------------------|
| `mode`                            | `TrainingMode`     | `"lora"`        | `"lora"`, `"qlora"`, `"full"`, `"pretrain"`, `"dpo"`, `"orpo"`, `"grpo"` |
| `output_dir`                      | `str`              | `"outputs"`     | Output directory for checkpoints and models           |
| `num_epochs`                      | `int`              | `3`             | Training epochs                                       |
| `per_device_train_batch_size`     | `int`              | `4`             | Micro-batch size per GPU (train)                      |
| `per_device_eval_batch_size`      | `int`              | `4`             | Micro-batch size per GPU (eval)                       |
| `gradient_accumulation_steps`     | `int`              | `4`             | Gradient accumulation steps                           |
| `learning_rate`                   | `float`            | `2e-4`          | Peak learning rate                                    |
| `weight_decay`                    | `float`            | `0.01`          | L2 weight decay                                       |
| `warmup_ratio`                    | `float`            | `0.03`          | Warmup fraction                                       |
| `warmup_steps`                    | `Optional[int]`    | `None`          | Exact warmup steps (overrides ratio)                  |
| `lr_scheduler_type`               | `Literal`          | `"cosine"`      | LR scheduler type                                     |
| `max_grad_norm`                   | `float`            | `1.0`           | Gradient clipping norm                                |
| `bf16`                            | `bool`             | `True`          | BF16 mixed precision                                  |
| `fp16`                            | `bool`             | `False`         | FP16 mixed precision                                  |
| `gradient_checkpointing`         | `bool`             | `False`         | Trade compute for memory                              |
| `optim`                           | `str`              | `"adamw_torch"` | Optimizer name                                        |
| `neftune_noise_alpha`             | `Optional[float]`  | `5.0`           | NEFTune noise alpha (null to disable)                 |
| `label_smoothing_factor`          | `float`            | `0.1`           | Label smoothing (0.0 to disable)                      |
| `average_tokens_across_devices`   | `bool`             | `True`          | Sync token counts for correct loss in distributed     |
| `use_unsloth`                     | `bool`             | `False`         | Use Unsloth accelerated kernels                       |
| `assistant_only_loss`             | `bool`             | `True`          | Train only on assistant tokens (TRL chat masking)     |
| `report_to`                       | `list[str]`        | `["wandb"]`     | Experiment trackers                                   |
| `resume_from_checkpoint`          | `Optional[str]`    | `None`          | Checkpoint path to resume from                        |

**Validation:** Cannot enable both `bf16` and `fp16`.

---

### DistributedConfig

```python
from llm_forge.config.schema import DistributedConfig
```

| Field                       | Type       | Default        | Description                                                     |
|-----------------------------|------------|----------------|-----------------------------------------------------------------|
| `enabled`                   | `bool`     | `False`        | Enable distributed training                                     |
| `framework`                 | `Literal`  | `"auto"`       | `"auto"`, `"fsdp"`, `"deepspeed"`, `"megatron"`               |
| `num_gpus`                  | `int`      | `1`            | Number of GPUs                                                  |
| `num_nodes`                 | `int`      | `1`            | Number of cluster nodes                                         |
| `fsdp_sharding_strategy`    | `Literal`  | `"FULL_SHARD"` | `"FULL_SHARD"`, `"SHARD_GRAD_OP"`, `"NO_SHARD"`, `"HYBRID_SHARD"` |
| `deepspeed_stage`           | `Literal`  | `2`            | ZeRO stage: `0`, `1`, `2`, `3`                                |
| `deepspeed_offload`         | `bool`     | `False`        | CPU offloading for optimizer/params                             |
| `tensor_parallel_degree`    | `int`      | `1`            | Tensor parallelism degree (Megatron)                            |
| `pipeline_parallel_degree`  | `int`      | `1`            | Pipeline parallelism degree (Megatron)                          |
| `fp8_enabled`               | `bool`     | `False`        | FP8 compute via Transformer Engine (Hopper GPUs)                |
| `fp8_format`                | `Literal`  | `"HYBRID"`     | FP8 format: `"E4M3"` or `"HYBRID"`                            |
| `auto_micro_batch`          | `bool`     | `False`        | Auto-find largest micro-batch                                   |

---

### EvalConfig

```python
from llm_forge.config.schema import EvalConfig
```

| Field              | Type              | Default                                | Description                             |
|--------------------|-------------------|----------------------------------------|-----------------------------------------|
| `enabled`          | `bool`            | `True`                                 | Run benchmarks after training           |
| `benchmarks`       | `list[str]`       | `["hellaswag", "arc_easy", "mmlu"]`    | Benchmark task names                    |
| `custom_eval_path` | `Optional[str]`   | `None`                                 | Path to custom evaluation dataset       |
| `num_fewshot`      | `int`             | `0`                                    | Few-shot examples for benchmarks        |
| `batch_size`       | `int`             | `8`                                    | Inference batch size                    |
| `generate_report`  | `bool`            | `True`                                 | Generate evaluation report              |

---

### RAGConfig

```python
from llm_forge.config.schema import RAGConfig
```

| Field                  | Type              | Default                                         | Description                      |
|------------------------|-------------------|-------------------------------------------------|----------------------------------|
| `enabled`              | `bool`            | `False`                                         | Enable RAG pipeline              |
| `knowledge_base_path`  | `Optional[str]`   | `None`                                          | Path to knowledge base           |
| `chunk_strategy`       | `Literal`         | `"recursive"`                                   | Chunking strategy                |
| `chunk_size`           | `int`             | `512`                                           | Target chunk size (64 - 8192)    |
| `chunk_overlap`        | `int`             | `64`                                            | Overlap between chunks           |
| `embedding_model`      | `str`             | `"sentence-transformers/all-MiniLM-L6-v2"`      | Embedding model                  |
| `vectorstore`          | `Literal`         | `"chromadb"`                                    | Backend: `"chromadb"`, `"faiss"` |
| `top_k`                | `int`             | `5`                                             | Chunks per query                 |
| `reranker_model`       | `Optional[str]`   | `None`                                          | Cross-encoder for reranking      |
| `hybrid_search`        | `bool`            | `False`                                         | Dense + BM25 retrieval           |
| `similarity_threshold` | `float`           | `0.7`                                           | Minimum similarity to keep       |

**Validation:** `chunk_overlap` must be less than `chunk_size`.

---

### ServingConfig

```python
from llm_forge.config.schema import ServingConfig
```

| Field               | Type              | Default     | Description                                                         |
|---------------------|-------------------|-------------|---------------------------------------------------------------------|
| `backend`           | `Literal`         | `"gradio"`  | `"gradio"`, `"fastapi"`, `"vllm"`                                  |
| `host`              | `str`             | `"0.0.0.0"` | Host to bind                                                        |
| `port`              | `int`             | `7860`      | Port number (1 - 65535)                                             |
| `export_format`     | `Optional[str]`   | `None`      | `"gguf"`, `"onnx"`, `"safetensors"`, `"awq"`, `"gptq"`            |
| `gguf_quantization` | `Optional[str]`   | `None`      | GGUF quant level, e.g. `"Q4_K_M"`                                  |
| `merge_adapter`     | `bool`            | `True`      | Merge LoRA adapter before serving/export                            |
| `inference_num_ctx` | `int`             | `2048`      | Context length for Ollama Modelfile `num_ctx`                       |

---

### validate_config_file

```python
from llm_forge.config import validate_config_file

config = validate_config_file("path/to/config.yaml")
```

Loads a YAML file, validates it against `LLMForgeConfig`, and returns the validated object. Raises `FileNotFoundError` if the path does not exist, or `ConfigValidationError` with detailed, field-level error messages if validation fails.

---

### validate_config_dict

```python
from llm_forge.config import validate_config_dict

config = validate_config_dict({
    "model": {"name": "meta-llama/Llama-3.2-1B"},
    "data": {"train_path": "tatsu-lab/alpaca"},
    "training": {"mode": "lora", "num_epochs": 1},
})
```

Validates a raw dictionary. Checks for unknown top-level keys and provides suggestions for common typos (e.g., `"lr"` suggests `"training.learning_rate"`). Prints warnings for non-existent file paths to stderr but still returns the config.

---

### detect_hardware

```python
from llm_forge.config import detect_hardware, HardwareProfile

profile: HardwareProfile = detect_hardware()
print(profile.summary())
```

Returns a `HardwareProfile` dataclass with:

| Property               | Type             | Description                                    |
|------------------------|------------------|------------------------------------------------|
| `gpu_count`            | `int`            | Number of detected NVIDIA GPUs                 |
| `gpus`                 | `list[GPUInfo]`  | Per-GPU details (name, VRAM, compute cap)      |
| `cuda_version`         | `str or None`    | CUDA runtime version                           |
| `driver_version`       | `str or None`    | NVIDIA driver version                          |
| `nvlink`               | `NVLinkTopology` | NVLink connectivity information                |
| `is_mps`               | `bool`           | Apple MPS backend available                    |
| `apple_chip`           | `str or None`    | Apple Silicon chip name (e.g. "Apple M4 Pro")  |
| `system_ram_mb`        | `int`            | Total system RAM in MB                         |
| `total_vram_gb`        | `float`          | Sum of all GPU VRAM (property)                 |
| `has_multi_gpu`        | `bool`           | True if 2+ GPUs detected (property)            |
| `all_support_bf16`     | `bool`           | All GPUs are Ampere+ (property)                |
| `any_supports_fp8`     | `bool`           | Any GPU is Hopper+ (property)                  |

Safe to call on any machine -- gracefully handles missing GPUs, missing `nvidia-smi`, and missing optional dependencies.

---

### auto_optimize_config

```python
from llm_forge.config import auto_optimize_config, detect_hardware

profile = detect_hardware()
config = auto_optimize_config(config, profile)
```

Adjusts the config in-place based on detected hardware. Key behaviors:

- **Apple Silicon / MPS**: Disables bf16/fp16, sets fp32, uses eager attention, adjusts batch sizes based on unified memory.
- **CPU-only**: Forces fp32, batch_size=1, gradient checkpointing, converts full/pretrain to LoRA.
- **Multi-GPU**: Enables distributed training, selects FSDP (if NVLink) or DeepSpeed (if PCIe).
- **Per-GPU class**: Sets precision, batch size, and memory optimizations based on GPU family (RTX 3090, RTX 4090, A100 40 GB, A100 80 GB, H100/H200).

Pass `profile=None` to let it call `detect_hardware()` internally.

---

### load_preset / list_presets

```python
from llm_forge.config import load_preset, list_presets

# List available presets
names = list_presets()
# ['dpo_default', 'full_finetune', 'lora_default', 'qlora_default', ...]

# Load a preset as a validated config
config = load_preset("lora_default")
```

Presets are YAML files in `llm_forge/config/presets/`. Each is validated at load time.

---

## llm_forge.data

### DataLoader

Universal data connector supporting files, directories, URLs, and HuggingFace datasets.

```python
from llm_forge.data import DataLoader

loader = DataLoader(
    path="tatsu-lab/alpaca",     # File, directory, URL, or HF dataset ID
    streaming=False,
    num_workers=4,
    max_samples=1000,            # None for all samples
    seed=42,
)

dataset = loader.load()          # Returns datasets.Dataset
```

| Method              | Description                              | Returns                |
|---------------------|------------------------------------------|------------------------|
| `load()`            | Auto-detect source type and load         | `Dataset`              |
| `load_streaming()`  | Load as a streaming iterator             | `Iterator[dict]`       |

**Supported formats:** `.jsonl`, `.json`, `.csv`, `.tsv`, `.parquet`, `.txt`, `.md`, `.pdf`, `.docx`, `.html`

The loader auto-detects the source type:
1. Starts with `http://` or `https://` -- downloads and loads as file.
2. Local path exists and is a directory -- recursively loads all supported files.
3. Local path exists and is a file -- loads based on extension.
4. Otherwise -- assumes HuggingFace dataset ID and calls `load_dataset()`.

---

### DataPreprocessor

Converts raw data into training-ready format.

```python
from llm_forge.data import DataPreprocessor

preprocessor = DataPreprocessor(
    format_type="sharegpt",        # "alpaca", "sharegpt", "completion", "custom"
    input_field="instruction",
    output_field="output",
    context_field="input",
    system_prompt="You are a helpful assistant.",
    max_seq_length=2048,
)
```

| Method                                              | Description                                    | Returns                     |
|-----------------------------------------------------|------------------------------------------------|-----------------------------|
| `format_dataset(dataset)`                           | Convert to unified text format                 | `Dataset`                   |
| `format_for_chat_template(dataset, tokenizer)`      | Format using the model's chat template         | `Dataset`                   |
| `tokenize_dataset(dataset, tokenizer, pack=False)`  | Tokenize with label masking                    | `Dataset`                   |
| `split_dataset(dataset, test_size=0.1, seed=42)`    | Train/eval split                               | `tuple[Dataset, Dataset]`   |

**ShareGPT format support:** The preprocessor handles three layouts:
- Classic ShareGPT: `conversations` list with `from`/`value` keys.
- OpenAI-style: `messages` list with `role`/`content` keys.
- Flat columns: `system`, `user`, `assistant` columns (e.g., Finance-Instruct-500k).

It outputs both a `text` column (backward compatible) and a `messages` column (for TRL chat-template pipeline and assistant-only loss masking).

---

### CleaningPipeline

Seven-step data cleaning pipeline. Each step is independently configurable.

```python
from llm_forge.data.cleaning import CleaningPipeline

pipeline = CleaningPipeline(
    config=cleaning_config,       # DataCleaningConfig, dict, or None
    text_field="text",
)

cleaned_dataset, stats = pipeline.run(dataset)
print(stats.summary())
```

**Pipeline steps (in order):**

| Step | Operation           | Config Key          | Description                               |
|------|---------------------|---------------------|-------------------------------------------|
| 1    | Unicode fix         | `unicode_fix`       | ftfy encoding repair, NFC normalization   |
| 2    | Language filter      | `language_filter`   | FastText lid.176.bin detection            |
| 3    | Heuristic filter     | `heuristic_filter`  | Gopher/C4/FineWeb quality rules           |
| 4    | Deduplication        | `dedup_enabled`     | Exact SHA-256, fuzzy MinHash, semantic    |
| 5    | Quality classifier   | `quality_preset`    | FastText + KenLM scoring                  |
| 6    | PII redaction        | `pii_redaction`     | Microsoft Presidio entity detection       |
| 7    | Toxicity filter      | `toxicity_filter`   | Detoxify model scoring                    |

**Returns:** `tuple[Dataset, CleaningStats]`

The `CleaningStats` dataclass provides:
- `initial_count`, `final_count`, `total_removed`, `retention_rate`
- Per-step counts: `removed_by_language`, `removed_by_heuristic`, `removed_by_dedup`, etc.
- Timing per step and total pipeline time
- `summary()` method for a human-readable report

---

### IFDScorer

Instruction-Following Difficulty scorer for data quality filtering. Implements the IFD metric from Li et al. (NAACL 2024).

```python
from llm_forge.data.ifd_scorer import IFDScorer

scorer = IFDScorer(model=model, tokenizer=tokenizer)
result = scorer.score_dataset(dataset)
# result.scores: list[float] -- IFD score per sample
# Higher IFD = instruction didn't help = more valuable for training
```

IFD(Q, A) = s(A|Q) / s(A), where s(A|Q) is the average per-token NLL of the response given the instruction, and s(A) is the NLL of the response alone.

---

### RefusalAugmentor

Mixes "I don't know" refusal examples into the training data (R-Tuning, Zhang et al. 2023).

```python
from llm_forge.data import RefusalAugmentor

augmentor = RefusalAugmentor(
    refusal_ratio=0.15,          # 15% of dataset becomes refusal examples
    refusal_responses=None,       # Uses default templates if None
    seed=42,
)

augmented = augmentor.augment_dataset(
    dataset=dataset,
    model=model,                  # Optional: used for scoring which samples to convert
    tokenizer=tokenizer,
)
```

Default refusal responses include:
- "I don't have enough information to answer that accurately."
- "I'm not confident in my knowledge about this topic."
- "I don't know the answer to that question."

---

## llm_forge.training

### FineTuner

Unified fine-tuning engine for LoRA, QLoRA, and full parameter training. Handles model loading, PEFT adapter application, SFT training via TRL, and merged model export.

```python
from llm_forge.training import FineTuner

finetuner = FineTuner(config)

# Step 1: Load model with appropriate precision and quantization
model, tokenizer = finetuner.setup_model()

# Step 2: Apply LoRA adapters (if mode is lora or qlora)
model = finetuner.apply_lora(model)

# Step 3: Train with SFTTrainer
result = finetuner.train(
    model=model,
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=None,               # Optional list of TrainerCallback
)

# Step 4: Merge LoRA weights and save
output_path = finetuner.merge_and_save(model=model, output_dir="outputs/merged")
```

| Method                                               | Description                               | Returns                                         |
|------------------------------------------------------|-------------------------------------------|-------------------------------------------------|
| `setup_model(config=None)`                           | Load base model with quantization         | `tuple[PreTrainedModel, PreTrainedTokenizerBase]`|
| `apply_lora(model, config=None)`                     | Apply PEFT LoRA adapters                  | `PeftModel`                                     |
| `train(model, dataset, config=None, eval_dataset=None, callbacks=None)` | Run SFT training           | `TrainOutput`                                   |
| `merge_and_save(model=None, output_dir=None)`        | Merge adapter and save full model         | `Path`                                          |

**Unsloth integration:** When `training.use_unsloth: true` and the `unsloth` package is installed, `setup_model()` uses Unsloth's `FastLanguageModel.from_pretrained()` for accelerated loading and LoRA application.

**Chat template handling:** When `assistant_only_loss: true`, the finetuner configures TRL's SFTConfig to mask non-assistant tokens. It uses `{% generation %}` markers in the Jinja chat template for accurate binary masking.

---

### PreTrainer

Pre-training engine for training language models from scratch.

```python
from llm_forge.training import PreTrainer

pretrainer = PreTrainer(config)

# Build a model from scratch using a size preset
model = pretrainer.build_model(model_size="1B")

# Or train a custom BPE tokenizer first
tokenizer = pretrainer.train_tokenizer(
    corpus_path="data/corpus.txt",
    vocab_size=32000,
)

# Run causal language modeling pre-training
result = pretrainer.train(
    model=model,
    dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```

| Method                                              | Description                         | Returns                     |
|-----------------------------------------------------|-------------------------------------|-----------------------------|
| `build_model(config=None, model_size=None, ...)`   | Create model from scratch           | `PreTrainedModel`           |
| `train_tokenizer(corpus_path, vocab_size=32000)`    | Train BPE tokenizer                 | `PreTrainedTokenizerFast`   |
| `train(model, dataset, config=None, ...)`           | Causal LM pre-training             | `TrainOutput`               |

**Model size presets:** `"125M"`, `"350M"`, `"760M"`, `"1B"` -- each maps to specific `hidden_size`, `num_hidden_layers`, `num_attention_heads`, and `intermediate_size` values based on the Llama architecture.

---

### AlignmentTrainer

Preference-based training using DPO, ORPO, GRPO, and PPO-based RLHF.

```python
from llm_forge.training import AlignmentTrainer

aligner = AlignmentTrainer(config)

# DPO training
model, ref_model, tokenizer = aligner.setup_dpo()
dataset = aligner.prepare_preference_dataset(raw_dataset)
result = aligner.train_dpo(model=model, dataset=dataset)

# RLHF training
reward_model = aligner.setup_reward_model("reward-model-name")
result = aligner.train_rlhf(model=model, dataset=dataset)
```

| Method                                               | Description                          | Returns                                  |
|------------------------------------------------------|--------------------------------------|------------------------------------------|
| `setup_dpo(model=None, config=None)`                 | Load policy + reference models       | `tuple[model, ref_model, tokenizer]`     |
| `prepare_preference_dataset(dataset, ...)`           | Validate and normalize columns       | `Dataset`                                |
| `train_dpo(model, dataset, ...)`                     | Run DPO training (TRL DPOTrainer)    | `TrainOutput`                            |
| `setup_reward_model(reward_model_name, config=None)` | Load reward model for RLHF           | `PreTrainedModel`                        |
| `train_rlhf(model, dataset, ...)`                    | Run PPO-based RLHF                   | `dict[str, Any]`                         |

**Supported alignment methods:** DPO (via `trl.DPOTrainer`), ORPO (via `trl.ORPOTrainer`), GRPO (via `trl.GRPOTrainer`), PPO (via `trl.PPOTrainer`). Set `training.mode` to `"dpo"`, `"orpo"`, or `"grpo"` accordingly.

---

### MLXTrainer

Native Apple Silicon training via the `mlx-lm` package. Mirrors the FineTuner API so the pipeline can transparently swap backends.

```python
from llm_forge.training import MLXTrainer, is_mlx_available

if is_mlx_available():
    trainer = MLXTrainer(config)
    model, tokenizer = trainer.setup_model()
    model = trainer.apply_lora(model)
    result = trainer.train(model=model, dataset=train_dataset)
```

| Method                                  | Description                                | Returns                     |
|-----------------------------------------|--------------------------------------------|-----------------------------|
| `setup_model(config=None)`              | Load model via mlx-lm                      | `tuple[model, tokenizer]`   |
| `apply_lora(model, config=None)`        | Convert linear layers to LoRA              | model                       |
| `train(model, dataset, ...)`            | Train with mlx-lm's trainer               | `dict`                      |

**Requirements:** `pip install 'mlx-lm[train]'` (macOS Apple Silicon only). The `is_mlx_available()` function checks for both `mlx` and `mlx-lm` imports.

---

### Training Callbacks

```python
from llm_forge.training.callbacks import (
    WandBCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    GPUMonitorCallback,
    RichProgressCallback,
    StopTrainingCallback,
)
```

| Callback                | Parameters                                                   | Description                                 |
|-------------------------|--------------------------------------------------------------|---------------------------------------------|
| `WandBCallback`         | `project="llm-forge"`, `run_name=None`, `tags=None`         | Weights & Biases logging                    |
| `CheckpointCallback`    | `save_every_n_minutes=30.0`, `max_checkpoints=5`            | Time-based checkpointing                    |
| `EarlyStoppingCallback` | `patience=3`, `min_delta=0.001`, `metric_name="eval_loss"`  | Early stopping on metric plateau            |
| `GPUMonitorCallback`    | `log_every_n_steps=50`                                       | GPU utilization and memory logging          |
| `RichProgressCallback`  | (none)                                                       | Rich terminal progress bars                 |
| `StopTrainingCallback`  | `stop_event: threading.Event`                                | External stop signal (used by UI dashboard) |

All callbacks extend `transformers.TrainerCallback`.

---

## llm_forge.pipeline

### PipelineRunner

Orchestrates the complete pipeline from configuration loading through data preparation, training, evaluation, and model export.

```python
from llm_forge.pipeline import PipelineRunner

runner = PipelineRunner()

# Full pipeline run
context = runner.run("config.yaml")

# With options
context = runner.run(
    config_path_or_config="config.yaml",
    resume_from="training",        # Skip stages before "training"
    auto_optimize=True,            # Hardware detection + auto-tune
    stop_event=None,               # threading.Event for graceful cancellation
)

# Dry run (preview pipeline plan without executing)
runner.dry_run("config.yaml")
```

| Method                                                   | Description                          | Returns              |
|----------------------------------------------------------|--------------------------------------|----------------------|
| `run(config_path_or_config, resume_from=None, ...)`     | Execute full pipeline end-to-end     | `dict[str, Any]`     |
| `dry_run(config_path_or_config)`                         | Preview pipeline plan                | `None`               |

**Pipeline stages (12 stages, in execution order):**

1. `data_loading` -- Load raw data from configured source
2. `cleaning` -- Run the 7-step data cleaning pipeline
3. `preprocessing` -- Format and tokenize data
4. `refusal_augmentation` -- Mix in refusal examples
5. `ifd_scoring` -- Score and filter by instruction-following difficulty
6. `training` -- LoRA/QLoRA/full/pretrain/DPO training
7. `alignment` -- DPO/ORPO/GRPO/RLHF post-training
8. `iti_probing` -- Inference-Time Intervention probing
9. `iti_baking` -- Bake ITI vectors into model weights
10. `model_merging` -- Merge LoRA adapter into base model
11. `evaluation` -- Run benchmarks and domain evaluation
12. `export` -- Export to GGUF/ONNX/safetensors, generate Modelfile

Stages are automatically enabled/disabled based on config. The runner tracks per-stage status (pending/running/completed/failed/skipped) and provides Rich terminal output.

**Config acceptance:** The `config_path_or_config` parameter accepts:
- `str` or `Path` -- path to a YAML file
- `dict` -- raw config dictionary (validated internally)
- `LLMForgeConfig` -- pre-validated config object

---

### DAGBuilder

Constructs the pipeline DAG from a config.

```python
from llm_forge.pipeline.dag_builder import DAGBuilder, PipelineStage

builder = DAGBuilder()
stages: list[PipelineStage] = builder.build_dag(config)

for stage in stages:
    print(f"{stage.name}: enabled={stage.enabled}, deps={stage.dependencies}")
```

Each `PipelineStage` has:

| Attribute      | Type                  | Description                                       |
|----------------|-----------------------|---------------------------------------------------|
| `name`         | `str`                 | Unique stage identifier                           |
| `callable`     | `Callable`            | Function that receives and returns context dict   |
| `dependencies` | `list[str]`           | Stage names that must complete first              |
| `config`       | `dict`                | Stage-specific config extracted from master config|
| `enabled`      | `bool`                | Whether the stage should execute                  |
| `description`  | `str`                 | Human-readable description                        |

---

## llm_forge.evaluation

### BenchmarkRunner

Integrates with EleutherAI's lm-evaluation-harness for standard benchmarks.

```python
from llm_forge.evaluation import BenchmarkRunner

runner = BenchmarkRunner(device=None, cache_dir=None)

# Run standard benchmarks
results = runner.run_benchmarks(
    model_path="outputs/merged",
    tasks=["mmlu", "hellaswag", "arc_challenge", "gsm8k"],
    num_fewshot=5,
    batch_size=8,
    limit=None,           # None for full dataset, int for subset
)

# Compare base vs fine-tuned
comparison = runner.compare_models(
    base_path="meta-llama/Llama-3.2-1B",
    finetuned_path="outputs/merged",
    tasks=["mmlu", "hellaswag"],
)

# Save results
runner.save_results(results, "eval_results/benchmark.json")

# List available tasks
tasks = runner.list_tasks()
```

**Standard benchmark registry:**

| Task              | Display Name       | Default Few-shot | Metric      |
|-------------------|--------------------|------------------|-------------|
| `mmlu`            | MMLU               | 5                | `acc`       |
| `hellaswag`       | HellaSwag          | 10               | `acc_norm`  |
| `arc_easy`        | ARC-Easy           | 25               | `acc_norm`  |
| `arc_challenge`   | ARC-Challenge      | 25               | `acc_norm`  |
| `winogrande`      | WinoGrande         | 5                | `acc`       |
| `truthfulqa_mc2`  | TruthfulQA (MC2)   | 0                | `acc`       |
| `gsm8k`           | GSM8K              | 5                | `acc`       |

Falls back to perplexity-based evaluation when lm-eval is not installed.

---

### DomainEvaluator

Generation-based evaluation on custom datasets.

```python
from llm_forge.evaluation import DomainEvaluator

evaluator = DomainEvaluator(
    metrics=["exact_match", "f1", "accuracy"],
    input_field="input",
    output_field="output",
)

# Load custom evaluation dataset (JSONL format)
eval_samples = evaluator.load_dataset("eval_data/finance_qa.jsonl")

# Run evaluation
results = evaluator.evaluate(
    model=model,
    tokenizer=tokenizer,
    eval_dataset=eval_samples,
    max_new_tokens=256,
    batch_size=8,
)
```

| Method                                                  | Description                     | Returns            |
|---------------------------------------------------------|---------------------------------|--------------------|
| `load_dataset(path, input_field=None, ...)`             | Load eval dataset from JSONL    | `list[EvalSample]` |
| `evaluate(model, tokenizer, eval_dataset, ...)`         | Run full generation + scoring   | `dict[str, Any]`   |
| `evaluate_predictions(predictions, references, ...)`    | Score pre-computed predictions  | `dict[str, Any]`   |

---

### LLMJudge

Uses a language model to evaluate generated responses on criteria like helpfulness, accuracy, and coherence. Implements the "LLM-as-a-Judge" methodology (Zheng et al., NeurIPS 2023).

```python
from llm_forge.evaluation.llm_judge import LLMJudge

judge = LLMJudge(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    device="auto",
)

# Single response scoring
score = judge.score(
    instruction="Explain stock options.",
    response="Stock options are...",
    criteria="accuracy",
)
# Returns: {"score": 8, "reasoning": "The response accurately explains..."}

# Pairwise comparison
winner = judge.compare(
    instruction="Explain stock options.",
    response_a="Stock options are...",
    response_b="A stock option is...",
    criteria="helpfulness",
)
# Returns: {"winner": "B", "reasoning": "Response B provides more detail..."}
```

**Default evaluation criteria:** `helpfulness`, `accuracy`, `coherence`, `relevance`.

---

### MetricsComputer

Unified metrics computation engine.

```python
from llm_forge.evaluation import MetricsComputer

mc = MetricsComputer()

# Individual metrics
mc.compute_perplexity(model, tokenizer, texts)
mc.compute_bleu(predictions, references, max_n=4)
mc.compute_rouge(predictions, references, rouge_types=["rouge1", "rougeL"])
mc.compute_exact_match(predictions, references, normalize=True)
mc.compute_f1(predictions, references)
mc.compute_accuracy(predictions, references)

# Compute multiple metrics at once
results = mc.compute_all(
    predictions=["the cat sat"],
    references=["a cat sat on the mat"],
    include=["f1", "exact_match", "bleu"],
)
```

| Method                                        | Returns  | Optional Deps  |
|-----------------------------------------------|----------|----------------|
| `compute_perplexity(model, tokenizer, texts)` | `dict`   | torch          |
| `compute_bleu(preds, refs, max_n=4)`          | `dict`   | nltk           |
| `compute_rouge(preds, refs, rouge_types=None)` | `dict`  | rouge-score    |
| `compute_exact_match(preds, refs)`            | `dict`   | (none)         |
| `compute_f1(preds, refs)`                     | `dict`   | (none)         |
| `compute_accuracy(preds, refs)`               | `dict`   | (none)         |
| `compute_all(preds, refs, include=None)`      | `dict`   | (varies)       |

All methods degrade gracefully when optional dependencies (rouge-score, nltk) are not installed.

---

## llm_forge.serving

### ModelExporter

Converts trained models to deployment formats and pushes to HuggingFace Hub.

```python
from llm_forge.serving import ModelExporter

exporter = ModelExporter(config=config)

# Merge LoRA adapter
merged_path = exporter.merge_adapter(
    model_path="outputs/checkpoint",
    output_path="outputs/merged",
)

# Export to GGUF (for Ollama)
gguf_path = exporter.export_gguf(
    model_path="outputs/merged",
    output_path="outputs/gguf",
    quantization="Q4_K_M",
)

# Export to ONNX
onnx_path = exporter.export_onnx(
    model_path="outputs/merged",
    output_path="outputs/onnx",
)

# Generate Ollama Modelfile
modelfile = exporter.generate_modelfile(
    model_path="outputs/gguf/model.gguf",
    system_prompt="You are a helpful assistant.",
    output_path="outputs/Modelfile",
)

# Push to HuggingFace Hub
exporter.push_to_hub(
    model_path="outputs/merged",
    repo_id="username/my-model",
    private=True,
)
```

**Supported export formats:** `safetensors`, `gguf`, `onnx`, `awq`, `gptq`

**GGUF export:** Requires `llama.cpp` tools. The exporter searches for the quantize binary in `~/llama.cpp/build/bin/` and other standard locations. Uses `sys.executable` (not `"python"`) for subprocess calls.

**Modelfile generation:** Uses `range .Messages` loop for multi-turn conversation support. Adds `num_ctx` parameter. Detects `$last` message for proper template termination.

---

### FastAPIServer

REST server with OpenAI-compatible endpoints and SSE streaming.

```python
from llm_forge.serving import FastAPIServer

server = FastAPIServer(
    model_path="outputs/merged",
    config=None,                  # Optional LLMForgeConfig
)
server.start(host="0.0.0.0", port=8000)
```

**Endpoints:**

| Method | Path           | Description                                          |
|--------|----------------|------------------------------------------------------|
| GET    | `/health`      | Health check                                         |
| GET    | `/model/info`  | Model metadata (name, parameters, precision)         |
| POST   | `/generate`    | Text generation with optional SSE streaming          |
| POST   | `/chat`        | OpenAI-compatible chat completion with SSE streaming |

**Generate request body:**

```json
{
  "prompt": "Explain stock options.",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

Includes optional Sentry integration (set `SENTRY_DSN` env var).

---

### GradioApp

Chat interface with streaming generation, adjustable parameters, and RAG integration.

```python
from llm_forge.serving import GradioApp

app = GradioApp(
    model_path="outputs/merged",
    config=None,
)
app.launch(host="0.0.0.0", port=7860, share=False)
```

Features:
- Streaming token generation via `TextIteratorStreamer`
- Adjustable generation parameters (temperature, top_p, max_tokens)
- Model information panel
- Optional RAG integration (context injection from vector store)

---

## llm_forge.rag

### RAGPipeline

Full retrieval-augmented generation workflow: document ingestion, retrieval, reranking, and answer generation.

```python
from llm_forge.rag import RAGPipeline

rag = RAGPipeline(config=rag_config)

# Ingest documents
rag.ingest(
    documents=["path/to/docs/"],    # Files, directories, or text
    chunk_strategy="semantic",
    chunk_size=512,
    chunk_overlap=64,
)

# Query
results = rag.query(
    question="What is the P/E ratio of Apple?",
    top_k=5,
)
# results.chunks: list of retrieved chunks with scores
# results.answer: generated answer (if model is loaded)
```

**RAGPipelineConfig parameters:**

| Parameter              | Default                                     | Description                              |
|------------------------|---------------------------------------------|------------------------------------------|
| `chunk_strategy`       | `"semantic"`                                | `"fixed"`, `"semantic"`, `"hierarchical"`, `"adaptive"` |
| `chunk_size`           | `512`                                       | Target chunk size in characters          |
| `chunk_overlap`        | `64`                                        | Overlap between chunks                   |
| `embedding_model`      | `"all-MiniLM-L6-v2"`                       | Sentence-transformers model              |
| `vectorstore_backend`  | `"chromadb"`                                | `"chromadb"` or `"faiss"`               |
| `top_k`                | `5`                                         | Chunks to retrieve per query             |
| `alpha`                | `0.7`                                       | Dense vs BM25 weight (1.0 = pure dense)  |
| `enable_bm25`          | `True`                                      | Hybrid dense + BM25 retrieval            |
| `reranker_model`       | `"cross-encoder/ms-marco-MiniLM-L-6-v2"`   | Cross-encoder for reranking              |

**Components (individually importable):**

```python
from llm_forge.rag.chunking import Chunker, Document
from llm_forge.rag.embeddings import EmbeddingEngine
from llm_forge.rag.retriever import HybridRetriever, RetrievedChunk
from llm_forge.rag.reranker import Reranker
from llm_forge.rag.vectorstore import VectorStore, create_vectorstore
```

---

## llm_forge.utils

### get_logger / setup_logging

Rich-formatted structured logging with file output support.

```python
from llm_forge.utils import get_logger, setup_logging

# Create a module-specific logger
logger = get_logger("my_module")
logger.info("Training started with %d samples", len(dataset))

# Configure logging globally (call once at startup)
setup_logging(
    level="INFO",              # DEBUG, INFO, WARNING, ERROR
    log_dir="logs/",           # Optional file logging directory
    log_file="train.log",      # Optional log file name
)
```

Loggers use Rich-formatted console output with color-coded levels. File logs use a detailed format: `[2024-12-01 14:30:05.123 UTC] [INFO ] [llm_forge.training] message`.

---

### diagnose_error

Error recovery engine that analyzes exceptions and provides actionable suggestions.

```python
from llm_forge.utils.error_recovery import diagnose_error, ErrorDiagnosis

try:
    trainer.train(...)
except Exception as e:
    diagnosis: ErrorDiagnosis = diagnose_error(e)
    print(f"Error type: {diagnosis.error_type}")
    for suggestion in diagnosis.suggestion_texts:
        print(f"  - {suggestion}")
    if diagnosis.auto_fixable:
        print(f"Auto-fix: {diagnosis.auto_fix_description}")
```

**Recognized error patterns:**

| Error Type                    | Example Trigger                             | Suggestions                                          |
|-------------------------------|---------------------------------------------|------------------------------------------------------|
| Out of Memory                 | `CUDA out of memory`                        | Reduce batch size, enable gradient checkpointing, use QLoRA |
| Training Instability          | `NaN`, `Inf`, loss divergence               | Reduce learning rate, increase warmup, check data     |
| Import/Dependency             | `ModuleNotFoundError`                       | Install missing package                               |
| CUDA/GPU                      | `CUDA error`, driver mismatch               | Check CUDA version, update drivers                    |
| Data Format                   | Column mismatch, empty dataset              | Check column names, verify data format                |
| Tokenizer                     | Missing pad token, special token issues     | Set pad token, check tokenizer config                 |

---

### gpu_utils

GPU introspection and memory estimation for the CLI `hardware` command.

```python
from llm_forge.utils.gpu_utils import get_gpu_info, SystemGPUInfo

info: SystemGPUInfo = get_gpu_info()
print(f"GPUs: {info.gpu_count}")
print(f"Total VRAM: {info.total_vram_gb:.1f} GB")
print(f"Free VRAM: {info.total_free_vram_gb:.1f} GB")
print(f"CUDA: {info.cuda_version}")
print(f"MPS available: {info.mps_available}")

for gpu in info.gpus:
    print(f"  [{gpu.index}] {gpu.name}: {gpu.total_memory_gb:.1f} GB "
          f"({gpu.free_memory_gb:.1f} GB free, {gpu.utilization_str} util)")
```

The `SystemGPUInfo` dataclass provides:

| Property              | Type            | Description                        |
|-----------------------|-----------------|------------------------------------|
| `cuda_available`      | `bool`          | CUDA backend available             |
| `cuda_version`        | `str or None`   | CUDA runtime version               |
| `torch_version`       | `str or None`   | PyTorch version                    |
| `gpu_count`           | `int`           | Number of GPUs                     |
| `gpus`                | `list[GPUInfo]` | Per-GPU details (live memory stats)|
| `mps_available`       | `bool`          | Apple MPS backend available        |
| `total_vram_gb`       | `float`         | Total VRAM across all GPUs         |
| `total_free_vram_gb`  | `float`         | Total free VRAM across all GPUs    |

---

## Enums

All enums are importable from `llm_forge.config.schema`:

```python
from llm_forge.config.schema import (
    TrainingMode,       # lora, qlora, full, pretrain, dpo, orpo, grpo
    DataFormat,         # alpaca, sharegpt, completion, custom
    PrecisionMode,      # fp32, fp16, bf16, fp8, int8, int4
    DeduplicationTier,  # exact, fuzzy, semantic
    QualityPreset,      # permissive, balanced, strict
)
```

| Enum                | Values                                            |
|---------------------|---------------------------------------------------|
| `TrainingMode`      | `lora`, `qlora`, `full`, `pretrain`, `dpo`, `orpo`, `grpo` |
| `DataFormat`        | `alpaca`, `sharegpt`, `completion`, `custom`       |
| `PrecisionMode`     | `fp32`, `fp16`, `bf16`, `fp8`, `int8`, `int4`     |
| `DeduplicationTier` | `exact`, `fuzzy`, `semantic`                       |
| `QualityPreset`     | `permissive`, `balanced`, `strict`                 |

---

## CLI Commands

All CLI commands are accessible via the `llm-forge` entry point (powered by Typer).

| Command                                    | Description                                |
|--------------------------------------------|--------------------------------------------|
| `llm-forge init <name> --template <tpl>`   | Initialize project with config template    |
| `llm-forge validate <config.yaml>`         | Validate a configuration file              |
| `llm-forge train --config <config.yaml>`   | Run training pipeline                      |
| `llm-forge train --config <cfg> --dry-run` | Preview training plan without training     |
| `llm-forge train --config <cfg> --no-auto-optimize` | Skip hardware auto-optimization |
| `llm-forge eval --config <cfg> --model-path <p>` | Evaluate a trained model            |
| `llm-forge serve --config <cfg> --model-path <p>` | Launch serving backend             |
| `llm-forge export --config <cfg> --model-path <p> --format <fmt>` | Export model    |
| `llm-forge clean --config <config.yaml>`   | Run data cleaning pipeline standalone      |
| `llm-forge synthetic --config <cfg>`       | Generate synthetic training data           |
| `llm-forge push --model-path <p> --repo-id <r>` | Push model to HuggingFace Hub       |
| `llm-forge hardware`                       | Detect and display hardware profile        |
| `llm-forge presets`                        | List available config presets              |
| `llm-forge rag build --config <cfg>`       | Build RAG vector index                     |
| `llm-forge rag query --config <cfg>`       | Interactive RAG query                      |
| `llm-forge ui`                             | Launch Gradio training dashboard           |
| `llm-forge ui --desktop`                   | Launch native desktop window (pywebview)   |
| `llm-forge --version`                      | Show version                               |

**Init templates:** `lora`, `qlora`, `pretrain`, `rag`, `full`

---

## Dependencies

### Core (always installed)

| Package          | Min Version | Purpose                          |
|------------------|-------------|----------------------------------|
| `transformers`   | `>=4.45`    | Model loading, training          |
| `torch`          | `>=2.4`     | Tensor computation               |
| `peft`           | `>=0.13`    | LoRA/QLoRA adapters              |
| `accelerate`     | `>=1.0`     | Distributed training             |
| `trl`            | `>=0.12`    | SFT, DPO, ORPO, GRPO trainers   |
| `datasets`       | `>=3.0`     | Dataset loading and processing   |
| `bitsandbytes`   | `>=0.44`    | Quantization                     |
| `pydantic`       | `>=2.0`     | Configuration schema             |
| `typer`          | --          | CLI framework                    |
| `rich`           | --          | Terminal UI and logging          |
| `pyyaml`         | --          | YAML parsing                     |

### Optional Extras

| Extra          | Install Command                       | Packages                                      |
|----------------|---------------------------------------|-----------------------------------------------|
| `rag`          | `pip install llm-forge[rag]`          | ChromaDB, sentence-transformers, rank-bm25    |
| `serve`        | `pip install llm-forge[serve]`        | Gradio, FastAPI, uvicorn, vLLM                |
| `eval`         | `pip install llm-forge[eval]`         | lm-eval-harness, rouge-score, nltk            |
| `cleaning`     | `pip install llm-forge[cleaning]`     | ftfy, presidio, detoxify, spaCy, pymupdf      |
| `distributed`  | `pip install llm-forge[distributed]`  | DeepSpeed, Transformer Engine                 |
| `desktop`      | `pip install llm-forge[desktop]`      | pywebview                                     |
| `dev`          | `pip install llm-forge[dev]`          | pytest, ruff, mypy, hypothesis                |
| `all`          | `pip install llm-forge[all]`          | Everything above                              |
