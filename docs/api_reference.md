# API Reference

Complete Python API reference for the llm-forge platform.

---

## Configuration Schema

### `LLMForgeConfig`

Top-level configuration model. A single YAML file parsed into this model drives the entire pipeline.

```python
from llm_forge.config.schema import LLMForgeConfig

config = LLMForgeConfig(
    model=ModelConfig(name="meta-llama/Llama-3.2-1B"),
    data=DataConfig(train_path="tatsu-lab/alpaca"),
)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `ModelConfig` | Pretrained model selection and loading options. **Required.** |
| `lora` | `LoRAConfig` | LoRA / QLoRA adapter hyper-parameters. |
| `quantization` | `QuantizationConfig` | BitsAndBytes quantization options. |
| `data` | `DataConfig` | Dataset paths, format, and cleaning options. **Required.** |
| `training` | `TrainingConfig` | Core training hyper-parameters. |
| `distributed` | `DistributedConfig` | Distributed / multi-GPU training settings. |
| `evaluation` | `EvalConfig` | Post-training evaluation settings. |
| `rag` | `RAGConfig` | RAG pipeline configuration. |
| `serving` | `ServingConfig` | Model serving and export configuration. |

**Auto-configuration:** When `training.mode` is `"qlora"`, the validator automatically enables 4-bit quantization (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`).

---

### `ModelConfig`

```python
from llm_forge.config.schema import ModelConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | **Required** | HuggingFace model name or local path |
| `revision` | `Optional[str]` | `None` | Git revision (branch, tag, or commit SHA) |
| `trust_remote_code` | `bool` | `False` | Trust and execute code shipped inside the model repo |
| `torch_dtype` | `PrecisionMode` | `"bf16"` | Dtype for loading model weights |
| `max_seq_length` | `int` | `2048` | Maximum sequence length. Range: 128-131072 |
| `attn_implementation` | `Literal` | `"flash_attention_2"` | Attention kernel: `"eager"`, `"sdpa"`, `"flash_attention_2"` |
| `rope_scaling` | `Optional[Dict]` | `None` | RoPE scaling config, e.g. `{"type": "dynamic", "factor": 2.0}` |

---

### `LoRAConfig`

```python
from llm_forge.config.schema import LoRAConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `r` | `int` | `16` | LoRA rank. Range: 1-256 |
| `alpha` | `int` | `32` | LoRA scaling factor |
| `dropout` | `float` | `0.05` | LoRA dropout. Range: 0.0-0.5 |
| `target_modules` | `List[str]` | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Module patterns to apply LoRA to |
| `bias` | `Literal` | `"none"` | Bias training: `"none"`, `"all"`, `"lora_only"` |
| `task_type` | `Literal` | `"CAUSAL_LM"` | PEFT task type |
| `use_rslora` | `bool` | `False` | Enable Rank-Stabilized LoRA |
| `use_dora` | `bool` | `False` | Enable DoRA (Weight-Decomposed LoRA) |

---

### `QuantizationConfig`

```python
from llm_forge.config.schema import QuantizationConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `load_in_4bit` | `bool` | `False` | Load model in 4-bit precision |
| `load_in_8bit` | `bool` | `False` | Load model in 8-bit precision |
| `bnb_4bit_compute_dtype` | `PrecisionMode` | `"bf16"` | Compute dtype for 4-bit inference |
| `bnb_4bit_quant_type` | `Literal` | `"nf4"` | Quantization type: `"nf4"` or `"fp4"` |
| `bnb_4bit_use_double_quant` | `bool` | `True` | Enable nested/double quantization |

**Validation:** Cannot enable both `load_in_4bit` and `load_in_8bit` simultaneously.

---

### `DataConfig`

```python
from llm_forge.config.schema import DataConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `train_path` | `str` | **Required** | Path or HuggingFace dataset ID for training data |
| `eval_path` | `Optional[str]` | `None` | Path for evaluation data |
| `format` | `DataFormat` | `"alpaca"` | Data format: `"alpaca"`, `"sharegpt"`, `"completion"`, `"custom"` |
| `input_field` | `str` | `"instruction"` | Column name for user instruction/input |
| `output_field` | `str` | `"output"` | Column name for expected model output |
| `context_field` | `Optional[str]` | `"input"` | Column name for optional context |
| `system_prompt` | `Optional[str]` | `None` | System prompt prepended to every sample |
| `max_samples` | `Optional[int]` | `None` | Cap training samples (for debugging) |
| `test_size` | `float` | `0.05` | Eval split fraction. Range: (0.0, 1.0) |
| `seed` | `int` | `42` | Random seed |
| `streaming` | `bool` | `False` | Stream dataset instead of loading into RAM |
| `num_workers` | `int` | `4` | Number of data-loader workers |
| `cleaning` | `DataCleaningConfig` | Default | Data cleaning sub-config |

---

### `DataCleaningConfig`

```python
from llm_forge.config.schema import DataCleaningConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Master switch for data cleaning |
| `quality_preset` | `QualityPreset` | `"balanced"` | Quality filtering: `"permissive"`, `"balanced"`, `"strict"` |
| `unicode_fix` | `bool` | `True` | Apply ftfy unicode fixing |
| `language_filter` | `Optional[List[str]]` | `None` | ISO-639 language codes to keep, e.g. `["en"]` |
| `language_confidence_threshold` | `float` | `0.65` | Min FastText confidence. Range: 0.0-1.0 |
| `heuristic_filter` | `bool` | `True` | Enable rule-based quality heuristics |
| `min_word_count` | `int` | `5` | Min words per document |
| `max_word_count` | `int` | `100000` | Max words per document |
| `min_char_count` | `int` | `20` | Min characters per document |
| `max_char_count` | `int` | `5000000` | Max characters per document |
| `alpha_ratio_threshold` | `float` | `0.6` | Min alphabetic character fraction |
| `symbol_to_word_ratio` | `float` | `0.1` | Max symbol-to-word ratio |
| `max_duplicate_line_fraction` | `float` | `0.3` | Max duplicate line fraction |
| `max_duplicate_para_fraction` | `float` | `0.3` | Max duplicate paragraph fraction |
| `toxicity_filter` | `bool` | `False` | Enable toxicity scoring |
| `toxicity_threshold` | `float` | `0.8` | Toxicity score threshold |
| `pii_redaction` | `bool` | `False` | Enable PII redaction |
| `pii_entities` | `List[str]` | `["PERSON", "EMAIL_ADDRESS", ...]` | Entity types to redact |
| `dedup_enabled` | `bool` | `True` | Enable deduplication |
| `dedup_tiers` | `List[DeduplicationTier]` | `["exact", "fuzzy"]` | Dedup strategies |
| `dedup_jaccard_threshold` | `float` | `0.85` | Jaccard threshold for fuzzy dedup |
| `dedup_num_perm` | `int` | `128` | MinHash permutations |
| `dedup_shingle_size` | `int` | `5` | N-gram size for MinHash |
| `semantic_dedup_enabled` | `bool` | `False` | Enable semantic deduplication |
| `semantic_dedup_threshold` | `float` | `0.95` | Cosine similarity threshold |
| `semantic_dedup_model` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | Embedding model |

---

### `TrainingConfig`

```python
from llm_forge.config.schema import TrainingConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `TrainingMode` | `"lora"` | Training strategy: `"lora"`, `"qlora"`, `"full"`, `"pretrain"`, `"dpo"` |
| `output_dir` | `str` | `"outputs"` | Output directory |
| `num_epochs` | `int` | `3` | Training epochs |
| `per_device_train_batch_size` | `int` | `4` | Micro-batch size per GPU (train) |
| `per_device_eval_batch_size` | `int` | `4` | Micro-batch size per GPU (eval) |
| `gradient_accumulation_steps` | `int` | `4` | Gradient accumulation steps |
| `learning_rate` | `float` | `2e-4` | Peak learning rate |
| `weight_decay` | `float` | `0.01` | L2 weight decay |
| `warmup_ratio` | `float` | `0.03` | Warmup fraction (ignored if `warmup_steps` is set) |
| `warmup_steps` | `Optional[int]` | `None` | Exact warmup steps (overrides `warmup_ratio`) |
| `lr_scheduler_type` | `Literal` | `"cosine"` | LR scheduler |
| `max_grad_norm` | `float` | `1.0` | Gradient clipping norm |
| `logging_steps` | `int` | `10` | Log every N steps |
| `eval_steps` | `Optional[int]` | `None` | Eval every N steps. `None` = every epoch |
| `eval_strategy` | `Literal` | `"epoch"` | When to eval: `"no"`, `"steps"`, `"epoch"` |
| `save_steps` | `int` | `500` | Checkpoint every N steps |
| `save_total_limit` | `int` | `3` | Max checkpoints to keep |
| `bf16` | `bool` | `True` | Enable BF16 mixed precision |
| `fp16` | `bool` | `False` | Enable FP16 mixed precision |
| `gradient_checkpointing` | `bool` | `False` | Trade compute for memory |
| `optim` | `str` | `"adamw_torch"` | Optimizer name |
| `group_by_length` | `bool` | `True` | Group samples by length |
| `report_to` | `List[str]` | `["wandb"]` | Experiment trackers |
| `resume_from_checkpoint` | `Optional[str]` | `None` | Path to checkpoint to resume from |
| `neftune_noise_alpha` | `Optional[float]` | `5.0` | NEFTune noise alpha (null to disable) |
| `label_smoothing_factor` | `float` | `0.1` | Label-smoothing coefficient (0.0 to disable) |
| `average_tokens_across_devices` | `bool` | `True` | Sync token counts for correct grad-accum loss |
| `use_unsloth` | `bool` | `False` | Use Unsloth accelerated kernels |

**Validation:** Cannot enable both `bf16` and `fp16` simultaneously.

---

### `DistributedConfig`

```python
from llm_forge.config.schema import DistributedConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable distributed training |
| `framework` | `Literal` | `"auto"` | Framework: `"auto"`, `"fsdp"`, `"deepspeed"`, `"megatron"` |
| `num_gpus` | `int` | `1` | Number of GPUs |
| `num_nodes` | `int` | `1` | Number of cluster nodes |
| `fsdp_sharding_strategy` | `Literal` | `"FULL_SHARD"` | FSDP sharding: `"FULL_SHARD"`, `"SHARD_GRAD_OP"`, `"NO_SHARD"`, `"HYBRID_SHARD"` |
| `deepspeed_stage` | `Literal` | `2` | DeepSpeed ZeRO stage: `0`, `1`, `2`, `3` |
| `deepspeed_offload` | `bool` | `False` | CPU offloading |
| `tensor_parallel_degree` | `int` | `1` | Tensor parallelism degree |
| `pipeline_parallel_degree` | `int` | `1` | Pipeline parallelism degree |
| `fp8_enabled` | `bool` | `False` | Enable FP8 compute (Hopper GPUs) |
| `fp8_format` | `Literal` | `"HYBRID"` | FP8 format: `"E4M3"` or `"HYBRID"` |
| `auto_micro_batch` | `bool` | `False` | Auto-find largest micro-batch |

---

### `EvalConfig`

```python
from llm_forge.config.schema import EvalConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Run benchmarks after training |
| `benchmarks` | `List[str]` | `["hellaswag", "arc_easy", "mmlu"]` | Benchmark task names |
| `custom_eval_path` | `Optional[str]` | `None` | Path to custom evaluation dataset |
| `num_fewshot` | `int` | `0` | Few-shot examples for benchmarks |
| `batch_size` | `int` | `8` | Inference batch size |
| `generate_report` | `bool` | `True` | Generate evaluation report |

---

### `RAGConfig`

```python
from llm_forge.config.schema import RAGConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable RAG pipeline |
| `knowledge_base_path` | `Optional[str]` | `None` | Path to knowledge base directory |
| `chunk_strategy` | `Literal` | `"recursive"` | Chunking: `"fixed"`, `"recursive"`, `"semantic"`, `"sentence"` |
| `chunk_size` | `int` | `512` | Target chunk size in tokens. Range: 64-8192 |
| `chunk_overlap` | `int` | `64` | Overlap between chunks |
| `embedding_model` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | Embedding model |
| `vectorstore` | `Literal` | `"chromadb"` | Backend: `"chromadb"`, `"faiss"`, `"qdrant"`, `"weaviate"` |
| `top_k` | `int` | `5` | Chunks to retrieve per query |
| `reranker_model` | `Optional[str]` | `None` | Cross-encoder for re-ranking |
| `hybrid_search` | `bool` | `False` | Combine dense + BM25 sparse retrieval |
| `similarity_threshold` | `float` | `0.7` | Minimum similarity to keep a chunk |

**Validation:** `chunk_overlap` must be less than `chunk_size`.

---

### `ServingConfig`

```python
from llm_forge.config.schema import ServingConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | `Literal` | `"gradio"` | Serving backend: `"gradio"`, `"fastapi"`, `"vllm"` |
| `host` | `str` | `"0.0.0.0"` | Host to bind |
| `port` | `int` | `7860` | Port number. Range: 1-65535 |
| `export_format` | `Optional[Literal]` | `None` | Export format: `"gguf"`, `"onnx"`, `"safetensors"`, `"awq"`, `"gptq"` |
| `gguf_quantization` | `Optional[str]` | `None` | GGUF quantization level, e.g. `"Q4_K_M"` |
| `merge_adapter` | `bool` | `True` | Merge LoRA adapter before serving/export |

---

### Enums

```python
from llm_forge.config.schema import (
    TrainingMode,      # lora, qlora, full, pretrain, dpo
    DataFormat,        # alpaca, sharegpt, completion, custom
    PrecisionMode,     # fp32, fp16, bf16, fp8, int8, int4
    DeduplicationTier, # exact, fuzzy, semantic
    QualityPreset,     # permissive, balanced, strict
)
```

---

## Configuration Validation

### `validate_config_file`

```python
from llm_forge.config.validator import validate_config_file

config = validate_config_file("config.yaml")
# Returns: LLMForgeConfig instance
# Raises: ConfigValidationError with detailed error messages
```

### `validate_config_dict`

```python
from llm_forge.config.validator import validate_config_dict

config = validate_config_dict({
    "model": {"name": "meta-llama/Llama-3.2-1B"},
    "data": {"train_path": "tatsu-lab/alpaca"},
})
```

### `load_preset`

```python
from llm_forge.config.validator import load_preset, list_presets

# List available presets
presets = list_presets()
# ['lora_default', 'qlora_default', 'full_finetune', 'pretrain_small']

# Load a preset
preset_dict = load_preset("lora_default")
```

---

## Data API

### `DataLoader`

```python
from llm_forge.data.loader import DataLoader

loader = DataLoader(
    path="tatsu-lab/alpaca",   # File, directory, URL, or HF dataset ID
    streaming=False,
    num_workers=4,
    max_samples=None,
    seed=42,
)

dataset = loader.load()          # Returns: datasets.Dataset
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `load()` | Auto-detect source and load | `Dataset` |
| `load_streaming()` | Load as streaming iterator | `Iterator[dict]` |

### `DataPreprocessor`

```python
from llm_forge.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(
    format_type="alpaca",       # "alpaca", "sharegpt", "completion", "custom"
    input_field="instruction",
    output_field="output",
    context_field="input",
    system_prompt=None,
    max_seq_length=2048,
)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `format_dataset(dataset)` | Convert to unified text format | `Dataset` |
| `format_for_chat_template(dataset, tokenizer)` | Format using model's chat template | `Dataset` |
| `tokenize_dataset(dataset, tokenizer, pack_sequences=False)` | Tokenize with label masking | `Dataset` |
| `split_dataset(dataset, test_size=0.1, seed=42)` | Train/eval split | `tuple[Dataset, Dataset]` |

### `DataMixer`

```python
from llm_forge.data.mixing import DataMixer

mixer = DataMixer(seed=42, temperature=1.0)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `mix_datasets(datasets, weights, total_samples)` | Mix datasets by weight | `Dataset` |
| `compute_optimal_weights(datasets, method)` | Compute mixing weights | `dict[str, float]` |
| `upsample_dataset(dataset, target_size)` | Upsample by repeating | `Dataset` |
| `downsample_dataset(dataset, target_size)` | Downsample by random selection | `Dataset` |

### `CleaningPipeline`

```python
from llm_forge.data.cleaning import CleaningPipeline

pipeline = CleaningPipeline(
    config=cleaning_config,     # DataCleaningConfig, dict, or None
    text_field="text",
)

cleaned_dataset, stats = pipeline.run(dataset)
print(stats.summary())
```

**Returns:** `tuple[Dataset, CleaningStats]`

### `SyntheticDataGenerator`

```python
from llm_forge.data.synthetic.generator import SyntheticDataGenerator

gen = SyntheticDataGenerator(
    teacher_model=None,
    temperature_range=(0.3, 0.9),
    max_pairs_per_chunk=3,
    seed=42,
)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `generate_from_dataset(dataset, text_field, num_samples)` | Generate pairs from dataset | `Dataset` |
| `generate_from_chunks(chunks, topics)` | Generate from pre-chunked text | `Dataset` |
| `load_teacher_model(model_name)` | Load teacher model for generation | `None` |

---

## Training API

### `Trainer`

The unified training orchestrator. Central entry point for all training workflows.

```python
from llm_forge.training.trainer import Trainer

trainer = Trainer(config=config, dry_run=False)
results = trainer.run()
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `run()` | Execute the full training pipeline | `dict[str, Any]` |

**Pipeline steps:**
1. Detect hardware (GPU count, VRAM, compute capability)
2. Auto-optimize settings (batch size, precision, attention)
3. Print Rich training plan
4. Dispatch to FineTuner, PreTrainer, or AlignmentTrainer
5. Load and preprocess dataset
6. Build callbacks
7. Train with progress display
8. Evaluate and export

### `FineTuner`

```python
from llm_forge.training.finetuner import FineTuner

finetuner = FineTuner(config)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `setup_model(config=None)` | Load base model with quantization | `tuple[PreTrainedModel, PreTrainedTokenizerBase]` |
| `apply_lora(model, config=None)` | Apply PEFT LoRA adapters | `PeftModel` |
| `train(model, dataset, config=None, eval_dataset=None, callbacks=None)` | Run SFT training | `TrainOutput` |
| `merge_and_save(model=None, output_dir=None)` | Merge LoRA and save | `Path` |

### `PreTrainer`

```python
from llm_forge.training.pretrainer import PreTrainer

pretrainer = PreTrainer(config)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `build_model(config=None, model_size=None, ...)` | Create model from scratch | `PreTrainedModel` |
| `train_tokenizer(corpus_path, vocab_size=32000, ...)` | Train BPE tokenizer | `PreTrainedTokenizerFast` |
| `train(model, dataset, config=None, eval_dataset=None, callbacks=None)` | Run causal LM pre-training | `TrainOutput` |

**Model size presets:** `"125M"`, `"350M"`, `"760M"`, `"1B"`

### `AlignmentTrainer`

```python
from llm_forge.training.alignment import AlignmentTrainer

aligner = AlignmentTrainer(config)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `setup_dpo(model=None, config=None)` | Load policy + reference models | `tuple[model, ref_model, tokenizer]` |
| `prepare_preference_dataset(dataset, ...)` | Validate and normalize columns | `Dataset` |
| `train_dpo(model, dataset, ...)` | Run DPO training | `TrainOutput` |
| `setup_reward_model(reward_model_name, config=None)` | Load reward model for RLHF | `PreTrainedModel` |
| `train_rlhf(model, dataset, ...)` | Run PPO-based RLHF | `dict[str, Any]` |

---

## Training Callbacks

```python
from llm_forge.training.callbacks import (
    WandBCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    GPUMonitorCallback,
    RichProgressCallback,
)
```

### `WandBCallback`

```python
WandBCallback(
    project="llm-forge",
    run_name=None,
    tags=None,
    log_model=False,
)
```

### `CheckpointCallback`

```python
CheckpointCallback(
    save_every_n_minutes=30.0,
    checkpoint_dir=None,
    max_checkpoints=5,
)
```

### `EarlyStoppingCallback`

```python
EarlyStoppingCallback(
    patience=3,
    min_delta=0.001,
    metric_name="eval_loss",
)
```

### `GPUMonitorCallback`

```python
GPUMonitorCallback(log_every_n_steps=50)
```

### `RichProgressCallback`

```python
RichProgressCallback()
```

All callbacks extend `transformers.TrainerCallback` and are compatible with the HuggingFace Trainer callback system.

---

## Evaluation API

### `BenchmarkRunner`

```python
from llm_forge.evaluation.benchmarks import BenchmarkRunner

runner = BenchmarkRunner(device=None, cache_dir=None)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `run_benchmarks(model_path, tasks=None, num_fewshot=None, batch_size=8, limit=None)` | Run lm-eval benchmarks | `dict[str, Any]` |
| `compare_models(base_path, finetuned_path, tasks=None, ...)` | Compare two models | `dict[str, Any]` |
| `save_results(results, output_path)` | Save results to JSON | `Path` |
| `list_tasks()` | List available benchmark tasks | `list[dict]` |

### `DomainEvaluator`

```python
from llm_forge.evaluation.domain_eval import DomainEvaluator

evaluator = DomainEvaluator(
    metrics=["exact_match", "f1", "accuracy"],
    input_field="input",
    output_field="output",
)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `load_dataset(path, input_field=None, output_field=None, max_samples=None)` | Load eval dataset | `list[EvalSample]` |
| `evaluate(model, tokenizer, eval_dataset, ...)` | Run full evaluation | `dict[str, Any]` |
| `evaluate_predictions(predictions, references, metrics=None)` | Evaluate pre-computed predictions | `dict[str, Any]` |

### `MetricsComputer`

```python
from llm_forge.evaluation.metrics import MetricsComputer

mc = MetricsComputer()
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `compute_perplexity(model, tokenizer, texts, ...)` | Compute perplexity | `dict` |
| `compute_bleu(predictions, references, max_n=4)` | Compute BLEU scores | `dict` |
| `compute_rouge(predictions, references, rouge_types=None)` | Compute ROUGE scores | `dict` |
| `compute_exact_match(predictions, references, normalize=True)` | Exact match accuracy | `dict` |
| `compute_f1(predictions, references)` | Token-level F1 | `dict` |
| `compute_accuracy(predictions, references, normalize_strings=True)` | Classification accuracy | `dict` |
| `compute_all(predictions, references, include=None)` | Compute multiple metrics | `dict` |

---

## Serving API

### `GradioApp`

```python
from llm_forge.serving.gradio_app import GradioApp

app = GradioApp(model_path="./outputs/my-model", config=None)
app.launch(host="0.0.0.0", port=7860, share=False)
```

### `FastAPIServer`

```python
from llm_forge.serving.fastapi_server import FastAPIServer

server = FastAPIServer(model_path="./outputs/my-model", config=None)
server.start(host="0.0.0.0", port=8000)
```

**REST Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |
| POST | `/generate` | Text generation (with optional SSE streaming) |
| POST | `/chat` | OpenAI-compatible chat completion (with optional SSE streaming) |

---

## Hardware Detection API

```python
from llm_forge.config.hardware_detector import detect_hardware, auto_optimize_config

# Detect available hardware
profile = detect_hardware()
# Returns: HardwareProfile with GPU info, VRAM, compute capabilities

# Auto-optimize a config based on detected hardware
optimized_config = auto_optimize_config(config)
```

---

## CLI Commands

All CLI commands are accessible via the `llm-forge` entry point.

| Command | Description |
|---------|-------------|
| `llm-forge init <name> --template <template>` | Initialize a project with a config template |
| `llm-forge validate <config.yaml>` | Validate a configuration file |
| `llm-forge train --config <config.yaml>` | Run training |
| `llm-forge train --config <config.yaml> --dry-run` | Preview training plan without training |
| `llm-forge eval --config <config.yaml> --model-path <path>` | Evaluate a trained model |
| `llm-forge serve --config <config.yaml> --model-path <path>` | Launch serving backend |
| `llm-forge export --config <config.yaml> --model-path <path> --format <fmt>` | Export model |
| `llm-forge clean --config <config.yaml>` | Run data cleaning pipeline |
| `llm-forge synthetic --config <config.yaml>` | Generate synthetic training data |
| `llm-forge push --model-path <path> --repo-id <repo>` | Push model to HuggingFace Hub |
| `llm-forge hardware` | Detect and display hardware profile |
| `llm-forge presets` | List available config presets |
| `llm-forge rag build --config <config.yaml>` | Build RAG vector index |
| `llm-forge rag query --config <config.yaml>` | Interactive RAG query |
| `llm-forge --version` | Show version |

**Available templates for `init`:** `lora`, `qlora`, `pretrain`, `rag`, `full`

---

## Dependencies

### Core (always installed)

| Package | Version | Purpose |
|---------|---------|---------|
| `transformers` | `>=4.45` | Model loading, training |
| `torch` | `>=2.4` | Tensor computation |
| `peft` | `>=0.13` | LoRA/QLoRA adapters |
| `accelerate` | `>=1.0` | Distributed training |
| `trl` | `>=0.12` | SFT and DPO trainers |
| `datasets` | `>=3.0` | Dataset loading |
| `bitsandbytes` | `>=0.44` | Quantization |
| `pydantic` | `>=2.0` | Configuration schema |
| `typer` | -- | CLI framework |
| `rich` | -- | Terminal UI |
| `pyyaml` | -- | YAML parsing |

### Optional Extras

| Extra | Install Command | Packages |
|-------|----------------|----------|
| `rag` | `pip install llm-forge[rag]` | ChromaDB, LlamaIndex, LangChain |
| `serve` | `pip install llm-forge[serve]` | Gradio, FastAPI, vLLM |
| `eval` | `pip install llm-forge[eval]` | lm-eval-harness, ROUGE, NLTK |
| `cleaning` | `pip install llm-forge[cleaning]` | ftfy, presidio, detoxify, spaCy |
| `distributed` | `pip install llm-forge[distributed]` | DeepSpeed, Transformer Engine |
| `dev` | `pip install llm-forge[dev]` | pytest, ruff, mypy |
| `all` | `pip install llm-forge[all]` | Everything |
