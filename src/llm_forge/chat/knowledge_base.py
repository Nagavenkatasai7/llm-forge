"""Deep knowledge base about LLM Forge architecture, configs, and best practices.

This module contains ALL the knowledge the LLM Forge manager needs to understand
the platform inside-out.  It is injected into the system prompt so Claude has
complete awareness of every config field, pipeline stage, data format, model
recommendation, common error, and security rule.
"""

FORGE_KNOWLEDGE = """
## LLM Forge Architecture Knowledge

### Project Structure
When LLM Forge is set up in a directory, it creates:
- configs/ --- YAML training configurations (reference configs + user configs)
- data/ --- User's training data (JSONL, CSV, TXT, PDF)
- examples/data/ --- Sample data in 3 formats (Alpaca, ShareGPT, completion)
- outputs/ --- Trained models, checkpoints, exports, GGUF files
- .llmforge/ --- Memory database (SQLite), project state
- config.yaml --- Active configuration file
- .gitignore --- Protects secrets, large files, model weights

### Supported Data Formats
1. **Alpaca** (recommended for beginners):
   {"instruction": "task description", "input": "optional context", "output": "expected response"}
   - Fields map to: input_field="instruction", output_field="output", context_field="input"
   - Great for single-turn instruction tuning

2. **ShareGPT** (for multi-turn conversations):
   {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
   - Supports system messages: {"from": "system", "value": "..."}
   - Best for chat/dialogue models

3. **Completion** (for pre-training or continued pre-training):
   Plain text, one document per line or continuous text
   - Use training.mode: "pretrain" with this format
   - No instruction/output separation

4. **Custom**: Map any column names using these config fields:
   - data.input_field: column containing user input (default: "instruction")
   - data.output_field: column containing expected output (default: "output")
   - data.context_field: column for optional context (default: "input")

### Pipeline Stages (12 stages, in execution order)
1. **data_loading** --- Load from file path, directory, URL, or HuggingFace Hub
   - Supports: JSONL, CSV, TXT, PDF, Parquet, Arrow, HuggingFace datasets
   - Config: data.train_path, data.eval_path, data.streaming, data.num_workers
2. **cleaning** --- Unicode fix, language filter, dedup, quality heuristics, PII, toxicity
   - Master switch: data.cleaning.enabled (default: true)
   - Presets: permissive, balanced (default), strict
   - Typically removes 50-70% of noisy/duplicate data
3. **preprocessing** --- Format conversion, tokenization, chat template application
   - Auto-detects format from data.format field
   - Applies system_prompt if configured
   - Splits into train/eval using data.test_size (default: 0.05)
4. **refusal_augmentation** --- Mix refusal examples for R-Tuning (anti-hallucination)
   - Config: refusal.enabled, refusal.refusal_ratio (default: 0.15)
   - Teaches model to say "I don't know" instead of hallucinating
5. **ifd_scoring** --- Score Instruction-Following Difficulty, filter by quality
   - Config: ifd.enabled, ifd.select_ratio (default: 0.5)
   - Keeps the most valuable training samples
6. **training** --- SFT with LoRA/QLoRA/full fine-tuning, or pretrain from scratch
   - Routes to PyTorch or MLX backend automatically
   - Applies LoRA adapters, builds callbacks, runs HuggingFace Trainer
   - Merges adapter into base model if serving.merge_adapter is true
7. **alignment** --- DPO/ORPO/GRPO preference optimization
   - Config: training.mode must be "dpo", "orpo", or "grpo"
   - Requires alignment.preference_dataset with prompt/chosen/rejected columns
8. **iti_probing** --- Discover truthfulness directions in attention heads
   - Config: iti.enabled, iti.num_probing_samples, iti.num_heads
   - Uses TruthfulQA dataset by default
9. **iti_baking** --- Bake anti-hallucination directions into model weights
   - Config: iti.bake_in (default: true when iti.enabled)
   - Zero inference cost --- directions become permanent o_proj biases
10. **model_merging** --- Merge models using linear, SLERP, or TIES strategy
    - Config: merge.enabled, merge.method, merge.models (list of paths)
    - TIES requires merge.base_model
11. **evaluation** --- Benchmarks (MMLU, GSM8K, ARC, HellaSwag, IFEval, TruthfulQA)
    - Config: evaluation.enabled, evaluation.benchmarks
    - Supports LLM-as-Judge and knowledge retention probes
    - Auto-generates HTML/Markdown report
12. **export** --- Safetensors, GGUF (for Ollama), ONNX
    - Config: serving.export_format ("gguf", "safetensors", "onnx")
    - GGUF auto-generates Ollama Modelfile with inference parameters

### Config Field Reference

#### model (ModelConfig) --- REQUIRED
- name: str --- HuggingFace model name or local path (REQUIRED)
- revision: str | None --- Git revision of model repo (default: None)
- trust_remote_code: bool --- Execute code from model repo (default: false)
- torch_dtype: str --- Weight precision: "fp32", "fp16", "bf16" (default: "bf16")
- max_seq_length: int --- Context window for training, 128-131072 (default: 2048)
- attn_implementation: str --- "eager", "sdpa", or "flash_attention_2" (default: "flash_attention_2")
- rope_scaling: dict | None --- RoPE scaling config (default: None)

#### lora (LoRAConfig)
- r: int --- LoRA rank, 1-256 (default: 16)
- alpha: int --- Scaling factor, effective scale = alpha/r (default: 32)
- dropout: float --- LoRA dropout, 0.0-0.5 (default: 0.05)
- target_modules: list[str] | str --- Module patterns to adapt (default: ["q_proj", "v_proj", "k_proj", "o_proj"])
  - Use "all-linear" for maximum adaptability (higher forgetting risk)
  - Attention-only is safest for knowledge preservation
- bias: str --- "none", "all", or "lora_only" (default: "none")
- task_type: str --- "CAUSAL_LM" (default), "SEQ_2_SEQ_LM", "TOKEN_CLS", "SEQ_CLS"
- use_rslora: bool --- Rank-Stabilized LoRA scaling (default: false)
- use_dora: bool --- Weight-Decomposed LoRA (default: false)

#### quantization (QuantizationConfig)
- load_in_4bit: bool --- 4-bit quantization (default: false)
- load_in_8bit: bool --- 8-bit quantization (default: false)
- bnb_4bit_compute_dtype: str --- Compute dtype for 4-bit (default: "bf16")
- bnb_4bit_quant_type: str --- "nf4" (default) or "fp4"
- bnb_4bit_use_double_quant: bool --- Nested quantization (default: true)
Note: Cannot enable both load_in_4bit and load_in_8bit simultaneously.

#### data (DataConfig) --- REQUIRED
- train_path: str --- Path or HF dataset ID (REQUIRED)
- eval_path: str | None --- Eval data path; if None, splits from train (default: None)
- format: str --- "alpaca" (default), "sharegpt", "completion", "custom"
- input_field: str --- Input column name (default: "instruction")
- output_field: str --- Output column name (default: "output")
- context_field: str | None --- Context column name (default: "input")
- system_prompt: str | None --- System prompt for every sample (default: None)
- max_samples: int | None --- Cap training samples (default: None = all)
- test_size: float --- Eval split fraction, 0.0-1.0 (default: 0.05)
- seed: int --- Random seed (default: 42)
- streaming: bool --- Stream instead of loading into RAM (default: false)
- num_workers: int --- Data loader workers (default: 4)
- cleaning: DataCleaningConfig --- Sub-config for data cleaning

#### data.cleaning (DataCleaningConfig)
- enabled: bool --- Master switch (default: true)
- quality_preset: str --- "permissive", "balanced" (default), "strict"
- unicode_fix: bool --- ftfy unicode repair (default: true)
- language_filter: list[str] | None --- ISO-639 codes, e.g. ["en"] (default: None)
- language_confidence_threshold: float --- Min confidence (default: 0.65)
- heuristic_filter: bool --- Rule-based quality checks (default: true)
- min_word_count: int --- Min words per sample (default: 5)
- max_word_count: int --- Max words per sample (default: 100000)
- min_char_count: int --- Min chars per sample (default: 20)
- max_char_count: int --- Max chars per sample (default: 5000000)
- alpha_ratio_threshold: float --- Min alphabetic fraction (default: 0.6)
- symbol_to_word_ratio: float --- Max symbol-to-word ratio (default: 0.1)
- max_duplicate_line_fraction: float --- Max dup lines in sample (default: 0.3)
- max_duplicate_para_fraction: float --- Max dup paragraphs (default: 0.3)
- toxicity_filter: bool --- Toxicity scoring (default: false, requires detoxify)
- toxicity_threshold: float --- Drop threshold (default: 0.8)
- pii_redaction: bool --- PII detection/redaction (default: false, requires presidio)
- pii_entities: list[str] --- Entity types to redact (default: PERSON, EMAIL, PHONE, etc.)
- dedup_enabled: bool --- Cross-document dedup (default: true)
- dedup_tiers: list[str] --- Strategies: "exact", "fuzzy", "semantic" (default: ["exact", "fuzzy"])
- dedup_jaccard_threshold: float --- Fuzzy dedup threshold (default: 0.85)
- dedup_num_perm: int --- MinHash permutations (default: 128)
- dedup_shingle_size: int --- N-gram size for MinHash (default: 5)
- semantic_dedup_enabled: bool --- Embedding-based dedup (default: false)
- semantic_dedup_threshold: float --- Cosine similarity threshold (default: 0.95)
- semantic_dedup_model: str --- Embedding model (default: "sentence-transformers/all-MiniLM-L6-v2")

#### training (TrainingConfig)
- mode: str --- "lora" (default), "qlora", "full", "pretrain", "dpo", "orpo", "grpo"
- output_dir: str --- Output directory (default: "outputs")
- num_epochs: int --- Training epochs (default: 1)
- per_device_train_batch_size: int --- Micro-batch size (default: 4)
- per_device_eval_batch_size: int --- Eval batch size (default: 4)
- gradient_accumulation_steps: int --- Grad accum steps (default: 4)
- learning_rate: float --- Peak LR (default: 2e-5)
- weight_decay: float --- L2 regularization (default: 0.01)
- warmup_ratio: float --- Fraction of steps for warmup (default: 0.03)
- warmup_steps: int | None --- Exact warmup steps, overrides ratio (default: None)
- lr_scheduler_type: str --- "cosine" (default), "linear", "constant", etc.
- max_grad_norm: float --- Gradient clipping (default: 1.0)
- logging_steps: int --- Log every N steps (default: 10)
- eval_steps: int | None --- Eval every N steps (default: None = every epoch)
- eval_strategy: str --- "epoch" (default), "steps", "no"
- save_steps: int --- Checkpoint every N steps (default: 500)
- save_total_limit: int --- Max checkpoints to keep (default: 3)
- bf16: bool --- bfloat16 mixed precision (default: true)
- fp16: bool --- float16 mixed precision (default: false)
- gradient_checkpointing: bool --- Trade compute for memory (default: false)
- optim: str --- Optimizer (default: "adamw_torch")
- group_by_length: bool --- Reduce padding waste (default: true)
- report_to: list[str] --- Experiment trackers (default: ["wandb"])
- resume_from_checkpoint: str | None --- Checkpoint path to resume (default: None)
- neftune_noise_alpha: float | None --- NEFTune noise, None=disabled (default: None)
- label_smoothing_factor: float --- Label smoothing (default: 0.0)
- completion_only_loss: bool | None --- Mask prompt tokens (default: true)
- assistant_only_loss: bool --- Train on assistant tokens only (default: true)
- average_tokens_across_devices: bool --- Sync token counts (default: true)
- use_unsloth: bool --- Unsloth accelerated kernels (default: false)
- pack_sequences: bool --- Pack short sequences together (default: false)

#### evaluation (EvalConfig)
- enabled: bool --- Run benchmarks (default: true)
- benchmarks: list[str] --- Benchmark names (default: ["hellaswag", "arc_easy", "mmlu", "truthfulqa_mc2", "ifeval"])
- custom_eval_path: str | None --- Custom eval script (default: None)
- num_fewshot: int --- Few-shot examples (default: 0)
- batch_size: int --- Eval batch size (default: 8)
- generate_report: bool --- Generate report (default: true)
- regression_check: bool --- Compare vs base (default: true)
- regression_threshold: float --- Max acceptable drop (default: -0.02)
- llm_judge: bool --- LLM-as-Judge eval (default: false)
- judge_model: str | None --- Judge model path (default: None = self-judge)
- judge_criteria: list[str] --- Evaluation criteria (default: ["helpfulness", "coherence"])
- judge_samples: int --- Number of samples (default: 50)
- retention_probes: bool --- Knowledge retention test (default: false)
- retention_threshold: float --- Min retention rate (default: 0.80)

#### serving (ServingConfig)
- backend: str --- "gradio" (default), "fastapi", "vllm"
- host: str --- Bind address (default: "0.0.0.0")
- port: int --- Server port (default: 7860)
- export_format: str | None --- "gguf", "safetensors", "onnx", "awq", "gptq" (default: None)
- gguf_quantization: str | None --- GGUF quant level e.g. "Q4_K_M" (default: None)
- merge_adapter: bool --- Merge LoRA before export (default: true)
- generate_modelfile: bool --- Auto-generate Ollama Modelfile (default: true)
- ollama_system_prompt: str | None --- Modelfile system prompt (default: None)
- inference_temperature: float --- Sampling temperature (default: 0.1)
- inference_top_p: float --- Nucleus sampling threshold (default: 0.9)
- inference_top_k: int --- Top-k sampling limit (default: 40)
- inference_repeat_penalty: float --- Repetition penalty (default: 1.1)
- inference_num_predict: int --- Max tokens per response (default: 256)
- inference_num_ctx: int --- KV-cache context window (default: 2048)

#### distributed (DistributedConfig)
- enabled: bool --- Enable distributed training (default: false)
- framework: str --- "auto" (default), "fsdp", "deepspeed", "megatron"
- num_gpus: int --- GPUs to use (default: 1)
- num_nodes: int --- Cluster nodes (default: 1)
- fsdp_sharding_strategy: str --- "FULL_SHARD" (default), "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"
- deepspeed_stage: int --- ZeRO stage 0-3 (default: 2)
- deepspeed_offload: bool --- CPU offload (default: false)
- tensor_parallel_degree: int --- Megatron TP (default: 1)
- pipeline_parallel_degree: int --- Megatron PP (default: 1)
- fp8_enabled: bool --- FP8 compute (default: false)
- fp8_format: str --- "E4M3" or "HYBRID" (default: "HYBRID")
- auto_micro_batch: bool --- Auto-find batch size (default: false)

#### iti (ITIConfig) --- Inference-Time Intervention
- enabled: bool --- Enable ITI probing and baking (default: false)
- probing_dataset: str --- Dataset for probing (default: "truthful_qa")
- num_probing_samples: int --- Probing samples (default: 500)
- num_heads: int --- Top-K attention heads (default: 48)
- alpha: float --- Intervention strength (default: 15.0)
- method: str --- "center_of_mass" (default) or "linear_probe"
- bake_in: bool --- Bake into weights (default: true)

#### refusal (RefusalConfig) --- R-Tuning
- enabled: bool --- Enable refusal augmentation (default: false)
- refusal_ratio: float --- Fraction to replace with refusals (default: 0.15)
- refusal_responses: list[str] --- Pool of refusal templates

#### ifd (IFDConfig) --- Instruction-Following Difficulty
- enabled: bool --- Enable IFD scoring (default: false)
- select_ratio: float --- Fraction to keep after filtering (default: 0.5)
- batch_size: int --- Scoring batch size (default: 4)
- max_length: int --- Max sequence length for scoring (default: 512)

#### merge (MergeConfig)
- enabled: bool --- Enable model merging (default: false)
- method: str --- "linear" (default), "slerp", "ties"
- models: list[str] --- Model paths to merge
- weights: list[float] --- Per-model weights (empty = equal)
- base_model: str | None --- Base model for TIES (required for TIES)
- slerp_t: float --- SLERP interpolation 0.0-1.0 (default: 0.5)
- ties_density: float --- TIES trimming density (default: 0.5)
- output_path: str | None --- Save path (default: output_dir/merged)

#### alignment (AlignmentConfig)
- preference_dataset: str | None --- Path to preference data (default: None)
- prompt_field: str --- Prompt column (default: "prompt")
- chosen_field: str --- Chosen response column (default: "chosen")
- rejected_field: str --- Rejected response column (default: "rejected")
- beta: float --- KL penalty coefficient (default: 0.1)
- max_prompt_length: int --- Max prompt tokens (default: 512)
- max_length: int --- Max total tokens (default: 1024)
- loss_type: str --- "sigmoid" (default), "hinge", "ipo", "kto_pair"
- num_generations: int --- GRPO completions per prompt (default: 4)
- max_completion_length: int --- GRPO max completion tokens (default: 256)

#### mlx (MLXConfig) --- Apple Silicon native training
- enabled: bool --- Use MLX backend (default: false)
- fine_tune_type: str --- "lora" (default), "dora", "full"
- num_layers: int --- Layers to adapt, -1=all (default: 16)
- lora_rank: int --- LoRA rank (default: 8)
- lora_scale: float --- LoRA scaling factor (default: 20.0)
- lora_dropout: float --- LoRA dropout (default: 0.0)
- iters: int --- Training iterations (default: 1000)
- batch_size: int --- Batch size (default: 4)
- learning_rate: float --- Peak LR (default: 1e-5)
- optimizer: str --- "adam" (default), "adamw", "sgd", "adafactor"
- max_seq_length: int --- Max sequence length (default: 2048)
- grad_checkpoint: bool --- Gradient checkpointing (default: false)
- grad_accumulation_steps: int --- Grad accum steps (default: 1)
- steps_per_report: int --- Log every N steps (default: 10)
- steps_per_eval: int --- Eval every N steps (default: 200)
- steps_per_save: int --- Save every N steps (default: 100)
- mask_prompt: bool --- Train on completions only (default: true)
- lr_schedule: str | None --- LR schedule (default: "cosine_decay")
- warmup_steps: int --- Warmup steps (default: 0)
- adapter_path: str --- Adapter save dir (default: "adapters")
- fuse_after_training: bool --- Fuse adapters (default: true)

#### compute (ComputeConfig)
- backend: str --- "local" (default), "slurm", "aws", "gcp", "azure", "lambda", "runpod", "ssh"
- ssh: SSHConfig | None --- SSH connection (required for slurm/ssh backends)
  - host, user, key_path, remote_dir, sync_exclude
- slurm: SLURMConfig | None --- SLURM settings
  - partition, qos, gres, cpus_per_task, mem_gb, time_limit, exclude_nodes, modules, conda_env, conda_prefix, extra_env, extra_sbatch_flags
- cloud: CloudGPUConfig | None --- Cloud GPU settings
  - instance_type, region, gpu_type, num_gpus, disk_gb, spot, max_price_per_hour, auto_shutdown, docker_image, setup_commands
- sync_code: bool --- Rsync code to remote (default: true)
- stream_logs: bool --- Stream training logs (default: true)
- pull_outputs: bool --- Pull results back (default: true)
- local_output_dir: str | None --- Local output directory (default: None)

#### mac (MacConfig) --- macOS optimizations
- smart_memory: bool --- Auto memory management (default: true)
- memory_pressure_threshold: float --- RAM trigger threshold (default: 0.85)
- thermal_aware: bool --- Detect thermal throttling (default: true)
- thermal_pause_seconds: int --- Pause duration on throttle (default: 30)
- battery_aware: bool --- Pause on low battery (default: true)
- min_battery_pct: int --- Min battery level (default: 20)
- mps_high_watermark_ratio: float --- MPS memory limit (default: 0.0 = no limit)

### Model Selection Guide
| VRAM Available   | Recommended Model                    | Training Mode | Notes                        |
|-----------------|--------------------------------------|---------------|-------------------------------|
| No GPU (CPU)    | HuggingFaceTB/SmolLM2-135M          | QLoRA         | CPU-only, very slow          |
| 8 GB            | unsloth/Llama-3.2-1B-Instruct       | QLoRA         | 4-bit quantization required  |
| 12 GB           | unsloth/Llama-3.2-1B-Instruct       | LoRA          | Full precision LoRA          |
| 16-24 GB        | meta-llama/Llama-3.2-3B-Instruct    | LoRA          | Ideal consumer GPU range     |
| 24+ GB          | microsoft/Phi-3-mini-4k-instruct    | LoRA          | 3.8B params, strong          |
| 40+ GB          | meta-llama/Llama-3.1-8B-Instruct    | LoRA          | Best quality for 1 GPU       |
| 80+ GB (A100)   | meta-llama/Llama-3.1-8B-Instruct    | Full or LoRA  | Can do full fine-tune        |
| Apple Silicon   | unsloth/Llama-3.2-1B-Instruct       | LoRA (MPS)    | Or use MLX backend           |
| Apple MLX       | mlx-community/Llama-3.2-1B-Instruct | MLX LoRA      | Native Apple Silicon         |

### Training Best Practices (from production experience)
- **LoRA rank 8-16** for domain adaptation (higher ranks = more forgetting risk)
- **Target attention-only modules** (q_proj, k_proj, v_proj, o_proj) on small datasets (<20K samples)
- **Learning rate 1e-5 to 5e-5** for Instruct models; higher values cause catastrophic forgetting
- **Single epoch** on small datasets (<20K) to avoid overfitting
- **Disable NEFTune** on datasets with fewer than 10K samples (amplifies forgetting)
- **assistant_only_loss: true** is critical --- trains only on response tokens, prevents system-prompt regurgitation
- **Always enable data cleaning** --- typically removes 50-70% noise and duplicates
- **Use Instruct models** as base (not raw base models) for instruction-tuning tasks
- **max_seq_length: 2048** supports 4-6 turn conversations; increase for longer context
- **gradient_checkpointing: true** when memory is tight (trades ~30% speed for ~50% memory savings)
- **pack_sequences: true** improves throughput 2-4x on variable-length instruction data
- **label_smoothing_factor: 0.1** only on large datasets (>50K); hurts MC benchmarks on small data
- For Ollama deployment: keep inference_temperature low (0.1-0.3), repeat_penalty at 1.1, num_predict at 256

### Common Errors and Fixes
| Error                                  | Cause                                          | Fix                                                                 |
|----------------------------------------|------------------------------------------------|----------------------------------------------------------------------|
| CUDA out of memory (OOM)               | Batch size or model too large for VRAM         | Reduce per_device_train_batch_size, enable gradient_checkpointing, use QLoRA, or use a smaller model |
| NaN loss during training               | Learning rate too high or data contains NaN/Inf | Reduce learning_rate to 1e-5, enable data cleaning, check for corrupt samples |
| Gibberish / random output              | Trained on wrong tokens (system/user)          | Set assistant_only_loss: true, ensure chat template has generation markers |
| Catastrophic forgetting                | Over-adapted to fine-tuning data               | Reduce LoRA rank to 8, target attention-only, lower learning_rate, use 1 epoch |
| Tokenizer pad_token errors             | Model has no pad token defined                 | Set pad_token = eos_token (done automatically by LLM Forge)         |
| "No module named transformers"         | Missing dependencies                           | Run: pip install llm-forge[train] or install_dependencies tool      |
| Config validation error                | Invalid field values or combinations           | Check schema constraints --- bf16/fp16 mutual exclusion, load_in_4bit/8bit exclusion |
| Slow training on Mac                   | Using CPU instead of MPS                       | Ensure torch.backends.mps.is_available(), set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 |
| "CUDA not available"                   | No GPU or wrong CUDA version                  | Install correct torch+CUDA version: pip install torch --index-url https://download.pytorch.org/whl/cu121 |
| Loss stuck at high value               | completion_only_loss: false with instruction data | Set completion_only_loss: true or assistant_only_loss: true          |
| Model repeats same phrase              | inference_repeat_penalty too low               | Increase repeat_penalty to 1.1-1.2 in serving config               |
| Multi-turn conversation fails          | Modelfile uses single-turn template            | Use range .Messages loop in Modelfile (auto-generated by export)    |
| Training loss 20-30 (expected 1-3)     | Training on all tokens including system/user   | Enable assistant_only_loss, verify chat template has generation markers |
| eval_steps ignored                     | eval_strategy set to "no"                      | Set eval_strategy: "steps" when using eval_steps                    |
| Both bf16 and fp16 enabled             | Config conflict                                | Enable only one: bf16 or fp16, not both                             |
| GGUF export fails                      | Missing llama.cpp tools                        | Install llama.cpp or use: pip install llama-cpp-python              |
| Ollama model gives empty responses     | Temperature too high for small model           | Set inference_temperature: 0.1, inference_num_predict: 256          |

### GGUF Quantization Guide
| Quantization | Size (1B model) | Quality    | Speed  | Recommendation                    |
|-------------|-----------------|------------|--------|-----------------------------------|
| Q2_K        | ~400 MB         | Poor       | Fast   | Not recommended, severe quality loss |
| Q3_K_M      | ~500 MB         | Fair       | Fast   | Acceptable for testing only       |
| Q4_0        | ~600 MB         | Good       | Fast   | Basic 4-bit, good tradeoff       |
| Q4_K_M      | ~650 MB         | Very Good  | Fast   | RECOMMENDED --- best quality/size balance |
| Q5_K_M      | ~750 MB         | Excellent  | Medium | When quality matters more than size |
| Q6_K        | ~850 MB         | Near-FP16  | Medium | Minimal quality loss              |
| Q8_0        | ~1.0 GB         | Excellent  | Slower | Highest quality quantization      |
| F16         | ~2.0 GB         | Lossless   | Slow   | Full precision, largest size      |

Use Q4_K_M for most deployments. Use Q5_K_M or Q6_K when running on hardware with
ample RAM and quality is paramount. Avoid Q2_K and Q3_K_S for production.

### Auto-Configuration Behavior
- When training.mode is "qlora", quantization is auto-configured:
  load_in_4bit=true, bnb_4bit_quant_type="nf4", double_quant=true
- The schema warns automatically about:
  - High effective learning rates (lr * alpha/r > 1e-3)
  - all-linear targets on small datasets
  - NEFTune on small datasets (<20K samples)
  - Multiple epochs on small datasets
  - Learning rates above 1e-3
- Config uses "extra": "forbid" --- unknown fields cause validation errors (catches typos)

### Security Rules
- Never store API keys, tokens, or passwords in YAML config files
- Never modify the user's existing data files without explicit permission
- Never write files outside the LLM Forge project directory
- Always validate configs before starting training (prevents silent misconfigurations)
- Protect .env files and secrets with .gitignore
- Never expose serving endpoints to the public internet without authentication
- PII redaction (data.cleaning.pii_redaction) should be enabled when training on user data
- Model weights downloaded from HuggingFace should be verified (trust_remote_code: false by default)
- SLURM configs should use exclude_nodes for known-bad nodes
- SSH keys should never be committed to version control
"""
