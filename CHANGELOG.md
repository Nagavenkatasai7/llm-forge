# Changelog

All notable changes to llm-forge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-19

### Added
- Initial public release of llm-forge
- YAML-first configuration system with Pydantic v2 validation (22 config classes, 183 fields)
- Universal data loader (JSONL, CSV, Parquet, PDF, DOCX, HTML, HuggingFace datasets)
- Full data cleaning pipeline (unicode fixing, language filtering, heuristic filters, quality classification, toxicity filtering, PII redaction, 3-tier deduplication)
- LoRA, QLoRA, and full fine-tuning with TRL SFTTrainer
- Pre-training from scratch with configurable Llama architecture
- Alignment training: DPO, ORPO, GRPO, and PPO-based RLHF
- Inference-Time Intervention (ITI) for anti-hallucination with probing and weight baking
- IFD (Instruction-Following Difficulty) scoring for data quality
- Refusal augmentation (R-Tuning) for safety training
- Model merging: linear interpolation, SLERP, and TIES-Merging
- Synthetic data generation with quality scoring
- Distributed training orchestration (FSDP, DeepSpeed ZeRO 0-3, Megatron-Core 3D parallelism)
- FP8 training support for Hopper GPUs via Transformer Engine
- Hardware auto-detection and config optimization for RTX 3090/4090, A100, H100, Apple Silicon
- lm-evaluation-harness benchmark integration (MMLU, GSM8K, ARC, HellaSwag, IFEval, etc.)
- LLM-as-Judge evaluation with configurable criteria
- Domain evaluation with custom datasets
- Knowledge retention probes (100 questions across 10 domains)
- RAG pipeline with hybrid retrieval (dense + BM25), cross-encoder reranking, ChromaDB/FAISS
- Gradio web dashboard with model browser, training UI, and chat interface
- FastAPI server with OpenAI-compatible API endpoints
- vLLM high-throughput serving integration
- Model export: safetensors, GGUF (16 quantization types), ONNX, AWQ
- Ollama Modelfile generation with multi-turn chat template support
- Typer CLI with 13 commands and Rich formatting
- Interactive training wizard for guided setup
- Apple Silicon support (MPS + MLX trainer)
- Mac-specific monitoring (thermal, battery, memory pressure)
- SLURM job generation for HPC clusters
- Training callbacks: Rich progress, GPU monitoring, W&B, early stopping, checkpointing
- Comprehensive error recovery with context-aware suggestions
- Security module: safetensors validation, credential masking
- 894 passing tests, 0 failures
- 20 ready-to-use YAML configs (quickstart, benchmarks, domain examples, cloud deployment)
- Example training data in Alpaca, ShareGPT, and completion formats
- GitHub Actions CI/CD (lint, test, build, PyPI release)

### Finance Specialist Model (v7)
- Trained and deployed to HuggingFace: Venkat9990/finance-specialist-v7
- Zero catastrophic forgetting: all benchmarks within 0-2% of base model
- Knowledge-preserving LoRA (r=8, attention-only, LR 1e-5)
- 97.7% token masking accuracy
- Available as safetensors (2.4 GB) and GGUF Q4_K_M (763 MB)
