"""Pipeline DAG construction for staged execution.

Builds a directed acyclic graph of pipeline stages where each stage
is a callable node with declared dependencies and per-stage configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("pipeline.dag_builder")


# ---------------------------------------------------------------------------
# Pipeline stage dataclass
# ---------------------------------------------------------------------------


@dataclass
class PipelineStage:
    """A single node in the pipeline DAG.

    Attributes
    ----------
    name : str
        Unique identifier for the stage.
    callable : Callable
        The function to execute for this stage.  Receives the pipeline
        context dict as its sole argument and must return an updated context.
    dependencies : list[str]
        Names of stages that must complete before this one runs.
    config : dict
        Stage-specific configuration extracted from the master config.
    enabled : bool
        Whether this stage should be executed.
    description : str
        Human-readable description of what this stage does.
    """

    name: str
    callable: Callable[..., Any]
    dependencies: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""


# ---------------------------------------------------------------------------
# Default stage implementations
# ---------------------------------------------------------------------------


def _stage_data_loading(context: dict[str, Any]) -> dict[str, Any]:
    """Load raw data from the configured source."""
    from llm_forge.data.loader import DataLoader

    config = context["config"]
    data_cfg = config.data

    loader = DataLoader(
        path=data_cfg.train_path,
        streaming=data_cfg.streaming,
        num_workers=data_cfg.num_workers,
        max_samples=data_cfg.max_samples,
        seed=data_cfg.seed,
    )

    dataset = loader.load()
    context["raw_dataset"] = dataset
    logger.info("Data loading complete: %d samples", len(dataset))

    # Load eval data if specified
    if data_cfg.eval_path is not None:
        eval_loader = DataLoader(
            path=data_cfg.eval_path,
            streaming=False,
            num_workers=data_cfg.num_workers,
            seed=data_cfg.seed,
        )
        context["raw_eval_dataset"] = eval_loader.load()
        logger.info("Eval data loaded: %d samples", len(context["raw_eval_dataset"]))

    return context


def _stage_cleaning(context: dict[str, Any]) -> dict[str, Any]:
    """Run the data-cleaning pipeline on the loaded dataset."""
    config = context["config"]
    cleaning_cfg = config.data.cleaning

    if not cleaning_cfg.enabled:
        logger.info("Data cleaning disabled, skipping")
        context["cleaned_dataset"] = context["raw_dataset"]
        return context

    dataset = context["raw_dataset"]
    original_len = len(dataset)

    # Determine the text field for cleaning based on dataset columns
    text_field = "text"
    if "text" not in dataset.column_names:
        # For structured formats (alpaca, etc.), use the output field
        # or fall back to the first text-like column
        data_cfg = config.data
        if data_cfg.output_field and data_cfg.output_field in dataset.column_names:
            text_field = data_cfg.output_field
        elif data_cfg.input_field and data_cfg.input_field in dataset.column_names:
            text_field = data_cfg.input_field
        elif dataset.column_names:
            text_field = dataset.column_names[0]

    try:
        from llm_forge.data.cleaning import CleaningPipeline

        pipeline = CleaningPipeline(
            config=cleaning_cfg,
            text_field=text_field,
        )
        dataset, stats = pipeline.run(dataset)
        context["cleaning_stats"] = stats
        logger.info(
            "Data cleaning complete: %d -> %d samples (removed %d)",
            original_len,
            len(dataset),
            original_len - len(dataset),
        )
    except ImportError as exc:
        logger.warning("Cleaning pipeline not available: %s", exc)
        logger.info("Skipping cleaning, passing raw data through")

    context["cleaned_dataset"] = dataset
    return context


def _stage_preprocessing(context: dict[str, Any]) -> dict[str, Any]:
    """Format and tokenize the cleaned dataset for training."""
    from llm_forge.data.preprocessor import DataPreprocessor

    config = context["config"]
    data_cfg = config.data

    preprocessor = DataPreprocessor(
        format_type=data_cfg.format,
        input_field=data_cfg.input_field,
        output_field=data_cfg.output_field,
        context_field=data_cfg.context_field,
        system_prompt=data_cfg.system_prompt,
        max_seq_length=config.model.max_seq_length,
    )

    dataset = context.get("cleaned_dataset", context.get("raw_dataset"))
    formatted = preprocessor.format_dataset(dataset)

    # Split into train/eval if no separate eval set
    eval_dataset = context.get("raw_eval_dataset")
    if eval_dataset is not None:
        eval_formatted = preprocessor.format_dataset(eval_dataset)
        context["train_dataset"] = formatted
        context["eval_dataset"] = eval_formatted
    else:
        train_ds, eval_ds = preprocessor.split_dataset(
            formatted,
            test_size=data_cfg.test_size,
            seed=data_cfg.seed,
        )
        context["train_dataset"] = train_ds
        context["eval_dataset"] = eval_ds

    logger.info(
        "Preprocessing complete: train=%d, eval=%d",
        len(context["train_dataset"]),
        len(context["eval_dataset"]),
    )

    return context


def _stage_training(context: dict[str, Any]) -> dict[str, Any]:
    """Run model fine-tuning (PyTorch or MLX backend)."""
    config = context["config"]

    # Route to MLX backend if enabled
    if hasattr(config, "mlx") and config.mlx.enabled:
        return _stage_training_mlx(context)

    return _stage_training_pytorch(context)


def _stage_training_pytorch(context: dict[str, Any]) -> dict[str, Any]:
    """Run PyTorch-based model fine-tuning."""
    from llm_forge.training.finetuner import FineTuner

    config = context["config"]

    finetuner = FineTuner(config)

    # Setup model
    model, tokenizer = finetuner.setup_model()

    # Apply LoRA if needed
    mode = config.training.mode
    if mode in ("lora", "qlora"):
        model = finetuner.apply_lora(model)

    # Build callbacks
    callbacks = []
    try:
        from llm_forge.training.callbacks import build_callbacks

        callbacks = build_callbacks(config)
    except (ImportError, Exception) as exc:
        logger.debug("Could not build custom callbacks: %s", exc)

    # Inject stop callback if a stop_event was provided by the UI
    stop_event = context.get("stop_event")
    if stop_event is not None:
        from llm_forge.training.callbacks import StopTrainingCallback

        callbacks.append(StopTrainingCallback(stop_event))

    # Train
    train_dataset = context["train_dataset"]
    eval_dataset = context.get("eval_dataset")
    if eval_dataset is not None and len(eval_dataset) == 0:
        eval_dataset = None

    result = finetuner.train(
        model=model,
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    context["train_result"] = result
    context["model"] = model
    context["tokenizer"] = tokenizer
    context["finetuner"] = finetuner

    # Merge LoRA if requested
    if mode in ("lora", "qlora") and config.serving.merge_adapter:
        merged_path = finetuner.merge_and_save(model)
        context["merged_model_path"] = str(merged_path)
        logger.info("Adapter merged and saved to %s", merged_path)

    return context


def _stage_training_mlx(context: dict[str, Any]) -> dict[str, Any]:
    """Run MLX-based model fine-tuning on Apple Silicon."""
    from llm_forge.training.mlx_trainer import MLXTrainer

    config = context["config"]
    logger.info("Using MLX backend for training")

    trainer = MLXTrainer(config)

    # Setup model
    model, tokenizer = trainer.setup_model()

    # Apply LoRA / DoRA if needed
    if config.mlx.fine_tune_type in ("lora", "dora"):
        model = trainer.apply_lora(model)

    # Train
    train_dataset = context["train_dataset"]
    eval_dataset = context.get("eval_dataset")

    result = trainer.train(
        model=model,
        dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    context["train_result"] = result
    context["model"] = model
    context["tokenizer"] = tokenizer
    context["mlx_trainer"] = trainer

    # Fuse adapters if requested
    if config.mlx.fine_tune_type in ("lora", "dora") and config.mlx.fuse_after_training:
        fused_path = trainer.fuse_and_save(model)
        context["merged_model_path"] = str(fused_path)
        logger.info("MLX adapters fused and saved to %s", fused_path)

    return context


def _stage_alignment(context: dict[str, Any]) -> dict[str, Any]:
    """Run preference-based alignment training (DPO or ORPO)."""
    from llm_forge.training.alignment import AlignmentTrainer

    config = context["config"]
    training_cfg = config.training
    alignment_cfg = config.alignment

    alignment_trainer = AlignmentTrainer(config)
    model, _ref_model, tokenizer = alignment_trainer.setup_dpo()

    # Prepare preference dataset
    pref_dataset = context.get("preference_dataset")
    if pref_dataset is None:
        from llm_forge.data.loader import DataLoader

        loader = DataLoader(
            path=alignment_cfg.preference_dataset,
            streaming=False,
            seed=config.data.seed if hasattr(config.data, "seed") else 42,
        )
        pref_dataset = loader.load()

    pref_dataset = alignment_trainer.prepare_preference_dataset(
        pref_dataset,
        prompt_field=alignment_cfg.prompt_field,
        chosen_field=alignment_cfg.chosen_field,
        rejected_field=alignment_cfg.rejected_field,
    )

    eval_dataset = context.get("eval_dataset")

    if training_cfg.mode == "dpo":
        result = alignment_trainer.train_dpo(
            model=model,
            dataset=pref_dataset,
            eval_dataset=eval_dataset,
            beta=alignment_cfg.beta,
            loss_type=alignment_cfg.loss_type,
            max_length=alignment_cfg.max_length,
            max_prompt_length=alignment_cfg.max_prompt_length,
        )
    elif training_cfg.mode == "orpo":
        result = alignment_trainer.train_orpo(
            model=model,
            dataset=pref_dataset,
            eval_dataset=eval_dataset,
            beta=alignment_cfg.beta,
            max_length=alignment_cfg.max_length,
            max_prompt_length=alignment_cfg.max_prompt_length,
        )
    elif training_cfg.mode == "grpo":
        result = alignment_trainer.train_grpo(
            model=model,
            dataset=pref_dataset,
            eval_dataset=eval_dataset,
            beta=alignment_cfg.beta,
            num_generations=alignment_cfg.num_generations,
            max_completion_length=alignment_cfg.max_completion_length,
        )
    else:
        raise ValueError(f"Unsupported alignment mode: {training_cfg.mode}")

    context["train_result"] = result
    context["model"] = model
    context["tokenizer"] = tokenizer
    context["alignment_trainer"] = alignment_trainer

    return context


def _stage_evaluation(context: dict[str, Any]) -> dict[str, Any]:
    """Run post-training evaluation benchmarks."""
    config = context["config"]
    eval_cfg = config.evaluation

    if not eval_cfg.enabled:
        logger.info("Evaluation disabled, skipping")
        return context

    try:
        from llm_forge.evaluation.benchmarks import BenchmarkRunner

        model_path = context.get("merged_model_path", config.training.output_dir)

        runner = BenchmarkRunner(
            model_path=model_path,
            benchmarks=eval_cfg.benchmarks,
            num_fewshot=eval_cfg.num_fewshot,
            batch_size=eval_cfg.batch_size,
        )

        results = runner.run()
        context["eval_results"] = results

        if eval_cfg.generate_report:
            try:
                from llm_forge.evaluation.report_generator import ReportGenerator

                report_gen = ReportGenerator()
                report_path = report_gen.generate(
                    results=results,
                    output_dir=config.training.output_dir,
                )
                context["eval_report_path"] = str(report_path)
                logger.info("Evaluation report saved: %s", report_path)
            except (ImportError, Exception) as exc:
                logger.warning("Report generation failed: %s", exc)

        logger.info("Evaluation complete: %s", results)

    except ImportError as exc:
        logger.warning("Evaluation skipped (missing dependency): %s", exc)
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        context["eval_error"] = str(exc)

    # LLM-as-Judge evaluation (optional)
    if getattr(eval_cfg, "llm_judge", False):
        try:
            from llm_forge.evaluation.llm_judge import LLMJudge

            eval_dataset = context.get("eval_dataset")
            model = context.get("model")
            tokenizer = context.get("tokenizer")

            if eval_dataset is None or model is None:
                logger.warning("No eval_dataset or model for LLM-as-Judge, skipping")
            else:
                judge_model = eval_cfg.judge_model
                if judge_model is None:
                    judge_model = context.get("merged_model_path", config.training.output_dir)

                judge = LLMJudge(
                    judge_model=judge_model,
                    max_new_tokens=256,
                )

                # Generate responses from eval dataset
                n_samples = min(getattr(eval_cfg, "judge_samples", 50), len(eval_dataset))

                # Extract instructions and generate responses
                data_cfg = config.data
                input_field = data_cfg.input_field or "instruction"
                columns = eval_dataset.column_names

                if "text" in columns:
                    instructions = [
                        row["text"][:256] for row in eval_dataset.select(range(n_samples))
                    ]
                elif input_field in columns:
                    instructions = [
                        row[input_field] for row in eval_dataset.select(range(n_samples))
                    ]
                else:
                    instructions = [
                        row[columns[0]] for row in eval_dataset.select(range(n_samples))
                    ]

                # Generate responses with the trained model
                import torch as _torch

                responses = []
                model.eval()
                for inst in instructions:
                    inputs = tokenizer(
                        inst, return_tensors="pt", truncation=True, max_length=256
                    ).to(next(model.parameters()).device)
                    with _torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                    resp = tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                    responses.append(resp)

                judge_result = judge.evaluate(
                    instructions=instructions,
                    responses=responses,
                    criteria=getattr(eval_cfg, "judge_criteria", None),
                )

                context["judge_result"] = judge_result
                logger.info(
                    "LLM-as-Judge evaluation: %d samples, scores=%s",
                    judge_result.num_evaluated,
                    judge_result.mean_scores,
                )

        except ImportError as exc:
            logger.warning("LLM-as-Judge skipped (missing dependency): %s", exc)
        except Exception as exc:
            logger.warning("LLM-as-Judge failed: %s", exc)

    # Display terminal-based post-training summary (quality report card)
    try:
        from llm_forge.evaluation.training_summary import (
            TrainingSummary,
            display_training_summary,
        )

        summary = TrainingSummary(
            model_name=config.model.name,
            training_loss_start=context.get("training_loss_start", 0.0),
            training_loss_end=context.get("training_loss_end", 0.0),
            duration_seconds=context.get("training_duration_seconds", 0.0),
            benchmark_results=context.get("eval_results", {}),
            baseline_results=context.get("baseline_results", {}),
            eval_loss=context.get("eval_loss"),
            num_samples=context.get("num_train_samples", 0),
            training_method=config.training.mode,
        )
        display_training_summary(summary)
        context["training_summary"] = summary
    except ImportError:
        logger.debug("Training summary display unavailable (missing rich)")
    except Exception as exc:
        logger.warning("Training summary display failed: %s", exc)

    return context


def _stage_export(context: dict[str, Any]) -> dict[str, Any]:
    """Export model to the configured format."""
    config = context["config"]
    serving_cfg = config.serving

    if serving_cfg.export_format is None:
        logger.info("No export format configured, skipping export stage")
        return context

    from llm_forge.serving.export import ModelExporter

    model_path = context.get("merged_model_path", config.training.output_dir)
    export_format = serving_cfg.export_format
    output_base = config.training.output_dir

    logger.info("Exporting model: format=%s, source=%s", export_format, model_path)

    if export_format == "safetensors":
        export_path = ModelExporter.export_safetensors(
            model=model_path,
            output_path=f"{output_base}/export_safetensors",
        )
    elif export_format == "gguf":
        from llm_forge.serving.export import (
            _clean_chat_template_for_export,
            generate_modelfile,
        )

        # Strip {% generation %} markers before GGUF conversion
        _clean_chat_template_for_export(Path(model_path))

        quant = serving_cfg.gguf_quantization or "q4_0"
        gguf_dir = f"{output_base}/gguf"
        gguf_filename = f"{Path(output_base).name}-{quant.upper()}.gguf"
        export_path = ModelExporter.export_gguf(
            model_path=model_path,
            output_path=f"{gguf_dir}/{gguf_filename}",
            quantization=quant,
        )

        # Auto-generate Ollama Modelfile alongside the GGUF
        if serving_cfg.generate_modelfile:
            system_prompt = serving_cfg.ollama_system_prompt or getattr(
                config.data, "system_prompt", None
            )
            generate_modelfile(
                gguf_path=export_path,
                output_dir=gguf_dir,
                system_prompt=system_prompt,
                temperature=serving_cfg.inference_temperature,
                top_p=serving_cfg.inference_top_p,
                top_k=serving_cfg.inference_top_k,
                repeat_penalty=serving_cfg.inference_repeat_penalty,
                num_predict=serving_cfg.inference_num_predict,
                num_ctx=serving_cfg.inference_num_ctx,
            )
            logger.info(
                "Ollama deployment ready: cd %s && ollama create <name> -f Modelfile",
                gguf_dir,
            )
    elif export_format == "onnx":
        export_path = ModelExporter.export_onnx(
            model=model_path,
            output_path=f"{output_base}/export_onnx",
        )
    else:
        logger.warning("Export format '%s' not yet implemented", export_format)
        return context

    context["export_path"] = str(export_path)
    logger.info("Export complete: %s", export_path)

    return context


def _stage_refusal_augmentation(context: dict[str, Any]) -> dict[str, Any]:
    """Mix refusal examples into training data before SFT (R-Tuning)."""
    config = context["config"]

    if not hasattr(config, "refusal") or not config.refusal.enabled:
        logger.info("Refusal augmentation disabled, skipping")
        return context

    from llm_forge.data.refusal_augmentor import RefusalAugmentor

    refusal_cfg = config.refusal
    augmentor = RefusalAugmentor(
        refusal_ratio=refusal_cfg.refusal_ratio,
        refusal_responses=refusal_cfg.refusal_responses,
        seed=getattr(config.data, "seed", 42),
    )

    train_dataset = context.get("train_dataset")
    if train_dataset is None:
        logger.warning("No train_dataset in context, skipping refusal augmentation")
        return context

    len(train_dataset)
    context["train_dataset"] = augmentor.augment_dataset(train_dataset)

    logger.info(
        "Refusal augmentation complete: %d samples (%.1f%% refusals)",
        len(context["train_dataset"]),
        100.0 * refusal_cfg.refusal_ratio,
    )

    return context


def _stage_ifd_scoring(context: dict[str, Any]) -> dict[str, Any]:
    """Score and filter training data using IFD (Instruction-Following Difficulty)."""
    config = context["config"]

    if not hasattr(config, "ifd") or not config.ifd.enabled:
        logger.info("IFD scoring disabled, skipping")
        return context

    from llm_forge.data.ifd_scorer import IFDScorer

    ifd_cfg = config.ifd
    train_dataset = context.get("train_dataset")
    if train_dataset is None:
        logger.warning("No train_dataset in context, skipping IFD scoring")
        return context

    # Load base model for scoring (before fine-tuning)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading base model for IFD scoring: %s", config.model.name)
        tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        import torch as _torch

        _device_map = "auto" if _torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            torch_dtype="auto",
            device_map=_device_map,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:
        logger.error("Failed to load model for IFD scoring: %s", exc)
        return context

    # Extract instruction and response texts
    columns = train_dataset.column_names
    data_cfg = config.data
    input_field = data_cfg.input_field or "instruction"
    output_field = data_cfg.output_field or "output"

    # Handle different dataset formats
    if "text" in columns:
        # Completion format — split on common separators
        instructions = [row.get("text", "")[:256] for row in train_dataset]
        responses = [row.get("text", "")[256:] for row in train_dataset]
    elif input_field in columns and output_field in columns:
        instructions = [row[input_field] for row in train_dataset]
        responses = [row[output_field] for row in train_dataset]
    else:
        logger.warning(
            "Cannot identify instruction/response columns (%s) for IFD scoring, skipping",
            columns,
        )
        return context

    # Score
    scorer = IFDScorer(
        max_length=ifd_cfg.max_length,
        batch_size=ifd_cfg.batch_size,
    )

    _device = next(model.parameters()).device  # noqa: F841
    result = scorer.score_dataset(model, tokenizer, instructions, responses)

    logger.info(
        "IFD scoring complete: %d samples, mean=%.3f, median=%.3f",
        result.num_scored,
        result.mean_ifd,
        result.median_ifd,
    )

    context["ifd_result"] = result

    # Filter dataset
    original_len = len(train_dataset)
    context["train_dataset"] = scorer.filter_by_ifd(
        train_dataset, result.scores, select_ratio=ifd_cfg.select_ratio
    )
    logger.info(
        "IFD filtering: %d -> %d samples (kept %.0f%%)",
        original_len,
        len(context["train_dataset"]),
        100.0 * ifd_cfg.select_ratio,
    )

    # Free scoring model to reclaim memory before training
    del model, tokenizer
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return context


def _stage_iti_probing(context: dict[str, Any]) -> dict[str, Any]:
    """Discover truthfulness directions in attention heads (ITI probing)."""
    config = context["config"]

    if not hasattr(config, "iti") or not config.iti.enabled:
        logger.info("ITI probing disabled, skipping")
        return context

    from llm_forge.evaluation.iti_prober import ITIProber

    iti_cfg = config.iti

    model = context.get("model")
    tokenizer = context.get("tokenizer")
    if model is None or tokenizer is None:
        logger.error("Model/tokenizer not in context, cannot run ITI probing")
        return context

    prober = ITIProber(
        num_probing_samples=iti_cfg.num_probing_samples,
        num_heads=iti_cfg.num_heads,
        method=iti_cfg.method,
    )

    result = prober.probe(model, tokenizer, iti_cfg)

    context["iti_directions"] = result.directions
    context["iti_top_heads"] = result.top_heads
    context["iti_sigmas"] = result.sigmas
    context["iti_probe_result"] = result

    # Log summary
    top_accs = [result.probe_accuracies.get(h, 0.0) for h in result.top_heads[:10]]
    logger.info(
        "ITI probing complete: %d heads selected, top-10 mean accuracy=%.3f",
        len(result.top_heads),
        sum(top_accs) / max(len(top_accs), 1),
    )

    return context


def _stage_iti_baking(context: dict[str, Any]) -> dict[str, Any]:
    """Bake ITI directions into model weights as o_proj biases."""
    config = context["config"]

    if not hasattr(config, "iti") or not config.iti.enabled or not config.iti.bake_in:
        logger.info("ITI baking disabled, skipping")
        return context

    from llm_forge.serving.iti_baker import ITIBaker

    model = context.get("model")
    directions = context.get("iti_directions")
    top_heads = context.get("iti_top_heads")
    sigmas = context.get("iti_sigmas")

    if model is None or directions is None:
        logger.error("Missing model or ITI directions, cannot bake")
        return context

    baker = ITIBaker()
    model = baker.bake_interventions(
        model=model,
        directions=directions,
        top_heads=top_heads,
        alpha=config.iti.alpha,
        sigmas=sigmas,
    )

    context["model"] = model

    # Re-save the model with baked-in biases
    output_dir = config.training.output_dir
    merged_path = context.get("merged_model_path", output_dir)

    try:
        import os

        save_path = os.path.join(str(merged_path), "iti_baked")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer = context.get("tokenizer")
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)
        context["merged_model_path"] = save_path
        logger.info("ITI-baked model saved to %s", save_path)
    except Exception as exc:
        logger.warning("Could not save ITI-baked model: %s", exc)

    return context


def _stage_model_merging(context: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple models using linear, SLERP, or TIES strategy."""
    config = context["config"]

    if not hasattr(config, "merge") or not config.merge.enabled:
        logger.info("Model merging disabled, skipping")
        return context

    merge_cfg = config.merge
    if not merge_cfg.models:
        logger.warning("No models specified for merging, skipping")
        return context

    from llm_forge.serving.model_merger import ModelMerger

    output_path = merge_cfg.output_path
    if output_path is None:
        import os

        output_path = os.path.join(config.training.output_dir, "merged")

    merged_path = ModelMerger.merge_models(
        method=merge_cfg.method,
        model_paths=merge_cfg.models,
        output_path=output_path,
        weights=merge_cfg.weights or None,
        base_model=merge_cfg.base_model,
        slerp_t=merge_cfg.slerp_t,
        ties_density=merge_cfg.ties_density,
    )

    context["merged_model_path"] = str(merged_path)
    logger.info("Model merging complete: %s", merged_path)

    return context


# ============================================================================
# DAGBuilder
# ============================================================================


class DAGBuilder:
    """Build a directed acyclic graph of pipeline stages from configuration.

    The default pipeline stages are:

    1. ``data_loading`` -- Load raw data
    2. ``cleaning`` -- Clean and filter data
    3. ``preprocessing`` -- Format and tokenize for training
    4. ``refusal_augmentation`` -- Mix refusal examples (R-Tuning)
    5. ``ifd_scoring`` -- Score and filter by Instruction-Following Difficulty
    6. ``training`` -- Fine-tune the model (SFT / LoRA / QLoRA)
    7. ``alignment`` -- Preference alignment (DPO / ORPO)
    8. ``iti_probing`` -- Discover truthfulness directions (ITI)
    9. ``iti_baking`` -- Bake ITI directions into weights
    10. ``model_merging`` -- Merge models (linear/SLERP/TIES)
    11. ``evaluation`` -- Benchmark and evaluate
    12. ``export`` -- Export to deployment format
    """

    # Default stage definitions
    _DEFAULT_STAGES: list[dict[str, Any]] = [
        {
            "name": "data_loading",
            "callable": _stage_data_loading,
            "dependencies": [],
            "description": "Load raw training and evaluation data",
        },
        {
            "name": "cleaning",
            "callable": _stage_cleaning,
            "dependencies": ["data_loading"],
            "description": "Clean, filter, and deduplicate data",
        },
        {
            "name": "preprocessing",
            "callable": _stage_preprocessing,
            "dependencies": ["cleaning"],
            "description": "Format and prepare data for training",
        },
        {
            "name": "refusal_augmentation",
            "callable": _stage_refusal_augmentation,
            "dependencies": ["preprocessing"],
            "description": "Mix refusal examples into training data (R-Tuning)",
        },
        {
            "name": "ifd_scoring",
            "callable": _stage_ifd_scoring,
            "dependencies": ["refusal_augmentation"],
            "description": "Score and filter data by Instruction-Following Difficulty",
        },
        {
            "name": "training",
            "callable": _stage_training,
            "dependencies": ["ifd_scoring"],
            "description": "Fine-tune the model",
        },
        {
            "name": "alignment",
            "callable": _stage_alignment,
            "dependencies": ["training"],
            "description": "Run preference-based alignment training (DPO or ORPO)",
        },
        {
            "name": "iti_probing",
            "callable": _stage_iti_probing,
            "dependencies": ["alignment"],
            "description": "Discover truthfulness directions in attention heads",
        },
        {
            "name": "iti_baking",
            "callable": _stage_iti_baking,
            "dependencies": ["iti_probing"],
            "description": "Bake ITI directions into model weights as o_proj biases",
        },
        {
            "name": "model_merging",
            "callable": _stage_model_merging,
            "dependencies": ["iti_baking"],
            "description": "Merge multiple models (linear/SLERP/TIES)",
        },
        {
            "name": "evaluation",
            "callable": _stage_evaluation,
            "dependencies": ["training"],
            "description": "Evaluate model on benchmarks",
        },
        {
            "name": "export",
            "callable": _stage_export,
            "dependencies": ["training"],
            "description": "Export model to deployment format",
        },
    ]

    def build_dag(self, config: Any) -> list[PipelineStage]:
        """Build a pipeline DAG based on the provided configuration.

        Parameters
        ----------
        config : LLMForgeConfig
            The master configuration object.

        Returns
        -------
        list[PipelineStage]
            Topologically sorted list of pipeline stages.
        """
        stages: list[PipelineStage] = []

        for stage_def in self._DEFAULT_STAGES:
            enabled = self._is_stage_enabled(stage_def["name"], config)
            stage = PipelineStage(
                name=stage_def["name"],
                callable=stage_def["callable"],
                dependencies=stage_def["dependencies"],
                config=self._extract_stage_config(stage_def["name"], config),
                enabled=enabled,
                description=stage_def["description"],
            )
            stages.append(stage)

        # Topological sort
        sorted_stages = self._topological_sort(stages)

        # Validate the DAG
        self._validate_dag(sorted_stages)

        enabled_names = [s.name for s in sorted_stages if s.enabled]
        logger.info("Pipeline DAG built: %s", " -> ".join(enabled_names))

        return sorted_stages

    def _is_stage_enabled(self, stage_name: str, config: Any) -> bool:
        """Determine whether a stage should be enabled."""
        if stage_name == "data_loading":
            return True  # Always needed

        if stage_name == "cleaning":
            return (
                hasattr(config, "data")
                and hasattr(config.data, "cleaning")
                and config.data.cleaning.enabled
            )

        if stage_name == "preprocessing":
            return True  # Always needed

        if stage_name == "training":
            return True  # Core stage

        if stage_name == "evaluation":
            return hasattr(config, "evaluation") and config.evaluation.enabled

        if stage_name == "alignment":
            return (
                hasattr(config, "alignment")
                and config.alignment.preference_dataset is not None
                and config.training.mode in ("dpo", "orpo", "grpo")
            )

        if stage_name == "refusal_augmentation":
            return hasattr(config, "refusal") and config.refusal.enabled

        if stage_name == "ifd_scoring":
            return hasattr(config, "ifd") and config.ifd.enabled

        if stage_name == "iti_probing":
            return hasattr(config, "iti") and config.iti.enabled

        if stage_name == "iti_baking":
            return hasattr(config, "iti") and config.iti.enabled and config.iti.bake_in

        if stage_name == "model_merging":
            return (
                hasattr(config, "merge") and config.merge.enabled and len(config.merge.models) >= 2
            )

        if stage_name == "export":
            return hasattr(config, "serving") and config.serving.export_format is not None

        return True

    def _extract_stage_config(self, stage_name: str, config: Any) -> dict[str, Any]:
        """Extract stage-specific config as a plain dict."""
        if stage_name == "data_loading":
            return {"train_path": config.data.train_path}
        if stage_name == "cleaning":
            return (
                config.data.cleaning.model_dump()
                if hasattr(config.data.cleaning, "model_dump")
                else {}
            )
        if stage_name == "preprocessing":
            return {"format": config.data.format}
        if stage_name == "training":
            return {"mode": config.training.mode, "output_dir": config.training.output_dir}
        if stage_name == "evaluation":
            return {"benchmarks": config.evaluation.benchmarks}
        if stage_name == "alignment":
            if hasattr(config, "alignment"):
                return (
                    config.alignment.model_dump() if hasattr(config.alignment, "model_dump") else {}
                )
            return {}
        if stage_name == "refusal_augmentation":
            if hasattr(config, "refusal"):
                return config.refusal.model_dump() if hasattr(config.refusal, "model_dump") else {}
            return {}
        if stage_name == "ifd_scoring":
            if hasattr(config, "ifd"):
                return config.ifd.model_dump() if hasattr(config.ifd, "model_dump") else {}
            return {}
        if stage_name == "iti_probing":
            if hasattr(config, "iti"):
                return config.iti.model_dump() if hasattr(config.iti, "model_dump") else {}
            return {}
        if stage_name == "iti_baking":
            if hasattr(config, "iti"):
                return {"alpha": config.iti.alpha, "bake_in": config.iti.bake_in}
            return {}
        if stage_name == "model_merging":
            if hasattr(config, "merge"):
                return config.merge.model_dump() if hasattr(config.merge, "model_dump") else {}
            return {}
        if stage_name == "export":
            return {"format": config.serving.export_format}
        return {}

    def _topological_sort(self, stages: list[PipelineStage]) -> list[PipelineStage]:
        """Topologically sort stages respecting dependency ordering."""
        stage_map = {s.name: s for s in stages}
        visited: set[str] = set()
        result: list[PipelineStage] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            stage = stage_map.get(name)
            if stage is None:
                return
            for dep in stage.dependencies:
                visit(dep)
            result.append(stage)

        for stage in stages:
            visit(stage.name)

        return result

    def _validate_dag(self, stages: list[PipelineStage]) -> None:
        """Validate that the DAG has no cycles and all dependencies exist."""
        stage_names = {s.name for s in stages}

        for stage in stages:
            for dep in stage.dependencies:
                if dep not in stage_names:
                    raise ValueError(f"Stage '{stage.name}' depends on unknown stage '{dep}'")

        # Cycle detection via DFS colouring
        WHITE, GRAY, BLACK = 0, 1, 2
        colour: dict[str, int] = {s.name: WHITE for s in stages}
        stage_map = {s.name: s for s in stages}

        def dfs(name: str) -> None:
            colour[name] = GRAY
            for dep in stage_map[name].dependencies:
                if colour[dep] == GRAY:
                    raise ValueError(f"Cycle detected in pipeline DAG involving stage '{dep}'")
                if colour[dep] == WHITE:
                    dfs(dep)
            colour[name] = BLACK

        for stage in stages:
            if colour[stage.name] == WHITE:
                dfs(stage.name)
