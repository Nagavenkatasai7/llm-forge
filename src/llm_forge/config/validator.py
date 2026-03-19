"""YAML configuration loading, validation, and user-friendly error reporting.

This module provides the primary entry points for reading a YAML config file,
validating it against :class:`LLMForgeConfig`, and returning actionable error
messages when the file is malformed or contains invalid values.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llm_forge.config.schema import LLMForgeConfig

# ---------------------------------------------------------------------------
# Preset directory (relative to this package)
# ---------------------------------------------------------------------------

_PRESETS_DIR = Path(__file__).resolve().parent / "presets"

# ---------------------------------------------------------------------------
# Common-mistake suggestions
# ---------------------------------------------------------------------------

_FIELD_SUGGESTIONS: dict[str, str] = {
    "model_name": "Did you mean 'model.name'?",
    "model_path": "Did you mean 'model.name' (accepts both HF ids and local paths)?",
    "train_data": "Did you mean 'data.train_path'?",
    "dataset": "Did you mean 'data.train_path'?",
    "epochs": "Did you mean 'training.num_epochs'?",
    "lr": "Did you mean 'training.learning_rate'?",
    "batch_size": "Did you mean 'training.per_device_train_batch_size'?",
    "lora_r": "Did you mean 'lora.r'?",
    "lora_rank": "Did you mean 'lora.r'?",
    "lora_alpha": "Did you mean 'lora.alpha'?",
    "quantize": "Did you mean 'quantization.load_in_4bit' or 'quantization.load_in_8bit'?",
    "4bit": "Did you mean 'quantization.load_in_4bit'?",
    "8bit": "Did you mean 'quantization.load_in_8bit'?",
    "deepspeed": "Did you mean 'distributed.deepspeed_stage'?",
    "fsdp": "Did you mean 'distributed.fsdp_sharding_strategy'?",
    "gpu": "Did you mean 'distributed.num_gpus'?",
    "gpus": "Did you mean 'distributed.num_gpus'?",
    "eval": "Did you mean 'evaluation'?",
    "serve": "Did you mean 'serving'?",
    "export": "Did you mean 'serving.export_format'?",
    "project_name": "This field is not supported. Remove it from your config.",
    "project": "This field is not supported. Remove it from your config.",
    "name": "Did you mean 'model.name'?",
    "lr_scheduler": "Did you mean 'training.lr_scheduler_type'?",
    "scheduler": "Did you mean 'training.lr_scheduler_type'?",
    "deduplication_enabled": "Did you mean 'data.cleaning.dedup_enabled'?",
    "wandb_project": "wandb_project is not a config field. Set it via the WANDB_PROJECT env variable.",
    "report_to": "Did you mean 'training.report_to' (must be a list, e.g. ['wandb'])?",
    "export_format": "Did you mean 'serving.export_format' (must be a string, not a list)?",
    "output": "Did you mean 'training.output_dir'?",
    "output_dir": "Did you mean 'training.output_dir'?",
    "data_path": "Did you mean 'data.train_path'?",
    "train": "Did you mean 'training'?",
    "seed": "Did you mean 'data.seed'?",
}


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------


def _format_validation_errors(exc: ValidationError) -> str:
    """Turn a Pydantic ``ValidationError`` into readable, actionable messages."""
    lines: list[str] = []
    lines.append(f"Configuration validation failed with {exc.error_count()} error(s):\n")

    for idx, err in enumerate(exc.errors(), start=1):
        # Build dotted field path  --  e.g. "training.learning_rate"
        loc_parts: list[str] = []
        for part in err["loc"]:
            loc_parts.append(str(part))
        field_path = ".".join(loc_parts) if loc_parts else "<root>"

        err_type = err["type"]
        msg = err["msg"]

        lines.append(f"  [{idx}] {field_path}")
        lines.append(f"      Error : {msg}")
        lines.append(f"      Type  : {err_type}")

        # Provide a helpful hint when possible
        hint = _hint_for_error(field_path, err_type, err.get("ctx"))
        if hint:
            lines.append(f"      Hint  : {hint}")

        lines.append("")

    return "\n".join(lines)


def _hint_for_error(
    field_path: str,
    err_type: str,
    ctx: dict[str, Any] | None = None,
) -> str | None:
    """Return an optional hint string for common errors."""
    ctx = ctx or {}

    if err_type == "missing":
        return f"'{field_path}' is required.  Add it to your YAML file."

    if err_type == "extra_forbidden":
        leaf = field_path.rsplit(".", maxsplit=1)[-1]
        suggestion = _FIELD_SUGGESTIONS.get(leaf)
        if suggestion:
            return suggestion
        return "This field is not recognised.  Check for typos."

    if err_type in {"greater_than_equal", "less_than_equal", "greater_than", "less_than"}:
        limit = ctx.get("ge") or ctx.get("le") or ctx.get("gt") or ctx.get("lt")
        if limit is not None:
            return f"Value must satisfy the constraint: {err_type.replace('_', ' ')} {limit}."

    if "enum" in err_type or err_type == "literal_error":
        expected = ctx.get("expected")
        if expected:
            return f"Allowed values: {expected}"

    return None


def _check_top_level_keys(raw: dict[str, Any]) -> list[str]:
    """Warn about unrecognised top-level keys before Pydantic validation."""
    known = set(LLMForgeConfig.model_fields.keys())
    warnings: list[str] = []
    for key in raw:
        if key not in known:
            suggestion = _FIELD_SUGGESTIONS.get(key, "")
            msg = f"Unknown top-level key '{key}'."
            if suggestion:
                msg += f"  {suggestion}"
            warnings.append(msg)
    return warnings


# ---------------------------------------------------------------------------
# File-path validation helpers
# ---------------------------------------------------------------------------


def _validate_file_paths(config: LLMForgeConfig) -> list[str]:
    """Check that referenced file / directory paths actually exist."""
    warnings: list[str] = []

    # Training data -- accept HF dataset IDs (contain '/') without checking
    train_path = config.data.train_path
    if "/" not in train_path and not Path(train_path).exists():
        warnings.append(
            f"data.train_path: Local path '{train_path}' does not exist.  "
            "If this is a HuggingFace dataset, use the format 'org/dataset-name'."
        )

    if config.data.eval_path is not None:
        eval_path = config.data.eval_path
        if "/" not in eval_path and not Path(eval_path).exists():
            warnings.append(f"data.eval_path: Local path '{eval_path}' does not exist.")

    if config.rag.enabled and config.rag.knowledge_base_path is not None:
        kb_path = config.rag.knowledge_base_path
        if not Path(kb_path).exists():
            warnings.append(f"rag.knowledge_base_path: Path '{kb_path}' does not exist.")

    if config.training.resume_from_checkpoint is not None:
        ckpt = config.training.resume_from_checkpoint
        if not Path(ckpt).exists():
            warnings.append(f"training.resume_from_checkpoint: Path '{ckpt}' does not exist.")

    if config.evaluation.custom_eval_path is not None:
        cep = config.evaluation.custom_eval_path
        if not Path(cep).exists():
            warnings.append(f"evaluation.custom_eval_path: Path '{cep}' does not exist.")

    return warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ConfigValidationError(Exception):
    """Raised when YAML config validation fails."""

    def __init__(self, message: str, warnings: list[str] | None = None) -> None:
        super().__init__(message)
        self.warnings = warnings or []


def validate_config_dict(raw: dict[str, Any]) -> LLMForgeConfig:
    """Validate a raw dictionary against the LLMForgeConfig schema.

    Parameters
    ----------
    raw:
        Parsed YAML content as a Python dict.

    Returns
    -------
    LLMForgeConfig
        Validated and normalised configuration object.

    Raises
    ------
    ConfigValidationError
        With a user-friendly description of every validation failure.
    """
    # Pre-validation warnings (unknown keys, etc.)
    pre_warnings = _check_top_level_keys(raw)

    try:
        config = LLMForgeConfig(**raw)
    except ValidationError as exc:
        detail = _format_validation_errors(exc)
        if pre_warnings:
            detail += "\nAdditional warnings:\n" + "\n".join(f"  - {w}" for w in pre_warnings)
        raise ConfigValidationError(detail, warnings=pre_warnings) from exc

    # Post-validation file-path warnings
    path_warnings = _validate_file_paths(config)
    if pre_warnings or path_warnings:
        all_warnings = pre_warnings + path_warnings
        # We still return the config, but print warnings to stderr
        for w in all_warnings:
            print(f"[llm-forge config warning] {w}", file=sys.stderr)

    return config


def validate_config_file(path: str | Path) -> LLMForgeConfig:
    """Load a YAML file and validate it against the LLMForgeConfig schema.

    Parameters
    ----------
    path:
        Filesystem path to a ``.yaml`` / ``.yml`` file.

    Returns
    -------
    LLMForgeConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ConfigValidationError
        If validation fails.
    """
    filepath = Path(path).resolve()
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    if not filepath.is_file():
        raise ValueError(f"Path is not a file: {filepath}")

    with open(filepath, encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigValidationError(f"Failed to parse YAML file '{filepath}':\n{exc}") from exc

    if raw is None:
        raise ConfigValidationError(f"YAML file '{filepath}' is empty.")

    if not isinstance(raw, dict):
        raise ConfigValidationError(
            f"YAML file '{filepath}' must contain a mapping at the top level, "
            f"got {type(raw).__name__}."
        )

    return validate_config_dict(raw)


def load_preset(name: str) -> LLMForgeConfig:
    """Load a built-in preset configuration by name.

    Parameters
    ----------
    name:
        Preset name without extension, e.g. ``"lora_default"``.
        Corresponds to a file ``<name>.yaml`` in the ``presets/`` directory.

    Returns
    -------
    LLMForgeConfig
        Validated configuration from the preset.

    Raises
    ------
    FileNotFoundError
        If no preset with the given name exists.
    ConfigValidationError
        If the preset file fails validation (should not happen for shipped presets).
    """
    # Allow callers to pass with or without .yaml extension
    stem = name.removesuffix(".yaml").removesuffix(".yml")
    preset_path = _PRESETS_DIR / f"{stem}.yaml"

    if not preset_path.exists():
        available = sorted(p.stem for p in _PRESETS_DIR.glob("*.yaml") if p.is_file())
        available_str = ", ".join(available) if available else "(none found)"
        raise FileNotFoundError(
            f"Preset '{name}' not found at {preset_path}.  Available presets: {available_str}"
        )

    return validate_config_file(preset_path)


def list_presets() -> list[str]:
    """Return names of all available built-in presets."""
    if not _PRESETS_DIR.exists():
        return []
    return sorted(p.stem for p in _PRESETS_DIR.glob("*.yaml") if p.is_file())
