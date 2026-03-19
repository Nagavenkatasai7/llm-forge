"""Error recovery suggestions for common training failures.

Analyses exception messages and training context to provide actionable
suggestions for resolving errors.  Used by the CLI and pipeline runner
to give users clear next steps when things go wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecoverySuggestion:
    """A single error recovery suggestion."""

    category: str
    message: str
    priority: int = 0  # 0 = highest priority


@dataclass
class ErrorDiagnosis:
    """Complete error analysis with categorised suggestions."""

    error_type: str
    error_message: str
    suggestions: list[RecoverySuggestion] = field(default_factory=list)
    auto_fixable: bool = False
    auto_fix_description: str | None = None

    @property
    def suggestion_texts(self) -> list[str]:
        """Return sorted suggestion messages."""
        return [s.message for s in sorted(self.suggestions, key=lambda s: s.priority)]


# ---------------------------------------------------------------------------
# Error pattern matchers
# ---------------------------------------------------------------------------

_ERROR_PATTERNS: list[dict[str, Any]] = [
    # Out of memory
    {
        "keywords": ["out of memory", "oom", "cuda.*memory", "mps.*memory", "allocat"],
        "type": "Out of Memory",
        "suggestions": [
            RecoverySuggestion("memory", "Reduce per_device_train_batch_size (try halving it)", 0),
            RecoverySuggestion("memory", "Enable gradient_checkpointing: true", 1),
            RecoverySuggestion("memory", "Switch to QLoRA (mode: qlora) for 4-bit quantization", 2),
            RecoverySuggestion("memory", "Reduce max_seq_length (e.g., 512 instead of 2048)", 3),
            RecoverySuggestion("memory", "On Mac: set mac.mps_high_watermark_ratio: 0.0", 4),
            RecoverySuggestion("memory", "Enable pack_sequences: true to reduce padding waste", 5),
        ],
    },
    # NaN / Inf in loss
    {
        "keywords": ["nan", "inf", "diverge", "loss.*nan", "nan.*loss"],
        "type": "Training Instability (NaN/Inf)",
        "suggestions": [
            RecoverySuggestion("stability", "Reduce learning_rate (try 1e-5 or lower)", 0),
            RecoverySuggestion("stability", "Increase warmup_ratio (try 0.1)", 1),
            RecoverySuggestion("stability", "Check dataset for corrupted or empty examples", 2),
            RecoverySuggestion("stability", "Use fp32 instead of fp16 if on CPU/MPS", 3),
            RecoverySuggestion("stability", "Reduce max_grad_norm (try 0.3)", 4),
            RecoverySuggestion("stability", "Disable label_smoothing (set to 0.0)", 5),
        ],
    },
    # Import errors
    {
        "keywords": ["no module named", "importerror", "modulenotfounderror"],
        "type": "Missing Dependencies",
        "suggestions": [
            RecoverySuggestion("deps", "Install all dependencies: pip install -e '.[all]'", 0),
            RecoverySuggestion("deps", "Verify your virtual environment is activated", 1),
            RecoverySuggestion("deps", "Run 'llm-forge doctor' to diagnose missing packages", 2),
            RecoverySuggestion("deps", "For QLoRA: pip install bitsandbytes", 3),
        ],
    },
    # Config / validation errors
    {
        "keywords": ["validation", "field required", "extra inputs", "pydantic"],
        "type": "Configuration Error",
        "suggestions": [
            RecoverySuggestion(
                "config", "Run 'llm-forge validate <config.yaml>' to check syntax", 0
            ),
            RecoverySuggestion("config", "Check YAML indentation (use spaces, not tabs)", 1),
            RecoverySuggestion("config", "Verify model name is a valid HuggingFace model ID", 2),
            RecoverySuggestion(
                "config", "Remove any unrecognised fields (extra: 'forbid' is set)", 3
            ),
        ],
    },
    # Network / download errors
    {
        "keywords": ["connection", "timeout", "network", "httperror", "resolve host"],
        "type": "Network Error",
        "suggestions": [
            RecoverySuggestion("network", "Check internet connectivity", 0),
            RecoverySuggestion(
                "network", "Set HF_HOME to a cache directory with pre-downloaded models", 1
            ),
            RecoverySuggestion(
                "network", "Try: huggingface-cli download <model-name> to pre-cache", 2
            ),
            RecoverySuggestion("network", "Use --offline mode if models are already cached", 3),
        ],
    },
    # CUDA errors
    {
        "keywords": ["cuda error", "cublas", "nccl", "device-side assert"],
        "type": "CUDA Error",
        "suggestions": [
            RecoverySuggestion("cuda", "Check CUDA and PyTorch version compatibility", 0),
            RecoverySuggestion("cuda", "Try: CUDA_VISIBLE_DEVICES=0 to use a single GPU", 1),
            RecoverySuggestion("cuda", "Update GPU drivers to the latest version", 2),
            RecoverySuggestion("cuda", "Check GPU temperature — throttling may cause errors", 3),
        ],
    },
    # Checkpoint / resume errors
    {
        "keywords": ["checkpoint", "resume", "state_dict", "loading.*weights"],
        "type": "Checkpoint Error",
        "suggestions": [
            RecoverySuggestion(
                "checkpoint", "Verify the checkpoint directory exists and is not corrupted", 0
            ),
            RecoverySuggestion(
                "checkpoint", "Try training from scratch (remove resume_from_checkpoint)", 1
            ),
            RecoverySuggestion("checkpoint", "Ensure model architecture matches the checkpoint", 2),
            RecoverySuggestion(
                "checkpoint", "Check disk space — incomplete checkpoints cause errors", 3
            ),
        ],
    },
    # Tokenizer errors
    {
        "keywords": ["tokenizer", "token.*id", "vocab", "encoding"],
        "type": "Tokenizer Error",
        "suggestions": [
            RecoverySuggestion("tokenizer", "Ensure the tokenizer matches the model", 0),
            RecoverySuggestion(
                "tokenizer", "Set trust_remote_code: true if using custom tokenizers", 1
            ),
            RecoverySuggestion("tokenizer", "Check that input text is valid UTF-8", 2),
        ],
    },
    # Dataset errors
    {
        "keywords": ["dataset", "column.*not found", "key.*error", "jsonl", "data.*loading"],
        "type": "Dataset Error",
        "suggestions": [
            RecoverySuggestion("data", "Verify the dataset path exists and is readable", 0),
            RecoverySuggestion(
                "data", "Check column names match config (input_field, output_field)", 1
            ),
            RecoverySuggestion("data", "Ensure JSONL file has valid JSON on each line", 2),
            RecoverySuggestion(
                "data",
                "Try: python -c \"import json; [json.loads(l) for l in open('data.jsonl')]\"",
                3,
            ),
        ],
    },
    # Disk space
    {
        "keywords": ["no space", "disk.*full", "errno 28", "oserror.*space"],
        "type": "Disk Space Error",
        "suggestions": [
            RecoverySuggestion("disk", "Free disk space (checkpoints can be 10+ GB each)", 0),
            RecoverySuggestion("disk", "Reduce save_total_limit to keep fewer checkpoints", 1),
            RecoverySuggestion("disk", "Change output_dir to a drive with more space", 2),
        ],
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diagnose_error(
    error: Exception,
    config: Any | None = None,
) -> ErrorDiagnosis:
    """Analyse an exception and return recovery suggestions.

    Parameters
    ----------
    error : Exception
        The caught exception from a training/pipeline failure.
    config : optional
        The LLMForgeConfig (for context-aware suggestions).

    Returns
    -------
    ErrorDiagnosis
        Categorised error analysis with actionable suggestions.
    """
    import re

    msg = str(error).lower()
    error_type_name = type(error).__name__

    diagnosis = ErrorDiagnosis(
        error_type=error_type_name,
        error_message=str(error),
    )

    matched = False
    for pattern in _ERROR_PATTERNS:
        for kw in pattern["keywords"]:
            if re.search(kw, msg):
                diagnosis.error_type = pattern["type"]
                diagnosis.suggestions.extend(pattern["suggestions"])
                matched = True
                break
        if matched:
            break

    if not matched:
        diagnosis.suggestions = [
            RecoverySuggestion("general", "Run 'llm-forge doctor' to check your environment", 0),
            RecoverySuggestion(
                "general", "Run 'llm-forge validate <config.yaml>' to check config", 1
            ),
            RecoverySuggestion("general", "Use --verbose for detailed error information", 2),
        ]

    # Context-aware suggestions
    if config is not None:
        _add_context_suggestions(diagnosis, config)

    return diagnosis


def _add_context_suggestions(diagnosis: ErrorDiagnosis, config: Any) -> None:
    """Add suggestions based on the specific configuration."""
    try:
        training = config.training
        mode = training.mode

        if diagnosis.error_type == "Out of Memory":
            if mode in ("lora", "full"):
                diagnosis.suggestions.append(
                    RecoverySuggestion(
                        "memory",
                        f"Currently using '{mode}' — switch to 'qlora' for 4x less memory",
                        0,
                    )
                )
            batch = training.per_device_train_batch_size
            if batch > 1:
                diagnosis.suggestions.append(
                    RecoverySuggestion(
                        "memory",
                        f"Current batch_size={batch} — try reducing to {max(1, batch // 2)}",
                        0,
                    )
                )
            seq_len = config.model.max_seq_length
            if seq_len > 512:
                diagnosis.suggestions.append(
                    RecoverySuggestion(
                        "memory",
                        f"Current max_seq_length={seq_len} — try reducing to 512",
                        1,
                    )
                )

        if diagnosis.error_type == "Training Instability (NaN/Inf)":
            lr = training.learning_rate
            if lr > 1e-4:
                diagnosis.suggestions.append(
                    RecoverySuggestion(
                        "stability",
                        f"Current learning_rate={lr} — try {lr / 10:.0e}",
                        0,
                    )
                )
    except (AttributeError, TypeError):
        pass


def format_diagnosis_plain(diagnosis: ErrorDiagnosis) -> str:
    """Format an ErrorDiagnosis as plain text."""
    lines = [
        f"Error Type: {diagnosis.error_type}",
        f"Message: {diagnosis.error_message[:200]}",
        "",
        "Suggested fixes:",
    ]
    for i, s in enumerate(diagnosis.suggestion_texts, 1):
        lines.append(f"  {i}. {s}")
    return "\n".join(lines)
