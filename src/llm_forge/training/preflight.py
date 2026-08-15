"""Pre-training compatibility checks.

Catches config/hardware mismatches before anything downloads a model or
allocates memory. The failure this exists to prevent: ``mode: qlora`` on Apple
Silicon builds a ``BitsAndBytesConfig``, which is CUDA-only, and dies inside
``from_pretrained`` -- several minutes and a multi-gigabyte download into what
looked like a working run, with a traceback that names neither the real cause
nor the fix.

Checks run at the point where a config is routed to a trainer, so both the
PyTorch and MLX paths are covered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from llm_forge.utils.logging import get_logger

    logger = get_logger("training.preflight")
except ImportError:  # pragma: no cover - logging util is optional
    import logging

    logger = logging.getLogger(__name__)


class PreflightError(RuntimeError):
    """A config cannot run on this machine. Message states the fix."""


@dataclass
class Problem:
    """One blocking incompatibility, with the config change that resolves it."""

    what: str
    why: str
    fix: str

    def render(self) -> str:
        return f"  - {self.what}\n      Why: {self.why}\n      Fix: {self.fix}"


def detect_backend() -> str:
    """Return ``"cuda"``, ``"mps"``, or ``"cpu"`` for the current machine."""
    try:
        import torch
    except ImportError:
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - unusual torch builds
        return "cpu"
    return "cpu"


def _mode_of(config: Any) -> str:
    mode = getattr(getattr(config, "training", None), "mode", None)
    # TrainingMode is a str-Enum; .value keeps this working either way.
    return str(getattr(mode, "value", mode) or "").lower()


def _mlx_enabled(config: Any) -> bool:
    mlx = getattr(config, "mlx", None)
    return bool(getattr(mlx, "enabled", False))


def check_config(config: Any, *, backend: str | None = None) -> list[Problem]:
    """Return every blocking incompatibility between ``config`` and this machine.

    An empty list means the config can run.
    """
    if backend is None:
        backend = detect_backend()

    problems: list[Problem] = []
    mode = _mode_of(config)
    mlx_on = _mlx_enabled(config)

    # bitsandbytes 4-bit has no Metal or CPU backend.
    if mode == "qlora" and backend != "cuda" and not mlx_on:
        problems.append(
            Problem(
                what=f"training.mode is 'qlora' but no CUDA GPU is available (backend={backend}).",
                why=(
                    "QLoRA loads the base model through bitsandbytes 4-bit "
                    "quantization, which is implemented only for CUDA. On Apple "
                    "Silicon the model load raises inside from_pretrained, after "
                    "the download has already completed."
                ),
                fix=(
                    "On Apple Silicon: set mlx.enabled: true and mlx.fine_tune_type: "
                    "lora, then quantize the base model with "
                    "`mlx_lm.convert --hf-path <model> -q`. "
                    "Otherwise set training.mode: lora, which runs in bf16 on MPS."
                ),
            )
        )

    if mode == "qlora" and backend == "cpu" and not mlx_on:
        problems.append(
            Problem(
                what="QLoRA requested on CPU.",
                why="4-bit quantized training has no CPU kernel.",
                fix="Use training.mode: lora with a small model, or train on a GPU.",
            )
        )

    # MLX is Apple-only.
    if mlx_on and backend != "mps":
        problems.append(
            Problem(
                what=f"mlx.enabled is true but this machine is not Apple Silicon (backend={backend}).",
                why="MLX is built on Metal and runs only on Apple Silicon.",
                fix="Set mlx.enabled: false and use training.mode: lora or qlora.",
            )
        )

    # 8-bit optimizers/quantization also come from bitsandbytes.
    quantization = getattr(config, "quantization", None)
    if getattr(quantization, "load_in_8bit", False) and backend != "cuda":
        problems.append(
            Problem(
                what=f"quantization.load_in_8bit is true but backend is {backend}.",
                why="8-bit quantization is a bitsandbytes feature and requires CUDA.",
                fix="Set quantization.load_in_8bit: false.",
            )
        )

    return problems


def assert_compatible(config: Any, *, backend: str | None = None) -> None:
    """Raise :class:`PreflightError` if ``config`` cannot run on this machine."""
    problems = check_config(config, backend=backend)
    if not problems:
        return

    detail = "\n".join(p.render() for p in problems)
    raise PreflightError(
        f"This config cannot run on your hardware ({len(problems)} problem(s)):\n\n"
        f"{detail}\n\n"
        "Nothing has been downloaded or allocated. Fix the config and re-run."
    )


def warn_if_tight(config: Any, num_params: float | None = None) -> str | None:
    """Warn when a run is projected to use most of the available memory.

    Returns the warning text, or None when there is comfortable headroom.
    Unified-memory machines are the motivating case: exceeding the budget there
    means swapping rather than a clean OOM, so the run gets pathologically slow
    instead of failing fast.
    """
    if not num_params:
        return None

    from llm_forge.chat.discovery import assess_fit

    backend = detect_backend()
    if backend == "mps":
        from llm_forge.training.mac_utils import usable_unified_memory_gb

        try:
            import psutil

            total = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return None
        budget = usable_unified_memory_gb(total)
    elif backend == "cuda":
        import torch

        budget = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        return None

    mode = _mode_of(config)
    method = {"qlora": "qlora", "full": "full", "lora": "lora"}.get(mode, "lora")
    if _mlx_enabled(config):
        method = "mlx_lora_4bit"

    seq_length = int(getattr(getattr(config, "data", None), "max_seq_length", 2048) or 2048)
    verdicts = {
        v.method: v
        for v in assess_fit(num_params, budget, seq_length=seq_length, backend=backend)
    }
    verdict = verdicts.get(method)
    if verdict is None:
        return None

    headroom = budget - verdict.required_gb
    if headroom > 2.0:
        return None

    return (
        f"Tight memory: this run needs about {verdict.required_gb:.1f} GB of a "
        f"{budget:.1f} GB budget ({headroom:.1f} GB spare). "
        + (
            "On unified memory, overshooting swaps rather than OOMs, so steps get "
            "very slow instead of failing. Close other apps, or reduce "
            "max_seq_length / batch size."
            if backend == "mps"
            else "Reduce batch size or max_seq_length if you hit an OOM."
        )
    )
