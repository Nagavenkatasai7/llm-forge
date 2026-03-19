"""llm-forge training engine.

Provides a unified training orchestrator and specialised engines for
fine-tuning (LoRA/QLoRA/full), pre-training from scratch, alignment
(DPO/RLHF), and MLX-based training on Apple Silicon.
"""

_all: list = []

try:
    from llm_forge.training.trainer import Trainer

    _all.append("Trainer")
except ImportError:
    pass

try:
    from llm_forge.training.finetuner import FineTuner

    _all.append("FineTuner")
except ImportError:
    pass

try:
    from llm_forge.training.pretrainer import PreTrainer

    _all.append("PreTrainer")
except ImportError:
    pass

try:
    from llm_forge.training.alignment import AlignmentTrainer

    _all.append("AlignmentTrainer")
except ImportError:
    pass

try:
    from llm_forge.training.mlx_trainer import MLXTrainer, is_mlx_available

    _all.extend(["MLXTrainer", "is_mlx_available"])
except ImportError:
    MLXTrainer = None  # type: ignore[assignment,misc]

    def is_mlx_available() -> bool:  # type: ignore[misc]
        return False

    _all.extend(["MLXTrainer", "is_mlx_available"])

__all__ = _all
