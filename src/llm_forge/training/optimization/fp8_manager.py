"""FP8 training manager using NVIDIA Transformer Engine.

Provides FP8 mixed-precision training support with automatic scaling,
amax history tracking, and graceful fallback when the transformer-engine
package is not installed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("training.optimization.fp8_manager")

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format

    _TE_AVAILABLE = True
except ImportError:
    _TE_AVAILABLE = False
    te = None  # type: ignore[assignment]
    DelayedScaling = None  # type: ignore[assignment, misc]
    Format = None  # type: ignore[assignment, misc]

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ============================================================================
# FP8Manager
# ============================================================================


class FP8Manager:
    """Manager for FP8 mixed-precision training via Transformer Engine.

    FP8 training uses 8-bit floating point for forward and backward passes
    to reduce memory and increase throughput on NVIDIA Hopper (H100) and
    later GPUs.  The manager handles recipe configuration, dynamic
    scaling, and context management.

    Parameters
    ----------
    enabled : bool
        Whether FP8 is enabled.  When ``False`` or when transformer-engine
        is not installed, all operations become no-ops.
    fp8_format : str
        FP8 format: ``"E4M3"`` for forward-only, ``"HYBRID"`` for
        E4M3 forward + E5M2 backward (recommended for training).
    amax_history_len : int
        Length of the amax history window for delayed scaling.
    amax_compute_algo : str
        Algorithm for computing amax: ``"max"`` or ``"most_recent"``.
    margin : int
        Margin for delayed scaling.
    """

    def __init__(
        self,
        enabled: bool = True,
        fp8_format: str = "HYBRID",
        amax_history_len: int = 1024,
        amax_compute_algo: str = "max",
        margin: int = 0,
    ) -> None:
        self.enabled = enabled and _TE_AVAILABLE
        self.fp8_format = fp8_format
        self.amax_history_len = amax_history_len
        self.amax_compute_algo = amax_compute_algo
        self.margin = margin
        self._recipe: Any | None = None
        self._is_setup: bool = False

        if enabled and not _TE_AVAILABLE:
            logger.warning(
                "FP8 training requested but transformer-engine is not installed. "
                "Install with: pip install transformer-engine. "
                "Falling back to standard precision."
            )

    @property
    def is_available(self) -> bool:
        """Check if FP8 training is available on this system."""
        if not _TE_AVAILABLE or not _TORCH_AVAILABLE:
            return False

        # FP8 requires Hopper (SM 8.9+) or later
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                capability = torch.cuda.get_device_capability(i)
                if capability[0] < 8 or (capability[0] == 8 and capability[1] < 9):
                    return False
            return True
        return False

    def setup_fp8(
        self,
        model: Any,
        recipe: str | None = None,
        amax_history_len: int | None = None,
    ) -> Any:
        """Configure FP8 training for a model.

        Wraps supported layers (Linear, LayerNorm, etc.) with
        Transformer Engine equivalents and configures the FP8 recipe.

        Parameters
        ----------
        model : PreTrainedModel
            The model to configure for FP8.
        recipe : str, optional
            FP8 recipe name: ``"delayed"`` (default).
        amax_history_len : int, optional
            Override amax history length.

        Returns
        -------
        model
            The (potentially modified) model with FP8 support.
        """
        if not self.enabled:
            logger.debug("FP8 not enabled; skipping setup")
            return model

        if not _TE_AVAILABLE or DelayedScaling is None or Format is None:
            logger.warning("transformer-engine not available; FP8 setup skipped")
            return model

        # Resolve format
        fp8_fmt = Format.E4M3 if self.fp8_format.upper() == "E4M3" else Format.HYBRID

        # Build delayed scaling recipe
        history_len = amax_history_len or self.amax_history_len
        self._recipe = DelayedScaling(
            fp8_format=fp8_fmt,
            amax_history_len=history_len,
            amax_compute_algo=self.amax_compute_algo,
            margin=self.margin,
        )

        self._is_setup = True

        logger.info(
            "FP8 configured: format=%s, amax_history=%d, algo=%s, margin=%d",
            self.fp8_format,
            history_len,
            self.amax_compute_algo,
            self.margin,
        )

        return model

    @contextlib.contextmanager
    def fp8_context(self) -> Iterator[None]:
        """Context manager for FP8 forward/backward passes.

        Use this to wrap the training step:

        .. code-block:: python

            with fp8_manager.fp8_context():
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

        Yields
        ------
        None
        """
        if not self.enabled or not self._is_setup or not _TE_AVAILABLE:
            yield
            return

        with te.fp8_autocast(enabled=True, fp8_recipe=self._recipe):
            yield

    def wrap_model(self, model: Any) -> Any:
        """Replace standard linear and norm layers with TE equivalents.

        Parameters
        ----------
        model : nn.Module
            PyTorch model.

        Returns
        -------
        nn.Module
            Model with TE-wrapped layers.
        """
        if not self.enabled or not _TE_AVAILABLE or not _TORCH_AVAILABLE:
            return model

        import torch.nn as nn

        replacements = 0
        for name, module in model.named_modules():
            # Replace nn.Linear with te.Linear where possible
            if isinstance(module, nn.Linear) and not isinstance(module, te.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model
                if parent_name:
                    for part in parent_name.split("."):
                        parent = getattr(parent, part)

                te_linear = te.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                )
                # Copy weights
                te_linear.weight.data.copy_(module.weight.data)
                if module.bias is not None and te_linear.bias is not None:
                    te_linear.bias.data.copy_(module.bias.data)

                setattr(parent, child_name, te_linear)
                replacements += 1

        if replacements > 0:
            logger.info("Replaced %d layers with Transformer Engine equivalents", replacements)
        else:
            logger.info("No layers replaced (model may already use TE layers)")

        return model

    def get_recipe(self) -> Any | None:
        """Return the current FP8 recipe, or None if not configured."""
        return self._recipe

    def get_status(self) -> dict[str, Any]:
        """Return the current FP8 configuration status.

        Returns
        -------
        dict[str, Any]
            Status dictionary with availability, configuration, and
            hardware compatibility information.
        """
        return {
            "enabled": self.enabled,
            "available": self.is_available,
            "transformer_engine_installed": _TE_AVAILABLE,
            "is_setup": self._is_setup,
            "fp8_format": self.fp8_format,
            "amax_history_len": self.amax_history_len,
            "amax_compute_algo": self.amax_compute_algo,
            "margin": self.margin,
        }
