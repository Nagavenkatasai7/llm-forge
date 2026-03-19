"""Batch size optimisation utilities.

Provides binary-search-based optimal micro-batch size finding and
automatic gradient accumulation step computation to hit a target
effective batch size or token throughput.
"""

from __future__ import annotations

import gc
import math
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("training.optimization.batch_optimizer")

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from datasets import Dataset

    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False


# ============================================================================
# BatchOptimizer
# ============================================================================


class BatchOptimizer:
    """Batch size and gradient accumulation optimiser.

    Determines the largest micro-batch size that fits in GPU memory using
    a binary search, and computes the gradient accumulation steps needed
    to achieve a target effective batch size or token throughput.
    """

    def find_optimal_batch_size(
        self,
        model: Any,
        dataset: Any,
        max_vram_gb: float | None = None,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        seq_len: int = 2048,
        start_batch_size: int = 4,
        safety_margin: float = 0.9,
    ) -> int:
        """Find the optimal micro-batch size via binary search.

        Iteratively tries forward + backward passes with increasing
        batch sizes until an out-of-memory condition is hit, then
        binary-searches for the largest batch that fits.

        Parameters
        ----------
        model : PreTrainedModel
            Model to test.
        dataset : Dataset
            Dataset to sample dummy batches from (or ``None`` for
            synthetic dummy data).
        max_vram_gb : float, optional
            Maximum VRAM budget in GB.  If ``None``, uses 90% of total
            available VRAM.
        min_batch_size : int
            Minimum batch size to try.
        max_batch_size : int
            Maximum batch size to try.
        seq_len : int
            Sequence length for dummy inputs.
        start_batch_size : int
            Initial batch size guess.
        safety_margin : float
            Fraction of max VRAM to use (default 0.9 = 90%).

        Returns
        -------
        int
            Optimal micro-batch size.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Returning default batch size of 1.")
            return 1

        if not torch.cuda.is_available():
            logger.warning(
                "No CUDA GPU available. Returning default batch size of %d.",
                start_batch_size,
            )
            return start_batch_size

        # Determine VRAM budget
        if max_vram_gb is None:
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            max_vram_gb = total_vram * safety_margin

        device = torch.device("cuda:0")
        model = model.to(device)
        model.train()

        # Get vocab size from model config
        vocab_size = 32000
        if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
            vocab_size = model.config.vocab_size

        # Binary search
        lo = min_batch_size
        hi = max_batch_size
        optimal = min_batch_size

        logger.info(
            "Starting batch size search: range=[%d, %d], vram=%.1f GB, seq_len=%d",
            lo,
            hi,
            max_vram_gb,
            seq_len,
        )

        while lo <= hi:
            mid = (lo + hi) // 2
            success = self._try_batch_size(model, mid, seq_len, vocab_size, device)

            if success:
                optimal = mid
                lo = mid + 1
                logger.debug("Batch size %d: SUCCESS (trying larger)", mid)
            else:
                hi = mid - 1
                logger.debug("Batch size %d: OOM (trying smaller)", mid)

            # Clean up
            self._cleanup_gpu()

        # Verify the optimal batch size one more time
        if optimal > min_batch_size:
            verified = self._try_batch_size(model, optimal, seq_len, vocab_size, device)
            if not verified:
                optimal = max(min_batch_size, optimal - 1)
                logger.info("Verification failed; reduced to batch_size=%d", optimal)

        self._cleanup_gpu()
        logger.info("Optimal micro-batch size: %d", optimal)
        return optimal

    def compute_gradient_accumulation(
        self,
        micro_batch_size: int,
        dp_degree: int,
        target_tokens: int,
        seq_len: int,
    ) -> int:
        """Compute gradient accumulation steps to reach a target token count.

        The effective batch size in tokens is:

            ``effective_tokens = micro_batch * dp * grad_accum * seq_len``

        This function solves for ``grad_accum``.

        Parameters
        ----------
        micro_batch_size : int
            Per-GPU micro-batch size.
        dp_degree : int
            Data-parallel degree (number of GPUs doing data parallelism).
        target_tokens : int
            Target number of tokens per optimiser step.  Common values:
            - 500K for small models
            - 1M for 7B models
            - 2M-4M for 70B+ models
        seq_len : int
            Sequence length.

        Returns
        -------
        int
            Number of gradient accumulation steps.

        Examples
        --------
        >>> opt = BatchOptimizer()
        >>> # 7B model: 4 micro-batch, 8 GPUs, target 1M tokens, seq_len=4096
        >>> opt.compute_gradient_accumulation(4, 8, 1_000_000, 4096)
        8
        """
        if micro_batch_size <= 0 or dp_degree <= 0 or seq_len <= 0:
            raise ValueError(
                "micro_batch_size, dp_degree, and seq_len must be positive. "
                f"Got: micro_batch={micro_batch_size}, dp={dp_degree}, seq={seq_len}"
            )

        if target_tokens <= 0:
            raise ValueError(f"target_tokens must be positive, got {target_tokens}")

        tokens_per_step = micro_batch_size * dp_degree * seq_len
        if tokens_per_step <= 0:
            return 1

        grad_accum = max(1, math.ceil(target_tokens / tokens_per_step))

        effective_tokens = tokens_per_step * grad_accum

        logger.info(
            "Gradient accumulation: micro_batch=%d, dp=%d, seq=%d, "
            "target_tokens=%d, grad_accum=%d, effective_tokens=%d",
            micro_batch_size,
            dp_degree,
            seq_len,
            target_tokens,
            grad_accum,
            effective_tokens,
        )

        return grad_accum

    def compute_effective_batch_size(
        self,
        micro_batch_size: int,
        dp_degree: int,
        gradient_accumulation_steps: int,
        seq_len: int,
    ) -> dict[str, int]:
        """Compute the effective batch size metrics.

        Parameters
        ----------
        micro_batch_size : int
            Per-GPU micro-batch size.
        dp_degree : int
            Data-parallel degree.
        gradient_accumulation_steps : int
            Gradient accumulation steps.
        seq_len : int
            Sequence length.

        Returns
        -------
        dict[str, int]
            Effective batch size metrics:
            - ``"effective_batch_size"``: total samples per step
            - ``"effective_tokens"``: total tokens per step
            - ``"micro_batch_size"``: per-GPU micro-batch
            - ``"dp_degree"``: data-parallel degree
            - ``"gradient_accumulation_steps"``: grad accum steps
        """
        effective_batch = micro_batch_size * dp_degree * gradient_accumulation_steps
        effective_tokens = effective_batch * seq_len

        return {
            "effective_batch_size": effective_batch,
            "effective_tokens": effective_tokens,
            "micro_batch_size": micro_batch_size,
            "dp_degree": dp_degree,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "seq_len": seq_len,
        }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _try_batch_size(
        model: Any,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        device: Any,
    ) -> bool:
        """Attempt a forward + backward pass with the given batch size.

        Returns
        -------
        bool
            ``True`` if the pass succeeded, ``False`` on OOM.
        """
        try:
            # Generate dummy input
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
            labels = input_ids.clone()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Cleanup
            del input_ids, attention_mask, labels, outputs, loss
            model.zero_grad(set_to_none=True)

            return True

        except torch.cuda.OutOfMemoryError:
            # Clean up after OOM
            model.zero_grad(set_to_none=True)
            return False

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                model.zero_grad(set_to_none=True)
                return False
            raise

    @staticmethod
    def _cleanup_gpu() -> None:
        """Force GPU memory cleanup."""
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
