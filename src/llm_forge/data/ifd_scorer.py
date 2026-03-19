"""IFD (Instruction-Following Difficulty) data scoring and filtering.

Implements the IFD metric from *"From Quantity to Quality: Boosting LLM
Performance with Self-Guided Data Selection for Instruction Tuning"*
(Li et al., NAACL 2024, arXiv:2308.12032).

IFD(Q, A) = s(A|Q) / s(A)

where s(A|Q) is the average per-token negative log-likelihood of the response
given the full instruction+response, and s(A) is the same metric with only
the response.  A high IFD means the instruction didn't help — these samples
tend to be more valuable for training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("llm_forge.data.ifd_scorer")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _no_grad_fallback(fn):
    """Identity decorator fallback when torch is not available."""
    return fn


try:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class IFDResult:
    """Result of IFD scoring across a dataset."""

    scores: list[float] = field(default_factory=list)
    conditioned_losses: list[float] = field(default_factory=list)
    direct_losses: list[float] = field(default_factory=list)
    num_scored: int = 0
    mean_ifd: float = 0.0
    median_ifd: float = 0.0


# ---------------------------------------------------------------------------
# IFDScorer
# ---------------------------------------------------------------------------


class IFDScorer:
    """Compute IFD scores for instruction-tuning datasets.

    Parameters
    ----------
    max_length : int
        Maximum sequence length for tokenisation.
    batch_size : int
        Mini-batch size for forward passes.
    """

    def __init__(self, max_length: int = 512, batch_size: int = 4):
        self.max_length = max_length
        self.batch_size = batch_size

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def score_dataset(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        instructions: list[str],
        responses: list[str],
    ) -> IFDResult:
        """Compute IFD scores for a list of (instruction, response) pairs.

        Parameters
        ----------
        model : PreTrainedModel
            Language model for computing log-likelihoods.
        tokenizer : PreTrainedTokenizerBase
            Tokenizer matching the model.
        instructions : list[str]
            Instruction / prompt texts.
        responses : list[str]
            Corresponding response texts.

        Returns
        -------
        IFDResult
            Per-sample IFD scores and aggregate statistics.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for IFD scoring")
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for IFD scoring")

        assert len(instructions) == len(responses), (
            "instructions and responses must have the same length"
        )

        model.eval()
        device = next(model.parameters()).device

        all_scores: list[float] = []
        all_cond: list[float] = []
        all_direct: list[float] = []

        for i in range(0, len(instructions), self.batch_size):
            batch_inst = instructions[i : i + self.batch_size]
            batch_resp = responses[i : i + self.batch_size]

            for inst, resp in zip(batch_inst, batch_resp, strict=False):
                cond_loss = self._compute_response_loss(
                    model, tokenizer, device, instruction=inst, response=resp
                )
                direct_loss = self._compute_response_loss(
                    model, tokenizer, device, instruction=None, response=resp
                )

                # Avoid division by zero — if direct_loss is ~0 the response
                # is trivially predictable; assign IFD=0 (very easy).
                ifd = 0.0 if direct_loss < 1e-08 else cond_loss / direct_loss

                all_scores.append(ifd)
                all_cond.append(cond_loss)
                all_direct.append(direct_loss)

        # Aggregate stats
        n = len(all_scores)
        mean_ifd = sum(all_scores) / max(n, 1)
        sorted_scores = sorted(all_scores)
        median_ifd = (
            (
                sorted_scores[n // 2]
                if n % 2 == 1
                else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
            )
            if n > 0
            else 0.0
        )

        return IFDResult(
            scores=all_scores,
            conditioned_losses=all_cond,
            direct_losses=all_direct,
            num_scored=n,
            mean_ifd=mean_ifd,
            median_ifd=median_ifd,
        )

    def filter_by_ifd(
        self,
        dataset: Any,
        scores: list[float],
        select_ratio: float = 0.5,
    ) -> Any:
        """Keep the top ``select_ratio`` fraction of samples ranked by IFD.

        Higher IFD samples are retained (they are harder for the model to
        follow and therefore more valuable for training).

        Parameters
        ----------
        dataset : Dataset
            HuggingFace dataset to filter.
        scores : list[float]
            IFD scores aligned with dataset rows.
        select_ratio : float
            Fraction of data to keep (0, 1].

        Returns
        -------
        Dataset
            Filtered dataset.
        """
        n = len(scores)
        k = max(1, int(n * select_ratio))

        # Get indices of top-k scores (highest IFD = most valuable)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = sorted([idx for idx, _ in indexed[:k]])

        filtered = dataset.select(top_indices)
        logger.info(
            "IFD filtering: kept %d / %d samples (ratio=%.2f, IFD range: %.3f - %.3f)",
            len(filtered),
            n,
            select_ratio,
            min(scores) if scores else 0,
            max(scores) if scores else 0,
        )
        return filtered

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @(torch.no_grad() if _TORCH_AVAILABLE else _no_grad_fallback)
    def _compute_response_loss(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        response: str,
        instruction: str | None = None,
    ) -> float:
        """Compute average per-token NLL of the response portion.

        When *instruction* is given, the full sequence is
        ``[instruction] [response]`` and only response tokens contribute to
        the loss.  When *instruction* is ``None``, only the response is fed.
        """
        if instruction is not None:
            full_text = instruction + "\n" + response
            # Tokenise instruction alone to know where response starts
            inst_ids = tokenizer.encode(instruction + "\n", add_special_tokens=True)
            inst_len = len(inst_ids)
        else:
            full_text = response
            inst_len = 0

        encoding = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding.get("attention_mask", torch.ones_like(input_ids)).to(device)

        # Build labels: mask instruction tokens with -100
        labels = input_ids.clone()
        if inst_len > 0:
            # Mask everything before the response portion
            mask_len = min(inst_len, labels.shape[1])
            labels[0, :mask_len] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # outputs.loss is the average cross-entropy over non-masked tokens
        return outputs.loss.item()
