"""Model merging utilities: Linear, SLERP, and TIES-Merging.

Implements three model merging strategies:

- **Linear**: Weighted average of model parameters.
- **SLERP**: Spherical linear interpolation between two models' parameters.
- **TIES**: Trim, Elect Sign & Merge (Yadav et al., NeurIPS 2023).
  Resolves interference by trimming small updates, electing consensus
  sign, and disjoint-merging aligned parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("llm_forge.serving.model_merger")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# ModelMerger
# ---------------------------------------------------------------------------


class ModelMerger:
    """Merge multiple models using linear, SLERP, or TIES strategies."""

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    @staticmethod
    def merge_linear(
        state_dicts: list[dict[str, torch.Tensor]],
        weights: list[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Weighted average of model state dicts.

        Parameters
        ----------
        state_dicts : list[dict]
            State dicts from each model.
        weights : list[float], optional
            Per-model weights (normalised internally). If ``None``,
            equal weights are used.

        Returns
        -------
        dict
            Merged state dict.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for model merging")

        n = len(state_dicts)
        if n == 0:
            raise ValueError("Need at least one state dict to merge")

        if weights is None:
            weights = [1.0 / n] * n
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        merged: dict[str, torch.Tensor] = {}
        keys = state_dicts[0].keys()

        for key in keys:
            merged[key] = sum(
                w * sd[key].float() for w, sd in zip(weights, state_dicts, strict=False)
            ).to(state_dicts[0][key].dtype)

        return merged

    @staticmethod
    def merge_slerp(
        state_dict_a: dict[str, torch.Tensor],
        state_dict_b: dict[str, torch.Tensor],
        t: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Spherical linear interpolation between two state dicts.

        Parameters
        ----------
        state_dict_a : dict
            First model's state dict (t=0).
        state_dict_b : dict
            Second model's state dict (t=1).
        t : float
            Interpolation factor in [0, 1].

        Returns
        -------
        dict
            Merged state dict.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for model merging")

        merged: dict[str, torch.Tensor] = {}

        for key in state_dict_a:
            a = state_dict_a[key].float().flatten()
            b = state_dict_b[key].float().flatten()

            merged[key] = (
                ModelMerger._slerp_tensors(a, b, t)
                .reshape(state_dict_a[key].shape)
                .to(state_dict_a[key].dtype)
            )

        return merged

    @staticmethod
    def merge_ties(
        base_state_dict: dict[str, torch.Tensor],
        finetuned_state_dicts: list[dict[str, torch.Tensor]],
        weights: list[float] | None = None,
        density: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """TIES-Merging: Trim, Elect Sign & Merge.

        Parameters
        ----------
        base_state_dict : dict
            State dict of the base (pre-trained) model.
        finetuned_state_dicts : list[dict]
            State dicts from fine-tuned models.
        weights : list[float], optional
            Per-model weights for the final merge. Equal if ``None``.
        density : float
            Fraction of parameters to keep in the trim step (top-k by
            absolute magnitude of the task vector).

        Returns
        -------
        dict
            Merged state dict.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for model merging")

        n = len(finetuned_state_dicts)
        if n == 0:
            raise ValueError("Need at least one fine-tuned model")

        if weights is None:
            weights = [1.0 / n] * n
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        merged: dict[str, torch.Tensor] = {}

        for key in base_state_dict:
            base_param = base_state_dict[key].float()

            # Step 1: Compute task vectors (delta from base)
            task_vectors = [sd[key].float() - base_param for sd in finetuned_state_dicts]

            # Step 2: Trim — zero out small-magnitude updates
            trimmed = [ModelMerger._trim_task_vector(tv, density) for tv in task_vectors]

            # Step 3: Elect Sign — majority vote on sign
            elected_sign = ModelMerger._elect_sign(trimmed, weights)

            # Step 4: Disjoint Merge — keep only values aligned with
            # elected sign, then weighted average
            disjoint_sum = torch.zeros_like(base_param)
            disjoint_count = torch.zeros_like(base_param)

            for tv, w in zip(trimmed, weights, strict=False):
                aligned = (tv.sign() == elected_sign) & (tv != 0)
                disjoint_sum += w * tv * aligned.float()
                disjoint_count += aligned.float()

            # Average where we have contributions
            avg_mask = disjoint_count > 0
            disjoint_merged = torch.where(
                avg_mask,
                disjoint_sum / disjoint_count.clamp(min=1),
                torch.zeros_like(base_param),
            )

            merged[key] = (base_param + disjoint_merged).to(base_state_dict[key].dtype)

        return merged

    @staticmethod
    def merge_models(
        method: str,
        model_paths: list[str],
        output_path: str,
        weights: list[float] | None = None,
        base_model: str | None = None,
        slerp_t: float = 0.5,
        ties_density: float = 0.5,
    ) -> Path:
        """High-level API: load models, merge, save.

        Parameters
        ----------
        method : str
            One of "linear", "slerp", "ties".
        model_paths : list[str]
            Paths or HuggingFace IDs of models to merge.
        output_path : str
            Where to save the merged model.
        weights : list[float], optional
            Per-model weights.
        base_model : str, optional
            Base model path (required for TIES).
        slerp_t : float
            SLERP interpolation parameter.
        ties_density : float
            TIES trim density.

        Returns
        -------
        Path
            Output directory of the merged model.
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for model merging. Install with: pip install transformers"
            )

        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        logger.info("Merging %d models with method=%s", len(model_paths), method)

        # Load state dicts
        state_dicts = []
        tokenizer = None
        for path in model_paths:
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map="cpu")
            state_dicts.append(model.state_dict())
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(path)
            del model

        if method == "linear":
            merged_sd = ModelMerger.merge_linear(state_dicts, weights)
        elif method == "slerp":
            if len(state_dicts) != 2:
                raise ValueError("SLERP requires exactly 2 models")
            merged_sd = ModelMerger.merge_slerp(state_dicts[0], state_dicts[1], t=slerp_t)
        elif method == "ties":
            if base_model is None:
                raise ValueError("TIES merging requires a base_model path")
            base = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype="auto", device_map="cpu"
            )
            base_sd = base.state_dict()
            del base
            merged_sd = ModelMerger.merge_ties(base_sd, state_dicts, weights, density=ties_density)
        else:
            raise ValueError(f"Unknown merge method: {method}")

        # Save merged model
        shell_model = AutoModelForCausalLM.from_pretrained(
            model_paths[0], torch_dtype="auto", device_map="cpu"
        )
        shell_model.load_state_dict(merged_sd)
        shell_model.save_pretrained(out)
        tokenizer.save_pretrained(out)

        logger.info("Merged model saved to %s", out)
        return out

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _slerp_tensors(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical linear interpolation between two flat tensors."""
        a_norm = a / a.norm().clamp(min=1e-8)
        b_norm = b / b.norm().clamp(min=1e-8)

        dot = torch.dot(a_norm, b_norm).clamp(-1.0, 1.0)
        omega = torch.acos(dot)

        # Fall back to linear interpolation for near-parallel vectors
        if omega.abs() < 1e-6:
            return (1.0 - t) * a + t * b

        sin_omega = torch.sin(omega)
        coeff_a = torch.sin((1.0 - t) * omega) / sin_omega
        coeff_b = torch.sin(t * omega) / sin_omega

        return coeff_a * a + coeff_b * b

    @staticmethod
    def _trim_task_vector(task_vector: torch.Tensor, density: float) -> torch.Tensor:
        """Zero out values below the top-k% magnitude threshold."""
        flat = task_vector.flatten()
        k = max(1, int(len(flat) * density))
        threshold = flat.abs().topk(k).values[-1]
        mask = flat.abs() >= threshold
        return (task_vector.flatten() * mask.float()).reshape(task_vector.shape)

    @staticmethod
    def _elect_sign(
        trimmed_vectors: list[torch.Tensor],
        weights: list[float],
    ) -> torch.Tensor:
        """Elect consensus sign by weighted vote."""
        weighted_signs = sum(w * tv.sign() for w, tv in zip(weights, trimmed_vectors, strict=False))
        # +1 where positive dominates, -1 where negative dominates, 0 ties
        return weighted_signs.sign()
