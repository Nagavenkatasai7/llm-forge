"""ITI Baker — bake truthfulness directions into model weights.

Takes the probing results from ITIProber and permanently encodes the
truthfulness shift as o_proj biases in selected attention layers.
This produces zero inference overhead and is compatible with Ollama/GGUF
export pipelines.

Reference: Li et al., "Inference-Time Intervention: Eliciting Truthful
Answers from a Language Model" (2023).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.nn import Parameter

logger = logging.getLogger(__name__)


class ITIBaker:
    """Bake ITI truthfulness directions into model weights.

    After baking, the model's o_proj layers in selected attention heads
    carry a permanent bias that steers activations toward the truthful
    direction.  No hooks or runtime overhead required.
    """

    def bake_interventions(
        self,
        model: Any,
        directions: dict[tuple[int, int], np.ndarray],
        top_heads: list[tuple[int, int]],
        alpha: float,
        sigmas: dict[tuple[int, int], float],
    ) -> Any:
        """Add truthfulness biases to the model's o_proj layers.

        Parameters
        ----------
        model : PreTrainedModel
            The fine-tuned model to modify in-place.
        directions : dict
            Mapping of (layer_idx, head_idx) -> direction vector (np.ndarray).
        top_heads : list
            List of (layer_idx, head_idx) tuples to intervene on.
        alpha : float
            Intervention strength scalar.
        sigmas : dict
            Mapping of (layer_idx, head_idx) -> std of projections for scaling.

        Returns
        -------
        model
            The modified model (also modified in-place).
        """
        layers = self._get_model_layers(model)
        device = next(model.parameters()).device

        # Detect geometry
        first_o_proj = self._get_o_proj(layers[0])
        hidden_size = first_o_proj.in_features
        model_config = model.config
        num_heads = getattr(
            model_config,
            "num_attention_heads",
            getattr(model_config, "n_head", 32),
        )
        head_dim = hidden_size // num_heads

        # Group heads by layer
        grouped: dict[int, list[tuple[int, np.ndarray, float]]] = defaultdict(list)
        for layer_idx, head_idx in top_heads:
            if (layer_idx, head_idx) not in directions:
                continue
            direction = directions[(layer_idx, head_idx)]
            sigma = sigmas.get((layer_idx, head_idx), 1.0)
            grouped[layer_idx].append((head_idx, direction, sigma))

        num_modified = 0

        for layer_idx, heads_info in grouped.items():
            if layer_idx >= len(layers):
                logger.warning(
                    "Layer %d out of range (model has %d layers), skipping",
                    layer_idx,
                    len(layers),
                )
                continue

            o_proj = self._get_o_proj(layers[layer_idx])
            if o_proj is None:
                logger.warning("No o_proj found for layer %d, skipping", layer_idx)
                continue

            # Start with existing bias or zeros
            if o_proj.bias is not None:
                displacement = o_proj.bias.data.clone()
            else:
                displacement = torch.zeros(hidden_size, device=device, dtype=o_proj.weight.dtype)

            for head_idx, direction, sigma in heads_info:
                direction_t = torch.tensor(direction, device=device, dtype=o_proj.weight.dtype)
                norm = direction_t.norm()
                if norm < 1e-8:
                    continue
                direction_t = direction_t / norm

                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim
                displacement[start:end] += alpha * sigma * direction_t

            # Set bias as a parameter
            o_proj.bias = Parameter(displacement)
            num_modified += 1

        # Mark that attention layers now have bias
        if hasattr(model, "config"):
            model.config.attention_bias = True

        logger.info(
            "ITI baking complete: modified %d layers, %d heads total (alpha=%.1f)",
            num_modified,
            len(top_heads),
            alpha,
        )

        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model_layers(self, model: Any) -> list:
        """Get the list of transformer layers from the model."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return list(model.transformer.h)
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            return list(model.gpt_neox.layers)
        raise ValueError(
            "Cannot find transformer layers. Supported architectures: "
            "LlamaForCausalLM, GPT2LMHeadModel, GPTNeoXForCausalLM"
        )

    def _get_o_proj(self, layer: Any) -> Any:
        """Get the o_proj linear layer from an attention layer."""
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            return layer.self_attn.o_proj
        if hasattr(layer, "attn") and hasattr(layer.attn, "c_proj"):
            return layer.attn.c_proj
        if hasattr(layer, "attention") and hasattr(layer.attention, "dense"):
            return layer.attention.dense
        return None
