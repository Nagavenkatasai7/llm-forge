"""Inference-Time Intervention (ITI) probing module.

Discovers 'truthfulness directions' in a model's attention heads by
probing with a labelled dataset (e.g. TruthfulQA).  The directions can
then be baked into the model weights by the ITIBaker.

Reference: Li et al., "Inference-Time Intervention: Eliciting Truthful
Answers from a Language Model" (2023).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Container for ITI probing results."""

    directions: dict[tuple[int, int], np.ndarray]  # (layer, head) -> direction vector
    top_heads: list[tuple[int, int]]  # ranked (layer, head) pairs
    sigmas: dict[tuple[int, int], float]  # (layer, head) -> std of projections
    probe_accuracies: dict[tuple[int, int], float]  # (layer, head) -> probe accuracy
    num_layers: int = 0
    num_heads_per_layer: int = 0
    head_dim: int = 0


class ITIProber:
    """Probe a model's attention heads to find truthfulness directions.

    Parameters
    ----------
    num_probing_samples : int
        Maximum samples to draw from the probing dataset.
    num_heads : int
        Number of top attention heads to select for intervention.
    method : str
        Direction-finding method: 'center_of_mass' or 'linear_probe'.
    """

    def __init__(
        self,
        num_probing_samples: int = 500,
        num_heads: int = 48,
        method: str = "center_of_mass",
    ) -> None:
        self.num_probing_samples = num_probing_samples
        self.num_heads = num_heads
        self.method = method

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe(
        self,
        model: Any,
        tokenizer: Any,
        config: Any,
    ) -> ProbeResult:
        """Full ITI probing pipeline.

        1. Load probing dataset
        2. Extract per-head activations
        3. Train probes / compute directions
        4. Select top-K heads
        5. Compute scaling factors

        Returns a ProbeResult with everything the ITIBaker needs.
        """
        dataset_name = getattr(config, "probing_dataset", "truthful_qa")
        statements, labels = self._load_probing_data(dataset_name, self.num_probing_samples)

        logger.info("Extracting activations from %d probing samples ...", len(statements))
        activations = self.extract_activations(model, tokenizer, statements)

        num_layers = len(activations)
        num_heads_per_layer = activations[0].shape[1]
        head_dim = activations[0].shape[2]

        logger.info(
            "Model geometry: %d layers, %d heads/layer, head_dim=%d",
            num_layers,
            num_heads_per_layer,
            head_dim,
        )

        labels_arr = np.array(labels)

        if self.method == "linear_probe":
            probe_accs = self.train_probes(activations, labels_arr)
        else:
            probe_accs = self._center_of_mass_accuracy(activations, labels_arr)

        top_heads = self.select_top_heads(probe_accs, self.num_heads)
        directions = self.compute_directions(activations, labels_arr, top_heads)
        sigmas = self.compute_scaling(activations, directions, top_heads)

        # Log top head accuracies
        for layer, head in top_heads[:10]:
            logger.info(
                "  Head (L=%d, H=%d): accuracy=%.3f",
                layer,
                head,
                probe_accs[(layer, head)],
            )

        return ProbeResult(
            directions=directions,
            top_heads=top_heads,
            sigmas=sigmas,
            probe_accuracies=probe_accs,
            num_layers=num_layers,
            num_heads_per_layer=num_heads_per_layer,
            head_dim=head_dim,
        )

    # ------------------------------------------------------------------
    # Activation extraction
    # ------------------------------------------------------------------

    def extract_activations(
        self,
        model: Any,
        tokenizer: Any,
        statements: list[str],
    ) -> list[np.ndarray]:
        """Run statements through the model and collect per-head activations.

        Uses forward hooks on each layer's ``self_attn.o_proj`` to capture
        the concatenated head outputs.  Extracts the last-token position
        and reshapes to ``[num_heads, head_dim]``.

        Returns
        -------
        list[np.ndarray]
            One array per layer, each of shape ``[num_samples, num_heads, head_dim]``.
        """
        model.eval()
        device = next(model.parameters()).device

        # Detect model architecture
        layers = self._get_model_layers(model)
        num_layers = len(layers)

        # Storage: layer_idx -> list of [num_heads, head_dim] arrays
        all_activations: dict[int, list[np.ndarray]] = {i: [] for i in range(num_layers)}

        # Register hooks
        hooks = []
        hook_storage: dict[int, Tensor | None] = {}

        for layer_idx, layer in enumerate(layers):
            o_proj = self._get_o_proj(layer)
            if o_proj is None:
                logger.warning("Could not find o_proj for layer %d", layer_idx)
                continue

            def make_hook(idx: int):
                def hook_fn(module, input):
                    # input[0] shape: [batch, seq_len, num_heads * head_dim]
                    hook_storage[idx] = input[0].detach()

                return hook_fn

            h = o_proj.register_forward_pre_hook(make_hook(layer_idx))
            hooks.append(h)

        # Determine geometry from first layer's o_proj
        first_o_proj = self._get_o_proj(layers[0])
        hidden_size = first_o_proj.in_features
        # Get num_heads from model config
        model_config = model.config
        num_heads = getattr(
            model_config,
            "num_attention_heads",
            getattr(model_config, "n_head", 32),
        )
        head_dim = hidden_size // num_heads

        try:
            for stmt in statements:
                inputs = tokenizer(
                    stmt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=False,
                ).to(device)

                with torch.no_grad():
                    model(**inputs)

                # Collect from hooks
                for layer_idx in range(num_layers):
                    if layer_idx not in hook_storage or hook_storage[layer_idx] is None:
                        continue
                    act = hook_storage[layer_idx]  # [1, seq_len, hidden]
                    # Take last token
                    last_token_act = act[0, -1, :]  # [hidden_size]
                    # Reshape to [num_heads, head_dim]
                    reshaped = last_token_act.reshape(num_heads, head_dim)
                    all_activations[layer_idx].append(reshaped.cpu().numpy())

                # Clear storage
                hook_storage.clear()
        finally:
            for h in hooks:
                h.remove()

        # Stack into arrays: [num_samples, num_heads, head_dim]
        result = []
        for layer_idx in range(num_layers):
            if all_activations[layer_idx]:
                result.append(np.stack(all_activations[layer_idx], axis=0))
            else:
                result.append(np.zeros((len(statements), num_heads, head_dim)))

        return result

    # ------------------------------------------------------------------
    # Probe training
    # ------------------------------------------------------------------

    def train_probes(
        self,
        activations: list[np.ndarray],
        labels: np.ndarray,
    ) -> dict[tuple[int, int], float]:
        """Train a LogisticRegression probe per (layer, head).

        Returns accuracy dict keyed by (layer_idx, head_idx).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        accuracies: dict[tuple[int, int], float] = {}

        for layer_idx, layer_acts in enumerate(activations):
            num_heads = layer_acts.shape[1]
            for head_idx in range(num_heads):
                head_acts = layer_acts[:, head_idx, :]  # [N, head_dim]
                clf = LogisticRegression(max_iter=1000, solver="lbfgs")
                try:
                    scores = cross_val_score(
                        clf, head_acts, labels, cv=min(5, len(labels)), scoring="accuracy"
                    )
                    accuracies[(layer_idx, head_idx)] = float(scores.mean())
                except Exception:
                    accuracies[(layer_idx, head_idx)] = 0.5

        return accuracies

    # ------------------------------------------------------------------
    # Head selection
    # ------------------------------------------------------------------

    def select_top_heads(
        self,
        probe_accuracies: dict[tuple[int, int], float],
        k: int,
    ) -> list[tuple[int, int]]:
        """Return the top-K heads ranked by probe accuracy."""
        sorted_heads = sorted(probe_accuracies.items(), key=lambda x: x[1], reverse=True)
        return [head for head, _ in sorted_heads[:k]]

    # ------------------------------------------------------------------
    # Direction computation
    # ------------------------------------------------------------------

    def compute_directions(
        self,
        activations: list[np.ndarray],
        labels: np.ndarray,
        top_heads: list[tuple[int, int]],
    ) -> dict[tuple[int, int], np.ndarray]:
        """Compute center-of-mass truthfulness direction per selected head.

        Direction = mean(true_activations) - mean(false_activations).
        """
        true_mask = labels == 1
        false_mask = labels == 0

        directions: dict[tuple[int, int], np.ndarray] = {}
        for layer_idx, head_idx in top_heads:
            head_acts = activations[layer_idx][:, head_idx, :]  # [N, head_dim]
            mean_true = head_acts[true_mask].mean(axis=0)
            mean_false = head_acts[false_mask].mean(axis=0)
            directions[(layer_idx, head_idx)] = mean_true - mean_false

        return directions

    def compute_scaling(
        self,
        activations: list[np.ndarray],
        directions: dict[tuple[int, int], np.ndarray],
        top_heads: list[tuple[int, int]],
    ) -> dict[tuple[int, int], float]:
        """Compute std of projections onto the direction for normalization."""
        sigmas: dict[tuple[int, int], float] = {}
        for layer_idx, head_idx in top_heads:
            head_acts = activations[layer_idx][:, head_idx, :]  # [N, head_dim]
            direction = directions[(layer_idx, head_idx)]
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                sigmas[(layer_idx, head_idx)] = 1.0
                continue
            unit_dir = direction / norm
            projections = head_acts @ unit_dir  # [N]
            sigmas[(layer_idx, head_idx)] = float(np.std(projections))

        return sigmas

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _center_of_mass_accuracy(
        self,
        activations: list[np.ndarray],
        labels: np.ndarray,
    ) -> dict[tuple[int, int], float]:
        """Estimate accuracy using center-of-mass classification.

        For each head, compute the direction (mean_true - mean_false),
        project all samples, and check classification accuracy using
        the midpoint as threshold.
        """
        true_mask = labels == 1
        false_mask = labels == 0

        accuracies: dict[tuple[int, int], float] = {}
        for layer_idx, layer_acts in enumerate(activations):
            num_heads = layer_acts.shape[1]
            for head_idx in range(num_heads):
                head_acts = layer_acts[:, head_idx, :]
                mean_true = head_acts[true_mask].mean(axis=0)
                mean_false = head_acts[false_mask].mean(axis=0)
                direction = mean_true - mean_false
                norm = np.linalg.norm(direction)
                if norm < 1e-8:
                    accuracies[(layer_idx, head_idx)] = 0.5
                    continue
                unit_dir = direction / norm
                projections = head_acts @ unit_dir
                threshold = projections.mean()
                preds = (projections > threshold).astype(int)
                accuracies[(layer_idx, head_idx)] = float((preds == labels).mean())

        return accuracies

    def _load_probing_data(
        self, dataset_name: str, max_samples: int
    ) -> tuple[list[str], list[int]]:
        """Load a probing dataset and return (statements, labels).

        For TruthfulQA, labels are 1 (true) or 0 (false).
        """
        from datasets import load_dataset

        statements: list[str] = []
        labels: list[int] = []

        if "truthful_qa" in dataset_name:
            ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
            for item in ds:
                if len(statements) >= max_samples:
                    break
                question = item["question"]
                choices = item["mc1_targets"]["choices"]
                choice_labels = item["mc1_targets"]["labels"]

                for choice, label in zip(choices, choice_labels, strict=False):
                    if len(statements) >= max_samples:
                        break
                    stmt = f"{question} {choice}"
                    statements.append(stmt)
                    labels.append(int(label))
        else:
            # Generic HF dataset — expect 'text' and 'label' columns
            ds = load_dataset(dataset_name, split="validation")
            for item in ds:
                if len(statements) >= max_samples:
                    break
                statements.append(str(item.get("text", item.get("sentence", ""))))
                labels.append(int(item.get("label", 0)))

        logger.info(
            "Loaded %d probing samples (%d true, %d false)",
            len(statements),
            sum(labels),
            len(labels) - sum(labels),
        )
        return statements, labels

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
        """Get the o_proj (output projection) linear layer from an attention layer."""
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            return layer.self_attn.o_proj
        if hasattr(layer, "attn") and hasattr(layer.attn, "c_proj"):
            return layer.attn.c_proj
        if hasattr(layer, "attention") and hasattr(layer.attention, "dense"):
            return layer.attention.dense
        return None
