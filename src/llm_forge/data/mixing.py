"""Domain/general data ratio optimization for training data mixing."""

from __future__ import annotations

import logging
import math
import random

from datasets import Dataset, concatenate_datasets

logger = logging.getLogger(__name__)

DEFAULT_DOMAIN_WEIGHTS = {
    "general": 0.4,
    "domain": 0.4,
    "synthetic": 0.2,
}


class DataMixer:
    """Mix multiple data sources with configurable ratios for optimal training."""

    def __init__(
        self,
        seed: int = 42,
        temperature: float = 1.0,
    ):
        self.seed = seed
        self.temperature = temperature
        self.rng = random.Random(seed)

    def mix_datasets(
        self,
        datasets: dict[str, Dataset],
        weights: dict[str, float] | None = None,
        total_samples: int | None = None,
    ) -> Dataset:
        """Mix multiple datasets according to specified weights.

        Args:
            datasets: Mapping of source name to Dataset.
            weights: Sampling weights per source. If None, uses uniform weights.
            total_samples: Target total samples. If None, uses sum of all datasets.

        Returns:
            Mixed dataset with _source metadata field.
        """
        if not datasets:
            raise ValueError("At least one dataset must be provided")

        if weights is None:
            weights = {name: 1.0 / len(datasets) for name in datasets}

        total_weight = sum(weights.get(name, 0) for name in datasets)
        if total_weight == 0:
            raise ValueError("Total weight must be positive")

        normalized_weights = {name: weights.get(name, 0) / total_weight for name in datasets}

        if total_samples is None:
            total_samples = sum(len(ds) for ds in datasets.values())

        logger.info(f"Mixing {len(datasets)} datasets into {total_samples} total samples")

        mixed_records = []
        for name, ds in datasets.items():
            target_count = int(total_samples * normalized_weights.get(name, 0))
            target_count = min(target_count, len(ds))

            if target_count <= 0:
                continue

            if target_count < len(ds):
                indices = self.rng.sample(range(len(ds)), target_count)
                indices.sort()
                selected = ds.select(indices)
            else:
                selected = ds

            for record in selected:
                record_dict = dict(record)
                record_dict["_source"] = name
                mixed_records.append(record_dict)

            logger.info(
                f"  {name}: {len(selected)} samples (weight={normalized_weights[name]:.2f})"
            )

        self.rng.shuffle(mixed_records)
        return Dataset.from_list(mixed_records)

    def compute_optimal_weights(
        self,
        datasets: dict[str, Dataset],
        method: str = "proportional",
    ) -> dict[str, float]:
        """Compute optimal mixing weights using various strategies.

        Args:
            datasets: Mapping of source name to Dataset.
            method: "proportional" (by size), "equal", "temperature" (size-scaled by temp).

        Returns:
            Optimal weight per source.
        """
        sizes = {name: len(ds) for name, ds in datasets.items()}
        total = sum(sizes.values())

        if method == "equal":
            return {name: 1.0 / len(datasets) for name in datasets}

        if method == "proportional":
            return {name: size / total for name, size in sizes.items()}

        if method == "temperature":
            scaled = {
                name: math.pow(size / total, 1.0 / self.temperature) for name, size in sizes.items()
            }
            scale_total = sum(scaled.values())
            return {name: v / scale_total for name, v in scaled.items()}

        raise ValueError(f"Unknown method: {method}. Use 'proportional', 'equal', or 'temperature'")

    def upsample_dataset(
        self,
        dataset: Dataset,
        target_size: int,
    ) -> Dataset:
        """Upsample a dataset to reach a target size by repeating examples."""
        current_size = len(dataset)
        if current_size >= target_size:
            return dataset

        repeats = target_size // current_size
        remainder = target_size % current_size

        parts = [dataset] * repeats
        if remainder > 0:
            indices = self.rng.sample(range(current_size), remainder)
            parts.append(dataset.select(indices))

        return concatenate_datasets(parts).shuffle(seed=self.seed)

    def downsample_dataset(
        self,
        dataset: Dataset,
        target_size: int,
    ) -> Dataset:
        """Downsample a dataset to a target size."""
        if len(dataset) <= target_size:
            return dataset

        indices = self.rng.sample(range(len(dataset)), target_size)
        return dataset.select(sorted(indices))
