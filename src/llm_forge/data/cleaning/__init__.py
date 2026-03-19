"""Data cleaning pipeline for llm-forge.

Orchestrates all cleaning steps in the canonical order:

1. Fix encoding (unicode repair, invisible chars, NFC normalization)
2. Identify language (FastText lid.176.bin)
3. Filter heuristically (Gopher/C4/FineWeb rules)
4. Deduplicate (exact SHA-256, fuzzy MinHash LSH, semantic embeddings)
5. Classify quality (FastText + KenLM with heuristic fallback)
6. Redact PII (Microsoft Presidio)
7. Filter toxicity (Detoxify)

Each step is optional and controlled via ``DataCleaningConfig``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_forge.data.cleaning.deduplication import (
    DeduplicationStats,
    Deduplicator,
    exact_dedup,
    fuzzy_dedup,
    semantic_dedup,
)
from llm_forge.data.cleaning.heuristic_filter import HeuristicFilter, HeuristicThresholds
from llm_forge.data.cleaning.language_filter import detect_language, filter_by_language
from llm_forge.data.cleaning.pii_redactor import PIIRedactor
from llm_forge.data.cleaning.quality_classifier import QUALITY_PRESETS, QualityClassifier
from llm_forge.data.cleaning.toxicity_filter import ToxicityFilter

# Import submodules for public re-export
from llm_forge.data.cleaning.unicode_fixer import fix_unicode
from llm_forge.data.cleaning.unicode_fixer import process_dataset as fix_unicode_dataset
from llm_forge.utils.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset

logger = get_logger("data.cleaning")

__all__ = [
    # Pipeline orchestrator
    "CleaningPipeline",
    "CleaningStats",
    # Unicode fixing
    "fix_unicode",
    "fix_unicode_dataset",
    # Language filtering
    "detect_language",
    "filter_by_language",
    # Heuristic filtering
    "HeuristicFilter",
    "HeuristicThresholds",
    # Deduplication
    "Deduplicator",
    "DeduplicationStats",
    "exact_dedup",
    "fuzzy_dedup",
    "semantic_dedup",
    # Quality classification
    "QualityClassifier",
    "QUALITY_PRESETS",
    # PII redaction
    "PIIRedactor",
    # Toxicity filtering
    "ToxicityFilter",
]


# ---------------------------------------------------------------------------
# Cleaning statistics
# ---------------------------------------------------------------------------


@dataclass
class CleaningStats:
    """Aggregate statistics from a full cleaning pipeline run.

    Tracks how many documents were present at each stage and how many were
    removed, along with timing information.
    """

    initial_count: int = 0
    final_count: int = 0

    # Per-step counts (records remaining after each step)
    after_unicode_fix: int = 0
    after_language_filter: int = 0
    after_heuristic_filter: int = 0
    after_deduplication: int = 0
    after_quality_filter: int = 0
    after_pii_redaction: int = 0
    after_toxicity_filter: int = 0

    # Per-step removed counts
    removed_by_language: int = 0
    removed_by_heuristic: int = 0
    removed_by_dedup: int = 0
    removed_by_quality: int = 0
    removed_by_toxicity: int = 0

    # Dedup sub-stats
    dedup_stats: DeduplicationStats | None = None

    # Timing (seconds per step)
    timing: dict[str, float] = field(default_factory=dict)

    # Steps that were skipped (disabled in config)
    skipped_steps: list[str] = field(default_factory=list)

    @property
    def total_removed(self) -> int:
        return self.initial_count - self.final_count

    @property
    def retention_rate(self) -> float:
        if self.initial_count == 0:
            return 0.0
        return self.final_count / self.initial_count

    @property
    def total_time(self) -> float:
        return sum(self.timing.values())

    def summary(self) -> str:
        """Return a human-readable summary of the cleaning run."""
        lines = [
            "=" * 60,
            "Data Cleaning Pipeline Summary",
            "=" * 60,
            f"Initial documents:  {self.initial_count:,}",
            f"Final documents:    {self.final_count:,}",
            f"Total removed:      {self.total_removed:,} ({(1 - self.retention_rate) * 100:.1f}%)",
            f"Retention rate:     {self.retention_rate * 100:.1f}%",
            "",
            "Per-step breakdown:",
        ]

        step_data = [
            ("Unicode fix", self.after_unicode_fix, 0),
            ("Language filter", self.after_language_filter, self.removed_by_language),
            ("Heuristic filter", self.after_heuristic_filter, self.removed_by_heuristic),
            ("Deduplication", self.after_deduplication, self.removed_by_dedup),
            ("Quality filter", self.after_quality_filter, self.removed_by_quality),
            ("PII redaction", self.after_pii_redaction, 0),
            ("Toxicity filter", self.after_toxicity_filter, self.removed_by_toxicity),
        ]

        for step_name, after_count, removed in step_data:
            if step_name.lower().replace(" ", "_") in [
                s.replace(" ", "_") for s in self.skipped_steps
            ]:
                lines.append(f"  {step_name:<22} [SKIPPED]")
            else:
                t = self.timing.get(step_name, 0.0)
                if removed > 0:
                    lines.append(
                        f"  {step_name:<22} {after_count:>8,} remaining (-{removed:,})  [{t:.1f}s]"
                    )
                else:
                    lines.append(f"  {step_name:<22} {after_count:>8,} remaining  [{t:.1f}s]")

        lines.append("")
        lines.append(f"Total pipeline time: {self.total_time:.1f}s")

        if self.skipped_steps:
            lines.append(f"Skipped steps: {', '.join(self.skipped_steps)}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Configuration interface
# ---------------------------------------------------------------------------
# We reference DataCleaningConfig from llm_forge.config.schema.
# At import time we try to load it; if the config module is not available
# (e.g. during isolated testing), we fall back to a simple dict-based
# interface.

try:
    from llm_forge.config.schema import DataCleaningConfig
except ImportError:
    DataCleaningConfig = None  # type: ignore[assignment,misc]


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Retrieve a value from a DataCleaningConfig, dict, or dataclass.

    Supports attribute access (Pydantic/dataclass) and dict-style access.
    """
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


# ---------------------------------------------------------------------------
# CleaningPipeline
# ---------------------------------------------------------------------------


class CleaningPipeline:
    """Orchestrates all data cleaning steps in canonical order.

    Parameters
    ----------
    config:
        A ``DataCleaningConfig`` (from ``llm_forge.config.schema``), a plain
        ``dict`` with the same keys, or *None* for all defaults.
    text_field:
        Name of the text column in the dataset.

    Examples
    --------
    >>> from llm_forge.data.cleaning import CleaningPipeline
    >>> pipeline = CleaningPipeline(config={"unicode_fix": True, "language_filter": ["en"]})
    >>> cleaned_dataset, stats = pipeline.run(dataset)
    >>> print(stats.summary())
    """

    def __init__(
        self,
        config: Any = None,
        text_field: str = "text",
    ) -> None:
        self.config = config
        self.text_field = text_field

        # Pre-initialize reusable components (lazy where expensive)
        self._heuristic_filter: HeuristicFilter | None = None
        self._deduplicator: Deduplicator | None = None
        self._quality_classifier: QualityClassifier | None = None
        self._pii_redactor: PIIRedactor | None = None
        self._toxicity_filter: ToxicityFilter | None = None

    # -- Lazy initialization of components ---------------------------------

    def _get_heuristic_filter(self) -> HeuristicFilter:
        if self._heuristic_filter is None:
            thresholds = HeuristicThresholds(
                min_word_count=int(_get_config_value(self.config, "min_word_count", 50)),
                max_word_count=int(_get_config_value(self.config, "max_word_count", 100_000)),
            )
            self._heuristic_filter = HeuristicFilter(thresholds=thresholds)
        return self._heuristic_filter

    def _get_deduplicator(self) -> Deduplicator:
        if self._deduplicator is None:
            self._deduplicator = Deduplicator(
                threshold=float(_get_config_value(self.config, "dedup_jaccard_threshold", 0.75)),
                num_perm=int(_get_config_value(self.config, "dedup_num_perm", 128)),
                shingle_size=int(_get_config_value(self.config, "dedup_shingle_size", 5)),
                semantic_model=str(
                    _get_config_value(
                        self.config,
                        "semantic_dedup_model",
                        "all-MiniLM-L6-v2",
                    )
                ),
                semantic_threshold=float(
                    _get_config_value(self.config, "semantic_dedup_threshold", 0.75)
                ),
            )
        return self._deduplicator

    def _get_quality_classifier(self) -> QualityClassifier:
        if self._quality_classifier is None:
            self._quality_classifier = QualityClassifier()
        return self._quality_classifier

    def _get_pii_redactor(self) -> PIIRedactor:
        if self._pii_redactor is None:
            entities = _get_config_value(self.config, "pii_entities", None)
            self._pii_redactor = PIIRedactor(
                entities=entities,
                strategy="redact",
            )
        return self._pii_redactor

    def _get_toxicity_filter(self) -> ToxicityFilter:
        if self._toxicity_filter is None:
            self._toxicity_filter = ToxicityFilter(model_name="unbiased")
        return self._toxicity_filter

    # -- Main pipeline run -------------------------------------------------

    def run(
        self,
        dataset: Dataset,
    ) -> tuple[Dataset, CleaningStats]:
        """Execute the full cleaning pipeline on a dataset.

        Runs cleaning steps in canonical order, skipping any that are
        disabled in the configuration.  Returns the cleaned dataset and
        detailed statistics.

        Parameters
        ----------
        dataset:
            A HuggingFace ``Dataset`` instance.

        Returns
        -------
        tuple[Dataset, CleaningStats]
            The cleaned dataset and statistics about what was filtered.
        """
        stats = CleaningStats(initial_count=len(dataset))
        enabled = _get_config_value(self.config, "enabled", True)

        if not enabled:
            logger.info("Data cleaning is disabled (config.enabled=False).")
            stats.final_count = len(dataset)
            return dataset, stats

        logger.info(
            "Starting data cleaning pipeline on %d records...",
            stats.initial_count,
        )

        # Validate text field exists
        if self.text_field not in dataset.column_names:
            raise ValueError(
                f"Text field '{self.text_field}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        # ---------------------------------------------------------------
        # Step 1: Fix unicode encoding
        # ---------------------------------------------------------------
        if _get_config_value(self.config, "unicode_fix", True):
            t0 = time.monotonic()
            logger.info("[1/7] Fixing unicode encoding...")
            dataset = fix_unicode_dataset(dataset, text_field=self.text_field)
            elapsed = time.monotonic() - t0
            stats.after_unicode_fix = len(dataset)
            stats.timing["Unicode fix"] = elapsed
        else:
            stats.after_unicode_fix = len(dataset)
            stats.skipped_steps.append("Unicode fix")

        # ---------------------------------------------------------------
        # Step 2: Language filtering
        # ---------------------------------------------------------------
        languages = _get_config_value(self.config, "language_filter", None)
        if languages:
            t0 = time.monotonic()
            count_before = len(dataset)
            threshold = float(_get_config_value(self.config, "language_confidence_threshold", 0.8))
            logger.info(
                "[2/7] Filtering by language (languages=%s, threshold=%.2f)...",
                languages,
                threshold,
            )
            dataset = filter_by_language(
                dataset,
                languages=languages,
                threshold=threshold,
                text_field=self.text_field,
            )
            elapsed = time.monotonic() - t0
            stats.after_language_filter = len(dataset)
            stats.removed_by_language = count_before - len(dataset)
            stats.timing["Language filter"] = elapsed
        else:
            stats.after_language_filter = len(dataset)
            stats.skipped_steps.append("Language filter")

        # ---------------------------------------------------------------
        # Step 3: Heuristic quality filtering
        # ---------------------------------------------------------------
        if _get_config_value(self.config, "heuristic_filter", True):
            t0 = time.monotonic()
            count_before = len(dataset)
            logger.info("[3/7] Applying heuristic quality filters...")
            hf = self._get_heuristic_filter()
            dataset = hf.filter_dataset(dataset, text_field=self.text_field)
            elapsed = time.monotonic() - t0
            stats.after_heuristic_filter = len(dataset)
            stats.removed_by_heuristic = count_before - len(dataset)
            stats.timing["Heuristic filter"] = elapsed
        else:
            stats.after_heuristic_filter = len(dataset)
            stats.skipped_steps.append("Heuristic filter")

        # ---------------------------------------------------------------
        # Step 4: Deduplication
        # ---------------------------------------------------------------
        if _get_config_value(self.config, "dedup_enabled", True):
            t0 = time.monotonic()
            count_before = len(dataset)

            # Build the list of tiers from config
            raw_tiers = _get_config_value(self.config, "dedup_tiers", ["exact", "fuzzy"])
            # Handle enum values from Pydantic
            tiers = [t.value if hasattr(t, "value") else str(t) for t in raw_tiers]

            # Check if semantic tier should be added
            if _get_config_value(self.config, "semantic_dedup_enabled", False):
                if "semantic" not in tiers:
                    tiers.append("semantic")

            logger.info("[4/7] Deduplicating (tiers=%s)...", tiers)
            try:
                deduplicator = self._get_deduplicator()
                dataset, dedup_stats = deduplicator.deduplicate(
                    dataset,
                    tiers=tiers,
                    text_field=self.text_field,
                )
                elapsed = time.monotonic() - t0
                stats.after_deduplication = len(dataset)
                stats.removed_by_dedup = count_before - len(dataset)
                stats.dedup_stats = dedup_stats
                stats.timing["Deduplication"] = elapsed
            except ImportError as exc:
                logger.warning("Deduplication skipped (missing dependency): %s", exc)
                stats.after_deduplication = len(dataset)
                stats.skipped_steps.append("Deduplication")
        else:
            stats.after_deduplication = len(dataset)
            stats.skipped_steps.append("Deduplication")

        # ---------------------------------------------------------------
        # Step 5: Quality classification
        # ---------------------------------------------------------------
        preset = _get_config_value(self.config, "quality_preset", "balanced")
        # Convert enum to string if necessary
        if hasattr(preset, "value"):
            preset = preset.value

        if preset:
            t0 = time.monotonic()
            count_before = len(dataset)
            logger.info("[5/7] Classifying quality (preset='%s')...", preset)
            try:
                qc = self._get_quality_classifier()
                dataset = qc.filter_dataset(dataset, preset=preset, text_field=self.text_field)
                elapsed = time.monotonic() - t0
                stats.after_quality_filter = len(dataset)
                stats.removed_by_quality = count_before - len(dataset)
                stats.timing["Quality filter"] = elapsed
            except ImportError as exc:
                logger.warning("Quality classification skipped (missing dependency): %s", exc)
                stats.after_quality_filter = len(dataset)
                stats.skipped_steps.append("Quality filter")
        else:
            stats.after_quality_filter = len(dataset)
            stats.skipped_steps.append("Quality filter")

        # ---------------------------------------------------------------
        # Step 6: PII redaction
        # ---------------------------------------------------------------
        if _get_config_value(self.config, "pii_redaction", False):
            t0 = time.monotonic()
            logger.info("[6/7] Redacting PII...")
            try:
                pii = self._get_pii_redactor()
                dataset = pii.redact_dataset(dataset, text_field=self.text_field)
            except ImportError as exc:
                logger.warning("PII redaction skipped: %s", exc)
                stats.skipped_steps.append("PII redaction")
            elapsed = time.monotonic() - t0
            stats.after_pii_redaction = len(dataset)
            stats.timing["PII redaction"] = elapsed
        else:
            stats.after_pii_redaction = len(dataset)
            stats.skipped_steps.append("PII redaction")

        # ---------------------------------------------------------------
        # Step 7: Toxicity filtering
        # ---------------------------------------------------------------
        if _get_config_value(self.config, "toxicity_filter", False):
            t0 = time.monotonic()
            count_before = len(dataset)
            toxicity_threshold = float(_get_config_value(self.config, "toxicity_threshold", 0.6))
            logger.info(
                "[7/7] Filtering toxicity (threshold=%.2f)...",
                toxicity_threshold,
            )
            try:
                tf = self._get_toxicity_filter()
                dataset = tf.filter_dataset(
                    dataset,
                    threshold=toxicity_threshold,
                    text_field=self.text_field,
                )
            except ImportError as exc:
                logger.warning("Toxicity filtering skipped: %s", exc)
                stats.skipped_steps.append("Toxicity filter")
            elapsed = time.monotonic() - t0
            stats.after_toxicity_filter = len(dataset)
            stats.removed_by_toxicity = count_before - len(dataset)
            stats.timing["Toxicity filter"] = elapsed
        else:
            stats.after_toxicity_filter = len(dataset)
            stats.skipped_steps.append("Toxicity filter")

        # ---------------------------------------------------------------
        # Finalize
        # ---------------------------------------------------------------
        stats.final_count = len(dataset)

        logger.info(
            "Data cleaning pipeline complete. %d -> %d records "
            "(retained %.1f%%, total time %.1fs).",
            stats.initial_count,
            stats.final_count,
            stats.retention_rate * 100,
            stats.total_time,
        )

        return dataset, stats
