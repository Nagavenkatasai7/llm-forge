"""Domain-specific evaluation module for llm-forge.

Loads custom evaluation datasets (JSONL format), runs generation-based
evaluation, and computes metrics like exact match, F1, and accuracy.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from llm_forge.evaluation.metrics import MetricsComputer
from llm_forge.utils.logging import get_logger

logger = get_logger("evaluation.domain_eval")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvalSample:
    """A single evaluation sample.

    Attributes
    ----------
    input_text : str
        The prompt / question / input to the model.
    reference : str
        The expected answer or output.
    metadata : dict
        Optional metadata (e.g. category, difficulty).
    """

    input_text: str
    reference: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result for a single evaluation sample.

    Attributes
    ----------
    input_text : str
        The original input.
    reference : str
        The expected output.
    prediction : str
        The model's generated output.
    metrics : dict
        Per-sample metrics.
    metadata : dict
        Forwarded metadata from the input sample.
    """

    input_text: str
    reference: str
    prediction: str
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path, input_field: str, output_field: str) -> list[EvalSample]:
    """Load evaluation samples from a JSONL file.

    Each line must be a JSON object containing at least *input_field* and
    *output_field*.  All other fields are stored in ``metadata``.
    """
    samples: list[EvalSample] = []

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", line_num, exc)
                continue

            if input_field not in record:
                logger.warning("Line %d missing field '%s'; skipping.", line_num, input_field)
                continue
            if output_field not in record:
                logger.warning("Line %d missing field '%s'; skipping.", line_num, output_field)
                continue

            metadata = {k: v for k, v in record.items() if k not in (input_field, output_field)}
            samples.append(
                EvalSample(
                    input_text=str(record[input_field]),
                    reference=str(record[output_field]),
                    metadata=metadata,
                )
            )

    logger.info("Loaded %d evaluation samples from '%s'.", len(samples), path)
    return samples


def _load_json_list(path: Path, input_field: str, output_field: str) -> list[EvalSample]:
    """Load evaluation samples from a JSON file containing a list of objects."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in '{path}', got {type(data).__name__}.")

    samples: list[EvalSample] = []
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            logger.warning("Item %d is not a dict; skipping.", idx)
            continue
        if input_field not in record or output_field not in record:
            logger.warning("Item %d missing required fields; skipping.", idx)
            continue
        metadata = {k: v for k, v in record.items() if k not in (input_field, output_field)}
        samples.append(
            EvalSample(
                input_text=str(record[input_field]),
                reference=str(record[output_field]),
                metadata=metadata,
            )
        )

    logger.info("Loaded %d evaluation samples from '%s'.", len(samples), path)
    return samples


# ---------------------------------------------------------------------------
# DomainEvaluator
# ---------------------------------------------------------------------------


class DomainEvaluator:
    """Evaluate a model on domain-specific, user-provided test sets.

    Supports generation-based evaluation where the model produces free-form
    answers that are compared against reference answers using configurable
    metrics (exact match, F1, accuracy, BLEU, ROUGE).

    Parameters
    ----------
    metrics : list[str], optional
        Metric names to compute.  Defaults to
        ``["exact_match", "f1", "accuracy"]``.
    input_field : str
        Name of the input / question field in the evaluation dataset.
    output_field : str
        Name of the reference / answer field in the evaluation dataset.

    Examples
    --------
    >>> evaluator = DomainEvaluator()
    >>> results = evaluator.evaluate(model, tokenizer, "test_data.jsonl")
    >>> print(results["aggregate"]["f1"])
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        input_field: str = "input",
        output_field: str = "output",
    ) -> None:
        self.metric_names = metrics or ["exact_match", "f1", "accuracy"]
        self.input_field = input_field
        self.output_field = output_field
        self._metrics = MetricsComputer()

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        path: str | Path,
        input_field: str | None = None,
        output_field: str | None = None,
        max_samples: int | None = None,
    ) -> list[EvalSample]:
        """Load an evaluation dataset from a file.

        Supports ``.jsonl`` and ``.json`` formats.

        Parameters
        ----------
        path:
            File path to the evaluation dataset.
        input_field:
            Override the input field name for this call.
        output_field:
            Override the output field name for this call.
        max_samples:
            Maximum number of samples to load (for quick testing).

        Returns
        -------
        list[EvalSample]
            Loaded evaluation samples.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {path}")

        in_f = input_field or self.input_field
        out_f = output_field or self.output_field

        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            samples = _load_jsonl(path, in_f, out_f)
        elif suffix == ".json":
            samples = _load_json_list(path, in_f, out_f)
        else:
            raise ValueError(f"Unsupported file format '{suffix}'. Use .jsonl or .json.")

        if max_samples is not None and max_samples < len(samples):
            samples = samples[:max_samples]
            logger.info("Truncated evaluation dataset to %d samples.", max_samples)

        return samples

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_predictions(
        self,
        model: Any,
        tokenizer: Any,
        samples: list[EvalSample],
        max_new_tokens: int = 256,
        batch_size: int = 4,
        temperature: float = 0.0,
        prompt_template: str | None = None,
        device: str | None = None,
    ) -> list[str]:
        """Generate model predictions for evaluation samples.

        Parameters
        ----------
        model:
            HuggingFace causal-LM model.
        tokenizer:
            Corresponding tokenizer.
        samples:
            Evaluation samples to generate predictions for.
        max_new_tokens:
            Maximum tokens to generate per sample.
        batch_size:
            Batch size for generation.
        temperature:
            Sampling temperature (0.0 = greedy).
        prompt_template:
            Optional template with ``{input}`` placeholder.
        device:
            Device override.

        Returns
        -------
        list[str]
            Generated predictions, one per sample.
        """
        if device is None:
            device = str(next(model.parameters()).device)

        model.eval()
        predictions: list[str] = []

        # Generation kwargs
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]

            # Format prompts
            prompts = []
            for sample in batch:
                if prompt_template:
                    prompt = prompt_template.format(input=sample.input_text)
                else:
                    prompt = sample.input_text
                prompts.append(prompt)

            encodings = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

            # Decode only the newly generated tokens
            for i, output_ids in enumerate(outputs):
                prompt_len = input_ids[i].shape[0]
                generated_ids = output_ids[prompt_len:]
                text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                predictions.append(text)

            if (start + batch_size) % (batch_size * 10) == 0:
                logger.info(
                    "Generated %d / %d predictions.",
                    min(start + batch_size, len(samples)),
                    len(samples),
                )

        return predictions

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        eval_dataset: str | Path | list[EvalSample],
        metrics: list[str] | None = None,
        max_new_tokens: int = 256,
        batch_size: int = 4,
        temperature: float = 0.0,
        prompt_template: str | None = None,
        max_samples: int | None = None,
        device: str | None = None,
        post_process_fn: Callable[[str], str] | None = None,
    ) -> dict[str, Any]:
        """Run domain-specific evaluation.

        Parameters
        ----------
        model:
            HuggingFace causal-LM model.
        tokenizer:
            Corresponding tokenizer.
        eval_dataset:
            Path to evaluation file (.jsonl / .json) or a pre-loaded list
            of :class:`EvalSample`.
        metrics:
            Metric names to compute (overrides instance default).
        max_new_tokens:
            Maximum new tokens per generation.
        batch_size:
            Batch size for generation.
        temperature:
            Sampling temperature.
        prompt_template:
            Optional template with ``{input}`` placeholder.
        max_samples:
            Cap on the number of evaluation samples.
        device:
            Device override.
        post_process_fn:
            Optional function applied to each prediction string before
            metric computation (e.g. extract final answer).

        Returns
        -------
        dict
            ``{"aggregate": {...}, "per_sample": [...], "metadata": {...}}``
        """
        metric_names = metrics or self.metric_names
        start_time = time.time()

        # Load dataset if path provided
        if isinstance(eval_dataset, (str, Path)):
            samples = self.load_dataset(eval_dataset, max_samples=max_samples)
        else:
            samples = eval_dataset
            if max_samples is not None and max_samples < len(samples):
                samples = samples[:max_samples]

        if not samples:
            logger.warning("No evaluation samples found; returning empty results.")
            return {"aggregate": {}, "per_sample": [], "metadata": {}}

        logger.info(
            "Evaluating %d samples with metrics: %s",
            len(samples),
            metric_names,
        )

        # Generate predictions
        predictions = self._generate_predictions(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            temperature=temperature,
            prompt_template=prompt_template,
            device=device,
        )

        # Apply post-processing
        if post_process_fn is not None:
            predictions = [post_process_fn(p) for p in predictions]

        references = [s.reference for s in samples]

        # Compute aggregate metrics
        aggregate = self._metrics.compute_all(
            predictions=predictions,
            references=references,
            include=metric_names,
        )

        # Build per-sample results
        per_sample: list[dict[str, Any]] = []
        for sample, pred in zip(samples, predictions, strict=False):
            sample_metrics = self._metrics.compute_all(
                predictions=[pred],
                references=[sample.reference],
                include=metric_names,
            )
            per_sample.append(
                {
                    "input": sample.input_text,
                    "reference": sample.reference,
                    "prediction": pred,
                    "metrics": sample_metrics,
                    "metadata": sample.metadata,
                }
            )

        elapsed = time.time() - start_time

        # Category breakdown if available
        category_results = self._compute_category_breakdown(per_sample, metric_names)

        result = {
            "aggregate": aggregate,
            "per_sample": per_sample,
            "category_breakdown": category_results,
            "metadata": {
                "num_samples": len(samples),
                "metrics_computed": metric_names,
                "elapsed_seconds": round(elapsed, 2),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        }

        logger.info(
            "Domain evaluation complete: %d samples in %.1f seconds. Aggregate: %s",
            len(samples),
            elapsed,
            {k: round(v, 4) if isinstance(v, float) else v for k, v in aggregate.items()},
        )

        return result

    # ------------------------------------------------------------------
    # Category breakdown
    # ------------------------------------------------------------------

    def _compute_category_breakdown(
        self,
        per_sample: list[dict[str, Any]],
        metric_names: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Group results by category metadata and compute per-category metrics."""
        categories: dict[str, list[dict[str, Any]]] = {}

        for item in per_sample:
            category = item.get("metadata", {}).get("category", "uncategorized")
            categories.setdefault(str(category), []).append(item)

        if len(categories) <= 1 and "uncategorized" in categories:
            return {}

        breakdown: dict[str, dict[str, Any]] = {}
        for cat_name, cat_items in sorted(categories.items()):
            preds = [item["prediction"] for item in cat_items]
            refs = [item["reference"] for item in cat_items]
            cat_metrics = self._metrics.compute_all(preds, refs, include=metric_names)
            breakdown[cat_name] = {
                "count": len(cat_items),
                "metrics": cat_metrics,
            }

        return breakdown

    # ------------------------------------------------------------------
    # Evaluate from pre-computed predictions
    # ------------------------------------------------------------------

    def evaluate_predictions(
        self,
        predictions: list[str],
        references: list[str],
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute metrics from pre-generated predictions and references.

        This is useful when predictions have been generated externally
        (e.g. via an API or a different serving framework).

        Parameters
        ----------
        predictions:
            List of model-generated strings.
        references:
            Corresponding ground-truth strings.
        metrics:
            Metric names to compute.

        Returns
        -------
        dict
            ``{"aggregate": {...}, "per_sample": [...]}``
        """
        metric_names = metrics or self.metric_names

        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references ({len(references)}) "
                "must have the same length."
            )

        aggregate = self._metrics.compute_all(predictions, references, include=metric_names)

        per_sample: list[dict[str, Any]] = []
        for pred, ref in zip(predictions, references, strict=False):
            sample_metrics = self._metrics.compute_all([pred], [ref], include=metric_names)
            per_sample.append(
                {
                    "prediction": pred,
                    "reference": ref,
                    "metrics": sample_metrics,
                }
            )

        return {
            "aggregate": aggregate,
            "per_sample": per_sample,
        }
