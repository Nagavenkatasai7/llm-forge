"""Benchmark runner for llm-forge evaluation.

Integrates with EleutherAI's lm-evaluation-harness for standard benchmarks
(MMLU, HellaSwag, ARC, WinoGrande, TruthfulQA, GSM8K).  Falls back to a
simple perplexity-based evaluation when lm-eval is not installed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from llm_forge.evaluation.metrics import MetricsComputer
from llm_forge.utils.logging import get_logger

logger = get_logger("evaluation.benchmarks")

# ---------------------------------------------------------------------------
# Optional lm-eval import
# ---------------------------------------------------------------------------

try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    _LM_EVAL_AVAILABLE = True
except ImportError:
    _LM_EVAL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Standard benchmark task registry
# ---------------------------------------------------------------------------

STANDARD_TASKS: dict[str, dict[str, Any]] = {
    "mmlu": {
        "task_name": "mmlu",
        "display_name": "MMLU",
        "description": "Massive Multitask Language Understanding",
        "default_fewshot": 5,
        "metric_key": "acc",
    },
    "hellaswag": {
        "task_name": "hellaswag",
        "display_name": "HellaSwag",
        "description": "Commonsense Natural Language Inference",
        "default_fewshot": 10,
        "metric_key": "acc_norm",
    },
    "arc_easy": {
        "task_name": "arc_easy",
        "display_name": "ARC-Easy",
        "description": "AI2 Reasoning Challenge (Easy)",
        "default_fewshot": 25,
        "metric_key": "acc_norm",
    },
    "arc_challenge": {
        "task_name": "arc_challenge",
        "display_name": "ARC-Challenge",
        "description": "AI2 Reasoning Challenge (Challenge)",
        "default_fewshot": 25,
        "metric_key": "acc_norm",
    },
    "winogrande": {
        "task_name": "winogrande",
        "display_name": "WinoGrande",
        "description": "Winograd Schema Challenge at Scale",
        "default_fewshot": 5,
        "metric_key": "acc",
    },
    "truthfulqa_mc2": {
        "task_name": "truthfulqa_mc2",
        "display_name": "TruthfulQA (MC2)",
        "description": "Measuring Truthfulness in Language Models",
        "default_fewshot": 0,
        "metric_key": "acc",
    },
    "gsm8k": {
        "task_name": "gsm8k",
        "display_name": "GSM8K",
        "description": "Grade School Math (8K)",
        "default_fewshot": 5,
        "metric_key": "exact_match",
    },
    "ifeval": {
        "task_name": "ifeval",
        "display_name": "IFEval",
        "description": "Instruction-Following Evaluation (verifiable instructions)",
        "default_fewshot": 0,
        "metric_key": "prompt_level_strict_acc",
    },
}

# Aliases so users can refer to tasks by common shorthand
TASK_ALIASES: dict[str, str] = {
    "arc": "arc_challenge",
    "truthfulqa": "truthfulqa_mc2",
    "gsm": "gsm8k",
    "instruction_following": "ifeval",
}


def _resolve_task_name(name: str) -> str:
    """Resolve a user-provided task name to its canonical form."""
    lower = name.lower().strip()
    return TASK_ALIASES.get(lower, lower)


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Runs standardised benchmarks and compares model performance.

    Integrates with EleutherAI's lm-evaluation-harness when available.
    Falls back to perplexity-based evaluation otherwise.

    Parameters
    ----------
    device : str, optional
        Device for inference (e.g. ``"cuda"``, ``"cpu"``).  Auto-detected
        when *None*.
    cache_dir : str or Path, optional
        Directory for caching downloaded evaluation datasets.

    Examples
    --------
    >>> runner = BenchmarkRunner()
    >>> results = runner.run_benchmarks("./my-model", tasks=["hellaswag"])
    >>> print(results["hellaswag"]["score"])
    """

    def __init__(
        self,
        device: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._metrics = MetricsComputer()

        if _LM_EVAL_AVAILABLE:
            logger.info("lm-evaluation-harness detected; full benchmark support enabled.")
        else:
            logger.warning(
                "lm-evaluation-harness not installed. Install with: "
                "pip install 'llm-forge[eval]'.  Falling back to perplexity evaluation."
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_benchmarks(
        self,
        model_path: str,
        tasks: list[str] | None = None,
        num_fewshot: int | None = None,
        batch_size: int = 8,
        limit: int | None = None,
        trust_remote_code: bool = False,
        apply_chat_template: bool = True,
    ) -> dict[str, Any]:
        """Run evaluation benchmarks on a model.

        Parameters
        ----------
        model_path:
            Path to a local model directory or a HuggingFace model identifier.
        tasks:
            List of benchmark task names to run. When *None*, defaults to
            ``["mmlu", "hellaswag", "arc_challenge", "winogrande"]``.
        num_fewshot:
            Number of few-shot examples. If *None*, uses the per-task default.
        batch_size:
            Batch size for inference.
        limit:
            Maximum number of samples per task (useful for quick testing).
        trust_remote_code:
            Whether to trust remote code in the model repository.
        apply_chat_template:
            Apply the model's chat template during evaluation.  Critical
            for instruction-tuned models, especially on IFEval.

        Returns
        -------
        dict
            Structured results with per-task scores and metadata.
        """
        if tasks is None:
            tasks = ["mmlu", "hellaswag", "arc_challenge", "winogrande"]

        resolved_tasks = [_resolve_task_name(t) for t in tasks]

        logger.info(
            "Running benchmarks on '%s' with tasks: %s",
            model_path,
            resolved_tasks,
        )

        start_time = time.time()

        if _LM_EVAL_AVAILABLE:
            results = self._run_lm_eval(
                model_path=model_path,
                tasks=resolved_tasks,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                limit=limit,
                trust_remote_code=trust_remote_code,
                apply_chat_template=apply_chat_template,
            )
        else:
            results = self._run_perplexity_fallback(
                model_path=model_path,
                trust_remote_code=trust_remote_code,
            )

        elapsed = time.time() - start_time
        results["_metadata"] = {
            "model_path": model_path,
            "tasks_requested": resolved_tasks,
            "backend": "lm_eval" if _LM_EVAL_AVAILABLE else "perplexity_fallback",
            "elapsed_seconds": round(elapsed, 2),
            "device": self.device,
        }

        logger.info("Benchmark evaluation completed in %.1f seconds.", elapsed)
        return results

    # ------------------------------------------------------------------
    # lm-evaluation-harness backend
    # ------------------------------------------------------------------

    def _run_lm_eval(
        self,
        model_path: str,
        tasks: list[str],
        num_fewshot: int | None,
        batch_size: int,
        limit: int | None,
        trust_remote_code: bool,
        apply_chat_template: bool = True,
    ) -> dict[str, Any]:
        """Run benchmarks using lm-evaluation-harness."""
        task_names_for_harness = []
        fewshot_map: dict[str, int] = {}

        for task_name in tasks:
            task_info = STANDARD_TASKS.get(task_name)
            if task_info:
                task_names_for_harness.append(task_info["task_name"])
                fewshot = num_fewshot if num_fewshot is not None else task_info["default_fewshot"]
                fewshot_map[task_info["task_name"]] = fewshot
            else:
                # Assume user knows the exact lm-eval task name
                task_names_for_harness.append(task_name)
                fewshot_map[task_name] = num_fewshot if num_fewshot is not None else 0

        # Use a single num_fewshot for the harness call; per-task fewshot
        # requires multiple calls, so we use the most common value.
        default_fewshot = max(set(fewshot_map.values()), key=list(fewshot_map.values()).count)

        try:
            lm_obj = HFLM(
                pretrained=model_path,
                device=self.device,
                batch_size=batch_size,
                trust_remote_code=trust_remote_code,
            )

            eval_results = lm_eval.simple_evaluate(
                model=lm_obj,
                tasks=task_names_for_harness,
                num_fewshot=default_fewshot,
                batch_size=batch_size,
                limit=limit,
                log_samples=False,
                apply_chat_template=apply_chat_template,
            )
        except Exception as exc:
            logger.error("lm-eval evaluation failed: %s", exc)
            logger.info("Falling back to perplexity evaluation.")
            return self._run_perplexity_fallback(
                model_path=model_path,
                trust_remote_code=trust_remote_code,
            )

        return self._parse_lm_eval_results(eval_results, tasks)

    def _parse_lm_eval_results(
        self, raw_results: dict[str, Any], requested_tasks: list[str]
    ) -> dict[str, Any]:
        """Parse lm-eval results into a standardised format."""
        parsed: dict[str, Any] = {}
        results_dict = raw_results.get("results", {})

        for task_name in requested_tasks:
            task_info = STANDARD_TASKS.get(task_name, {})
            harness_name = task_info.get("task_name", task_name)
            metric_key = task_info.get("metric_key", "acc")
            display_name = task_info.get("display_name", task_name)

            task_results = results_dict.get(harness_name, {})

            if not task_results:
                logger.warning("No results found for task '%s'.", harness_name)
                parsed[task_name] = {
                    "display_name": display_name,
                    "score": None,
                    "metric": metric_key,
                    "raw": {},
                }
                continue

            # Extract the primary score
            # lm-eval stores metrics with comma-separated filter suffix
            score = None
            for key, value in task_results.items():
                if key.startswith(metric_key) and isinstance(value, (int, float)):
                    score = value
                    break

            # Also check the stderr variant
            stderr = None
            for key, value in task_results.items():
                if key.startswith(f"{metric_key}_stderr") and isinstance(value, (int, float)):
                    stderr = value
                    break

            parsed[task_name] = {
                "display_name": display_name,
                "score": score,
                "score_stderr": stderr,
                "metric": metric_key,
                "num_fewshot": task_results.get("num_fewshot"),
                "raw": {
                    k: v for k, v in task_results.items() if isinstance(v, (int, float, str, bool))
                },
            }

        # Compute aggregate
        scores = [
            v["score"]
            for v in parsed.values()
            if isinstance(v, dict) and v.get("score") is not None
        ]
        if scores:
            parsed["_aggregate"] = {
                "average_score": sum(scores) / len(scores),
                "num_tasks": len(scores),
            }

        return parsed

    # ------------------------------------------------------------------
    # Perplexity fallback
    # ------------------------------------------------------------------

    def _run_perplexity_fallback(
        self,
        model_path: str,
        trust_remote_code: bool = False,
    ) -> dict[str, Any]:
        """Compute perplexity as a basic evaluation when lm-eval is unavailable."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Running perplexity-based fallback evaluation on '%s'.", model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        if self.device != "cuda":
            model = model.to(self.device)

        # Use a small set of diverse evaluation texts
        eval_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In machine learning, a neural network is a computational model inspired by biological neurons.",
            "Paris is the capital city of France and one of the most visited cities in the world.",
            "The Pythagorean theorem states that a squared plus b squared equals c squared.",
            "Climate change refers to long-term shifts in global temperatures and weather patterns.",
            "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
            "Shakespeare wrote many famous plays including Hamlet, Macbeth, and Romeo and Juliet.",
            "The human genome contains approximately three billion base pairs of DNA.",
        ]

        perplexity_result = self._metrics.compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=eval_texts,
            device=self.device,
        )

        return {
            "perplexity_eval": {
                "display_name": "Perplexity (fallback)",
                "score": perplexity_result["perplexity"],
                "avg_loss": perplexity_result["avg_loss"],
                "num_tokens": perplexity_result["num_tokens"],
                "metric": "perplexity",
                "note": "Lower is better. lm-eval not installed; install with pip install 'llm-forge[eval]'.",
            },
        }

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        base_path: str,
        finetuned_path: str,
        tasks: list[str] | None = None,
        num_fewshot: int | None = None,
        batch_size: int = 8,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Compare a base model against a fine-tuned model.

        Runs the same benchmarks on both models and computes the delta
        for each task.

        Parameters
        ----------
        base_path:
            Path or identifier for the base (pre-fine-tuning) model.
        finetuned_path:
            Path or identifier for the fine-tuned model.
        tasks:
            Benchmark tasks to evaluate.
        num_fewshot:
            Number of few-shot examples.
        batch_size:
            Batch size for inference.
        limit:
            Maximum samples per task.

        Returns
        -------
        dict
            Contains ``base_results``, ``finetuned_results``, and
            ``comparison`` with per-task deltas.
        """
        logger.info("Comparing base model '%s' vs fine-tuned '%s'.", base_path, finetuned_path)

        base_results = self.run_benchmarks(
            model_path=base_path,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
        )

        finetuned_results = self.run_benchmarks(
            model_path=finetuned_path,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
        )

        comparison = self._compute_comparison(base_results, finetuned_results)

        return {
            "base_results": base_results,
            "finetuned_results": finetuned_results,
            "comparison": comparison,
        }

    def _compute_comparison(
        self,
        base_results: dict[str, Any],
        finetuned_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute per-task score deltas."""
        comparison: dict[str, Any] = {}

        # Find common task keys (excluding metadata keys)
        base_tasks = {k for k in base_results if not k.startswith("_")}
        ft_tasks = {k for k in finetuned_results if not k.startswith("_")}
        common_tasks = base_tasks & ft_tasks

        for task in common_tasks:
            base_entry = base_results[task]
            ft_entry = finetuned_results[task]

            base_score = base_entry.get("score") if isinstance(base_entry, dict) else None
            ft_score = ft_entry.get("score") if isinstance(ft_entry, dict) else None

            if base_score is not None and ft_score is not None:
                delta = ft_score - base_score
                pct_change = (delta / base_score * 100) if base_score != 0 else float("inf")
                improved = delta > 0
            else:
                delta = None
                pct_change = None
                improved = None

            display_name = (
                base_entry.get("display_name", task) if isinstance(base_entry, dict) else task
            )

            comparison[task] = {
                "display_name": display_name,
                "base_score": base_score,
                "finetuned_score": ft_score,
                "delta": delta,
                "pct_change": round(pct_change, 2) if pct_change is not None else None,
                "improved": improved,
            }

        # Summary
        deltas = [v["delta"] for v in comparison.values() if v["delta"] is not None]
        if deltas:
            comparison["_summary"] = {
                "avg_delta": round(sum(deltas) / len(deltas), 4),
                "num_improved": sum(1 for d in deltas if d > 0),
                "num_degraded": sum(1 for d in deltas if d < 0),
                "num_unchanged": sum(1 for d in deltas if d == 0),
                "total_tasks": len(deltas),
            }

        return comparison

    # ------------------------------------------------------------------
    # Regression detection
    # ------------------------------------------------------------------

    @staticmethod
    def check_regression(
        comparison: dict[str, Any],
        threshold: float = -0.02,
    ) -> dict[str, Any]:
        """Analyse a comparison dict for performance regressions.

        Parameters
        ----------
        comparison:
            Output of :meth:`_compute_comparison` (or the ``"comparison"``
            key from :meth:`compare_models`).
        threshold:
            Maximum acceptable score drop.  A delta below this value
            for any benchmark is flagged as a regression.

        Returns
        -------
        dict
            ``passed`` (bool), ``regressions`` (list of task dicts),
            ``grade`` (str, e.g. "A+", "B", "D").
        """
        regressions = []
        improvements = []

        for task, data in comparison.items():
            if task.startswith("_"):
                continue
            delta = data.get("delta")
            if delta is not None and delta < threshold:
                regressions.append(
                    {
                        "task": task,
                        "display_name": data.get("display_name", task),
                        "base_score": data.get("base_score"),
                        "finetuned_score": data.get("finetuned_score"),
                        "delta": delta,
                    }
                )
            elif delta is not None and delta > 0:
                improvements.append(delta)

        summary = comparison.get("_summary", {})
        avg_delta = summary.get("avg_delta", 0)

        # Grade based on average improvement
        if avg_delta > 0.20:
            grade = "A+"
        elif avg_delta > 0.10:
            grade = "A"
        elif avg_delta > 0.05:
            grade = "B+"
        elif avg_delta > 0.0:
            grade = "B"
        elif avg_delta == 0:
            grade = "C"
        else:
            grade = "D"

        passed = len(regressions) == 0

        return {
            "passed": passed,
            "grade": grade,
            "regressions": regressions,
            "num_improved": len(improvements),
            "avg_delta": avg_delta,
        }

    # ------------------------------------------------------------------
    # Utility: save results
    # ------------------------------------------------------------------

    def save_results(
        self,
        results: dict[str, Any],
        output_path: str | Path,
    ) -> Path:
        """Save benchmark results to a JSON file.

        Parameters
        ----------
        results:
            The results dictionary from :meth:`run_benchmarks` or
            :meth:`compare_models`.
        output_path:
            File path to write (creates parent directories as needed).

        Returns
        -------
        Path
            Resolved path to the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("Benchmark results saved to '%s'.", output_path)
        return output_path

    # ------------------------------------------------------------------
    # List available tasks
    # ------------------------------------------------------------------

    @staticmethod
    def list_tasks() -> list[dict[str, str]]:
        """Return a list of supported standard benchmark tasks.

        Returns
        -------
        list[dict]
            Each dict has ``name``, ``display_name``, and ``description``.
        """
        return [
            {
                "name": name,
                "display_name": info["display_name"],
                "description": info["description"],
                "default_fewshot": info["default_fewshot"],
            }
            for name, info in STANDARD_TASKS.items()
        ]
