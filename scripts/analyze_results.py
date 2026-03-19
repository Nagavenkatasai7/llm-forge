#!/usr/bin/env python3
"""Finance Specialist Pipeline — Results Analyzer.

Parses training logs, extracts metrics, analyzes benchmark results,
and generates a comprehensive quality report with pass/fail verdict.

Usage:
    python scripts/analyze_results.py --output-dir outputs/finance-specialist-llama1b
    python scripts/analyze_results.py --output-dir outputs/finance-specialist-llama1b --log-file logs/pipeline_full.log
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Benchmark pass thresholds (minimum acceptable scores)
BENCHMARK_THRESHOLDS = {
    "hellaswag": 0.35,
    "arc_easy": 0.45,
    "mmlu": 0.30,
    "truthfulqa_mc2": 0.35,
    "ifeval": 0.25,
    "winogrande": 0.50,
    "gsm8k": 0.10,
}

# Training quality thresholds
MAX_FINAL_LOSS = 1.5
MIN_LOSS_DECREASE_PCT = 20.0  # Minimum % loss decrease from start to end
MAX_NAN_FRACTION = 0.0  # Zero tolerance for NaN


# ---------------------------------------------------------------------------
# Log Parsing
# ---------------------------------------------------------------------------


def parse_training_log(log_path: Path) -> Dict[str, Any]:
    """Parse a training log file and extract metrics."""
    metrics: Dict[str, Any] = {
        "loss_values": [],
        "eval_losses": [],
        "learning_rates": [],
        "steps": [],
        "epochs": [],
        "nan_count": 0,
        "total_steps": 0,
        "wall_time_seconds": None,
        "gpu_count": None,
        "model_name": None,
        "errors": [],
        "warnings": [],
    }

    if not log_path.exists():
        metrics["errors"].append(f"Log file not found: {log_path}")
        return metrics

    text = log_path.read_text(errors="replace")

    # Extract loss values from HF Trainer log lines
    # Pattern: {'loss': 1.234, 'learning_rate': 0.0002, 'epoch': 0.5}
    loss_pattern = re.compile(
        r"\{'loss':\s*([\d.]+),\s*'learning_rate':\s*([\d.e-]+),\s*'epoch':\s*([\d.]+)\}"
    )
    for m in loss_pattern.finditer(text):
        loss = float(m.group(1))
        lr = float(m.group(2))
        epoch = float(m.group(3))
        metrics["loss_values"].append(loss)
        metrics["learning_rates"].append(lr)
        metrics["epochs"].append(epoch)
        if loss != loss:  # NaN check
            metrics["nan_count"] += 1

    # Extract eval loss
    eval_pattern = re.compile(r"\{'eval_loss':\s*([\d.]+)")
    for m in eval_pattern.finditer(text):
        metrics["eval_losses"].append(float(m.group(1)))

    # Extract step counts
    step_pattern = re.compile(r"Step\s+(\d+)/(\d+)")
    for m in step_pattern.finditer(text):
        metrics["steps"].append(int(m.group(1)))
        metrics["total_steps"] = max(metrics["total_steps"], int(m.group(2)))

    # Extract GPU count
    gpu_match = re.search(r"GPU count:\s*(\d+)", text)
    if gpu_match:
        metrics["gpu_count"] = int(gpu_match.group(1))

    # Extract model name
    model_match = re.search(r"Config validated:\s*(.+)", text)
    if model_match:
        metrics["model_name"] = model_match.group(1).strip()

    # Extract wall time
    wall_match = re.search(r"Wall time:\s*(\d+)\s*seconds", text)
    if wall_match:
        metrics["wall_time_seconds"] = int(wall_match.group(1))

    # Count errors/warnings
    for line in text.splitlines():
        lower = line.lower()
        if "error" in lower and "traceback" not in lower:
            metrics["errors"].append(line.strip()[:200])
        if "warning" in lower and len(metrics["warnings"]) < 20:
            metrics["warnings"].append(line.strip()[:200])

    return metrics


# ---------------------------------------------------------------------------
# Benchmark Parsing
# ---------------------------------------------------------------------------


def parse_benchmark_results(output_dir: Path) -> Dict[str, float]:
    """Search for benchmark result files and extract scores."""
    results: Dict[str, float] = {}

    # Check for lm-eval results JSON files
    for pattern in ["**/results*.json", "**/eval_results*.json", "**/*benchmark*.json"]:
        for f in output_dir.glob(pattern):
            try:
                data = json.loads(f.read_text())
                if "results" in data:
                    for task_name, task_data in data["results"].items():
                        # lm-eval format: results -> task_name -> metric
                        if isinstance(task_data, dict):
                            for key, val in task_data.items():
                                if isinstance(val, (int, float)):
                                    clean_name = task_name.split("|")[0] if "|" in task_name else task_name
                                    results[f"{clean_name}/{key}"] = val
                elif isinstance(data, dict):
                    for key, val in data.items():
                        if isinstance(val, (int, float)):
                            results[key] = val
            except (json.JSONDecodeError, KeyError):
                continue

    # Check for report HTML/Markdown
    for f in output_dir.glob("**/*report*"):
        if f.suffix in (".html", ".md"):
            text = f.read_text(errors="replace")
            # Try to extract scores from table rows
            score_pattern = re.compile(r"(hellaswag|arc_easy|mmlu|truthfulqa|ifeval|winogrande|gsm8k)\D+([\d.]+)")
            for m in score_pattern.finditer(text.lower()):
                task = m.group(1)
                score = float(m.group(2))
                if score <= 1.0:  # Likely a proper score
                    results[task] = score

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_training(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze training metrics and produce quality assessment."""
    analysis: Dict[str, Any] = {
        "training_quality": "UNKNOWN",
        "issues": [],
        "strengths": [],
        "stats": {},
    }

    losses = metrics["loss_values"]
    if not losses:
        analysis["training_quality"] = "NO DATA"
        analysis["issues"].append("No loss values found in training log")
        return analysis

    first_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)
    max_loss = max(losses)
    avg_loss = sum(losses) / len(losses)
    loss_decrease_pct = ((first_loss - final_loss) / first_loss) * 100 if first_loss > 0 else 0

    analysis["stats"] = {
        "first_loss": round(first_loss, 4),
        "final_loss": round(final_loss, 4),
        "min_loss": round(min_loss, 4),
        "max_loss": round(max_loss, 4),
        "avg_loss": round(avg_loss, 4),
        "loss_decrease_pct": round(loss_decrease_pct, 1),
        "total_logged_steps": len(losses),
        "nan_count": metrics["nan_count"],
    }

    # Quality checks
    quality_score = 100

    # Check 1: NaN values
    if metrics["nan_count"] > 0:
        analysis["issues"].append(f"NaN loss detected ({metrics['nan_count']} times)")
        quality_score -= 50

    # Check 2: Final loss threshold
    if final_loss > MAX_FINAL_LOSS:
        analysis["issues"].append(f"Final loss {final_loss:.4f} exceeds threshold {MAX_FINAL_LOSS}")
        quality_score -= 20
    else:
        analysis["strengths"].append(f"Final loss {final_loss:.4f} is within acceptable range")

    # Check 3: Loss decrease
    if loss_decrease_pct < MIN_LOSS_DECREASE_PCT:
        analysis["issues"].append(
            f"Loss decreased only {loss_decrease_pct:.1f}% (threshold: {MIN_LOSS_DECREASE_PCT}%)"
        )
        quality_score -= 15
    else:
        analysis["strengths"].append(f"Loss decreased {loss_decrease_pct:.1f}% (healthy convergence)")

    # Check 4: Loss stability (no sudden spikes in last 10%)
    last_segment = losses[int(len(losses) * 0.9):]
    if last_segment:
        last_avg = sum(last_segment) / len(last_segment)
        if max(last_segment) > last_avg * 1.5:
            analysis["issues"].append("Loss instability detected in final 10% of training")
            quality_score -= 10

    # Check 5: Eval loss
    if metrics["eval_losses"]:
        final_eval = metrics["eval_losses"][-1]
        analysis["stats"]["final_eval_loss"] = round(final_eval, 4)
        if final_eval < final_loss * 1.2:
            analysis["strengths"].append("No significant overfitting (eval loss close to train loss)")
        else:
            analysis["issues"].append(
                f"Possible overfitting: eval loss {final_eval:.4f} >> train loss {final_loss:.4f}"
            )
            quality_score -= 10

    # Wall time
    if metrics["wall_time_seconds"]:
        hours = metrics["wall_time_seconds"] / 3600
        analysis["stats"]["wall_time_hours"] = round(hours, 2)
        throughput = len(losses) / hours if hours > 0 else 0
        analysis["stats"]["steps_per_hour"] = round(throughput, 0)

    # Overall verdict
    if quality_score >= 80:
        analysis["training_quality"] = "EXCELLENT"
    elif quality_score >= 60:
        analysis["training_quality"] = "GOOD"
    elif quality_score >= 40:
        analysis["training_quality"] = "ACCEPTABLE"
    else:
        analysis["training_quality"] = "POOR"

    analysis["stats"]["quality_score"] = quality_score

    return analysis


def analyze_benchmarks(benchmark_results: Dict[str, float]) -> Dict[str, Any]:
    """Analyze benchmark results against thresholds."""
    analysis: Dict[str, Any] = {
        "benchmark_quality": "UNKNOWN",
        "passed": [],
        "failed": [],
        "scores": {},
    }

    if not benchmark_results:
        analysis["benchmark_quality"] = "NO DATA"
        return analysis

    for task, threshold in BENCHMARK_THRESHOLDS.items():
        # Find matching score (may be task/metric format)
        score = None
        for key, val in benchmark_results.items():
            if task in key.lower():
                score = val
                break

        if score is not None:
            analysis["scores"][task] = round(score, 4)
            if score >= threshold:
                analysis["passed"].append(f"{task}: {score:.4f} (>= {threshold})")
            else:
                analysis["failed"].append(f"{task}: {score:.4f} (< {threshold})")

    total = len(analysis["passed"]) + len(analysis["failed"])
    if total == 0:
        analysis["benchmark_quality"] = "NO MATCHING RESULTS"
    elif len(analysis["failed"]) == 0:
        analysis["benchmark_quality"] = "ALL PASSED"
    elif len(analysis["failed"]) <= 2:
        analysis["benchmark_quality"] = "MOSTLY PASSED"
    else:
        analysis["benchmark_quality"] = "NEEDS IMPROVEMENT"

    return analysis


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def generate_report(
    training_analysis: Dict[str, Any],
    benchmark_analysis: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Generate a human-readable analysis report."""
    lines: List[str] = []

    lines.append("=" * 70)
    lines.append("  FINANCE SPECIALIST PIPELINE — RESULTS ANALYSIS")
    lines.append("=" * 70)
    lines.append("")

    # Overall verdict
    train_q = training_analysis["training_quality"]
    bench_q = benchmark_analysis["benchmark_quality"]
    overall = "PASS" if train_q in ("EXCELLENT", "GOOD") else "REVIEW NEEDED"
    lines.append(f"  OVERALL VERDICT: {overall}")
    lines.append(f"  Training Quality:  {train_q}")
    lines.append(f"  Benchmark Quality: {bench_q}")
    lines.append("")

    # Training stats
    lines.append("-" * 70)
    lines.append("TRAINING METRICS")
    lines.append("-" * 70)
    stats = training_analysis.get("stats", {})
    if stats:
        lines.append(f"  First loss:        {stats.get('first_loss', 'N/A')}")
        lines.append(f"  Final loss:        {stats.get('final_loss', 'N/A')}")
        lines.append(f"  Min loss:          {stats.get('min_loss', 'N/A')}")
        lines.append(f"  Loss decrease:     {stats.get('loss_decrease_pct', 'N/A')}%")
        lines.append(f"  NaN count:         {stats.get('nan_count', 'N/A')}")
        lines.append(f"  Logged steps:      {stats.get('total_logged_steps', 'N/A')}")
        if "final_eval_loss" in stats:
            lines.append(f"  Final eval loss:   {stats['final_eval_loss']}")
        if "wall_time_hours" in stats:
            lines.append(f"  Wall time:         {stats['wall_time_hours']} hours")
            lines.append(f"  Throughput:        {stats.get('steps_per_hour', 'N/A')} steps/hour")
        lines.append(f"  Quality score:     {stats.get('quality_score', 'N/A')}/100")
    else:
        lines.append("  No training metrics available.")
    lines.append("")

    # Strengths
    if training_analysis.get("strengths"):
        lines.append("  Strengths:")
        for s in training_analysis["strengths"]:
            lines.append(f"    + {s}")
        lines.append("")

    # Issues
    if training_analysis.get("issues"):
        lines.append("  Issues:")
        for i in training_analysis["issues"]:
            lines.append(f"    ! {i}")
        lines.append("")

    # Benchmark results
    lines.append("-" * 70)
    lines.append("BENCHMARK RESULTS")
    lines.append("-" * 70)
    if benchmark_analysis.get("scores"):
        for task, score in sorted(benchmark_analysis["scores"].items()):
            threshold = BENCHMARK_THRESHOLDS.get(task, 0)
            status = "PASS" if score >= threshold else "FAIL"
            lines.append(f"  {task:20s}  {score:.4f}  (threshold: {threshold})  [{status}]")
    else:
        lines.append("  No benchmark results found.")
        lines.append("  (Results will be available after the evaluation stage completes)")
    lines.append("")

    if benchmark_analysis.get("passed"):
        lines.append(f"  Passed: {len(benchmark_analysis['passed'])} benchmarks")
    if benchmark_analysis.get("failed"):
        lines.append(f"  Failed: {len(benchmark_analysis['failed'])} benchmarks")
    lines.append("")

    # Output artifacts
    lines.append("-" * 70)
    lines.append("OUTPUT ARTIFACTS")
    lines.append("-" * 70)
    if output_dir.exists():
        for f in sorted(output_dir.iterdir()):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                lines.append(f"  {f.name:40s}  {size_mb:>8.1f} MB")
        # Check subdirs
        for d in sorted(output_dir.iterdir()):
            if d.is_dir():
                total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                lines.append(f"  {d.name + '/':40s}  {total / (1024 * 1024):>8.1f} MB (dir)")
    else:
        lines.append(f"  Output directory not found: {output_dir}")
    lines.append("")

    # Errors from log
    if metrics.get("errors"):
        lines.append("-" * 70)
        lines.append("ERRORS IN LOG")
        lines.append("-" * 70)
        for err in metrics["errors"][:10]:
            lines.append(f"  {err}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("  END OF ANALYSIS REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance Specialist Pipeline — Results Analyzer")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to the training output directory",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to the pipeline log file (default: output-dir/logs/pipeline_full.log)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable text",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    log_file = args.log_file or output_dir / "logs" / "pipeline_full.log"

    # Parse
    print(f"Analyzing results from: {output_dir}")
    print(f"Log file: {log_file}")
    print()

    metrics = parse_training_log(log_file)
    benchmark_results = parse_benchmark_results(output_dir)

    # Analyze
    training_analysis = analyze_training(metrics)
    benchmark_analysis = analyze_benchmarks(benchmark_results)

    if args.json:
        result = {
            "training": training_analysis,
            "benchmarks": benchmark_analysis,
            "raw_metrics": {
                "loss_count": len(metrics["loss_values"]),
                "final_loss": metrics["loss_values"][-1] if metrics["loss_values"] else None,
                "gpu_count": metrics["gpu_count"],
                "model_name": metrics["model_name"],
                "wall_time_seconds": metrics["wall_time_seconds"],
            },
        }
        print(json.dumps(result, indent=2))
    else:
        report = generate_report(training_analysis, benchmark_analysis, metrics, output_dir)
        print(report)

        # Save report to file
        report_path = output_dir / "analysis_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
