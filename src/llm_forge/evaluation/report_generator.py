"""HTML report generator for llm-forge evaluation results.

Generates self-contained HTML reports with embedded CSS and JavaScript,
including benchmark score tables, training loss curves (SVG bar charts),
per-task breakdowns, training config summaries, and sample model outputs.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("evaluation.report_generator")


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Generate professional HTML evaluation reports.

    Produces a self-contained HTML file with no external dependencies at
    viewing time.  All CSS, JavaScript, and data are embedded inline.

    Parameters
    ----------
    title : str
        Report title shown in the header.

    Examples
    --------
    >>> gen = ReportGenerator(title="My Model Evaluation")
    >>> gen.generate_report(results, config, "report.html")
    """

    def __init__(self, title: str = "llm-forge Evaluation Report") -> None:
        self.title = title

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(
        self,
        results: dict[str, Any],
        config: dict[str, Any] | None = None,
        output_path: str | Path = "evaluation_report.html",
        training_history: list[dict[str, float]] | None = None,
        sample_outputs: list[dict[str, str]] | None = None,
        comparison: dict[str, Any] | None = None,
    ) -> Path:
        """Generate a self-contained HTML evaluation report.

        Parameters
        ----------
        results:
            Benchmark or domain evaluation results dict.
        config:
            Training configuration dictionary to include in the report.
        output_path:
            File path for the generated HTML report.
        training_history:
            List of dicts with training metrics over time, e.g.
            ``[{"step": 100, "loss": 2.1, "lr": 1e-4}, ...]``.
        sample_outputs:
            List of sample model outputs, each with keys
            ``"input"``, ``"output"``, and optionally ``"reference"``.
        comparison:
            Model comparison results from ``BenchmarkRunner.compare_models``.

        Returns
        -------
        Path
            Path to the generated HTML file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sections: list[str] = []

        # Header
        sections.append(self._build_header())

        # Quality report card (top-level pass/fail + grade)
        quality_card = self.generate_quality_card(
            results=results,
            comparison=comparison,
            training_history=training_history,
        )
        sections.append(self._build_quality_card_section(quality_card))

        # Summary
        sections.append(self._build_summary_section(results))

        # Benchmark scores table
        sections.append(self._build_benchmark_table(results))

        # Model comparison
        if comparison:
            sections.append(self._build_comparison_section(comparison))

        # Per-task breakdown
        if "per_sample" in results or "category_breakdown" in results:
            sections.append(self._build_breakdown_section(results))

        # Training loss curves
        if training_history:
            sections.append(self._build_training_curves(training_history))

        # Config summary
        if config:
            sections.append(self._build_config_section(config))

        # Sample outputs
        if sample_outputs:
            sections.append(self._build_samples_section(sample_outputs))

        html_content = self._assemble_html(sections, training_history)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info("Evaluation report generated at '%s'.", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_header(self) -> str:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"""
        <header>
            <h1>{_esc(self.title)}</h1>
            <p class="timestamp">Generated on {timestamp}</p>
        </header>
        """

    # ------------------------------------------------------------------
    # Quality Report Card
    # ------------------------------------------------------------------

    @staticmethod
    def generate_quality_card(
        results: dict[str, Any] | None = None,
        comparison: dict[str, Any] | None = None,
        training_history: list[dict[str, float]] | None = None,
        regression_threshold: float = -0.02,
    ) -> dict[str, Any]:
        """Produce a structured quality report card.

        Aggregates benchmark scores, regression analysis, and training
        metrics into a single pass/fail verdict with a letter grade and
        per-dimension scores.  The returned dict is JSON-serialisable for
        programmatic consumption.

        Parameters
        ----------
        results:
            Benchmark results from ``BenchmarkRunner.run_benchmarks()``.
        comparison:
            Comparison dict from ``BenchmarkRunner.compare_models()``.
        training_history:
            List of ``{"step": ..., "loss": ...}`` dicts from training.
        regression_threshold:
            Maximum acceptable per-benchmark score drop.

        Returns
        -------
        dict
            Quality card with keys: ``verdict``, ``grade``, ``overall_score``,
            ``dimensions``, ``recommendations``.
        """
        dimensions: dict[str, dict[str, Any]] = {}
        recommendations: list[str] = []

        # --- Dimension 1: Benchmark performance ---
        bench_score = 0.0
        bench_pass = True
        if results:
            aggregate = results.get("_aggregate", results.get("aggregate", {}))
            avg = aggregate.get("average_score")
            if avg is not None:
                bench_score = float(avg)
                if bench_score < 0.25:
                    bench_pass = False
                    recommendations.append(
                        "Benchmark average is below 25%.  Consider more training data or epochs."
                    )
            else:
                bench_score = 0.0
                bench_pass = False
                recommendations.append("No aggregate benchmark score available.")
        else:
            bench_pass = False
            recommendations.append("No benchmark results provided.")

        dimensions["benchmark_performance"] = {
            "label": "Benchmark Performance",
            "score": round(bench_score, 4),
            "passed": bench_pass,
        }

        # --- Dimension 2: Regression analysis ---
        reg_pass = True
        reg_grade = "N/A"
        regressions: list[str] = []
        if comparison:
            summary = comparison.get("_summary", {})
            avg_delta = summary.get("avg_delta", 0.0)
            summary.get("num_degraded", 0)

            # Compute grade (same logic as BenchmarkRunner.check_regression)
            if avg_delta > 0.20:
                reg_grade = "A+"
            elif avg_delta > 0.10:
                reg_grade = "A"
            elif avg_delta > 0.05:
                reg_grade = "B+"
            elif avg_delta > 0.0:
                reg_grade = "B"
            elif avg_delta == 0.0:
                reg_grade = "C"
            else:
                reg_grade = "D"

            # Check per-task regressions
            for key, value in comparison.items():
                if key.startswith("_") or not isinstance(value, dict):
                    continue
                delta = value.get("delta")
                if delta is not None and delta < regression_threshold:
                    reg_pass = False
                    task_name = value.get("display_name", key)
                    regressions.append(task_name)

            if regressions:
                recommendations.append(
                    f"Regressions detected on: {', '.join(regressions)}.  "
                    "Review training data or reduce learning rate."
                )
        else:
            reg_grade = "N/A"

        dimensions["regression"] = {
            "label": "Regression Check",
            "grade": reg_grade,
            "passed": reg_pass,
            "regressions": regressions,
        }

        # --- Dimension 3: Training stability ---
        train_pass = True
        loss_reduction = 0.0
        if training_history and len(training_history) >= 2:
            losses = [
                h.get("loss", h.get("train_loss"))
                for h in training_history
                if h.get("loss", h.get("train_loss")) is not None
            ]
            if len(losses) >= 2:
                first_loss = losses[0]
                last_loss = losses[-1]
                if first_loss > 0:
                    loss_reduction = (first_loss - last_loss) / first_loss

                # Check for NaN/Inf
                import math

                if any(math.isnan(l) or math.isinf(l) for l in losses):
                    train_pass = False
                    recommendations.append(
                        "Training produced NaN/Inf losses.  Reduce learning rate."
                    )
                elif last_loss >= first_loss:
                    train_pass = False
                    recommendations.append(
                        "Loss did not decrease during training.  Check data quality and learning rate."
                    )
        else:
            # No training history — neutral (don't fail)
            loss_reduction = 0.0

        dimensions["training_stability"] = {
            "label": "Training Stability",
            "loss_reduction_pct": round(loss_reduction * 100, 1),
            "passed": train_pass,
        }

        # --- Overall verdict ---
        all_passed = all(d["passed"] for d in dimensions.values())

        # Overall score: weighted combination (0-100)
        score_parts = []
        # Benchmark performance (40% weight)
        score_parts.append(min(bench_score, 1.0) * 40)
        # Regression (30% weight): A+=100, A=90, B+=80, B=70, C=50, D=20, N/A=50
        grade_scores = {"A+": 100, "A": 90, "B+": 80, "B": 70, "C": 50, "D": 20, "N/A": 50}
        score_parts.append(grade_scores.get(reg_grade, 50) / 100 * 30)
        # Training stability (30% weight)
        stability_score = min(max(loss_reduction, 0.0), 1.0) * 100 if train_pass else 0
        score_parts.append(stability_score / 100 * 30)
        overall_score = round(sum(score_parts), 1)

        # Assign overall letter grade
        if overall_score >= 90:
            overall_grade = "A+"
        elif overall_score >= 80:
            overall_grade = "A"
        elif overall_score >= 70:
            overall_grade = "B+"
        elif overall_score >= 60:
            overall_grade = "B"
        elif overall_score >= 50:
            overall_grade = "C"
        else:
            overall_grade = "D"

        if not recommendations:
            recommendations.append("All quality checks passed.  Model is ready for deployment.")

        return {
            "verdict": "PASS" if all_passed else "FAIL",
            "grade": overall_grade,
            "overall_score": overall_score,
            "dimensions": dimensions,
            "recommendations": recommendations,
        }

    def _build_quality_card_section(self, card: dict[str, Any]) -> str:
        """Build the quality report card HTML section."""
        verdict = card.get("verdict", "N/A")
        grade = card.get("grade", "N/A")
        score = card.get("overall_score", 0)
        dimensions = card.get("dimensions", {})
        recommendations = card.get("recommendations", [])

        verdict_class = "positive" if verdict == "PASS" else "negative"

        # Dimension rows
        dim_rows = []
        for dim_data in dimensions.values():
            label = dim_data.get("label", "")
            passed = dim_data.get("passed", False)
            status = "PASS" if passed else "FAIL"
            status_class = "positive" if passed else "negative"

            detail_parts = []
            if "score" in dim_data:
                detail_parts.append(f"Score: {dim_data['score']:.4f}")
            if "grade" in dim_data:
                detail_parts.append(f"Grade: {dim_data['grade']}")
            if "loss_reduction_pct" in dim_data:
                detail_parts.append(f"Loss reduction: {dim_data['loss_reduction_pct']}%")
            if dim_data.get("regressions"):
                detail_parts.append(f"Regressions: {', '.join(dim_data['regressions'])}")
            detail = " | ".join(detail_parts) if detail_parts else ""

            dim_rows.append(f"""
            <tr>
                <td>{_esc(label)}</td>
                <td class="{status_class}"><strong>{status}</strong></td>
                <td>{_esc(detail)}</td>
            </tr>
            """)

        dim_html = "\n".join(dim_rows)

        # Recommendations
        rec_items = "\n".join(f"<li>{_esc(r)}</li>" for r in recommendations)

        return f"""
        <section>
            <h2>Quality Report Card</h2>
            <div class="cards">
                <div class="card">
                    <div class="card-value {verdict_class}">{_esc(verdict)}</div>
                    <div class="card-label">Verdict</div>
                </div>
                <div class="card">
                    <div class="card-value">{_esc(grade)}</div>
                    <div class="card-label">Grade</div>
                </div>
                <div class="card">
                    <div class="card-value">{score:.1f}</div>
                    <div class="card-label">Quality Score</div>
                </div>
            </div>
            <table>
                <thead>
                    <tr><th>Dimension</th><th>Status</th><th>Details</th></tr>
                </thead>
                <tbody>{dim_html}</tbody>
            </table>
            <h3>Recommendations</h3>
            <ul>{rec_items}</ul>
        </section>
        """

    def _build_summary_section(self, results: dict[str, Any]) -> str:
        """Build summary cards for aggregate metrics."""
        aggregate = results.get("_aggregate", results.get("aggregate", {}))
        metadata = results.get("_metadata", results.get("metadata", {}))

        cards = []

        if isinstance(aggregate, dict):
            avg_score = aggregate.get("average_score")
            if avg_score is not None:
                cards.append(self._card("Average Score", f"{avg_score:.4f}"))

            num_tasks = aggregate.get("num_tasks")
            if num_tasks is not None:
                cards.append(self._card("Tasks Evaluated", str(num_tasks)))

            # Domain eval aggregates
            for key in ("exact_match", "f1", "accuracy", "bleu", "perplexity"):
                val = aggregate.get(key)
                if val is not None:
                    label = key.replace("_", " ").title()
                    cards.append(self._card(label, f"{val:.4f}"))

        if isinstance(metadata, dict):
            backend = metadata.get("backend")
            if backend:
                cards.append(self._card("Backend", str(backend)))
            elapsed = metadata.get("elapsed_seconds")
            if elapsed is not None:
                cards.append(self._card("Eval Time", f"{elapsed:.1f}s"))
            num_samples = metadata.get("num_samples")
            if num_samples is not None:
                cards.append(self._card("Samples", str(num_samples)))

        if not cards:
            return ""

        cards_html = "\n".join(cards)
        return f"""
        <section>
            <h2>Summary</h2>
            <div class="cards">{cards_html}</div>
        </section>
        """

    def _card(self, label: str, value: str) -> str:
        return f"""
        <div class="card">
            <div class="card-value">{_esc(value)}</div>
            <div class="card-label">{_esc(label)}</div>
        </div>
        """

    def _build_benchmark_table(self, results: dict[str, Any]) -> str:
        """Build a table of benchmark scores."""
        rows: list[str] = []

        for key, value in results.items():
            if key.startswith("_") or not isinstance(value, dict):
                continue
            if "score" not in value and "display_name" not in value:
                continue

            display_name = value.get("display_name", key)
            score = value.get("score")
            metric = value.get("metric", "")
            stderr = value.get("score_stderr")
            fewshot = value.get("num_fewshot", "")

            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
            stderr_str = f" +/- {stderr:.4f}" if stderr is not None else ""
            bar_width = max(0, min(100, score * 100)) if isinstance(score, (int, float)) else 0

            rows.append(f"""
            <tr>
                <td>{_esc(display_name)}</td>
                <td>{_esc(metric)}</td>
                <td>{fewshot}</td>
                <td>
                    <div class="score-cell">
                        <span class="score-value">{score_str}{stderr_str}</span>
                        <div class="bar-bg"><div class="bar-fill" style="width:{bar_width}%"></div></div>
                    </div>
                </td>
            </tr>
            """)

        if not rows:
            return ""

        rows_html = "\n".join(rows)
        return f"""
        <section>
            <h2>Benchmark Scores</h2>
            <table>
                <thead>
                    <tr>
                        <th>Task</th>
                        <th>Metric</th>
                        <th>Few-shot</th>
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </section>
        """

    def _build_comparison_section(self, comparison: dict[str, Any]) -> str:
        """Build a model comparison table with delta indicators."""
        rows: list[str] = []
        summary = comparison.get("_summary", {})

        for key, value in comparison.items():
            if key.startswith("_") or not isinstance(value, dict):
                continue

            display_name = value.get("display_name", key)
            base = value.get("base_score")
            ft = value.get("finetuned_score")
            delta = value.get("delta")
            pct = value.get("pct_change")
            value.get("improved")

            base_str = f"{base:.4f}" if isinstance(base, (int, float)) else "N/A"
            ft_str = f"{ft:.4f}" if isinstance(ft, (int, float)) else "N/A"

            if delta is not None:
                delta_sign = "+" if delta > 0 else ""
                delta_str = f"{delta_sign}{delta:.4f}"
                delta_class = "positive" if delta > 0 else ("negative" if delta < 0 else "neutral")
                pct_str = f" ({delta_sign}{pct}%)" if pct is not None else ""
            else:
                delta_str = "N/A"
                delta_class = "neutral"
                pct_str = ""

            rows.append(f"""
            <tr>
                <td>{_esc(display_name)}</td>
                <td>{base_str}</td>
                <td>{ft_str}</td>
                <td class="{delta_class}">{delta_str}{pct_str}</td>
            </tr>
            """)

        if not rows:
            return ""

        rows_html = "\n".join(rows)

        summary_html = ""
        if summary:
            avg_delta = summary.get("avg_delta", 0)
            sign = "+" if avg_delta > 0 else ""
            summary_html = f"""
            <p class="comparison-summary">
                Average delta: <strong class="{"positive" if avg_delta > 0 else "negative"}">{sign}{avg_delta:.4f}</strong>
                | Improved: {summary.get("num_improved", 0)}
                | Degraded: {summary.get("num_degraded", 0)}
                | Unchanged: {summary.get("num_unchanged", 0)}
            </p>
            """

        return f"""
        <section>
            <h2>Model Comparison</h2>
            {summary_html}
            <table>
                <thead>
                    <tr>
                        <th>Task</th>
                        <th>Base</th>
                        <th>Fine-tuned</th>
                        <th>Delta</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </section>
        """

    def _build_breakdown_section(self, results: dict[str, Any]) -> str:
        """Build category breakdown or per-sample details."""
        category_breakdown = results.get("category_breakdown", {})
        per_sample = results.get("per_sample", [])

        sections: list[str] = []

        if category_breakdown:
            rows = []
            for cat_name, cat_data in category_breakdown.items():
                count = cat_data.get("count", 0)
                metrics = cat_data.get("metrics", {})
                metric_strs = [
                    f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in metrics.items()
                    if v is not None
                ]
                rows.append(f"""
                <tr>
                    <td>{_esc(cat_name)}</td>
                    <td>{count}</td>
                    <td>{_esc(", ".join(metric_strs))}</td>
                </tr>
                """)

            if rows:
                rows_html = "\n".join(rows)
                sections.append(f"""
                <h3>Category Breakdown</h3>
                <table>
                    <thead>
                        <tr><th>Category</th><th>Count</th><th>Metrics</th></tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """)

        # Show first N per-sample results
        if per_sample:
            max_show = min(20, len(per_sample))
            rows = []
            for i, item in enumerate(per_sample[:max_show]):
                inp = item.get("input", "")[:100]
                ref = item.get("reference", "")[:100]
                pred = item.get("prediction", "")[:100]
                sample_metrics = item.get("metrics", {})
                em = sample_metrics.get("exact_match")
                f1 = sample_metrics.get("f1")
                score_parts = []
                if em is not None:
                    score_parts.append(f"EM: {em:.2f}")
                if f1 is not None:
                    score_parts.append(f"F1: {f1:.2f}")

                rows.append(f"""
                <tr>
                    <td>{i + 1}</td>
                    <td title="{_esc(item.get("input", ""))}">{_esc(inp)}</td>
                    <td title="{_esc(item.get("reference", ""))}">{_esc(ref)}</td>
                    <td title="{_esc(item.get("prediction", ""))}">{_esc(pred)}</td>
                    <td>{_esc(", ".join(score_parts))}</td>
                </tr>
                """)

            if rows:
                rows_html = "\n".join(rows)
                sections.append(f"""
                <h3>Per-Sample Results (first {max_show} of {len(per_sample)})</h3>
                <table class="small-text">
                    <thead>
                        <tr><th>#</th><th>Input</th><th>Reference</th><th>Prediction</th><th>Scores</th></tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """)

        if not sections:
            return ""

        inner = "\n".join(sections)
        return f"""
        <section>
            <h2>Detailed Breakdown</h2>
            {inner}
        </section>
        """

    def _build_training_curves(self, training_history: list[dict[str, float]]) -> str:
        """Build SVG bar chart for training loss over steps."""
        if not training_history:
            return ""

        # Extract loss values for chart
        losses = [h.get("loss", h.get("train_loss", 0.0)) for h in training_history]
        steps = [h.get("step", i) for i, h in enumerate(training_history)]

        if not losses:
            return ""

        # Build SVG bar chart
        chart_width = 800
        chart_height = 300
        padding_left = 60
        padding_bottom = 40
        padding_top = 20
        padding_right = 20
        plot_w = chart_width - padding_left - padding_right
        plot_h = chart_height - padding_top - padding_bottom

        max_loss = max(losses) if losses else 1.0
        min_loss = min(losses) if losses else 0.0
        loss_range = max(max_loss - min_loss, 0.001)

        num_bars = len(losses)
        bar_width = max(2, plot_w / max(num_bars, 1) - 1)

        bars = []
        for i, loss in enumerate(losses):
            x = padding_left + i * (plot_w / max(num_bars, 1))
            normalized = (loss - min_loss) / loss_range
            bar_h = max(1, normalized * plot_h)
            y = padding_top + plot_h - bar_h

            color = f"hsl({120 - normalized * 120}, 70%, 50%)"
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_h:.1f}" fill="{color}" opacity="0.8">'
                f"<title>Step {steps[i]}: {loss:.4f}</title></rect>"
            )

        # Y-axis labels
        y_labels = []
        for i in range(5):
            frac = i / 4
            val = min_loss + frac * loss_range
            y_pos = padding_top + plot_h - frac * plot_h
            y_labels.append(
                f'<text x="{padding_left - 5}" y="{y_pos:.1f}" '
                f'text-anchor="end" font-size="11" fill="#666">{val:.3f}</text>'
            )
            y_labels.append(
                f'<line x1="{padding_left}" y1="{y_pos:.1f}" '
                f'x2="{chart_width - padding_right}" y2="{y_pos:.1f}" '
                f'stroke="#eee" stroke-width="1"/>'
            )

        # X-axis labels (show ~6 tick marks)
        x_labels = []
        tick_count = min(6, num_bars)
        for i in range(tick_count):
            idx = int(i * (num_bars - 1) / max(tick_count - 1, 1))
            x = padding_left + idx * (plot_w / max(num_bars, 1))
            x_labels.append(
                f'<text x="{x:.1f}" y="{chart_height - 5}" '
                f'text-anchor="middle" font-size="11" fill="#666">{steps[idx]}</text>'
            )

        bars_svg = "\n".join(bars)
        y_labels_svg = "\n".join(y_labels)
        x_labels_svg = "\n".join(x_labels)

        return f"""
        <section>
            <h2>Training Loss Curve</h2>
            <svg width="{chart_width}" height="{chart_height}" class="chart">
                {y_labels_svg}
                {bars_svg}
                {x_labels_svg}
                <text x="{padding_left - 40}" y="{chart_height // 2}"
                      text-anchor="middle" font-size="12" fill="#666"
                      transform="rotate(-90, {padding_left - 40}, {chart_height // 2})">Loss</text>
                <text x="{chart_width // 2}" y="{chart_height - 2}"
                      text-anchor="middle" font-size="12" fill="#666">Step</text>
            </svg>
        </section>
        """

    def _build_config_section(self, config: dict[str, Any]) -> str:
        """Build a collapsible config summary section."""

        def _flatten(d: dict[str, Any], prefix: str = "") -> list[tuple]:
            items = []
            for key, val in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(val, dict):
                    items.extend(_flatten(val, full_key))
                else:
                    items.append((full_key, val))
            return items

        flat = _flatten(config)
        rows = []
        for key, val in flat:
            val_str = str(val)
            if len(val_str) > 120:
                val_str = val_str[:120] + "..."
            rows.append(f"<tr><td>{_esc(key)}</td><td>{_esc(val_str)}</td></tr>")

        if not rows:
            return ""

        rows_html = "\n".join(rows)
        config_json = json.dumps(config, indent=2, default=str)

        return f"""
        <section>
            <h2>Training Configuration</h2>
            <details>
                <summary>Show/hide full configuration ({len(flat)} parameters)</summary>
                <table class="small-text">
                    <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
                <details>
                    <summary>Raw JSON</summary>
                    <pre>{_esc(config_json)}</pre>
                </details>
            </details>
        </section>
        """

    def _build_samples_section(self, sample_outputs: list[dict[str, str]]) -> str:
        """Build section showing sample model outputs."""
        if not sample_outputs:
            return ""

        items = []
        for i, sample in enumerate(sample_outputs[:10], 1):
            inp = sample.get("input", "")
            out = sample.get("output", sample.get("prediction", ""))
            ref = sample.get("reference", "")

            ref_html = ""
            if ref:
                ref_html = f"""
                <div class="sample-field">
                    <strong>Reference:</strong>
                    <div class="sample-text ref">{_esc(ref)}</div>
                </div>
                """

            items.append(f"""
            <div class="sample-card">
                <div class="sample-header">Sample {i}</div>
                <div class="sample-field">
                    <strong>Input:</strong>
                    <div class="sample-text input">{_esc(inp)}</div>
                </div>
                <div class="sample-field">
                    <strong>Output:</strong>
                    <div class="sample-text output">{_esc(out)}</div>
                </div>
                {ref_html}
            </div>
            """)

        items_html = "\n".join(items)
        return f"""
        <section>
            <h2>Sample Model Outputs</h2>
            {items_html}
        </section>
        """

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble_html(
        self,
        sections: list[str],
        training_history: list[dict[str, float]] | None = None,
    ) -> str:
        """Assemble the complete HTML document."""
        body = "\n".join(s for s in sections if s)

        # Embed training history as JSON for potential JS interactivity
        data_json = ""
        if training_history:
            data_json = f"""
            <script>
                window.__LLM_FORGE_TRAINING_DATA__ = {json.dumps(training_history, default=str)};
            </script>
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(self.title)}</title>
<style>
{_CSS}
</style>
</head>
<body>
<div class="container">
{body}
<footer>
    <p>Generated by <strong>llm-forge</strong> evaluation system.</p>
</footer>
</div>
{data_json}
<script>
{_JS}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(text))


# ---------------------------------------------------------------------------
# Embedded CSS
# ---------------------------------------------------------------------------

_CSS = """
:root {
    --bg: #f8f9fa;
    --card-bg: #ffffff;
    --border: #e0e0e0;
    --text: #333333;
    --text-muted: #666666;
    --primary: #2563eb;
    --positive: #16a34a;
    --negative: #dc2626;
    --neutral: #6b7280;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
}

.container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
}

header {
    text-align: center;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 2px solid var(--primary);
}

header h1 {
    font-size: 1.8rem;
    color: var(--primary);
    margin-bottom: 0.5rem;
}

.timestamp {
    color: var(--text-muted);
    font-size: 0.9rem;
}

section {
    background: var(--card-bg);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

h2 {
    font-size: 1.3rem;
    margin-bottom: 1rem;
    color: var(--primary);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}

h3 {
    font-size: 1.1rem;
    margin: 1.2rem 0 0.8rem;
    color: var(--text);
}

.cards {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
}

.card {
    flex: 1 1 140px;
    text-align: center;
    padding: 1rem;
    background: var(--bg);
    border-radius: 8px;
    border: 1px solid var(--border);
    min-width: 120px;
}

.card-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary);
}

.card-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.5rem;
}

th, td {
    padding: 0.6rem 0.8rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}

th {
    background: var(--bg);
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-muted);
}

tr:hover {
    background: #f0f4ff;
}

.small-text td, .small-text th {
    font-size: 0.85rem;
    padding: 0.4rem 0.6rem;
}

.score-cell {
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.score-value {
    font-weight: 600;
    min-width: 80px;
}

.bar-bg {
    flex: 1;
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
    min-width: 80px;
}

.bar-fill {
    height: 100%;
    background: var(--primary);
    border-radius: 4px;
    transition: width 0.3s ease;
}

.positive { color: var(--positive); font-weight: 600; }
.negative { color: var(--negative); font-weight: 600; }
.neutral { color: var(--neutral); }

.comparison-summary {
    margin-bottom: 1rem;
    padding: 0.8rem;
    background: var(--bg);
    border-radius: 6px;
    font-size: 0.95rem;
}

.chart {
    display: block;
    margin: 0 auto;
    max-width: 100%;
}

details {
    margin-top: 0.8rem;
}

summary {
    cursor: pointer;
    color: var(--primary);
    font-weight: 500;
    padding: 0.4rem 0;
}

summary:hover {
    text-decoration: underline;
}

pre {
    background: #1e293b;
    color: #e2e8f0;
    padding: 1rem;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 0.85rem;
    line-height: 1.5;
    margin-top: 0.5rem;
}

.sample-card {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.sample-header {
    font-weight: 600;
    margin-bottom: 0.8rem;
    color: var(--primary);
    font-size: 0.95rem;
}

.sample-field {
    margin-bottom: 0.6rem;
}

.sample-text {
    padding: 0.5rem 0.8rem;
    border-radius: 4px;
    font-size: 0.9rem;
    margin-top: 0.3rem;
    white-space: pre-wrap;
    word-wrap: break-word;
}

.sample-text.input { background: #eff6ff; border-left: 3px solid var(--primary); }
.sample-text.output { background: #f0fdf4; border-left: 3px solid var(--positive); }
.sample-text.ref { background: #fefce8; border-left: 3px solid #ca8a04; }

footer {
    text-align: center;
    padding: 1.5rem 0;
    color: var(--text-muted);
    font-size: 0.85rem;
}

@media (max-width: 768px) {
    .container { padding: 1rem; }
    .cards { gap: 0.5rem; }
    .card { flex: 1 1 100px; padding: 0.7rem; }
    .card-value { font-size: 1.2rem; }
    table { font-size: 0.85rem; }
    th, td { padding: 0.4rem; }
}
"""

# ---------------------------------------------------------------------------
# Embedded JavaScript
# ---------------------------------------------------------------------------

_JS = """
// Interactive tooltip for SVG chart bars
document.querySelectorAll('.chart rect').forEach(function(rect) {
    rect.addEventListener('mouseover', function(e) {
        this.style.opacity = '1';
        this.style.strokeWidth = '1';
        this.style.stroke = '#333';
    });
    rect.addEventListener('mouseout', function(e) {
        this.style.opacity = '0.8';
        this.style.strokeWidth = '0';
    });
});

// Sort tables by clicking column headers
document.querySelectorAll('th').forEach(function(th) {
    th.style.cursor = 'pointer';
    th.addEventListener('click', function() {
        var table = th.closest('table');
        var tbody = table.querySelector('tbody');
        if (!tbody) return;
        var rows = Array.from(tbody.querySelectorAll('tr'));
        var colIndex = Array.from(th.parentNode.children).indexOf(th);
        var ascending = th.dataset.sortDir !== 'asc';
        th.dataset.sortDir = ascending ? 'asc' : 'desc';

        rows.sort(function(a, b) {
            var aText = a.children[colIndex] ? a.children[colIndex].textContent.trim() : '';
            var bText = b.children[colIndex] ? b.children[colIndex].textContent.trim() : '';
            var aNum = parseFloat(aText);
            var bNum = parseFloat(bText);
            if (!isNaN(aNum) && !isNaN(bNum)) {
                return ascending ? aNum - bNum : bNum - aNum;
            }
            return ascending ? aText.localeCompare(bText) : bText.localeCompare(aText);
        });

        rows.forEach(function(row) { tbody.appendChild(row); });
    });
});
"""
