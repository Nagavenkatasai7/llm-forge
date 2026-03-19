"""Terminal-based post-training quality report card.

Displays a Rich-formatted summary after every training run showing:
- Overall quality grade (A+ to D)
- Benchmark before/after comparison
- Strengths and watch areas
- Actionable recommendations
- Export options
"""

from __future__ import annotations

from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Grading scale
# ---------------------------------------------------------------------------

GRADE_THRESHOLDS: list[tuple[str, float, str]] = [
    ("A+", 0.20, "Exceptional improvement"),
    ("A", 0.10, "Significant improvement"),
    ("B+", 0.05, "Good improvement"),
    ("B", 0.00, "Modest improvement"),
    ("C", -0.02, "No meaningful change"),
    ("D", float("-inf"), "Regression detected"),
]


def compute_grade(improvement_pct: float) -> tuple[str, str]:
    """Map an improvement percentage to a letter grade and description."""
    for grade, threshold, desc in GRADE_THRESHOLDS:
        if improvement_pct >= threshold:
            return grade, desc
    return "D", "Regression detected"


def grade_color(grade: str) -> str:
    """Return a Rich colour for a given grade."""
    if grade.startswith("A"):
        return "green"
    if grade.startswith("B"):
        return "cyan"
    if grade == "C":
        return "yellow"
    return "red"


# ---------------------------------------------------------------------------
# TrainingSummary data class
# ---------------------------------------------------------------------------


class TrainingSummary:
    """Container for post-training summary data.

    Parameters
    ----------
    model_name : str
        Name of the trained model.
    training_loss_start : float
        Training loss at the start of training.
    training_loss_end : float
        Training loss at the end of training.
    duration_seconds : float
        Total training duration in seconds.
    benchmark_results : dict
        Mapping of benchmark names to scores (0-1).
    baseline_results : dict
        Mapping of benchmark names to baseline (pre-training) scores.
    eval_loss : float or None
        Final evaluation loss.
    num_samples : int
        Number of training samples used.
    training_method : str
        Training method used (lora, qlora, full).
    """

    def __init__(
        self,
        model_name: str = "",
        training_loss_start: float = 0.0,
        training_loss_end: float = 0.0,
        duration_seconds: float = 0.0,
        benchmark_results: dict[str, float] | None = None,
        baseline_results: dict[str, float] | None = None,
        eval_loss: float | None = None,
        num_samples: int = 0,
        training_method: str = "lora",
    ) -> None:
        self.model_name = model_name
        self.training_loss_start = training_loss_start
        self.training_loss_end = training_loss_end
        self.duration_seconds = duration_seconds
        self.benchmark_results = benchmark_results or {}
        self.baseline_results = baseline_results or {}
        self.eval_loss = eval_loss
        self.num_samples = num_samples
        self.training_method = training_method

    @property
    def loss_improvement(self) -> float:
        """Fractional improvement in training loss."""
        if self.training_loss_start == 0:
            return 0.0
        return (self.training_loss_start - self.training_loss_end) / self.training_loss_start

    @property
    def mean_benchmark_improvement(self) -> float:
        """Average improvement across all benchmarks with baselines."""
        improvements = []
        for bench, score in self.benchmark_results.items():
            baseline = self.baseline_results.get(bench)
            if baseline is not None and baseline > 0:
                improvements.append((score - baseline) / baseline)
        if not improvements:
            return self.loss_improvement
        return sum(improvements) / len(improvements)

    @property
    def overall_grade(self) -> tuple[str, str]:
        """Compute the overall quality grade."""
        return compute_grade(self.mean_benchmark_improvement)

    @property
    def duration_formatted(self) -> str:
        """Human-readable duration string."""
        total = int(self.duration_seconds)
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def strengths(self) -> list[str]:
        """Identify strengths from benchmark improvements."""
        items = []
        for bench, score in self.benchmark_results.items():
            baseline = self.baseline_results.get(bench, 0)
            if baseline > 0 and score > baseline:
                pct = (score - baseline) / baseline * 100
                items.append(f"{bench}: +{pct:.1f}% improvement")
        if self.loss_improvement > 0.1:
            items.append(f"Training loss decreased {self.loss_improvement * 100:.0f}%")
        if not items:
            items.append("Training completed successfully")
        return items

    def watch_areas(self) -> list[str]:
        """Identify areas that need attention."""
        items = []
        for bench, score in self.benchmark_results.items():
            baseline = self.baseline_results.get(bench, 0)
            if baseline > 0 and score < baseline:
                pct = (baseline - score) / baseline * 100
                items.append(f"{bench}: -{pct:.1f}% regression (possible catastrophic forgetting)")
        if self.training_loss_end > self.training_loss_start:
            items.append("Training loss increased (possible instability)")
        return items

    def recommendations(self) -> list[str]:
        """Generate actionable recommendations."""
        recs = []
        grade, _ = self.overall_grade

        if grade in ("C", "D"):
            recs.append("Add more diverse training data to improve quality")
            recs.append("Try increasing training epochs by 1-2")
            recs.append("Consider adding 10-15% general knowledge data")
        if any("regression" in w for w in self.watch_areas()):
            recs.append("Include knowledge retention data to prevent forgetting")
            recs.append("Enable ITI (anti-hallucination) with: iti.enabled: true")
        if self.loss_improvement < 0.05 and grade not in ("A+", "A"):
            recs.append("Increase LoRA rank (r: 32) for more capacity")
            recs.append("Try a slightly higher learning rate")
        if not recs:
            recs.append("Model is performing well - consider deploying")
            if self.training_method in ("lora", "qlora"):
                recs.append("Export to GGUF for Ollama: serving.export_format: gguf")
        return recs


# ---------------------------------------------------------------------------
# Display function
# ---------------------------------------------------------------------------


def display_training_summary(
    summary: TrainingSummary,
    console: Any | None = None,
) -> None:
    """Display a Rich-formatted post-training quality report card."""
    if console is None:
        if _RICH_AVAILABLE:
            console = Console()
        else:
            _display_plain(summary)
            return

    grade, grade_desc = summary.overall_grade
    gc = grade_color(grade)

    # Header
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold {gc}]OVERALL GRADE: {grade}[/bold {gc}]  "
                f"({grade_desc})\n\n"
                f"Model: [bold]{summary.model_name}[/bold]  |  "
                f"Method: [bold]{summary.training_method}[/bold]  |  "
                f"Duration: [bold]{summary.duration_formatted}[/bold]  |  "
                f"Samples: [bold]{summary.num_samples:,}[/bold]"
            ),
            title="[bold]QUALITY REPORT CARD[/bold]",
            border_style=gc,
            padding=(1, 2),
        )
    )

    # Benchmark comparison table
    if summary.benchmark_results:
        bench_table = Table(title="Benchmark Results", expand=True)
        bench_table.add_column("Benchmark", style="bold", width=20)
        bench_table.add_column("Before", justify="right", width=10)
        bench_table.add_column("After", justify="right", width=10)
        bench_table.add_column("Change", justify="right", width=12)

        for bench, score in summary.benchmark_results.items():
            baseline = summary.baseline_results.get(bench)
            before_str = f"{baseline * 100:.1f}%" if baseline is not None else "—"
            after_str = f"{score * 100:.1f}%"

            if baseline is not None and baseline > 0:
                change = (score - baseline) / baseline * 100
                if change > 0:
                    change_str = f"[green]+{change:.1f}%[/green]"
                elif change < -2:
                    change_str = f"[red]{change:.1f}%[/red]"
                else:
                    change_str = f"[yellow]{change:.1f}%[/yellow]"
            else:
                change_str = "—"

            bench_table.add_row(bench, before_str, after_str, change_str)

        # Training loss row
        bench_table.add_row(
            "Training Loss",
            f"{summary.training_loss_start:.4f}",
            f"{summary.training_loss_end:.4f}",
            f"[green]-{summary.loss_improvement * 100:.1f}%[/green]"
            if summary.loss_improvement > 0
            else f"[red]+{abs(summary.loss_improvement) * 100:.1f}%[/red]",
        )

        console.print(bench_table)

    # Strengths
    strengths = summary.strengths()
    if strengths:
        strength_text = "\n".join(f"  [green]+[/green] {s}" for s in strengths)
        console.print(Panel(strength_text, title="Strengths", border_style="green"))

    # Watch areas
    watch = summary.watch_areas()
    if watch:
        watch_text = "\n".join(f"  [yellow]![/yellow] {w}" for w in watch)
        console.print(Panel(watch_text, title="Watch Areas", border_style="yellow"))

    # Recommendations
    recs = summary.recommendations()
    if recs:
        rec_text = "\n".join(f"  {i}. {r}" for i, r in enumerate(recs, 1))
        console.print(Panel(rec_text, title="Recommendations", border_style="cyan"))

    # Export options
    console.print(
        Panel(
            "  [1] GGUF (for Ollama, llama.cpp) — Recommended for local use\n"
            "  [2] SafeTensors (for Transformers/HuggingFace)\n"
            "  [3] AWQ (4-bit quantized — best quality/size ratio)\n\n"
            "  Set in config: [cyan]serving.export_format: gguf[/cyan]",
            title="Export Options",
            border_style="dim",
        )
    )


def _display_plain(summary: TrainingSummary) -> None:
    """Fallback plain-text display when Rich is not available."""
    grade, grade_desc = summary.overall_grade
    print(f"\n{'=' * 50}")
    print("QUALITY REPORT CARD")
    print(f"{'=' * 50}")
    print(f"Grade: {grade} ({grade_desc})")
    print(f"Model: {summary.model_name}")
    print(f"Duration: {summary.duration_formatted}")
    print(f"Loss: {summary.training_loss_start:.4f} -> {summary.training_loss_end:.4f}")
    for s in summary.strengths():
        print(f"  + {s}")
    for w in summary.watch_areas():
        print(f"  ! {w}")
    for i, r in enumerate(summary.recommendations(), 1):
        print(f"  {i}. {r}")
    print(f"{'=' * 50}\n")
