"""Curated model catalog and HuggingFace Hub search for the UI.

Provides a built-in catalog of popular fine-tuning models grouped by size,
plus live search against the HuggingFace Hub API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("ui.model_catalog")


@dataclass
class ModelEntry:
    """A model entry in the catalog."""

    model_id: str
    name: str
    size: str
    params: str
    license: str
    description: str
    category: str


# ---------------------------------------------------------------------------
# Curated catalog — models known to work well with llm-forge
# ---------------------------------------------------------------------------

CATALOG: list[ModelEntry] = [
    # --- Tiny (testing / CPU) ---
    ModelEntry(
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        name="SmolLM2 135M Instruct",
        size="tiny",
        params="135M",
        license="Apache-2.0",
        description="Smallest model for quick testing. Runs on CPU. Good for verifying configs.",
        category="Tiny (CPU testing)",
    ),
    ModelEntry(
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        name="SmolLM2 360M Instruct",
        size="tiny",
        params="360M",
        license="Apache-2.0",
        description="Slightly larger test model. Still runs on CPU with small batch sizes.",
        category="Tiny (CPU testing)",
    ),
    # --- Small (laptop GPU / Apple M-series) ---
    ModelEntry(
        model_id="unsloth/Llama-3.2-1B-Instruct",
        name="Llama 3.2 1B Instruct",
        size="small",
        params="1.2B",
        license="Llama 3.2",
        description="Excellent for domain-specific fine-tuning. Fast to train, good quality. "
        "Unsloth-optimized for faster download.",
        category="Small (8-16 GB)",
    ),
    ModelEntry(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        name="Qwen 2.5 1.5B Instruct",
        size="small",
        params="1.5B",
        license="Apache-2.0",
        description="Strong multilingual model from Alibaba. Good at code and math.",
        category="Small (8-16 GB)",
    ),
    ModelEntry(
        model_id="microsoft/Phi-3.5-mini-instruct",
        name="Phi 3.5 Mini",
        size="small",
        params="3.8B",
        license="MIT",
        description="Microsoft's compact model. Punches above its weight on reasoning tasks.",
        category="Small (8-16 GB)",
    ),
    ModelEntry(
        model_id="unsloth/Llama-3.2-3B-Instruct",
        name="Llama 3.2 3B Instruct",
        size="small",
        params="3.2B",
        license="Llama 3.2",
        description="Good balance of quality and speed. Fits on most GPUs with QLoRA.",
        category="Small (8-16 GB)",
    ),
    # --- Medium (24-48 GB GPU) ---
    ModelEntry(
        model_id="unsloth/Meta-Llama-3.1-8B-Instruct",
        name="Llama 3.1 8B Instruct",
        size="medium",
        params="8B",
        license="Llama 3.1",
        description="The workhorse of open-source LLMs. Strong general-purpose model. "
        "Use QLoRA to fit on a 24 GB GPU.",
        category="Medium (24-48 GB GPU)",
    ),
    ModelEntry(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        name="Mistral 7B v0.3 Instruct",
        size="medium",
        params="7.2B",
        license="Apache-2.0",
        description="Efficient architecture with sliding window attention. "
        "Apache-2.0 licensed — no restrictions.",
        category="Medium (24-48 GB GPU)",
    ),
    ModelEntry(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        name="Qwen 2.5 7B Instruct",
        size="medium",
        params="7.6B",
        license="Apache-2.0",
        description="Excellent multilingual + code model. Top performer in its size class.",
        category="Medium (24-48 GB GPU)",
    ),
    ModelEntry(
        model_id="google/gemma-2-9b-it",
        name="Gemma 2 9B Instruct",
        size="medium",
        params="9.2B",
        license="Gemma",
        description="Google's open model. Strong on knowledge tasks and creative writing.",
        category="Medium (24-48 GB GPU)",
    ),
    # --- Large (80 GB GPU / multi-GPU) ---
    ModelEntry(
        model_id="meta-llama/Llama-3.1-70B-Instruct",
        name="Llama 3.1 70B Instruct",
        size="large",
        params="70B",
        license="Llama 3.1",
        description="Frontier-class open model. Requires A100 80GB or multi-GPU setup. "
        "QLoRA can fit on 1x A100.",
        category="Large (80 GB+ / multi-GPU)",
    ),
    ModelEntry(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        name="Qwen 2.5 72B Instruct",
        size="large",
        params="72.7B",
        license="Qwen",
        description="One of the strongest open models. Matches GPT-4 on many benchmarks.",
        category="Large (80 GB+ / multi-GPU)",
    ),
]

# Group for display
CATEGORIES = [
    "Tiny (CPU testing)",
    "Small (8-16 GB)",
    "Medium (24-48 GB GPU)",
    "Large (80 GB+ / multi-GPU)",
]


def get_catalog_choices() -> list[str]:
    """Return model IDs grouped by category for a dropdown."""
    return [m.model_id for m in CATALOG]


def get_catalog_table() -> str:
    """Return a Markdown table of all catalog models."""
    lines = [
        "| Model | Size | License | Description |",
        "|---|---|---|---|",
    ]
    current_cat = ""
    for m in CATALOG:
        if m.category != current_cat:
            current_cat = m.category
            lines.append(f"| **{current_cat}** | | | |")
        lines.append(f"| `{m.model_id}` | {m.params} | {m.license} | {m.description} |")
    return "\n".join(lines)


def get_model_info(model_id: str) -> str:
    """Return description for a catalog model, or empty string if not found."""
    for m in CATALOG:
        if m.model_id == model_id:
            return f"**{m.name}** ({m.params})\nLicense: {m.license}\n{m.description}"
    return ""


def search_hub_models(
    query: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search HuggingFace Hub for text-generation models.

    Returns a list of dicts with keys: id, downloads, likes, pipeline_tag.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        models = api.list_models(
            search=query,
            pipeline_tag="text-generation",
            sort="downloads",
            limit=limit,
        )
        results = []
        for m in models:
            results.append(
                {
                    "id": m.id,
                    "downloads": _format_number(m.downloads or 0),
                    "likes": m.likes or 0,
                    "pipeline_tag": m.pipeline_tag or "",
                }
            )
        return results
    except Exception as exc:
        logger.warning("HuggingFace Hub search failed: %s", exc)
        return []


def format_hub_results(results: list[dict[str, Any]]) -> str:
    """Format hub search results as a Markdown table."""
    if not results:
        return "No models found. Try a different search term."
    lines = [
        "| Model ID | Downloads | Likes |",
        "|---|---|---|",
    ]
    for r in results:
        lines.append(f"| `{r['id']}` | {r['downloads']} | {r['likes']} |")
    return "\n".join(lines)


def _format_number(n: int) -> str:
    """Format large numbers with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
