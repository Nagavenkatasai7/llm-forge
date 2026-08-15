"""Model and dataset discovery with evidence, for picking what to train.

Two jobs the agent cannot do well without this module:

1. **Find a base model that actually fits.** Search results are ranked by
   downloads, which says nothing about whether a model fits in the user's
   memory budget. :func:`search_models` reads the real parameter count from
   each repo's safetensors index and reports which training methods fit.

2. **Find a dataset with real ground truth.** "Ground truth" means each example
   carries a verifiably correct answer you can grade against -- not just
   plausible-looking text. :func:`search_datasets` scores that explicitly from
   Hub metadata (benchmark registration, held-out splits, a citing paper,
   labelled task category) and reports the evidence, so a recommendation can be
   checked rather than taken on faith.

Web search runs through Anthropic's server-side ``web_search`` tool, reusing
the API key the chat session already needs. No extra credential, and results
carry source URLs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Memory model
# ---------------------------------------------------------------------------

# Bytes of memory per parameter, by training method. Weights + gradients +
# optimizer state. Activations are added separately since they scale with
# batch size and sequence length, not parameter count.
BYTES_PER_PARAM: dict[str, float] = {
    # bf16 weights + bf16 grads + fp32 Adam moments and master weights
    "full": 16.0,
    # Same, but with an 8-bit optimizer (bitsandbytes / MLX equivalents)
    "full_8bit_optim": 8.0,
    # Frozen bf16 base + small trainable adapter
    "lora": 2.6,
    # 4-bit quantized frozen base + bf16 adapter, via bitsandbytes (CUDA only)
    "qlora": 0.9,
    # 4-bit quantized frozen base + adapter, via MLX (Apple Silicon only)
    "mlx_lora_4bit": 0.9,
}

METHOD_LABELS: dict[str, str] = {
    "full": "full fine-tune (bf16 + Adam)",
    "full_8bit_optim": "full fine-tune (bf16 + 8-bit optimizer)",
    "lora": "LoRA (bf16 base, frozen)",
    "qlora": "QLoRA (4-bit base, bitsandbytes/CUDA)",
    "mlx_lora_4bit": "MLX LoRA (4-bit base, Apple Silicon)",
}

# Which backends each method can actually run on.
METHOD_BACKENDS: dict[str, set[str]] = {
    "full": {"cuda", "mps", "mlx"},
    "full_8bit_optim": {"cuda", "mps", "mlx"},
    "lora": {"cuda", "mps", "mlx"},
    # bitsandbytes has no Metal backend.
    "qlora": {"cuda"},
    # MLX is Apple-only by construction.
    "mlx_lora_4bit": {"mps", "mlx"},
}


def activation_gb(seq_length: int, batch_size: int, hidden_size: int, layers: int) -> float:
    """Rough activation-memory estimate in GB, with gradient checkpointing on.

    Checkpointing stores one activation tensor per layer boundary rather than
    every intermediate, so this scales with ``layers`` rather than the far
    larger per-layer intermediate count.
    """
    bytes_per_elem = 2  # bf16
    per_layer = seq_length * batch_size * hidden_size * bytes_per_elem
    return (per_layer * layers) / (1024**3)


@dataclass
class FitVerdict:
    """Whether one training method fits a memory budget, and why."""

    method: str
    label: str
    required_gb: float
    budget_gb: float
    fits: bool
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "label": self.label,
            "required_gb": round(self.required_gb, 1),
            "budget_gb": round(self.budget_gb, 1),
            "fits": self.fits,
            "note": self.note,
        }


def assess_fit(
    num_params: float,
    budget_gb: float,
    *,
    seq_length: int = 2048,
    batch_size: int = 1,
    backend: str = "cuda",
) -> list[FitVerdict]:
    """Report which training methods fit ``num_params`` into ``budget_gb``.

    ``backend`` is ``"cuda"``, ``"mps"`` (Apple Silicon) or ``"mlx"``. On the
    Apple backends, 4-bit QLoRA via bitsandbytes is unavailable -- it is CUDA
    only -- so that verdict is reported as unsupported rather than as a fit.
    """
    # Activation term. Without config.json we approximate hidden size and depth
    # from parameter count; the estimate only needs to be the right order.
    hidden = max(512, int((num_params / 1e9) ** 0.5 * 2048))
    layers = max(8, int((num_params / 1e9) ** 0.4 * 24))
    act = activation_gb(seq_length, batch_size, hidden, layers)

    verdicts: list[FitVerdict] = []
    for method, bpp in BYTES_PER_PARAM.items():
        required = (num_params * bpp) / (1024**3) + act
        note = ""
        fits = required <= budget_gb

        if backend not in METHOD_BACKENDS[method]:
            fits = False
            if method == "qlora":
                note = (
                    "bitsandbytes 4-bit quantization is CUDA-only. On Apple "
                    "Silicon use mlx_lora_4bit instead (mlx_lm.convert -q)."
                )
            elif method == "mlx_lora_4bit":
                note = "MLX runs only on Apple Silicon."
            else:
                note = f"Not supported on the {backend} backend."
        elif method == "full" and fits and backend in {"mps", "mlx"}:
            note = "Feasible, but expect slow steps -- MPS has no fused optimizer."

        verdicts.append(
            FitVerdict(
                method=method,
                label=METHOD_LABELS[method],
                required_gb=required,
                budget_gb=budget_gb,
                fits=fits,
                note=note,
            )
        )
    return verdicts


def recommended_method(verdicts: list[FitVerdict]) -> str | None:
    """Pick the most capable method that fits.

    Preference order is quality-first: a full fine-tune updates every weight
    and beats LoRA when it fits, and LoRA beats QLoRA because quantizing the
    base model costs accuracy.
    """
    for method in ("full", "full_8bit_optim", "lora", "qlora", "mlx_lora_4bit"):
        for v in verdicts:
            if v.method == method and v.fits:
                return method
    return None


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

_PARAM_SUFFIX = re.compile(r"(\d+(?:\.\d+)?)\s*([bm])\b", re.IGNORECASE)


def params_from_name(repo_id: str) -> float | None:
    """Guess a parameter count from a repo name like ``Llama-3.2-1B``.

    Only a fallback -- the safetensors index is authoritative. Returns None
    when the name carries no size hint.
    """
    # Strip version-like numbers ("3.2", "v0.1") so they are not read as sizes.
    cleaned = re.sub(r"\b\d+\.\d+\b(?![bm])", " ", repo_id, flags=re.IGNORECASE)
    matches = _PARAM_SUFFIX.findall(cleaned)
    if not matches:
        return None

    # Take the largest match, not the last. MoE repos are named like
    # "Qwen3-Coder-30B-A3B" where 30B is total parameters and A3B is the active
    # count -- memory is bounded by the total, so reading the last match would
    # underestimate tenfold and recommend a model that OOMs.
    sizes = [
        float(value) * (1e9 if unit.lower() == "b" else 1e6) for value, unit in matches
    ]
    return max(sizes)


def params_from_hub(api: Any, repo_id: str) -> float | None:
    """Read the true parameter count from a repo's safetensors metadata."""
    try:
        info = api.model_info(repo_id, expand=["safetensors"])
    except Exception:
        return None

    safetensors = getattr(info, "safetensors", None)
    if not safetensors:
        return None

    total = getattr(safetensors, "total", None)
    if isinstance(total, int) and total > 0:
        return float(total)

    parameters = getattr(safetensors, "parameters", None)
    if isinstance(parameters, dict) and parameters:
        summed = sum(v for v in parameters.values() if isinstance(v, int))
        if summed > 0:
            return float(summed)
    return None


# ---------------------------------------------------------------------------
# Ground-truth scoring
# ---------------------------------------------------------------------------

# Task categories whose examples carry a checkable label.
_LABELLED_TASKS = {
    "question-answering",
    "text-classification",
    "token-classification",
    "multiple-choice",
    "zero-shot-classification",
    "translation",
    "summarization",
    "sentence-similarity",
    "table-question-answering",
    "visual-question-answering",
}

_PERMISSIVE_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd",
    "bsd-3-clause",
    "bsd-2-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "odc-by",
    "cdla-permissive-2.0",
}


@dataclass
class GroundTruthAssessment:
    """How verifiable a dataset's answers are, with the evidence for it."""

    score: int = 0
    max_score: int = 6
    evidence: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.score >= 5:
            return "strong"
        if self.score >= 3:
            return "moderate"
        if self.score >= 1:
            return "weak"
        return "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": f"{self.score}/{self.max_score}",
            "verdict": self.verdict,
            "evidence": self.evidence,
            "caveats": self.caveats,
        }


def _tag_values(tags: list[str], prefix: str) -> list[str]:
    """Pull values out of HF's ``prefix:value`` tag convention."""
    return [t.split(":", 1)[1] for t in tags if t.startswith(f"{prefix}:")]


def assess_ground_truth(
    tags: list[str],
    card_data: dict[str, Any] | None,
    splits: list[str] | None = None,
) -> GroundTruthAssessment:
    """Score how verifiable a dataset's answers are.

    "Ground truth" here means each example has an answer you can grade a model
    against automatically. The signals below are all things the Hub records, so
    the score is reproducible rather than a vibe.
    """
    assessment = GroundTruthAssessment()
    card_data = card_data or {}
    tags = tags or []

    # 1. Registered as a benchmark on the Hub -- the strongest single signal.
    if any(t.startswith("benchmark:") for t in tags):
        assessment.score += 2
        benchmarks = ", ".join(_tag_values(tags, "benchmark")) or "yes"
        assessment.evidence.append(f"Registered Hub benchmark ({benchmarks})")
    else:
        assessment.caveats.append("Not registered as a Hub benchmark")

    # 2. Task category implies labelled examples.
    task_categories = card_data.get("task_categories") or _tag_values(tags, "task_categories")
    if isinstance(task_categories, str):
        task_categories = [task_categories]
    labelled = sorted(set(task_categories or []) & _LABELLED_TASKS)
    if labelled:
        assessment.score += 1
        assessment.evidence.append(f"Labelled task type: {', '.join(labelled)}")
    elif task_categories:
        assessment.caveats.append(
            f"Task type {', '.join(task_categories)} has no inherent correct answer "
            "-- grading needs a judge model or human review"
        )
    else:
        assessment.caveats.append("No task category declared")

    # 3. A held-out split you can evaluate on without training on it.
    split_names = {s.lower() for s in (splits or [])}
    holdout = sorted(split_names & {"test", "validation", "valid", "eval", "dev"})
    if holdout:
        assessment.score += 1
        assessment.evidence.append(f"Held-out split available: {', '.join(holdout)}")
    elif splits:
        assessment.caveats.append(
            f"Only {', '.join(sorted(split_names))} split(s) -- you must carve out "
            "your own eval set to avoid measuring on training data"
        )

    # 4. A citing paper means the construction method is documented.
    arxiv_ids = _tag_values(tags, "arxiv")
    if arxiv_ids:
        assessment.score += 1
        assessment.evidence.append(f"Described in a paper (arXiv:{arxiv_ids[0]})")
    else:
        assessment.caveats.append("No linked paper describing how labels were produced")

    # 5. Human-produced or expert-produced annotations beat synthetic ones.
    annotators = _tag_values(tags, "annotations_creators")
    if any(a in {"crowdsourced", "expert-generated", "found"} for a in annotators):
        assessment.score += 1
        assessment.evidence.append(f"Human annotations ({', '.join(annotators)})")
    elif "machine-generated" in annotators:
        assessment.caveats.append(
            "Labels are machine-generated -- they inherit the errors of whatever "
            "model produced them"
        )

    return assessment


def _license_note(license_id: str | None) -> str:
    if not license_id:
        return "No license declared -- check the dataset card before any non-personal use"
    if license_id.lower() in _PERMISSIVE_LICENSES:
        return f"Permissive license ({license_id})"
    return f"License is {license_id} -- verify it permits your intended use"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _normalise_search_type(search_type: str) -> str:
    """Accept singular and plural forms.

    The tool schema in tools.py used the plural, while research_tools.py
    defaulted to the singular. The mismatch made every singular call fall
    through both branches and return an empty result set with no error.
    """
    value = (search_type or "").strip().lower()
    if value in {"model", "models"}:
        return "models"
    if value in {"dataset", "datasets"}:
        return "datasets"
    raise ValueError(f"search_type must be 'model' or 'dataset', got {search_type!r}")


def _first(value: Any) -> str | None:
    """HF card fields are sometimes a scalar, sometimes a one-element list."""
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value else None


def _card_dict(card_data: Any) -> dict[str, Any]:
    """Normalise HF's ``CardData`` object to a plain dict.

    ``CardData`` is dict-like but not a Mapping -- ``dict(card)`` raises
    KeyError because its ``__getitem__`` is keyed on attribute names while
    ``dict()`` tries integer indices.
    """
    if not card_data:
        return {}
    to_dict = getattr(card_data, "to_dict", None)
    if callable(to_dict):
        try:
            return dict(to_dict())
        except Exception:
            pass
    return dict(getattr(card_data, "__dict__", {}) or {})


def search_datasets(query: str, limit: int = 5, *, api: Any = None) -> dict[str, Any]:
    """Search Hub datasets and assess each one's ground truth."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    results: list[dict[str, Any]] = []
    for ds in api.list_datasets(search=query, sort="downloads", limit=limit, full=True):
        tags = list(ds.tags or [])
        card = _card_dict(ds.card_data)

        splits: list[str] = []
        for cfg in card.get("configs", []) or []:
            for entry in (cfg or {}).get("data_files", []) or []:
                split = (entry or {}).get("split")
                if split:
                    splits.append(str(split))
        if not splits:
            for info in card.get("dataset_info", []) or []:
                for split in (info or {}).get("splits", []) or []:
                    name = (split or {}).get("name")
                    if name:
                        splits.append(str(name))

        license_id = _first(card.get("license")) or _first(_tag_values(tags, "license"))
        assessment = assess_ground_truth(tags, card, splits)

        results.append(
            {
                "id": ds.id,
                "url": f"https://huggingface.co/datasets/{ds.id}",
                "downloads": ds.downloads,
                "likes": ds.likes,
                "gated": bool(getattr(ds, "gated", False)),
                "last_modified": str(getattr(ds, "last_modified", "") or ""),
                "size_category": _first(card.get("size_categories"))
                or _first(_tag_values(tags, "size_categories")),
                "languages": _tag_values(tags, "language")[:5],
                "splits": sorted(set(splits)),
                "license": license_id,
                "license_note": _license_note(license_id),
                "ground_truth": assessment.as_dict(),
            }
        )

    results.sort(
        key=lambda r: (r["ground_truth"]["score"], r["downloads"] or 0),
        reverse=True,
    )
    return {
        "query": query,
        "type": "datasets",
        "count": len(results),
        "results": results,
        "how_to_read": (
            "ground_truth.score rates how automatically verifiable the answers "
            "are. Prefer 'strong' for anything you intend to benchmark against; "
            "'weak' or 'none' means you will need an LLM judge or human review "
            "to measure quality. Always check license_note before use."
        ),
    }


def search_models(
    query: str,
    limit: int = 5,
    *,
    budget_gb: float | None = None,
    backend: str = "cuda",
    seq_length: int = 2048,
    api: Any = None,
) -> dict[str, Any]:
    """Search Hub models, with a real parameter count and a memory-fit verdict."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    results: list[dict[str, Any]] = []
    for m in api.list_models(search=query, sort="downloads", limit=limit, full=True):
        tags = list(m.tags or [])
        license_id = _first(_tag_values(tags, "license"))

        num_params = params_from_hub(api, m.id)
        params_source = "safetensors metadata"
        if num_params is None:
            num_params = params_from_name(m.id)
            params_source = "inferred from repo name (approximate)"

        entry: dict[str, Any] = {
            "id": m.id,
            "url": f"https://huggingface.co/{m.id}",
            "downloads": m.downloads,
            "likes": m.likes,
            "pipeline_tag": m.pipeline_tag,
            "library": m.library_name,
            "license": license_id,
            "license_note": _license_note(license_id),
            # "manual"/"auto" both mean access must be requested on the Hub.
            "gated": bool(getattr(m, "gated", False)),
            "parameters": int(num_params) if num_params else None,
            "parameters_human": f"{num_params / 1e9:.1f}B" if num_params else "unknown",
            "parameters_source": params_source if num_params else "unavailable",
        }

        if entry["gated"]:
            entry["access_note"] = (
                "Gated repo -- you must accept its terms on the Hub and be "
                "logged in (`huggingface-cli login`) before download will work."
            )

        if num_params and budget_gb:
            verdicts = assess_fit(
                num_params, budget_gb, seq_length=seq_length, backend=backend
            )
            best = recommended_method(verdicts)
            entry["fit"] = {
                "budget_gb": round(budget_gb, 1),
                "backend": backend,
                "recommended_method": best,
                "recommended_label": METHOD_LABELS[best] if best else None,
                "methods": [v.as_dict() for v in verdicts],
            }
            if best is None:
                entry["fit"]["note"] = (
                    f"Does not fit in {budget_gb:.0f} GB by any method. "
                    "Pick a smaller base model."
                )
        elif budget_gb:
            entry["fit"] = {
                "note": "Parameter count unavailable -- cannot assess memory fit."
            }

        results.append(entry)

    return {
        "query": query,
        "type": "models",
        "count": len(results),
        "results": results,
    }


def search_huggingface(
    query: str,
    search_type: str = "model",
    *,
    limit: int = 5,
    budget_gb: float | None = None,
    backend: str = "cuda",
    api: Any = None,
) -> str:
    """JSON-returning entry point used by the agent tools."""
    try:
        kind = _normalise_search_type(search_type)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    try:
        if kind == "datasets":
            payload = search_datasets(query, limit=limit, api=api)
        else:
            payload = search_models(
                query, limit=limit, budget_gb=budget_gb, backend=backend, api=api
            )
    except ImportError:
        return json.dumps(
            {"error": "huggingface_hub not installed. Run: pip install huggingface_hub"}
        )
    except Exception as exc:  # network failures, Hub outages, rate limits
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "query": query})

    return json.dumps(payload, indent=2, default=str)


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

# Dynamic-filtering variant. Supported on Opus 5 / 4.8 / 4.7 / 4.6, Sonnet 5,
# and Sonnet 4.6 -- all of the models in our catalogue.
WEB_SEARCH_TOOL_TYPE = "web_search_20260209"


def web_search(query: str, *, max_uses: int = 5, client: Any = None) -> str:
    """Search the web via Anthropic's server-side web_search tool.

    Reuses the Anthropic key the chat session already needs, so this adds no
    new credential. Results come back with source URLs so claims can be
    checked.
    """
    from llm_forge.chat.claude_models import DEFAULT_MODEL, model_id

    if client is None:
        from llm_forge.chat.api_keys import MissingAPIKeyError, require_anthropic_api_key

        try:
            key = require_anthropic_api_key()
        except MissingAPIKeyError as exc:
            return json.dumps({"error": str(exc), "query": query})

        import anthropic

        client = anthropic.Anthropic(api_key=key)

    try:
        response = client.messages.create(
            model=model_id(DEFAULT_MODEL),
            max_tokens=4096,
            tools=[
                {
                    "type": WEB_SEARCH_TOOL_TYPE,
                    "name": "web_search",
                    "max_uses": max_uses,
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Search the web and answer: {query}\n\n"
                        "Cite the source URL for every factual claim. If sources "
                        "disagree, say so rather than picking one silently."
                    ),
                }
            ],
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "query": query})

    answer_parts: list[str] = []
    sources: list[dict[str, str]] = []
    seen: set[str] = set()

    for block in response.content:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            answer_parts.append(block.text)
        elif block_type == "web_search_tool_result":
            content = getattr(block, "content", None)
            # An error comes back as a single object, a success as a list.
            if not isinstance(content, list):
                code = getattr(content, "error_code", "unknown")
                sources.append({"error": f"web search failed: {code}"})
                continue
            for item in content:
                url = getattr(item, "url", "")
                if url and url not in seen:
                    seen.add(url)
                    sources.append({"url": url, "title": getattr(item, "title", "")})

    return json.dumps(
        {
            "query": query,
            "answer": "\n".join(answer_parts).strip(),
            "sources": sources,
            "stop_reason": response.stop_reason,
        },
        indent=2,
    )
