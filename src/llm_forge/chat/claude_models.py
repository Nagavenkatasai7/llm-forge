"""Claude model catalogue — the single source of truth for LLM Forge.

``engine.py`` and ``orchestrator.py`` previously kept two hand-maintained copies
of this table, which drifted. Both now import from here.

Model IDs are exact strings with no date suffix. Costs are USD per million
tokens, input/output, at first-party Anthropic API rates.
"""

from __future__ import annotations

CLAUDE_MODELS: dict[str, dict[str, str]] = {
    "opus-5": {
        "id": "claude-opus-5",
        "name": "Claude Opus 5",
        "context": "1M",
        "cost": "$5/$25",
        "blurb": "Best for complex agentic coding and long-horizon work.",
    },
    "sonnet-5": {
        "id": "claude-sonnet-5",
        "name": "Claude Sonnet 5",
        "context": "1M",
        "cost": "$3/$15",
        "blurb": "Near-Opus quality on coding and agentic work, at lower cost.",
    },
    "haiku-4.5": {
        "id": "claude-haiku-4-5",
        "name": "Claude Haiku 4.5",
        "context": "200K",
        "cost": "$1/$5",
        "blurb": "Fastest and cheapest. Good for simple, high-volume tasks.",
    },
    "opus-4.8": {
        "id": "claude-opus-4-8",
        "name": "Claude Opus 4.8",
        "context": "1M",
        "cost": "$5/$25",
        "blurb": "Previous-generation Opus.",
    },
}

# Opus 5 is the default: this agent does long-horizon, multi-step work
# (dataset research, config generation, training supervision) where capability
# matters more than per-token cost.
DEFAULT_MODEL = "opus-5"

# Cheap model for background summarisation (session summaries, memory
# compaction) where the task is simple and volume is high.
SUMMARY_MODEL_ID = "claude-haiku-4-5"

# Friendly aliases accepted by the /model slash command.
MODEL_ALIASES: dict[str, str] = {
    "opus": "opus-5",
    "sonnet": "sonnet-5",
    "haiku": "haiku-4.5",
    "opus5": "opus-5",
    "opus-5": "opus-5",
    "sonnet5": "sonnet-5",
    "sonnet-5": "sonnet-5",
    "haiku4.5": "haiku-4.5",
    "opus4.8": "opus-4.8",
    "opus-4.8": "opus-4.8",
    # Retired keys from older releases, mapped forward so existing configs and
    # muscle memory keep working instead of erroring.
    "opus-4.6": "opus-5",
    "opus4.6": "opus-5",
    "opus-4.5": "opus-4.8",
    "opus4.5": "opus-4.8",
    "sonnet-4.6": "sonnet-5",
    "sonnet4.6": "sonnet-5",
    "sonnet-4.5": "sonnet-5",
    "sonnet4.5": "sonnet-5",
}


def resolve_model_key(key: str | None) -> str:
    """Map a user-supplied model name to a key in ``CLAUDE_MODELS``.

    Falls back to :data:`DEFAULT_MODEL` for unknown input so a typo degrades to
    a working session rather than a crash.
    """
    if not key:
        return DEFAULT_MODEL
    normalised = key.strip().lower()
    if normalised in CLAUDE_MODELS:
        return normalised
    return MODEL_ALIASES.get(normalised, DEFAULT_MODEL)


def model_id(key: str | None) -> str:
    """Return the exact API model ID for a model key or alias."""
    return CLAUDE_MODELS[resolve_model_key(key)]["id"]
