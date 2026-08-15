"""Provider-agnostic LLM access for the utility tools.

Several tools need *an* LLM to do their job -- generating synthetic training
data, judging model outputs, writing a preprocessing script. They were each
hardcoded to NVIDIA NIM, which meant a user with a perfectly good Ollama key
and a live session already talking to a model was told:

    "Synthetic data generation needs an NVIDIA API key -- don't have one."

The session's own provider is right there. This module picks whichever
OpenAI-compatible provider is actually configured and hands back a client, so
those tools work with whatever the user has rather than one specific vendor.

Priority is Ollama (cloud key, or a local server needing no key), then NVIDIA,
then OpenAI. Anthropic is deliberately not used here: these are bulk
generation tasks where a cheaper endpoint is the right call, and the Anthropic
client has a different request shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class NoUtilityProviderError(RuntimeError):
    """No LLM provider is configured for utility tasks."""


SETUP_HELP = (
    "This tool needs an LLM provider. Configure any one of:\n"
    "  OLLAMA_API_KEY   https://ollama.com/settings/keys  (18+ models)\n"
    "  NVIDIA_API_KEY   https://build.nvidia.com/         (free tier)\n"
    "  OPENAI_API_KEY   https://platform.openai.com/api-keys\n"
    "Or run a local Ollama server (`ollama serve`), which needs no key."
)


@dataclass
class UtilityProvider:
    """A resolved provider: an OpenAI-compatible client plus a default model."""

    name: str
    client: Any
    model: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.name} ({self.model})"


def resolve_provider(prefer_model: str | None = None) -> UtilityProvider:
    """Return the best available OpenAI-compatible provider.

    Raises :class:`NoUtilityProviderError` with setup instructions when nothing
    is configured.
    """
    from llm_forge.chat.ollama_provider import (
        _client as ollama_client,
        default_model as ollama_default,
        has_ollama_api_key,
        is_local_ollama_running,
    )

    if has_ollama_api_key() or is_local_ollama_running():
        client = ollama_client()
        model = prefer_model or ollama_default(client=client)
        if model:
            return UtilityProvider("ollama", client, model)

    from llm_forge.chat.nvidia_provider import has_nvidia_api_key, nvidia_client

    if has_nvidia_api_key():
        return UtilityProvider(
            "nvidia", nvidia_client(), prefer_model or "meta/llama-3.3-70b-instruct"
        )

    import os

    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI

        return UtilityProvider("openai", OpenAI(), prefer_model or "gpt-4o-mini")

    raise NoUtilityProviderError(SETUP_HELP)


def complete(
    prompt: str,
    system: str | None = None,
    max_tokens: int = 4096,
    temperature: float | None = None,
    model: str | None = None,
    provider: UtilityProvider | None = None,
) -> tuple[str, UtilityProvider]:
    """Run one completion on whichever provider is available.

    Returns ``(text, provider)`` so callers can report which model produced the
    result -- worth surfacing when the output is training data.
    """
    if provider is None:
        provider = resolve_provider(prefer_model=model)

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": provider.model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        response = provider.client.chat.completions.create(**kwargs)
    except Exception as exc:
        if provider.name == "ollama":
            from llm_forge.chat.ollama_provider import _explain

            raise NoUtilityProviderError(_explain(exc)) from exc
        raise NoUtilityProviderError(f"{provider.name} request failed: {exc}") from exc

    return (response.choices[0].message.content or ""), provider


__all__ = [
    "SETUP_HELP",
    "NoUtilityProviderError",
    "UtilityProvider",
    "complete",
    "resolve_provider",
]
