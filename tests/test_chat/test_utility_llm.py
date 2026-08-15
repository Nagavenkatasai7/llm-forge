"""Tests for provider-agnostic LLM access used by the utility tools."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_forge.chat.utility_llm import (
    NoUtilityProviderError,
    UtilityProvider,
    complete,
    resolve_provider,
)


class TestProviderResolution:
    """The bug: a user with a working Ollama key was told to get an NVIDIA one."""

    def test_ollama_key_resolves_to_ollama(self) -> None:
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "fake"}, clear=True), patch(
            "llm_forge.chat.ollama_provider._client", return_value=MagicMock()
        ), patch(
            "llm_forge.chat.ollama_provider.default_model", return_value="kimi-k2.6"
        ):
            provider = resolve_provider()
        assert provider.name == "ollama"
        assert provider.model == "kimi-k2.6"

    def test_local_ollama_needs_no_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "llm_forge.chat.ollama_provider.is_local_ollama_running", return_value=True
        ), patch(
            "llm_forge.chat.ollama_provider._client", return_value=MagicMock()
        ), patch(
            "llm_forge.chat.ollama_provider.default_model", return_value="llama3"
        ):
            assert resolve_provider().name == "ollama"

    def test_nvidia_used_when_no_ollama(self) -> None:
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-x"}, clear=True), patch(
            "llm_forge.chat.ollama_provider.is_local_ollama_running", return_value=False
        ), patch("llm_forge.chat.nvidia_provider.nvidia_client", return_value=MagicMock()):
            assert resolve_provider().name == "nvidia"

    def test_ollama_preferred_over_nvidia(self) -> None:
        """The session's own provider should be used before a second vendor."""
        env = {"OLLAMA_API_KEY": "fake", "NVIDIA_API_KEY": "nvapi-x"}
        with patch.dict(os.environ, env, clear=True), patch(
            "llm_forge.chat.ollama_provider._client", return_value=MagicMock()
        ), patch(
            "llm_forge.chat.ollama_provider.default_model", return_value="kimi-k2.6"
        ):
            assert resolve_provider().name == "ollama"

    def test_nothing_configured_lists_every_option(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "llm_forge.chat.ollama_provider.is_local_ollama_running", return_value=False
        ):
            with pytest.raises(NoUtilityProviderError) as excinfo:
                resolve_provider()

        message = str(excinfo.value)
        for option in ("OLLAMA_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY"):
            assert option in message


class TestComplete:
    def _provider(self, text="result"):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )
        return UtilityProvider("ollama", client, "kimi-k2.6")

    def test_returns_text_and_provider(self) -> None:
        provider = self._provider("generated")
        text, used = complete("prompt", provider=provider)
        assert text == "generated"
        assert used.model == "kimi-k2.6"

    def test_uses_the_resolved_model_not_a_hardcoded_one(self) -> None:
        provider = self._provider()
        complete("prompt", provider=provider)
        kwargs = provider.client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "kimi-k2.6"

    def test_system_prompt_leads(self) -> None:
        provider = self._provider()
        complete("ask", system="you are a judge", provider=provider)
        messages = provider.client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "you are a judge"}

    def test_temperature_omitted_when_unset(self) -> None:
        provider = self._provider()
        complete("prompt", provider=provider)
        assert "temperature" not in provider.client.chat.completions.create.call_args.kwargs

    def test_failure_becomes_actionable_error(self) -> None:
        provider = self._provider()
        provider.client.chat.completions.create.side_effect = RuntimeError("boom")
        with pytest.raises(NoUtilityProviderError):
            complete("prompt", provider=provider)
