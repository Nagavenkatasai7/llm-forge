"""Tests for the Ollama Cloud provider."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_forge.chat.ollama_provider import (
    OLLAMA_CLOUD_BASE_URL,
    OllamaError,
    _explain,
    default_model,
    describe_models,
    get_ollama_api_key,
    list_models,
    stream_ollama,
    supports_tools,
    to_openai_messages,
    to_openai_tools,
)


def fake_client(models=None, message=None, chunks=None):
    client = MagicMock()
    if models is not None:
        client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id=m) for m in models]
        )
    if message is not None:
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )
    if chunks is not None:
        client.chat.completions.create.return_value = iter(chunks)
    return client


class TestKeyResolution:
    def test_env_var_wins(self) -> None:
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "from-env"}):
            assert get_ollama_api_key() == "from-env"

    def test_no_bundled_fallback(self) -> None:
        """Regression guard, mirroring the other providers."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_ollama_api_key() == ""


class TestModelListing:
    def test_lists_sorted(self) -> None:
        client = fake_client(models=["qwen3.5:397b", "gpt-oss:20b"])
        assert list_models(client=client) == ["gpt-oss:20b", "qwen3.5:397b"]

    def test_api_failure_becomes_ollama_error(self) -> None:
        client = MagicMock()
        client.models.list.side_effect = RuntimeError("network down")
        with pytest.raises(OllamaError):
            list_models(client=client)

    def test_describe_reports_the_error_not_a_traceback(self) -> None:
        with patch(
            "llm_forge.chat.ollama_provider.list_models",
            side_effect=OllamaError("key rejected"),
        ):
            assert "key rejected" in describe_models()


class TestDefaultModel:
    def test_prefers_a_capable_model(self) -> None:
        client = fake_client(models=["gemma4:31b", "qwen3.5:397b", "gpt-oss:20b"])
        assert default_model(client=client) == "qwen3.5:397b"

    def test_falls_back_to_whatever_exists(self) -> None:
        """An Ollama model released after this code still has to work."""
        client = fake_client(models=["brand-new-model:1t"])
        assert default_model(client=client) == "brand-new-model:1t"

    def test_no_models_returns_none(self) -> None:
        assert default_model(client=fake_client(models=[])) is None


class TestToolSupportProbe:
    def test_model_that_calls_a_tool_passes(self) -> None:
        message = SimpleNamespace(tool_calls=[SimpleNamespace(id="1")], content=None)
        ok, reason = supports_tools("qwen3.5:397b", client=fake_client(message=message))
        assert ok
        assert "verified" in reason

    def test_model_that_only_talks_is_rejected(self) -> None:
        """Accepting it would leave the assistant chatty but unable to act."""
        message = SimpleNamespace(tool_calls=None, content="Sure, I can help!")
        ok, reason = supports_tools("chatty:1b", client=fake_client(message=message))
        assert not ok
        assert "cannot run the assistant" in reason


class TestErrorExplanations:
    def test_payment_required_names_the_billing_page(self) -> None:
        exc = RuntimeError("this model uses extra usage only ... balance is empty")
        assert "billing" in _explain(exc)

    def test_401_points_at_the_key(self) -> None:
        exc = RuntimeError("unauthorized")
        exc.status_code = 401
        assert "API key" in _explain(exc)

    def test_404_suggests_listing_models(self) -> None:
        exc = RuntimeError("not found")
        exc.status_code = 404
        assert "/model" in _explain(exc)


class TestMessageTranslation:
    def test_system_prompt_leads(self) -> None:
        out = to_openai_messages([], "you are forge")
        assert out[0] == {"role": "system", "content": "you are forge"}

    def test_plain_user_message(self) -> None:
        out = to_openai_messages([{"role": "user", "content": "hi"}], "sys")
        assert out[-1] == {"role": "user", "content": "hi"}

    def test_anthropic_tool_result_becomes_a_tool_message(self) -> None:
        """The engine stores tool results Anthropic-style; they must survive."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "{}"}
                ],
            }
        ]
        out = to_openai_messages(messages, "sys")
        assert out[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "{}"}

    def test_assistant_tool_calls_round_trip(self) -> None:
        """Second loop iteration breaks if these are dropped."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "detect_hardware", "arguments": "{}"},
                    }
                ],
            }
        ]
        out = to_openai_messages(messages, "sys")
        assert out[-1]["tool_calls"][0]["id"] == "call_1"
        # OpenAI rejects null content alongside tool_calls.
        assert out[-1]["content"] == ""

    def test_tool_role_messages_pass_through(self) -> None:
        messages = [{"role": "tool", "tool_call_id": "c1", "content": "result"}]
        out = to_openai_messages(messages, "sys")
        assert out[-1]["role"] == "tool"


class TestToolTranslation:
    def test_anthropic_schema_becomes_openai_function(self) -> None:
        tools = [
            {
                "name": "scan_data",
                "description": "Scan a dataset",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        out = to_openai_tools(tools)
        assert out[0]["type"] == "function"
        assert out[0]["function"]["name"] == "scan_data"
        assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


class TestStreaming:
    def _chunk(self, content=None, tool_calls=None):
        delta = SimpleNamespace(content=content, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

    def test_text_is_accumulated_and_streamed(self) -> None:
        seen: list[str] = []
        client = fake_client(chunks=[self._chunk("Hel"), self._chunk("lo")])
        response = stream_ollama([], "sys", model="m", on_text=seen.append, client=client)
        assert "".join(seen) == "Hello"
        assert response.choices[0].message.content == "Hello"

    def test_tool_arguments_are_reassembled_across_chunks(self) -> None:
        """Arguments arrive fragmented and only parse once concatenated."""
        tc_start = SimpleNamespace(
            index=0,
            id="call_1",
            function=SimpleNamespace(name="scan_data", arguments='{"path"'),
        )
        tc_more = SimpleNamespace(
            index=0, id=None, function=SimpleNamespace(name=None, arguments=': "d.jsonl"}')
        )
        client = fake_client(chunks=[self._chunk(tool_calls=[tc_start]),
                                     self._chunk(tool_calls=[tc_more])])
        response = stream_ollama([], "sys", model="m", client=client)

        call = response.choices[0].message.tool_calls[0]
        assert call.id == "call_1"
        assert call.function.name == "scan_data"
        import json

        assert json.loads(call.function.arguments) == {"path": "d.jsonl"}

    def test_interrupt_stops_consuming(self) -> None:
        client = fake_client(chunks=[self._chunk("a"), self._chunk("b")])
        response = stream_ollama(
            [], "sys", model="m", interrupt_check=lambda: True, client=client
        )
        assert response.choices[0].message.content == ""


class TestBaseUrl:
    def test_cloud_url_is_the_openai_compatible_one(self) -> None:
        assert OLLAMA_CLOUD_BASE_URL.endswith("/v1")
