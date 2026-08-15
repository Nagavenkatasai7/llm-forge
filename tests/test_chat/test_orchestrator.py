"""Tests for the OrchestratorEngine (Claude + ADK sub-agents)."""
from __future__ import annotations
import json
import os
from unittest.mock import MagicMock, patch
import pytest


class TestOrchestratorInit:
    def test_ships_no_builtin_keys(self, tmp_path) -> None:
        """Regression guard: the package must never bundle credentials.

        v3.0.0 shipped XOR+base64-obfuscated Anthropic and Google keys in
        api_keys.py. Those were published to a public repo and are compromised.
        With no env var and no ~/.llm-forge/.env, key lookup must come back empty
        rather than falling back to something baked into the source.
        """
        from llm_forge.chat import api_keys

        with patch.dict(os.environ, {}, clear=True), patch.object(
            api_keys, "ENV_FILE", tmp_path / "absent.env"
        ):
            assert api_keys.get_anthropic_api_key() == ""
            assert api_keys.get_google_api_key() == ""
            assert not api_keys.has_anthropic_api_key()

    def test_source_contains_no_obfuscated_key_blobs(self) -> None:
        """The deobfuscation machinery itself must be gone, not just unused."""
        import inspect

        from llm_forge.chat import api_keys

        source = inspect.getsource(api_keys)
        for marker in ("_XOR_KEY", "_deobfuscate", "b64decode", "_ANTHROPIC_OBF"):
            assert marker not in source, f"leftover key-obfuscation code: {marker}"

    def test_env_var_is_used(self) -> None:
        """A user-supplied env var is what gets returned."""
        from llm_forge.chat.api_keys import get_anthropic_api_key

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-user-custom"}):
            assert get_anthropic_api_key() == "sk-user-custom"

    def test_env_file_is_fallback(self, tmp_path) -> None:
        """~/.llm-forge/.env is read when the env var is unset."""
        from llm_forge.chat import api_keys

        env_file = tmp_path / ".env"
        env_file.write_text("# comment\nANTHROPIC_API_KEY=sk-from-file\n")

        with patch.dict(os.environ, {}, clear=True), patch.object(
            api_keys, "ENV_FILE", env_file
        ):
            assert api_keys.get_anthropic_api_key() == "sk-from-file"

    def test_require_raises_with_setup_instructions(self, tmp_path) -> None:
        """Missing key produces an actionable error, not a silent fallback."""
        from llm_forge.chat import api_keys

        with patch.dict(os.environ, {}, clear=True), patch.object(
            api_keys, "ENV_FILE", tmp_path / "absent.env"
        ):
            with pytest.raises(api_keys.MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
                api_keys.require_anthropic_api_key()

    def test_has_send_method(self) -> None:
        from llm_forge.chat.orchestrator import OrchestratorEngine
        assert callable(getattr(OrchestratorEngine, "send", None))

    def test_has_end_session_method(self) -> None:
        from llm_forge.chat.orchestrator import OrchestratorEngine
        assert callable(getattr(OrchestratorEngine, "end_session", None))


class TestOrchestratorSystemPrompt:
    def test_system_prompt_mentions_agents(self) -> None:
        from llm_forge.chat.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT
        assert "data_agent" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "config_agent" in ORCHESTRATOR_SYSTEM_PROMPT

    def test_system_prompt_mentions_delegation(self) -> None:
        from llm_forge.chat.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT
        assert "delegate" in ORCHESTRATOR_SYSTEM_PROMPT.lower()


class TestOrchestratorToolList:
    def test_delegate_tool_in_tools(self) -> None:
        from llm_forge.chat.orchestrator import ORCHESTRATOR_TOOLS
        tool_names = [t["name"] for t in ORCHESTRATOR_TOOLS]
        assert "delegate_to_agent" in tool_names

    def test_memory_tools_in_tools(self) -> None:
        from llm_forge.chat.orchestrator import ORCHESTRATOR_TOOLS
        tool_names = [t["name"] for t in ORCHESTRATOR_TOOLS]
        assert "save_memory" in tool_names
        assert "recall_memory" in tool_names
