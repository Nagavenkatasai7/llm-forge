"""Tests for the OrchestratorEngine (Claude + ADK sub-agents)."""
from __future__ import annotations
import json
import os
import pathlib
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

    def test_no_module_embeds_an_obfuscated_credential(self) -> None:
        """Scan the WHOLE package, not just api_keys.py.

        The first version of this guard only checked api_keys.py, and missed a
        third embedded credential -- an NVIDIA key using the identical
        XOR+base64 scheme in nvidia_provider.py. A guard scoped to the one file
        you already knew about does not catch the file you didn't.
        """
        import re
        from pathlib import Path

        import llm_forge

        package_root = Path(llm_forge.__file__).parent
        offenders: list[str] = []

        # A long base64-ish literal in source is how every embedded key here
        # has been smuggled in so far.
        blob = re.compile(r'["\'][A-Za-z0-9+/]{60,}={0,2}["\']')

        for py_file in package_root.rglob("*.py"):
            source = py_file.read_text(encoding="utf-8", errors="replace")
            rel = py_file.relative_to(package_root)

            for marker in ("_XOR_KEY", "_deobfuscate", "_OBFUSCATED_KEY"):
                if marker in source:
                    offenders.append(f"{rel}: {marker}")

            for match in blob.finditer(source):
                offenders.append(f"{rel}: long base64 literal {match.group()[:24]}...")

        assert not offenders, "possible embedded credentials:\n  " + "\n  ".join(offenders)

    def test_no_provider_falls_back_to_a_bundled_key(self) -> None:
        """Every provider must return empty rather than a baked-in credential."""
        from llm_forge.chat.nvidia_provider import get_nvidia_api_key
        from llm_forge.chat.ollama_provider import get_ollama_api_key
        from llm_forge.chat import api_keys

        with patch.dict(os.environ, {}, clear=True), patch.object(
            api_keys, "ENV_FILE", pathlib.Path("/nonexistent/.env")
        ):
            assert api_keys.get_anthropic_api_key() == ""
            assert api_keys.get_google_api_key() == ""
            assert get_nvidia_api_key() == ""
            assert get_ollama_api_key() == ""

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
