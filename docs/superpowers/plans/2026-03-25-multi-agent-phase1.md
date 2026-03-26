# Multi-Agent Orchestration — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-LLM ChatEngine with a Claude orchestrator that delegates to Google ADK sub-agents, starting with Data Agent and Config Agent.

**Architecture:** Claude Sonnet 4.6 orchestrates via `delegate_to_agent` tool. Each sub-agent is a Google ADK `Agent` backed by Gemini 2.5 Flash. Tool implementations are extracted from the monolithic `tools.py` into per-agent modules. The existing MemoryManager and PermissionSystem are preserved. Google ADK is fully async — we bridge with `asyncio.run()` in the sync `delegate()` method.

**Tech Stack:** anthropic SDK, google-adk, google-genai, Python 3.12, Pydantic v2, pytest

**Design Spec:** `docs/design/multi_agent_orchestration_spec.md`

---

## File Structure

### New Files (create)

| File | Responsibility |
|------|---------------|
| `src/llm_forge/chat/orchestrator.py` | `OrchestratorEngine` — Claude orchestrator, delegates to sub-agents |
| `src/llm_forge/chat/agents/__init__.py` | Package init, exports `AgentManager` |
| `src/llm_forge/chat/agents/base.py` | `BaseAgent` wrapper around Google ADK Agent + Runner |
| `src/llm_forge/chat/agents/data_agent.py` | Data Agent — scans data, validates quality |
| `src/llm_forge/chat/agents/config_agent.py` | Config Agent — writes/validates YAML configs |
| `src/llm_forge/chat/agent_tools/__init__.py` | Package init, tool registry |
| `src/llm_forge/chat/agent_tools/data_tools.py` | Extracted data tool implementations |
| `src/llm_forge/chat/agent_tools/config_tools.py` | Extracted config tool implementations |
| `tests/test_chat/test_agents/__init__.py` | Test package |
| `tests/test_chat/test_agents/test_base_agent.py` | Tests for BaseAgent wrapper |
| `tests/test_chat/test_agents/test_data_agent.py` | Tests for Data Agent |
| `tests/test_chat/test_agents/test_config_agent.py` | Tests for Config Agent |
| `tests/test_chat/test_orchestrator.py` | Tests for OrchestratorEngine |
| `tests/test_chat/test_agent_tools/` | Tests for extracted tool modules |

### Modified Files

| File | Changes |
|------|---------|
| `src/llm_forge/chat/__init__.py` | Export `OrchestratorEngine` alongside `ChatEngine` |
| `src/llm_forge/chat/engine.py` | Add `OrchestratorEngine` import, keep `ChatEngine` as legacy |
| `src/llm_forge/chat/ui.py` | Update `launch_chat()` to use `OrchestratorEngine` when both API keys present |
| `src/llm_forge/cli.py` | Add `--gemini-key` option, update provider logic |
| `pyproject.toml` | Add `google-adk>=1.0`, `google-genai>=1.0` to chat extras |

### Kept Unchanged

| File | Reason |
|------|--------|
| `src/llm_forge/chat/tools.py` | Kept intact in Phase 1 (tools extracted as copies, not moved yet) |
| `src/llm_forge/chat/memory.py` | Reused as-is by OrchestratorEngine |
| `src/llm_forge/chat/execution.py` | Reused as-is (PermissionSystem, execution tools) |
| `src/llm_forge/chat/training_monitor.py` | Reused as-is |
| `src/llm_forge/chat/tui.py` | Updated in Phase 4 |
| `src/llm_forge/chat/slash_commands.py` | Updated in Phase 4 |
| All existing tests | Must continue passing |

---

## Task 1: Add google-adk dependency

**Files:**
- Modify: `pyproject.toml` (lines 55-60, chat extras section)

- [ ] **Step 1: Read current chat extras**

Run: `grep -A 10 'chat = \[' pyproject.toml` from project root
Expected: See current `anthropic`, `openai`, `prompt_toolkit`, `textual` deps

- [ ] **Step 2: Add google-adk and google-genai to chat extras**

In `pyproject.toml`, update the `chat` extras:

```toml
chat = [
    "anthropic>=0.40",
    "openai>=1.50",
    "google-adk>=1.0,<2.0",
    "google-genai>=1.0,<2.0",
    "prompt_toolkit>=3.0",
    "textual>=1.0",
]
```

- [ ] **Step 3: Install updated deps in local venv**

Run: `cd "/Users/nagavenkatasaichennu/Library/Mobile Documents/com~apple~CloudDocs/build_your_own_llm/llm-forge" && .venv/bin/pip install -e ".[chat]"`
Expected: google-adk and google-genai install successfully

- [ ] **Step 4: Verify imports work**

Run: `.venv/bin/python -c "from google.adk.agents import Agent; from google.adk.sessions import InMemorySessionService; from google.adk.runners import InMemoryRunner; print('ADK imports: ok')"`
Expected: Prints "ADK imports: ok" without ImportError

Run: `.venv/bin/python -c "from google.genai import types; c = types.Content(role='user', parts=[types.Part(text='test')]); print('GenAI types: ok')"`
Expected: Prints "GenAI types: ok"

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add google-adk and google-genai to chat extras"
```

---

## Task 2: Extract data tools from tools.py

**Files:**
- Create: `src/llm_forge/chat/agent_tools/__init__.py`
- Create: `src/llm_forge/chat/agent_tools/data_tools.py`
- Create: `tests/test_chat/test_agent_tools/__init__.py`
- Create: `tests/test_chat/test_agent_tools/test_data_tools.py`
- Reference: `src/llm_forge/chat/tools.py` (do NOT modify — copy functions out)

These functions are **copied** from `tools.py` into `data_tools.py`. The originals stay in `tools.py` so existing `ChatEngine` and all 894 tests continue working. We'll remove duplicates in Phase 4.

- [ ] **Step 1: Create agent_tools package**

Create `src/llm_forge/chat/agent_tools/__init__.py`:

```python
"""Per-agent tool modules extracted from the monolithic tools.py."""
```

- [ ] **Step 2: Write failing test for data tools**

Create `tests/test_chat/test_agent_tools/__init__.py` (empty).

Create `tests/test_chat/test_agent_tools/test_data_tools.py`:

```python
"""Tests for extracted data tools."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestScanData:
    """Tests for scan_data tool."""

    def test_scan_jsonl_file(self, tmp_path: Path) -> None:
        """scan_data returns format info for a JSONL file."""
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(
            '{"instruction": "Hi", "input": "", "output": "Hello"}\n'
            '{"instruction": "Bye", "input": "", "output": "Goodbye"}\n'
        )
        from llm_forge.chat.agent_tools.data_tools import scan_data

        result = json.loads(scan_data(str(data_file)))
        assert result["status"] == "success"
        assert result["num_samples"] >= 2

    def test_scan_nonexistent_path(self) -> None:
        """scan_data returns error for missing path."""
        from llm_forge.chat.agent_tools.data_tools import scan_data

        result = json.loads(scan_data("/nonexistent/path.jsonl"))
        assert result["status"] == "error"

    def test_scan_csv_file(self, tmp_path: Path) -> None:
        """scan_data handles CSV files."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("instruction,output\nHi,Hello\nBye,Goodbye\n")
        from llm_forge.chat.agent_tools.data_tools import scan_data

        result = json.loads(scan_data(str(csv_file)))
        assert result["status"] == "success"


class TestDetectHardware:
    """Tests for detect_hardware tool."""

    def test_returns_valid_json(self) -> None:
        """detect_hardware returns parseable JSON."""
        from llm_forge.chat.agent_tools.data_tools import detect_hardware

        result = json.loads(detect_hardware())
        assert "os" in result or "platform" in result or "cpu" in result


class TestSearchHuggingface:
    """Tests for search_huggingface tool (mocked)."""

    def test_search_returns_json(self) -> None:
        """search_huggingface returns valid JSON."""
        from llm_forge.chat.agent_tools.data_tools import search_huggingface

        result = json.loads(search_huggingface("test", "model"))
        # Should return a result structure (may be empty list if no network)
        assert isinstance(result, (dict, list))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_chat/test_agent_tools/test_data_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_forge.chat.agent_tools.data_tools'`

- [ ] **Step 4: Create data_tools.py by extracting from tools.py**

Create `src/llm_forge/chat/agent_tools/data_tools.py`. Extract these functions from `tools.py`, making them public (remove leading underscore) and self-contained:

```python
"""Data-related tools for the Data Agent.

Extracted from the monolithic chat/tools.py. Each function is a standalone
tool that can be registered with a Google ADK Agent.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path


def scan_data(path: str) -> str:
    """Analyze a dataset file, directory, or HuggingFace dataset ID.

    Returns JSON with: status, format, num_samples, columns, preview, size.
    """
    # Import the original implementation to avoid code duplication
    from llm_forge.chat.tools import _scan_data

    return _scan_data(path)


def detect_hardware() -> str:
    """Detect OS, CPU, RAM, GPU (NVIDIA CUDA, Apple MPS).

    Returns JSON with hardware details and training recommendations.
    """
    from llm_forge.chat.tools import _detect_hardware

    return _detect_hardware()


def search_huggingface(query: str, search_type: str = "model") -> str:
    """Search HuggingFace Hub for models or datasets.

    Args:
        query: Search query string.
        search_type: Either "model" or "dataset".

    Returns JSON with matched results.
    """
    from llm_forge.chat.tools import _search_huggingface

    return _search_huggingface(query, search_type)


def download_model(model_name: str, cache_dir: str | None = None) -> str:
    """Download a model from HuggingFace Hub.

    Args:
        model_name: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B").
        cache_dir: Optional local cache directory.

    Returns JSON with download status and path.
    """
    from llm_forge.chat.tools import _download_model

    return _download_model(model_name, cache_dir)


def show_model_info(model_path: str) -> str:
    """Show model size, architecture, training config, available checkpoints.

    Args:
        model_path: Path to local model directory or HuggingFace model ID.

    Returns JSON with model metadata.
    """
    from llm_forge.chat.tools import _show_model_info

    return _show_model_info(model_path)


# Google ADK tool definitions (JSON schemas for Gemini tool_use)
DATA_TOOL_DEFINITIONS = [
    {
        "name": "scan_data",
        "description": (
            "Analyze a dataset file (JSONL, CSV, Parquet), directory, or HuggingFace "
            "dataset ID. Returns format, sample count, columns, preview, and size."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path, directory, or HuggingFace dataset ID",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "detect_hardware",
        "description": (
            "Detect hardware: OS, CPU, RAM, GPU (NVIDIA/Apple MPS). "
            "Returns training recommendations based on available resources."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "search_huggingface",
        "description": "Search HuggingFace Hub for models or datasets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "search_type": {
                    "type": "string",
                    "enum": ["model", "dataset"],
                    "description": "Search for models or datasets",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "download_model",
        "description": "Download a model from HuggingFace Hub.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "HuggingFace model ID (e.g., meta-llama/Llama-3.2-1B)",
                },
                "cache_dir": {
                    "type": "string",
                    "description": "Optional local cache directory",
                },
            },
            "required": ["model_name"],
        },
    },
    {
        "name": "show_model_info",
        "description": "Show model size, architecture, and available checkpoints.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "Path to model or HuggingFace model ID",
                },
            },
            "required": ["model_path"],
        },
    },
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_chat/test_agent_tools/test_data_tools.py -v`
Expected: All tests PASS

- [ ] **Step 6: Verify existing tests still pass**

Run: `.venv/bin/python -m pytest tests/test_chat/test_tools.py -v --tb=short`
Expected: All existing tool tests PASS (no regressions)

- [ ] **Step 7: Commit**

```bash
git add src/llm_forge/chat/agent_tools/ tests/test_chat/test_agent_tools/
git commit -m "feat: extract data tools from monolithic tools.py"
```

---

## Task 3: Extract config tools from tools.py

**Files:**
- Create: `src/llm_forge/chat/agent_tools/config_tools.py`
- Create: `tests/test_chat/test_agent_tools/test_config_tools.py`
- Reference: `src/llm_forge/chat/tools.py` (do NOT modify)

- [ ] **Step 1: Write failing test for config tools**

Create `tests/test_chat/test_agent_tools/test_config_tools.py`:

```python
"""Tests for extracted config tools."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestWriteConfig:
    """Tests for write_config tool."""

    def test_write_valid_config(self, tmp_path: Path) -> None:
        """write_config creates a YAML file."""
        from llm_forge.chat.agent_tools.config_tools import write_config

        output = tmp_path / "test_config.yaml"
        config = {
            "project_name": "test",
            "model": {"name": "SmolLM2-135M"},
            "data": {"train_path": "data/train.jsonl"},
            "training": {"mode": "lora", "num_train_epochs": 1},
        }
        result = json.loads(write_config(str(output), config))
        assert result["status"] == "success"
        assert output.exists()

    def test_write_config_creates_parent_dirs(self, tmp_path: Path) -> None:
        """write_config creates parent directories if needed."""
        from llm_forge.chat.agent_tools.config_tools import write_config

        output = tmp_path / "nested" / "dir" / "config.yaml"
        config = {"project_name": "test"}
        result = json.loads(write_config(str(output), config))
        assert output.parent.exists()


class TestValidateConfig:
    """Tests for validate_config tool."""

    def test_validate_nonexistent_file(self) -> None:
        """validate_config returns error for missing file."""
        from llm_forge.chat.agent_tools.config_tools import validate_config

        result = json.loads(validate_config("/nonexistent/config.yaml"))
        assert "error" in result.get("status", "").lower() or "error" in json.dumps(result).lower()


class TestListConfigs:
    """Tests for list_configs tool."""

    def test_list_returns_json(self) -> None:
        """list_configs returns valid JSON."""
        from llm_forge.chat.agent_tools.config_tools import list_configs

        result = json.loads(list_configs())
        assert isinstance(result, (dict, list))


class TestEstimateTraining:
    """Tests for estimate_training tool."""

    def test_estimate_returns_json(self) -> None:
        """estimate_training returns parseable JSON."""
        from llm_forge.chat.agent_tools.config_tools import estimate_training

        result = json.loads(
            estimate_training(
                model_name="SmolLM2-135M",
                mode="lora",
                num_samples=1000,
            )
        )
        assert isinstance(result, dict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_chat/test_agent_tools/test_config_tools.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create config_tools.py**

Create `src/llm_forge/chat/agent_tools/config_tools.py`:

```python
"""Config-related tools for the Config Agent.

Extracted from the monolithic chat/tools.py. Each function is a standalone
tool that can be registered with a Google ADK Agent.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_config(output_path: str, config: dict) -> str:
    """Write a YAML training configuration file.

    Args:
        output_path: Where to save the YAML file.
        config: Configuration dictionary.

    Returns JSON with status and path.
    """
    from llm_forge.chat.tools import _write_config

    return _write_config(output_path, config)


def validate_config(config_path: str) -> str:
    """Validate a YAML config against the llm-forge Pydantic schema.

    Args:
        config_path: Path to the YAML config file.

    Returns JSON with validation result or errors.
    """
    from llm_forge.chat.tools import _validate_config

    return _validate_config(config_path)


def list_configs() -> str:
    """List available example/template configs.

    Returns JSON with config names and descriptions.
    """
    from llm_forge.chat.tools import _list_configs

    return _list_configs()


def estimate_training(
    model_name: str,
    mode: str,
    num_samples: int = 1000,
    num_epochs: int = 1,
    batch_size: int = 4,
    seq_length: int = 2048,
) -> str:
    """Estimate training time, memory usage, and hardware requirements.

    Args:
        model_name: Model name or HuggingFace ID.
        mode: Training mode (lora, qlora, full).
        num_samples: Number of training samples.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        seq_length: Max sequence length.

    Returns JSON with estimates.
    """
    from llm_forge.chat.tools import _estimate_training

    return _estimate_training(model_name, mode, num_samples, num_epochs, batch_size, seq_length)


# Google ADK tool definitions
CONFIG_TOOL_DEFINITIONS = [
    {
        "name": "write_config",
        "description": (
            "Write a YAML training config file. Takes a config dictionary "
            "and saves it to the specified path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "output_path": {
                    "type": "string",
                    "description": "Where to save the YAML config file",
                },
                "config": {
                    "type": "object",
                    "description": "Configuration dictionary with model, data, training sections",
                },
            },
            "required": ["output_path", "config"],
        },
    },
    {
        "name": "validate_config",
        "description": "Validate a YAML config against the llm-forge schema.",
        "parameters": {
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "Path to the YAML config file",
                },
            },
            "required": ["config_path"],
        },
    },
    {
        "name": "list_configs",
        "description": "List available example/template configs.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "estimate_training",
        "description": (
            "Estimate training time, VRAM usage, and hardware fit for a "
            "given model, dataset size, and training mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Model name (e.g., SmolLM2-135M, Llama-3.2-1B)",
                },
                "mode": {
                    "type": "string",
                    "enum": ["lora", "qlora", "full"],
                    "description": "Training mode",
                },
                "num_samples": {
                    "type": "integer",
                    "description": "Number of training samples",
                },
                "num_epochs": {"type": "integer", "description": "Number of epochs"},
                "batch_size": {"type": "integer", "description": "Per-device batch size"},
                "seq_length": {"type": "integer", "description": "Max sequence length"},
            },
            "required": ["model_name", "mode"],
        },
    },
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_chat/test_agent_tools/test_config_tools.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_forge/chat/agent_tools/config_tools.py tests/test_chat/test_agent_tools/test_config_tools.py
git commit -m "feat: extract config tools from monolithic tools.py"
```

---

## Task 4: Create BaseAgent wrapper for Google ADK

**Files:**
- Create: `src/llm_forge/chat/agents/__init__.py`
- Create: `src/llm_forge/chat/agents/base.py`
- Create: `tests/test_chat/test_agents/__init__.py`
- Create: `tests/test_chat/test_agents/test_base_agent.py`

- [ ] **Step 1: Write failing test for BaseAgent**

Create `tests/test_chat/test_agents/__init__.py` (empty).

Create `tests/test_chat/test_agents/test_base_agent.py`:

```python
"""Tests for BaseAgent wrapper around Google ADK."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestAgentManager:
    """Tests for the AgentManager that holds all sub-agents."""

    def test_agent_manager_init(self) -> None:
        """AgentManager initializes without errors."""
        from llm_forge.chat.agents.base import AgentManager

        # Should raise if no API key — test the validation
        with pytest.raises((ValueError, KeyError, Exception)):
            AgentManager(gemini_api_key="")

    def test_agent_manager_has_registry(self) -> None:
        """AgentManager exposes an agent registry."""
        from llm_forge.chat.agents.base import AgentManager

        # Verify the class has the expected interface
        assert hasattr(AgentManager, "delegate")

    def test_agent_names_constant(self) -> None:
        """AGENT_NAMES lists all available agent names."""
        from llm_forge.chat.agents.base import AGENT_NAMES

        assert "data_agent" in AGENT_NAMES
        assert "config_agent" in AGENT_NAMES


class TestDelegateToolSchema:
    """Tests for the delegate_to_agent tool definition."""

    def test_delegate_tool_schema_exists(self) -> None:
        """DELEGATE_TOOL defines the tool schema for Claude."""
        from llm_forge.chat.agents.base import DELEGATE_TOOL

        assert DELEGATE_TOOL["name"] == "delegate_to_agent"
        assert "agent" in DELEGATE_TOOL["input_schema"]["properties"]
        assert "task" in DELEGATE_TOOL["input_schema"]["properties"]

    def test_delegate_tool_agent_enum(self) -> None:
        """DELEGATE_TOOL has agent names in enum."""
        from llm_forge.chat.agents.base import DELEGATE_TOOL

        agent_enum = DELEGATE_TOOL["input_schema"]["properties"]["agent"]["enum"]
        assert "data_agent" in agent_enum
        assert "config_agent" in agent_enum
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_chat/test_agents/test_base_agent.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create the agents package and base.py**

Create `src/llm_forge/chat/agents/__init__.py`:

```python
"""Multi-agent system for llm-forge.

Each agent is a Google ADK Agent backed by Gemini, managed by AgentManager.
The Claude orchestrator delegates tasks via the delegate_to_agent tool.
"""

from llm_forge.chat.agents.base import AGENT_NAMES, DELEGATE_TOOL, AgentManager

__all__ = ["AgentManager", "AGENT_NAMES", "DELEGATE_TOOL"]
```

Create `src/llm_forge/chat/agents/base.py`:

```python
"""Base agent infrastructure wrapping Google ADK.

AgentManager initializes all sub-agents and provides a unified
``delegate(agent_name, task, context)`` interface for the orchestrator.

IMPORTANT: Google ADK is fully async. We bridge sync/async with
asyncio.run() in the delegate() method so the rest of llm-forge
(which is synchronous) doesn't need to change.

Verified import paths (from Context7, google-adk v1.x):
  from google.adk.agents import Agent
  from google.adk.sessions import InMemorySessionService
  from google.genai import types  # types.Content, types.Part
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# All available agent names — extend as new agents are added
AGENT_NAMES: list[str] = [
    "data_agent",
    "config_agent",
    # Phase 2: "training_agent", "eval_agent"
    # Phase 3: "export_agent", "research_agent"
]

# Tool schema for Claude orchestrator to invoke sub-agents
DELEGATE_TOOL: dict[str, Any] = {
    "name": "delegate_to_agent",
    "description": (
        "Delegate a task to a specialist agent. The agent will use its "
        "own tools to complete the task and return a structured result. "
        "Available agents:\n"
        "- data_agent: Scan datasets, detect format, validate quality, search HuggingFace\n"
        "- config_agent: Write YAML configs, validate, tune hyperparameters, estimate training\n"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "enum": AGENT_NAMES,
                "description": "Which specialist agent to invoke",
            },
            "task": {
                "type": "string",
                "description": "What the agent should do (natural language instruction)",
            },
            "context": {
                "type": "object",
                "description": (
                    "Optional context: file paths, previous results, user preferences. "
                    "The agent receives this as additional context with its task."
                ),
            },
        },
        "required": ["agent", "task"],
    },
}


class ADKRunner:
    """Wraps Google ADK InMemoryRunner with a sync `run(message) -> str` interface.

    Google ADK's runner is fully async. This wrapper bridges to sync using
    asyncio.run() so the rest of llm-forge doesn't need async propagation.

    Shared by all agent modules (data_agent, config_agent, etc.) to avoid
    duplicating the async bridging logic.
    """

    def __init__(self, agent: Any, api_key: str) -> None:
        import os
        os.environ.setdefault("GOOGLE_API_KEY", api_key)

        from google.adk.runners import InMemoryRunner

        self._runner = InMemoryRunner(agent=agent)
        self._user_id = "orchestrator"
        self._session_id = f"{agent.name}_session"

    def run(self, message: str) -> str:
        """Send a message to the agent and return its text response (sync)."""

        async def _run_async() -> list:
            return await self._runner.run_debug(
                message,
                user_id=self._user_id,
                session_id=self._session_id,
                quiet=True,
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                events = pool.submit(asyncio.run, _run_async()).result()
        else:
            events = asyncio.run(_run_async())

        # Extract text from events
        texts = []
        for event in events:
            if hasattr(event, "content") and event.content:
                if hasattr(event.content, "parts"):
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            texts.append(part.text)

        return "\n".join(texts) if texts else "(No response from agent)"


class FallbackRunner:
    """Fallback runner when google-adk is not installed.

    Parses the user message for tool-call intent and dispatches directly.
    No LLM reasoning — just pattern matching and tool execution.

    Shared by all agent modules so they don't each reimplement this.
    """

    def __init__(self, name: str, dispatch_fn: Callable, system_prompt: str) -> None:
        self._name = name
        self._dispatch = dispatch_fn
        self._prompt = system_prompt

    def run(self, message: str) -> str:
        """Simple keyword-based tool dispatch."""
        msg_lower = message.lower()

        if any(kw in msg_lower for kw in ["scan", "analyze", "check data", "look at"]):
            for word in message.split():
                if "/" in word or "." in word:
                    return self._dispatch("scan_data", {"path": word.strip("'\"")})

        if any(kw in msg_lower for kw in ["hardware", "gpu", "cpu", "vram"]):
            return self._dispatch("detect_hardware", {})

        if any(kw in msg_lower for kw in ["search", "find", "huggingface", "hf"]):
            query = message.split("for")[-1].strip() if "for" in message else message
            return self._dispatch("search_huggingface", {"query": query, "search_type": "model"})

        if any(kw in msg_lower for kw in ["config", "yaml", "write", "generate"]):
            return self._dispatch("list_configs", {})

        if any(kw in msg_lower for kw in ["estimate", "time", "memory", "vram"]):
            return self._dispatch("estimate_training", {"model_name": "SmolLM2-135M", "mode": "lora", "num_samples": 1000})

        if any(kw in msg_lower for kw in ["validate", "check config"]):
            for word in message.split():
                if word.endswith(".yaml") or word.endswith(".yml"):
                    return self._dispatch("validate_config", {"config_path": word.strip("'\"")})

        return json.dumps({
            "status": "info",
            "agent": self._name,
            "message": f"Agent received: {message}. Please be more specific about what you need.",
        })


class AgentManager:
    """Manages Google ADK agent lifecycle and sessions.

    Each sub-agent is a Google ADK Agent with:
    - A specialized system prompt
    - A set of tools it can call
    - Its own session state (via InMemoryRunner)

    The orchestrator calls ``delegate()`` to send a task to a sub-agent
    and receive a structured response. The delegate() method is synchronous
    (bridges async ADK via asyncio.run).
    """

    def __init__(self, gemini_api_key: str) -> None:
        if not gemini_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required for sub-agents. "
                "Get one at: https://aistudio.google.com/apikey"
            )
        self._api_key = gemini_api_key
        self._agents: dict[str, Any] = {}

    def _get_or_create_agent(self, agent_name: str) -> Any:
        """Get an existing agent or create it on first use."""
        if agent_name in self._agents:
            return self._agents[agent_name]

        if agent_name == "data_agent":
            from llm_forge.chat.agents.data_agent import create_data_agent

            agent = create_data_agent(self._api_key)
        elif agent_name == "config_agent":
            from llm_forge.chat.agents.config_agent import create_config_agent

            agent = create_config_agent(self._api_key)
        else:
            raise ValueError(
                f"Unknown agent: {agent_name}. Available: {AGENT_NAMES}"
            )

        self._agents[agent_name] = agent
        return agent

    def delegate(self, agent_name: str, task: str, context: dict | None = None) -> str:
        """Send a task to a sub-agent and return its response (sync).

        Args:
            agent_name: Which agent to invoke (must be in AGENT_NAMES).
            task: Natural language description of what the agent should do.
            context: Optional dict with file paths, previous results, etc.

        Returns:
            The agent's response as a string (may contain JSON or natural language).
        """
        if agent_name not in AGENT_NAMES:
            raise ValueError(
                f"Unknown agent: {agent_name}. Available: {AGENT_NAMES}"
            )

        agent_runner = self._get_or_create_agent(agent_name)

        # Build the message with context
        message = task
        if context:
            message += f"\n\nContext:\n```json\n{json.dumps(context, indent=2)}\n```"

        logger.info("Delegating to %s: %s", agent_name, task[:100])

        try:
            response = agent_runner.run(message)
            logger.info("Agent %s completed", agent_name)
            return response
        except Exception as e:
            logger.error("Agent %s failed: %s", agent_name, e)
            return json.dumps({
                "status": "error",
                "agent": agent_name,
                "error": str(e),
                "suggestion": "The agent encountered an error. Check your API key and try again.",
            })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_chat/test_agents/test_base_agent.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_forge/chat/agents/ tests/test_chat/test_agents/
git commit -m "feat: add BaseAgent wrapper and AgentManager for Google ADK"
```

---

## Task 5: Create Data Agent

**Files:**
- Create: `src/llm_forge/chat/agents/data_agent.py`
- Create: `tests/test_chat/test_agents/test_data_agent.py`
- Reference: `src/llm_forge/chat/agent_tools/data_tools.py`

- [ ] **Step 1: Write failing test for Data Agent**

Create `tests/test_chat/test_agents/test_data_agent.py`:

```python
"""Tests for the Data Agent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


class TestDataAgentCreation:
    """Tests for create_data_agent factory."""

    def test_create_data_agent_returns_runner(self) -> None:
        """create_data_agent returns a callable agent runner."""
        from llm_forge.chat.agents.data_agent import create_data_agent

        # This will fail without a valid API key, but we test the interface
        with pytest.raises(Exception):
            # Empty key should raise
            create_data_agent("")

    def test_data_agent_system_prompt_exists(self) -> None:
        """DATA_AGENT_PROMPT is defined and non-empty."""
        from llm_forge.chat.agents.data_agent import DATA_AGENT_PROMPT

        assert len(DATA_AGENT_PROMPT) > 100
        assert "data" in DATA_AGENT_PROMPT.lower()

    def test_data_agent_has_tools(self) -> None:
        """Data agent tool list is defined."""
        from llm_forge.chat.agents.data_agent import DATA_AGENT_TOOLS

        tool_names = [t.__name__ if callable(t) else t["name"] for t in DATA_AGENT_TOOLS]
        assert "scan_data" in tool_names
        assert "detect_hardware" in tool_names


class TestDataAgentToolDispatch:
    """Tests for Data Agent tool dispatch (no API calls)."""

    def test_scan_data_via_tool(self, tmp_path) -> None:
        """Data agent's scan_data tool produces valid output."""
        from llm_forge.chat.agent_tools.data_tools import scan_data

        data_file = tmp_path / "test.jsonl"
        data_file.write_text('{"instruction": "test", "output": "ok"}\n')

        result = json.loads(scan_data(str(data_file)))
        assert result["status"] == "success"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_chat/test_agents/test_data_agent.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create data_agent.py**

Create `src/llm_forge/chat/agents/data_agent.py`:

```python
"""Data Agent — scans datasets, validates quality, searches HuggingFace.

This agent is backed by Gemini 2.5 Flash via Google ADK. It uses tools
from agent_tools/data_tools.py to analyze training data.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from llm_forge.chat.agent_tools.data_tools import (
    DATA_TOOL_DEFINITIONS,
    detect_hardware,
    download_model,
    scan_data,
    search_huggingface,
    show_model_info,
)

logger = logging.getLogger(__name__)

DATA_AGENT_PROMPT = """\
You are the Data Agent for llm-forge, an LLM training platform. Your job is to analyze
training datasets and help users understand their data before fine-tuning.

When analyzing data, always report:
- **Format**: JSONL, CSV, Parquet, HuggingFace dataset
- **Sample count**: Total number of training examples
- **Columns/fields**: What fields are present (instruction, input, output, etc.)
- **Data preview**: First 2-3 examples formatted clearly
- **Estimated token count**: Rough token estimate based on character count
- **Quality issues**: Empty fields, duplicates, encoding problems, very short/long examples

When the user asks about datasets, search HuggingFace Hub for relevant options.
When asked about hardware, detect and report the user's GPU/CPU/RAM.

Always return structured, actionable information. If data has quality issues,
recommend specific cleaning steps.
"""

# Tools available to this agent (callable references for Google ADK)
DATA_AGENT_TOOLS = [scan_data, detect_hardware, search_huggingface, download_model, show_model_info]


def _dispatch_tool(name: str, args: dict) -> str:
    """Dispatch a tool call to the right function."""
    tool_map = {
        "scan_data": scan_data,
        "detect_hardware": detect_hardware,
        "search_huggingface": search_huggingface,
        "download_model": download_model,
        "show_model_info": show_model_info,
    }
    fn = tool_map.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(**args)
    except Exception as e:
        return json.dumps({"error": str(e)})


def create_data_agent(api_key: str) -> Any:
    """Create a Data Agent backed by Gemini via Google ADK.

    Args:
        api_key: Google API key for Gemini access.

    Returns:
        An agent runner with a `run(message)` method (sync interface).
    """
    if not api_key:
        raise ValueError("Google API key is required for Data Agent")

    try:
        from google.adk.agents import Agent

        agent = Agent(
            name="data_agent",
            model="gemini-2.5-flash",
            description="Analyzes training datasets for format, quality, and readiness",
            instruction=DATA_AGENT_PROMPT,
            tools=DATA_AGENT_TOOLS,
        )

        from llm_forge.chat.agents.base import ADKRunner

        return ADKRunner(agent, api_key)

    except ImportError:
        logger.warning(
            "google-adk not installed. Data Agent will use tool-dispatch mode "
            "(no Gemini reasoning, just direct tool calls)."
        )
        from llm_forge.chat.agents.base import FallbackRunner

        return FallbackRunner("data_agent", _dispatch_tool, DATA_AGENT_PROMPT)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_chat/test_agents/test_data_agent.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_forge/chat/agents/data_agent.py tests/test_chat/test_agents/test_data_agent.py
git commit -m "feat: add Data Agent with scan_data, detect_hardware, search tools"
```

---

## Task 6: Create Config Agent

**Files:**
- Create: `src/llm_forge/chat/agents/config_agent.py`
- Create: `tests/test_chat/test_agents/test_config_agent.py`
- Reference: `src/llm_forge/chat/agent_tools/config_tools.py`

- [ ] **Step 1: Write failing test for Config Agent**

Create `tests/test_chat/test_agents/test_config_agent.py`:

```python
"""Tests for the Config Agent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


class TestConfigAgentCreation:
    """Tests for create_config_agent factory."""

    def test_create_config_agent_rejects_empty_key(self) -> None:
        """create_config_agent raises on empty API key."""
        from llm_forge.chat.agents.config_agent import create_config_agent

        with pytest.raises((ValueError, Exception)):
            create_config_agent("")

    def test_config_agent_system_prompt_exists(self) -> None:
        """CONFIG_AGENT_PROMPT is defined and mentions config/YAML."""
        from llm_forge.chat.agents.config_agent import CONFIG_AGENT_PROMPT

        assert len(CONFIG_AGENT_PROMPT) > 100
        assert "config" in CONFIG_AGENT_PROMPT.lower() or "yaml" in CONFIG_AGENT_PROMPT.lower()

    def test_config_agent_has_tools(self) -> None:
        """Config agent tool list includes write_config and validate_config."""
        from llm_forge.chat.agents.config_agent import CONFIG_AGENT_TOOLS

        tool_names = [t.__name__ if callable(t) else t.get("name", "") for t in CONFIG_AGENT_TOOLS]
        assert "write_config" in tool_names
        assert "validate_config" in tool_names


class TestConfigAgentToolDispatch:
    """Tests for Config Agent tool dispatch (no API calls)."""

    def test_write_config_via_tool(self, tmp_path) -> None:
        """Config agent's write_config tool creates a YAML file."""
        from llm_forge.chat.agent_tools.config_tools import write_config

        output = tmp_path / "generated.yaml"
        config = {"project_name": "test", "model": {"name": "SmolLM2-135M"}}
        result = json.loads(write_config(str(output), config))
        assert result["status"] == "success"

    def test_estimate_training_via_tool(self) -> None:
        """Config agent's estimate_training tool returns estimates."""
        from llm_forge.chat.agent_tools.config_tools import estimate_training

        result = json.loads(estimate_training("SmolLM2-135M", "lora", 500))
        assert isinstance(result, dict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_chat/test_agents/test_config_agent.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create config_agent.py**

Create `src/llm_forge/chat/agents/config_agent.py`:

```python
"""Config Agent — generates, validates, and optimizes YAML training configs.

This agent is backed by Gemini 2.5 Flash via Google ADK. It uses tools
from agent_tools/config_tools.py to manage training configurations.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from llm_forge.chat.agent_tools.config_tools import (
    CONFIG_TOOL_DEFINITIONS,
    estimate_training,
    list_configs,
    validate_config,
    write_config,
)

logger = logging.getLogger(__name__)

CONFIG_AGENT_PROMPT = """\
You are the Config Agent for llm-forge, an LLM training platform. Your job is to generate,
validate, and optimize YAML training configurations.

You have deep knowledge of the Pydantic v2 schema (LLMForgeConfig) and all sub-configs.

Key rules for generating configs:
- **LoRA r and alpha**: r must be a power of 2 (8, 16, 32, 64). alpha = 2*r is a good default.
- **Learning rate**: 1e-4 for LoRA, 2e-5 for full fine-tuning, 1e-5 for conservative.
- **Batch size**: Auto-detect based on VRAM: 8GB->bs=1, 16GB->bs=2, 24GB->bs=4, 80GB->bs=16.
- **max_seq_length**: At least 2048 for instruction tuning with multi-turn conversations.
- **assistant_only_loss**: true for chat/instruction data (masks system/user tokens from loss).
- **New features**: Disabled by default (enabled: false).
- **Chat template**: Must include {% generation %} markers for TRL assistant_only_loss.

When generating a config, always:
1. Ask about the model, data, and training mode if not specified
2. Detect hardware to set appropriate batch size
3. Validate the config against the schema before returning
4. Explain your hyperparameter choices

Config structure:
```yaml
project_name: "my_project"
model:
  name: "meta-llama/Llama-3.2-1B-Instruct"
  max_seq_length: 2048
data:
  train_path: "data/train.jsonl"
  format: "alpaca"
  cleaning:
    enabled: true
training:
  mode: "lora"
  num_train_epochs: 1
  per_device_train_batch_size: 4
  learning_rate: 1.0e-4
  lora:
    r: 16
    alpha: 32
    target_modules: ["q_proj", "v_proj"]
  assistant_only_loss: true
```
"""

# Tools available to this agent
CONFIG_AGENT_TOOLS = [write_config, validate_config, list_configs, estimate_training]


def _dispatch_tool(name: str, args: dict) -> str:
    """Dispatch a tool call to the right function."""
    tool_map = {
        "write_config": write_config,
        "validate_config": validate_config,
        "list_configs": list_configs,
        "estimate_training": estimate_training,
    }
    fn = tool_map.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(**args)
    except Exception as e:
        return json.dumps({"error": str(e)})


def create_config_agent(api_key: str) -> Any:
    """Create a Config Agent backed by Gemini via Google ADK.

    Args:
        api_key: Google API key for Gemini access.

    Returns:
        An agent runner with a `run(message)` method (sync interface).
    """
    if not api_key:
        raise ValueError("Google API key is required for Config Agent")

    try:
        from google.adk.agents import Agent

        agent = Agent(
            name="config_agent",
            model="gemini-2.5-flash",
            description="Generates, validates, and optimizes YAML training configurations",
            instruction=CONFIG_AGENT_PROMPT,
            tools=CONFIG_AGENT_TOOLS,
        )

        from llm_forge.chat.agents.base import ADKRunner

        return ADKRunner(agent, api_key)

    except ImportError:
        logger.warning(
            "google-adk not installed. Config Agent will use tool-dispatch mode."
        )
        from llm_forge.chat.agents.base import FallbackRunner

        return FallbackRunner("config_agent", _dispatch_tool, CONFIG_AGENT_PROMPT)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_chat/test_agents/test_config_agent.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_forge/chat/agents/config_agent.py tests/test_chat/test_agents/test_config_agent.py
git commit -m "feat: add Config Agent with write_config, validate, estimate tools"
```

---

## Task 7: Create OrchestratorEngine

**Files:**
- Create: `src/llm_forge/chat/orchestrator.py`
- Create: `tests/test_chat/test_orchestrator.py`
- Reference: `src/llm_forge/chat/engine.py` (do NOT modify yet)

- [ ] **Step 1: Write failing test for OrchestratorEngine**

Create `tests/test_chat/test_orchestrator.py`:

```python
"""Tests for the OrchestratorEngine (Claude + ADK sub-agents)."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest


class TestOrchestratorInit:
    """Tests for OrchestratorEngine initialization."""

    def test_requires_anthropic_key(self) -> None:
        """OrchestratorEngine raises without ANTHROPIC_API_KEY."""
        from llm_forge.chat.orchestrator import OrchestratorEngine

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises((ValueError, KeyError, Exception)):
                OrchestratorEngine(gemini_api_key="fake-key")

    def test_requires_gemini_key(self) -> None:
        """OrchestratorEngine raises without gemini_api_key."""
        from llm_forge.chat.orchestrator import OrchestratorEngine

        with pytest.raises((ValueError, TypeError)):
            OrchestratorEngine(gemini_api_key="")

    def test_has_send_method(self) -> None:
        """OrchestratorEngine exposes send() interface."""
        from llm_forge.chat.orchestrator import OrchestratorEngine

        assert callable(getattr(OrchestratorEngine, "send", None))

    def test_has_end_session_method(self) -> None:
        """OrchestratorEngine exposes end_session() interface."""
        from llm_forge.chat.orchestrator import OrchestratorEngine

        assert callable(getattr(OrchestratorEngine, "end_session", None))


class TestOrchestratorSystemPrompt:
    """Tests for the orchestrator's system prompt."""

    def test_system_prompt_mentions_agents(self) -> None:
        """ORCHESTRATOR_SYSTEM_PROMPT references available agents."""
        from llm_forge.chat.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

        assert "data_agent" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "config_agent" in ORCHESTRATOR_SYSTEM_PROMPT

    def test_system_prompt_mentions_delegation(self) -> None:
        """System prompt instructs Claude to delegate."""
        from llm_forge.chat.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

        assert "delegate" in ORCHESTRATOR_SYSTEM_PROMPT.lower()


class TestOrchestratorToolList:
    """Tests for the tools available to the orchestrator."""

    def test_delegate_tool_in_tools(self) -> None:
        """Orchestrator includes delegate_to_agent in its tool list."""
        from llm_forge.chat.orchestrator import ORCHESTRATOR_TOOLS

        tool_names = [t["name"] for t in ORCHESTRATOR_TOOLS]
        assert "delegate_to_agent" in tool_names

    def test_memory_tools_in_tools(self) -> None:
        """Orchestrator includes memory tools."""
        from llm_forge.chat.orchestrator import ORCHESTRATOR_TOOLS

        tool_names = [t["name"] for t in ORCHESTRATOR_TOOLS]
        assert "save_memory" in tool_names
        assert "recall_memory" in tool_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_chat/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create orchestrator.py**

Create `src/llm_forge/chat/orchestrator.py`:

```python
"""Claude-powered orchestrator that delegates to Google ADK sub-agents.

Replaces the single-LLM ChatEngine with a multi-agent architecture:
- Claude Sonnet 4.6 handles user interaction, planning, and synthesis
- Google ADK sub-agents (Gemini Flash) handle specialized tasks
- Memory system preserved from ChatEngine
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

from llm_forge.chat.agents.base import AGENT_NAMES, DELEGATE_TOOL, AgentManager
from llm_forge.chat.execution import (
    EXECUTION_TOOL_NAMES,
    PermissionSystem,
    execute_execution_tool,
)
from llm_forge.chat.memory import MemoryManager
from llm_forge.chat.tools import TOOLS as LEGACY_TOOLS

logger = logging.getLogger(__name__)

# Claude models available for orchestration
CLAUDE_MODELS = {
    "opus-4.6": "claude-opus-4-6",
    "sonnet-4.6": "claude-sonnet-4-6",
    "haiku-4.5": "claude-haiku-4-5",
    "sonnet-4.5": "claude-sonnet-4-5",
}
DEFAULT_MODEL = "sonnet-4.6"

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the orchestrator of llm-forge, a production-grade LLM training platform.

You have specialist agents you can delegate tasks to using the `delegate_to_agent` tool:

## Available Agents

- **data_agent**: Scans training datasets, detects format (JSONL/CSV/Parquet/HuggingFace), \
validates data quality, counts tokens, searches HuggingFace Hub for datasets and models, \
detects hardware (GPU/CPU/RAM).

- **config_agent**: Generates YAML training configs, validates against the Pydantic schema, \
tunes hyperparameters (LoRA rank, learning rate, batch size), estimates training time and \
VRAM usage, lists available config templates.

## Your Role

1. **Understand** what the user wants to accomplish
2. **Plan** the steps needed (which agents, in what order)
3. **Delegate** each step to the right specialist agent
4. **Synthesize** results into a clear, actionable response
5. **Handle errors** gracefully — suggest alternatives if an agent fails

## When to Delegate vs Answer Directly

**Delegate to agents when the task requires:**
- Scanning or analyzing files/data → data_agent
- Generating or validating configs → config_agent
- Hardware detection → data_agent
- Searching HuggingFace → data_agent
- Estimating training requirements → config_agent

**Answer directly when:**
- The user asks a general question about LLM training concepts
- The user needs clarification on terminology
- The user asks about your capabilities
- Simple conversational responses

## Memory

You have memory tools to save and recall information across sessions:
- `save_memory`: Save insights, decisions, user preferences
- `recall_memory`: Search past memories
- `get_project_state`: Current project status
- `get_session_history`: Past conversation summaries

## Important Rules

- Always delegate to specialists instead of trying to do everything yourself
- When chaining agents (e.g., data scan → config generation), pass results as context
- Report what each agent found in plain language — don't just dump JSON
- If an agent fails, explain what went wrong and suggest next steps
"""

# Memory tool definitions (same as ChatEngine uses)
_MEMORY_TOOLS = [
    {
        "name": "save_memory",
        "description": "Save a memory for future sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Category: user_preference, project_decision, training_lesson",
                },
                "content": {"type": "string", "description": "What to remember"},
                "relevance": {"type": "number", "description": "Importance 0-1 (default 1.0)"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall_memory",
        "description": "Search memories by keyword or topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_project_state",
        "description": "Get current project state (configs, models, data files).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_session_history",
        "description": "Get summaries of past conversation sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max sessions (default 5)"},
            },
        },
    },
]

# All tools the orchestrator can use
ORCHESTRATOR_TOOLS: list[dict] = [DELEGATE_TOOL] + _MEMORY_TOOLS


def _get_anthropic_client():
    """Get or create cached Anthropic client."""
    import anthropic

    return anthropic.Anthropic()


class OrchestratorEngine:
    """Claude-powered orchestrator that delegates to ADK sub-agents.

    Drop-in replacement for ChatEngine with the same send()/end_session()
    interface, but internally uses multi-agent delegation instead of
    a single LLM doing everything.
    """

    def __init__(
        self,
        project_dir: str | None = None,
        model_key: str = DEFAULT_MODEL,
        gemini_api_key: str | None = None,
    ) -> None:
        # Validate API keys
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError(
                "ANTHROPIC_API_KEY is required for the orchestrator. "
                "Get one at: https://console.anthropic.com/"
            )

        resolved_gemini_key = gemini_api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not resolved_gemini_key:
            raise ValueError(
                "GOOGLE_API_KEY is required for sub-agents. "
                "Get one at: https://aistudio.google.com/apikey"
            )

        self.model_key = model_key
        self.messages: list[dict] = []
        self.memory = MemoryManager(project_dir=project_dir or ".")
        self.permissions = PermissionSystem(auto_approve=True)
        self.agent_manager = AgentManager(gemini_api_key=resolved_gemini_key)
        self._client = _get_anthropic_client()

        # Build system prompt with memory context
        self.system = self._build_system_prompt()

        # UI callbacks for agent activity
        self.on_agent_start: Callable[[str, str], None] | None = None
        self.on_agent_end: Callable[[str, str], None] | None = None
        self.on_tool_start: Callable | None = None
        self.on_tool_end: Callable | None = None

    def _build_system_prompt(self) -> str:
        """Build system prompt with injected memory context."""
        context_block = self.memory.build_context_block()
        if context_block.strip():
            return f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\n---\n\n{context_block}"
        return ORCHESTRATOR_SYSTEM_PROMPT

    def send(
        self,
        user_input: str,
        on_text: Callable[[str], None] | None = None,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> str:
        """Send a user message through the Claude orchestrator.

        Claude decides which agents to invoke, in what order, and
        synthesizes their results into a coherent response.

        Args:
            user_input: The user's message.
            on_text: Callback for streaming text chunks.
            interrupt_check: Callback to check if user pressed Esc.

        Returns:
            The orchestrator's final response text.
        """
        self.messages.append({"role": "user", "content": user_input})

        # Check if context needs compaction
        if self.memory.needs_compaction(self.messages):
            self.messages = self.memory.compact_messages(
                self.messages, client=self._client
            )

        max_iterations = 15
        iteration = 0

        while True:
            # Call Claude with orchestrator tools
            model_id = CLAUDE_MODELS.get(self.model_key, CLAUDE_MODELS[DEFAULT_MODEL])
            response = self._client.messages.create(
                model=model_id,
                max_tokens=16000,
                system=self.system,
                messages=self.messages,
                tools=ORCHESTRATOR_TOOLS,
            )

            # Check for interruption
            if interrupt_check and interrupt_check():
                partial = self._extract_text(response) or "[interrupted]"
                self.messages.append({"role": "assistant", "content": partial})
                return partial

            # Parse response
            text, tool_calls = self._parse_response(response)

            if not tool_calls:
                self.messages.append({"role": "assistant", "content": text})
                return text

            # Guard against infinite loops
            iteration += 1
            if iteration >= max_iterations:
                partial = text or ""
                partial += "\n\n[Agent iteration limit reached]"
                self.messages.append({"role": "assistant", "content": partial})
                return partial

            # Handle tool calls
            self.messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for tc in tool_calls:
                result = self._execute_tool(tc["name"], tc["input"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result,
                })

            self.messages.append({"role": "user", "content": tool_results})

    def _execute_tool(self, name: str, input_data: dict) -> str:
        """Execute a tool call — either delegation or memory."""
        if self.on_tool_start:
            try:
                self.on_tool_start(name, input_data)
            except Exception:
                pass

        if name == "delegate_to_agent":
            result = self._handle_delegation(
                agent_name=input_data["agent"],
                task=input_data["task"],
                context=input_data.get("context"),
            )
        elif name == "save_memory":
            result = self.memory.save_memory(
                category=input_data.get("category", "general"),
                content=input_data["content"],
                relevance=input_data.get("relevance", 1.0),
            )
        elif name == "recall_memory":
            result = self.memory.recall_memory(
                query=input_data["query"],
                limit=input_data.get("limit", 10),
            )
        elif name == "get_project_state":
            result = json.dumps(self.memory.project_state, indent=2)
        elif name == "get_session_history":
            result = self.memory.get_session_history(
                limit=input_data.get("limit", 5)
            )
        elif name in EXECUTION_TOOL_NAMES:
            allowed, reason = self.permissions.check(name, input_data)
            if not allowed:
                result = json.dumps({"status": "blocked", "reason": reason})
            else:
                result = execute_execution_tool(name, input_data)
        else:
            result = json.dumps({"error": f"Unknown tool: {name}"})

        if self.on_tool_end:
            try:
                self.on_tool_end(name, input_data, result)
            except Exception:
                pass

        return result

    def _handle_delegation(
        self, agent_name: str, task: str, context: dict | None = None
    ) -> str:
        """Delegate a task to a sub-agent."""
        if self.on_agent_start:
            try:
                self.on_agent_start(agent_name, task)
            except Exception:
                pass

        result = self.agent_manager.delegate(agent_name, task, context)

        if self.on_agent_end:
            try:
                self.on_agent_end(agent_name, result)
            except Exception:
                pass

        return result

    def _parse_response(self, response) -> tuple[str, list[dict]]:
        """Parse Claude response into text + tool calls."""
        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return "\n".join(text_parts), tool_calls

    def _extract_text(self, response) -> str:
        """Extract just text from a response."""
        return "\n".join(
            b.text for b in response.content if b.type == "text"
        )

    def end_session(self) -> None:
        """End session gracefully, saving memory."""
        self.memory.end_session(self.messages, client=self._client)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_chat/test_orchestrator.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_forge/chat/orchestrator.py tests/test_chat/test_orchestrator.py
git commit -m "feat: add OrchestratorEngine — Claude orchestrator with agent delegation"
```

---

## Task 8: Wire OrchestratorEngine into the chat package and CLI

**Files:**
- Modify: `src/llm_forge/chat/__init__.py`
- Modify: `src/llm_forge/chat/ui.py`
- Modify: `src/llm_forge/cli.py`

- [ ] **Step 1: Read current files to understand what to modify**

Run: `cat src/llm_forge/chat/__init__.py`
Run: `head -50 src/llm_forge/chat/ui.py`
Run: `grep -n "launch_chat\|provider\|ChatEngine" src/llm_forge/chat/ui.py | head -20`

- [ ] **Step 2: Update chat/__init__.py to export OrchestratorEngine**

Modify `src/llm_forge/chat/__init__.py`:

```python
from llm_forge.chat.engine import ChatEngine
from llm_forge.chat.orchestrator import OrchestratorEngine

__all__ = ["ChatEngine", "OrchestratorEngine"]
```

- [ ] **Step 3: Update ui.py to use OrchestratorEngine when both keys present**

In `src/llm_forge/chat/ui.py`, modify the `launch_chat()` function to check for both
`ANTHROPIC_API_KEY` and `GOOGLE_API_KEY`. If both are present, use `OrchestratorEngine`.
Otherwise fall back to `ChatEngine`:

Add this logic to the `launch_chat()` function (exact line numbers depend on current file):

```python
import os

def launch_chat(provider=None):
    # Check if we can use the new multi-agent orchestrator
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_gemini = bool(os.environ.get("GOOGLE_API_KEY"))

    if has_anthropic and has_gemini and provider != "nvidia":
        from llm_forge.chat.orchestrator import OrchestratorEngine
        engine = OrchestratorEngine(
            project_dir=".",
            gemini_api_key=os.environ["GOOGLE_API_KEY"],
        )
    else:
        from llm_forge.chat.engine import ChatEngine
        engine = ChatEngine(provider=provider, project_dir=".")

    # ... rest of the chat loop (unchanged)
```

- [ ] **Step 4: Run full test suite to verify no regressions**

Run: `.venv/bin/python -m pytest tests/test_chat/ -v --tb=short -q`
Expected: All 894+ tests PASS (0 failures)

- [ ] **Step 5: Commit**

```bash
git add src/llm_forge/chat/__init__.py src/llm_forge/chat/ui.py
git commit -m "feat: wire OrchestratorEngine into chat package and CLI"
```

---

## Task 9: Full integration test and regression check

**Files:**
- No new files — run existing tests + manual verification

- [ ] **Step 1: Run the complete test suite**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short -q`
Expected: All tests pass, 0 failures, baseline maintained

- [ ] **Step 2: Verify config validation still works**

Run: `.venv/bin/python -c "from pathlib import Path; from llm_forge.config.validator import validate_config_file; configs = list(Path('configs').rglob('*.yaml')); [validate_config_file(c) for c in configs]; print(f'{len(configs)} configs validated')"`
Expected: All configs validate

- [ ] **Step 3: Verify imports work cleanly**

Run:
```bash
.venv/bin/python -c "
from llm_forge.chat import ChatEngine, OrchestratorEngine
from llm_forge.chat.agents import AgentManager, AGENT_NAMES, DELEGATE_TOOL
from llm_forge.chat.agent_tools.data_tools import scan_data, detect_hardware
from llm_forge.chat.agent_tools.config_tools import write_config, validate_config
print('All imports successful')
print(f'Agents: {AGENT_NAMES}')
print(f'Delegate tool: {DELEGATE_TOOL[\"name\"]}')
"
```
Expected: Prints all imports successful, lists agents

- [ ] **Step 4: Commit final state**

```bash
git add src/llm_forge/chat/agents/ src/llm_forge/chat/agent_tools/ src/llm_forge/chat/orchestrator.py src/llm_forge/chat/__init__.py src/llm_forge/chat/ui.py tests/test_chat/test_agents/ tests/test_chat/test_agent_tools/ tests/test_chat/test_orchestrator.py pyproject.toml
git commit -m "chore: Phase 1 complete — orchestrator + data agent + config agent"
```

**NOTE for Phase 4**: `src/llm_forge/chat/ui.py` imports from `llm_forge.chat.project_setup` (detect_project_type, scaffold_project). When `project_setup.py` is deleted in Phase 4, update ui.py to remove those imports.

---

## Summary

| Task | Description | New Files | Tests |
|------|-------------|-----------|-------|
| 1 | Add google-adk dependency | 0 | 0 |
| 2 | Extract data tools | 3 | 5 |
| 3 | Extract config tools | 2 | 6 |
| 4 | Create BaseAgent + AgentManager | 3 | 5 |
| 5 | Create Data Agent | 2 | 4 |
| 6 | Create Config Agent | 2 | 5 |
| 7 | Create OrchestratorEngine | 2 | 7 |
| 8 | Wire into CLI/UI | 0 (modify 3) | 0 |
| 9 | Integration test | 0 | 0 |
| **Total** | | **14 new files** | **32 new tests** |

**After Phase 1**: Users with both `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` get the new multi-agent orchestrator. Users with only `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` fall back to the existing `ChatEngine`. No regressions.
