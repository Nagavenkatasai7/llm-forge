"""Tests for the chat tools module.

Covers hardware detection, data scanning, config writing/validation,
config listing, training status, and the execute_tool dispatcher.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from llm_forge.chat.tools import execute_tool

# ===================================================================
# Hardware detection
# ===================================================================


class TestDetectHardware:
    """Test _detect_hardware() via execute_tool."""

    def test_detect_hardware_returns_json(self) -> None:
        """_detect_hardware() returns valid JSON with os and cpu keys."""
        result = json.loads(execute_tool("detect_hardware", {}))
        assert "os" in result
        assert "cpu" in result
        assert "python_version" in result
        # Should be a recognized OS
        assert result["os"] in ("Darwin", "Linux", "Windows")

    def test_detect_hardware_has_gpu_info(self) -> None:
        """_detect_hardware() includes gpu_type key."""
        result = json.loads(execute_tool("detect_hardware", {}))
        assert "gpu_type" in result


# ===================================================================
# Data scanning
# ===================================================================


class TestScanData:
    """Test _scan_data() with various inputs."""

    def test_scan_data_local_jsonl(self, tmp_path: Path) -> None:
        """_scan_data() on a real JSONL file returns format + preview."""
        jsonl_file = tmp_path / "train.jsonl"
        records = [
            {"instruction": "What is AI?", "output": "Artificial Intelligence"},
            {"instruction": "What is ML?", "output": "Machine Learning"},
            {"instruction": "What is DL?", "output": "Deep Learning"},
        ]
        jsonl_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        result = json.loads(execute_tool("scan_data", {"path": str(jsonl_file)}))
        assert result["status"] == "ok"
        assert result["source"] == "local_file"
        assert result["sample_count"] == 3
        assert result["detected_format"] == "alpaca"
        assert len(result["preview"]) <= 3

    def test_scan_data_not_found(self) -> None:
        """_scan_data() on nonexistent path returns error."""
        result = json.loads(execute_tool("scan_data", {"path": "/nonexistent/path/data.jsonl"}))
        assert result["status"] == "not_found"
        assert "error" in result

    def test_scan_data_directory(self, tmp_path: Path) -> None:
        """_scan_data() on a directory returns file count and size."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file1.txt").write_text("hello world")
        (data_dir / "file2.txt").write_text("goodbye world")

        result = json.loads(execute_tool("scan_data", {"path": str(data_dir)}))
        assert result["status"] == "ok"
        assert result["source"] == "local_directory"
        assert result["file_count"] == 2

    def test_scan_data_sharegpt_format(self, tmp_path: Path) -> None:
        """_scan_data() detects sharegpt format correctly."""
        jsonl_file = tmp_path / "sharegpt.jsonl"
        records = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi!"},
                ]
            }
        ]
        jsonl_file.write_text(json.dumps(records[0]) + "\n")
        result = json.loads(execute_tool("scan_data", {"path": str(jsonl_file)}))
        assert result["detected_format"] == "sharegpt"

    def test_scan_data_text_file(self, tmp_path: Path) -> None:
        """_scan_data() handles plain text files."""
        txt_file = tmp_path / "corpus.txt"
        txt_file.write_text("This is a training corpus with many words.")
        result = json.loads(execute_tool("scan_data", {"path": str(txt_file)}))
        assert result["status"] == "ok"
        assert result["detected_format"] == "completion"
        assert "word_count" in result


# ===================================================================
# Config writing
# ===================================================================


class TestWriteConfig:
    """Test _write_config() creates a valid YAML file."""

    def test_write_config_creates_file(self, tmp_path: Path) -> None:
        """_write_config() creates a YAML file at the specified path."""
        output_path = str(tmp_path / "generated_config.yaml")
        config = {
            "model": {"name": "Llama-3.2-1B"},
            "data": {"train_path": "data/train.jsonl"},
            "training": {"mode": "lora", "num_epochs": 1},
        }
        result = json.loads(
            execute_tool("write_config", {"output_path": output_path, "config": config})
        )
        assert result["status"] == "ok"
        assert Path(output_path).exists()

        # Verify the file is valid YAML with expected content
        with open(output_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["model"]["name"] == "Llama-3.2-1B"
        assert loaded["training"]["mode"] == "lora"

    def test_write_config_creates_parent_dirs(self, tmp_path: Path) -> None:
        """_write_config() creates parent directories if needed."""
        output_path = str(tmp_path / "nested" / "dir" / "config.yaml")
        config = {"model": {"name": "test"}}
        result = json.loads(
            execute_tool("write_config", {"output_path": output_path, "config": config})
        )
        assert result["status"] == "ok"
        assert Path(output_path).exists()


# ===================================================================
# Config validation
# ===================================================================


class TestValidateConfig:
    """Test _validate_config() with valid and invalid configs."""

    def test_validate_config_valid(self, tmp_path: Path) -> None:
        """_validate_config() on a valid config returns 'valid'."""
        config_path = tmp_path / "valid.yaml"
        config = {
            "model": {"name": "meta-llama/Llama-3.2-1B"},
            "data": {"train_path": "tatsu-lab/alpaca"},
        }
        config_path.write_text(yaml.dump(config, default_flow_style=False))

        result = json.loads(execute_tool("validate_config", {"config_path": str(config_path)}))
        assert result["status"] == "valid"

    def test_validate_config_invalid(self, tmp_path: Path) -> None:
        """_validate_config() on an empty file returns error."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        result = json.loads(execute_tool("validate_config", {"config_path": str(config_path)}))
        # Empty YAML returns None, which triggers an error
        assert result["status"] == "error"

    def test_validate_config_missing_required(self, tmp_path: Path) -> None:
        """_validate_config() on config missing required fields returns invalid."""
        config_path = tmp_path / "incomplete.yaml"
        # Missing model and data sections entirely
        config_path.write_text("training:\n  mode: lora\n")

        result = json.loads(execute_tool("validate_config", {"config_path": str(config_path)}))
        assert result["status"] == "invalid"
        assert "errors" in result


# ===================================================================
# List configs
# ===================================================================


class TestListConfigs:
    """Test _list_configs() finding config files."""

    def test_list_configs_finds_files(self, tmp_path: Path) -> None:
        """_list_configs() returns config files from the configs directory."""
        from unittest.mock import patch

        # Create a configs dir with known files
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        (configs_dir / "train.yaml").write_text("model:\n  name: test\n")
        (configs_dir / "eval.yaml").write_text("model:\n  name: test2\n")

        # Patch the internal _list_configs to look at our temp dir only
        from llm_forge.chat import tools as tools_module

        fake_file = tmp_path / "src" / "llm_forge" / "chat" / "tools.py"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.touch()

        with patch.object(tools_module, "__file__", str(fake_file)):
            result = json.loads(execute_tool("list_configs", {}))

        assert result["count"] == 2
        names = [c["name"] for c in result["configs"]]
        assert "eval.yaml" in names
        assert "train.yaml" in names


# ===================================================================
# Training status
# ===================================================================


class TestCheckTrainingStatus:
    """Test _check_training_status()."""

    def test_check_training_status_idle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns idle when no training running and no outputs dir."""
        monkeypatch.chdir(tmp_path)
        result = json.loads(execute_tool("check_training_status", {}))
        assert result["status"] == "idle"
        assert "No training detected" in result["message"]


# ===================================================================
# Execute tool dispatcher
# ===================================================================


class TestExecuteToolDispatcher:
    """Test the execute_tool() routing logic."""

    def test_execute_tool_unknown(self) -> None:
        """execute_tool('fake_tool', {}) returns error JSON."""
        result = json.loads(execute_tool("fake_tool", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_execute_tool_catches_exceptions(self) -> None:
        """execute_tool() catches exceptions and returns error JSON."""
        # scan_data requires a "path" key; omitting it should trigger KeyError
        result = json.loads(execute_tool("scan_data", {}))
        assert "error" in result
