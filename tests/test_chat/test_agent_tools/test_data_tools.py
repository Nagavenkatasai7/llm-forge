"""Tests for extracted data tools."""
from __future__ import annotations
import json
from pathlib import Path
import pytest

class TestScanData:
    def test_scan_jsonl_file(self, tmp_path: Path) -> None:
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(
            '{"instruction": "Hi", "input": "", "output": "Hello"}\n'
            '{"instruction": "Bye", "input": "", "output": "Goodbye"}\n'
        )
        from llm_forge.chat.agent_tools.data_tools import scan_data
        result = json.loads(scan_data(str(data_file)))
        assert result["status"] in ("success", "ok")
        assert result.get("num_samples", result.get("sample_count", 0)) >= 2

    def test_scan_nonexistent_path(self) -> None:
        from llm_forge.chat.agent_tools.data_tools import scan_data
        result = json.loads(scan_data("/nonexistent/path.jsonl"))
        assert result["status"] in ("error", "not_found")

    def test_scan_csv_file(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("instruction,output\nHi,Hello\nBye,Goodbye\n")
        from llm_forge.chat.agent_tools.data_tools import scan_data
        result = json.loads(scan_data(str(csv_file)))
        assert result["status"] in ("success", "ok")

class TestDetectHardware:
    def test_returns_valid_json(self) -> None:
        from llm_forge.chat.agent_tools.data_tools import detect_hardware
        result = json.loads(detect_hardware())
        assert "os" in result or "platform" in result or "cpu" in result

class TestSearchHuggingface:
    def test_search_returns_json(self) -> None:
        from llm_forge.chat.agent_tools.data_tools import search_huggingface
        result = json.loads(search_huggingface("test", "model"))
        assert isinstance(result, (dict, list))
