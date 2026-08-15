"""Tests for the TUI status bar and tool rendering."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

textual = pytest.importorskip("textual")

from llm_forge.chat.tui import StatusBar, describe_hardware  # noqa: E402
from llm_forge.chat.ui import _format_tool_detail, _summarize_tool_result  # noqa: E402


class TestStatusBar:
    def test_shows_model_and_hardware(self) -> None:
        bar = StatusBar()
        bar.update_info(model="opus-5", hardware="Apple M4 Pro · 18/24 GB usable (mps)")
        rendered = bar.render()
        assert "opus-5" in rendered
        assert "18/24 GB usable" in rendered

    def test_idle_shows_key_hints(self) -> None:
        bar = StatusBar()
        assert "esc interrupt" in bar.render()

    def test_activity_replaces_hints(self) -> None:
        bar = StatusBar()
        bar.set_activity("Search the web: best 1B model")
        rendered = bar.render()
        assert "Search the web" in rendered
        assert "esc interrupt" not in rendered

    def test_activity_clears(self) -> None:
        bar = StatusBar()
        bar.set_activity("thinking")
        bar.set_activity("")
        assert "esc interrupt" in bar.render()

    def test_zero_memory_omits_the_segment(self) -> None:
        bar = StatusBar()
        bar.update_info(model="opus-5", memory=0, sessions=0)
        assert "insights" not in bar.render()


class TestDescribeHardware:
    def test_apple_reports_usable_not_installed(self) -> None:
        """The two differ, and the usable figure is the one that binds."""
        payload = json.dumps(
            {
                "backend": "mps",
                "gpu_name": "Apple M4 Pro",
                "unified_memory_gb": 24.0,
                "usable_memory_gb": 18.0,
            }
        )
        with patch("llm_forge.chat.tools._detect_hardware", return_value=payload):
            summary = describe_hardware()
        assert "18/24 GB usable" in summary
        assert "Apple M4 Pro" in summary

    def test_cuda_reports_vram(self) -> None:
        payload = json.dumps(
            {
                "backend": "cuda",
                "usable_memory_gb": 24.0,
                "gpus": [{"name": "NVIDIA GeForce RTX 4090"}],
            }
        )
        with patch("llm_forge.chat.tools._detect_hardware", return_value=payload):
            assert "RTX 4090" in describe_hardware()

    def test_cpu_says_so(self) -> None:
        payload = json.dumps({"backend": "cpu", "usable_memory_gb": 0})
        with patch("llm_forge.chat.tools._detect_hardware", return_value=payload):
            assert "CPU only" in describe_hardware()

    def test_detection_failure_degrades_to_empty(self) -> None:
        """A broken detector must not take the whole status bar down."""
        with patch("llm_forge.chat.tools._detect_hardware", side_effect=OSError("boom")):
            assert describe_hardware() == ""


class TestToolRendering:
    def test_web_search_has_a_label(self) -> None:
        """A tool with no label renders as a raw identifier."""
        detail = _format_tool_detail("web_search", {"query": "best 1B base model"})
        assert "Search the web" in detail
        assert "best 1B base model" in detail

    def test_long_query_is_truncated(self) -> None:
        detail = _format_tool_detail("web_search", {"query": "x" * 200})
        assert len(detail) < 100

    def test_hf_search_shows_kind(self) -> None:
        detail = _format_tool_detail(
            "search_huggingface", {"query": "gsm8k", "search_type": "datasets"}
        )
        assert "datasets" in detail

    def test_dataset_result_leads_with_ground_truth(self) -> None:
        result = json.dumps(
            {
                "type": "datasets",
                "results": [{"id": "openai/gsm8k", "ground_truth": {"verdict": "strong"}}],
            }
        )
        summary = _summarize_tool_result("search_huggingface", result)
        assert "openai/gsm8k" in summary
        assert "strong" in summary

    def test_model_result_leads_with_fit(self) -> None:
        result = json.dumps(
            {
                "type": "models",
                "results": [
                    {
                        "id": "meta-llama/Llama-3.2-1B",
                        "parameters_human": "1.2B",
                        "fit": {"recommended_method": "full"},
                    }
                ],
            }
        )
        summary = _summarize_tool_result("search_huggingface", result)
        assert "1.2B" in summary
        assert "full" in summary

    def test_hardware_summary_uses_usable_memory(self) -> None:
        result = json.dumps(
            {
                "gpu_type": "apple_mps",
                "ram_total_gb": 24.0,
                "usable_memory_gb": 18.0,
                "recommendation": {"mode": "lora"},
            }
        )
        summary = _summarize_tool_result("detect_hardware", result)
        assert "18 GB usable" in summary
        assert "24" not in summary

    def test_search_error_is_surfaced(self) -> None:
        result = json.dumps({"error": "hub unreachable"})
        summary = _summarize_tool_result("search_huggingface", result)
        assert "hub unreachable" in summary
