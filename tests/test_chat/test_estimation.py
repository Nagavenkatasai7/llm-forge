"""Tests for the estimate_training tool."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from llm_forge.chat.tools import (
    _estimate_training,
    _parse_model_params,
    execute_tool,
)

# ---------------------------------------------------------------------------
# _parse_model_params unit tests
# ---------------------------------------------------------------------------


class TestParseModelParams:
    def test_parse_1b(self):
        assert _parse_model_params("meta-llama/Llama-3.2-1B") == 1.0

    def test_parse_3b(self):
        assert _parse_model_params("meta-llama/Llama-3.2-3B-Instruct") == 3.0

    def test_parse_7b(self):
        assert _parse_model_params("mistralai/Mistral-7B-v0.1") == 7.0

    def test_parse_13b(self):
        assert _parse_model_params("meta-llama/Llama-2-13b-chat-hf") == 13.0

    def test_parse_135m(self):
        assert _parse_model_params("HuggingFaceTB/SmolLM2-135M") == pytest.approx(0.135, abs=0.001)

    def test_parse_360m(self):
        assert _parse_model_params("HuggingFaceTB/SmolLM2-360M") == pytest.approx(0.360, abs=0.001)

    def test_parse_1_5b(self):
        assert _parse_model_params("Qwen/Qwen2-1.5B") == 1.5

    def test_parse_70b(self):
        assert _parse_model_params("meta-llama/Llama-3-70B") == 70.0

    def test_fallback_unknown(self):
        """If no size pattern is found, fallback to 1B."""
        assert _parse_model_params("some-custom-model") == 1.0


# ---------------------------------------------------------------------------
# Estimation logic tests
# ---------------------------------------------------------------------------


class TestEstimateTraining:
    def _result(self, **kwargs) -> dict:
        """Helper: call _estimate_training and parse the JSON result."""
        raw = _estimate_training(**kwargs)
        return json.loads(raw)

    def test_estimate_small_model(self):
        """A 135M model should return reasonable estimates."""
        with patch(
            "llm_forge.chat.tools._detect_available_vram",
            return_value=(16.0, "consumer_gpu"),
        ):
            result = self._result(
                model_name="HuggingFaceTB/SmolLM2-135M",
                mode="lora",
                num_samples=1000,
            )
        assert result["status"] == "ok"
        assert result["estimated_params_billion"] == pytest.approx(0.135, abs=0.001)
        # A 135M model with LoRA should need very little VRAM (< 4 GB)
        assert result["estimated_vram_gb"] < 4.0
        assert result["fits_in_memory"] is True
        assert result["steps_total"] > 0
        assert result["estimated_time_minutes"] > 0

    def test_estimate_large_model_warns(self):
        """A 7B model on 8 GB VRAM should warn about memory."""
        with patch(
            "llm_forge.chat.tools._detect_available_vram",
            return_value=(8.0, "consumer_gpu"),
        ):
            result = self._result(
                model_name="mistralai/Mistral-7B-v0.1",
                mode="lora",
                num_samples=5000,
            )
        assert result["status"] == "ok"
        assert result["fits_in_memory"] is False
        assert result["estimated_vram_gb"] > 8.0
        assert (
            "smaller model" in result["recommendation"].lower()
            or "qlora" in result["recommendation"].lower()
        )

    def test_estimate_time_calculation(self):
        """Verify step count and time math are correct."""
        with patch(
            "llm_forge.chat.tools._detect_available_vram",
            return_value=(24.0, "consumer_gpu"),
        ):
            result = self._result(
                model_name="meta-llama/Llama-3.2-1B",
                mode="lora",
                num_samples=1000,
                num_epochs=2,
                batch_size=4,
            )
        # steps = ceil(1000 * 2 / 4) = 500
        assert result["steps_total"] == 500
        # time = 500 steps * 0.3 s/step * 1.0 (1B model) = 150s = 2.5 min
        assert result["estimated_time_minutes"] == pytest.approx(2.5, abs=0.1)

    def test_estimate_qlora_uses_less_memory(self):
        """QLoRA should need less VRAM than LoRA for the same model."""
        with patch(
            "llm_forge.chat.tools._detect_available_vram",
            return_value=(24.0, "consumer_gpu"),
        ):
            lora_result = self._result(
                model_name="meta-llama/Llama-3.2-3B",
                mode="lora",
                num_samples=1000,
            )
            qlora_result = self._result(
                model_name="meta-llama/Llama-3.2-3B",
                mode="qlora",
                num_samples=1000,
            )
        assert qlora_result["estimated_vram_gb"] < lora_result["estimated_vram_gb"]

    def test_fits_in_memory_true(self):
        """A small model on a big GPU should fit."""
        with patch(
            "llm_forge.chat.tools._detect_available_vram",
            return_value=(80.0, "a100"),
        ):
            result = self._result(
                model_name="meta-llama/Llama-3.2-1B",
                mode="lora",
                num_samples=5000,
            )
        assert result["fits_in_memory"] is True
        assert result["available_vram_gb"] == 80.0

    def test_fits_in_memory_false(self):
        """A big model on a small GPU should not fit."""
        with patch(
            "llm_forge.chat.tools._detect_available_vram",
            return_value=(8.0, "consumer_gpu"),
        ):
            result = self._result(
                model_name="meta-llama/Llama-3-70B",
                mode="full",
                num_samples=100,
            )
        assert result["fits_in_memory"] is False
        assert result["estimated_vram_gb"] > 8.0

    def test_no_gpu_detected(self):
        """When no GPU is available, fits_in_memory is False with helpful message."""
        with patch(
            "llm_forge.chat.tools._detect_available_vram",
            return_value=(0.0, "cpu"),
        ):
            result = self._result(
                model_name="meta-llama/Llama-3.2-1B",
                mode="lora",
                num_samples=500,
            )
        assert result["fits_in_memory"] is False
        assert "no gpu" in result["recommendation"].lower()

    def test_breakdown_present(self):
        """The result should include a VRAM breakdown."""
        with patch(
            "llm_forge.chat.tools._detect_available_vram",
            return_value=(24.0, "consumer_gpu"),
        ):
            result = self._result(
                model_name="meta-llama/Llama-3.2-1B",
                mode="lora",
                num_samples=1000,
            )
        breakdown = result["breakdown"]
        assert "model_weights_gb" in breakdown
        assert "gradients_gb" in breakdown
        assert "optimizer_gb" in breakdown
        assert "activations_gb" in breakdown

    def test_execute_tool_dispatch(self):
        """estimate_training is reachable via execute_tool."""
        with patch(
            "llm_forge.chat.tools._detect_available_vram",
            return_value=(24.0, "consumer_gpu"),
        ):
            raw = execute_tool(
                "estimate_training",
                {
                    "model_name": "meta-llama/Llama-3.2-1B",
                    "mode": "lora",
                    "num_samples": 100,
                },
            )
        result = json.loads(raw)
        assert result["status"] == "ok"

    def test_full_mode_uses_more_memory(self):
        """Full fine-tuning should need more memory than LoRA."""
        with patch(
            "llm_forge.chat.tools._detect_available_vram",
            return_value=(80.0, "a100"),
        ):
            lora = self._result(
                model_name="meta-llama/Llama-3.2-1B",
                mode="lora",
                num_samples=1000,
            )
            full = self._result(
                model_name="meta-llama/Llama-3.2-1B",
                mode="full",
                num_samples=1000,
            )
        assert full["estimated_vram_gb"] > lora["estimated_vram_gb"]

    def test_default_epochs_and_batch(self):
        """When num_epochs and batch_size are not provided, defaults apply."""
        with patch(
            "llm_forge.chat.tools._detect_available_vram",
            return_value=(24.0, "consumer_gpu"),
        ):
            result = self._result(
                model_name="meta-llama/Llama-3.2-1B",
                mode="lora",
                num_samples=400,
            )
        # Default: epochs=1, batch_size=4 -> steps = ceil(400/4) = 100
        assert result["steps_total"] == 100
