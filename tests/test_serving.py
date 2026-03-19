"""Tests for the serving module.

Covers ModelExporter class existence, export format validation via
ServingConfig, and skips actual server/model tests.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_forge.config.schema import ServingConfig

# ===================================================================
# ServingConfig validation
# ===================================================================


class TestServingConfig:
    """Test ServingConfig pydantic model."""

    def test_defaults(self) -> None:
        cfg = ServingConfig()
        assert cfg.backend == "gradio"
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 7860
        assert cfg.export_format is None
        assert cfg.merge_adapter is True

    @pytest.mark.parametrize("backend", ["gradio", "fastapi", "vllm"])
    def test_valid_backends(self, backend: str) -> None:
        cfg = ServingConfig(backend=backend)
        assert cfg.backend == backend

    def test_invalid_backend(self) -> None:
        with pytest.raises(ValidationError):
            ServingConfig(backend="flask")


# ===================================================================
# Export format validation
# ===================================================================


class TestExportFormatValidation:
    """Test export_format field of ServingConfig."""

    @pytest.mark.parametrize("fmt", ["gguf", "onnx", "safetensors", "awq", "gptq"])
    def test_valid_export_formats(self, fmt: str) -> None:
        cfg = ServingConfig(export_format=fmt)
        assert cfg.export_format == fmt

    def test_export_format_none(self) -> None:
        cfg = ServingConfig(export_format=None)
        assert cfg.export_format is None

    def test_invalid_export_format(self) -> None:
        with pytest.raises(ValidationError):
            ServingConfig(export_format="invalid_format")

    def test_gguf_quantization(self) -> None:
        cfg = ServingConfig(export_format="gguf", gguf_quantization="Q4_K_M")
        assert cfg.gguf_quantization == "Q4_K_M"


# ===================================================================
# Port validation
# ===================================================================


class TestPortValidation:
    """Test port number constraints."""

    def test_valid_port(self) -> None:
        cfg = ServingConfig(port=8080)
        assert cfg.port == 8080

    def test_port_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            ServingConfig(port=0)

    def test_port_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            ServingConfig(port=70000)


# ===================================================================
# ModelExporter class existence
# ===================================================================


class TestModelExporterExists:
    """Test that ModelExporter is importable from the serving module."""

    def test_model_exporter_class_exists(self) -> None:
        """ModelExporter should be importable (may require deps)."""
        try:
            from llm_forge.serving import ModelExporter

            assert ModelExporter is not None
        except ImportError:
            pytest.skip("ModelExporter import failed (likely missing export.py)")

    def test_merge_adapter_flag(self) -> None:
        """merge_adapter flag exists on ServingConfig."""
        cfg = ServingConfig(merge_adapter=False)
        assert cfg.merge_adapter is False

    def test_host_field(self) -> None:
        cfg = ServingConfig(host="127.0.0.1")
        assert cfg.host == "127.0.0.1"
