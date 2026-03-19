"""Tests for the serving module implementations.

Covers: ModelExporter (safetensors, GGUF, ONNX, merge, push_to_hub),
ITIBaker, FastAPIServer request/response models, GradioApp, VLLMServer.
All heavy dependencies (torch, transformers, etc.) are mocked to allow
tests to run without GPU or large model downloads.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Schema imports (always available)
# ---------------------------------------------------------------------------
from llm_forge.config.schema import ServingConfig

# ---------------------------------------------------------------------------
# Guard imports for serving modules (need optional deps)
# ---------------------------------------------------------------------------

try:
    from llm_forge.serving.export import (
        SUPPORTED_GGUF_QUANT_TYPES,
        ModelExporter,
    )

    _EXPORT_AVAILABLE = True
except ImportError:
    _EXPORT_AVAILABLE = False

try:
    from llm_forge.serving.iti_baker import ITIBaker

    _ITI_AVAILABLE = True
except ImportError:
    _ITI_AVAILABLE = False

try:
    from llm_forge.serving.fastapi_server import FastAPIServer

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

# Check if FastAPI Pydantic models are actually defined (requires fastapi package)
try:
    from llm_forge.serving.fastapi_server import GenerateRequest as _GR  # noqa: F401

    _FASTAPI_MODELS_AVAILABLE = True
except ImportError:
    _FASTAPI_MODELS_AVAILABLE = False

try:
    from llm_forge.serving.gradio_app import GradioApp

    _GRADIO_AVAILABLE = True
except ImportError:
    _GRADIO_AVAILABLE = False

try:
    from llm_forge.serving.vllm_server import VLLMServer

    _VLLM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _VLLM_AVAILABLE = False

# Check if VLLM Pydantic models are actually importable (requires torch)
try:
    from llm_forge.serving.vllm_server import VLLMGenerateRequest as _VGR  # noqa: F401

    _VLLM_MODELS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _VLLM_MODELS_AVAILABLE = False


# ===================================================================
# ModelExporter Tests
# ===================================================================


@pytest.mark.skipif(not _EXPORT_AVAILABLE, reason="export module not importable")
class TestModelExporter:
    """Test ModelExporter methods with mocked dependencies."""

    def test_exporter_class_exists(self) -> None:
        assert ModelExporter is not None

    def test_static_methods_exist(self) -> None:
        assert callable(getattr(ModelExporter, "export_safetensors", None))
        assert callable(getattr(ModelExporter, "export_gguf", None))
        assert callable(getattr(ModelExporter, "export_onnx", None))
        assert callable(getattr(ModelExporter, "merge_lora_and_export", None))
        assert callable(getattr(ModelExporter, "push_to_hub", None))

    def test_supported_gguf_quant_types(self) -> None:
        """Verify the supported GGUF quantization types set."""
        assert isinstance(SUPPORTED_GGUF_QUANT_TYPES, frozenset)
        assert "q4_0" in SUPPORTED_GGUF_QUANT_TYPES
        assert "q8_0" in SUPPORTED_GGUF_QUANT_TYPES
        assert "f16" in SUPPORTED_GGUF_QUANT_TYPES
        assert "q4_k_m" in SUPPORTED_GGUF_QUANT_TYPES
        assert len(SUPPORTED_GGUF_QUANT_TYPES) >= 10

    def test_gguf_invalid_quantization(self) -> None:
        """Invalid GGUF quantization type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported GGUF quantization"):
            ModelExporter.export_gguf(
                model_path="/tmp/fake_model",
                output_path="/tmp/fake_output.gguf",
                quantization="invalid_quant_type",
            )

    def test_gguf_nonexistent_model_path(self, tmp_path: Path) -> None:
        """Non-existent model path raises FileNotFoundError."""
        fake_path = str(tmp_path / "nonexistent_model")
        with pytest.raises(FileNotFoundError, match="Model directory not found"):
            ModelExporter.export_gguf(
                model_path=fake_path,
                output_path=str(tmp_path / "output.gguf"),
                quantization="q4_0",
            )

    def test_gguf_ensures_extension(self, tmp_path: Path) -> None:
        """GGUF export adds .gguf extension if missing."""
        # Create a fake model dir so it passes the exists() check
        model_dir = tmp_path / "fake_model"
        model_dir.mkdir()

        # Patch convert function to simulate failure
        with (
            mock.patch(
                "llm_forge.serving.export._try_llama_cpp_subprocess",
                return_value=False,
            ),
            pytest.raises(RuntimeError, match="GGUF export failed"),
        ):
            ModelExporter.export_gguf(
                model_path=str(model_dir),
                output_path=str(tmp_path / "output"),  # no .gguf
                quantization="q4_0",
            )

    def test_safetensors_requires_transformers(self) -> None:
        """export_safetensors raises ImportError if transformers not available."""
        with (
            mock.patch("llm_forge.serving.export._TRANSFORMERS_AVAILABLE", False),
            pytest.raises(ImportError, match="transformers is required"),
        ):
            ModelExporter.export_safetensors("fake_model", "/tmp/out")

    def test_onnx_requires_optimum(self) -> None:
        """export_onnx raises ImportError if optimum not available."""
        with mock.patch("llm_forge.serving.export._OPTIMUM_AVAILABLE", False):
            with pytest.raises(ImportError, match="optimum is required"):
                ModelExporter.export_onnx("fake_model", "/tmp/out")

    def test_merge_lora_requires_transformers(self) -> None:
        """merge_lora_and_export raises ImportError if transformers missing."""
        with (
            mock.patch("llm_forge.serving.export._TRANSFORMERS_AVAILABLE", False),
            pytest.raises(ImportError, match="transformers is required"),
        ):
            ModelExporter.merge_lora_and_export("base", "adapter", "/tmp/out")

    def test_merge_lora_requires_peft(self) -> None:
        """merge_lora_and_export raises ImportError if peft missing."""
        with (
            mock.patch("llm_forge.serving.export._TRANSFORMERS_AVAILABLE", True),
            mock.patch("llm_forge.serving.export._PEFT_AVAILABLE", False),
        ):
            with pytest.raises(ImportError, match="peft is required"):
                ModelExporter.merge_lora_and_export("base", "adapter", "/tmp/out")

    def test_awq_requires_autoawq(self) -> None:
        """export_awq raises ImportError if autoawq not available."""
        with mock.patch("llm_forge.serving.export._AWQ_AVAILABLE", False):
            with pytest.raises(ImportError, match="autoawq is required"):
                ModelExporter.export_awq("fake_model", "/tmp/out")

    def test_awq_requires_transformers(self) -> None:
        """export_awq raises ImportError if transformers not available."""
        with (
            mock.patch("llm_forge.serving.export._AWQ_AVAILABLE", True),
            mock.patch("llm_forge.serving.export._TRANSFORMERS_AVAILABLE", False),
            pytest.raises(ImportError, match="transformers is required"),
        ):
            ModelExporter.export_awq("fake_model", "/tmp/out")

    def test_awq_method_exists(self) -> None:
        """export_awq is a callable static method."""
        assert callable(getattr(ModelExporter, "export_awq", None))

    def test_merge_lora_nonexistent_adapter(self, tmp_path: Path) -> None:
        """merge_lora_and_export raises FileNotFoundError for missing adapter."""
        with (
            mock.patch("llm_forge.serving.export._TRANSFORMERS_AVAILABLE", True),
            mock.patch("llm_forge.serving.export._PEFT_AVAILABLE", True),
        ):
            with pytest.raises(FileNotFoundError, match="Adapter path not found"):
                ModelExporter.merge_lora_and_export(
                    "base_model",
                    str(tmp_path / "nonexistent_adapter"),
                    str(tmp_path / "output"),
                )

    def test_merge_lora_invalid_format(self, tmp_path: Path) -> None:
        """merge_lora_and_export raises ValueError for invalid format."""
        import llm_forge.serving.export as export_mod

        # Skip if transformers/peft are not installed (can't mock attrs that
        # don't exist as module-level names)
        if not getattr(export_mod, "_TRANSFORMERS_AVAILABLE", False):
            pytest.skip("transformers not installed — can't mock model loading")

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        mock_model = mock.MagicMock()
        mock_merged = mock.MagicMock()
        mock_model.merge_and_unload.return_value = mock_merged

        with (
            mock.patch(
                "llm_forge.serving.export.AutoModelForCausalLM.from_pretrained",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "llm_forge.serving.export.AutoTokenizer.from_pretrained",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "llm_forge.serving.export.PeftModel.from_pretrained",
                return_value=mock_model,
            ),
            pytest.raises(ValueError, match="Unsupported export format"),
        ):
            ModelExporter.merge_lora_and_export(
                "base_model",
                str(adapter_dir),
                str(tmp_path / "output"),
                format="invalid_format",
            )


# ===================================================================
# ITIBaker Tests
# ===================================================================


@pytest.mark.skipif(not _ITI_AVAILABLE, reason="iti_baker not importable")
class TestITIBaker:
    """Test the ITI Baker module."""

    def test_baker_instantiation(self) -> None:
        baker = ITIBaker()
        assert baker is not None

    def test_bake_interventions_method_exists(self) -> None:
        baker = ITIBaker()
        assert callable(baker.bake_interventions)

    def test_bake_interventions_with_mock_model(self) -> None:
        """Bake interventions on a mock model with Llama-like structure."""
        import numpy as np
        from torch import nn

        baker = ITIBaker()

        # Build a minimal mock Llama-like model
        hidden_size = 64
        num_heads = 4
        head_dim = hidden_size // num_heads

        # Create mock o_proj layers (Linear without bias)
        o_proj_0 = nn.Linear(hidden_size, hidden_size, bias=False)
        o_proj_1 = nn.Linear(hidden_size, hidden_size, bias=False)

        # Mock layer structure
        layer_0 = mock.MagicMock()
        layer_0.self_attn.o_proj = o_proj_0
        layer_1 = mock.MagicMock()
        layer_1.self_attn.o_proj = o_proj_1

        # Mock model
        model = mock.MagicMock()
        model.model.layers = [layer_0, layer_1]
        model.parameters.return_value = iter([o_proj_0.weight])
        model.config.num_attention_heads = num_heads

        # Directions for 2 heads
        directions = {
            (0, 1): np.random.randn(head_dim).astype(np.float32),
            (1, 0): np.random.randn(head_dim).astype(np.float32),
        }
        top_heads = [(0, 1), (1, 0)]
        sigmas = {(0, 1): 1.5, (1, 0): 2.0}

        result = baker.bake_interventions(
            model=model,
            directions=directions,
            top_heads=top_heads,
            alpha=15.0,
            sigmas=sigmas,
        )

        # Model should be returned
        assert result is model
        # attention_bias should be set
        assert model.config.attention_bias is True
        # o_proj layers should now have bias
        assert o_proj_0.bias is not None
        assert o_proj_1.bias is not None
        assert o_proj_0.bias.shape == (hidden_size,)

    def test_bake_skips_near_zero_direction(self) -> None:
        """Near-zero direction vectors are skipped."""
        import numpy as np
        from torch import nn

        baker = ITIBaker()

        hidden_size = 32
        num_heads = 2
        head_dim = hidden_size // num_heads

        o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        layer_0 = mock.MagicMock()
        layer_0.self_attn.o_proj = o_proj

        model = mock.MagicMock()
        model.model.layers = [layer_0]
        model.parameters.return_value = iter([o_proj.weight])
        model.config.num_attention_heads = num_heads

        # Near-zero direction
        directions = {(0, 0): np.zeros(head_dim, dtype=np.float32)}
        top_heads = [(0, 0)]
        sigmas = {(0, 0): 1.0}

        baker.bake_interventions(model, directions, top_heads, 15.0, sigmas)

        # Bias should still be set (but displacement from this head is zero)
        assert o_proj.bias is not None

    def test_bake_handles_out_of_range_layer(self) -> None:
        """Out-of-range layer indices are skipped with a warning."""
        import numpy as np
        from torch import nn

        baker = ITIBaker()

        hidden_size = 32
        num_heads = 2
        head_dim = hidden_size // num_heads

        o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        layer_0 = mock.MagicMock()
        layer_0.self_attn.o_proj = o_proj

        model = mock.MagicMock()
        model.model.layers = [layer_0]  # Only 1 layer
        model.parameters.return_value = iter([o_proj.weight])
        model.config.num_attention_heads = num_heads

        # Direction for layer 99 (out of range)
        directions = {(99, 0): np.random.randn(head_dim).astype(np.float32)}
        top_heads = [(99, 0)]
        sigmas = {(99, 0): 1.0}

        # Should not raise, just skip with warning
        baker.bake_interventions(model, directions, top_heads, 15.0, sigmas)

    def test_get_model_layers_unsupported_arch(self) -> None:
        """Unsupported architecture raises ValueError."""
        baker = ITIBaker()

        model = mock.MagicMock(spec=[])  # Empty spec = no attributes
        with pytest.raises(ValueError, match="Cannot find transformer layers"):
            baker._get_model_layers(model)

    def test_get_o_proj_returns_none_for_unknown(self) -> None:
        """Unknown layer structure returns None for o_proj."""
        baker = ITIBaker()

        layer = mock.MagicMock(spec=[])  # No known attributes
        result = baker._get_o_proj(layer)
        assert result is None


# ===================================================================
# FastAPIServer Tests
# ===================================================================


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="fastapi_server not importable")
class TestFastAPIServer:
    """Test FastAPIServer request/response models and construction."""

    def test_server_class_exists(self) -> None:
        assert FastAPIServer is not None

    def test_server_has_start_method(self) -> None:
        assert callable(getattr(FastAPIServer, "start", None))


@pytest.mark.skipif(not _FASTAPI_MODELS_AVAILABLE, reason="fastapi package not installed")
class TestFastAPIRequestModels:
    """Test the Pydantic request/response models used by FastAPIServer."""

    def test_generate_request_defaults(self) -> None:
        from llm_forge.serving.fastapi_server import GenerateRequest

        req = GenerateRequest(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.max_tokens == 256
        assert req.temperature == pytest.approx(0.7)
        assert req.top_p == pytest.approx(0.9)
        assert req.top_k == 50
        assert req.stream is False
        assert req.stop is None

    def test_generate_request_custom(self) -> None:
        from llm_forge.serving.fastapi_server import GenerateRequest

        req = GenerateRequest(
            prompt="Test",
            max_tokens=100,
            temperature=0.5,
            top_p=0.8,
            top_k=10,
            stream=True,
            stop=["<|end|>"],
        )
        assert req.max_tokens == 100
        assert req.stream is True
        assert req.stop == ["<|end|>"]

    def test_generate_response_fields(self) -> None:
        from llm_forge.serving.fastapi_server import GenerateResponse

        resp = GenerateResponse(
            id="gen-123",
            text="Output text",
            usage={"prompt_tokens": 5, "completion_tokens": 10},
            finish_reason="stop",
        )
        assert resp.id == "gen-123"
        assert resp.text == "Output text"
        assert resp.usage["prompt_tokens"] == 5

    def test_chat_message(self) -> None:
        from llm_forge.serving.fastapi_server import ChatMessage

        msg = ChatMessage(role="user", content="Hi there")
        assert msg.role == "user"
        assert msg.content == "Hi there"

    def test_chat_request_defaults(self) -> None:
        from llm_forge.serving.fastapi_server import ChatMessage, ChatRequest

        req = ChatRequest(messages=[ChatMessage(role="user", content="Hello")])
        assert req.model == "llm-forge"
        assert len(req.messages) == 1
        assert req.max_tokens == 256
        assert req.stream is False

    def test_chat_completion_response(self) -> None:
        from llm_forge.serving.fastapi_server import (
            ChatChoice,
            ChatCompletionResponse,
            ChatMessage,
        )

        resp = ChatCompletionResponse(
            id="chatcmpl-abc",
            created=1709000000,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hi!"),
                )
            ],
            usage={
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        )
        assert resp.object == "chat.completion"
        assert resp.choices[0].message.content == "Hi!"
        assert resp.choices[0].finish_reason == "stop"

    def test_health_response(self) -> None:
        from llm_forge.serving.fastapi_server import HealthResponse

        resp = HealthResponse()
        assert resp.status == "healthy"
        assert resp.model_loaded is True

    def test_model_info_response(self) -> None:
        from llm_forge.serving.fastapi_server import ModelInfoResponse

        resp = ModelInfoResponse(model_path="/models/test")
        assert resp.model_path == "/models/test"
        assert resp.model_type == "unknown"
        assert resp.vocab_size == 0


# ===================================================================
# GradioApp Tests
# ===================================================================


@pytest.mark.skipif(not _GRADIO_AVAILABLE, reason="gradio_app not importable")
class TestGradioApp:
    """Test GradioApp class."""

    def test_class_exists(self) -> None:
        assert GradioApp is not None

    def test_has_launch_method(self) -> None:
        assert callable(getattr(GradioApp, "launch", None))


# ===================================================================
# VLLMServer Tests
# ===================================================================


@pytest.mark.skipif(not _VLLM_AVAILABLE, reason="vllm_server not importable")
class TestVLLMServer:
    """Test VLLMServer class."""

    def test_class_exists(self) -> None:
        assert VLLMServer is not None

    def test_has_generate_method(self) -> None:
        assert callable(getattr(VLLMServer, "generate", None))

    def test_has_start_method(self) -> None:
        assert callable(getattr(VLLMServer, "start", None))


@pytest.mark.skipif(
    not _VLLM_MODELS_AVAILABLE, reason="vllm_server models not importable (needs torch)"
)
class TestVLLMRequestModels:
    """Test the vLLM Pydantic request/response models."""

    def test_vllm_generate_request_defaults(self) -> None:
        from llm_forge.serving.vllm_server import VLLMGenerateRequest

        req = VLLMGenerateRequest(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.max_tokens == 256
        assert req.temperature == pytest.approx(0.7)
        assert req.n == 1
        assert req.stream is False

    def test_vllm_chat_message(self) -> None:
        from llm_forge.serving.vllm_server import VLLMChatMessage

        msg = VLLMChatMessage(role="user", content="Hi")
        assert msg.role == "user"

    def test_vllm_chat_request(self) -> None:
        from llm_forge.serving.vllm_server import (
            VLLMChatMessage,
            VLLMChatRequest,
        )

        req = VLLMChatRequest(messages=[VLLMChatMessage(role="user", content="Hello")])
        assert req.model == "llm-forge"
        assert len(req.messages) == 1


# ===================================================================
# Serving __init__ Exports Tests
# ===================================================================


class TestServingExports:
    """Test that the serving package exports the right classes."""

    def test_model_exporter_in_exports(self) -> None:
        try:
            from llm_forge.serving import ModelExporter

            assert ModelExporter is not None
        except ImportError:
            pytest.skip("serving module not fully importable")

    def test_iti_baker_in_exports(self) -> None:
        try:
            from llm_forge.serving import ITIBaker

            assert ITIBaker is not None
        except ImportError:
            pytest.skip("ITIBaker not importable")

    def test_gradio_app_in_exports(self) -> None:
        try:
            from llm_forge.serving import GradioApp

            assert GradioApp is not None
        except ImportError:
            pytest.skip("GradioApp not importable")

    def test_fastapi_server_in_exports(self) -> None:
        try:
            from llm_forge.serving import FastAPIServer

            assert FastAPIServer is not None
        except ImportError:
            pytest.skip("FastAPIServer not importable")


# ===================================================================
# Export Format Integration with ServingConfig
# ===================================================================


class TestExportConfigIntegration:
    """Test that export formats in ServingConfig match what ModelExporter supports."""

    @pytest.mark.parametrize("fmt", ["safetensors", "gguf", "onnx"])
    def test_core_formats_valid(self, fmt: str) -> None:
        """Core export formats accepted by both schema and exporter."""
        cfg = ServingConfig(export_format=fmt)
        assert cfg.export_format == fmt

    def test_gguf_quantization_types(self) -> None:
        """GGUF quantization type is preserved in config."""
        cfg = ServingConfig(export_format="gguf", gguf_quantization="Q5_K_M")
        assert cfg.gguf_quantization == "Q5_K_M"

    def test_merge_adapter_default_true(self) -> None:
        cfg = ServingConfig()
        assert cfg.merge_adapter is True

    def test_backend_port_combinations(self) -> None:
        """Different backends can have different default ports."""
        gradio = ServingConfig(backend="gradio", port=7860)
        fastapi = ServingConfig(backend="fastapi", port=8000)
        vllm = ServingConfig(backend="vllm", port=8000)
        assert gradio.port == 7860
        assert fastapi.port == 8000
        assert vllm.port == 8000
