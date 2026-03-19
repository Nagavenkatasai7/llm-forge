"""vLLM-based high-throughput model serving with OpenAI-compatible API.

Uses the vLLM engine for continuous batching, PagedAttention, and
high-throughput inference.  Falls back gracefully when vLLM is not installed.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("serving.vllm_server")

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

try:
    import uvicorn

    _UVICORN_AVAILABLE = True
except ImportError:
    _UVICORN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pydantic models (shared with the OpenAI-compatible API surface)
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:

    class VLLMGenerateRequest(PydanticBaseModel):
        """Request body for /generate."""

        prompt: str = Field(..., description="Text prompt.")
        max_tokens: int = Field(default=256, ge=1, le=8192)
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        top_p: float = Field(default=0.9, ge=0.0, le=1.0)
        top_k: int = Field(default=-1, description="-1 = disabled.")
        n: int = Field(default=1, ge=1, le=16)
        stream: bool = Field(default=False)
        stop: list[str] | None = None

    class VLLMChatMessage(PydanticBaseModel):
        role: str
        content: str

    class VLLMChatRequest(PydanticBaseModel):
        model: str = Field(default="llm-forge")
        messages: list[VLLMChatMessage] = Field(...)
        max_tokens: int = Field(default=256, ge=1, le=8192)
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        top_p: float = Field(default=0.9, ge=0.0, le=1.0)
        top_k: int = Field(default=-1)
        n: int = Field(default=1, ge=1, le=16)
        stream: bool = Field(default=False)
        stop: list[str] | None = None


# ============================================================================
# VLLMServer
# ============================================================================


class VLLMServer:
    """High-throughput model serving using vLLM.

    Wraps the vLLM engine and exposes an OpenAI-compatible REST API for
    completions and chat completions with continuous batching and
    PagedAttention for optimal GPU utilisation.

    Parameters
    ----------
    model_path : str
        Local directory path or HuggingFace model identifier.
    config : object, optional
        An ``LLMForgeConfig`` instance.
    tensor_parallel_size : int
        Number of GPUs for tensor parallelism.
    gpu_memory_utilization : float
        Fraction of GPU memory vLLM is allowed to use.
    max_model_len : int, optional
        Maximum context length.  If *None*, inferred from the model config.
    trust_remote_code : bool
        Whether to trust remote code in the model repo.
    dtype : str
        Data type for model weights (``"auto"``, ``"float16"``, ``"bfloat16"``).
    """

    def __init__(
        self,
        model_path: str,
        config: Any | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        trust_remote_code: bool = False,
        dtype: str = "auto",
    ) -> None:
        if not _VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is required for high-throughput serving but is not installed.\n"
                "Install with: pip install vllm\n\n"
                "Note: vLLM requires a CUDA-capable GPU (compute capability >= 7.0).\n"
                "For CPU-only environments, consider using the FastAPIServer or "
                "GradioApp backends instead."
            )
        if not _FASTAPI_AVAILABLE:
            raise ImportError(
                "fastapi is required for the vLLM server. Install with: pip install fastapi"
            )

        self.model_path = model_path
        self.config = config

        # Resolve settings from config
        if config is not None and hasattr(config, "model"):
            model_cfg = config.model
            trust_remote_code = model_cfg.trust_remote_code
            if max_model_len is None:
                max_model_len = model_cfg.max_seq_length
            dtype_str = str(model_cfg.torch_dtype)
            if dtype_str in ("bf16", "bfloat16"):
                dtype = "bfloat16"
            elif dtype_str in ("fp16", "float16"):
                dtype = "float16"

        if config is not None and hasattr(config, "distributed"):
            dist_cfg = config.distributed
            if dist_cfg.enabled and dist_cfg.num_gpus > 1:
                tensor_parallel_size = max(tensor_parallel_size, dist_cfg.num_gpus)

        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype

        self.engine: LLM | None = None
        self.app: FastAPI | None = None

        self._setup_engine()
        self.app = self._create_app()

    # ------------------------------------------------------------------ #
    # Engine setup
    # ------------------------------------------------------------------ #

    def _setup_engine(self) -> None:
        """Initialise the vLLM engine."""
        logger.info(
            "Initialising vLLM engine: model=%s, tp=%d, gpu_util=%.2f, dtype=%s",
            self.model_path,
            self.tensor_parallel_size,
            self.gpu_memory_utilization,
            self.dtype,
        )

        engine_kwargs: dict[str, Any] = {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
        }

        if self.max_model_len is not None:
            engine_kwargs["max_model_len"] = self.max_model_len

        self.engine = LLM(**engine_kwargs)
        logger.info("vLLM engine initialised successfully")

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        n: int = 1,
        stop: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run batch inference on a list of prompts.

        Parameters
        ----------
        prompts : list[str]
            Input prompts.
        max_tokens : int
            Maximum tokens to generate per prompt.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus sampling threshold.
        top_k : int
            Top-k sampling (-1 = disabled).
        n : int
            Number of completions per prompt.
        stop : list[str], optional
            Stop sequences.

        Returns
        -------
        list[dict]
            One result dict per prompt, each containing ``"text"``,
            ``"prompt_tokens"``, ``"completion_tokens"``, and ``"finish_reason"``.
        """
        if self.engine is None:
            raise RuntimeError("vLLM engine not initialised")

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=n,
            stop=stop,
        )

        outputs = self.engine.generate(prompts, sampling_params)

        results: list[dict[str, Any]] = []
        for output in outputs:
            for completion in output.outputs:
                results.append(
                    {
                        "text": completion.text,
                        "prompt_tokens": len(output.prompt_token_ids),
                        "completion_tokens": len(completion.token_ids),
                        "finish_reason": completion.finish_reason or "stop",
                    }
                )

        return results

    def _build_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        """Convert messages to a prompt string using the tokenizer's chat template."""
        tokenizer = self.engine.get_tokenizer() if self.engine is not None else None

        if (
            tokenizer is not None
            and hasattr(tokenizer, "chat_template")
            and tokenizer.chat_template is not None
        ):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback: simple concatenation
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # FastAPI app
    # ------------------------------------------------------------------ #

    def _create_app(self) -> FastAPI:
        """Build the FastAPI application."""
        app = FastAPI(
            title="llm-forge vLLM Server",
            description="High-throughput inference with vLLM and OpenAI-compatible API.",
            version="0.1.0",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        server = self

        @app.get("/health")
        async def health() -> dict[str, Any]:
            return {
                "status": "healthy",
                "engine": "vllm",
                "model_loaded": server.engine is not None,
            }

        @app.get("/model/info")
        async def model_info() -> dict[str, Any]:
            info: dict[str, Any] = {
                "model_path": server.model_path,
                "engine": "vllm",
                "tensor_parallel_size": server.tensor_parallel_size,
                "gpu_memory_utilization": server.gpu_memory_utilization,
                "dtype": server.dtype,
            }
            if server.max_model_len is not None:
                info["max_model_len"] = server.max_model_len
            return info

        @app.post("/generate")
        async def generate_endpoint(request: VLLMGenerateRequest) -> Any:
            if server.engine is None:
                raise HTTPException(status_code=503, detail="Engine not ready")

            results = server.generate(
                prompts=[request.prompt],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                n=request.n,
                stop=request.stop,
            )

            result = results[0]
            return {
                "id": f"gen-{uuid.uuid4().hex[:12]}",
                "text": result["text"],
                "usage": {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                },
                "finish_reason": result["finish_reason"],
            }

        @app.post("/chat")
        async def chat_completions(request: VLLMChatRequest) -> Any:
            if server.engine is None:
                raise HTTPException(status_code=503, detail="Engine not ready")

            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            prompt = server._build_chat_prompt(messages)

            results = server.generate(
                prompts=[prompt],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                n=request.n,
                stop=request.stop,
            )

            response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            choices = []
            for i, result in enumerate(results):
                choices.append(
                    {
                        "index": i,
                        "message": {
                            "role": "assistant",
                            "content": result["text"],
                        },
                        "finish_reason": result["finish_reason"],
                    }
                )

            total_prompt = results[0]["prompt_tokens"] if results else 0
            total_completion = sum(r["completion_tokens"] for r in results)

            return {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": choices,
                "usage": {
                    "prompt_tokens": total_prompt,
                    "completion_tokens": total_completion,
                    "total_tokens": total_prompt + total_completion,
                },
            }

        @app.post("/v1/completions")
        async def v1_completions(request: VLLMGenerateRequest) -> Any:
            """OpenAI /v1/completions compatible endpoint."""
            return await generate_endpoint(request)

        @app.post("/v1/chat/completions")
        async def v1_chat_completions(request: VLLMChatRequest) -> Any:
            """OpenAI /v1/chat/completions compatible endpoint."""
            return await chat_completions(request)

        return app

    # ------------------------------------------------------------------ #
    # Server start
    # ------------------------------------------------------------------ #

    def start(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the vLLM server.

        Parameters
        ----------
        host : str
            Network interface to bind to.
        port : int
            Port number.
        """
        if not _UVICORN_AVAILABLE:
            raise ImportError(
                "uvicorn is required to start the server. Install with: pip install uvicorn"
            )

        if self.config is not None and hasattr(self.config, "serving"):
            serving_cfg = self.config.serving
            if host == "0.0.0.0":
                host = serving_cfg.host
            if port == 8000:
                port = serving_cfg.port

        if self.app is None:
            raise RuntimeError("FastAPI app not created. Call _create_app() first.")

        logger.info("Starting vLLM server on %s:%d", host, port)
        uvicorn.run(self.app, host=host, port=port, log_level="info")
