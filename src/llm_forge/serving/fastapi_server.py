"""FastAPI REST server for model inference with OpenAI-compatible endpoints.

Provides ``/generate``, ``/chat``, ``/health``, and ``/model/info`` endpoints
with streaming support via Server-Sent Events.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from threading import Thread
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("serving.fastapi_server")

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        TextIteratorStreamer,
    )

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

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

try:
    import os as _os

    import sentry_sdk

    _sentry_dsn = _os.environ.get("SENTRY_DSN", "")
    if _sentry_dsn:
        sentry_sdk.init(
            dsn=_sentry_dsn,
            traces_sample_rate=0.2,
            send_default_pii=False,
            environment=_os.environ.get("LLM_FORGE_ENV", "production"),
        )
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Precision mapping
# ---------------------------------------------------------------------------

_DTYPE_MAP: dict[str, Any] = {}
if _TORCH_AVAILABLE:
    _DTYPE_MAP = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }


# ---------------------------------------------------------------------------
# Pydantic request/response models (only if FastAPI is available)
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:

    class GenerateRequest(PydanticBaseModel):
        """Request body for the ``/generate`` endpoint."""

        prompt: str = Field(..., description="Text prompt for generation.")
        max_tokens: int = Field(
            default=256, ge=1, le=8192, description="Maximum new tokens to generate."
        )
        temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature.")
        top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold.")
        top_k: int = Field(default=50, ge=0, le=200, description="Top-k sampling. 0 = disabled.")
        stream: bool = Field(default=False, description="Stream tokens via Server-Sent Events.")
        stop: list[str] | None = Field(default=None, description="Stop sequences.")

    class GenerateResponse(PydanticBaseModel):
        """Response body for the ``/generate`` endpoint."""

        id: str = Field(description="Unique response identifier.")
        text: str = Field(description="Generated text.")
        usage: dict[str, int] = Field(description="Token usage statistics.")
        finish_reason: str = Field(default="stop", description="Why generation stopped.")

    class ChatMessage(PydanticBaseModel):
        """A single chat message."""

        role: str = Field(..., description="Message role: system, user, or assistant.")
        content: str = Field(..., description="Message content.")

    class ChatRequest(PydanticBaseModel):
        """OpenAI-compatible chat completion request."""

        model: str = Field(default="llm-forge", description="Model identifier.")
        messages: list[ChatMessage] = Field(..., description="Conversation messages.")
        max_tokens: int = Field(default=256, ge=1, le=8192, description="Maximum new tokens.")
        temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature.")
        top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling.")
        stream: bool = Field(default=False, description="Stream via SSE.")
        stop: list[str] | None = Field(default=None, description="Stop sequences.")

    class ChatChoice(PydanticBaseModel):
        """A single choice in a chat completion response."""

        index: int = 0
        message: ChatMessage
        finish_reason: str = "stop"

    class ChatCompletionResponse(PydanticBaseModel):
        """OpenAI-compatible chat completion response."""

        id: str
        object: str = "chat.completion"
        created: int
        model: str = "llm-forge"
        choices: list[ChatChoice]
        usage: dict[str, int]

    class HealthResponse(PydanticBaseModel):
        """Response for the ``/health`` endpoint."""

        status: str = "healthy"
        model_loaded: bool = True

    class ModelInfoResponse(PydanticBaseModel):
        """Response for the ``/model/info`` endpoint."""

        model_path: str
        model_type: str = "unknown"
        parameters: str = "N/A"
        dtype: str = "N/A"
        device: str = "N/A"
        max_position_embeddings: int = 0
        vocab_size: int = 0


# ============================================================================
# FastAPIServer
# ============================================================================


class FastAPIServer:
    """REST API server for model inference.

    Exposes OpenAI-compatible ``/chat`` completions, a simple ``/generate``
    endpoint, ``/health`` checks, and ``/model/info`` metadata.

    Parameters
    ----------
    model_path : str
        Local directory or HuggingFace model identifier.
    config : object, optional
        An ``LLMForgeConfig`` instance.
    """

    def __init__(self, model_path: str, config: Any | None = None) -> None:
        if not _FASTAPI_AVAILABLE:
            raise ImportError(
                "fastapi is required for the REST server. Install with: pip install fastapi"
            )
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for inference. Install with: pip install transformers"
            )

        self.model_path = model_path
        self.config = config
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self._model_info: dict[str, Any] = {}

        self._load_model()
        self.app = self._create_app()

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        """Load model and tokenizer."""
        logger.info("Loading model for FastAPI serving: %s", self.model_path)

        torch_dtype = torch.float16 if _TORCH_AVAILABLE else None
        trust_remote_code = False
        attn_implementation: str | None = None

        if self.config is not None and hasattr(self.config, "model"):
            model_cfg = self.config.model
            torch_dtype = _DTYPE_MAP.get(str(model_cfg.torch_dtype), torch.float16)
            trust_remote_code = model_cfg.trust_remote_code
            attn_implementation = model_cfg.attn_implementation

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Model
        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "device_map": "auto",
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        except Exception:
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)

        self.model.eval()

        # Collect info
        self._model_info = self._collect_model_info()
        logger.info("Model loaded for FastAPI serving")

    def _collect_model_info(self) -> dict[str, Any]:
        """Gather model metadata."""
        info: dict[str, Any] = {"model_path": self.model_path}
        if self.model is not None:
            mc = self.model.config
            info["model_type"] = getattr(mc, "model_type", "unknown")
            info["vocab_size"] = getattr(mc, "vocab_size", 0)
            info["max_position_embeddings"] = getattr(mc, "max_position_embeddings", 0)
            total_params = sum(p.numel() for p in self.model.parameters())
            info["parameters"] = f"{total_params:,}"
            try:
                fp = next(self.model.parameters())
                info["device"] = str(fp.device)
                info["dtype"] = str(fp.dtype)
            except StopIteration:
                info["device"] = "N/A"
                info["dtype"] = "N/A"
        return info

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def _generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Synchronous generation returning the full text at once."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        prompt_length = input_ids.shape[1]

        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if top_k > 0:
                gen_kwargs["top_k"] = top_k

        with torch.inference_mode():
            output_ids = self.model.generate(**gen_kwargs)

        generated_ids = output_ids[0][prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Trim at stop sequences
        finish_reason = "stop"
        if stop:
            for seq in stop:
                idx = text.find(seq)
                if idx != -1:
                    text = text[:idx]
                    finish_reason = "stop"
                    break
            else:
                if len(generated_ids) >= max_tokens:
                    finish_reason = "length"
        elif len(generated_ids) >= max_tokens:
            finish_reason = "length"

        return {
            "text": text,
            "prompt_tokens": prompt_length,
            "completion_tokens": len(generated_ids),
            "finish_reason": finish_reason,
        }

    def _generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        """Streaming generation yielding token chunks."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if top_k > 0:
                gen_kwargs["top_k"] = top_k

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        accumulated = ""
        for chunk in streamer:
            accumulated += chunk
            # Check stop sequences
            if stop:
                should_stop = False
                for seq in stop:
                    idx = accumulated.find(seq)
                    if idx != -1:
                        # Yield only up to the stop sequence
                        final_chunk = accumulated[:idx]
                        if final_chunk:
                            yield final_chunk[len(accumulated) - len(chunk) :]
                        should_stop = True
                        break
                if should_stop:
                    break
            yield chunk

        thread.join()

    # ------------------------------------------------------------------ #
    # Chat helpers
    # ------------------------------------------------------------------ #

    def _build_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        """Convert chat messages to a model prompt string."""
        if (
            self.tokenizer is not None
            and hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        ):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback
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
    # FastAPI app construction
    # ------------------------------------------------------------------ #

    def _create_app(self) -> FastAPI:
        """Build the FastAPI application with all endpoints."""
        app = FastAPI(
            title="llm-forge Inference API",
            description="REST API for model inference with OpenAI-compatible endpoints.",
            version="0.1.0",
        )

        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        server = self

        # ---- GET /health ----
        @app.get("/health", response_model=HealthResponse)
        async def health_check() -> HealthResponse:
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                model_loaded=server.model is not None,
            )

        # ---- GET /model/info ----
        @app.get("/model/info", response_model=ModelInfoResponse)
        async def model_info() -> ModelInfoResponse:
            """Return model metadata."""
            info = server._model_info
            return ModelInfoResponse(
                model_path=info.get("model_path", server.model_path),
                model_type=info.get("model_type", "unknown"),
                parameters=info.get("parameters", "N/A"),
                dtype=info.get("dtype", "N/A"),
                device=info.get("device", "N/A"),
                max_position_embeddings=info.get("max_position_embeddings", 0),
                vocab_size=info.get("vocab_size", 0),
            )

        # ---- POST /generate ----
        @app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest) -> Any:
            """Generate text from a prompt."""
            if server.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            if request.stream:
                return StreamingResponse(
                    _stream_generate(request),
                    media_type="text/event-stream",
                )

            result = server._generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop,
            )

            return GenerateResponse(
                id=f"gen-{uuid.uuid4().hex[:12]}",
                text=result["text"],
                usage={
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                },
                finish_reason=result["finish_reason"],
            )

        async def _stream_generate(
            request: GenerateRequest,
        ) -> Iterator[str]:
            """SSE streaming for /generate."""
            response_id = f"gen-{uuid.uuid4().hex[:12]}"
            for chunk in server._generate_streaming(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop,
            ):
                data = json.dumps({"id": response_id, "text": chunk, "finish_reason": None})
                yield f"data: {data}\n\n"

            # Final message
            yield f"data: {json.dumps({'id': response_id, 'text': '', 'finish_reason': 'stop'})}\n\n"
            yield "data: [DONE]\n\n"

        # ---- POST /chat ----
        @app.post("/chat")
        async def chat_completions(request: ChatRequest) -> Any:
            """OpenAI-compatible chat completion endpoint."""
            if server.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            prompt = server._build_chat_prompt(messages)

            if request.stream:
                return StreamingResponse(
                    _stream_chat(request, prompt),
                    media_type="text/event-stream",
                )

            result = server._generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )

            response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            return ChatCompletionResponse(
                id=response_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=result["text"]),
                        finish_reason=result["finish_reason"],
                    )
                ],
                usage={
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                },
            )

        async def _stream_chat(
            request: ChatRequest,
            prompt: str,
        ) -> Iterator[str]:
            """SSE streaming for /chat (OpenAI-compatible format)."""
            response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            for chunk in server._generate_streaming(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            ):
                data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"

            # Final chunk
            final = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return app

    # ------------------------------------------------------------------ #
    # Server start
    # ------------------------------------------------------------------ #

    def start(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the FastAPI server using uvicorn.

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

        # Use config values as defaults
        if self.config is not None and hasattr(self.config, "serving"):
            serving_cfg = self.config.serving
            if host == "0.0.0.0":
                host = serving_cfg.host
            if port == 8000:
                port = serving_cfg.port

        logger.info("Starting FastAPI server on %s:%d", host, port)
        uvicorn.run(self.app, host=host, port=port, log_level="info")
