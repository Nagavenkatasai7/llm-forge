"""Gradio-based chat UI for interactive model serving.

Provides a rich chat interface with streaming token generation,
adjustable generation parameters, optional RAG integration, and
a model information panel.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from threading import Thread
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("serving.gradio_app")

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
    import gradio as gr

    _GRADIO_AVAILABLE = True
except ImportError:
    _GRADIO_AVAILABLE = False

try:
    import os as _os

    import sentry_sdk

    _sentry_dsn = _os.environ.get("SENTRY_DSN", "")
    if _sentry_dsn:
        sentry_sdk.init(
            dsn=_sentry_dsn,
            traces_sample_rate=0.1,
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
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }


# ============================================================================
# GradioApp
# ============================================================================


class GradioApp:
    """Interactive Gradio chat interface for a trained LLM.

    Provides streaming token generation, adjustable sampling parameters,
    optional RAG context injection, and model metadata display.

    Parameters
    ----------
    model_path : str
        Local directory path or HuggingFace model identifier.
    config : object, optional
        An ``LLMForgeConfig`` instance (or any object exposing ``.model``,
        ``.serving``, ``.rag`` sub-configs).  When provided, model loading
        parameters and RAG settings are drawn from the config.
    """

    def __init__(self, model_path: str, config: Any | None = None) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for the Gradio app. "
                "Install with: pip install transformers"
            )

        self.model_path = model_path
        self.config = config
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self._rag_retriever: Any | None = None
        self._model_info: dict[str, Any] = {}

        self._load_model()
        self._setup_rag()

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        """Load model and tokenizer from *model_path*."""
        model_path = self.model_path
        logger.info("Loading model for Gradio serving: %s", model_path)

        # Determine dtype
        torch_dtype = torch.float16 if _TORCH_AVAILABLE else None
        trust_remote_code = False
        attn_implementation: str | None = None

        if self.config is not None and hasattr(self.config, "model"):
            model_cfg = self.config.model
            torch_dtype = _DTYPE_MAP.get(str(model_cfg.torch_dtype), torch.float16)
            trust_remote_code = model_cfg.trust_remote_code
            attn_implementation = model_cfg.attn_implementation

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "device_map": "auto",
        }

        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        except Exception:
            # Fallback: try without attn_implementation (e.g. flash_attention_2 not available)
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        self.model.eval()

        # Gather model info
        self._model_info = self._collect_model_info()
        logger.info("Model loaded successfully for Gradio serving")

    def _collect_model_info(self) -> dict[str, Any]:
        """Collect model metadata for the info panel."""
        info: dict[str, Any] = {"model_path": self.model_path}

        if self.model is not None:
            model_config = self.model.config
            info["model_type"] = getattr(model_config, "model_type", "unknown")
            info["hidden_size"] = getattr(model_config, "hidden_size", "N/A")
            info["num_hidden_layers"] = getattr(model_config, "num_hidden_layers", "N/A")
            info["num_attention_heads"] = getattr(model_config, "num_attention_heads", "N/A")
            info["vocab_size"] = getattr(model_config, "vocab_size", "N/A")
            info["max_position_embeddings"] = getattr(
                model_config, "max_position_embeddings", "N/A"
            )

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            info["total_parameters"] = f"{total_params:,}"
            info["total_parameters_billions"] = f"{total_params / 1e9:.2f}B"

            # Device and dtype
            try:
                first_param = next(self.model.parameters())
                info["device"] = str(first_param.device)
                info["dtype"] = str(first_param.dtype)
            except StopIteration:
                info["device"] = "N/A"
                info["dtype"] = "N/A"

        return info

    # ------------------------------------------------------------------ #
    # RAG setup
    # ------------------------------------------------------------------ #

    def _setup_rag(self) -> None:
        """Optionally initialise RAG retriever from config."""
        if self.config is None:
            return
        if not hasattr(self.config, "rag"):
            return
        rag_cfg = self.config.rag
        if not rag_cfg.enabled:
            return
        if rag_cfg.knowledge_base_path is None:
            return

        kb_path = Path(rag_cfg.knowledge_base_path)
        if not kb_path.exists():
            logger.warning("RAG knowledge base path does not exist: %s. RAG disabled.", kb_path)
            return

        try:
            from llm_forge.rag import RAGPipeline  # type: ignore[import-untyped]

            self._rag_retriever = RAGPipeline(self.config)
            self._rag_retriever.build_index(str(kb_path))
            logger.info("RAG retriever initialised from %s", kb_path)
        except (ImportError, Exception) as exc:
            logger.warning("Failed to initialise RAG retriever: %s", exc)
            self._rag_retriever = None

    # ------------------------------------------------------------------ #
    # Generation helpers
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self,
        message: str,
        history: list[dict[str, str]],
        system_prompt: str,
        use_rag: bool,
    ) -> str:
        """Build a full prompt from message, history, and optional RAG context."""
        # Retrieve RAG context if enabled
        rag_context = ""
        if use_rag and self._rag_retriever is not None:
            try:
                chunks = self._rag_retriever.retrieve(message)
                if chunks:
                    rag_context = "\n\n".join(
                        c if isinstance(c, str) else c.get("text", str(c)) for c in chunks
                    )
            except Exception as exc:
                logger.warning("RAG retrieval failed: %s", exc)

        # Build messages list for chat template
        messages: list[dict[str, str]] = []

        if system_prompt.strip():
            sys_content = system_prompt.strip()
            if rag_context:
                sys_content += (
                    "\n\nUse the following context to help answer the user's "
                    "question:\n\n" + rag_context
                )
            messages.append({"role": "system", "content": sys_content})
        elif rag_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Use the following context to help answer the user's "
                        "question:\n\n" + rag_context
                    ),
                }
            )

        # Append conversation history
        for turn in history:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

        # Append current user message
        messages.append({"role": "user", "content": message})

        # Try to use chat template, fall back to simple concatenation
        if (
            self.tokenizer is not None
            and hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        ):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                return prompt
            except Exception:
                pass

        # Fallback: simple text concatenation
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

    def _generate_streaming(
        self,
        message: str,
        history: list[dict[str, str]],
        system_prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        use_rag: bool,
    ) -> Generator[str, None, None]:
        """Generate response tokens in a streaming fashion."""
        if self.model is None or self.tokenizer is None:
            yield "Error: Model not loaded."
            return

        prompt = self._build_prompt(message, history, system_prompt, use_rag)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # Configure streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Build generation kwargs
        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "streamer": streamer,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
        else:
            # Greedy decoding
            gen_kwargs["do_sample"] = False

        # Run generation in a background thread
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # Yield tokens as they become available
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text

        thread.join()

    # ------------------------------------------------------------------ #
    # Launch
    # ------------------------------------------------------------------ #

    def launch(
        self,
        host: str = "0.0.0.0",
        port: int = 7860,
        share: bool = False,
    ) -> None:
        """Build and launch the Gradio interface.

        Parameters
        ----------
        host : str
            Network interface to bind to.
        port : int
            Port number for the web server.
        share : bool
            Whether to create a public Gradio share link.
        """
        if not _GRADIO_AVAILABLE:
            raise ImportError(
                "gradio is required for the chat UI. Install with: pip install gradio"
            )

        # Use config values as defaults if available
        if self.config is not None and hasattr(self.config, "serving"):
            serving_cfg = self.config.serving
            if host == "0.0.0.0":
                host = serving_cfg.host
            if port == 7860:
                port = serving_cfg.port

        app = self._build_interface()

        logger.info("Launching Gradio app on %s:%d (share=%s)", host, port, share)
        app.launch(
            server_name=host,
            server_port=port,
            share=share,
        )

    def _build_interface(self) -> gr.Blocks:
        """Construct the Gradio Blocks interface."""
        # Format model info for display
        info_lines = []
        for key, value in self._model_info.items():
            display_key = key.replace("_", " ").title()
            info_lines.append(f"**{display_key}:** {value}")
        model_info_md = "\n\n".join(info_lines) if info_lines else "No model info."

        rag_available = self._rag_retriever is not None

        with gr.Blocks(
            title="llm-forge Chat",
            theme=gr.themes.Soft(),
        ) as app:
            gr.Markdown("# llm-forge Chat Interface\nInteractive chat with your fine-tuned model.")

            with gr.Row():
                # ---- Main chat column ----
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Chat",
                        height=500,
                        type="messages",
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Message",
                            placeholder="Type your message here...",
                            lines=2,
                            scale=4,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")
                        stop_btn = gr.Button("Stop")

                # ---- Settings sidebar ----
                with gr.Column(scale=1):
                    gr.Markdown("### Generation Settings")

                    system_prompt = gr.Textbox(
                        label="System Prompt",
                        value="You are a helpful, harmless, and honest AI assistant.",
                        lines=3,
                    )

                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.05,
                        label="Temperature",
                        info="Higher = more creative, Lower = more focused",
                    )

                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p (Nucleus Sampling)",
                    )

                    top_k = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Top-k",
                        info="0 = disabled",
                    )

                    max_tokens = gr.Slider(
                        minimum=16,
                        maximum=4096,
                        value=512,
                        step=16,
                        label="Max New Tokens",
                    )

                    if rag_available:
                        use_rag = gr.Checkbox(
                            label="Enable RAG",
                            value=True,
                            info="Retrieve context from knowledge base",
                        )
                    else:
                        use_rag = gr.Checkbox(
                            label="Enable RAG",
                            value=False,
                            interactive=False,
                            info="No knowledge base configured",
                        )

                    with gr.Accordion("Model Information", open=False):
                        gr.Markdown(model_info_md)

            # ---- Event handlers ----

            def user_message(
                message: str,
                history: list[dict[str, str]],
            ) -> tuple[str, list[dict[str, str]]]:
                """Add user message to history and clear input."""
                if not message.strip():
                    return "", history
                history = history + [{"role": "user", "content": message}]
                return "", history

            def bot_response(
                history: list[dict[str, str]],
                sys_prompt: str,
                temp: float,
                tp: float,
                tk: int,
                mt: int,
                rag_toggle: bool,
            ) -> Generator[list[dict[str, str]], None, None]:
                """Stream the bot response token by token."""
                if not history:
                    return

                last_user_msg = history[-1]["content"]

                # Build history without the last user message for context
                prior_history = history[:-1]

                # Add empty assistant message placeholder
                history = history + [{"role": "assistant", "content": ""}]

                for partial_text in self._generate_streaming(
                    message=last_user_msg,
                    history=prior_history,
                    system_prompt=sys_prompt,
                    temperature=temp,
                    top_p=tp,
                    top_k=int(tk),
                    max_tokens=int(mt),
                    use_rag=rag_toggle,
                ):
                    history[-1]["content"] = partial_text
                    yield history

            # Wire up events
            submit_event = msg_input.submit(
                fn=user_message,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot],
                queue=False,
            ).then(
                fn=bot_response,
                inputs=[
                    chatbot,
                    system_prompt,
                    temperature,
                    top_p,
                    top_k,
                    max_tokens,
                    use_rag,
                ],
                outputs=chatbot,
            )

            click_event = send_btn.click(
                fn=user_message,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot],
                queue=False,
            ).then(
                fn=bot_response,
                inputs=[
                    chatbot,
                    system_prompt,
                    temperature,
                    top_p,
                    top_k,
                    max_tokens,
                    use_rag,
                ],
                outputs=chatbot,
            )

            stop_btn.click(
                fn=None,
                inputs=None,
                outputs=None,
                cancels=[submit_event, click_event],
            )

            clear_btn.click(
                fn=lambda: [],
                inputs=None,
                outputs=chatbot,
                queue=False,
            )

        return app
