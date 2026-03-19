"""Model export utilities: safetensors, GGUF, ONNX, LoRA merging, and Hub upload.

Provides a unified ``ModelExporter`` class for converting trained models
to various deployment formats and pushing them to HuggingFace Hub.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from llm_forge.utils.logging import get_logger

logger = get_logger("serving.export")

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
    )

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from safetensors.torch import save_file as safetensors_save_file

    _SAFETENSORS_AVAILABLE = True
except ImportError:
    _SAFETENSORS_AVAILABLE = False

try:
    from peft import PeftModel

    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

try:
    from optimum.exporters.onnx import main_export as onnx_export

    _OPTIMUM_AVAILABLE = True
except ImportError:
    _OPTIMUM_AVAILABLE = False

try:
    from huggingface_hub import HfApi, create_repo

    _HF_HUB_AVAILABLE = True
except ImportError:
    _HF_HUB_AVAILABLE = False

try:
    from awq import AutoAWQForCausalLM  # type: ignore[import-untyped]

    _AWQ_AVAILABLE = True
except ImportError:
    _AWQ_AVAILABLE = False


# ---------------------------------------------------------------------------
# GGUF quantization types
# ---------------------------------------------------------------------------

SUPPORTED_GGUF_QUANT_TYPES = frozenset(
    {
        "q4_0",
        "q4_1",
        "q5_0",
        "q5_1",
        "q8_0",
        "f16",
        "f32",
        # llama.cpp extended types
        "q2_k",
        "q3_k_s",
        "q3_k_m",
        "q3_k_l",
        "q4_k_s",
        "q4_k_m",
        "q5_k_s",
        "q5_k_m",
        "q6_k",
    }
)


# ============================================================================
# ModelExporter
# ============================================================================


class ModelExporter:
    """Export trained models to various deployment formats.

    Supports safetensors, GGUF (via llama.cpp), ONNX (via optimum),
    LoRA adapter merging, and HuggingFace Hub publishing.
    """

    # ------------------------------------------------------------------ #
    # Safetensors export
    # ------------------------------------------------------------------ #

    @staticmethod
    def export_safetensors(
        model: PreTrainedModel | str,
        output_path: str,
    ) -> Path:
        """Export model weights in safetensors format.

        Parameters
        ----------
        model : PreTrainedModel or str
            A loaded model object, or a path to a model directory.
        output_path : str
            Destination directory for the exported model.

        Returns
        -------
        Path
            The output directory.
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for safetensors export. "
                "Install with: pip install transformers"
            )

        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        # If model is a string path, load it
        if isinstance(model, str):
            logger.info("Loading model from %s for safetensors export", model)
            loaded_model = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype="auto", device_map="cpu"
            )
            tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            loaded_model = model
            tokenizer = None

        logger.info("Exporting model to safetensors at %s", out)
        loaded_model.save_pretrained(str(out), safe_serialization=True)

        # Also save tokenizer if we loaded one
        if tokenizer is not None:
            tokenizer.save_pretrained(str(out))

        logger.info("Safetensors export complete: %s", out)
        return out

    # ------------------------------------------------------------------ #
    # GGUF export
    # ------------------------------------------------------------------ #

    @staticmethod
    def export_gguf(
        model_path: str,
        output_path: str,
        quantization: str = "q4_0",
    ) -> Path:
        """Export model to GGUF format for llama.cpp inference.

        Attempts to use ``llama-cpp-python`` first, then falls back to
        invoking the ``llama.cpp`` ``convert_hf_to_gguf.py`` script and
        ``llama-quantize`` binary via subprocess.

        Parameters
        ----------
        model_path : str
            Path to the HuggingFace-format model directory.
        output_path : str
            Destination path for the GGUF file.
        quantization : str
            Quantization type. Supported: q4_0, q4_1, q5_0, q5_1, q8_0, etc.

        Returns
        -------
        Path
            Path to the exported GGUF file.
        """
        quant_lower = quantization.lower()
        if quant_lower not in SUPPORTED_GGUF_QUANT_TYPES:
            raise ValueError(
                f"Unsupported GGUF quantization type: '{quantization}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_GGUF_QUANT_TYPES))}"
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Ensure the output path has a .gguf extension
        if out.suffix != ".gguf":
            out = out.with_suffix(".gguf")

        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        logger.info(
            "Exporting to GGUF: model=%s, quant=%s, output=%s",
            model_path,
            quant_lower,
            out,
        )

        # Try llama.cpp convert script + quantize binary
        converted = _try_llama_cpp_subprocess(model_dir, out, quant_lower)
        if converted:
            logger.info("GGUF export via llama.cpp tools complete: %s", out)
            return out

        raise RuntimeError(
            "GGUF export failed. llama.cpp tools were not found.\n\n"
            "Install llama.cpp:\n"
            "  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp\n"
            "  cmake -B build && cmake --build build\n"
            "Then ensure 'convert_hf_to_gguf.py' and 'llama-quantize' are "
            "accessible (set LLAMA_CPP_DIR env var or add to PATH)."
        )

    # ------------------------------------------------------------------ #
    # ONNX export
    # ------------------------------------------------------------------ #

    @staticmethod
    def export_onnx(
        model: PreTrainedModel | str,
        output_path: str,
    ) -> Path:
        """Export model to ONNX format using the optimum library.

        Parameters
        ----------
        model : PreTrainedModel or str
            A loaded model or path to a model directory.
        output_path : str
            Destination directory for ONNX artefacts.

        Returns
        -------
        Path
            The output directory containing the ONNX model.
        """
        if not _OPTIMUM_AVAILABLE:
            raise ImportError(
                "optimum is required for ONNX export. Install with: pip install optimum[exporters]"
            )

        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        # Determine model path
        if isinstance(model, str):
            model_path = model
        elif hasattr(model, "name_or_path"):
            model_path = model.name_or_path
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_path = model.config._name_or_path
        else:
            raise ValueError(
                "Cannot determine model path for ONNX export. "
                "Pass a model path string or a model with name_or_path attribute."
            )

        logger.info("Exporting model to ONNX: %s -> %s", model_path, out)

        try:
            onnx_export(
                model_name_or_path=model_path,
                output=str(out),
                task="text-generation-with-past",
            )
        except Exception as exc:
            # Fall back to a simpler task string
            logger.warning(
                "ONNX export with 'text-generation-with-past' failed: %s. "
                "Retrying with 'text-generation'.",
                exc,
            )
            onnx_export(
                model_name_or_path=model_path,
                output=str(out),
                task="text-generation",
            )

        logger.info("ONNX export complete: %s", out)
        return out

    # ------------------------------------------------------------------ #
    # AWQ quantization export
    # ------------------------------------------------------------------ #

    @staticmethod
    def export_awq(
        model: PreTrainedModel | str,
        output_path: str,
        quant_config: dict[str, Any] | None = None,
        calib_data: str | None = None,
    ) -> Path:
        """Export model with AWQ (Activation-aware Weight Quantization).

        AWQ provides best-in-class 4-bit quantization quality by preserving
        salient weights identified through activation-aware analysis.

        Parameters
        ----------
        model : PreTrainedModel or str
            A loaded model or path to a model directory.
        output_path : str
            Destination directory for the AWQ-quantized model.
        quant_config : dict, optional
            AWQ quantization config.  Defaults to ``{"zero_point": True,
            "q_group_size": 128, "w_bit": 4, "version": "GEMM"}``.
        calib_data : str, optional
            Calibration dataset name or path for activation analysis.
            Defaults to ``"pileval"`` (small calibration corpus).

        Returns
        -------
        Path
            The output directory containing the quantized model.
        """
        if not _AWQ_AVAILABLE:
            raise ImportError(
                "autoawq is required for AWQ export. Install with: pip install autoawq"
            )
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for AWQ export. Install with: pip install transformers"
            )

        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        # Resolve model path
        if isinstance(model, str):
            model_path = model
        elif hasattr(model, "name_or_path"):
            model_path = model.name_or_path
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_path = model.config._name_or_path
        else:
            raise ValueError(
                "Cannot determine model path for AWQ export. "
                "Pass a model path string or a model with name_or_path attribute."
            )

        # Default quantization config
        if quant_config is None:
            quant_config = {
                "zero_point": True,
                "q_group_size": 128,
                "w_bit": 4,
                "version": "GEMM",
            }

        calib_data = calib_data or "pileval"

        logger.info(
            "AWQ quantization: model=%s, config=%s, calib=%s",
            model_path,
            quant_config,
            calib_data,
        )

        # Load model via AutoAWQ
        awq_model = AutoAWQForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Quantize
        awq_model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

        # Save
        awq_model.save_quantized(str(out))
        tokenizer.save_pretrained(str(out))

        logger.info("AWQ export complete: %s", out)
        return out

    # ------------------------------------------------------------------ #
    # LoRA merge + export
    # ------------------------------------------------------------------ #

    @staticmethod
    def merge_lora_and_export(
        base_model: str,
        adapter_path: str,
        output_path: str,
        format: Literal["safetensors", "gguf", "onnx", "awq"] = "safetensors",
        gguf_quantization: str = "q4_0",
    ) -> Path:
        """Merge LoRA adapter into the base model and export.

        Parameters
        ----------
        base_model : str
            HuggingFace model name or local path for the base model.
        adapter_path : str
            Path to the directory containing the LoRA adapter weights.
        output_path : str
            Destination path for the merged and exported model.
        format : str
            Export format: ``"safetensors"``, ``"gguf"``, or ``"onnx"``.
        gguf_quantization : str
            GGUF quantization type (only used when ``format="gguf"``).

        Returns
        -------
        Path
            The output path.
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for LoRA merging. Install with: pip install transformers"
            )
        if not _PEFT_AVAILABLE:
            raise ImportError("peft is required for LoRA merging. Install with: pip install peft")

        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_dir}")

        out = Path(output_path)

        logger.info("Merging LoRA adapter: base=%s, adapter=%s", base_model, adapter_path)

        # Load base model in float16 explicitly (not "auto") to avoid
        # inheriting QLoRA 4-bit quantization — which would leave
        # bitsandbytes .absmax artifacts in the merged checkpoint.
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if _TORCH_AVAILABLE else "auto",
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Load and merge adapter
        model = PeftModel.from_pretrained(base, adapter_path)
        merged_model = model.merge_and_unload()

        logger.info("LoRA adapter merged successfully")

        # Save merged model to a temporary or final directory
        if format == "safetensors":
            out.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(str(out), safe_serialization=True)
            tokenizer.save_pretrained(str(out))
            logger.info("Merged model saved in safetensors format: %s", out)
            return out

        elif format == "gguf":
            # Save merged model to a temporary directory, then convert to GGUF
            merged_dir = out.parent / f"{out.stem}_merged_tmp"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(merged_dir))

            # Strip {% generation %} markers before GGUF conversion
            _clean_chat_template_for_export(merged_dir)

            try:
                result = ModelExporter.export_gguf(
                    model_path=str(merged_dir),
                    output_path=str(out),
                    quantization=gguf_quantization,
                )
            finally:
                # Clean up temporary merged directory
                shutil.rmtree(merged_dir, ignore_errors=True)
            return result

        elif format == "onnx":
            # Save merged model then export to ONNX
            merged_dir = out.parent / f"{out.stem}_merged_tmp"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(merged_dir))

            try:
                result = ModelExporter.export_onnx(
                    model=str(merged_dir),
                    output_path=str(out),
                )
            finally:
                shutil.rmtree(merged_dir, ignore_errors=True)
            return result

        elif format == "awq":
            # Save merged model then quantize with AWQ
            merged_dir = out.parent / f"{out.stem}_merged_tmp"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(merged_dir))

            try:
                result = ModelExporter.export_awq(
                    model=str(merged_dir),
                    output_path=str(out),
                )
            finally:
                shutil.rmtree(merged_dir, ignore_errors=True)
            return result

        else:
            raise ValueError(
                f"Unsupported export format: '{format}'. Supported: safetensors, gguf, onnx, awq"
            )

    # ------------------------------------------------------------------ #
    # Push to Hub
    # ------------------------------------------------------------------ #

    @staticmethod
    def push_to_hub(
        model_path: str,
        repo_id: str,
        token: str | None = None,
        private: bool = False,
        commit_message: str = "Upload model via llm-forge",
    ) -> str:
        """Push a model directory to HuggingFace Hub with an auto-generated model card.

        Parameters
        ----------
        model_path : str
            Local directory containing the model files.
        repo_id : str
            HuggingFace repository identifier (e.g. ``"username/my-model"``).
        token : str, optional
            HuggingFace API token.  If *None*, uses the cached token from
            ``huggingface-cli login``.
        private : bool
            Whether to create a private repository.
        commit_message : str
            Git commit message for the upload.

        Returns
        -------
        str
            URL of the uploaded model on HuggingFace Hub.
        """
        if not _HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required for Hub uploads. "
                "Install with: pip install huggingface_hub"
            )

        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model path not found: {model_dir}")

        logger.info("Pushing model to Hub: %s -> %s", model_path, repo_id)

        api = HfApi(token=token)

        # Create or get the repo
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="model",
        )

        # Auto-generate a model card if one does not exist
        readme_path = model_dir / "README.md"
        model_card_generated = False
        if not readme_path.exists():
            model_card = _generate_model_card(model_dir, repo_id)
            readme_path.write_text(model_card, encoding="utf-8")
            model_card_generated = True
            logger.info("Generated model card: %s", readme_path)

        # Upload all files
        try:
            api.upload_folder(
                folder_path=str(model_dir),
                repo_id=repo_id,
                commit_message=commit_message,
                repo_type="model",
            )
        finally:
            # Remove auto-generated card to avoid polluting the local dir
            if model_card_generated and readme_path.exists():
                readme_path.unlink()

        hub_url = f"https://huggingface.co/{repo_id}"
        logger.info("Model pushed successfully: %s", hub_url)
        return hub_url


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_llama_cpp_subprocess(
    model_dir: Path,
    output_path: Path,
    quantization: str,
) -> bool:
    """Attempt GGUF conversion using llama.cpp CLI tools."""
    # Step 1: Find the convert script
    convert_script = _find_convert_script()
    if convert_script is None:
        logger.debug("llama.cpp convert_hf_to_gguf.py not found")
        return False

    # Step 2: Convert HF model to f16 GGUF
    f16_path = output_path.parent / f"{output_path.stem}-f16.gguf"

    logger.info("Converting HF model to GGUF f16 via %s", convert_script)
    result = subprocess.run(
        [
            sys.executable,
            str(convert_script),
            str(model_dir),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.returncode != 0:
        logger.warning(
            "GGUF conversion failed:\nstdout: %s\nstderr: %s",
            result.stdout[-500:] if result.stdout else "",
            result.stderr[-500:] if result.stderr else "",
        )
        return False

    # Step 3: Quantize if needed (not f16 or f32)
    if quantization in ("f16", "f32"):
        # Already in the desired format
        if f16_path != output_path:
            shutil.move(str(f16_path), str(output_path))
        return True

    quantize_bin = shutil.which("llama-quantize")
    if quantize_bin is None:
        # Also try common alternative names
        for alt_name in ("quantize", "llama_quantize"):
            quantize_bin = shutil.which(alt_name)
            if quantize_bin is not None:
                break

    # Check common build locations if not on PATH
    if quantize_bin is None:
        llama_dir = os.environ.get("LLAMA_CPP_DIR")
        build_dirs = [
            Path(llama_dir) / "build" / "bin" if llama_dir else None,
            Path.home() / "llama.cpp" / "build" / "bin",
            Path("/opt/llama.cpp/build/bin"),
        ]
        for d in build_dirs:
            if d is None:
                continue
            candidate = d / "llama-quantize"
            if candidate.exists():
                quantize_bin = str(candidate)
                break

    if quantize_bin is None:
        logger.warning(
            "llama-quantize binary not found on PATH. Saving f16 GGUF only; quantization skipped."
        )
        if f16_path != output_path:
            shutil.move(str(f16_path), str(output_path))
        return True

    logger.info("Quantizing GGUF: %s -> %s (%s)", f16_path, output_path, quantization)
    result = subprocess.run(
        [quantize_bin, str(f16_path), str(output_path), quantization.upper()],
        capture_output=True,
        text=True,
        timeout=3600,
    )

    # Clean up f16 intermediate
    if f16_path.exists() and f16_path != output_path:
        f16_path.unlink()

    if result.returncode != 0:
        logger.warning(
            "GGUF quantization failed:\nstdout: %s\nstderr: %s",
            result.stdout[-500:] if result.stdout else "",
            result.stderr[-500:] if result.stderr else "",
        )
        return False

    return True


def _find_convert_script() -> Path | None:
    """Locate the llama.cpp ``convert_hf_to_gguf.py`` script."""
    # Check PATH
    which_result = shutil.which("convert_hf_to_gguf.py")
    if which_result:
        return Path(which_result)

    # Check common locations
    common_paths = [
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/opt/llama.cpp/convert_hf_to_gguf.py"),
        Path("./llama.cpp/convert_hf_to_gguf.py"),
    ]

    # Also check LLAMA_CPP_DIR environment variable
    llama_dir = os.environ.get("LLAMA_CPP_DIR")
    if llama_dir:
        common_paths.insert(0, Path(llama_dir) / "convert_hf_to_gguf.py")

    for p in common_paths:
        if p.exists():
            return p

    return None


def _clean_chat_template_for_export(model_dir: Path) -> None:
    """Strip ``{% generation %}`` markers from the chat template before GGUF export.

    TRL v0.20's ``assistant_only_loss`` injects ``{% generation %}...{% endgeneration %}``
    markers into the Jinja2 chat template during training.  These are HuggingFace-only
    extensions that are **not** valid Jinja2 — they cause errors in llama.cpp /
    Ollama / vLLM.  This function removes them so the exported model works everywhere.
    """
    # Check chat_template.jinja file (preferred by convert_hf_to_gguf.py)
    jinja_path = model_dir / "chat_template.jinja"
    if jinja_path.exists():
        content = jinja_path.read_text(encoding="utf-8")
        if "{% generation %}" in content:
            cleaned = content.replace("{% generation %}", "").replace("{% endgeneration %}", "")
            jinja_path.write_text(cleaned, encoding="utf-8")
            logger.info("Stripped {%% generation %%} markers from %s", jinja_path)

    # Also check tokenizer_config.json (fallback location for chat_template)
    tok_config_path = model_dir / "tokenizer_config.json"
    if tok_config_path.exists():
        try:
            with open(tok_config_path, encoding="utf-8") as f:
                tok_config = json.load(f)
            template = tok_config.get("chat_template", "")
            if isinstance(template, str) and "{% generation %}" in template:
                tok_config["chat_template"] = template.replace("{% generation %}", "").replace(
                    "{% endgeneration %}", ""
                )
                with open(tok_config_path, "w", encoding="utf-8") as f:
                    json.dump(tok_config, f, indent=2, ensure_ascii=False)
                logger.info("Stripped {%% generation %%} markers from tokenizer_config.json")
        except (json.JSONDecodeError, OSError):
            pass


def generate_modelfile(
    gguf_path: str | Path,
    output_dir: str | Path,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.1,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    num_predict: int = 256,
    num_ctx: int = 2048,
) -> Path:
    """Generate an Ollama Modelfile for the given GGUF model.

    Uses the ``range .Messages`` template pattern for proper multi-turn
    conversation support (the legacy ``.Prompt``/``.Response`` pattern
    destroys turn boundaries in multi-turn chat).

    Parameters
    ----------
    gguf_path : str or Path
        Path to the GGUF model file.
    output_dir : str or Path
        Directory to write the Modelfile into.
    system_prompt : str, optional
        System prompt.  If None, a generic assistant prompt is used.
    temperature, top_p, top_k, repeat_penalty, num_predict
        Inference sampling parameters.
    num_ctx : int
        KV-cache context window size for Ollama.  Should match or
        slightly exceed the training ``max_seq_length``.

    Returns
    -------
    Path
        Path to the generated Modelfile.
    """
    gguf_path = Path(gguf_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modelfile_path = out_dir / "Modelfile"

    # Resolve GGUF path relative to the Modelfile directory
    try:
        relative_gguf = gguf_path.relative_to(out_dir)
    except ValueError:
        relative_gguf = gguf_path

    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant."

    # Ollama TEMPLATE block for Llama 3 format — uses range .Messages
    # for proper multi-turn conversation support.  The legacy .Prompt/
    # .Response pattern is single-turn only: it flattens conversation
    # history into two strings, destroying turn boundaries.
    # NOTE: Do NOT include <|begin_of_text|> — Ollama adds BOS automatically.
    template_block = (
        'TEMPLATE """<|start_header_id|>system<|end_header_id|>\n\n'
        "Cutting Knowledge Date: December 2023\n\n"
        "{{ if .System }}{{ .System }}\n"
        "{{- end }}<|eot_id|>\n"
        "{{- range $i, $_ := .Messages }}\n"
        "{{- $last := eq (len (slice $.Messages $i)) 1 }}\n"
        '{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>\n\n'
        "{{ .Content }}<|eot_id|>"
        "{{ if $last }}<|start_header_id|>assistant<|end_header_id|>\n\n"
        "{{ end }}\n"
        '{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>\n\n'
        "{{ .Content }}"
        "{{ if not $last }}<|eot_id|>{{ end }}\n"
        "{{- end }}\n"
        '{{- end }}"""'
    )

    content = (
        f"FROM ./{relative_gguf}\n"
        f"\n"
        f'SYSTEM "{system_prompt}"\n'
        f"\n"
        f"{template_block}\n"
        f"\n"
        f"PARAMETER temperature {temperature}\n"
        f"PARAMETER top_p {top_p}\n"
        f"PARAMETER top_k {top_k}\n"
        f"PARAMETER repeat_penalty {repeat_penalty}\n"
        f"PARAMETER repeat_last_n 128\n"
        f'PARAMETER stop "<|eot_id|>"\n'
        f'PARAMETER stop "<|end_of_text|>"\n'
        f'PARAMETER stop "<|start_header_id|>"\n'
        f"PARAMETER num_predict {num_predict}\n"
        f"PARAMETER num_ctx {num_ctx}\n"
    )

    modelfile_path.write_text(content, encoding="utf-8")
    logger.info(
        "Modelfile generated: %s (temp=%.1f, repeat_penalty=%.1f, num_predict=%d)",
        modelfile_path,
        temperature,
        repeat_penalty,
        num_predict,
    )
    return modelfile_path


def _generate_model_card(model_dir: Path, repo_id: str) -> str:
    """Generate a basic model card (README.md) for the uploaded model."""
    # Try to read model config for metadata
    config_path = model_dir / "config.json"
    model_type = "unknown"
    hidden_size = "N/A"
    num_layers = "N/A"

    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type", "unknown")
            hidden_size = cfg.get("hidden_size", "N/A")
            num_layers = cfg.get("num_hidden_layers", "N/A")
        except (json.JSONDecodeError, OSError):
            pass

    # Check for adapter_config (indicates LoRA was used)
    adapter_path = model_dir / "adapter_config.json"
    training_note = ""
    if adapter_path.exists():
        try:
            with open(adapter_path, encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            lora_r = adapter_cfg.get("r", "N/A")
            lora_alpha = adapter_cfg.get("lora_alpha", "N/A")
            training_note = (
                f"\n## Training\n\n"
                f"This model was fine-tuned using LoRA (r={lora_r}, alpha={lora_alpha}).\n"
            )
        except (json.JSONDecodeError, OSError):
            pass

    # List files in the model directory
    files = sorted(f.name for f in model_dir.iterdir() if f.is_file())
    files_section = "\n".join(f"- `{f}`" for f in files[:20])
    if len(files) > 20:
        files_section += f"\n- ... and {len(files) - 20} more files"

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    card = f"""---
tags:
- llm-forge
- text-generation
library_name: transformers
pipeline_tag: text-generation
---

# {repo_id}

This model was trained and exported using [llm-forge](https://github.com/llm-forge/llm-forge).

## Model Details

| Property | Value |
|----------|-------|
| Model Type | {model_type} |
| Hidden Size | {hidden_size} |
| Layers | {num_layers} |
| Uploaded | {timestamp} |
{training_note}
## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Files

{files_section}

---

*Generated by llm-forge on {timestamp}.*
"""
    return card
