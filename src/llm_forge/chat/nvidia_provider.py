"""NVIDIA NIM provider for LLM Forge.

Gives every user free access to 189+ models via NVIDIA's NIM API.
Uses an embedded API key (obfuscated, not plaintext) for the community.
Users can also provide their own NVIDIA_API_KEY for higher limits.
"""

from __future__ import annotations

import base64
import os

# NVIDIA NIM API configuration
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Obfuscated community API key (XOR + base64)
# Users can override with NVIDIA_API_KEY env var
_OBFUSCATED_KEY = "RFxLWkMHS2ViTmkaGBhjWBoYY1ldTRNtWE4fRH1lcEVBc2xyeHp/bUJaYhNDbGJlRRJwTGNCRV4eWExIbB9JWWRraEJ7SQ=="
_XOR_KEY = 42  # Simple XOR for basic obfuscation


def _deobfuscate(encoded: str) -> str:
    """Deobfuscate the embedded API key."""
    raw = base64.b64decode(encoded)
    return "".join(chr(b ^ _XOR_KEY) for b in raw)


def get_nvidia_api_key() -> str:
    """Get NVIDIA API key -- user's own or embedded community key."""
    # User's own key takes priority
    user_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if user_key:
        return user_key
    # Fall back to embedded community key
    return _deobfuscate(_OBFUSCATED_KEY)


# Available models organized by category
NVIDIA_MODELS = {
    # Recommended (best quality/speed balance)
    "llama-3.3-70b": {
        "id": "meta/llama-3.3-70b-instruct",
        "name": "Llama 3.3 70B",
        "params": "70B",
        "category": "recommended",
    },
    "llama-3.2-3b": {
        "id": "meta/llama-3.2-3b-instruct",
        "name": "Llama 3.2 3B",
        "params": "3B",
        "category": "recommended",
    },
    "llama-3.1-405b": {
        "id": "meta/llama-3.1-405b-instruct",
        "name": "Llama 3.1 405B",
        "params": "405B",
        "category": "recommended",
    },
    "deepseek-v3.2": {
        "id": "deepseek-ai/deepseek-v3.2",
        "name": "DeepSeek V3.2",
        "params": "Large",
        "category": "recommended",
    },
    # Llama family
    "llama-3.2-1b": {
        "id": "meta/llama-3.2-1b-instruct",
        "name": "Llama 3.2 1B",
        "params": "1B",
        "category": "llama",
    },
    "llama-3.1-70b": {
        "id": "meta/llama-3.1-70b-instruct",
        "name": "Llama 3.1 70B",
        "params": "70B",
        "category": "llama",
    },
    "llama-3.1-8b": {
        "id": "meta/llama-3.1-8b-instruct",
        "name": "Llama 3.1 8B",
        "params": "8B",
        "category": "llama",
    },
    "llama-4-scout": {
        "id": "meta/llama-4-scout-17b-16e-instruct",
        "name": "Llama 4 Scout",
        "params": "17B",
        "category": "llama",
    },
    "llama-4-maverick": {
        "id": "meta/llama-4-maverick-17b-128e-instruct",
        "name": "Llama 4 Maverick",
        "params": "17B",
        "category": "llama",
    },
    # DeepSeek
    "deepseek-r1-8b": {
        "id": "deepseek-ai/deepseek-r1-distill-llama-8b",
        "name": "DeepSeek R1 8B",
        "params": "8B",
        "category": "deepseek",
    },
    "deepseek-r1-32b": {
        "id": "deepseek-ai/deepseek-r1-distill-qwen-32b",
        "name": "DeepSeek R1 32B",
        "params": "32B",
        "category": "deepseek",
    },
    "deepseek-v3.1": {
        "id": "deepseek-ai/deepseek-v3.1",
        "name": "DeepSeek V3.1",
        "params": "Large",
        "category": "deepseek",
    },
    # Google Gemma
    "gemma-3-27b": {
        "id": "google/gemma-3-27b-it",
        "name": "Gemma 3 27B",
        "params": "27B",
        "category": "google",
    },
    "gemma-3-12b": {
        "id": "google/gemma-3-12b-it",
        "name": "Gemma 3 12B",
        "params": "12B",
        "category": "google",
    },
    "gemma-3-4b": {
        "id": "google/gemma-3-4b-it",
        "name": "Gemma 3 4B",
        "params": "4B",
        "category": "google",
    },
    # Mistral
    "mistral-large-2": {
        "id": "mistralai/mistral-large-2-instruct",
        "name": "Mistral Large 2",
        "params": "123B",
        "category": "mistral",
    },
    # NVIDIA
    "nemotron-70b": {
        "id": "nvidia/llama-3.1-nemotron-70b-instruct",
        "name": "Nemotron 70B",
        "params": "70B",
        "category": "nvidia",
    },
    # Code
    "starcoder2-15b": {
        "id": "bigcode/starcoder2-15b",
        "name": "StarCoder2 15B",
        "params": "15B",
        "category": "code",
    },
    "deepseek-coder-7b": {
        "id": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "name": "DeepSeek Coder 7B",
        "params": "7B",
        "category": "code",
    },
}

DEFAULT_NVIDIA_MODEL = "llama-3.3-70b"


def call_nvidia(
    messages: list[dict],
    system: str,
    model_key: str | None = None,
    max_tokens: int = 4096,
) -> dict:
    """Call NVIDIA NIM API. OpenAI-compatible."""
    from openai import OpenAI

    model_info = NVIDIA_MODELS.get(model_key or DEFAULT_NVIDIA_MODEL)
    if model_info is None:
        model_info = NVIDIA_MODELS[DEFAULT_NVIDIA_MODEL]

    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=get_nvidia_api_key(),
    )

    # Prepend system message
    full_messages = [{"role": "system", "content": system}] + messages

    response = client.chat.completions.create(
        model=model_info["id"],
        messages=full_messages,
        max_tokens=max_tokens,
    )
    return response


def stream_nvidia(
    messages: list[dict],
    system: str,
    model_key: str | None = None,
    on_text=None,
    interrupt_check=None,
    max_tokens: int = 4096,
):
    """Stream NVIDIA NIM API response."""
    from openai import OpenAI

    model_info = NVIDIA_MODELS.get(model_key or DEFAULT_NVIDIA_MODEL)
    if model_info is None:
        model_info = NVIDIA_MODELS[DEFAULT_NVIDIA_MODEL]

    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=get_nvidia_api_key(),
    )

    full_messages = [{"role": "system", "content": system}] + messages

    stream = client.chat.completions.create(
        model=model_info["id"],
        messages=full_messages,
        max_tokens=max_tokens,
        stream=True,
    )

    collected_text = []

    for chunk in stream:
        if interrupt_check and interrupt_check():
            break
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            collected_text.append(text)
            if on_text:
                on_text(text)

    # Build a response-like object for compatibility
    from types import SimpleNamespace

    response = SimpleNamespace()
    response.choices = [SimpleNamespace()]
    response.choices[0].message = SimpleNamespace()
    response.choices[0].message.content = "".join(collected_text)
    response.choices[0].message.tool_calls = None

    return response


def list_nvidia_models() -> str:
    """Return formatted list of available NVIDIA models."""
    import json

    categories: dict[str, list] = {}
    for key, info in NVIDIA_MODELS.items():
        cat = info["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({"key": key, **info})
    return json.dumps(categories, indent=2)
