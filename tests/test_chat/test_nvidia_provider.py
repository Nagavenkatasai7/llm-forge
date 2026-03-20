"""Tests for the NVIDIA NIM provider module."""

from __future__ import annotations

import base64
import json
import os

import pytest


def test_deobfuscate_roundtrip():
    """Obfuscating then deobfuscating should return the original key."""
    from llm_forge.chat.nvidia_provider import _XOR_KEY, _deobfuscate

    original = "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"
    # Obfuscate
    obfuscated = base64.b64encode(bytes(ord(c) ^ _XOR_KEY for c in original)).decode()
    # Deobfuscate
    result = _deobfuscate(obfuscated)
    assert result == original


def test_get_nvidia_api_key_env_override(monkeypatch):
    """When NVIDIA_API_KEY env var is set, it should take priority."""
    from llm_forge.chat.nvidia_provider import get_nvidia_api_key

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-user-custom-key")
    assert get_nvidia_api_key() == "nvapi-user-custom-key"


def test_get_nvidia_api_key_fallback(monkeypatch):
    """When no NVIDIA_API_KEY env var, should fall back to embedded key."""
    from llm_forge.chat.nvidia_provider import get_nvidia_api_key

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    key = get_nvidia_api_key()
    assert key.startswith("nvapi-")
    assert len(key) > 20


def test_nvidia_models_has_entries():
    """NVIDIA_MODELS dict should have at least 10 models."""
    from llm_forge.chat.nvidia_provider import NVIDIA_MODELS

    assert len(NVIDIA_MODELS) >= 10
    # Each model should have required keys
    for key, info in NVIDIA_MODELS.items():
        assert "id" in info, f"Model {key} missing 'id'"
        assert "name" in info, f"Model {key} missing 'name'"
        assert "params" in info, f"Model {key} missing 'params'"
        assert "category" in info, f"Model {key} missing 'category'"


def test_default_model_exists():
    """DEFAULT_NVIDIA_MODEL must exist in NVIDIA_MODELS."""
    from llm_forge.chat.nvidia_provider import DEFAULT_NVIDIA_MODEL, NVIDIA_MODELS

    assert DEFAULT_NVIDIA_MODEL in NVIDIA_MODELS


def test_list_nvidia_models_returns_json():
    """list_nvidia_models should return valid JSON grouped by category."""
    from llm_forge.chat.nvidia_provider import list_nvidia_models

    result = list_nvidia_models()
    parsed = json.loads(result)
    assert isinstance(parsed, dict)
    # Should have at least 3 categories
    assert len(parsed) >= 3
    # Each category should be a list
    for _cat, models in parsed.items():
        assert isinstance(models, list)
        assert len(models) >= 1
        # Each model in a category should have a 'key' field
        for m in models:
            assert "key" in m
            assert "id" in m


def test_embedded_key_deobfuscates_correctly():
    """The embedded _OBFUSCATED_KEY should deobfuscate to a valid nvapi key."""
    from llm_forge.chat.nvidia_provider import _OBFUSCATED_KEY, _deobfuscate

    key = _deobfuscate(_OBFUSCATED_KEY)
    assert key.startswith("nvapi-")
    assert len(key) == 70  # NVIDIA API key length for this key


def test_nvidia_base_url():
    """NVIDIA_BASE_URL should point to the NIM API."""
    from llm_forge.chat.nvidia_provider import NVIDIA_BASE_URL

    assert "nvidia" in NVIDIA_BASE_URL.lower() or "integrate.api" in NVIDIA_BASE_URL
    assert NVIDIA_BASE_URL.startswith("https://")


def test_model_categories_present():
    """Ensure expected categories exist in the model catalog."""
    from llm_forge.chat.nvidia_provider import NVIDIA_MODELS

    categories = {info["category"] for info in NVIDIA_MODELS.values()}
    assert "recommended" in categories
    assert "llama" in categories
    assert "deepseek" in categories
    assert "google" in categories
    assert "code" in categories
