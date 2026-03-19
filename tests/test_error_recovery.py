"""Tests for the error recovery suggestions system.

Covers pattern matching for all 10 error categories, context-aware
suggestions, and plain-text formatting of diagnoses.
"""

from __future__ import annotations

from unittest import mock

import pytest

try:
    from llm_forge.utils.error_recovery import (
        ErrorDiagnosis,
        RecoverySuggestion,
        diagnose_error,
        format_diagnosis_plain,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE,
    reason="llm_forge.utils.error_recovery not importable",
)


# ===================================================================
# RecoverySuggestion / ErrorDiagnosis data classes
# ===================================================================


class TestRecoverySuggestion:
    """Test the RecoverySuggestion dataclass."""

    def test_defaults(self) -> None:
        s = RecoverySuggestion(category="test", message="Do something")
        assert s.category == "test"
        assert s.message == "Do something"
        assert s.priority == 0

    def test_custom_priority(self) -> None:
        s = RecoverySuggestion(category="test", message="Later", priority=5)
        assert s.priority == 5


class TestErrorDiagnosis:
    """Test the ErrorDiagnosis dataclass."""

    def test_suggestion_texts_sorted_by_priority(self) -> None:
        d = ErrorDiagnosis(
            error_type="Test",
            error_message="test",
            suggestions=[
                RecoverySuggestion("a", "Third", 2),
                RecoverySuggestion("a", "First", 0),
                RecoverySuggestion("a", "Second", 1),
            ],
        )
        texts = d.suggestion_texts
        assert texts == ["First", "Second", "Third"]

    def test_empty_suggestions(self) -> None:
        d = ErrorDiagnosis(error_type="Test", error_message="test")
        assert d.suggestion_texts == []


# ===================================================================
# diagnose_error() — pattern matching
# ===================================================================


class TestDiagnoseErrorPatterns:
    """Test error pattern matching for all 10 categories."""

    def test_out_of_memory(self) -> None:
        err = RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        d = diagnose_error(err)
        assert d.error_type == "Out of Memory"
        assert len(d.suggestions) >= 3

    def test_oom_mps(self) -> None:
        err = RuntimeError("MPS backend: out of memory")
        d = diagnose_error(err)
        assert d.error_type == "Out of Memory"

    def test_nan_loss(self) -> None:
        err = ValueError("Loss is NaN at step 100")
        d = diagnose_error(err)
        assert d.error_type == "Training Instability (NaN/Inf)"

    def test_inf_loss(self) -> None:
        err = ValueError("Loss diverged to Inf")
        d = diagnose_error(err)
        assert d.error_type == "Training Instability (NaN/Inf)"

    def test_import_error(self) -> None:
        err = ImportError("No module named 'bitsandbytes'")
        d = diagnose_error(err)
        assert d.error_type == "Missing Dependencies"

    def test_module_not_found(self) -> None:
        err = ModuleNotFoundError("No module named 'peft'")
        d = diagnose_error(err)
        assert d.error_type == "Missing Dependencies"

    def test_validation_error(self) -> None:
        err = ValueError("validation error: field required 'model'")
        d = diagnose_error(err)
        assert d.error_type == "Configuration Error"

    def test_pydantic_error(self) -> None:
        err = ValueError("pydantic validation failed: extra inputs not permitted")
        d = diagnose_error(err)
        assert d.error_type == "Configuration Error"

    def test_network_error(self) -> None:
        err = ConnectionError("Could not resolve host: huggingface.co")
        d = diagnose_error(err)
        assert d.error_type == "Network Error"

    def test_timeout_error(self) -> None:
        err = TimeoutError("Connection timeout after 30s")
        d = diagnose_error(err)
        assert d.error_type == "Network Error"

    def test_cuda_error(self) -> None:
        err = RuntimeError("CUDA error: device-side assert triggered")
        d = diagnose_error(err)
        assert d.error_type == "CUDA Error"

    def test_checkpoint_error(self) -> None:
        err = FileNotFoundError("Could not load checkpoint from state_dict")
        d = diagnose_error(err)
        assert d.error_type == "Checkpoint Error"

    def test_tokenizer_error(self) -> None:
        err = ValueError("Invalid token id 999999 for tokenizer")
        d = diagnose_error(err)
        assert d.error_type == "Tokenizer Error"

    def test_dataset_error(self) -> None:
        err = KeyError("Column 'instruction' not found in dataset")
        d = diagnose_error(err)
        assert d.error_type == "Dataset Error"

    def test_disk_space_error(self) -> None:
        err = OSError("[Errno 28] No space left on device")
        d = diagnose_error(err)
        assert d.error_type == "Disk Space Error"

    def test_unknown_error_gets_general_suggestions(self) -> None:
        err = RuntimeError("Something completely unexpected happened")
        d = diagnose_error(err)
        assert d.error_type == "RuntimeError"
        assert len(d.suggestions) >= 1
        assert any("doctor" in s.message.lower() for s in d.suggestions)


# ===================================================================
# Context-aware suggestions
# ===================================================================


class TestContextAwareSuggestions:
    """Test that config context adds targeted suggestions."""

    def _make_config(
        self, mode: str = "lora", batch_size: int = 4, seq_len: int = 2048, lr: float = 1e-4
    ) -> mock.MagicMock:
        cfg = mock.MagicMock()
        cfg.training.mode = mode
        cfg.training.per_device_train_batch_size = batch_size
        cfg.training.learning_rate = lr
        cfg.model.max_seq_length = seq_len
        return cfg

    def test_oom_suggests_qlora_switch(self) -> None:
        """OOM with LoRA → suggest switching to QLoRA."""
        err = RuntimeError("CUDA out of memory")
        cfg = self._make_config(mode="lora")
        d = diagnose_error(err, config=cfg)
        texts = d.suggestion_texts
        assert any("qlora" in t.lower() for t in texts)

    def test_oom_suggests_batch_reduction(self) -> None:
        """OOM with batch>1 → suggest reducing batch size."""
        err = RuntimeError("CUDA out of memory")
        cfg = self._make_config(batch_size=8)
        d = diagnose_error(err, config=cfg)
        texts = d.suggestion_texts
        assert any("batch_size=8" in t for t in texts)

    def test_oom_suggests_seq_len_reduction(self) -> None:
        """OOM with long seq → suggest reducing seq length."""
        err = RuntimeError("CUDA out of memory")
        cfg = self._make_config(seq_len=4096)
        d = diagnose_error(err, config=cfg)
        texts = d.suggestion_texts
        assert any("4096" in t for t in texts)

    def test_nan_suggests_lr_reduction(self) -> None:
        """NaN with high LR → suggest lowering it."""
        err = ValueError("Loss is NaN")
        cfg = self._make_config(lr=1e-3)
        d = diagnose_error(err, config=cfg)
        texts = d.suggestion_texts
        assert any("learning_rate" in t for t in texts)


# ===================================================================
# format_diagnosis_plain()
# ===================================================================


class TestFormatDiagnosisPlain:
    """Test plain-text formatting of error diagnoses."""

    def test_format_basic(self) -> None:
        d = ErrorDiagnosis(
            error_type="Out of Memory",
            error_message="CUDA out of memory",
            suggestions=[
                RecoverySuggestion("memory", "Reduce batch size", 0),
                RecoverySuggestion("memory", "Use QLoRA", 1),
            ],
        )
        text = format_diagnosis_plain(d)
        assert "Out of Memory" in text
        assert "Reduce batch size" in text
        assert "Use QLoRA" in text

    def test_format_truncates_long_message(self) -> None:
        d = ErrorDiagnosis(
            error_type="Test",
            error_message="x" * 500,
        )
        text = format_diagnosis_plain(d)
        # Message in output is truncated to 200 chars
        assert len(text.split("\n")[1]) < 250

    def test_format_numbered_suggestions(self) -> None:
        d = ErrorDiagnosis(
            error_type="Test",
            error_message="test",
            suggestions=[
                RecoverySuggestion("a", "First fix", 0),
                RecoverySuggestion("a", "Second fix", 1),
            ],
        )
        text = format_diagnosis_plain(d)
        assert "1." in text
        assert "2." in text
