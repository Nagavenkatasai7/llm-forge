"""Tests for pre-training hardware/config compatibility checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_forge.training.preflight import (
    PreflightError,
    assert_compatible,
    check_config,
)


def make_config(mode: str = "lora", mlx: bool = False, load_in_8bit: bool = False):
    return SimpleNamespace(
        training=SimpleNamespace(mode=mode),
        mlx=SimpleNamespace(enabled=mlx),
        quantization=SimpleNamespace(load_in_8bit=load_in_8bit),
        data=SimpleNamespace(max_seq_length=2048),
    )


class TestQLoRAOnAppleSilicon:
    """The motivating failure: QLoRA dies inside from_pretrained on a Mac.

    bitsandbytes has no Metal backend, so the config is unrunnable -- but
    nothing caught it until after the model had downloaded.
    """

    def test_qlora_on_mps_is_blocked(self) -> None:
        problems = check_config(make_config(mode="qlora"), backend="mps")
        assert problems
        assert any("bitsandbytes" in p.why for p in problems)

    def test_the_fix_names_mlx(self) -> None:
        problems = check_config(make_config(mode="qlora"), backend="mps")
        assert any("mlx" in p.fix.lower() for p in problems)

    def test_qlora_on_cpu_is_blocked(self) -> None:
        assert check_config(make_config(mode="qlora"), backend="cpu")

    def test_qlora_on_cuda_is_fine(self) -> None:
        assert check_config(make_config(mode="qlora"), backend="cuda") == []

    def test_mlx_config_on_mps_is_fine(self) -> None:
        """With mlx enabled the qlora mode is handled by MLX, not bitsandbytes."""
        assert check_config(make_config(mode="qlora", mlx=True), backend="mps") == []


class TestMLXPortability:
    def test_mlx_on_cuda_is_blocked(self) -> None:
        problems = check_config(make_config(mlx=True), backend="cuda")
        assert problems
        assert any("Apple Silicon" in p.why for p in problems)

    def test_mlx_on_mps_is_fine(self) -> None:
        assert check_config(make_config(mlx=True), backend="mps") == []


class TestEightBit:
    def test_8bit_needs_cuda(self) -> None:
        problems = check_config(make_config(load_in_8bit=True), backend="mps")
        assert any("load_in_8bit" in p.what for p in problems)
        assert any("bitsandbytes" in p.why for p in problems)

    def test_8bit_on_cuda_is_fine(self) -> None:
        assert check_config(make_config(load_in_8bit=True), backend="cuda") == []


class TestLoRAEverywhere:
    @pytest.mark.parametrize("backend", ["cuda", "mps", "cpu"])
    def test_plain_lora_runs_on_every_backend(self, backend: str) -> None:
        assert check_config(make_config(mode="lora"), backend=backend) == []

    @pytest.mark.parametrize("backend", ["cuda", "mps", "cpu"])
    def test_full_finetune_runs_on_every_backend(self, backend: str) -> None:
        assert check_config(make_config(mode="full"), backend=backend) == []


class TestAssertCompatible:
    def test_raises_with_actionable_message(self) -> None:
        with pytest.raises(PreflightError) as excinfo:
            assert_compatible(make_config(mode="qlora"), backend="mps")

        message = str(excinfo.value)
        assert "Fix:" in message
        assert "mlx.enabled" in message
        # The user must know nothing was downloaded before they hit this.
        assert "Nothing has been downloaded" in message

    def test_silent_when_compatible(self) -> None:
        assert_compatible(make_config(mode="lora"), backend="mps")


class TestEnumModes:
    def test_reads_str_enum_mode(self) -> None:
        """TrainingMode is a str-Enum, so .value must be unwrapped."""
        import enum

        class TrainingMode(str, enum.Enum):
            qlora = "qlora"

        config = make_config()
        config.training = SimpleNamespace(mode=TrainingMode.qlora)
        assert check_config(config, backend="mps")
