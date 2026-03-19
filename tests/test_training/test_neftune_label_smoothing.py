"""Tests for NEFTune and label_smoothing_factor config fields.

Phase 1 Foundation features:
- NEFTune noise injection: default alpha=None (disabled), ref arxiv:2310.05914
- Label smoothing: default 0.0 (disabled), reduces overconfidence when enabled
"""

from __future__ import annotations

import pytest

try:
    from llm_forge.config.schema import LLMForgeConfig, TrainingConfig

    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _SCHEMA_AVAILABLE,
    reason="llm_forge.config.schema not importable",
)


# ===================================================================
# NEFTune Config Tests
# ===================================================================


class TestNEFTuneConfig:
    """Verify NEFTune noise alpha defaults and validation."""

    def test_default_alpha_is_none(self) -> None:
        """Default NEFTune alpha is None (disabled by default)."""
        cfg = TrainingConfig()
        assert cfg.neftune_noise_alpha is None

    def test_can_disable_by_none(self) -> None:
        """Users can disable NEFTune by setting alpha to None."""
        cfg = TrainingConfig(neftune_noise_alpha=None)
        assert cfg.neftune_noise_alpha is None

    def test_can_set_custom_alpha(self) -> None:
        """Users can set a custom alpha value."""
        cfg = TrainingConfig(neftune_noise_alpha=10.0)
        assert cfg.neftune_noise_alpha == 10.0

    def test_zero_alpha_accepted(self) -> None:
        """Alpha=0 is valid (effectively disabled)."""
        cfg = TrainingConfig(neftune_noise_alpha=0.0)
        assert cfg.neftune_noise_alpha == 0.0

    def test_negative_alpha_rejected(self) -> None:
        """Negative alpha should be rejected by ge=0.0 validator."""
        with pytest.raises(Exception):
            TrainingConfig(neftune_noise_alpha=-1.0)

    def test_full_config_with_neftune(self) -> None:
        """Full config with explicit NEFTune alpha validates."""
        config = LLMForgeConfig(
            model={"name": "test-model"},
            data={"train_path": "test-data"},
            training={"neftune_noise_alpha": 5.0},
        )
        assert config.training.neftune_noise_alpha == 5.0

    def test_full_config_default_neftune(self) -> None:
        """Full config without explicit NEFTune gets alpha=None."""
        config = LLMForgeConfig(
            model={"name": "test-model"},
            data={"train_path": "test-data"},
        )
        assert config.training.neftune_noise_alpha is None


# ===================================================================
# Label Smoothing Config Tests
# ===================================================================


class TestLabelSmoothingConfig:
    """Verify label_smoothing_factor defaults and validation."""

    def test_default_is_zero(self) -> None:
        """Default label_smoothing_factor is 0.0 (disabled)."""
        cfg = TrainingConfig()
        assert cfg.label_smoothing_factor == 0.0

    def test_field_exists(self) -> None:
        """TrainingConfig has the label_smoothing_factor field."""
        cfg = TrainingConfig()
        assert hasattr(cfg, "label_smoothing_factor")

    def test_can_disable(self) -> None:
        """Users can disable label smoothing by setting to 0.0."""
        cfg = TrainingConfig(label_smoothing_factor=0.0)
        assert cfg.label_smoothing_factor == 0.0

    def test_can_set_custom_value(self) -> None:
        """Users can set a custom smoothing factor."""
        cfg = TrainingConfig(label_smoothing_factor=0.2)
        assert cfg.label_smoothing_factor == 0.2

    def test_max_value_one(self) -> None:
        """label_smoothing_factor=1.0 is valid (upper bound)."""
        cfg = TrainingConfig(label_smoothing_factor=1.0)
        assert cfg.label_smoothing_factor == 1.0

    def test_negative_rejected(self) -> None:
        """Negative smoothing factor should be rejected."""
        with pytest.raises(Exception):
            TrainingConfig(label_smoothing_factor=-0.1)

    def test_above_one_rejected(self) -> None:
        """Smoothing factor > 1.0 should be rejected."""
        with pytest.raises(Exception):
            TrainingConfig(label_smoothing_factor=1.5)

    def test_full_config_with_label_smoothing(self) -> None:
        """Full config with explicit label_smoothing validates."""
        config = LLMForgeConfig(
            model={"name": "test-model"},
            data={"train_path": "test-data"},
            training={"label_smoothing_factor": 0.1},
        )
        assert config.training.label_smoothing_factor == 0.1

    def test_yaml_roundtrip(self, tmp_path) -> None:
        """Config with both features survives YAML roundtrip."""
        import yaml

        config_dict = {
            "model": {"name": "test-model"},
            "data": {"train_path": "test-data"},
            "training": {
                "neftune_noise_alpha": 5.0,
                "label_smoothing_factor": 0.1,
            },
        }
        yaml_path = tmp_path / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f)

        with open(yaml_path) as f:
            loaded = yaml.safe_load(f)

        config = LLMForgeConfig(**loaded)
        assert config.training.neftune_noise_alpha == 5.0
        assert config.training.label_smoothing_factor == 0.1


# ===================================================================
# All-Linear Target Modules Tests
# ===================================================================


class TestAllLinearTargetModules:
    """Verify default target_modules includes all linear layers."""

    def test_default_targets_attention(self) -> None:
        """Default LoRA target_modules targets attention projections (safer default)."""
        from llm_forge.config.schema import LoRAConfig

        cfg = LoRAConfig()
        assert cfg.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]

    def test_can_override_modules(self) -> None:
        """Users can restrict target_modules to a subset."""
        from llm_forge.config.schema import LoRAConfig

        cfg = LoRAConfig(target_modules=["q_proj", "v_proj"])
        assert cfg.target_modules == ["q_proj", "v_proj"]
