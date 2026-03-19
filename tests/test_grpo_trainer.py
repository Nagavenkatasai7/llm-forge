"""Tests for GRPO (Group Relative Policy Optimisation) trainer.

Covers:
- TrainingMode.grpo in schema
- AlignmentConfig GRPO-specific fields (num_generations, max_completion_length)
- AlignmentTrainer.train_grpo method
- DAG builder alignment stage routing to GRPO
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Import guards
# ---------------------------------------------------------------------------

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from llm_forge.config.schema import (
        AlignmentConfig,
        LLMForgeConfig,
        TrainingMode,
    )

    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False

try:
    from llm_forge.pipeline.dag_builder import DAGBuilder

    _DAG_AVAILABLE = True
except ImportError:
    _DAG_AVAILABLE = False

try:
    from llm_forge.training.alignment import AlignmentTrainer

    _ALIGNMENT_AVAILABLE = True
except ImportError:
    _ALIGNMENT_AVAILABLE = False


# ===================================================================
# Schema tests
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema deps missing")
class TestGRPOInTrainingMode:
    """Verify GRPO is a valid training mode."""

    def test_grpo_exists(self):
        assert TrainingMode.grpo.value == "grpo"

    def test_grpo_in_full_config(self):
        cfg = LLMForgeConfig.model_validate(
            {
                "model": {"name": "test/model"},
                "data": {"train_path": "test.jsonl"},
                "training": {"mode": "grpo", "output_dir": "/tmp/test"},
            }
        )
        assert cfg.training.mode == TrainingMode.grpo


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema deps missing")
class TestGRPOAlignmentConfig:
    """Verify GRPO-specific fields in AlignmentConfig."""

    def test_num_generations_default(self):
        cfg = AlignmentConfig()
        assert cfg.num_generations == 4

    def test_max_completion_length_default(self):
        cfg = AlignmentConfig()
        assert cfg.max_completion_length == 256

    def test_num_generations_custom(self):
        cfg = AlignmentConfig(num_generations=8)
        assert cfg.num_generations == 8

    def test_num_generations_minimum(self):
        with pytest.raises(Exception):
            AlignmentConfig(num_generations=1)  # must be >= 2

    def test_max_completion_length_minimum(self):
        with pytest.raises(Exception):
            AlignmentConfig(max_completion_length=8)  # must be >= 16

    def test_grpo_config_in_master(self):
        cfg = LLMForgeConfig.model_validate(
            {
                "model": {"name": "test/model"},
                "data": {"train_path": "test.jsonl"},
                "training": {"mode": "grpo", "output_dir": "/tmp/test"},
                "alignment": {
                    "preference_dataset": "some/ds",
                    "num_generations": 6,
                    "max_completion_length": 512,
                    "beta": 0.2,
                },
            }
        )
        assert cfg.alignment.num_generations == 6
        assert cfg.alignment.max_completion_length == 512
        assert cfg.alignment.beta == 0.2

    def test_yaml_roundtrip(self):
        raw = {
            "model": {"name": "test/model"},
            "data": {"train_path": "test.jsonl"},
            "training": {"mode": "grpo", "output_dir": "/tmp/test"},
            "alignment": {
                "preference_dataset": "ds",
                "num_generations": 4,
                "max_completion_length": 128,
            },
        }
        cfg = LLMForgeConfig.model_validate(raw)
        dumped = cfg.model_dump(mode="json")
        reloaded = yaml.safe_load(yaml.dump(dumped))
        cfg2 = LLMForgeConfig.model_validate(reloaded)
        assert cfg2.training.mode == "grpo"
        assert cfg2.alignment.num_generations == 4
        assert cfg2.alignment.max_completion_length == 128


# ===================================================================
# DAG builder tests
# ===================================================================


@pytest.mark.skipif(not _DAG_AVAILABLE, reason="dag_builder deps missing")
class TestGRPODAGStage:
    """Verify alignment stage handles GRPO mode."""

    def _build_config(self, **overrides):
        base = {
            "model": {"name": "test/model"},
            "data": {"train_path": "test.jsonl"},
            "training": {"output_dir": "/tmp/test"},
        }
        base.update(overrides)
        return LLMForgeConfig.model_validate(base)

    def test_alignment_enabled_for_grpo(self):
        cfg = self._build_config(
            training={"mode": "grpo", "output_dir": "/tmp/test"},
            alignment={"preference_dataset": "some/ds"},
        )
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        alignment = next(s for s in stages if s.name == "alignment")
        assert alignment.enabled

    def test_alignment_disabled_for_grpo_without_dataset(self):
        cfg = self._build_config(
            training={"mode": "grpo", "output_dir": "/tmp/test"},
        )
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        alignment = next(s for s in stages if s.name == "alignment")
        assert not alignment.enabled


# ===================================================================
# AlignmentTrainer.train_grpo method tests
# ===================================================================


@pytest.mark.skipif(not _ALIGNMENT_AVAILABLE, reason="alignment module deps missing")
class TestAlignmentTrainerGRPO:
    """Verify train_grpo method exists and has correct interface."""

    def test_train_grpo_method_exists(self):
        assert hasattr(AlignmentTrainer, "train_grpo")

    def test_train_grpo_callable(self):
        assert callable(getattr(AlignmentTrainer, "train_grpo", None))

    def test_train_grpo_signature(self):
        import inspect

        sig = inspect.signature(AlignmentTrainer.train_grpo)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "model" in params
        assert "dataset" in params
        assert "num_generations" in params
        assert "max_completion_length" in params
        assert "beta" in params


# ===================================================================
# Existing configs still valid
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema deps missing")
class TestExistingConfigsWithGRPO:
    """Ensure adding GRPO didn't break existing configs."""

    def test_all_modes_still_valid(self):
        for mode in ("lora", "qlora", "full", "pretrain", "dpo", "orpo", "grpo"):
            cfg = LLMForgeConfig.model_validate(
                {
                    "model": {"name": "test/model"},
                    "data": {"train_path": "test.jsonl"},
                    "training": {"mode": mode, "output_dir": "/tmp/test"},
                }
            )
            assert cfg.training.mode == mode

    def test_project_configs_validate(self):
        configs_dir = Path(__file__).resolve().parents[1] / "configs"
        if not configs_dir.exists():
            pytest.skip("configs directory not found")
        for c in configs_dir.glob("*.yaml"):
            with open(c) as f:
                raw = yaml.safe_load(f)
            LLMForgeConfig.model_validate(raw)
