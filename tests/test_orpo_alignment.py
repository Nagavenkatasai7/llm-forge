"""Tests for ORPO alignment training and AlignmentConfig.

Covers:
- AlignmentConfig schema validation
- TrainingMode enum (orpo, dpo)
- DAG builder alignment stage wiring
- AlignmentTrainer.train_orpo method existence and guard
- YAML config round-trip with alignment settings
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Import guards — system Python may lack heavy deps
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
    from llm_forge.pipeline.dag_builder import DAGBuilder, PipelineStage

    _DAG_AVAILABLE = True
except ImportError:
    _DAG_AVAILABLE = False

try:
    from llm_forge.training.alignment import AlignmentTrainer

    _ALIGNMENT_AVAILABLE = True
except ImportError:
    _ALIGNMENT_AVAILABLE = False


# ===================================================================
# AlignmentConfig schema tests
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema deps missing")
class TestAlignmentConfig:
    """Validate AlignmentConfig Pydantic model."""

    def test_default_alignment_config(self):
        cfg = AlignmentConfig()
        assert cfg.preference_dataset is None
        assert cfg.prompt_field == "prompt"
        assert cfg.chosen_field == "chosen"
        assert cfg.rejected_field == "rejected"
        assert cfg.beta == 0.1
        assert cfg.max_prompt_length == 512
        assert cfg.max_length == 1024
        assert cfg.loss_type == "sigmoid"

    def test_alignment_config_custom_fields(self):
        cfg = AlignmentConfig(
            preference_dataset="Anthropic/hh-rlhf",
            prompt_field="question",
            chosen_field="best",
            rejected_field="worst",
            beta=0.05,
            max_prompt_length=256,
            max_length=2048,
            loss_type="ipo",
        )
        assert cfg.preference_dataset == "Anthropic/hh-rlhf"
        assert cfg.prompt_field == "question"
        assert cfg.chosen_field == "best"
        assert cfg.rejected_field == "worst"
        assert cfg.beta == 0.05
        assert cfg.max_prompt_length == 256
        assert cfg.max_length == 2048
        assert cfg.loss_type == "ipo"

    def test_alignment_beta_must_be_positive(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            AlignmentConfig(beta=0.0)

    def test_alignment_beta_negative_rejected(self):
        with pytest.raises(Exception):
            AlignmentConfig(beta=-0.5)

    def test_alignment_max_prompt_length_minimum(self):
        with pytest.raises(Exception):
            AlignmentConfig(max_prompt_length=32)  # below ge=64

    def test_alignment_max_length_minimum(self):
        with pytest.raises(Exception):
            AlignmentConfig(max_length=64)  # below ge=128

    def test_alignment_loss_type_invalid(self):
        with pytest.raises(Exception):
            AlignmentConfig(loss_type="invalid_loss")

    def test_alignment_loss_type_all_valid(self):
        for loss in ("sigmoid", "hinge", "ipo", "kto_pair"):
            cfg = AlignmentConfig(loss_type=loss)
            assert cfg.loss_type == loss


# ===================================================================
# TrainingMode enum tests
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema deps missing")
class TestTrainingModeORPO:
    """Verify ORPO is a valid training mode."""

    def test_orpo_in_training_mode(self):
        assert TrainingMode.orpo.value == "orpo"

    def test_dpo_in_training_mode(self):
        assert TrainingMode.dpo.value == "dpo"

    def test_orpo_mode_in_full_config(self):
        cfg = LLMForgeConfig.model_validate(
            {
                "model": {"name": "test/model"},
                "data": {"train_path": "test.jsonl"},
                "training": {"mode": "orpo", "output_dir": "/tmp/test"},
            }
        )
        assert cfg.training.mode == TrainingMode.orpo

    def test_dpo_mode_in_full_config(self):
        cfg = LLMForgeConfig.model_validate(
            {
                "model": {"name": "test/model"},
                "data": {"train_path": "test.jsonl"},
                "training": {"mode": "dpo", "output_dir": "/tmp/test"},
            }
        )
        assert cfg.training.mode == TrainingMode.dpo


# ===================================================================
# LLMForgeConfig alignment integration tests
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema deps missing")
class TestAlignmentInMasterConfig:
    """Verify alignment config embeds correctly in LLMForgeConfig."""

    def test_default_alignment_in_config(self):
        cfg = LLMForgeConfig.model_validate(
            {
                "model": {"name": "test/model"},
                "data": {"train_path": "test.jsonl"},
                "training": {"output_dir": "/tmp/test"},
            }
        )
        assert cfg.alignment.preference_dataset is None
        assert cfg.alignment.beta == 0.1

    def test_alignment_with_preference_dataset(self):
        cfg = LLMForgeConfig.model_validate(
            {
                "model": {"name": "test/model"},
                "data": {"train_path": "test.jsonl"},
                "training": {"mode": "dpo", "output_dir": "/tmp/test"},
                "alignment": {
                    "preference_dataset": "Anthropic/hh-rlhf",
                    "beta": 0.2,
                    "loss_type": "hinge",
                },
            }
        )
        assert cfg.alignment.preference_dataset == "Anthropic/hh-rlhf"
        assert cfg.alignment.beta == 0.2
        assert cfg.alignment.loss_type == "hinge"

    def test_orpo_config_with_alignment(self):
        cfg = LLMForgeConfig.model_validate(
            {
                "model": {"name": "test/model"},
                "data": {"train_path": "test.jsonl"},
                "training": {"mode": "orpo", "output_dir": "/tmp/test"},
                "alignment": {
                    "preference_dataset": "HuggingFaceH4/ultrafeedback_binarized",
                    "beta": 0.15,
                    "max_length": 2048,
                },
            }
        )
        assert cfg.training.mode == TrainingMode.orpo
        assert cfg.alignment.preference_dataset == "HuggingFaceH4/ultrafeedback_binarized"
        assert cfg.alignment.max_length == 2048

    def test_yaml_roundtrip_with_alignment(self):
        raw = {
            "model": {"name": "test/model"},
            "data": {"train_path": "test.jsonl"},
            "training": {"mode": "orpo", "output_dir": "/tmp/test"},
            "alignment": {
                "preference_dataset": "some/dataset",
                "beta": 0.3,
            },
        }
        cfg = LLMForgeConfig.model_validate(raw)
        dumped = cfg.model_dump(mode="json")
        yaml_str = yaml.dump(dumped, default_flow_style=False)
        reloaded = yaml.safe_load(yaml_str)
        cfg2 = LLMForgeConfig.model_validate(reloaded)
        assert cfg2.alignment.preference_dataset == "some/dataset"
        assert cfg2.alignment.beta == 0.3
        assert cfg2.training.mode == "orpo"


# ===================================================================
# DAG builder alignment stage tests
# ===================================================================


@pytest.mark.skipif(not _DAG_AVAILABLE, reason="dag_builder deps missing")
class TestAlignmentDAGStage:
    """Verify alignment stage is wired into the DAG builder."""

    def _build_config(self, **overrides):
        base = {
            "model": {"name": "test/model"},
            "data": {"train_path": "test.jsonl"},
            "training": {"output_dir": "/tmp/test"},
        }
        base.update(overrides)
        return LLMForgeConfig.model_validate(base)

    def test_alignment_stage_exists_in_defaults(self):
        builder = DAGBuilder()
        names = [s["name"] for s in builder._DEFAULT_STAGES]
        assert "alignment" in names

    def test_alignment_stage_depends_on_training(self):
        builder = DAGBuilder()
        for s in builder._DEFAULT_STAGES:
            if s["name"] == "alignment":
                assert "training" in s["dependencies"]
                break

    def test_iti_probing_depends_on_alignment(self):
        builder = DAGBuilder()
        for s in builder._DEFAULT_STAGES:
            if s["name"] == "iti_probing":
                assert "alignment" in s["dependencies"]
                break

    def test_alignment_disabled_by_default(self):
        cfg = self._build_config()
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        alignment = next(s for s in stages if s.name == "alignment")
        assert not alignment.enabled

    def test_alignment_disabled_without_preference_dataset(self):
        cfg = self._build_config(
            training={"mode": "dpo", "output_dir": "/tmp/test"},
        )
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        alignment = next(s for s in stages if s.name == "alignment")
        assert not alignment.enabled

    def test_alignment_disabled_for_lora_mode(self):
        cfg = self._build_config(
            training={"mode": "lora", "output_dir": "/tmp/test"},
            alignment={"preference_dataset": "some/dataset"},
        )
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        alignment = next(s for s in stages if s.name == "alignment")
        assert not alignment.enabled

    def test_alignment_enabled_for_dpo(self):
        cfg = self._build_config(
            training={"mode": "dpo", "output_dir": "/tmp/test"},
            alignment={"preference_dataset": "Anthropic/hh-rlhf"},
        )
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        alignment = next(s for s in stages if s.name == "alignment")
        assert alignment.enabled

    def test_alignment_enabled_for_orpo(self):
        cfg = self._build_config(
            training={"mode": "orpo", "output_dir": "/tmp/test"},
            alignment={"preference_dataset": "HuggingFaceH4/ultrafeedback"},
        )
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        alignment = next(s for s in stages if s.name == "alignment")
        assert alignment.enabled

    def test_alignment_config_extraction(self):
        cfg = self._build_config(
            training={"mode": "dpo", "output_dir": "/tmp/test"},
            alignment={
                "preference_dataset": "some/ds",
                "beta": 0.5,
            },
        )
        builder = DAGBuilder()
        extracted = builder._extract_stage_config("alignment", cfg)
        assert extracted["preference_dataset"] == "some/ds"
        assert extracted["beta"] == 0.5

    def test_pipeline_order_alignment_after_training(self):
        cfg = self._build_config(
            training={"mode": "orpo", "output_dir": "/tmp/test"},
            alignment={"preference_dataset": "ds"},
        )
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        names = [s.name for s in stages]
        training_idx = names.index("training")
        alignment_idx = names.index("alignment")
        assert alignment_idx > training_idx

    def test_pipeline_order_alignment_before_iti_probing(self):
        cfg = self._build_config(
            training={"mode": "orpo", "output_dir": "/tmp/test"},
            alignment={"preference_dataset": "ds"},
            iti={"enabled": True},
        )
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        names = [s.name for s in stages]
        alignment_idx = names.index("alignment")
        iti_idx = names.index("iti_probing")
        assert iti_idx > alignment_idx

    def test_full_pipeline_stage_count(self):
        cfg = self._build_config()
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        # 12 stages: data_loading, cleaning, preprocessing,
        # refusal_augmentation, ifd_scoring, training, alignment,
        # iti_probing, iti_baking, model_merging, evaluation, export
        assert len(stages) == 12


# ===================================================================
# AlignmentTrainer.train_orpo method tests
# ===================================================================


@pytest.mark.skipif(not _ALIGNMENT_AVAILABLE, reason="alignment module deps missing")
class TestAlignmentTrainerORPO:
    """Verify train_orpo method exists and has correct interface."""

    def test_train_orpo_method_exists(self):
        assert hasattr(AlignmentTrainer, "train_orpo")

    def test_train_orpo_callable(self):
        assert callable(getattr(AlignmentTrainer, "train_orpo", None))

    def test_train_orpo_signature(self):
        import inspect

        sig = inspect.signature(AlignmentTrainer.train_orpo)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "model" in params
        assert "dataset" in params
        assert "beta" in params
        assert "max_length" in params
        assert "max_prompt_length" in params


# ===================================================================
# Existing configs still validate
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema deps missing")
class TestExistingConfigsValid:
    """Ensure adding alignment didn't break existing configs."""

    def test_minimal_config(self):
        cfg = LLMForgeConfig.model_validate(
            {
                "model": {"name": "test/model"},
                "data": {"train_path": "test.jsonl"},
                "training": {"output_dir": "/tmp/test"},
            }
        )
        assert cfg.alignment.preference_dataset is None

    def test_full_lora_config(self):
        cfg = LLMForgeConfig.model_validate(
            {
                "model": {"name": "test/model", "max_seq_length": 512},
                "data": {
                    "train_path": "test.jsonl",
                    "format": "alpaca",
                    "max_samples": 1000,
                },
                "lora": {"r": 16, "alpha": 32},
                "training": {
                    "mode": "lora",
                    "output_dir": "/tmp/test",
                    "num_epochs": 3,
                    "learning_rate": 2e-4,
                },
                "evaluation": {"enabled": True, "benchmarks": ["hellaswag"]},
                "serving": {"export_format": "safetensors"},
            }
        )
        assert cfg.training.mode == TrainingMode.lora
        assert cfg.alignment.preference_dataset is None
