"""Tests for IFD (Instruction-Following Difficulty) data scoring and filtering.

Covers:
- IFDConfig schema validation
- IFDScorer class interface
- IFDResult dataclass
- DAG builder ifd_scoring stage wiring
- YAML config round-trip
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
    from llm_forge.config.schema import IFDConfig, LLMForgeConfig

    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False

try:
    from llm_forge.data.ifd_scorer import IFDResult, IFDScorer

    _SCORER_AVAILABLE = True
except ImportError:
    _SCORER_AVAILABLE = False

try:
    from llm_forge.pipeline.dag_builder import DAGBuilder

    _DAG_AVAILABLE = True
except ImportError:
    _DAG_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ===================================================================
# IFDConfig schema tests
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema deps missing")
class TestIFDConfig:
    """Validate IFDConfig Pydantic model."""

    def test_defaults(self):
        cfg = IFDConfig()
        assert cfg.enabled is False
        assert cfg.select_ratio == 0.5
        assert cfg.batch_size == 4
        assert cfg.max_length == 512

    def test_custom_values(self):
        cfg = IFDConfig(
            enabled=True,
            select_ratio=0.1,
            batch_size=8,
            max_length=1024,
        )
        assert cfg.enabled is True
        assert cfg.select_ratio == 0.1
        assert cfg.batch_size == 8
        assert cfg.max_length == 1024

    def test_select_ratio_must_be_positive(self):
        with pytest.raises(Exception):
            IFDConfig(select_ratio=0.0)

    def test_select_ratio_must_be_at_most_one(self):
        with pytest.raises(Exception):
            IFDConfig(select_ratio=1.5)

    def test_select_ratio_boundary_one(self):
        cfg = IFDConfig(select_ratio=1.0)
        assert cfg.select_ratio == 1.0

    def test_batch_size_minimum(self):
        with pytest.raises(Exception):
            IFDConfig(batch_size=0)

    def test_max_length_minimum(self):
        with pytest.raises(Exception):
            IFDConfig(max_length=32)

    def test_max_length_boundary(self):
        cfg = IFDConfig(max_length=64)
        assert cfg.max_length == 64


# ===================================================================
# IFDConfig in LLMForgeConfig
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema deps missing")
class TestIFDInMasterConfig:
    """Verify IFD config in the master LLMForgeConfig."""

    def test_ifd_disabled_by_default(self):
        cfg = LLMForgeConfig.model_validate(
            {
                "model": {"name": "test/model"},
                "data": {"train_path": "test.jsonl"},
                "training": {"output_dir": "/tmp/test"},
            }
        )
        assert cfg.ifd.enabled is False

    def test_ifd_enabled(self):
        cfg = LLMForgeConfig.model_validate(
            {
                "model": {"name": "test/model"},
                "data": {"train_path": "test.jsonl"},
                "training": {"output_dir": "/tmp/test"},
                "ifd": {"enabled": True, "select_ratio": 0.2},
            }
        )
        assert cfg.ifd.enabled is True
        assert cfg.ifd.select_ratio == 0.2

    def test_yaml_roundtrip(self):
        raw = {
            "model": {"name": "test/model"},
            "data": {"train_path": "test.jsonl"},
            "training": {"output_dir": "/tmp/test"},
            "ifd": {"enabled": True, "select_ratio": 0.3, "batch_size": 16},
        }
        cfg = LLMForgeConfig.model_validate(raw)
        dumped = cfg.model_dump(mode="json")
        reloaded = yaml.safe_load(yaml.dump(dumped))
        cfg2 = LLMForgeConfig.model_validate(reloaded)
        assert cfg2.ifd.enabled is True
        assert cfg2.ifd.select_ratio == 0.3
        assert cfg2.ifd.batch_size == 16


# ===================================================================
# IFDScorer class tests
# ===================================================================


@pytest.mark.skipif(not _SCORER_AVAILABLE, reason="ifd_scorer deps missing")
class TestIFDScorer:
    """Test IFDScorer class interface."""

    def test_class_exists(self):
        assert IFDScorer is not None

    def test_instantiation(self):
        scorer = IFDScorer(max_length=256, batch_size=2)
        assert scorer.max_length == 256
        assert scorer.batch_size == 2

    def test_default_params(self):
        scorer = IFDScorer()
        assert scorer.max_length == 512
        assert scorer.batch_size == 4

    def test_score_dataset_method_exists(self):
        assert hasattr(IFDScorer, "score_dataset")
        assert callable(IFDScorer.score_dataset)

    def test_filter_by_ifd_method_exists(self):
        assert hasattr(IFDScorer, "filter_by_ifd")
        assert callable(IFDScorer.filter_by_ifd)


@pytest.mark.skipif(not _SCORER_AVAILABLE, reason="ifd_scorer deps missing")
class TestIFDResult:
    """Test IFDResult dataclass."""

    def test_default_result(self):
        result = IFDResult()
        assert result.scores == []
        assert result.conditioned_losses == []
        assert result.direct_losses == []
        assert result.num_scored == 0
        assert result.mean_ifd == 0.0
        assert result.median_ifd == 0.0

    def test_custom_result(self):
        result = IFDResult(
            scores=[0.8, 1.2, 0.5],
            conditioned_losses=[1.5, 2.0, 1.0],
            direct_losses=[1.875, 1.667, 2.0],
            num_scored=3,
            mean_ifd=0.833,
            median_ifd=0.8,
        )
        assert len(result.scores) == 3
        assert result.num_scored == 3


@pytest.mark.skipif(
    not _SCORER_AVAILABLE or not _TORCH_AVAILABLE,
    reason="torch or ifd_scorer deps missing",
)
class TestIFDScorerRequiresTorchModel:
    """Verify IFD scorer requires proper model/tokenizer."""

    def test_score_dataset_requires_torch(self):
        scorer = IFDScorer()
        # Should raise when given non-model objects
        with pytest.raises((AttributeError, TypeError)):
            scorer.score_dataset(
                model=None,
                tokenizer=None,
                instructions=["test"],
                responses=["answer"],
            )

    def test_score_dataset_length_mismatch(self):
        scorer = IFDScorer()
        with pytest.raises(AssertionError):
            scorer.score_dataset(
                model=object(),
                tokenizer=object(),
                instructions=["a", "b"],
                responses=["c"],
            )


# ===================================================================
# DAG builder IFD stage tests
# ===================================================================


@pytest.mark.skipif(not _DAG_AVAILABLE, reason="dag_builder deps missing")
class TestIFDDAGStage:
    """Verify ifd_scoring stage is wired into the DAG builder."""

    def _build_config(self, **overrides):
        base = {
            "model": {"name": "test/model"},
            "data": {"train_path": "test.jsonl"},
            "training": {"output_dir": "/tmp/test"},
        }
        base.update(overrides)
        return LLMForgeConfig.model_validate(base)

    def test_ifd_stage_exists_in_defaults(self):
        builder = DAGBuilder()
        names = [s["name"] for s in builder._DEFAULT_STAGES]
        assert "ifd_scoring" in names

    def test_ifd_stage_depends_on_refusal_augmentation(self):
        builder = DAGBuilder()
        for s in builder._DEFAULT_STAGES:
            if s["name"] == "ifd_scoring":
                assert "refusal_augmentation" in s["dependencies"]
                break

    def test_training_depends_on_ifd(self):
        builder = DAGBuilder()
        for s in builder._DEFAULT_STAGES:
            if s["name"] == "training":
                assert "ifd_scoring" in s["dependencies"]
                break

    def test_ifd_disabled_by_default(self):
        cfg = self._build_config()
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        ifd = next(s for s in stages if s.name == "ifd_scoring")
        assert not ifd.enabled

    def test_ifd_enabled_when_configured(self):
        cfg = self._build_config(ifd={"enabled": True, "select_ratio": 0.3})
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        ifd = next(s for s in stages if s.name == "ifd_scoring")
        assert ifd.enabled

    def test_ifd_config_extraction(self):
        cfg = self._build_config(ifd={"enabled": True, "select_ratio": 0.2, "batch_size": 8})
        builder = DAGBuilder()
        extracted = builder._extract_stage_config("ifd_scoring", cfg)
        assert extracted["enabled"] is True
        assert extracted["select_ratio"] == 0.2
        assert extracted["batch_size"] == 8

    def test_pipeline_order_ifd_after_refusal(self):
        cfg = self._build_config(
            ifd={"enabled": True},
            refusal={"enabled": True},
        )
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        names = [s.name for s in stages]
        refusal_idx = names.index("refusal_augmentation")
        ifd_idx = names.index("ifd_scoring")
        assert ifd_idx > refusal_idx

    def test_pipeline_order_ifd_before_training(self):
        cfg = self._build_config(ifd={"enabled": True})
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        names = [s.name for s in stages]
        ifd_idx = names.index("ifd_scoring")
        train_idx = names.index("training")
        assert train_idx > ifd_idx

    def test_total_stage_count(self):
        cfg = self._build_config()
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        assert len(stages) == 12


# ===================================================================
# Existing configs still valid
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema deps missing")
class TestExistingConfigsWithIFD:
    """Ensure adding IFDConfig didn't break existing configs."""

    def test_minimal_config_still_valid(self):
        cfg = LLMForgeConfig.model_validate(
            {
                "model": {"name": "test/model"},
                "data": {"train_path": "test.jsonl"},
                "training": {"output_dir": "/tmp/test"},
            }
        )
        assert cfg.ifd.enabled is False

    def test_project_configs_validate(self):
        configs_dir = Path(__file__).resolve().parents[1] / "configs"
        if not configs_dir.exists():
            pytest.skip("configs directory not found")
        configs = list(configs_dir.glob("*.yaml"))
        if not configs:
            pytest.skip("no config files found")
        for c in configs:
            with open(c) as f:
                raw = yaml.safe_load(f)
            LLMForgeConfig.model_validate(raw)
