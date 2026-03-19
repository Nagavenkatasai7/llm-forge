"""Tests for model merging: Linear, SLERP, TIES.

Covers:
- MergeConfig schema validation
- ModelMerger class methods
- Merge algorithm correctness (linear, slerp, ties)
- DAG builder model_merging stage wiring
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
    from llm_forge.config.schema import LLMForgeConfig, MergeConfig

    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False

try:
    from llm_forge.serving.model_merger import ModelMerger

    _MERGER_AVAILABLE = True
except ImportError:
    _MERGER_AVAILABLE = False

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
# MergeConfig schema tests
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema deps missing")
class TestMergeConfig:
    """Validate MergeConfig Pydantic model."""

    def test_defaults(self):
        cfg = MergeConfig()
        assert cfg.enabled is False
        assert cfg.method == "linear"
        assert cfg.models == []
        assert cfg.weights == []
        assert cfg.base_model is None
        assert cfg.slerp_t == 0.5
        assert cfg.ties_density == 0.5
        assert cfg.output_path is None

    def test_custom_linear(self):
        cfg = MergeConfig(
            enabled=True,
            method="linear",
            models=["model_a", "model_b"],
            weights=[0.6, 0.4],
        )
        assert cfg.enabled
        assert len(cfg.models) == 2
        assert cfg.weights == [0.6, 0.4]

    def test_custom_slerp(self):
        cfg = MergeConfig(
            enabled=True,
            method="slerp",
            models=["a", "b"],
            slerp_t=0.3,
        )
        assert cfg.method == "slerp"
        assert cfg.slerp_t == 0.3

    def test_custom_ties(self):
        cfg = MergeConfig(
            enabled=True,
            method="ties",
            models=["a", "b"],
            base_model="base",
            ties_density=0.3,
        )
        assert cfg.method == "ties"
        assert cfg.base_model == "base"
        assert cfg.ties_density == 0.3

    def test_invalid_method(self):
        with pytest.raises(Exception):
            MergeConfig(method="invalid")

    def test_slerp_t_bounds(self):
        MergeConfig(slerp_t=0.0)  # should work
        MergeConfig(slerp_t=1.0)  # should work
        with pytest.raises(Exception):
            MergeConfig(slerp_t=-0.1)
        with pytest.raises(Exception):
            MergeConfig(slerp_t=1.1)

    def test_ties_density_bounds(self):
        MergeConfig(ties_density=1.0)  # should work
        with pytest.raises(Exception):
            MergeConfig(ties_density=0.0)
        with pytest.raises(Exception):
            MergeConfig(ties_density=1.5)

    def test_in_master_config(self):
        cfg = LLMForgeConfig.model_validate(
            {
                "model": {"name": "test/model"},
                "data": {"train_path": "test.jsonl"},
                "training": {"output_dir": "/tmp/test"},
                "merge": {
                    "enabled": True,
                    "method": "slerp",
                    "models": ["a", "b"],
                },
            }
        )
        assert cfg.merge.enabled
        assert cfg.merge.method == "slerp"

    def test_disabled_by_default(self):
        cfg = LLMForgeConfig.model_validate(
            {
                "model": {"name": "test/model"},
                "data": {"train_path": "test.jsonl"},
                "training": {"output_dir": "/tmp/test"},
            }
        )
        assert cfg.merge.enabled is False

    def test_yaml_roundtrip(self):
        raw = {
            "model": {"name": "test/model"},
            "data": {"train_path": "test.jsonl"},
            "training": {"output_dir": "/tmp/test"},
            "merge": {
                "enabled": True,
                "method": "ties",
                "models": ["a", "b", "c"],
                "base_model": "base",
                "ties_density": 0.3,
            },
        }
        cfg = LLMForgeConfig.model_validate(raw)
        dumped = cfg.model_dump(mode="json")
        reloaded = yaml.safe_load(yaml.dump(dumped))
        cfg2 = LLMForgeConfig.model_validate(reloaded)
        assert cfg2.merge.method == "ties"
        assert cfg2.merge.ties_density == 0.3
        assert len(cfg2.merge.models) == 3


# ===================================================================
# ModelMerger class tests
# ===================================================================


@pytest.mark.skipif(not _MERGER_AVAILABLE, reason="merger deps missing")
class TestModelMergerInterface:
    """Test ModelMerger class interface exists."""

    def test_class_exists(self):
        assert ModelMerger is not None

    def test_merge_linear_exists(self):
        assert hasattr(ModelMerger, "merge_linear")
        assert callable(ModelMerger.merge_linear)

    def test_merge_slerp_exists(self):
        assert hasattr(ModelMerger, "merge_slerp")
        assert callable(ModelMerger.merge_slerp)

    def test_merge_ties_exists(self):
        assert hasattr(ModelMerger, "merge_ties")
        assert callable(ModelMerger.merge_ties)

    def test_merge_models_exists(self):
        assert hasattr(ModelMerger, "merge_models")
        assert callable(ModelMerger.merge_models)


@pytest.mark.skipif(
    not _MERGER_AVAILABLE or not _TORCH_AVAILABLE,
    reason="torch or merger deps missing",
)
class TestMergeLinear:
    """Test linear merge algorithm."""

    def test_equal_weights(self):
        sd_a = {"w": torch.tensor([1.0, 2.0, 3.0])}
        sd_b = {"w": torch.tensor([3.0, 2.0, 1.0])}
        merged = ModelMerger.merge_linear([sd_a, sd_b])
        expected = torch.tensor([2.0, 2.0, 2.0])
        assert torch.allclose(merged["w"], expected)

    def test_custom_weights(self):
        sd_a = {"w": torch.tensor([0.0, 0.0])}
        sd_b = {"w": torch.tensor([10.0, 10.0])}
        merged = ModelMerger.merge_linear([sd_a, sd_b], weights=[0.3, 0.7])
        expected = torch.tensor([7.0, 7.0])
        assert torch.allclose(merged["w"], expected)

    def test_three_models(self):
        sd_a = {"w": torch.tensor([3.0])}
        sd_b = {"w": torch.tensor([6.0])}
        sd_c = {"w": torch.tensor([9.0])}
        merged = ModelMerger.merge_linear([sd_a, sd_b, sd_c])
        expected = torch.tensor([6.0])
        assert torch.allclose(merged["w"], expected)

    def test_single_model(self):
        sd = {"w": torch.tensor([5.0, 10.0])}
        merged = ModelMerger.merge_linear([sd])
        assert torch.allclose(merged["w"], sd["w"])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ModelMerger.merge_linear([])


@pytest.mark.skipif(
    not _MERGER_AVAILABLE or not _TORCH_AVAILABLE,
    reason="torch or merger deps missing",
)
class TestMergeSLERP:
    """Test SLERP merge algorithm."""

    def test_t_zero_returns_first(self):
        sd_a = {"w": torch.tensor([1.0, 0.0])}
        sd_b = {"w": torch.tensor([0.0, 1.0])}
        merged = ModelMerger.merge_slerp(sd_a, sd_b, t=0.0)
        assert torch.allclose(merged["w"], sd_a["w"], atol=1e-5)

    def test_t_one_returns_second(self):
        sd_a = {"w": torch.tensor([1.0, 0.0])}
        sd_b = {"w": torch.tensor([0.0, 1.0])}
        merged = ModelMerger.merge_slerp(sd_a, sd_b, t=1.0)
        assert torch.allclose(merged["w"], sd_b["w"], atol=1e-5)

    def test_t_half_midpoint(self):
        sd_a = {"w": torch.tensor([1.0, 0.0])}
        sd_b = {"w": torch.tensor([0.0, 1.0])}
        merged = ModelMerger.merge_slerp(sd_a, sd_b, t=0.5)
        # SLERP midpoint on unit circle: both components ~0.707
        assert merged["w"][0].item() == pytest.approx(0.707, abs=0.01)
        assert merged["w"][1].item() == pytest.approx(0.707, abs=0.01)

    def test_parallel_vectors_fallback(self):
        """Near-parallel vectors should fall back to linear interpolation."""
        sd_a = {"w": torch.tensor([1.0, 1.0])}
        sd_b = {"w": torch.tensor([2.0, 2.0])}
        merged = ModelMerger.merge_slerp(sd_a, sd_b, t=0.5)
        expected = torch.tensor([1.5, 1.5])
        assert torch.allclose(merged["w"], expected, atol=0.01)


@pytest.mark.skipif(
    not _MERGER_AVAILABLE or not _TORCH_AVAILABLE,
    reason="torch or merger deps missing",
)
class TestMergeTIES:
    """Test TIES merge algorithm."""

    def test_basic_ties(self):
        base = {"w": torch.tensor([0.0, 0.0, 0.0, 0.0])}
        ft_a = {"w": torch.tensor([1.0, -0.5, 0.01, 0.0])}
        ft_b = {"w": torch.tensor([0.8, -0.3, 0.02, 0.0])}
        merged = ModelMerger.merge_ties(base, [ft_a, ft_b], density=1.0)
        # All params kept (density=1.0), both agree on sign
        assert merged["w"][0].item() > 0  # both positive
        assert merged["w"][1].item() < 0  # both negative

    def test_trim_zeros_small_values(self):
        base = {"w": torch.zeros(100)}
        ft = {"w": torch.randn(100)}
        merged = ModelMerger.merge_ties(base, [ft], density=0.1)
        # With density=0.1, ~90% should be close to base (0)
        zero_count = (merged["w"].abs() < 1e-6).sum().item()
        assert zero_count >= 80  # at least 80 of 100 trimmed

    def test_ties_returns_base_for_zero_deltas(self):
        base = {"w": torch.tensor([5.0, 10.0])}
        ft = {"w": torch.tensor([5.0, 10.0])}  # no change
        merged = ModelMerger.merge_ties(base, [ft], density=0.5)
        assert torch.allclose(merged["w"], base["w"])

    def test_ties_empty_finetuned_raises(self):
        with pytest.raises(ValueError):
            ModelMerger.merge_ties({"w": torch.zeros(4)}, [])


# ===================================================================
# DAG builder model_merging stage tests
# ===================================================================


@pytest.mark.skipif(not _DAG_AVAILABLE, reason="dag_builder deps missing")
class TestMergingDAGStage:
    """Verify model_merging stage is wired into the DAG builder."""

    def _build_config(self, **overrides):
        base = {
            "model": {"name": "test/model"},
            "data": {"train_path": "test.jsonl"},
            "training": {"output_dir": "/tmp/test"},
        }
        base.update(overrides)
        return LLMForgeConfig.model_validate(base)

    def test_stage_exists(self):
        builder = DAGBuilder()
        names = [s["name"] for s in builder._DEFAULT_STAGES]
        assert "model_merging" in names

    def test_stage_depends_on_iti_baking(self):
        builder = DAGBuilder()
        for s in builder._DEFAULT_STAGES:
            if s["name"] == "model_merging":
                assert "iti_baking" in s["dependencies"]
                break

    def test_evaluation_depends_on_training(self):
        """Evaluation depends on training (not model_merging, which is optional)."""
        builder = DAGBuilder()
        for s in builder._DEFAULT_STAGES:
            if s["name"] == "evaluation":
                assert "training" in s["dependencies"]
                assert "model_merging" not in s["dependencies"]
                break

    def test_disabled_by_default(self):
        cfg = self._build_config()
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        merging = next(s for s in stages if s.name == "model_merging")
        assert not merging.enabled

    def test_disabled_without_enough_models(self):
        cfg = self._build_config(
            merge={"enabled": True, "models": ["only_one"]},
        )
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        merging = next(s for s in stages if s.name == "model_merging")
        assert not merging.enabled

    def test_enabled_with_two_models(self):
        cfg = self._build_config(
            merge={
                "enabled": True,
                "method": "slerp",
                "models": ["a", "b"],
            },
        )
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        merging = next(s for s in stages if s.name == "model_merging")
        assert merging.enabled

    def test_config_extraction(self):
        cfg = self._build_config(
            merge={
                "enabled": True,
                "method": "ties",
                "models": ["a", "b"],
                "ties_density": 0.3,
            },
        )
        builder = DAGBuilder()
        extracted = builder._extract_stage_config("model_merging", cfg)
        assert extracted["method"] == "ties"
        assert extracted["ties_density"] == 0.3

    def test_pipeline_order(self):
        cfg = self._build_config(
            merge={"enabled": True, "models": ["a", "b"]},
        )
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        names = [s.name for s in stages]
        merge_idx = names.index("model_merging")
        eval_idx = names.index("evaluation")
        export_idx = names.index("export")
        assert eval_idx > merge_idx
        assert export_idx > merge_idx

    def test_total_stage_count(self):
        cfg = self._build_config()
        builder = DAGBuilder()
        stages = builder.build_dag(cfg)
        assert len(stages) == 12
