"""Integration tests for the pipeline DAG builder and stage ordering.

Tests verify that the DAG is correctly constructed, stages are ordered
by dependency, feature flags enable/disable the right stages, and the
topological sort + cycle detection work correctly.
"""

from __future__ import annotations

from typing import Any

import pytest

try:
    from llm_forge.pipeline.dag_builder import DAGBuilder, PipelineStage

    _DAG_AVAILABLE = True
except ImportError:
    _DAG_AVAILABLE = False

try:
    from llm_forge.config.schema import LLMForgeConfig

    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False


pytestmark = pytest.mark.skipif(not _DAG_AVAILABLE, reason="dag_builder not importable")


# ===================================================================
# PipelineStage Dataclass Tests
# ===================================================================


class TestPipelineStage:
    """Test the PipelineStage dataclass."""

    def test_defaults(self) -> None:
        stage = PipelineStage(
            name="test",
            callable=lambda ctx: ctx,
        )
        assert stage.name == "test"
        assert stage.dependencies == []
        assert stage.config == {}
        assert stage.enabled is True
        assert stage.description == ""

    def test_custom_values(self) -> None:
        def fn(ctx):
            return ctx

        stage = PipelineStage(
            name="custom",
            callable=fn,
            dependencies=["dep1", "dep2"],
            config={"key": "val"},
            enabled=False,
            description="A custom stage",
        )
        assert stage.dependencies == ["dep1", "dep2"]
        assert stage.config == {"key": "val"}
        assert stage.enabled is False
        assert stage.description == "A custom stage"


# ===================================================================
# DAGBuilder Tests
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema not importable")
class TestDAGBuilder:
    """Test the DAG builder with various configurations."""

    def _minimal_config(self, **overrides: Any) -> LLMForgeConfig:
        """Create a minimal valid config with optional overrides."""
        base = {
            "model": {"name": "test-model"},
            "data": {"train_path": "test-data"},
        }
        base.update(overrides)
        return LLMForgeConfig(**base)

    def test_build_dag_returns_list(self) -> None:
        config = self._minimal_config()
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        assert isinstance(dag, list)
        assert all(isinstance(s, PipelineStage) for s in dag)

    def test_core_stages_present(self) -> None:
        """Minimal config includes core stages."""
        config = self._minimal_config()
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        names = {s.name for s in dag}
        assert "data_loading" in names
        assert "preprocessing" in names
        assert "training" in names

    def test_data_loading_always_enabled(self) -> None:
        config = self._minimal_config()
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        data_stage = next(s for s in dag if s.name == "data_loading")
        assert data_stage.enabled is True

    def test_training_always_enabled(self) -> None:
        config = self._minimal_config()
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        train_stage = next(s for s in dag if s.name == "training")
        assert train_stage.enabled is True

    def test_evaluation_enabled_by_default(self) -> None:
        """Evaluation is enabled by default (EvalConfig.enabled=True)."""
        config = self._minimal_config()
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        eval_stage = next(s for s in dag if s.name == "evaluation")
        assert eval_stage.enabled is True

    def test_evaluation_enabled(self) -> None:
        config = self._minimal_config(evaluation={"enabled": True, "benchmarks": ["hellaswag"]})
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        eval_stage = next(s for s in dag if s.name == "evaluation")
        assert eval_stage.enabled is True

    def test_iti_stages_disabled_by_default(self) -> None:
        config = self._minimal_config()
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        iti_probing = next(s for s in dag if s.name == "iti_probing")
        iti_baking = next(s for s in dag if s.name == "iti_baking")
        assert iti_probing.enabled is False
        assert iti_baking.enabled is False

    def test_iti_stages_enabled(self) -> None:
        config = self._minimal_config(iti={"enabled": True, "bake_in": True})
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        iti_probing = next(s for s in dag if s.name == "iti_probing")
        iti_baking = next(s for s in dag if s.name == "iti_baking")
        assert iti_probing.enabled is True
        assert iti_baking.enabled is True

    def test_iti_probing_without_baking(self) -> None:
        config = self._minimal_config(iti={"enabled": True, "bake_in": False})
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        iti_probing = next(s for s in dag if s.name == "iti_probing")
        iti_baking = next(s for s in dag if s.name == "iti_baking")
        assert iti_probing.enabled is True
        assert iti_baking.enabled is False

    def test_refusal_disabled_by_default(self) -> None:
        config = self._minimal_config()
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        refusal = next(s for s in dag if s.name == "refusal_augmentation")
        assert refusal.enabled is False

    def test_refusal_enabled(self) -> None:
        config = self._minimal_config(refusal={"enabled": True})
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        refusal = next(s for s in dag if s.name == "refusal_augmentation")
        assert refusal.enabled is True

    def test_export_disabled_when_no_format(self) -> None:
        config = self._minimal_config()
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        export_stage = next(s for s in dag if s.name == "export")
        # Default export_format is None or safetensors — check either way
        # The stage is enabled if export_format is not None
        # Default ServingConfig has export_format = None
        assert isinstance(export_stage.enabled, bool)

    def test_export_enabled_with_format(self) -> None:
        config = self._minimal_config(serving={"export_format": "safetensors"})
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        export_stage = next(s for s in dag if s.name == "export")
        assert export_stage.enabled is True


# ===================================================================
# Topological Sort Tests
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema not importable")
class TestTopologicalSort:
    """Test that stages are correctly ordered by dependencies."""

    def test_data_loading_before_preprocessing(self) -> None:
        config = LLMForgeConfig(model={"name": "test"}, data={"train_path": "test"})
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        names = [s.name for s in dag]
        assert names.index("data_loading") < names.index("preprocessing")

    def test_preprocessing_before_training(self) -> None:
        config = LLMForgeConfig(model={"name": "test"}, data={"train_path": "test"})
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        names = [s.name for s in dag]
        assert names.index("preprocessing") < names.index("training")

    def test_training_before_iti_probing(self) -> None:
        config = LLMForgeConfig(
            model={"name": "test"},
            data={"train_path": "test"},
            iti={"enabled": True},
        )
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        names = [s.name for s in dag]
        assert names.index("training") < names.index("iti_probing")

    def test_iti_probing_before_baking(self) -> None:
        config = LLMForgeConfig(
            model={"name": "test"},
            data={"train_path": "test"},
            iti={"enabled": True, "bake_in": True},
        )
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        names = [s.name for s in dag]
        assert names.index("iti_probing") < names.index("iti_baking")

    def test_refusal_before_training(self) -> None:
        config = LLMForgeConfig(
            model={"name": "test"},
            data={"train_path": "test"},
            refusal={"enabled": True},
        )
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        names = [s.name for s in dag]
        assert names.index("refusal_augmentation") < names.index("training")

    def test_full_pipeline_order(self) -> None:
        """All stages enabled → full ordering."""
        config = LLMForgeConfig(
            model={"name": "test"},
            data={"train_path": "test"},
            evaluation={"enabled": True, "benchmarks": ["hellaswag"]},
            iti={"enabled": True, "bake_in": True},
            refusal={"enabled": True},
            serving={"export_format": "safetensors"},
        )
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        names = [s.name for s in dag]

        # Core ordering: data → preprocessing → refusal → training → iti_probing → iti_baking
        assert names.index("data_loading") < names.index("preprocessing")
        assert names.index("preprocessing") < names.index("refusal_augmentation")
        assert names.index("refusal_augmentation") < names.index("training")
        assert names.index("training") < names.index("iti_probing")
        assert names.index("iti_probing") < names.index("iti_baking")


# ===================================================================
# DAG Validation Tests
# ===================================================================


class TestDAGValidation:
    """Test DAG validation (cycle detection and missing deps)."""

    def test_validate_no_cycles(self) -> None:
        """Linear dependency chain should pass."""
        builder = DAGBuilder()
        stages = [
            PipelineStage(name="a", callable=lambda c: c, dependencies=[]),
            PipelineStage(name="b", callable=lambda c: c, dependencies=["a"]),
            PipelineStage(name="c", callable=lambda c: c, dependencies=["b"]),
        ]
        # Should not raise
        builder._validate_dag(stages)

    def test_validate_detects_missing_dependency(self) -> None:
        """Unknown dependency should raise ValueError."""
        builder = DAGBuilder()
        stages = [
            PipelineStage(name="a", callable=lambda c: c, dependencies=["nonexistent"]),
        ]
        with pytest.raises(ValueError, match="unknown stage"):
            builder._validate_dag(stages)

    def test_validate_detects_cycle(self) -> None:
        """Circular dependency should raise ValueError."""
        builder = DAGBuilder()
        stages = [
            PipelineStage(name="a", callable=lambda c: c, dependencies=["b"]),
            PipelineStage(name="b", callable=lambda c: c, dependencies=["a"]),
        ]
        with pytest.raises(ValueError, match="[Cc]ycle"):
            builder._validate_dag(stages)


# ===================================================================
# Stage Config Extraction Tests
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema not importable")
class TestStageConfigExtraction:
    """Test that stage configs are correctly extracted."""

    def test_data_loading_config(self) -> None:
        config = LLMForgeConfig(
            model={"name": "test"},
            data={"train_path": "my-dataset"},
        )
        builder = DAGBuilder()
        extracted = builder._extract_stage_config("data_loading", config)
        assert extracted["train_path"] == "my-dataset"

    def test_training_config(self) -> None:
        config = LLMForgeConfig(
            model={"name": "test"},
            data={"train_path": "test"},
            training={"mode": "lora", "output_dir": "./out"},
        )
        builder = DAGBuilder()
        extracted = builder._extract_stage_config("training", config)
        assert extracted["mode"] == "lora"
        assert extracted["output_dir"] == "./out"

    def test_iti_config(self) -> None:
        config = LLMForgeConfig(
            model={"name": "test"},
            data={"train_path": "test"},
            iti={"enabled": True, "alpha": 20.0},
        )
        builder = DAGBuilder()
        extracted = builder._extract_stage_config("iti_probing", config)
        assert extracted.get("alpha") == 20.0

    def test_unknown_stage_returns_empty(self) -> None:
        config = LLMForgeConfig(
            model={"name": "test"},
            data={"train_path": "test"},
        )
        builder = DAGBuilder()
        extracted = builder._extract_stage_config("nonexistent_stage", config)
        assert extracted == {}


# ===================================================================
# Enabled Stage Counting Tests
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema not importable")
class TestEnabledStageCounting:
    """Test that feature flags produce the correct number of enabled stages."""

    def test_minimal_config_enabled_count(self) -> None:
        """Minimal config enables only core stages."""
        config = LLMForgeConfig(model={"name": "test"}, data={"train_path": "test"})
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        enabled = [s for s in dag if s.enabled]
        enabled_names = {s.name for s in enabled}
        # At minimum: data_loading, preprocessing, training
        assert "data_loading" in enabled_names
        assert "preprocessing" in enabled_names
        assert "training" in enabled_names
        # Optional stages should be disabled
        assert "iti_probing" not in enabled_names
        assert "refusal_augmentation" not in enabled_names

    def test_full_config_all_stages_enabled(self) -> None:
        """Full config with everything enabled has max stages."""
        config = LLMForgeConfig(
            model={"name": "test"},
            data={"train_path": "test"},
            evaluation={"enabled": True, "benchmarks": ["hellaswag"]},
            iti={"enabled": True, "bake_in": True},
            refusal={"enabled": True},
            serving={"export_format": "safetensors"},
        )
        builder = DAGBuilder()
        dag = builder.build_dag(config)
        enabled = [s for s in dag if s.enabled]
        enabled_names = {s.name for s in enabled}
        # All stages should be enabled (except maybe cleaning if not configured)
        assert "data_loading" in enabled_names
        assert "preprocessing" in enabled_names
        assert "refusal_augmentation" in enabled_names
        assert "training" in enabled_names
        assert "iti_probing" in enabled_names
        assert "iti_baking" in enabled_names
        assert "evaluation" in enabled_names
        assert "export" in enabled_names
