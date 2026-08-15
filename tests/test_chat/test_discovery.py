"""Tests for model/dataset discovery, memory fit, and ground-truth scoring."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_forge.chat.discovery import (
    METHOD_BACKENDS,
    _normalise_search_type,
    assess_fit,
    assess_ground_truth,
    params_from_name,
    recommended_method,
    search_datasets,
    search_huggingface,
    search_models,
)


class TestSearchTypeNormalisation:
    """The singular/plural mismatch that silently returned no results."""

    @pytest.mark.parametrize("value", ["model", "models", "MODEL", " Models "])
    def test_model_forms_all_normalise(self, value: str) -> None:
        assert _normalise_search_type(value) == "models"

    @pytest.mark.parametrize("value", ["dataset", "datasets", "DataSets"])
    def test_dataset_forms_all_normalise(self, value: str) -> None:
        assert _normalise_search_type(value) == "datasets"

    def test_unknown_type_raises_rather_than_returning_empty(self) -> None:
        with pytest.raises(ValueError, match="search_type"):
            _normalise_search_type("modles")

    def test_singular_model_reaches_the_model_branch(self) -> None:
        """Regression: 'model' used to fall through and return zero results.

        The tool schema advertised the plural while research_tools defaulted to
        the singular, so every agent call took neither branch and reported an
        empty result set with no error.
        """
        fake_api = SimpleNamespace(
            list_models=lambda **kw: iter(
                [
                    SimpleNamespace(
                        id="org/tiny-1B",
                        downloads=10,
                        likes=1,
                        pipeline_tag="text-generation",
                        library_name="transformers",
                        tags=["license:mit"],
                        gated=False,
                    )
                ]
            ),
            list_datasets=lambda **kw: iter([]),
            model_info=lambda *a, **kw: SimpleNamespace(safetensors=None),
        )
        payload = json.loads(search_huggingface("tiny", "model", api=fake_api))
        assert payload["type"] == "models"
        assert payload["count"] == 1
        assert payload["results"][0]["id"] == "org/tiny-1B"


class TestParamsFromName:
    @pytest.mark.parametrize(
        ("repo", "expected"),
        [
            ("meta-llama/Llama-3.2-1B-Instruct", 1e9),
            ("HuggingFaceTB/SmolLM2-360M", 360e6),
            ("Qwen/Qwen2.5-7B", 7e9),
            ("mistralai/Mistral-7B-v0.1", 7e9),
        ],
    )
    def test_reads_size_suffix(self, repo: str, expected: float) -> None:
        assert params_from_name(repo) == expected

    def test_moe_name_uses_total_not_active_params(self) -> None:
        """A "30B-A3B" MoE needs memory for 30B, not the 3B active count.

        Reading the trailing match would underestimate tenfold and recommend a
        model that OOMs, so the largest match wins.
        """
        assert params_from_name("Qwen/Qwen3-Coder-30B-A3B-Instruct") == 30e9

    def test_no_size_hint_returns_none(self) -> None:
        assert params_from_name("openai/whisper-large") is None


class TestMemoryFit:
    def test_small_model_fits_everything_on_cuda(self) -> None:
        verdicts = {v.method: v for v in assess_fit(360e6, 24.0, backend="cuda")}
        assert verdicts["full"].fits
        assert verdicts["lora"].fits
        assert verdicts["qlora"].fits

    def test_required_memory_grows_with_model_size(self) -> None:
        small = {v.method: v.required_gb for v in assess_fit(1e9, 24.0)}
        large = {v.method: v.required_gb for v in assess_fit(8e9, 24.0)}
        for method in small:
            assert large[method] > small[method]

    def test_full_finetune_needs_more_than_lora(self) -> None:
        verdicts = {v.method: v.required_gb for v in assess_fit(1e9, 24.0)}
        assert verdicts["full"] > verdicts["full_8bit_optim"] > verdicts["lora"]

    def test_bitsandbytes_qlora_never_fits_on_apple(self) -> None:
        """bitsandbytes has no Metal backend, so QLoRA must not be recommended.

        Reporting it as a fit is how a user ends up with an ImportError several
        minutes into what they thought was a working run.
        """
        verdicts = {v.method: v for v in assess_fit(1e9, 18.0, backend="mps")}
        assert not verdicts["qlora"].fits
        assert "CUDA-only" in verdicts["qlora"].note

    def test_mlx_path_available_on_apple_but_not_cuda(self) -> None:
        apple = {v.method: v for v in assess_fit(8e9, 18.0, backend="mps")}
        cuda = {v.method: v for v in assess_fit(8e9, 18.0, backend="cuda")}
        assert apple["mlx_lora_4bit"].fits
        assert not cuda["mlx_lora_4bit"].fits

    def test_8b_on_24gb_mac_routes_to_mlx(self) -> None:
        """The user's actual machine: 24 GB unified, ~18 GB usable."""
        assert recommended_method(assess_fit(8e9, 18.0, backend="mps")) == "mlx_lora_4bit"

    def test_1b_on_24gb_mac_routes_to_lora(self) -> None:
        """A 1.24B full fine-tune needs ~18.6 GB -- just over an 18 GB budget.

        Optimizer state dominates (12 of the 16 bytes/param), so trimming
        sequence length does not rescue it. LoRA is the honest answer.
        """
        assert recommended_method(assess_fit(1.24e9, 18.0, backend="mps")) == "lora"

    def test_360m_full_finetune_fits_on_24gb_mac(self) -> None:
        """The largest model that can have every weight updated on this machine."""
        assert recommended_method(assess_fit(360e6, 18.0, backend="mps")) == "full"

    def test_8bit_optimizer_never_offered_on_apple(self) -> None:
        """adamw_8bit is a bitsandbytes optimizer, and MLX has no equivalent.

        Recommending it on Apple Silicon would be advice the user cannot act on
        -- the same class of error as recommending QLoRA there.
        """
        verdicts = {v.method: v for v in assess_fit(1e9, 18.0, backend="mps")}
        assert not verdicts["full_8bit_optim"].fits
        assert "CUDA only" in verdicts["full_8bit_optim"].note

    def test_8bit_optimizer_offered_on_cuda(self) -> None:
        verdicts = {v.method: v for v in assess_fit(1e9, 18.0, backend="cuda")}
        assert verdicts["full_8bit_optim"].fits

    def test_oversized_model_has_no_recommendation(self) -> None:
        assert recommended_method(assess_fit(400e9, 18.0, backend="mps")) is None

    def test_every_method_declares_its_backends(self) -> None:
        from llm_forge.chat.discovery import BYTES_PER_PARAM

        assert set(BYTES_PER_PARAM) == set(METHOD_BACKENDS)


class TestGroundTruthAssessment:
    def test_benchmark_with_paper_and_holdout_scores_strong(self) -> None:
        assessment = assess_ground_truth(
            tags=[
                "benchmark:official",
                "arxiv:2110.14168",
                "annotations_creators:crowdsourced",
                "task_categories:question-answering",
            ],
            card_data={"task_categories": ["question-answering"]},
            splits=["train", "test"],
        )
        assert assessment.verdict == "strong"
        assert any("benchmark" in e.lower() for e in assessment.evidence)

    def test_bare_dataset_scores_none_with_caveats(self) -> None:
        assessment = assess_ground_truth(tags=[], card_data={}, splits=[])
        assert assessment.score == 0
        assert assessment.verdict == "none"
        assert assessment.caveats

    def test_train_only_split_is_flagged(self) -> None:
        assessment = assess_ground_truth(tags=[], card_data={}, splits=["train"])
        assert any("eval set" in c for c in assessment.caveats)

    def test_machine_generated_labels_are_flagged(self) -> None:
        assessment = assess_ground_truth(
            tags=["annotations_creators:machine-generated"], card_data={}, splits=["test"]
        )
        assert any("machine-generated" in c for c in assessment.caveats)

    def test_unlabelled_task_type_is_flagged(self) -> None:
        assessment = assess_ground_truth(
            tags=[], card_data={"task_categories": ["text-generation"]}, splits=["test"]
        )
        assert any("no inherent correct answer" in c for c in assessment.caveats)


class TestDatasetSearchRanking:
    def test_results_rank_ground_truth_above_downloads(self) -> None:
        """A popular unverifiable dataset must not outrank a verifiable one."""
        popular_but_unverified = SimpleNamespace(
            id="org/popular",
            downloads=1_000_000,
            likes=999,
            gated=False,
            last_modified="2026-01-01",
            tags=[],
            card_data=None,
        )
        benchmark = SimpleNamespace(
            id="org/benchmark",
            downloads=10,
            likes=1,
            gated=False,
            last_modified="2026-01-01",
            tags=[
                "benchmark:official",
                "arxiv:1234.5678",
                "annotations_creators:expert-generated",
                "task_categories:question-answering",
                "license:mit",
            ],
            card_data=None,
        )
        fake_api = SimpleNamespace(
            list_datasets=lambda **kw: iter([popular_but_unverified, benchmark])
        )
        payload = search_datasets("x", api=fake_api)
        assert payload["results"][0]["id"] == "org/benchmark"

    def test_missing_license_is_surfaced(self) -> None:
        entry = SimpleNamespace(
            id="org/x",
            downloads=1,
            likes=0,
            gated=False,
            last_modified="",
            tags=[],
            card_data=None,
        )
        fake_api = SimpleNamespace(list_datasets=lambda **kw: iter([entry]))
        result = search_datasets("x", api=fake_api)["results"][0]
        assert result["license"] is None
        assert "No license declared" in result["license_note"]


class TestModelSearchFit:
    def _fake_api(self, repo_id: str, gated: bool = False):
        return SimpleNamespace(
            list_models=lambda **kw: iter(
                [
                    SimpleNamespace(
                        id=repo_id,
                        downloads=100,
                        likes=5,
                        pipeline_tag="text-generation",
                        library_name="transformers",
                        tags=["license:apache-2.0"],
                        gated=gated,
                    )
                ]
            ),
            model_info=lambda *a, **kw: SimpleNamespace(safetensors=None),
        )

    def test_fit_reported_against_budget(self) -> None:
        payload = search_models(
            "x", budget_gb=18.0, backend="mps", api=self._fake_api("org/model-1B")
        )
        fit = payload["results"][0]["fit"]
        assert fit["budget_gb"] == 18.0
        assert fit["recommended_method"] is not None

    def test_oversized_model_says_so_plainly(self) -> None:
        payload = search_models(
            "x", budget_gb=18.0, backend="mps", api=self._fake_api("org/model-405B")
        )
        fit = payload["results"][0]["fit"]
        assert fit["recommended_method"] is None
        assert "does not fit" in fit["note"].lower()

    def test_gated_repo_warns_about_access(self) -> None:
        payload = search_models("x", api=self._fake_api("org/model-1B", gated=True))
        assert "access_note" in payload["results"][0]

    def test_no_budget_means_no_fit_section(self) -> None:
        payload = search_models("x", api=self._fake_api("org/model-1B"))
        assert "fit" not in payload["results"][0]


class TestErrorHandling:
    def test_hub_failure_returns_error_json_not_exception(self) -> None:
        def boom(**kwargs):
            raise ConnectionError("hub unreachable")

        fake_api = SimpleNamespace(list_models=boom, list_datasets=boom)
        payload = json.loads(search_huggingface("x", "models", api=fake_api))
        assert "error" in payload
        assert "hub unreachable" in payload["error"]
