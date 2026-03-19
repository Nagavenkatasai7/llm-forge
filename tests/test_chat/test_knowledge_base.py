"""Tests for the LLM Forge chat knowledge base module."""

from __future__ import annotations

from llm_forge.chat.knowledge_base import FORGE_KNOWLEDGE
from llm_forge.chat.system_prompt import SYSTEM_PROMPT


class TestKnowledgeBaseType:
    """Verify the knowledge base is a non-empty string constant."""

    def test_knowledge_base_is_string(self) -> None:
        assert isinstance(FORGE_KNOWLEDGE, str)
        assert len(FORGE_KNOWLEDGE) > 1000, (
            "Knowledge base should be a substantial document, "
            f"but is only {len(FORGE_KNOWLEDGE)} chars"
        )


class TestKnowledgeBaseProjectStructure:
    """Verify the knowledge base documents the project structure."""

    def test_knowledge_base_has_project_structure(self) -> None:
        assert "### Project Structure" in FORGE_KNOWLEDGE
        assert "configs/" in FORGE_KNOWLEDGE
        assert "data/" in FORGE_KNOWLEDGE
        assert "outputs/" in FORGE_KNOWLEDGE
        assert ".llmforge/" in FORGE_KNOWLEDGE
        assert "config.yaml" in FORGE_KNOWLEDGE
        assert ".gitignore" in FORGE_KNOWLEDGE


class TestKnowledgeBasePipelineStages:
    """Verify the knowledge base documents all 12 pipeline stages."""

    def test_knowledge_base_has_pipeline_stages(self) -> None:
        assert "### Pipeline Stages" in FORGE_KNOWLEDGE
        # All 12 stages must be documented
        stages = [
            "data_loading",
            "cleaning",
            "preprocessing",
            "refusal_augmentation",
            "ifd_scoring",
            "training",
            "alignment",
            "iti_probing",
            "iti_baking",
            "model_merging",
            "evaluation",
            "export",
        ]
        for stage in stages:
            assert stage in FORGE_KNOWLEDGE, f"Pipeline stage '{stage}' not documented"

    def test_pipeline_stages_in_order(self) -> None:
        """Stages should appear in the correct execution order in the pipeline section."""
        # Use the numbered stage markers to avoid false matches elsewhere
        numbered_stages = [
            "1. **data_loading**",
            "2. **cleaning**",
            "3. **preprocessing**",
            "4. **refusal_augmentation**",
            "5. **ifd_scoring**",
            "6. **training**",
            "7. **alignment**",
            "8. **iti_probing**",
            "9. **iti_baking**",
            "10. **model_merging**",
            "11. **evaluation**",
            "12. **export**",
        ]
        positions = [FORGE_KNOWLEDGE.index(s) for s in numbered_stages]
        assert positions == sorted(positions), "Pipeline stages are not in order"


class TestKnowledgeBaseConfigReference:
    """Verify the knowledge base includes a comprehensive config field reference."""

    def test_knowledge_base_has_config_reference(self) -> None:
        assert "### Config Field Reference" in FORGE_KNOWLEDGE

    def test_config_sections_present(self) -> None:
        sections = [
            "#### model",
            "#### lora",
            "#### quantization",
            "#### data",
            "#### data.cleaning",
            "#### training",
            "#### evaluation",
            "#### serving",
            "#### distributed",
            "#### iti",
            "#### refusal",
            "#### ifd",
            "#### merge",
            "#### alignment",
            "#### mlx",
            "#### compute",
            "#### mac",
        ]
        for section in sections:
            assert section in FORGE_KNOWLEDGE, f"Config section '{section}' not documented"

    def test_key_fields_documented(self) -> None:
        """Critical config fields that users commonly need."""
        key_fields = [
            "learning_rate",
            "num_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "assistant_only_loss",
            "completion_only_loss",
            "neftune_noise_alpha",
            "max_seq_length",
            "export_format",
            "gguf_quantization",
            "merge_adapter",
            "target_modules",
        ]
        for field in key_fields:
            assert field in FORGE_KNOWLEDGE, f"Key config field '{field}' not documented"


class TestKnowledgeBaseModelGuide:
    """Verify the knowledge base includes a model selection guide."""

    def test_knowledge_base_has_model_guide(self) -> None:
        assert "### Model Selection Guide" in FORGE_KNOWLEDGE
        # Must include VRAM-based recommendations
        assert "VRAM" in FORGE_KNOWLEDGE
        assert "QLoRA" in FORGE_KNOWLEDGE
        assert "LoRA" in FORGE_KNOWLEDGE
        assert "Apple Silicon" in FORGE_KNOWLEDGE or "Apple MLX" in FORGE_KNOWLEDGE

    def test_model_names_present(self) -> None:
        """Specific model recommendations should be present."""
        models = [
            "SmolLM2-135M",
            "Llama-3.2-1B-Instruct",
            "Llama-3.2-3B",
        ]
        for model in models:
            assert model in FORGE_KNOWLEDGE, f"Model '{model}' not in selection guide"


class TestKnowledgeBaseCommonErrors:
    """Verify the knowledge base includes common errors and fixes."""

    def test_knowledge_base_has_common_errors(self) -> None:
        assert "### Common Errors and Fixes" in FORGE_KNOWLEDGE

    def test_critical_errors_documented(self) -> None:
        """Must document the most frequent user-facing errors."""
        errors = [
            "OOM",
            "NaN loss",
            "Gibberish",
            "Catastrophic forgetting",
        ]
        for error in errors:
            assert error.lower() in FORGE_KNOWLEDGE.lower(), (
                f"Common error '{error}' not documented"
            )


class TestKnowledgeBaseSecurityRules:
    """Verify the knowledge base includes security rules."""

    def test_knowledge_base_has_security_rules(self) -> None:
        assert "### Security Rules" in FORGE_KNOWLEDGE
        assert "API keys" in FORGE_KNOWLEDGE or "api keys" in FORGE_KNOWLEDGE.lower()
        assert ".gitignore" in FORGE_KNOWLEDGE
        assert "permission" in FORGE_KNOWLEDGE.lower()


class TestKnowledgeBaseExtras:
    """Verify additional knowledge sections."""

    def test_has_data_formats(self) -> None:
        assert "### Supported Data Formats" in FORGE_KNOWLEDGE
        assert "Alpaca" in FORGE_KNOWLEDGE
        assert "ShareGPT" in FORGE_KNOWLEDGE
        assert "Completion" in FORGE_KNOWLEDGE

    def test_has_training_best_practices(self) -> None:
        assert "### Training Best Practices" in FORGE_KNOWLEDGE

    def test_has_gguf_quantization_guide(self) -> None:
        assert "### GGUF Quantization Guide" in FORGE_KNOWLEDGE
        assert "Q4_K_M" in FORGE_KNOWLEDGE


class TestSystemPromptIncludesKnowledge:
    """Verify that the assembled system prompt includes the knowledge base."""

    def test_system_prompt_includes_knowledge(self) -> None:
        # The knowledge base content must appear in the assembled SYSTEM_PROMPT
        assert FORGE_KNOWLEDGE in SYSTEM_PROMPT, "SYSTEM_PROMPT does not contain FORGE_KNOWLEDGE"

    def test_system_prompt_has_core_instructions(self) -> None:
        assert "You are LLM Forge" in SYSTEM_PROMPT
        assert "Your Memory System" in SYSTEM_PROMPT

    def test_knowledge_follows_core_prompt(self) -> None:
        """Knowledge base should appear AFTER core instructions."""
        core_pos = SYSTEM_PROMPT.index("You are LLM Forge")
        knowledge_pos = SYSTEM_PROMPT.index("## LLM Forge Architecture Knowledge")
        assert knowledge_pos > core_pos, "Knowledge base should appear after core instructions"
