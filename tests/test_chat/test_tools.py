"""Tests for the chat tools module.

Covers hardware detection, data scanning, config writing/validation,
config listing, training status, and the execute_tool dispatcher.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from llm_forge.chat.tools import execute_tool

# ===================================================================
# Hardware detection
# ===================================================================


class TestDetectHardware:
    """Test _detect_hardware() via execute_tool."""

    def test_detect_hardware_returns_json(self) -> None:
        """_detect_hardware() returns valid JSON with os and cpu keys."""
        result = json.loads(execute_tool("detect_hardware", {}))
        assert "os" in result
        assert "cpu" in result
        assert "python_version" in result
        # Should be a recognized OS
        assert result["os"] in ("Darwin", "Linux", "Windows")

    def test_detect_hardware_has_gpu_info(self) -> None:
        """_detect_hardware() includes gpu_type key."""
        result = json.loads(execute_tool("detect_hardware", {}))
        assert "gpu_type" in result


# ===================================================================
# Data scanning
# ===================================================================


class TestScanData:
    """Test _scan_data() with various inputs."""

    def test_scan_data_local_jsonl(self, tmp_path: Path) -> None:
        """_scan_data() on a real JSONL file returns format + preview."""
        jsonl_file = tmp_path / "train.jsonl"
        records = [
            {"instruction": "What is AI?", "output": "Artificial Intelligence"},
            {"instruction": "What is ML?", "output": "Machine Learning"},
            {"instruction": "What is DL?", "output": "Deep Learning"},
        ]
        jsonl_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        result = json.loads(execute_tool("scan_data", {"path": str(jsonl_file)}))
        assert result["status"] == "ok"
        assert result["source"] == "local_file"
        assert result["sample_count"] == 3
        assert result["detected_format"] == "alpaca"
        assert len(result["preview"]) <= 3

    def test_scan_data_not_found(self) -> None:
        """_scan_data() on nonexistent path returns error."""
        result = json.loads(execute_tool("scan_data", {"path": "/nonexistent/path/data.jsonl"}))
        assert result["status"] == "not_found"
        assert "error" in result

    def test_scan_data_directory(self, tmp_path: Path) -> None:
        """_scan_data() on a directory returns file count and size."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file1.txt").write_text("hello world")
        (data_dir / "file2.txt").write_text("goodbye world")

        result = json.loads(execute_tool("scan_data", {"path": str(data_dir)}))
        assert result["status"] == "ok"
        assert result["source"] == "local_directory"
        assert result["file_count"] == 2

    def test_scan_data_sharegpt_format(self, tmp_path: Path) -> None:
        """_scan_data() detects sharegpt format correctly."""
        jsonl_file = tmp_path / "sharegpt.jsonl"
        records = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi!"},
                ]
            }
        ]
        jsonl_file.write_text(json.dumps(records[0]) + "\n")
        result = json.loads(execute_tool("scan_data", {"path": str(jsonl_file)}))
        assert result["detected_format"] == "sharegpt"

    def test_scan_data_text_file(self, tmp_path: Path) -> None:
        """_scan_data() handles plain text files."""
        txt_file = tmp_path / "corpus.txt"
        txt_file.write_text("This is a training corpus with many words.")
        result = json.loads(execute_tool("scan_data", {"path": str(txt_file)}))
        assert result["status"] == "ok"
        assert result["detected_format"] == "completion"
        assert "word_count" in result


# ===================================================================
# Config writing
# ===================================================================


class TestWriteConfig:
    """Test _write_config() creates a valid YAML file."""

    def test_write_config_creates_file(self, tmp_path: Path) -> None:
        """_write_config() creates a YAML file at the specified path."""
        output_path = str(tmp_path / "generated_config.yaml")
        config = {
            "model": {"name": "Llama-3.2-1B"},
            "data": {"train_path": "data/train.jsonl"},
            "training": {"mode": "lora", "num_epochs": 1},
        }
        result = json.loads(
            execute_tool("write_config", {"output_path": output_path, "config": config})
        )
        assert result["status"] == "ok"
        assert Path(output_path).exists()

        # Verify the file is valid YAML with expected content
        with open(output_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["model"]["name"] == "Llama-3.2-1B"
        assert loaded["training"]["mode"] == "lora"

    def test_write_config_creates_parent_dirs(self, tmp_path: Path) -> None:
        """_write_config() creates parent directories if needed."""
        output_path = str(tmp_path / "nested" / "dir" / "config.yaml")
        config = {"model": {"name": "test"}}
        result = json.loads(
            execute_tool("write_config", {"output_path": output_path, "config": config})
        )
        assert result["status"] == "ok"
        assert Path(output_path).exists()


# ===================================================================
# Config validation
# ===================================================================


class TestValidateConfig:
    """Test _validate_config() with valid and invalid configs."""

    def test_validate_config_valid(self, tmp_path: Path) -> None:
        """_validate_config() on a valid config returns 'valid'."""
        config_path = tmp_path / "valid.yaml"
        config = {
            "model": {"name": "meta-llama/Llama-3.2-1B"},
            "data": {"train_path": "tatsu-lab/alpaca"},
        }
        config_path.write_text(yaml.dump(config, default_flow_style=False))

        result = json.loads(execute_tool("validate_config", {"config_path": str(config_path)}))
        assert result["status"] == "valid"

    def test_validate_config_invalid(self, tmp_path: Path) -> None:
        """_validate_config() on an empty file returns error."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        result = json.loads(execute_tool("validate_config", {"config_path": str(config_path)}))
        # Empty YAML returns None, which triggers an error
        assert result["status"] == "error"

    def test_validate_config_missing_required(self, tmp_path: Path) -> None:
        """_validate_config() on config missing required fields returns invalid."""
        config_path = tmp_path / "incomplete.yaml"
        # Missing model and data sections entirely
        config_path.write_text("training:\n  mode: lora\n")

        result = json.loads(execute_tool("validate_config", {"config_path": str(config_path)}))
        assert result["status"] == "invalid"
        assert "errors" in result


# ===================================================================
# List configs
# ===================================================================


class TestListConfigs:
    """Test _list_configs() finding config files."""

    def test_list_configs_finds_files(self, tmp_path: Path) -> None:
        """_list_configs() returns config files from the configs directory."""
        from unittest.mock import patch

        # Create a configs dir with known files
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        (configs_dir / "train.yaml").write_text("model:\n  name: test\n")
        (configs_dir / "eval.yaml").write_text("model:\n  name: test2\n")

        # Patch the internal _list_configs to look at our temp dir only
        from llm_forge.chat import tools as tools_module

        fake_file = tmp_path / "src" / "llm_forge" / "chat" / "tools.py"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.touch()

        with patch.object(tools_module, "__file__", str(fake_file)):
            result = json.loads(execute_tool("list_configs", {}))

        assert result["count"] == 2
        names = [c["name"] for c in result["configs"]]
        assert "eval.yaml" in names
        assert "train.yaml" in names


# ===================================================================
# Training status
# ===================================================================


class TestCheckTrainingStatus:
    """Test _check_training_status()."""

    def test_check_training_status_idle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns idle when no training running and no outputs dir."""
        monkeypatch.chdir(tmp_path)
        result = json.loads(execute_tool("check_training_status", {}))
        assert result["status"] == "idle"
        assert "No training detected" in result["message"]


# ===================================================================
# Execute tool dispatcher
# ===================================================================


class TestExecuteToolDispatcher:
    """Test the execute_tool() routing logic."""

    def test_execute_tool_unknown(self) -> None:
        """execute_tool('fake_tool', {}) returns error JSON."""
        result = json.loads(execute_tool("fake_tool", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_execute_tool_catches_exceptions(self) -> None:
        """execute_tool() catches exceptions and returns error JSON."""
        # scan_data requires a "path" key; omitting it should trigger KeyError
        result = json.loads(execute_tool("scan_data", {}))
        assert "error" in result


# ===================================================================
# NVIDIA-powered tools: generate_training_data
# ===================================================================


class TestGenerateTrainingData:
    """Test _generate_training_data() via execute_tool (mocked NVIDIA API)."""

    def _make_mock_response(self, content: str):
        """Build a mock OpenAI ChatCompletion-like response."""
        from types import SimpleNamespace

        choice = SimpleNamespace()
        choice.message = SimpleNamespace()
        choice.message.content = content
        resp = SimpleNamespace()
        resp.choices = [choice]
        return resp

    def test_generate_training_data_basic(self, tmp_path: Path) -> None:
        """Generates samples and writes JSONL to the output path."""
        from unittest.mock import MagicMock, patch

        fake_lines = "\n".join(
            [
                json.dumps({"instruction": f"Q{i}?", "input": "", "output": f"A{i}."})
                for i in range(3)
            ]
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(fake_lines)

        with patch("openai.OpenAI", return_value=mock_client):
            output_path = str(tmp_path / "synthetic.jsonl")
            result = json.loads(
                execute_tool(
                    "generate_training_data",
                    {
                        "topic": "Python programming",
                        "num_samples": 3,
                        "output_path": output_path,
                    },
                )
            )

        assert result["status"] == "ok"
        assert result["samples_generated"] == 3
        assert result["topic"] == "Python programming"
        assert Path(output_path).exists()

        # Verify the JSONL content
        with open(output_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 3
        assert lines[0]["instruction"] == "Q0?"

    def test_generate_training_data_with_examples(self, tmp_path: Path) -> None:
        """Examples text is included in the prompt sent to the model."""
        from unittest.mock import MagicMock, patch

        fake_line = json.dumps({"instruction": "Q?", "input": "", "output": "A."})
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(fake_line)

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "generate_training_data",
                    {
                        "topic": "cooking",
                        "num_samples": 1,
                        "examples": ["Q: What is sauteing? A: Pan-cooking with oil."],
                        "output_path": str(tmp_path / "out.jsonl"),
                    },
                )
            )

        assert result["status"] == "ok"
        # Verify examples were passed in the prompt
        call_args = mock_client.chat.completions.create.call_args
        prompt_text = call_args.kwargs["messages"][0]["content"]
        assert "sauteing" in prompt_text

    def test_generate_training_data_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Output path with nested directories is created automatically."""
        from unittest.mock import MagicMock, patch

        fake_line = json.dumps({"instruction": "Q?", "input": "", "output": "A."})
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(fake_line)

        output_path = str(tmp_path / "deep" / "nested" / "dir" / "data.jsonl")
        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "generate_training_data",
                    {"topic": "math", "num_samples": 1, "output_path": output_path},
                )
            )

        assert result["status"] == "ok"
        assert Path(output_path).exists()

    def test_generate_training_data_api_error(self, tmp_path: Path) -> None:
        """Returns error status when the NVIDIA API call fails."""
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API timeout")

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "generate_training_data",
                    {
                        "topic": "finance",
                        "num_samples": 5,
                        "output_path": str(tmp_path / "out.jsonl"),
                    },
                )
            )

        assert result["status"] == "error"
        assert "API timeout" in result["error"]
        assert result["generated_so_far"] == 0

    def test_generate_training_data_skips_malformed_json(self, tmp_path: Path) -> None:
        """Malformed JSON lines in the model response are silently skipped."""
        from unittest.mock import MagicMock, patch

        content = (
            '{"instruction": "Good Q?", "input": "", "output": "Good A."}\n'
            "this is not json\n"
            '{"instruction": "Also good?", "output": "Yes."}\n'
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(content)

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "generate_training_data",
                    {
                        "topic": "science",
                        "num_samples": 3,
                        "output_path": str(tmp_path / "out.jsonl"),
                    },
                )
            )

        assert result["status"] == "ok"
        # Only 2 valid JSON lines out of 3
        assert result["samples_generated"] == 2

    def test_generate_training_data_default_output_path(self) -> None:
        """When no output_path is given, defaults to data/synthetic_train.jsonl."""
        from unittest.mock import MagicMock, patch

        fake_line = json.dumps({"instruction": "Q?", "input": "", "output": "A."})
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(fake_line)

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "generate_training_data",
                    {"topic": "history", "num_samples": 1},
                )
            )

        assert result["status"] == "ok"
        assert result["output_path"] == "data/synthetic_train.jsonl"
        # Clean up
        Path("data/synthetic_train.jsonl").unlink(missing_ok=True)


# ===================================================================
# NVIDIA-powered tools: evaluate_with_llm
# ===================================================================


class TestEvaluateWithLlm:
    """Test _evaluate_with_llm() via execute_tool (mocked NVIDIA API)."""

    def _make_mock_response(self, content: str):
        """Build a mock OpenAI ChatCompletion-like response."""
        from types import SimpleNamespace

        choice = SimpleNamespace()
        choice.message = SimpleNamespace()
        choice.message.content = content
        resp = SimpleNamespace()
        resp.choices = [choice]
        return resp

    def test_evaluate_with_llm_basic(self) -> None:
        """Evaluates a single Q&A pair and returns scores."""
        from unittest.mock import MagicMock, patch

        judge_response = json.dumps(
            {
                "scores": {"relevance": 4, "accuracy": 5, "helpfulness": 4},
                "overall": 4.3,
                "feedback": "Good response.",
            }
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(judge_response)

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "evaluate_with_llm",
                    {
                        "questions": ["What is Python?"],
                        "model_outputs": ["Python is a programming language."],
                    },
                )
            )

        assert result["status"] == "ok"
        assert result["samples_evaluated"] == 1
        assert result["average_score"] == 4.3
        assert len(result["evaluations"]) == 1
        assert result["evaluations"][0]["overall"] == 4.3

    def test_evaluate_with_llm_multiple(self) -> None:
        """Evaluates multiple Q&A pairs and averages scores."""
        from unittest.mock import MagicMock, patch

        responses = [
            json.dumps(
                {
                    "scores": {"relevance": 5, "accuracy": 5, "helpfulness": 5},
                    "overall": 5.0,
                    "feedback": "Excellent.",
                }
            ),
            json.dumps(
                {
                    "scores": {"relevance": 3, "accuracy": 3, "helpfulness": 3},
                    "overall": 3.0,
                    "feedback": "Average.",
                }
            ),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            self._make_mock_response(r) for r in responses
        ]

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "evaluate_with_llm",
                    {
                        "questions": ["Q1?", "Q2?"],
                        "model_outputs": ["A1.", "A2."],
                    },
                )
            )

        assert result["status"] == "ok"
        assert result["samples_evaluated"] == 2
        assert result["average_score"] == 4.0  # (5.0 + 3.0) / 2

    def test_evaluate_with_llm_custom_criteria(self) -> None:
        """Custom criteria string is passed to the judge prompt."""
        from unittest.mock import MagicMock, patch

        judge_response = json.dumps(
            {
                "scores": {"clarity": 4, "depth": 3},
                "overall": 3.5,
                "feedback": "Decent.",
            }
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(judge_response)

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "evaluate_with_llm",
                    {
                        "questions": ["Explain AI"],
                        "model_outputs": ["AI is intelligence."],
                        "criteria": "clarity, depth",
                    },
                )
            )

        assert result["status"] == "ok"
        assert result["criteria"] == "clarity, depth"
        # Verify criteria was passed in the prompt
        call_args = mock_client.chat.completions.create.call_args
        prompt_text = call_args.kwargs["messages"][0]["content"]
        assert "clarity, depth" in prompt_text

    def test_evaluate_with_llm_api_error(self) -> None:
        """API error for one question produces error entry but does not crash."""
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Rate limited")

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "evaluate_with_llm",
                    {
                        "questions": ["What is ML?"],
                        "model_outputs": ["ML is machine learning."],
                    },
                )
            )

        assert result["status"] == "ok"
        assert result["samples_evaluated"] == 1
        assert result["average_score"] == 0  # No valid scores
        assert "error" in result["evaluations"][0]
        assert "Rate limited" in result["evaluations"][0]["error"]

    def test_evaluate_with_llm_unparseable_response(self) -> None:
        """When judge returns non-JSON, an error entry is recorded."""
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "I cannot evaluate this in JSON format."
        )

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "evaluate_with_llm",
                    {
                        "questions": ["Test?"],
                        "model_outputs": ["Answer."],
                    },
                )
            )

        assert result["status"] == "ok"
        assert result["samples_evaluated"] == 1
        assert "error" in result["evaluations"][0]

    def test_evaluate_with_llm_question_truncation(self) -> None:
        """Long questions are truncated to 100 chars in the evaluation result."""
        from unittest.mock import MagicMock, patch

        long_question = "Q" * 200
        judge_response = json.dumps(
            {
                "scores": {"relevance": 4, "accuracy": 4, "helpfulness": 4},
                "overall": 4.0,
                "feedback": "Good.",
            }
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(judge_response)

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "evaluate_with_llm",
                    {
                        "questions": [long_question],
                        "model_outputs": ["Short answer."],
                    },
                )
            )

        assert result["status"] == "ok"
        assert len(result["evaluations"][0]["question"]) == 100


# ===================================================================
# NVIDIA-powered tools: test_model
# ===================================================================


class TestTestModel:
    """Test _test_model() via execute_tool (mocked NVIDIA API)."""

    def _make_mock_response(self, content: str, total_tokens: int = 42):
        """Build a mock OpenAI ChatCompletion-like response with usage."""
        from types import SimpleNamespace

        choice = SimpleNamespace()
        choice.message = SimpleNamespace()
        choice.message.content = content
        usage = SimpleNamespace()
        usage.total_tokens = total_tokens
        resp = SimpleNamespace()
        resp.choices = [choice]
        resp.usage = usage
        return resp

    def test_test_model_basic(self) -> None:
        """Tests a single question and returns the model's answer."""
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "Python is a programming language.", total_tokens=50
        )

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "test_model",
                    {
                        "model": "meta/llama-3.2-3b-instruct",
                        "question": "What is Python?",
                    },
                )
            )

        assert result["status"] == "ok"
        assert result["model"] == "meta/llama-3.2-3b-instruct"
        assert result["total_questions"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["question"] == "What is Python?"
        assert result["results"][0]["answer"] == "Python is a programming language."
        assert result["results"][0]["tokens_used"] == 50

    def test_test_model_multiple_questions(self) -> None:
        """Tests multiple newline-separated questions."""
        from unittest.mock import MagicMock, patch

        responses = [
            self._make_mock_response("Answer 1", total_tokens=30),
            self._make_mock_response("Answer 2", total_tokens=40),
            self._make_mock_response("Answer 3", total_tokens=35),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = responses

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "test_model",
                    {
                        "model": "meta/llama-3.1-8b-instruct",
                        "question": "What is AI?\nWhat is ML?\nWhat is DL?",
                        "num_questions": 3,
                    },
                )
            )

        assert result["status"] == "ok"
        assert result["total_questions"] == 3
        assert len(result["results"]) == 3
        assert result["results"][0]["question"] == "What is AI?"
        assert result["results"][0]["answer"] == "Answer 1"
        assert result["results"][1]["question"] == "What is ML?"
        assert result["results"][2]["question"] == "What is DL?"

    def test_test_model_error_handling(self) -> None:
        """API error for a question produces an error entry, not a crash."""
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Model not found")

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "test_model",
                    {
                        "model": "nonexistent/model",
                        "question": "Hello?",
                    },
                )
            )

        assert result["status"] == "ok"
        assert result["total_questions"] == 1
        assert "error" in result["results"][0]
        assert "Model not found" in result["results"][0]["error"]
        assert result["results"][0]["model"] == "nonexistent/model"


# ===================================================================
# NVIDIA-powered tools: generate_embeddings
# ===================================================================


class TestGenerateEmbeddings:
    """Test _generate_embeddings() via execute_tool (mocked NVIDIA API)."""

    def _make_mock_embedding_response(self, texts: list[str], dim: int = 1024):
        """Build a mock OpenAI Embeddings response."""
        from types import SimpleNamespace

        data = []
        for i, _text in enumerate(texts):
            item = SimpleNamespace()
            item.index = i
            item.embedding = [0.1 * (i + 1)] * dim
            data.append(item)

        resp = SimpleNamespace()
        resp.data = data
        return resp

    def test_generate_embeddings_basic(self) -> None:
        """Embeds texts and returns previews with metadata."""
        from unittest.mock import MagicMock, patch

        texts = ["What is machine learning?", "How does fine-tuning work?"]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._make_mock_embedding_response(texts)

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "generate_embeddings",
                    {"texts": texts},
                )
            )

        assert result["status"] == "ok"
        assert result["model"] == "nvidia/nv-embedqa-e5-v5"
        assert result["num_texts"] == 2
        assert result["embedding_dimension"] == 1024
        assert len(result["previews"]) == 2
        assert result["previews"][0]["embedding_dim"] == 1024
        # Preview should only contain first 5 embedding values
        assert len(result["previews"][0]["embedding"]) == 5
        # Text preview should be truncated to 100 chars
        assert result["previews"][0]["text"] == "What is machine learning?"

    def test_generate_embeddings_save_to_file(self, tmp_path: Path) -> None:
        """When output_path is provided, full embeddings are saved to JSON."""
        from unittest.mock import MagicMock, patch

        texts = ["Hello world", "Goodbye world"]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._make_mock_embedding_response(
            texts, dim=512
        )

        output_path = str(tmp_path / "nested" / "embeddings.json")
        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "generate_embeddings",
                    {
                        "texts": texts,
                        "model": "baai/bge-m3",
                        "output_path": output_path,
                    },
                )
            )

        assert result["status"] == "ok"
        assert result["saved_to"] == output_path
        assert result["model"] == "baai/bge-m3"
        assert Path(output_path).exists()

        # Verify saved content has full embeddings
        with open(output_path) as f:
            saved = json.load(f)
        assert len(saved) == 2
        assert saved[0]["text"] == "Hello world"
        assert len(saved[0]["embedding"]) == 512

    def test_generate_embeddings_error(self) -> None:
        """Returns error status when the NVIDIA embedding API call fails."""
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = RuntimeError("Model not available")

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "generate_embeddings",
                    {"texts": ["test text"]},
                )
            )

        assert result["status"] == "error"
        assert "Model not available" in result["error"]


# ===================================================================
# NVIDIA-powered tools: generate_script
# ===================================================================


class TestGenerateScript:
    """Test _generate_script() via execute_tool (mocked NVIDIA API)."""

    def _make_mock_response(self, content: str):
        """Build a mock OpenAI ChatCompletion-like response."""
        from types import SimpleNamespace

        choice = SimpleNamespace()
        choice.message = SimpleNamespace()
        choice.message.content = content
        resp = SimpleNamespace()
        resp.choices = [choice]
        return resp

    def test_generate_script_basic(self, tmp_path: Path) -> None:
        """Generates a script and saves it to the output path."""
        from unittest.mock import MagicMock, patch

        script_code = (
            "import pathlib\n\ndef main():\n    print('hello')\n\n"
            "if __name__ == '__main__':\n    main()\n"
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(script_code)

        output_path = str(tmp_path / "convert.py")
        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "generate_script",
                    {
                        "task_description": "convert CSV to JSONL",
                        "output_path": output_path,
                    },
                )
            )

        assert result["status"] == "ok"
        assert result["output_path"] == output_path
        assert result["lines"] > 0
        assert Path(output_path).exists()

        # Verify the script content was written
        written = Path(output_path).read_text()
        assert "def main():" in written

    def test_generate_script_extracts_code_block(self, tmp_path: Path) -> None:
        """Extracts code from markdown ```python code blocks."""
        from unittest.mock import MagicMock, patch

        wrapped_code = (
            "Here is the script:\n"
            "```python\n"
            "import sys\n\ndef main():\n    print('extracted')\n\n"
            "if __name__ == '__main__':\n    main()\n"
            "```\n"
            "This script converts files."
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(wrapped_code)

        output_path = str(tmp_path / "script.py")
        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "generate_script",
                    {
                        "task_description": "do something",
                        "output_path": output_path,
                    },
                )
            )

        assert result["status"] == "ok"
        written = Path(output_path).read_text()
        # Should NOT contain the markdown fence or explanation text
        assert "```" not in written
        assert "Here is the script" not in written
        assert "import sys" in written

    def test_generate_script_with_input_output_files(self, tmp_path: Path) -> None:
        """Input/output file names are passed to the prompt."""
        from unittest.mock import MagicMock, patch

        script_code = "print('done')\n"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(script_code)

        output_path = str(tmp_path / "proc.py")
        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "generate_script",
                    {
                        "task_description": "process data",
                        "output_path": output_path,
                        "input_file": "data.csv",
                        "output_file": "data.jsonl",
                    },
                )
            )

        assert result["status"] == "ok"
        # Verify file context was passed in the prompt
        call_args = mock_client.chat.completions.create.call_args
        prompt_text = call_args.kwargs["messages"][1]["content"]
        assert "data.csv" in prompt_text
        assert "data.jsonl" in prompt_text

    def test_generate_script_api_error(self, tmp_path: Path) -> None:
        """Returns error status when the NVIDIA API call fails."""
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "generate_script",
                    {
                        "task_description": "something",
                        "output_path": str(tmp_path / "fail.py"),
                    },
                )
            )

        assert result["status"] == "error"
        assert "API down" in result["error"]


# ===================================================================
# NVIDIA-powered tools: compare_models
# ===================================================================


class TestCompareModels:
    """Test _compare_models() via execute_tool (mocked NVIDIA API)."""

    def _make_mock_response(self, content: str):
        """Build a mock OpenAI ChatCompletion-like response."""
        from types import SimpleNamespace

        choice = SimpleNamespace()
        choice.message = SimpleNamespace()
        choice.message.content = content
        resp = SimpleNamespace()
        resp.choices = [choice]
        return resp

    def test_compare_models_basic(self) -> None:
        """Compares two models on one question and returns a verdict."""
        from unittest.mock import MagicMock, patch

        responses = [
            self._make_mock_response("Answer from model A"),
            self._make_mock_response("Answer from model B"),
            self._make_mock_response('{"winner": "A", "reason": "More detailed"}'),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = responses

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "compare_models",
                    {
                        "model_a": "meta/llama-3.2-1b-instruct",
                        "model_b": "meta/llama-3.2-3b-instruct",
                        "questions": ["What is machine learning?"],
                    },
                )
            )

        assert result["status"] == "ok"
        assert result["model_a"] == "meta/llama-3.2-1b-instruct"
        assert result["model_b"] == "meta/llama-3.2-3b-instruct"
        assert result["summary"]["model_a_wins"] == 1
        assert result["summary"]["model_b_wins"] == 0
        assert result["summary"]["ties"] == 0
        assert result["summary"]["total"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["winner"] == "A"

    def test_compare_models_with_winner_b(self) -> None:
        """When model B wins, verdict reflects that."""
        from unittest.mock import MagicMock, patch

        responses = [
            self._make_mock_response("Weak answer from A"),
            self._make_mock_response("Strong answer from B"),
            self._make_mock_response('{"winner": "B", "reason": "Much better"}'),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = responses

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "compare_models",
                    {
                        "model_a": "model-a",
                        "model_b": "model-b",
                        "questions": ["Explain AI"],
                    },
                )
            )

        assert result["summary"]["model_b_wins"] == 1
        assert result["verdict"] == "model-b wins"

    def test_compare_models_multiple_questions(self) -> None:
        """Works with multiple questions and tallies results correctly."""
        from unittest.mock import MagicMock, patch

        responses = [
            # Q1: model A answer, model B answer, judge
            self._make_mock_response("A1"),
            self._make_mock_response("B1"),
            self._make_mock_response('{"winner": "A", "reason": "Better"}'),
            # Q2: model A answer, model B answer, judge
            self._make_mock_response("A2"),
            self._make_mock_response("B2"),
            self._make_mock_response('{"winner": "B", "reason": "Better"}'),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = responses

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "compare_models",
                    {
                        "model_a": "model-a",
                        "model_b": "model-b",
                        "questions": ["Q1?", "Q2?"],
                    },
                )
            )

        assert result["summary"]["model_a_wins"] == 1
        assert result["summary"]["model_b_wins"] == 1
        assert result["summary"]["ties"] == 0
        assert result["summary"]["total"] == 2
        assert result["verdict"] == "Tie"

    def test_compare_models_judge_unavailable(self) -> None:
        """When judge API fails, result is counted as a tie."""
        from unittest.mock import MagicMock, patch

        responses = [
            self._make_mock_response("A answer"),
            self._make_mock_response("B answer"),
        ]
        mock_client = MagicMock()
        # First two calls succeed (model responses), third fails (judge)
        mock_client.chat.completions.create.side_effect = [
            *responses,
            RuntimeError("Judge API down"),
        ]

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                execute_tool(
                    "compare_models",
                    {
                        "model_a": "model-a",
                        "model_b": "model-b",
                        "questions": ["Q?"],
                    },
                )
            )

        assert result["summary"]["ties"] == 1
        assert result["results"][0]["winner"] == "tie"
        assert result["results"][0]["reason"] == "Judge unavailable"
