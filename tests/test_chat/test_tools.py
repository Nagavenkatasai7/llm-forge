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
