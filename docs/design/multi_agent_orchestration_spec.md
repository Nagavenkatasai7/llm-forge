# Multi-Agent Orchestration Design Spec

**Author**: Claude Code (Opus 4.6) + Naga Venkata Sai Chennu
**Date**: 2026-03-25
**Status**: DRAFT — Awaiting approval before implementation
**Architecture**: Hybrid — Claude Orchestrator + Google ADK Sub-agents (Gemini)

---

## 1. Problem Statement

The current chat system (`src/llm_forge/chat/`) has a single-LLM architecture where one model
(Claude/OpenAI/NVIDIA) tries to handle everything: config writing, training monitoring, data
analysis, evaluation, web research, and deployment. Key issues:

1. **NVIDIA NIM provider has zero tool support** — `tool_calls = []` always, so the default
   free-tier path literally cannot execute actions.
2. **Single-model bottleneck** — One LLM can't be an expert at config tuning AND loss curve
   analysis AND web research AND deployment automation.
3. **No feedback loop** — The chat can't see live training metrics, detect loss divergence,
   or auto-adjust hyperparameters.
4. **No web browsing** — Can't read docs, papers, or URLs the user shares.
5. **System prompt is aspirational** — Describes capabilities (e.g., "I can monitor your
   training in real-time") that the system cannot deliver.

## 2. Solution: Hybrid Multi-Agent Architecture

```
                        ┌──────────────────────────┐
                        │         USER              │
                        │   (CLI / Gradio / TUI)    │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │     CLAUDE ORCHESTRATOR        │
                     │  (Anthropic API — Sonnet 4.6)  │
                     │                                │
                     │  - Intent detection            │
                     │  - Multi-step planning         │
                     │  - Agent delegation             │
                     │  - Result synthesis             │
                     │  - Error diagnosis              │
                     │  - Memory management            │
                     └──┬──┬──┬──┬──┬──┬─────────────┘
                        │  │  │  │  │  │
              ┌─────────┘  │  │  │  │  └─────────┐
              ▼            ▼  │  ▼  ▼            ▼
    ┌──────────────┐ ┌────────┤ ┌──────────┐ ┌──────────────┐
    │  DATA AGENT  │ │ CONFIG ││ │ TRAINING │ │  RESEARCH    │
    │  (Gemini)    │ │ AGENT  ││ │ AGENT    │ │  AGENT       │
    │              │ │(Gemini)││ │ (Gemini) │ │  (Gemini)    │
    │ - scan data  │ │        ││ │          │ │              │
    │ - validate   │ │ - write││ │ - launch │ │ - web search │
    │ - recommend  │ │ - tune ││ │ - monitor│ │ - read docs  │
    │   cleaning   │ │ - valid││ │ - diagnose│ │ - find data │
    └──────────────┘ └────────┘│ └──────────┘ └──────────────┘
                               │
                    ┌──────────┤──────────┐
                    ▼          ▼          ▼
              ┌──────────┐ ┌──────────┐ ┌──────────────┐
              │   EVAL   │ │  EXPORT  │ │  (future)    │
              │  AGENT   │ │  AGENT   │ │  More agents │
              │ (Gemini) │ │ (Gemini) │ │  as needed   │
              │          │ │          │ │              │
              │ - bench  │ │ - GGUF   │ └──────────────┘
              │ - compare│ │ - Ollama │
              │ - report │ │ - HF Hub │
              └──────────┘ └──────────┘
```

### Why Hybrid?

| Concern | Claude Orchestrator | Google ADK Sub-agents |
|---------|--------------------|-----------------------|
| **Reasoning** | Superior multi-step planning, error diagnosis | Good for focused tasks |
| **Cost** | $3/M input, $15/M output (Sonnet 4.6) | Lower cost for grunt work |
| **Speed** | Deliberate, thorough | Fast for structured tasks |
| **Tool use** | Full tool-use protocol (proven in engine.py) | ADK handles tool routing natively |
| **Session mgmt** | We manage conversation history | ADK provides Session + Runner |

## 3. Scope

### 3.1 REMOVE (Delete Files)

| File | Reason |
|------|--------|
| `chat/nvidia_provider.py` | Replaced by Gemini via ADK — no more free NVIDIA tier |
| `chat/wizard_fallback.py` | No more offline mode — API keys required |
| `chat/wizard.py` | Replaced by Data Agent + Config Agent conversation |
| `chat/system_prompt.py` | Replaced by per-agent system prompts |
| `chat/knowledge_base.py` | Knowledge embedded in agent system prompts instead |
| `chat/context_detector.py` | Claude orchestrator handles intent natively |
| `chat/input_handler.py` | Thin wrapper — inline into CLI |
| `chat/project_setup.py` | Absorbed into Config Agent |
| `tests/test_chat/test_nvidia_provider.py` | Provider removed |
| `tests/test_chat/test_wizard_fallback.py` | Wizard removed |
| `tests/test_chat/test_context_detector.py` | Detector removed |
| `tests/test_chat/test_input_handler.py` | Handler removed |

### 3.2 KEEP (No Changes)

| Component | Path | Reason |
|-----------|------|--------|
| Pipeline backend | `pipeline/` | Production-grade, 12-stage DAG — the core value |
| Config schema | `config/schema.py` | 25+ Pydantic v2 models — the contract |
| Config validator | `config/validator.py` | YAML→Pydantic validation |
| Training backend | `training/` | Finetuner, alignment, distributed |
| Data pipeline | `data/` | Loading, cleaning, preprocessing |
| Evaluation | `evaluation/` | Benchmarks, LLM judge |
| Serving/Export | `serving/` | GGUF, safetensors, Ollama, HF Hub |
| Gradio UI | `ui/` | Web dashboard |
| CLI commands | `cli.py` | `train`, `validate`, `export`, `eval` |
| Test suite | `tests/` (except removed tests) | Quality baseline |

### 3.3 MODIFY (Refactor)

| File | Changes |
|------|---------|
| `chat/engine.py` | Gut and rebuild as `OrchestratorEngine` — Claude-only, delegates to ADK agents |
| `chat/tools.py` | Split into per-agent tool sets (see Section 5) |
| `chat/memory.py` | Keep 3-layer memory, add agent-level context |
| `chat/execution.py` | Keep permission system, wire to Training/Export agents |
| `chat/tui.py` | Update to show agent activity (which agent is working) |
| `chat/slash_commands.py` | Map commands to agent invocations |
| `chat/training_monitor.py` | Wire to Training Agent as a tool |
| `cli.py` | Update `chat` command — remove provider/NVIDIA flags, add `--gemini-key` |
| `pyproject.toml` | Add `google-adk>=1.0` to `[chat]` extras |

### 3.4 ADD (New Files)

| File | Purpose |
|------|---------|
| `chat/orchestrator.py` | Claude orchestrator — intent detection, agent delegation |
| `chat/agents/__init__.py` | Agent package |
| `chat/agents/base.py` | Base agent class (wraps Google ADK Agent) |
| `chat/agents/data_agent.py` | Data scanning, validation, quality analysis |
| `chat/agents/config_agent.py` | YAML config generation, hyperparameter tuning |
| `chat/agents/training_agent.py` | Training launch, monitoring, failure detection |
| `chat/agents/eval_agent.py` | Benchmark execution, comparison reports |
| `chat/agents/export_agent.py` | GGUF/safetensors export, Ollama/HF deployment |
| `chat/agents/research_agent.py` | Web search, doc reading, dataset discovery |
| `chat/agent_tools/` | Per-agent tool definitions (moved from monolithic tools.py) |
| `tests/test_chat/test_orchestrator.py` | Orchestrator tests |
| `tests/test_chat/test_agents/` | Per-agent test suites |

## 4. Agent Definitions

### 4.1 Claude Orchestrator

**Model**: Claude Sonnet 4.6 (or user-selected Claude model)
**Role**: The brain — understands user intent, plans multi-step workflows, delegates to
specialists, synthesizes results, handles errors.

**System Prompt** (condensed):
```
You are the orchestrator of llm-forge, an LLM training platform. You have 6 specialist
agents you can delegate to. Your job is to:
1. Understand what the user wants to accomplish
2. Break it into steps
3. Delegate each step to the right agent
4. Synthesize results and present them clearly
5. Handle errors and suggest alternatives

You do NOT execute tools directly (except memory tools). You delegate to agents.
```

**Tools available to orchestrator**:
- `delegate_to_agent(agent_name, task, context)` — Invoke a sub-agent
- `save_memory(...)` / `recall_memory(...)` — Memory management
- `get_project_state()` — Current project status
- `get_session_history(...)` — Conversation context

**Delegation patterns**:
```
User: "I want to fine-tune Llama 3 on my customer support data"
Orchestrator plan:
  1. DATA_AGENT: scan user's data → format, quality, size
  2. CONFIG_AGENT: generate YAML config → model, LoRA, hyperparams
  3. [User confirms config]
  4. TRAINING_AGENT: launch training → monitor loss, detect issues
  5. EVAL_AGENT: run benchmarks → compare base vs fine-tuned
  6. EXPORT_AGENT: convert to GGUF → deploy to Ollama

User: "What's the latest research on DPO vs GRPO?"
Orchestrator plan:
  1. RESEARCH_AGENT: search papers, read docs → synthesize findings

User: "My training loss is NaN"
Orchestrator plan:
  1. TRAINING_AGENT: read logs, check config → diagnose root cause
  2. CONFIG_AGENT: suggest config fix → lower LR, gradient clipping
```

### 4.2 Data Agent

**Model**: Gemini 2.0 Flash (fast, cheap for file analysis)
**Google ADK Agent name**: `data_agent`

**System Prompt**:
```
You are the Data Agent for llm-forge. You analyze training datasets to determine format,
quality, and readiness for LLM fine-tuning. You understand: JSONL, Parquet, CSV, HuggingFace
datasets, Alpaca format, ShareGPT format, completion format, and custom formats.

When scanning data, report: format detected, sample count, column names, data preview,
estimated token count, quality issues (empty fields, duplicates, encoding problems).
```

**Tools**:
| Tool | Source | Description |
|------|--------|-------------|
| `scan_data(path)` | Existing `tools.py::_scan_data()` | Scan file/dir/HF dataset |
| `preview_data(path, n)` | New | Show first N records formatted |
| `count_tokens(path, tokenizer)` | New | Token count with specific tokenizer |
| `detect_format(path)` | Existing `tools.py::_scan_data()` (subset) | Auto-detect data format |
| `validate_data_quality(path)` | New | Check for empty fields, encoding issues, duplicates |
| `recommend_cleaning(path)` | New | Suggest cleaning config based on data analysis |
| `search_huggingface(query, type)` | Existing `tools.py::_search_huggingface()` | Search HF Hub |
| `download_dataset(dataset_id)` | New | Download HF dataset to local |

### 4.3 Config Agent

**Model**: Gemini 2.0 Flash (fast config generation)
**Google ADK Agent name**: `config_agent`

**System Prompt**:
```
You are the Config Agent for llm-forge. You generate, validate, and optimize YAML training
configurations. You have deep knowledge of the Pydantic v2 schema (LLMForgeConfig) and
all its sub-configs.

Key rules:
- LoRA r and alpha must be powers of 2. alpha = 2*r is a good default.
- Learning rate: 1e-4 for LoRA, 2e-5 for full fine-tuning, 1e-5 for conservative.
- Batch size: auto-detect based on VRAM. 8GB→bs=1, 16GB→bs=2, 24GB→bs=4, 80GB→bs=16.
- max_seq_length: at least 2048 for instruction tuning with multi-turn.
- assistant_only_loss: true for chat/instruction data (masks system/user tokens).
- New features disabled by default (enabled: false).

Always validate against the schema before returning.
```

**Tools**:
| Tool | Source | Description |
|------|--------|-------------|
| `write_config(output_path, config)` | Existing `tools.py::_write_config()` | Write YAML config |
| `validate_config(config_path)` | Existing `tools.py::_validate_config()` | Validate against schema |
| `detect_hardware()` | Existing `tools.py::_detect_hardware()` | GPU/CPU detection |
| `estimate_training(model, data, hw)` | Existing `tools.py::_estimate_training()` | VRAM/time estimation |
| `list_configs()` | Existing `tools.py::_list_configs()` | Show config templates |
| `read_schema()` | New | Return current Pydantic schema as JSON |
| `suggest_hyperparams(model, data_size, hw)` | New | Auto-tune hyperparameters |
| `diff_configs(config_a, config_b)` | New | Show differences between two configs |

### 4.4 Training Agent

**Model**: Gemini 2.0 Flash
**Google ADK Agent name**: `training_agent`

**System Prompt**:
```
You are the Training Agent for llm-forge. You launch, monitor, and diagnose training runs.
You can read live training logs, detect loss anomalies (NaN, divergence, plateaus), and
suggest fixes.

Common failure patterns you know:
- Loss NaN → learning rate too high, data has NaN values, gradient explosion
- Loss plateau → learning rate too low, data quality issue, model too small
- OOM → reduce batch size, enable gradient checkpointing, use QLoRA
- Slow training → enable bf16, increase batch size, use Flash Attention
- Gibberish output → assistant_only_loss not enabled, bad chat template
```

**Tools**:
| Tool | Source | Description |
|------|--------|-------------|
| `start_training(config_path)` | Existing `tools.py::_start_training()` | Launch pipeline |
| `check_training_status()` | Existing `tools.py::_check_training_status()` | Is training running? |
| `read_training_logs(output_dir)` | Existing `tools.py::_read_training_logs()` | Live metrics |
| `stop_training()` | New (via StopTrainingCallback) | Graceful stop |
| `read_loss_curve(output_dir)` | New | Parse trainer_state.json for loss history |
| `diagnose_failure(error_log)` | New | Analyze error + suggest fixes |
| `tail_log(log_path, n)` | New | Read last N lines of training log |
| `get_gpu_utilization()` | Existing `tools.py::_detect_hardware()` | Current GPU usage |

### 4.5 Evaluation Agent

**Model**: Gemini 2.0 Flash
**Google ADK Agent name**: `eval_agent`

**System Prompt**:
```
You are the Evaluation Agent for llm-forge. You run benchmarks (MMLU, GSM8K, IFEval,
HellaSwag, ARC, Winogrande) via lm-eval-harness, compare base vs fine-tuned models,
and generate human-readable reports.

Interpretation guide:
- MMLU: general knowledge (46% is baseline for 1B models)
- GSM8K: math reasoning (33% baseline for 1B)
- IFEval: instruction following (43% baseline for 1B)
- A drop >5% = catastrophic forgetting (bad)
- A drop <2% = acceptable trade-off
- Domain metrics matching base = conservative training (not bad, not great)
```

**Tools**:
| Tool | Source | Description |
|------|--------|-------------|
| `run_evaluation(model_path, benchmarks)` | Existing `tools.py::_run_evaluation()` | Run lm-eval |
| `compare_models(model_a, model_b, benchmarks)` | Existing `tools.py::_compare_models()` | A/B comparison |
| `test_model(model_name, questions)` | Existing `tools.py::_test_model()` | Quick chat test |
| `read_eval_results(results_dir)` | New | Parse eval JSON output |
| `generate_report(results)` | New | Markdown report with tables |
| `check_catastrophic_forgetting(base, tuned)` | New | Compare key metrics for regression |

### 4.6 Export Agent

**Model**: Gemini 2.0 Flash
**Google ADK Agent name**: `export_agent`

**System Prompt**:
```
You are the Export Agent for llm-forge. You handle model format conversion (safetensors,
GGUF, ONNX, AWQ), Ollama deployment, and HuggingFace Hub upload.

GGUF export requires llama.cpp tools. Common quantization types:
- Q4_K_M: best balance of size vs quality (recommended for 1-3B models)
- Q8_0: higher quality, larger file
- Q4_0: smallest, lowest quality

Ollama Modelfile must use `{{ range .Messages }}` for multi-turn support.
Never include `<|begin_of_text|>` — Ollama adds BOS automatically.
```

**Tools**:
| Tool | Source | Description |
|------|--------|-------------|
| `export_model(model_path, format)` | Existing `tools.py::_export_model()` | Convert format |
| `deploy_to_ollama(model_path, name)` | Existing `tools.py::_deploy_to_ollama()` | Ollama deploy |
| `deploy_to_huggingface(model_path)` | Existing `tools.py::_deploy_to_huggingface()` | HF Hub push |
| `merge_lora(base_model, adapter, output)` | New | LoRA merge |
| `show_model_info(model_path)` | Existing `tools.py::_show_model_info()` | Model metadata |
| `generate_modelfile(config)` | New (from export.py) | Create Ollama Modelfile |
| `check_ollama_running()` | New | Verify Ollama is available |

### 4.7 Research Agent

**Model**: Gemini 2.0 Flash
**Google ADK Agent name**: `research_agent`

**System Prompt**:
```
You are the Research Agent for llm-forge. You search the web, read documentation, find
datasets on HuggingFace, and summarize research papers. You help users discover:
- The right base model for their use case
- Suitable training datasets
- Best practices from recent papers
- Library documentation and API references

Always cite sources with URLs when providing information.
```

**Tools**:
| Tool | Source | Description |
|------|--------|-------------|
| `web_search(query)` | Google ADK built-in (google_search) | Web search |
| `read_url(url)` | New (via httpx/trafilatura) | Fetch and parse web page |
| `search_huggingface(query, type)` | Existing `tools.py::_search_huggingface()` | HF Hub search |
| `search_papers(query)` | New (ArXiv API) | Search academic papers |
| `summarize_text(text)` | Gemini native | Summarize long documents |
| `read_documentation(library, topic)` | New (Context7-like) | Library docs |

## 5. Tool Migration Plan

Current `tools.py` has 50+ tools in a single file. These get split:

```
chat/agent_tools/
├── __init__.py              # Tool registry
├── data_tools.py            # scan_data, preview, token count, HF search
├── config_tools.py          # write_config, validate, estimate, hardware detect
├── training_tools.py        # start_training, check_status, read_logs
├── eval_tools.py            # run_evaluation, compare_models, test_model
├── export_tools.py          # export_model, deploy_ollama, deploy_hf
├── research_tools.py        # web_search, read_url, search_papers
├── memory_tools.py          # save_memory, recall_memory (kept in orchestrator)
└── execution_tools.py       # run_command, read_file, write_file (kept, gated)
```

Each tool file exports:
1. **Tool definitions** (JSON schema for the agent's tool-use protocol)
2. **Tool implementations** (Python functions that do the actual work)

The tool implementations stay almost identical — we're just reorganizing, not rewriting.

## 6. Google ADK Integration

### 6.1 ADK Agent Structure

Each sub-agent follows this pattern:

```python
from google.adk import Agent

data_agent = Agent(
    name="data_agent",
    model="gemini-2.0-flash",
    description="Analyzes training datasets for format, quality, and readiness",
    instruction=DATA_AGENT_SYSTEM_PROMPT,
    tools=[scan_data, preview_data, count_tokens, ...],
)
```

### 6.2 ADK Runner

The orchestrator invokes sub-agents via the ADK `Runner`:

```python
from google.adk import Runner, InMemorySessionService

session_service = InMemorySessionService()
runner = Runner(
    agent=data_agent,
    app_name="llm_forge",
    session_service=session_service,
)

# Run the agent with a task
response = await runner.run(
    user_id="user",
    session_id="session_123",
    new_message=Content(parts=[Part(text="Scan the data at ./data/train.jsonl")]),
)
```

### 6.3 Orchestrator→Agent Communication

The Claude orchestrator has a meta-tool `delegate_to_agent`:

```python
{
    "name": "delegate_to_agent",
    "description": "Delegate a task to a specialist agent",
    "input_schema": {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "enum": ["data_agent", "config_agent", "training_agent",
                         "eval_agent", "export_agent", "research_agent"]
            },
            "task": {
                "type": "string",
                "description": "What the agent should do"
            },
            "context": {
                "type": "object",
                "description": "Relevant context (file paths, config, previous results)"
            }
        },
        "required": ["agent", "task"]
    }
}
```

When Claude calls `delegate_to_agent`, the orchestrator:
1. Selects the ADK agent
2. Creates/resumes a session
3. Passes the task + context
4. Collects the agent's response (may include tool calls)
5. Returns the result to Claude for synthesis

### 6.4 Agent Session Management

```python
class AgentManager:
    """Manages Google ADK agent lifecycle and sessions."""

    def __init__(self, gemini_api_key: str):
        self.session_service = InMemorySessionService()
        self.agents: dict[str, Agent] = {}
        self.runners: dict[str, Runner] = {}
        self._init_agents()

    def _init_agents(self):
        """Initialize all sub-agents."""
        self.agents = {
            "data_agent": _create_data_agent(),
            "config_agent": _create_config_agent(),
            "training_agent": _create_training_agent(),
            "eval_agent": _create_eval_agent(),
            "export_agent": _create_export_agent(),
            "research_agent": _create_research_agent(),
        }
        for name, agent in self.agents.items():
            self.runners[name] = Runner(
                agent=agent,
                app_name="llm_forge",
                session_service=self.session_service,
            )

    async def delegate(self, agent_name: str, task: str, context: dict) -> str:
        """Send a task to a sub-agent and return its response."""
        runner = self.runners[agent_name]
        # Build message with context
        message = f"{task}\n\nContext: {json.dumps(context)}"
        response = await runner.run(
            user_id="orchestrator",
            session_id=f"{agent_name}_session",
            new_message=Content(parts=[Part(text=message)]),
        )
        return self._extract_response(response)
```

## 7. New OrchestratorEngine

The `ChatEngine` gets replaced by `OrchestratorEngine`:

```python
class OrchestratorEngine:
    """Claude-powered orchestrator that delegates to ADK sub-agents."""

    def __init__(
        self,
        project_dir: str | None = None,
        model_key: str = "sonnet-4.6",
        gemini_api_key: str | None = None,
    ):
        self.model_key = model_key
        self.messages: list[dict] = []
        self.memory = MemoryManager(project_dir=project_dir or ".")
        self.agent_manager = AgentManager(
            gemini_api_key=gemini_api_key or os.environ["GOOGLE_API_KEY"]
        )
        self._client = _get_anthropic_client()
        self.system = self._build_system_prompt()

        # UI callbacks
        self.on_agent_start: Callable | None = None  # (agent_name, task) -> None
        self.on_agent_end: Callable | None = None     # (agent_name, result) -> None

    def send(self, user_input: str, on_text=None, interrupt_check=None) -> str:
        """Send user message through Claude orchestrator.

        Claude decides which agents to invoke, in what order, and synthesizes
        their results into a coherent response.
        """
        self.messages.append({"role": "user", "content": user_input})

        while True:
            response = _stream_anthropic(
                self.messages, self.system,
                client=self._client, on_text=on_text,
                interrupt_check=interrupt_check, model_key=self.model_key,
            )
            text, tool_calls = self._parse_response(response)

            if not tool_calls:
                self.messages.append({"role": "assistant", "content": text})
                return text

            # Handle tool calls (delegation or memory)
            self._handle_tool_calls(response, tool_calls)

    async def _handle_delegation(self, agent_name: str, task: str, context: dict) -> str:
        """Delegate to a sub-agent and return result."""
        if self.on_agent_start:
            self.on_agent_start(agent_name, task)

        result = await self.agent_manager.delegate(agent_name, task, context)

        if self.on_agent_end:
            self.on_agent_end(agent_name, result)

        return result
```

## 8. Dependency Changes

### pyproject.toml additions

```toml
[project.optional-dependencies]
chat = [
    "anthropic>=0.40",       # Claude orchestrator (REQUIRED)
    "google-adk>=1.0",       # Google ADK for sub-agents (REQUIRED)
    "google-genai>=1.0",     # Gemini API client
    "openai>=1.50",          # REMOVED from chat extras (no longer needed)
    "prompt_toolkit>=3.0",   # Keep for CLI input
    "textual>=1.0",          # Keep for TUI
    "httpx>=0.27",           # For Research Agent web requests
    "trafilatura>=1.12",     # For Research Agent HTML→text
]
```

### Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Yes | Claude orchestrator |
| `GOOGLE_API_KEY` | Yes | Gemini sub-agents via ADK |
| `WANDB_API_KEY` | No | Experiment tracking |
| `HF_TOKEN` | No | HuggingFace Hub access |

## 9. Implementation Phases

### Phase 1: Foundation (Week 1)

**Goal**: Orchestrator + 2 agents working end-to-end

1. Install `google-adk` and verify Gemini API connectivity
2. Create `chat/agents/base.py` — base agent wrapper
3. Create `chat/agents/data_agent.py` — migrate data tools
4. Create `chat/agents/config_agent.py` — migrate config tools
5. Create `chat/orchestrator.py` — Claude orchestrator with `delegate_to_agent`
6. Refactor `chat/engine.py` → `OrchestratorEngine`
7. Update CLI to use `OrchestratorEngine`
8. Write tests for orchestrator + 2 agents
9. Verify: User can say "scan my data and generate a config" end-to-end

### Phase 2: Training + Eval (Week 2)

**Goal**: Full training lifecycle through agents

1. Create `chat/agents/training_agent.py` — migrate training tools
2. Create `chat/agents/eval_agent.py` — migrate eval tools
3. Add live training monitoring to Training Agent
4. Add benchmark comparison to Eval Agent
5. Wire Training Agent → loss curve analysis → auto-diagnosis
6. Write tests
7. Verify: User can train, monitor, and evaluate through conversation

### Phase 3: Export + Research (Week 3)

**Goal**: Complete agent suite

1. Create `chat/agents/export_agent.py` — migrate export tools
2. Create `chat/agents/research_agent.py` — web search, docs, papers
3. Add web browsing capability to Research Agent
4. Wire Export Agent → Ollama deployment → model testing
5. Write tests
6. Verify: Full end-to-end workflow (data → config → train → eval → export → deploy)

### Phase 4: Cleanup + Polish (Week 4)

**Goal**: Remove old code, update UI, stabilize

1. Delete removed files (nvidia_provider, wizard, etc.)
2. Update Gradio UI to show agent activity
3. Update TUI with agent status indicators
4. Update system prompts based on testing
5. Run full test suite — establish new baseline
6. Update CLAUDE.md and documentation
7. Performance optimization (agent startup time, session reuse)

## 10. Testing Strategy

### Unit Tests (per agent)

```python
# tests/test_chat/test_agents/test_data_agent.py
def test_data_agent_scans_jsonl():
    """Data agent can scan a JSONL file and return format info."""

def test_data_agent_detects_format():
    """Data agent correctly identifies alpaca vs sharegpt vs completion format."""

def test_data_agent_counts_tokens():
    """Data agent returns accurate token counts."""
```

### Integration Tests (orchestrator + agents)

```python
# tests/test_chat/test_orchestrator.py
def test_orchestrator_delegates_data_scan():
    """Orchestrator routes 'scan my data' to data_agent."""

def test_orchestrator_chains_data_then_config():
    """Orchestrator chains data_agent → config_agent for 'prepare training'."""

def test_orchestrator_handles_agent_failure():
    """Orchestrator gracefully handles sub-agent errors."""
```

### Mock Strategy

- **Mock Gemini API** for unit tests (no real API calls)
- **Mock Claude API** for orchestrator unit tests
- **Integration tests** can use real APIs with small payloads (gated by env var)

## 11. Error Handling

### Agent Failure

If a sub-agent fails (API error, timeout, bad response):
1. Orchestrator catches the error
2. Logs the failure with context
3. Tells the user: "The [agent] encountered an issue: [error]. Let me try an alternative approach."
4. Falls back to direct tool execution or suggests manual action

### API Key Missing

```python
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ANTHROPIC_API_KEY required. Get one at: https://console.anthropic.com/")
    sys.exit(1)

if not os.environ.get("GOOGLE_API_KEY"):
    print("GOOGLE_API_KEY required. Get one at: https://aistudio.google.com/apikey")
    sys.exit(1)
```

### Rate Limiting

- Claude: 4000 RPM (Sonnet) — unlikely to hit
- Gemini: 1500 RPM (Flash) — sub-agents share the quota
- Implement exponential backoff in `AgentManager.delegate()`

## 12. Migration Path for Existing Users

1. Users currently using NVIDIA free tier → Must set `ANTHROPIC_API_KEY` + `GOOGLE_API_KEY`
2. Users using `llm-forge wizard` → Use the new conversational flow instead
3. CLI users → `llm-forge train --config x.yaml` unchanged (no agent system needed)
4. Gradio users → UI updated to show agent activity

## 13. Cost Estimation

Per typical training session (scan data → config → train → eval → export):

| Agent | Calls | Tokens/call | Cost/call | Total |
|-------|-------|-------------|-----------|-------|
| Claude Orchestrator | ~10 | ~2K in, ~1K out | ~$0.02 | ~$0.20 |
| Data Agent (Gemini Flash) | ~3 | ~1K in, ~500 out | ~$0.001 | ~$0.003 |
| Config Agent (Gemini Flash) | ~5 | ~2K in, ~1K out | ~$0.002 | ~$0.01 |
| Training Agent (Gemini Flash) | ~10 | ~1K in, ~500 out | ~$0.001 | ~$0.01 |
| Eval Agent (Gemini Flash) | ~3 | ~2K in, ~1K out | ~$0.002 | ~$0.006 |
| Export Agent (Gemini Flash) | ~2 | ~1K in, ~500 out | ~$0.001 | ~$0.002 |
| Research Agent (Gemini Flash) | ~5 | ~3K in, ~1K out | ~$0.003 | ~$0.015 |
| **Total per session** | | | | **~$0.25** |

Gemini 2.0 Flash is extremely cheap. The dominant cost is Claude orchestrator at ~$0.20/session.

---

## Appendix A: File Tree (After Implementation)

```
src/llm_forge/chat/
├── __init__.py
├── orchestrator.py          # NEW — Claude orchestrator engine
├── engine.py                # MODIFIED — thin wrapper around OrchestratorEngine
├── memory.py                # KEPT — 3-layer memory system
├── execution.py             # KEPT — permission-gated execution
├── tui.py                   # MODIFIED — agent activity display
├── ui.py                    # KEPT — UI routing
├── slash_commands.py        # MODIFIED — route to agents
├── training_monitor.py      # KEPT — wired to Training Agent
├── agents/
│   ├── __init__.py
│   ├── base.py              # Base agent wrapper (Google ADK)
│   ├── data_agent.py
│   ├── config_agent.py
│   ├── training_agent.py
│   ├── eval_agent.py
│   ├── export_agent.py
│   └── research_agent.py
└── agent_tools/
    ├── __init__.py
    ├── data_tools.py
    ├── config_tools.py
    ├── training_tools.py
    ├── eval_tools.py
    ├── export_tools.py
    ├── research_tools.py
    ├── memory_tools.py
    └── execution_tools.py
```

## Appendix B: Removed Files

```
DELETED:
  src/llm_forge/chat/nvidia_provider.py
  src/llm_forge/chat/wizard_fallback.py
  src/llm_forge/chat/wizard.py
  src/llm_forge/chat/system_prompt.py
  src/llm_forge/chat/knowledge_base.py
  src/llm_forge/chat/context_detector.py
  src/llm_forge/chat/input_handler.py
  src/llm_forge/chat/project_setup.py
  src/llm_forge/wizard.py
  tests/test_chat/test_nvidia_provider.py
  tests/test_chat/test_wizard_fallback.py
  tests/test_chat/test_context_detector.py
  tests/test_chat/test_input_handler.py
```
