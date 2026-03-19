# LLM Forge Intelligent Memory System — Design Spec

## Problem

LLM Forge's chat assistant is stateless. Every session starts fresh. The conversation context grows unbounded until it hits limits. The assistant has no awareness of the project directory, past training runs, or user preferences. A non-developer user shouldn't have to re-explain their setup every time.

## Solution

A three-layer memory system stored in `.llmforge/` that makes the assistant intelligent across sessions and resilient during long conversations. The user never interacts with the memory directly — Forge manages it autonomously.

## Architecture

### Storage: `.llmforge/` directory

```
.llmforge/
  memory.db          — SQLite: sessions, training history, user profile, preferences
  project_state.json — Auto-scanned: configs, models, data files in project dir
```

Single SQLite database for all persistent state. One auto-generated JSON for project scanning (rebuilt every startup).

### Layer 1: Working Memory (conversation context management)

**Purpose**: Keep the current conversation coherent even during very long sessions.

**How it works**:
- Track token count of the messages list
- When approaching 80% of context window (~160K tokens for Claude):
  1. Summarize the oldest 60% of messages into a compact summary
  2. Store the summary in SQLite (`session_summaries` table)
  3. Replace those messages with a single system message containing the summary
  4. Keep the most recent 40% of messages in full detail
- All tool results from the current active workflow are preserved (never summarized away)
- Use Claude API's compaction beta when available as an enhancement

**New tools for the manager**:
- `save_memory`: Store a key insight about the user or project (called by Claude proactively)
- `recall_memory`: Search past memories by topic (called when user references past work)

### Layer 2: Project Memory (auto-scanned on startup)

**Purpose**: Forge knows what's in the project directory without being told.

**What gets scanned**:
- `configs/` — all YAML configs with their key settings
- `outputs/` — trained models with size, date, training config used
- Training logs — latest loss values, whether training is complete
- Data files — JSONL/CSV/TXT with sample counts and formats
- GGUF files — exported quantized models
- `config.yaml` in project root (active config)

**Injection**: The scanned state is injected into the system prompt at session start:

```
## Your Current Project State
Hardware: Apple M4 Pro, 36 GB RAM, MPS GPU
Configs: 3 found (finance_bot.yaml, legal_summarizer.yaml, quickstart_tiny.yaml)
Models: 2 trained (finance-bot-v1 in outputs/finance-bot/, legal-v1 in outputs/legal/)
Active training: None
Data: 2 datasets (data/faqs.jsonl — 500 samples, data/contracts/ — 200 PDFs)
```

### Layer 3: Long-Term Memory (persistent across sessions)

**Purpose**: Forge learns who the user is and what works for them.

**SQLite tables**:

```sql
-- User profile (hardware, preferences, skill level)
CREATE TABLE user_profile (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP
);

-- Session history with summaries
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    summary TEXT,
    turns INTEGER,
    tokens_used INTEGER
);

-- Training run history
CREATE TABLE training_runs (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    config_path TEXT,
    model_name TEXT,
    base_model TEXT,
    mode TEXT,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    final_loss REAL,
    eval_loss REAL,
    status TEXT,
    output_dir TEXT,
    notes TEXT
);

-- Key memories (things Forge should remember)
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    category TEXT,
    content TEXT,
    created_at TIMESTAMP,
    relevance_score REAL DEFAULT 1.0
);
```

**Categories for memories**:
- `user_preference`: "User prefers LoRA over QLoRA", "User wants terse responses"
- `project_decision`: "Using Llama-3.2-1B for this project because of VRAM limits"
- `training_lesson`: "LoRA r=16 caused forgetting on this dataset, switched to r=8"
- `user_behavior`: "User is a non-developer, needs simple explanations"

### New Tools (added to chat/tools.py)

| Tool | Purpose | Called by |
|------|---------|----------|
| `save_memory` | Store an insight about user/project | Claude (proactively) |
| `recall_memory` | Search past memories by keyword | Claude (when user references past) |
| `get_project_state` | Return the auto-scanned project state | Claude (at session start) |
| `get_session_history` | Get summaries of past sessions | Claude (when resuming) |
| `log_training_run` | Record a training run's outcome | Claude (after training completes) |

### Context Compaction Strategy

For a session approaching context limits:

```
Messages: [msg1, msg2, ..., msg50, msg51, ..., msg100]
                                    ↑ compaction point

Before compaction (approaching limit):
  System prompt + project state + 100 full messages = ~180K tokens

After compaction:
  System prompt + project state
  + [SUMMARY of msg1-msg60: "User wanted a finance chatbot. We detected
     RTX 4090, chose Llama-3.2-1B with LoRA r=8. Config saved to
     finance_bot.yaml. Training started, reached loss 1.42 at step 200."]
  + msg61-msg100 (recent messages in full)
  = ~60K tokens
```

The user sees no difference. Forge stays coherent.

### Session Lifecycle

```
1. User types 'llm-forge'
2. Forge initializes:
   a. Load/create .llmforge/memory.db
   b. Scan project directory → project_state.json
   c. Load user profile from DB
   d. Load last session summary from DB
   e. Build system prompt with all context
3. Forge greets user with awareness:
   "Welcome back! Your finance chatbot trained to loss 1.34.
    Want to evaluate it or start something new?"
4. Conversation proceeds with full context
5. During conversation:
   - Forge proactively calls save_memory for important insights
   - Token count monitored, compaction triggered when needed
6. On exit (quit/Ctrl+C):
   a. Summarize current session
   b. Store session summary in DB
   c. Update user profile if learned something new
```

## Files to Create/Modify

### New files:
- `src/llm_forge/chat/memory.py` — MemoryManager class (SQLite + project scanner + compaction)

### Modified files:
- `src/llm_forge/chat/engine.py` — Integrate MemoryManager into conversation loop
- `src/llm_forge/chat/tools.py` — Add 5 new memory tools
- `src/llm_forge/chat/system_prompt.py` — Dynamic system prompt with project state
- `src/llm_forge/chat/ui.py` — Graceful shutdown saves session

## Success Criteria

1. User closes and reopens llm-forge — Forge remembers their hardware, last project, training history
2. A 2-hour conversation session doesn't crash or lose coherence
3. Forge proactively uses past training lessons ("Last time r=16 caused forgetting...")
4. Non-developer user never has to understand or touch .llmforge/ directory
5. Project directory scan takes <2 seconds on a typical project
