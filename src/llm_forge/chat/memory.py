"""Intelligent memory system for LLM Forge.

Three-layer memory:
  Layer 1: Working Memory — conversation context compaction
  Layer 2: Project Memory — auto-scanned project directory state
  Layer 3: Long-Term Memory — persistent SQLite across sessions
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLMFORGE_DIR = ".llmforge"
DB_NAME = "memory.db"
PROJECT_STATE_FILE = "project_state.json"

# Compaction triggers when message token estimate exceeds this
COMPACTION_THRESHOLD_TOKENS = 120_000
# Keep this many recent messages after compaction
KEEP_RECENT_MESSAGES = 20
# Rough token estimate: 4 chars per token
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------


class MemoryManager:
    """Manages all three memory layers for the LLM Forge assistant."""

    def __init__(self, project_dir: str | Path | None = None):
        self.project_dir = Path(project_dir or ".").resolve()
        self.llmforge_dir = self.project_dir / LLMFORGE_DIR
        self.llmforge_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.llmforge_dir / DB_NAME
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start = datetime.now()

        self._init_db()
        self.project_state = self._scan_project()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TIMESTAMP,
                    ended_at TIMESTAMP,
                    summary TEXT,
                    turns INTEGER DEFAULT 0,
                    tokens_used INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    relevance_score REAL DEFAULT 1.0
                );
                """
            )

        # Register this session
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, started_at) VALUES (?, ?)",
                (self.session_id, self.session_start.isoformat()),
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    # ------------------------------------------------------------------
    # Layer 2: Project scanning
    # ------------------------------------------------------------------

    def _scan_project(self) -> dict:
        """Scan the project directory for configs, models, data, and training state."""
        state: dict = {
            "scanned_at": datetime.now().isoformat(),
            "project_dir": str(self.project_dir),
            "configs": [],
            "trained_models": [],
            "data_sources": [],
            "active_training": None,
        }

        # Scan configs
        configs_dir = self.project_dir / "configs"
        if configs_dir.exists():
            for f in sorted(configs_dir.glob("*.yaml")):
                try:
                    import yaml

                    with open(f) as fh:
                        raw = yaml.safe_load(fh)
                    if raw and isinstance(raw, dict):
                        state["configs"].append(
                            {
                                "name": f.name,
                                "model": raw.get("model", {}).get("name", "?"),
                                "mode": raw.get("training", {}).get("mode", "?"),
                            }
                        )
                except Exception:
                    state["configs"].append({"name": f.name, "error": "parse_failed"})

        # Scan outputs for trained models
        outputs_dir = self.project_dir / "outputs"
        if outputs_dir.exists():
            for d in sorted(outputs_dir.iterdir()):
                if not d.is_dir():
                    continue
                model_info: dict = {"name": d.name, "path": str(d)}

                # Check if training is complete
                if (d / "model.safetensors").exists() or (
                    d / "merged" / "model.safetensors"
                ).exists():
                    model_info["status"] = "complete"
                    safetensors = list(d.rglob("*.safetensors"))
                    if safetensors:
                        model_info["size_mb"] = round(
                            sum(f.stat().st_size for f in safetensors) / (1024 * 1024), 1
                        )
                elif list(d.glob("checkpoint-*")):
                    model_info["status"] = "has_checkpoints"
                    model_info["checkpoints"] = len(list(d.glob("checkpoint-*")))
                else:
                    model_info["status"] = "empty"

                # Check for GGUF
                gguf_files = list(d.rglob("*.gguf"))
                if gguf_files:
                    model_info["gguf"] = [f.name for f in gguf_files]

                # Read trainer state for metrics
                trainer_state = d / "trainer_state.json"
                if not trainer_state.exists():
                    trainer_state = d / "merged" / "trainer_state.json"
                if trainer_state.exists():
                    try:
                        ts = json.loads(trainer_state.read_text())
                        log_history = ts.get("log_history", [])
                        if log_history:
                            last = log_history[-1]
                            model_info["last_loss"] = last.get("loss", last.get("train_loss"))
                            model_info["last_step"] = last.get("step")
                    except Exception:
                        pass

                state["trained_models"].append(model_info)

        # Scan for data files
        data_dir = self.project_dir / "data"
        if data_dir.exists():
            for f in data_dir.rglob("*"):
                if f.is_file() and f.suffix.lower() in (
                    ".jsonl",
                    ".json",
                    ".csv",
                    ".txt",
                    ".parquet",
                ):
                    state["data_sources"].append(
                        {
                            "name": f.name,
                            "path": str(f.relative_to(self.project_dir)),
                            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                        }
                    )

        # Check for active config in project root
        root_config = self.project_dir / "config.yaml"
        if root_config.exists():
            state["active_config"] = "config.yaml"

        # Save project state
        state_path = self.llmforge_dir / PROJECT_STATE_FILE
        state_path.write_text(json.dumps(state, indent=2))

        return state

    # ------------------------------------------------------------------
    # Layer 3: Long-term memory operations
    # ------------------------------------------------------------------

    def save_memory(self, category: str, content: str, relevance: float = 1.0) -> str:
        """Store a memory. Called by Claude proactively."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memories (category, content, relevance_score) VALUES (?, ?, ?)",
                (category, content, relevance),
            )
            conn.commit()
        return json.dumps({"status": "ok", "message": f"Memory saved: {category}"})

    def recall_memory(self, query: str, limit: int = 10) -> str:
        """Search memories by keyword. Returns matching memories."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT category, content, created_at, relevance_score "
                "FROM memories WHERE content LIKE ? OR category LIKE ? "
                "ORDER BY relevance_score DESC, created_at DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()

        memories = [
            {"category": r[0], "content": r[1], "created_at": r[2], "relevance": r[3]} for r in rows
        ]
        return json.dumps({"memories": memories, "count": len(memories)})

    def get_session_history(self, limit: int = 5) -> str:
        """Get recent session summaries."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, started_at, ended_at, summary, turns "
                "FROM sessions WHERE summary IS NOT NULL "
                "ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        sessions = [
            {
                "id": r[0],
                "started": r[1],
                "ended": r[2],
                "summary": r[3],
                "turns": r[4],
            }
            for r in rows
        ]
        return json.dumps({"sessions": sessions, "count": len(sessions)})

    def log_training_run(
        self,
        config_path: str,
        model_name: str,
        base_model: str,
        mode: str,
        output_dir: str,
        final_loss: float | None = None,
        eval_loss: float | None = None,
        status: str = "started",
        notes: str = "",
    ) -> str:
        """Record a training run."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO training_runs "
                "(session_id, config_path, model_name, base_model, mode, "
                "started_at, final_loss, eval_loss, status, output_dir, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    config_path,
                    model_name,
                    base_model,
                    mode,
                    datetime.now().isoformat(),
                    final_loss,
                    eval_loss,
                    status,
                    output_dir,
                    notes,
                ),
            )
            conn.commit()
        return json.dumps({"status": "ok", "message": "Training run logged"})

    def get_user_profile(self) -> dict:
        """Get full user profile."""
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM user_profile").fetchall()
        return {r[0]: r[1] for r in rows}

    def set_user_profile(self, key: str, value: str) -> None:
        """Set a user profile value."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_profile (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, datetime.now().isoformat()),
            )
            conn.commit()

    def get_training_history(self, limit: int = 10) -> list[dict]:
        """Get recent training runs."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT model_name, base_model, mode, started_at, final_loss, "
                "eval_loss, status, output_dir, notes "
                "FROM training_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "model_name": r[0],
                "base_model": r[1],
                "mode": r[2],
                "started_at": r[3],
                "final_loss": r[4],
                "eval_loss": r[5],
                "status": r[6],
                "output_dir": r[7],
                "notes": r[8],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Layer 1: Context compaction
    # ------------------------------------------------------------------

    def estimate_tokens(self, messages: list[dict]) -> int:
        """Rough token estimate for a message list."""
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total_chars += len(json.dumps(block))
                    else:
                        total_chars += len(str(block))
            else:
                total_chars += len(str(content))
        return total_chars // CHARS_PER_TOKEN

    def needs_compaction(self, messages: list[dict]) -> bool:
        """Check if messages need compaction."""
        return self.estimate_tokens(messages) > COMPACTION_THRESHOLD_TOKENS

    def compact_messages(self, messages: list[dict], client=None) -> list[dict]:
        """Compact older messages into a summary, keeping recent ones.

        If a Claude client is provided, uses Claude to generate the summary.
        Otherwise, uses a simple extraction approach.
        """
        if not self.needs_compaction(messages):
            return messages

        # Split: older messages to summarize, recent to keep
        split_point = max(len(messages) - KEEP_RECENT_MESSAGES, 0)
        old_messages = messages[:split_point]
        recent_messages = messages[split_point:]

        # Generate summary
        if client and old_messages:
            summary = self._summarize_with_claude(old_messages, client)
        else:
            summary = self._summarize_simple(old_messages)

        # Store the summary
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET summary = ? WHERE id = ?",
                (summary, self.session_id),
            )
            conn.commit()

        # Build compacted message list
        compacted = [
            {
                "role": "user",
                "content": f"[CONVERSATION CONTEXT - Earlier in this session, here's what happened:\n{summary}\n\nNow continuing the conversation:]",
            },
            {
                "role": "assistant",
                "content": "Understood. I have the context from our earlier discussion. Let's continue.",
            },
        ]
        compacted.extend(recent_messages)
        return compacted

    def _summarize_with_claude(self, messages: list[dict], client) -> str:
        """Use Claude to summarize older messages."""
        # Build a text representation of old messages
        text_parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                text_parts.append(f"{role}: {content[:500]}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(f"{role}: {block.get('text', '')[:500]}")
                    elif isinstance(block, dict) and block.get("type") == "tool_result":
                        text_parts.append(f"tool_result: {str(block.get('content', ''))[:200]}")

        conversation_text = "\n".join(text_parts[-50:])  # Last 50 entries max

        try:
            response = client.messages.create(
                # claude-haiku-4-5 is the correct model ID (released after
                # training data cutoff; verified against current Anthropic API).
                model="claude-haiku-4-5",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Summarize this conversation between a user and an LLM training assistant. "
                            "Focus on: what the user wants to build, what hardware they have, what data "
                            "they're using, what config was created, training status, and any key decisions. "
                            "Be concise but include all important details.\n\n"
                            f"{conversation_text}"
                        ),
                    }
                ],
            )
            return response.content[0].text
        except Exception:
            return self._summarize_simple(messages)

    def _summarize_simple(self, messages: list[dict]) -> str:
        """Simple extraction-based summary (no API call)."""
        summary_parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and msg.get("role") == "assistant":
                # Extract first sentence of assistant messages
                first_sentence = content.split(".")[0] + "." if "." in content else content[:100]
                if len(first_sentence) > 20:
                    summary_parts.append(first_sentence)

        return " | ".join(summary_parts[-10:]) or "Previous conversation context unavailable."

    # ------------------------------------------------------------------
    # Build dynamic system prompt context
    # ------------------------------------------------------------------

    def build_context_block(self) -> str:
        """Build the dynamic context block injected into the system prompt."""
        parts = []

        # User profile
        profile = self.get_user_profile()
        if profile:
            parts.append("## User Profile")
            for key, value in profile.items():
                parts.append(f"- {key}: {value}")

        # Project state
        ps = self.project_state
        parts.append("\n## Current Project State")
        parts.append(f"Project directory: {ps['project_dir']}")

        if ps["configs"]:
            parts.append(f"Configs: {len(ps['configs'])} found")
            for c in ps["configs"][:5]:
                parts.append(
                    f"  - {c['name']} (model: {c.get('model', '?')}, mode: {c.get('mode', '?')})"
                )

        if ps["trained_models"]:
            parts.append(f"Trained models: {len(ps['trained_models'])} found")
            for m in ps["trained_models"][:5]:
                status = m.get("status", "?")
                loss = m.get("last_loss")
                loss_str = f", loss: {loss:.3f}" if loss else ""
                parts.append(f"  - {m['name']} ({status}{loss_str})")

        if ps["data_sources"]:
            parts.append(f"Data files: {len(ps['data_sources'])} found")
            for d in ps["data_sources"][:5]:
                parts.append(f"  - {d['name']} ({d['size_mb']} MB)")

        if ps.get("active_config"):
            parts.append(f"Active config: {ps['active_config']}")

        # Training history
        history = self.get_training_history(limit=3)
        if history:
            parts.append("\n## Recent Training History")
            for h in history:
                loss_str = f"loss={h['final_loss']:.3f}" if h["final_loss"] else "in progress"
                parts.append(f"  - {h['model_name']} ({h['mode']}, {loss_str}, {h['status']})")

        # Recent memories
        rows = []
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT category, content FROM memories "
                    "ORDER BY relevance_score DESC, created_at DESC LIMIT 10"
                ).fetchall()
        except Exception:
            pass

        if rows:
            parts.append("\n## Key Memories")
            for category, content in rows:
                parts.append(f"- [{category}] {content}")

        # Last session summary
        last_session = None
        try:
            with self._connect() as conn:
                last_session = conn.execute(
                    "SELECT summary FROM sessions WHERE summary IS NOT NULL AND id != ? "
                    "ORDER BY started_at DESC LIMIT 1",
                    (self.session_id,),
                ).fetchone()
        except Exception:
            pass

        if last_session and last_session[0]:
            parts.append(f"\n## Last Session Summary\n{last_session[0]}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def end_session(self, messages: list[dict], client=None) -> None:
        """End the current session: summarize and save."""
        # Count turns
        turns = sum(1 for m in messages if m.get("role") == "user")

        # Generate summary
        if client and len(messages) > 4:
            summary = self._summarize_with_claude(messages, client)
        else:
            summary = self._summarize_simple(messages)

        # Update session record
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at = ?, summary = ?, turns = ?, tokens_used = ? WHERE id = ?",
                (
                    datetime.now().isoformat(),
                    summary,
                    turns,
                    self.estimate_tokens(messages),
                    self.session_id,
                ),
            )
            conn.commit()
