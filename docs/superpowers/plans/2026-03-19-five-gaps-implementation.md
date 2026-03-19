# LLM Forge — Five Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan. All 5 tasks are independent and can run in parallel.

**Goal:** Close the 5 remaining gaps to make LLM Forge production-ready for non-developer users.

**Architecture:** Each gap is an independent module that plugs into the existing chat/CLI system. No cross-dependencies between the 5 tasks.

**Tech Stack:** Python 3.10+, Typer, Rich, SQLite, Anthropic SDK, pytest

---

## Task 1: Free Guided Wizard Fallback

**Goal:** When no API key is set, `llm-forge` launches an interactive guided wizard instead of showing an error. The wizard walks users through building an LLM using predefined decision trees — no LLM API needed.

**Files:**
- Create: `src/llm_forge/chat/wizard_fallback.py`
- Modify: `src/llm_forge/chat/ui.py` (replace "No API key" error with wizard launch)
- Test: `tests/test_chat/test_wizard_fallback.py`

---

## Task 2: Chat Module Tests

**Goal:** Comprehensive test coverage for the chat module: engine, tools, memory, and UI.

**Files:**
- Create: `tests/test_chat/__init__.py`
- Create: `tests/test_chat/test_memory.py`
- Create: `tests/test_chat/test_tools.py`
- Create: `tests/test_chat/test_engine.py`

---

## Task 3: One-Command Quickstart Demo

**Goal:** `llm-forge demo` trains a tiny model (SmolLM2-135M) end-to-end in ~5 minutes using included sample data, then lets the user chat with it. Zero configuration needed.

**Files:**
- Modify: `src/llm_forge/cli.py` (add `demo` command)
- Test: `tests/test_cli_demo.py`

---

## Task 4: Live Training Progress Streaming

**Goal:** When training is running, the chat assistant can show real-time progress updates by tailing the training logs with a background monitor.

**Files:**
- Create: `src/llm_forge/chat/training_monitor.py`
- Modify: `src/llm_forge/chat/tools.py` (enhance `start_training` to use monitor)
- Test: `tests/test_chat/test_training_monitor.py`

---

## Task 5: Cost and Time Estimation

**Goal:** Before training starts, estimate how long it will take and whether the model fits the hardware. Add an `estimate_training` tool that the manager calls automatically before `start_training`.

**Files:**
- Modify: `src/llm_forge/chat/tools.py` (add `estimate_training` tool)
- Modify: `src/llm_forge/chat/system_prompt.py` (tell Claude to always estimate before training)
- Test: `tests/test_chat/test_estimation.py`
