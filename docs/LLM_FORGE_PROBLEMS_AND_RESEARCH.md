# LLM Forge: Current Problems & Deep Research Prompt

## Context

LLM Forge is a conversational CLI tool where users type `llm-forge` in any terminal and get an AI assistant that helps them build custom language models. The assistant is powered by Claude API with tool use. The vision is: **user talks, Forge does everything.** Like Claude Code — the user gives instructions, the AI executes.

GitHub: https://github.com/Nagavenkatasai7/llm-forge
PyPI: pip install llm-forge-new
Install: curl -fsSL https://raw.githubusercontent.com/Nagavenkatasai7/llm-forge/main/install.sh | bash

---

## Problem 1: Forge Cannot Execute Shell Commands

### What happens now:
User says "convert this DOCX file to training data" and Forge responds with:
```
Run this command in your terminal:
  textutil -convert txt "file.docx"
```

### What should happen:
Forge should RUN the command itself and report the result:
```
Converting file.docx to text... done (42 KB extracted).
Found 156 Q&A pairs. Saving to data/nz_visa_guide.jsonl.
Ready to train. Start now?
```

### Root cause:
The tool system only has high-level tools (scan_data, write_config, start_training). There is NO tool that can execute arbitrary shell commands on the user's machine. Claude Code has a `Bash` tool — LLM Forge does not.

### What needs to change:
Add a `run_command` tool that:
- Executes shell commands (subprocess)
- Returns stdout/stderr
- Has safety guardrails (no rm -rf, no sudo, etc.)
- Asks for permission before destructive commands (configurable)
- Works on macOS, Linux, Windows

---

## Problem 2: Input Collision During Streaming

### What happens now:
When Forge is streaming a response (text appearing word-by-word), if the user starts typing, their keystrokes get mixed into the output. The terminal shows garbled text because stdin and stdout share the same terminal.

### What should happen:
Like Claude Code: when the AI is responding, user input is buffered or blocked. The user can press Esc to interrupt, but their keystrokes don't appear in the output stream. After the response finishes, the input prompt appears cleanly.

### Root cause:
The current implementation uses simple `print()` for streaming and `input()` for user input. There's no terminal raw mode handling, no input buffering during output, no separation of input/output streams.

### What needs to change:
- Use a proper terminal UI library (e.g., `prompt_toolkit`, `textual`, `blessed`, or raw `termios`)
- During AI response streaming: capture and buffer any keystrokes (don't echo them)
- Esc key detection should work during streaming (currently requires `termios` which is fragile)
- After response ends: show the buffered input or a clean prompt
- Consider a two-panel layout: output above, input below (like Claude Code)

---

## Problem 3: Forge is an Advisor, Not an Executor

### What happens now:
User: "I want to build a NZ visa expert"
Forge: "Here's a step-by-step plan... do step 1, then step 2, then come back"

### What should happen:
User: "I want to build a NZ visa expert"
Forge: [detects hardware] [asks about data] [user provides DOCX path]
Forge: [converts DOCX to text] [extracts Q&A pairs] [writes training config]
Forge: "Data prepared: 156 Q&A pairs. Training will take ~45 min. Start?"
User: "yes"
Forge: [launches training] [monitors progress] [exports to Ollama]
Forge: "Done! Run: ollama run nz-visa-expert"

### Root cause:
The system prompt says "you ARE the application, you take action" but the tools don't support the actions needed:
- No file conversion tool (DOCX→TXT, PDF→TXT)
- No shell command execution
- No file read/write tool (can't create JSONL from extracted text)
- No web scraping tool (can't fetch from immigration.govt.nz)
- No data transformation tool (raw text → structured Q&A pairs)

### What needs to change:
Add these execution tools:
1. `run_command` — Execute shell commands with safety guards
2. `read_file` — Read any file's contents
3. `write_file` — Write/create files
4. `convert_document` — DOCX/PDF/HTML → plain text
5. `fetch_url` — Download web pages (for scraping FAQ data)
6. `create_training_data` — Transform raw text into Q&A JSONL format
7. `install_package` — pip install / brew install with permission

---

## Problem 4: No File System Access

### What happens now:
Forge can scan directories (list files) but cannot:
- Read file contents
- Write new files (except YAML configs)
- Move/copy/rename files
- Convert between formats

### What should happen:
Forge should have full file system access within the project directory:
- Read any file the user points to
- Write training data files
- Convert documents
- Organize the project structure

### Root cause:
The `scan_data` tool only lists files and reads previews. There's no general-purpose file read/write capability.

---

## Problem 5: No Dependency Management at Runtime

### What happens now:
If `python-docx` isn't installed, Forge can't convert DOCX files and tells the user to install it manually.

### What should happen:
Forge detects the missing package and installs it automatically:
```
Converting DOCX... python-docx not installed. Installing... done.
Converting DOCX... extracted 42 KB of text.
```

### Root cause:
The `install_dependencies` tool only installs llm-forge extras. It doesn't handle arbitrary package installation for data processing needs.

---

## Problem 6: No Autonomous Workflow Execution

### What happens now:
Every step requires the user to go back and forth:
1. User says what they want
2. Forge explains what to do
3. User does it manually
4. User comes back
5. Forge explains the next step
6. Repeat

### What should happen:
User gives the goal. Forge executes the entire workflow autonomously:
1. User: "Build a NZ visa expert from this DOCX"
2. Forge: executes 10 steps silently, reports progress
3. Forge: "Model trained and deployed to Ollama. Try: ollama run nz-visa-expert"

### Root cause:
The agentic loop only handles one tool call at a time with Claude deciding the next step. There's no concept of a "workflow" that chains multiple actions. Claude Code solves this by giving the AI access to Bash, Read, Write, Edit, Glob, Grep tools — the AI plans and executes a multi-step workflow using these primitives.

---

## Problem 7: The Assistant Can't Learn Mid-Conversation

### What happens now:
The system prompt is fixed at session start. If the user says "I'm a complete beginner, explain everything simply", Forge might still give technical responses because the system prompt wasn't updated.

### What should happen:
The memory system should update the system prompt dynamically:
- User says "I'm a beginner" → save_memory("user_behavior", "complete beginner")
- Next response adjusts language automatically
- Mid-conversation adaptation, not just cross-session

---

## Research Questions

1. **How does Claude Code handle shell command execution?** What safety mechanisms does it use? How does it handle permissions? Can we replicate this for LLM Forge?

2. **What terminal UI libraries allow clean input/output separation during streaming?** How do tools like `rich`, `prompt_toolkit`, `textual`, or `blessed` handle concurrent stdin/stdout? What does Claude Code use?

3. **How do agentic CLI tools (Claude Code, Aider, Cursor, Continue) structure their tool systems?** What's the minimal set of tools needed for full autonomy? How do they handle:
   - File read/write
   - Shell command execution
   - Package installation
   - Web access
   - Approval gates for destructive actions

4. **What's the best architecture for an autonomous workflow engine?** Should the LLM plan all steps first then execute, or should it execute step-by-step with the LLM deciding the next action? How does Claude Code's agentic loop work?

5. **How to handle terminal input during streaming output?** Raw mode vs cooked mode, input buffering strategies, Esc key detection across platforms (macOS, Linux, Windows WSL).

6. **Can we use the Claude Agent SDK instead of raw API calls?** The Agent SDK has built-in Bash, Read, Write, Edit, Glob, Grep tools. Would switching from our custom tool system to the Agent SDK give us full execution capability out of the box?

7. **How do we add web scraping capability safely?** For use cases like "scrape immigration.govt.nz for FAQ data" — what libraries, rate limiting, and legal considerations apply?

---

## Architecture Comparison

### Current LLM Forge:
```
User → input() → Claude API (with custom tools) → print response
                   ↓
            Tools: scan_data, write_config, detect_hardware, etc.
            (high-level, no file/shell access)
```

### What it should be (like Claude Code):
```
User → Terminal UI (prompt_toolkit/textual) → Claude API (with system tools)
                                                ↓
                                         Tools:
                                           - Bash (run any command)
                                           - Read (read any file)
                                           - Write (create/overwrite files)
                                           - Edit (modify files)
                                           - Glob (find files)
                                           - Grep (search files)
                                           + Our custom tools:
                                             - detect_hardware
                                             - start_training
                                             - deploy_to_ollama
                                             - etc.
```

### Or use Claude Agent SDK directly:
```
User → Agent SDK (has Bash/Read/Write/Edit built-in)
         + Custom MCP tools for ML-specific actions
         + Memory system
         + Project awareness
```

---

## Priority Order for Fixes

1. **P0: Add Bash/shell execution tool** — Without this, Forge cannot do anything. This is the #1 blocker.
2. **P0: Add file read/write tools** — Forge needs to read data files and write training data.
3. **P1: Fix terminal input/output** — Streaming + typing collision makes the UI feel broken.
4. **P1: Add document conversion** — DOCX/PDF to text is needed for 80% of use cases.
5. **P2: Autonomous workflow chains** — Execute multi-step workflows without user intervention.
6. **P2: Dynamic system prompt updates** — Adapt to user mid-conversation.
7. **P3: Web scraping capability** — Fetch data from URLs for dataset creation.

---

## Key Decision: Build Custom Tools vs Use Agent SDK

### Option A: Add Bash/Read/Write tools to our existing system
- More work but full control
- Keep our existing 24 tools + add 5-6 system tools
- We manage safety, permissions, everything

### Option B: Switch to Claude Agent SDK
- Built-in Bash, Read, Write, Edit, Glob, Grep
- Built-in permission system
- Built-in streaming with proper terminal handling
- We add our ML tools as MCP servers
- Less code to maintain
- BUT: requires Claude Code CLI installed, ties us to Anthropic's infrastructure

### Option C: Hybrid
- Use Agent SDK for the execution layer (Bash, Read, Write)
- Keep our custom tools for ML-specific actions
- Keep our memory system
- Best of both worlds

**Recommendation: Option C (Hybrid)** — Use Agent SDK for system-level execution, keep our domain-specific tools and memory system.

---

## What Success Looks Like

```
$ llm-forge

Forge: Welcome back! What do you want to build?

You: I have a DOCX file with NZ visa info. Build me a visa expert chatbot.

Forge: Got it. Let me handle everything.

  [1/6] Converting DOCX to text... done (42 KB)
  [2/6] Extracting Q&A pairs... found 156 pairs
  [3/6] Writing training data... saved to data/nz_visa.jsonl
  [4/6] Configuring training... Qwen2.5-3B, LoRA r=8, ~45 min on your M4
  [5/6] Training... ████████████████████ 100% loss: 1.28 (45 min)
  [6/6] Deploying to Ollama... done!

  Your NZ visa expert is ready! Try it:
    ollama run nz-visa-expert

You: nice! can it handle questions about work visas?

Forge: Let me test it for you...
  Q: "What are the requirements for a NZ work visa?"
  A: "To apply for a New Zealand work visa, you need..."

  Looks good! The model correctly covers work visa requirements.
  Want me to run a full evaluation?
```

**That's the vision. Zero manual steps. User talks, Forge does.**
