# CLAUDE.md — llm-forge Master Development Prompt

> **What is this file?** This is the master instruction file for Claude Code. When you open Claude Code in the llm-forge project directory, Claude Code automatically reads this file and follows every instruction in it. Think of it as the "brain configuration" for your AI coding assistant — it tells Claude Code exactly how to think, what tools to use, when to use them, and how to execute every task.
>
> **Why does this matter?** Without this file, Claude Code operates with generic behavior. With this file, Claude Code becomes a specialized ML platform engineer that follows a proven 8-step methodology, uses MCP tools for verification at every step, and never skips quality checks.

---

## TABLE OF CONTENTS

1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [MCP Tools — Complete Installation Guide](#2-mcp-tools--complete-installation-guide)
3. [Claude Code Configuration Files](#3-claude-code-configuration-files)
4. [Core Execution Mindset](#4-core-execution-mindset)
5. [The 8-Step Feature Development Lifecycle](#5-the-8-step-feature-development-lifecycle)
6. [MCP Tool Decision Matrix](#6-mcp-tool-decision-matrix)
7. [Claude Code Hooks Configuration](#7-claude-code-hooks-configuration)
8. [Custom Slash Commands](#8-custom-slash-commands)
9. [Testing Methodology for ML Code](#9-testing-methodology-for-ml-code)
10. [Agent Deployment Patterns](#10-agent-deployment-patterns)
11. [Anti-Patterns to Avoid](#11-anti-patterns-to-avoid)
12. [Branching & Versioning Strategy](#12-branching--versioning-strategy)
13. [Memory File Conventions](#13-memory-file-conventions)
14. [Troubleshooting Common MCP Issues](#14-troubleshooting-common-mcp-issues)
15. [Execution Checklist Template](#15-execution-checklist-template)

---

## 1. PREREQUISITES & ENVIRONMENT SETUP

Before installing any MCP tools, you need these foundations in place. If any of these are missing, the MCP servers will fail to start.

### 1.1 Required Software

**Node.js (v18 or higher) — Required for all npx-based MCP servers**

Most MCP servers are distributed as npm packages and launched via `npx`. Without Node.js, none of them will work.

```bash
# Check if Node.js is installed and what version
node --version
# You need v18.0.0 or higher. If not installed or outdated:

# Option A: Install via nvm (recommended — lets you switch versions)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc   # or source ~/.zshrc on macOS
nvm install 20     # Install Node.js 20 (LTS)
nvm use 20         # Activate it
node --version     # Verify: should show v20.x.x

# Option B: Install directly from nodejs.org
# Download from https://nodejs.org/ and run the installer
```

**Python (3.10 or higher) — Required for Python-based MCP servers and our project**

```bash
# Check Python version
python --version   # or python3 --version
# You need 3.10+. If not installed:

# macOS:
brew install python@3.12

# Ubuntu/Debian:
sudo apt update && sudo apt install python3.12 python3.12-venv python3-pip

# Verify
python3 --version  # Should show 3.10+
```

**uv (Python package runner) — Required for uvx-based MCP servers**

Some MCP servers use `uvx` instead of `npx`. `uvx` is the Python equivalent — it runs Python packages without permanent installation.

```bash
# Install uv (the fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or source ~/.zshrc

# Verify
uv --version       # Should show uv 0.x.x
uvx --version      # Should work after uv is installed
```

**Claude Code CLI — The AI assistant itself**

```bash
# Install Claude Code globally
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version

# If you get "command not found":
# Check where npm installs global packages
npm config get prefix
# Add that path's /bin to your PATH:
export PATH="$(npm config get prefix)/bin:$PATH"
# Add this line to your ~/.bashrc or ~/.zshrc to make it permanent
```

### 1.2 Verify Everything Works

Run this checklist before proceeding to MCP installation:

```bash
echo "=== Environment Check ==="
echo "Node.js: $(node --version 2>/dev/null || echo 'NOT INSTALLED')"
echo "npm: $(npm --version 2>/dev/null || echo 'NOT INSTALLED')"
echo "Python: $(python3 --version 2>/dev/null || echo 'NOT INSTALLED')"
echo "uv: $(uv --version 2>/dev/null || echo 'NOT INSTALLED')"
echo "Claude Code: $(claude --version 2>/dev/null || echo 'NOT INSTALLED')"
echo "Git: $(git --version 2>/dev/null || echo 'NOT INSTALLED')"
```

All should show version numbers, not "NOT INSTALLED".

---

## 2. MCP TOOLS — COMPLETE INSTALLATION GUIDE

MCP (Model Context Protocol) servers are plugins that extend Claude Code's capabilities. Each MCP server gives Claude Code access to external tools, APIs, or services. When you install an MCP server, Claude Code can automatically use it during conversations.

### 2.1 How MCP Servers Work (Conceptual Overview)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code CLI                          │
│                                                                 │
│  You type: "Run pytest on my project"                          │
│                                                                 │
│  Claude Code thinks:                                            │
│    "I have an MCP server called 'code-checker' that can         │
│     run pytest. Let me use its 'run_pytest_check' tool."       │
│                                                                 │
│  Claude Code calls → MCP Server (code-checker)                  │
│                       → Runs pytest on your project             │
│                       → Returns results to Claude Code          │
│                                                                 │
│  Claude Code shows you the results and suggests fixes           │
└─────────────────────────────────────────────────────────────────┘
```

There are three types of MCP connections:
- **stdio** — The MCP server runs as a local process on your machine. Claude Code launches it, communicates via stdin/stdout. Most common for dev tools.
- **http** — The MCP server runs remotely on a server. Claude Code connects via HTTP. Used for cloud services like Notion, Asana.
- **sse** — (Deprecated) Server-Sent Events transport. Being replaced by http.

### 2.2 MCP Server Scopes (Where Configs Are Stored)

When you install an MCP server, you choose a "scope" — this determines where the configuration is saved and who can access it:

| Scope | Config Location | Who Can Use It | When to Use |
|-------|----------------|----------------|-------------|
| **local** (default) | `~/.claude.json` under your project's path | Only you, only in this project | Project-specific tools (code-checker for this project) |
| **user** | `~/.claude.json` at the top level | Only you, in ALL projects | Tools you want everywhere (Context7, GitHub) |
| **project** | `.mcp.json` in the project root (committed to git) | Everyone on the team | Shared team tools |

**Recommendation for llm-forge**: Install most tools at `user` scope (so they're available in all your projects), and project-specific ones at `local` scope.

### 2.3 Two Ways to Install MCP Servers

**Method A: CLI Command (Quick, one-at-a-time)**

```bash
claude mcp add <name> -- <command> [args...]
# or with options:
claude mcp add --scope user --transport stdio <name> -- <command> [args...]
```

**Method B: Edit Config File Directly (Recommended for bulk setup)**

Edit `~/.claude.json` directly. This is better when you're setting up many servers at once, because you can see all configurations in one place.

```bash
# Open the config file
code ~/.claude.json      # VS Code
nano ~/.claude.json      # Terminal editor
vim ~/.claude.json       # Vim
```

The file structure looks like this:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "@package/name"],
      "env": {
        "API_KEY": "your-key-here"
      }
    }
  }
}
```

### 2.4 Installing Each MCP Server (Step-by-Step)

---

#### TOOL 1: Context7 — Live Library Documentation

**What it does**: Fetches up-to-date, version-specific documentation for any library (PyTorch, HuggingFace, Pydantic, etc.) directly into Claude Code's context. This prevents Claude from using outdated or hallucinated API calls.

**Why you need it**: Claude Code's training data has a knowledge cutoff. When you ask about `transformers` v4.45 or `peft` v0.13, Claude might use outdated method signatures. Context7 fetches the CURRENT docs so Claude generates correct code.

**Tools it provides**:
- `resolve-library-id` — Finds the Context7-compatible library ID for any library name
- `get-library-docs` — Fetches current documentation for a specific library and topic

**When to use in our methodology**: Step 1 (Research Spike), Step 2 (Schema — verify Pydantic APIs), Step 4 (TDD — verify any new library API before using it), Step 6 (Quality Audit)

**Installation — Method A (CLI)**:

```bash
# Install at user scope (available in all projects)
claude mcp add --scope user context7 -- npx -y @upstash/context7-mcp@latest
```

**Installation — Method B (Config File)**:

Add this to `~/.claude.json`:

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

**Verification**:

```bash
# Check it's registered
claude mcp list
# You should see: context7 (connected)

# Inside Claude Code, run:
/mcp
# Should show: • context7: connected
```

**How to use in prompts**:

```
# Explicit usage — add "use context7" to any prompt
"Show me how to use peft LoRA with transformers. use context7"

# Claude Code will automatically:
# 1. Call resolve-library-id to find "peft" and "transformers"
# 2. Call get-library-docs to fetch current documentation
# 3. Generate code using the CURRENT API, not outdated training data
```

---

#### TOOL 2: mcp-code-checker — Automated pytest + pylint + mypy

**What it does**: Runs Python code quality checks (pytest for tests, pylint for code style/bugs, mypy for type checking) directly from Claude Code. Returns results formatted for AI interpretation, so Claude can automatically understand failures and suggest fixes.

**Why you need it**: Instead of manually running pytest, copying output, pasting it back to Claude, this tool lets Claude run tests and fix issues in one continuous flow. Critical for our TDD methodology.

**Tools it provides**:
- `run_pytest_check` — Runs pytest on your project, returns results + AI-friendly analysis prompts
- `run_pylint_check` — Runs pylint, returns code quality issues with fix suggestions
- `run_mypy_check` — Runs mypy type checking, catches type errors
- `run_all_checks` — Runs all three simultaneously, returns combined report

**When to use in our methodology**: Step 4 (every RED/GREEN/REFACTOR cycle), Step 5 (Quality Gates)

**Prerequisites**:

```bash
# These must be installed in your project's Python environment
pip install pytest pylint mypy --break-system-packages
# Or if using a virtual environment (recommended):
source venv/bin/activate
pip install pytest pylint mypy
```

**Installation — Method A (CLI)**:

```bash
# First, install the package
pip install git+https://github.com/MarcusJellinghaus/mcp-code-checker.git --break-system-packages

# Then register with Claude Code
# Replace /path/to/llm-forge with your actual project path
claude mcp add --scope local code-checker -- \
  mcp-code-checker \
  --project-dir /path/to/llm-forge \
  --target-directories src tests
```

**Installation — Method B (Config File)**:

Add to `~/.claude.json` under your project path:

```json
{
  "projects": {
    "/path/to/llm-forge": {
      "mcpServers": {
        "code-checker": {
          "command": "mcp-code-checker",
          "args": [
            "--project-dir", "/path/to/llm-forge",
            "--target-directories", "src", "tests"
          ]
        }
      }
    }
  }
}
```

**Verification**:

```bash
# Verify the CLI works
mcp-code-checker --help

# Inside Claude Code (in the llm-forge directory):
/mcp
# Should show: • code-checker: connected
```

**How to use in prompts**:

```
# Run all checks at once
"Run all code quality checks on the project"
# Claude calls run_all_checks → gets pytest + pylint + mypy results

# Run just tests
"Run the pytest suite and fix any failures"
# Claude calls run_pytest_check → reads failures → edits code → re-runs

# Run type checking
"Check for type errors in the schema module"
# Claude calls run_mypy_check → identifies type mismatches
```

**Configuration options**:

```bash
# Disable specific pylint codes (e.g., missing docstrings)
--disable-codes C0114 C0116

# Only check specific directories
--target-directories src/llm_forge tests/unit

# Use a specific Python executable
--python-executable /path/to/venv/bin/python

# Use a specific virtual environment
--venv-path /path/to/venv
```

---

#### TOOL 3: GitHub MCP Server — Full GitHub Integration

**What it does**: Gives Claude Code direct access to your GitHub repositories — creating branches, managing pull requests, checking CI status, viewing workflow logs, reading code scanning alerts, and managing issues.

**Why you need it**: Enables our trunk-based development workflow. Claude can create feature branches, push code, create PRs, and check CI status without you leaving the terminal.

**Tools it provides**: Repository browsing, file operations, branch management, PR creation and review, issue management, CI/CD workflow status, code scanning alerts, and more.

**When to use in our methodology**: Step 3 (create feature branch), Step 5 (create PR after quality gates pass), all steps (version control)

**Prerequisites**:

You need a GitHub Personal Access Token (PAT):

```
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a descriptive name: "Claude Code MCP - llm-forge"
4. Select scopes:
   - repo (full control of private repositories)
   - workflow (update GitHub Actions workflows)
   - read:org (read org membership — if using org repos)
5. Click "Generate token"
6. COPY THE TOKEN IMMEDIATELY — you can't see it again
7. Store it securely (e.g., in a .env file that's in .gitignore)
```

**Installation — Method A (CLI)**:

```bash
# Set your token as an environment variable first
export GITHUB_PERSONAL_ACCESS_TOKEN="ghp_your_token_here"

# Install using the official GitHub MCP server (HTTP transport)
claude mcp add --scope user --transport http github \
  https://api.githubcopilot.com/mcp \
  --header "Authorization: Bearer $GITHUB_PERSONAL_ACCESS_TOKEN"
```

**Installation — Method B (Config File)**:

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "github": {
      "type": "http",
      "url": "https://api.githubcopilot.com/mcp",
      "headers": {
        "Authorization": "Bearer ghp_your_token_here"
      }
    }
  }
}
```

**Alternative: Docker-based installation** (more isolated):

```json
{
  "mcpServers": {
    "github": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
      }
    }
  }
}
```

**Verification**:

```bash
claude mcp list
# Should show: github (connected)

# Test inside Claude Code:
"List the branches in my llm-forge repository"
```

**How to use in prompts**:

```
# Create a feature branch
"Create a new branch called feature/neftune-implementation from main"

# Create a PR
"Create a pull request from feature/neftune-implementation to main with title 'Add NEFTune noise injection' and describe the changes"

# Check CI status
"What's the status of the latest CI workflow run on the feature branch?"

# View issues
"Show me all open issues labeled 'bug' in the repository"
```

---

#### TOOL 4: ArXiv MCP Server — Research Paper Search

**What it does**: Searches and retrieves academic papers from ArXiv directly within Claude Code. Returns paper titles, abstracts, authors, and PDF links.

**Why you need it**: Our methodology starts with a Research Spike (Step 1). For ML techniques like NEFTune, LoRA, DPO, label smoothing — they all have ArXiv papers. This tool lets Claude find and analyze the papers without you switching to a browser.

**Tools it provides**:
- `search_papers` — Search ArXiv by query, returns matching papers with metadata
- `get_paper` — Get full details of a specific paper by ArXiv ID

**When to use in our methodology**: Step 1 (Research Spike)

**Installation — Method A (CLI)**:

```bash
claude mcp add --scope user arxiv -- npx -y arxiv-mcp-server
```

**Installation — Method B (Config File)**:

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "npx",
      "args": ["-y", "arxiv-mcp-server"]
    }
  }
}
```

**Verification**:

```bash
/mcp
# Should show: • arxiv: connected
```

**How to use in prompts**:

```
# Search for papers on a technique
"Search ArXiv for papers about NEFTune noise injection for LLM fine-tuning"

# Get a specific paper
"Get the full details of the paper arxiv:2310.05914 (NEFTune paper)"

# Research comparison
"Find papers comparing LoRA vs full fine-tuning for instruction tuning"
```

---

#### TOOL 5: SQLite MCP Server — Local Benchmark Database

**What it does**: Provides SQL query access to local SQLite databases. You can create databases, insert data, query results — all through Claude Code.

**Why you need it**: For Step 7 (Local Training Validation), we need to store and compare training metrics across runs. Instead of eyeballing loss curves, we store metrics in SQLite and Claude can query historical data to verify improvements.

**Tools it provides**:
- `read_query` — Execute a SELECT query and return results
- `write_query` — Execute INSERT, UPDATE, DELETE queries
- `create_table` — Create a new table
- `list_tables` — List all tables in the database
- `describe_table` — Show a table's schema

**When to use in our methodology**: Step 7 (Training Validation — store and compare metrics)

**Installation — Method A (CLI)**:

```bash
claude mcp add --scope local sqlite -- \
  npx -y @modelcontextprotocol/server-sqlite \
  /path/to/llm-forge/metrics.db
```

**Installation — Method B (Config File)**:

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sqlite",
        "/path/to/llm-forge/metrics.db"
      ]
    }
  }
}
```

**Verification**:

```bash
/mcp
# Should show: • sqlite: connected
```

**How to use in prompts**:

```
# Create a metrics tracking table
"Create a SQLite table called 'training_runs' with columns: id, feature_name, timestamp, loss_start, loss_end, duration_minutes, config_file, notes"

# Store training results
"Insert a new training run: feature='neftune', loss_start=2.45, loss_end=1.89, duration=12 minutes"

# Compare results
"Show me all training runs for the 'neftune' feature, sorted by loss_end ascending"

# Regression detection
"Compare the average final loss across features to check for regressions"
```

---

#### TOOL 6: Snyk MCP — Security Scanning (Already Installed)

**What it does**: Performs SAST (Static Application Security Testing), SCA (Software Composition Analysis), and SBOM (Software Bill of Materials) scanning on your codebase. Finds security vulnerabilities in your code and dependencies.

**Why you need it**: Step 6 (MCP Quality Audit) requires a security scan before any feature is considered complete. Catches dependency vulnerabilities, secrets in code, and insecure coding patterns.

**Status**: Already installed in your environment. No action needed.

**How to use in prompts**:

```
# Full security scan
"Run a Snyk security scan on the llm-forge project"

# Check for dependency vulnerabilities
"Check if any of our Python dependencies have known vulnerabilities"

# SBOM generation
"Generate a software bill of materials for the project"
```

---

#### TOOL 7: Sentry MCP — Error Monitoring (Already Installed)

**What it does**: Connects to your Sentry error monitoring dashboard. Claude can check for unresolved issues, view error details, and help debug production errors.

**Why you need it**: Step 6 (MCP Quality Audit) checks that no new errors have been introduced.

**Status**: Already installed in your environment. No action needed.

**How to use in prompts**:

```
# Check for new issues
"Check Sentry for any unresolved issues in the llm-forge project"

# Investigate a specific error
"What are the details of the latest Sentry issue?"
```

---

#### TOOL 8: Weights & Biases (W&B) MCP — Experiment Tracking (Already Installed)

**What it does**: Connects to your W&B experiment tracking dashboard. Claude can log metrics, compare runs, and verify that training integrations are working.

**Why you need it**: Step 6 (MCP Quality Audit) verifies W&B integration, and Step 7 (Training) uses it for metric logging.

**Status**: Already installed in your environment. No action needed.

**How to use in prompts**:

```
# Check recent runs
"Show me the latest W&B training runs for llm-forge"

# Compare metrics
"Compare the loss curves between the last two training runs"
```

---

#### TOOL 9: HuggingFace MCP — Model/Dataset/Paper Search (Already Installed)

**What it does**: Searches the HuggingFace Hub for models, datasets, papers, and spaces. Provides 8 different search tools.

**Why you need it**: Step 1 (Research Spike) uses this to find reference implementations, pre-trained models, and datasets.

**Status**: Already installed in your environment. No action needed.

**How to use in prompts**:

```
# Find models
"Search HuggingFace for SmolLM2 models suitable for fine-tuning"

# Find datasets
"Find instruction-tuning datasets on HuggingFace with at least 10k examples"

# Find papers
"Search for HuggingFace papers about NEFTune"
```

---

#### TOOL 10: Apify MCP — Web Scraping & Research (Already Installed)

**What it does**: Provides access to Apify's web scraping actors. The most useful ones for development are:
- `apify/rag-web-browser` — General web research, doc lookups
- `apify/website-content-crawler` — Deep-crawl documentation sites
- `easyapi/arxiv-search-scraper` — Batch ArXiv search
- `pear_fight/stackoverflow-scraper` — Find StackOverflow solutions
- `nexgendata/github-scraper` — Scrape GitHub repositories

**Why you need it**: Step 1 (Research Spike) uses this for comprehensive web research when Context7 doesn't cover a library.

**Status**: Already installed in your environment. No action needed.

**How to use in prompts**:

```
# General research
"Use Apify RAG browser to research how HuggingFace TRL implements gradient accumulation loss scaling"

# Deep-crawl docs
"Use Apify website crawler to crawl the PyTorch documentation for DataLoader optimization"

# StackOverflow solutions
"Use Apify StackOverflow scraper to find solutions for 'CUDA out of memory during gradient accumulation'"
```

---

### 2.5 TIER 2 TOOLS — Install When Needed

These tools are not needed immediately but become valuable at specific phases:

#### mcp-pypi — Python Package Intelligence

```bash
# Install when you need to check package versions, find alternatives, or audit dependencies
claude mcp add --scope user pypi -- uvx mcp-pypi
```

**When to install**: When adding new dependencies to the project, or when auditing for version conflicts.

#### mcp-system-monitor — Hardware/GPU Monitoring

```bash
# Install when you start running local training (Step 7)
claude mcp add --scope user sysmon -- uvx mcp-system-monitor
```

**When to install**: Before Phase 1 training validation, or when debugging CUDA memory issues.

#### Docker MCP — Container Management

```bash
# Install when you reach Phase 3 (Ollama/GGUF export, model serving)
claude mcp add --scope user docker -- npx -y @quantgeekdev/docker-mcp
```

**When to install**: Phase 3 of llm-forge development, when packaging models for deployment.

#### Sequential Thinking MCP — Complex Problem Decomposition

```bash
claude mcp add --scope user sequential-thinking -- npx -y @modelcontextprotocol/server-sequential-thinking
```

**When to install**: When facing complex architectural decisions that benefit from step-by-step reasoning.

---

### 2.6 Complete ~/.claude.json Configuration

Here is what your complete configuration file should look like after installing all Tier 1 tools. You can copy this entire file:

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    },
    "arxiv": {
      "command": "npx",
      "args": ["-y", "arxiv-mcp-server"]
    },
    "github": {
      "type": "http",
      "url": "https://api.githubcopilot.com/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_GITHUB_PAT_HERE"
      }
    }
  },
  "projects": {
    "/path/to/llm-forge": {
      "mcpServers": {
        "code-checker": {
          "command": "mcp-code-checker",
          "args": [
            "--project-dir", "/path/to/llm-forge",
            "--target-directories", "src", "tests"
          ]
        },
        "sqlite": {
          "command": "npx",
          "args": [
            "-y",
            "@modelcontextprotocol/server-sqlite",
            "/path/to/llm-forge/metrics.db"
          ]
        }
      }
    }
  }
}
```

**IMPORTANT**: Replace all `/path/to/llm-forge` with your actual project path, and `YOUR_GITHUB_PAT_HERE` with your actual GitHub Personal Access Token.

---

## 3. CLAUDE CODE CONFIGURATION FILES

Claude Code uses several configuration files. Here's where each one lives and what it does:

### 3.1 File Locations

| File | Location | Purpose | Committed to Git? |
|------|----------|---------|-------------------|
| `CLAUDE.md` | Project root (`/llm-forge/CLAUDE.md`) | Master instructions for Claude Code behavior | Yes |
| `.claude/settings.json` | Project root (`/llm-forge/.claude/`) | Hooks, permissions, feature flags | Yes |
| `.mcp.json` | Project root (`/llm-forge/.mcp.json`) | Shared MCP server configs (team-wide) | Yes |
| `~/.claude.json` | Home directory | Personal MCP configs, user-level settings | No (personal) |
| `MEMORY.md` | Project root | Project memory — status, decisions, context for future sessions | Yes |

### 3.2 CLAUDE.md (This File)

This file is the most important configuration. Claude Code reads it at the START of every session. It defines:
- The development methodology (8 steps)
- When to use which MCP tool
- Quality gates and checks
- Anti-patterns to avoid
- Agent deployment patterns

**Location**: `/llm-forge/CLAUDE.md` (project root)

**How Claude Code uses it**: Automatically loaded when Claude Code is launched from the llm-forge directory. Claude Code treats all instructions in this file as its "operating manual."

### 3.3 .claude/settings.json (Hooks & Permissions)

This file configures Claude Code's hooks (automatic actions that run before/after tool usage) and permissions.

**Location**: `/llm-forge/.claude/settings.json`

See Section 7 for the complete hooks configuration.

---

## 4. CORE EXECUTION MINDSET

**You (Claude Code) are a senior ML platform engineer building production-grade LLM fine-tuning infrastructure.** Every action you take must follow the methodology defined in this document. Never skip steps. Never assume — always verify.

### 4.1 Guiding Principles

These are non-negotiable rules that govern every action:

1. **Research before code** — Never implement anything without first understanding the technique from papers, docs, and reference implementations. Deploy research agents (Step 1) before writing a single line of code.

2. **Contract-first** — Define Pydantic v2 schemas and interfaces in `schema.py` before writing any business logic. The schema IS the contract — CLI, pipeline, and tests all derive from it.

3. **Walking skeleton** — Wire the empty pipeline stage end-to-end before adding logic. If the skeleton breaks the pipeline, the feature doesn't fit the architecture — redesign before implementing.

4. **Test-driven** — Write failing tests FIRST (RED), then make them pass (GREEN), then clean up (REFACTOR). Never write implementation code without a corresponding test.

5. **Zero regressions** — Every change must pass ALL existing tests (baseline: 248 passed, 10 skipped, 0 failures) and validate ALL 20+ YAML configs. No exceptions.

6. **Verify with tools** — Use Context7 to verify every library API call. Use mcp-code-checker to run tests after every change. Use Snyk to scan for vulnerabilities. Never trust cached knowledge.

7. **Document as you go** — Update MEMORY.md and relevant docs after every feature completion. Future sessions must be able to understand what was done.

### 4.2 Decision-Making Framework

When facing ANY technical decision, follow this sequence:

```
STEP 1: What does the research say?
  → ArXiv MCP: Find the paper
  → Context7: Verify the library API
  → HuggingFace MCP: Find reference implementations

STEP 2: What do reference implementations do?
  → Check HuggingFace TRL (primary reference)
  → Check Axolotl, LitGPT, torchtune (secondary references)
  → Use Apify rag-web-browser if needed

STEP 3: What does our existing architecture support?
  → Read schema.py for current data models
  → Read dag_builder.py for pipeline structure
  → Check MEMORY.md for past decisions and constraints

STEP 4: What's the simplest thing that could work?
  → Walking skeleton first (empty stage, wired end-to-end)
  → Minimum viable implementation
  → Iterate from there

STEP 5: How do we verify correctness?
  → Write the test BEFORE the code
  → Use mcp-code-checker to run tests
  → Use Context7 to verify APIs
  → Run a tiny training validation if applicable
```

### 4.3 The "Never Trust, Always Verify" Rule

Claude Code's training data has a knowledge cutoff. Library APIs change. Functions get deprecated. New parameters are added. **NEVER** rely on memory for API signatures. **ALWAYS** verify with Context7 before using any external library API in code.

**Bad** (relying on cached knowledge):
```python
# Claude remembers: model.generate(input_ids, max_length=100)
# But the API changed 3 months ago to: model.generate(input_ids, max_new_tokens=100)
```

**Good** (verified with Context7):
```
# Before writing code, Claude asks Context7:
"What is the current signature for transformers model.generate()? use context7"
# Context7 returns the CURRENT docs showing max_new_tokens
# Claude writes correct code
```

---

## 5. THE 8-STEP FEATURE DEVELOPMENT LIFECYCLE

**MANDATORY**: Follow these 8 steps in order for EVERY feature, bug fix, or enhancement. Do not skip or reorder steps.

### Step 1: Research Spike (Time-boxed: 2-4 hours max)

**Objective**: Understand the technique thoroughly before writing any code.

**What you (Claude Code) do**:

1. Deploy 3-5 research agents in parallel to investigate:
   - **Agent 1**: Search ArXiv for the technique's paper (use ArXiv MCP)
   - **Agent 2**: Verify library APIs we'll use (use Context7)
   - **Agent 3**: Search HuggingFace for reference implementations (use HuggingFace MCP)
   - **Agent 4**: General web research for tutorials and blog posts (use Apify rag-web-browser)
   - **Agent 5**: Search for known issues or gotchas (use Apify stackoverflow-scraper)

2. Wait for ALL agents to complete before synthesizing findings.

3. Cross-reference findings — if sources disagree, prioritize:
   - Official library documentation (via Context7) — highest authority
   - The original paper (via ArXiv) — for theoretical correctness
   - HuggingFace TRL implementation — for practical implementation patterns
   - Community sources — for edge cases and gotchas

4. Produce a decision document saved to `[feature]_research.md` in the memory directory.

**MCP Tools Used**:

| Tool | How to Invoke | What Claude Code Should Say |
|------|--------------|----------------------------|
| ArXiv MCP | `search_papers` / `get_paper` | "Let me search ArXiv for papers on [technique]..." |
| Context7 | `resolve-library-id` + `get-library-docs` | "Let me verify the current API with Context7..." |
| HuggingFace MCP | `model_search` / `paper_search` | "Let me check HuggingFace for implementations..." |
| Apify rag-web-browser | `apify/rag-web-browser` | "Let me do broader web research..." |
| Apify stackoverflow-scraper | `pear_fight/stackoverflow-scraper` | "Let me check StackOverflow for known issues..." |

**Rules**:
- Fixed time box: 2-4 hours max. After that, make a decision and move on.
- If research is inconclusive, default to the HuggingFace TRL approach (our primary reference implementation).
- Always verify API signatures with Context7 before proceeding — never trust cached knowledge.

**Output**: A file `[feature]_research.md` saved to the project memory directory.

---

### Step 2: Contract-First Design (Schema Definition)

**Objective**: Define the data contract before any implementation.

**What you (Claude Code) do**:

1. Open `src/llm_forge/schema.py`
2. Define or extend Pydantic v2 config models for the new feature
3. Write an example YAML config snippet that exercises the new schema
4. Run validation: ALL 20+ existing YAML configs must still pass
5. Run mypy on schema.py to verify type correctness

**MCP Tools Used**:

| Tool | When | What For |
|------|------|----------|
| Context7 | Before writing Pydantic validators | Verify `model_validator`, `field_validator`, `ConfigDict` APIs |
| mcp-code-checker | After defining schema | Run `run_mypy_check` to verify types |

**Rules**:
- Every new feature MUST have a corresponding schema definition in `schema.py`
- Use `Optional[X] = None` for features disabled by default (config-driven feature flags)
- Add clear `Field(description="...")` for every new config field
- New features are DISABLED by default — users opt in with `enabled: true`
- The schema IS the contract: if it's not in the schema, it doesn't exist

**Validation Command**:
```python
# Claude Code should run this after any schema change:
python -c "
from pathlib import Path
from llm_forge.schema import load_and_validate_config
configs = list(Path('configs').rglob('*.yaml'))
passed, failed = 0, 0
for c in configs:
    try:
        load_and_validate_config(c)
        passed += 1
    except Exception as e:
        print(f'FAIL: {c.name} — {e}')
        failed += 1
print(f'Results: {passed} passed, {failed} failed out of {len(configs)} configs')
assert failed == 0, f'{failed} configs failed validation!'
"
```

---

### Step 3: Walking Skeleton

**Objective**: Wire an empty pipeline stage end-to-end before adding any business logic.

**What you (Claude Code) do**:

1. Create an empty/stub implementation of the new component (function/class that does nothing but pass-through)
2. Wire it into `dag_builder.py` at the correct pipeline stage
3. Run the full pipeline end-to-end — it should pass through the empty stage without breaking
4. Write ONE integration test: "pipeline completes with this feature enabled (even though it's a no-op)"

**MCP Tools Used**:

| Tool | When | What For |
|------|------|----------|
| GitHub MCP | At start | Create a feature branch: `feature/[name]` |
| mcp-code-checker | After wiring | Run `run_pytest_check` to verify nothing is broken |

**Example skeleton**:

```python
# In src/llm_forge/techniques/neftune.py (empty skeleton)
class NEFTuneHandler:
    """NEFTune noise injection for embedding layers. (Skeleton — no logic yet)"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = getattr(config, 'neftune_alpha', None) is not None
    
    def apply(self, model):
        """Apply NEFTune to model. Currently a no-op skeleton."""
        if not self.enabled:
            return model
        # TODO: Implementation goes in Step 4
        return model
```

**Rules**:
- The skeleton MUST "walk" (full pipeline passes) before you add any logic
- If the empty stage breaks ANYTHING, fix the architecture issue before proceeding
- This step validates that the pipeline architecture supports the new feature
- Commit the skeleton to the feature branch

---

### Step 4: Red-Green-Refactor (TDD Implementation)

**Objective**: Implement the feature using strict test-driven development.

**What you (Claude Code) do**:

**Phase A — Write the Test List** (before ANY implementation code):

```python
# Example test list for NEFTune:
# tests/test_neftune.py

# 1. Config validation tests
def test_neftune_config_valid():       # Valid config with alpha=5.0 is accepted
def test_neftune_config_invalid():     # Alpha < 0 is rejected
def test_neftune_config_disabled():    # No alpha field means disabled

# 2. Unit tests
def test_neftune_adds_noise():         # Forward hook adds noise to embeddings
def test_neftune_noise_magnitude():    # Noise magnitude matches alpha / sqrt(seq_len * dim)
def test_neftune_only_during_train():  # No noise during eval mode

# 3. Edge cases
def test_neftune_zero_alpha():         # Alpha=0 means no noise
def test_neftune_very_large_alpha():   # Large alpha doesn't cause NaN

# 4. ML-specific tests
def test_neftune_loss_decreases():     # Loss decreases over 10 steps with NEFTune
def test_neftune_output_shape():       # Output shape unchanged after applying NEFTune

# 5. Integration tests
def test_neftune_pipeline_integration():  # NEFTune works in full pipeline
```

**Phase B — RED → GREEN → REFACTOR for each test**:

For EACH test in the list:
1. **RED**: Write the failing test. Run with `mcp-code-checker run_pytest_check`. Confirm it fails.
2. **GREEN**: Write the MINIMUM code to make the test pass. Run again. Confirm it passes.
3. **REFACTOR**: Clean up the code without changing behavior. Run again. Confirm still passes. Run `mcp-server-analyzer ruff-check` for linting.

**Phase C — Implementation Order** (layer by layer):

```
Layer 1: Config validation (tests 1-3) → Edit schema.py
Layer 2: Component logic (tests 4-7)   → Edit techniques/neftune.py
Layer 3: Pipeline integration (test 8)  → Edit dag_builder.py
Layer 4: End-to-end (test 9-10)        → Full pipeline test
```

**MCP Tools Used**:

| Tool | When | What For |
|------|------|----------|
| mcp-code-checker | Every RED/GREEN/REFACTOR cycle | `run_pytest_check` — non-negotiable |
| mcp-code-checker | After each layer | `run_mypy_check` — catch type errors early |
| mcp-server-analyzer | During REFACTOR phase | `ruff-check` for linting, `ruff-format` for formatting |
| Context7 | Before any new API call | Verify API signatures before using them |
| Apify stackoverflow | When stuck | Find solutions to specific technical problems |

**Rules**:
- NEVER write implementation code without a corresponding test
- NEVER skip the RED phase — seeing the test FAIL first confirms it's testing the right thing
- After EVERY change, run tests via mcp-code-checker — this is not optional
- If a test is flaky (passes sometimes, fails sometimes), fix it immediately — flaky tests erode trust

---

### Step 5: Quality Gates (Definition of Done)

**Objective**: Run the FULL quality checklist. ALL checks must pass.

**The Checklist** (run in this order):

| # | Check | MCP Tool | Command | Pass Criteria |
|---|-------|----------|---------|---------------|
| 1 | All existing tests pass | mcp-code-checker | `run_pytest_check` | 248+ passed, 0 failed |
| 2 | New tests exist for this feature | Manual review | Count new test functions | At least 5 new tests |
| 3 | All 20+ YAML configs validate | Built-in | See validation script above | 0 failures |
| 4 | Type checking passes | mcp-code-checker | `run_mypy_check` | 0 errors |
| 5 | Linting passes | mcp-server-analyzer | `ruff-check` | 0 errors |
| 6 | No dead code introduced | mcp-server-analyzer | `vulture-scan` | No new dead code |
| 7 | Code quality score | mcp-server-analyzer | `analyze-code` | Score ≥ 80/100 |
| 8 | No unresolved TODOs | Manual | `grep -rn "TODO\|FIXME" src/` | All TODOs have issue links |

**Rules**:
- If ANY check fails, fix it before proceeding to Step 6
- The test baseline is sacred: 248 passed, 10 skipped, 0 failures — never go below this
- A feature without tests is NOT done
- A feature with type errors is NOT done
- A feature with linting errors is NOT done

---

### Step 6: MCP Quality Audit (Security & Correctness)

**Objective**: Use ALL available MCP tools to verify security, API correctness, and integration health.

**The 4-Tool Audit** (deploy as parallel agents if possible):

| # | Tool | What to Check | Pass Criteria |
|---|------|--------------|---------------|
| 1 | **Context7** | ALL library API calls in new code use correct signatures. No deprecated methods. Correct parameter names and types. | 0 incorrect API calls |
| 2 | **Snyk** | Full SAST + SCA scan. No HIGH or CRITICAL vulnerabilities. | 0 HIGH/CRITICAL vulns |
| 3 | **Sentry** | No new unresolved error issues introduced. | 0 new issues |
| 4 | **W&B** | Experiment tracking integration points work. Logging calls use correct metric names. | Integration verified |

**How to run**:

```
# Agent 1: Context7 verification
"Use Context7 to verify all library API calls in [new file]. Check every import, every function call, every parameter name."

# Agent 2: Snyk scan
"Run a Snyk security scan on the project. Report any HIGH or CRITICAL vulnerabilities."

# Agent 3: Sentry check
"Check Sentry for any new unresolved issues in the last 24 hours."

# Agent 4: W&B integration check
"Verify that the W&B logging integration is working by checking the latest logged metrics."
```

**Rules**:
- ALL four checks must pass before the feature is complete
- If Context7 reveals a deprecated API, fix it immediately
- If Snyk finds HIGH/CRITICAL vulnerabilities, fix them before merging

---

### Step 7: Local Validation Training (If Feature Affects Training)

**Objective**: If the feature modifies the training pipeline, verify with a real training run.

**What you (Claude Code) do**:

1. Use `configs/quickstart_tiny.yaml` (SmolLM2-135M, ~5-minute run)
2. Run training with the feature enabled
3. Monitor GPU usage with mcp-system-monitor (if installed)
4. Store metrics in SQLite via the sqlite MCP server
5. Compare before/after:
   - Loss should decrease (or behave as expected)
   - No NaN or Inf values
   - Memory usage is reasonable (no OOM)
   - Training speed is not degraded by >10%

**MCP Tools Used**:

| Tool | What For |
|------|----------|
| SQLite MCP | Store metrics: `INSERT INTO training_runs (feature, loss_start, loss_end, ...)` |
| W&B | Log training metrics for visualization |
| mcp-system-monitor | Monitor GPU utilization and memory (if installed) |

**Rules**:
- Skip this step ONLY if the feature doesn't touch the training pipeline at all
- If training fails or metrics regress, go back to Step 4 and fix
- Always store metrics for historical comparison

---

### Step 8: Documentation & Memory Update

**Objective**: Document everything for future sessions.

**What you (Claude Code) do**:

1. Update `MEMORY.md`:
   - What was built and why
   - Key technical decisions and rationale
   - Any gotchas or non-obvious design choices
   - Updated test baseline numbers
   - Cross-references to research files

2. Update topic-specific memory files (e.g., `neftune_research.md`)

3. Add/update docstrings for all new public functions and classes

4. If new CLI options were added, update help text

5. Commit with a descriptive message: `feat: add [feature] — [brief description]`

**Rules**:
- A feature is NOT complete until MEMORY.md is updated
- Include timestamps for when features were completed
- Never delete memory entries — mark them as `[RESOLVED]` or `[SUPERSEDED BY: ...]`

---

## 6. MCP TOOL DECISION MATRIX

Use this lookup table to instantly determine which tool(s) to use for any situation:

```
╔══════════════════════════════════════════╦══════════════════════════════════════════════════════════╗
║ SITUATION                                ║ TOOL(S) TO USE                                         ║
╠══════════════════════════════════════════╬══════════════════════════════════════════════════════════╣
║ "How does this library API work?"        ║ Context7 FIRST (authoritative), then Apify if needed   ║
║ "Is this the correct function signature?"║ Context7 (ALWAYS — never trust cached knowledge)       ║
║ "Find research papers on [technique]"    ║ ArXiv MCP + HuggingFace paper_search                   ║
║ "What models/datasets exist for [task]?" ║ HuggingFace MCP (model_search + dataset_search)        ║
║ "Run tests on my code"                   ║ mcp-code-checker: run_pytest_check                     ║
║ "Check code quality"                     ║ mcp-code-checker: run_all_checks                       ║
║ "Find type errors"                       ║ mcp-code-checker: run_mypy_check                       ║
║ "Lint my code"                           ║ mcp-server-analyzer: ruff-check + ruff-format           ║
║ "Find dead code"                         ║ mcp-server-analyzer: vulture-scan                       ║
║ "Get a code quality score"               ║ mcp-server-analyzer: analyze-code                       ║
║ "Scan for security vulnerabilities"      ║ Snyk FIRST, then Semgrep for deeper analysis            ║
║ "Check for production errors"            ║ Sentry                                                  ║
║ "Monitor GPU during training"            ║ mcp-system-monitor (Tier 2 — install when needed)       ║
║ "Store/compare training metrics"         ║ SQLite MCP + W&B                                        ║
║ "Check dependency versions/vulns"        ║ mcp-pypi (Tier 2) + Snyk                               ║
║ "Create branch / PR / check CI"          ║ GitHub MCP                                              ║
║ "Deep-crawl a documentation site"        ║ Apify: website-content-crawler                          ║
║ "General web research"                   ║ Apify: rag-web-browser                                  ║
║ "Find StackOverflow solutions"           ║ Apify: stackoverflow-scraper                            ║
║ "Search GitHub for code examples"        ║ Apify: github-scraper (nexgendata)                      ║
║ "Track model rankings/benchmarks"        ║ Apify: chatbot-arena-leaderboard-scraper                ║
╚══════════════════════════════════════════╩══════════════════════════════════════════════════════════╝
```

---

## 7. CLAUDE CODE HOOKS CONFIGURATION

Hooks are automatic scripts that run BEFORE or AFTER Claude Code uses a tool. They enforce quality without you having to remember to run checks.

### 7.1 Create the Settings File

Create the file `/llm-forge/.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "cd \"$CLAUDE_PROJECT_DIR\" && python -m ruff check --fix $TOOL_INPUT_path 2>/dev/null; exit 0"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/safety-check.sh"
          }
        ]
      }
    ]
  }
}
```

### 7.2 What Each Hook Does

**PostToolUse — Auto-lint After Every File Edit**:
- **Trigger**: Every time Claude Code writes or edits a file
- **Action**: Automatically runs `ruff check --fix` on the edited file
- **Why**: Catches linting issues immediately, not 30 minutes later at quality gates
- **Result**: Code stays clean throughout development, not just at the end

**PreToolUse — Safety Check Before Bash Commands**:
- **Trigger**: Every time Claude Code tries to run a bash command
- **Action**: Checks for dangerous operations (force push, recursive delete of critical dirs)
- **Why**: Prevents catastrophic mistakes like `rm -rf src/` or `git push --force main`
- **Result**: Dangerous commands are blocked with a warning message

### 7.3 Create the Safety Check Script

Create `/llm-forge/.claude/hooks/safety-check.sh`:

```bash
#!/bin/bash
# Safety check hook — blocks dangerous operations in Claude Code

INPUT="$TOOL_INPUT_command"

# Block force pushes to main (use --force-with-lease on feature branches only)
if echo "$INPUT" | grep -q "git push.*--force"; then
  echo "BLOCKED: Force push detected."
  echo "Use --force-with-lease on feature branches only."
  echo "Never force push to main."
  exit 1
fi

# Block recursive deletion of critical directories
if echo "$INPUT" | grep -qE "rm.*-rf.*(src|tests|configs|\.claude)"; then
  echo "BLOCKED: Recursive deletion of critical directory detected."
  echo "Directories src/, tests/, configs/, and .claude/ are protected."
  exit 1
fi

# Block accidental deletion of the database
if echo "$INPUT" | grep -qE "rm.*metrics\.db"; then
  echo "BLOCKED: Deletion of metrics database detected."
  echo "The metrics database contains historical training data."
  exit 1
fi

# Warn about pip install without --break-system-packages
if echo "$INPUT" | grep -q "pip install" && ! echo "$INPUT" | grep -q "break-system-packages"; then
  echo "WARNING: Consider using --break-system-packages flag with pip install."
  echo "Continuing anyway..."
fi

# All checks passed
exit 0
```

Make it executable:

```bash
chmod +x /llm-forge/.claude/hooks/safety-check.sh
```

---

## 8. CUSTOM SLASH COMMANDS

These are shortcuts you can type inside Claude Code to trigger common workflows:

### /quality-check

Tell Claude: "Run the full quality gate checklist" and Claude will:

```
1. Run pytest via mcp-code-checker → run_all_checks
2. Validate all YAML configs
3. Check for dead code via mcp-server-analyzer → vulture-scan
4. Get code quality score via mcp-server-analyzer → analyze-code
5. Report: PASS/FAIL with summary of each check
```

### /run-tests

Tell Claude: "Run the test suite with coverage" and Claude will:

```bash
pytest tests/ -v --cov=llm_forge --cov-report=term-missing --tb=short
```

### /validate-configs

Tell Claude: "Validate all YAML configs" and Claude will run the validation script from Step 2.

### /research-spike [topic]

Tell Claude: "Do a research spike on [topic]" and Claude will deploy 3-5 parallel research agents as described in Step 1.

### /train-local

Tell Claude: "Run a local validation training" and Claude will:

```
1. Run: python -m llm_forge.train --config configs/quickstart_tiny.yaml
2. Monitor GPU (if mcp-system-monitor installed)
3. Log metrics to W&B
4. Store results in SQLite
5. Compare with previous runs
```

---

## 9. TESTING METHODOLOGY FOR ML CODE

ML code requires special testing patterns because outputs are non-deterministic. Here are the patterns to use:

### 9.1 Deterministic Seeding

ALWAYS seed random number generators in tests:

```python
import torch
import random
import numpy as np

def seed_everything(seed=42):
    """Set all random seeds for reproducibility in tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 9.2 Approximate Assertions

Use `pytest.approx` for floating-point comparisons:

```python
# BAD — will fail due to floating-point precision
assert loss.item() == 2.3456

# GOOD — allows small tolerance
assert loss.item() == pytest.approx(2.3456, abs=0.05)
```

### 9.3 BoringModel Pattern (from Lightning AI)

Use trivially simple models for infrastructure tests — you're testing the pipeline, not the model:

```python
class BoringModel(nn.Module):
    """Trivially simple model for testing pipeline infrastructure."""
    def __init__(self, in_features=32, out_features=2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)
```

### 9.4 Property-Based Testing

Use Hypothesis for testing invariants (things that should ALWAYS be true):

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=0.01, max_value=100.0), min_size=2, max_size=100))
def test_softmax_sums_to_one(values):
    """Softmax output should always sum to 1.0, regardless of input."""
    t = torch.tensor(values)
    result = torch.softmax(t, dim=0)
    assert result.sum().item() == pytest.approx(1.0, abs=1e-5)
```

### 9.5 Testing Pyramid

```
            /\
           /  \          ~5% System/E2E Tests
          / E2E\         Full training run with tiny model (minutes)
         /------\
        /        \       ~15-20% Integration Tests
       / Integr.  \      Pipeline chaining, config→execution (seconds-minutes)
      /------------\
     /              \    ~15-20% ML-Specific Tests
    /   ML Tests     \   Loss decreases, output shapes, overfit-1-batch (seconds)
   /------------------\
  /                    \ ~60% Unit Tests
 /    Unit Tests        \ Config validation, data transforms, utilities (milliseconds)
/________________________\
```

---

## 10. AGENT DEPLOYMENT PATTERNS

### 10.1 Parallel Research Agents (Step 1)

Deploy 3-5 agents simultaneously. Each agent focuses on one research area:

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Spike Deployment                  │
│                                                              │
│  Agent 1: ArXiv + Web Research                              │
│    → Search for the technique's paper                       │
│    → Find related papers and citations                      │
│    → Tools: ArXiv MCP, Apify rag-web-browser                │
│                                                              │
│  Agent 2: Reference Implementation Analysis                  │
│    → How does HuggingFace TRL do it?                        │
│    → How does Axolotl do it?                                │
│    → Tools: HuggingFace MCP, Apify github-scraper           │
│                                                              │
│  Agent 3: Library API Verification                           │
│    → What's the current API for libraries we'll use?        │
│    → Are there any deprecated methods to avoid?             │
│    → Tools: Context7                                         │
│                                                              │
│  Agent 4: Community Solutions                                │
│    → What issues have others encountered?                    │
│    → Are there known bugs or gotchas?                        │
│    → Tools: Apify stackoverflow-scraper, rag-web-browser     │
│                                                              │
│  Agent 5: Security & Compatibility                           │
│    → Any security concerns with this technique?              │
│    → Are all required packages compatible?                   │
│    → Tools: Snyk, mcp-pypi                                   │
│                                                              │
│  ⏳ WAIT for ALL agents to complete                          │
│  📝 SYNTHESIZE findings into decision document               │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 Quality Audit Agents (Step 6)

Deploy the 4-tool audit as parallel checks:

```
┌─────────────────────────────────────────────────────────────┐
│                    Quality Audit Deployment                   │
│                                                              │
│  Agent 1: Context7 → API correctness verification           │
│  Agent 2: Snyk    → Security vulnerability scan              │
│  Agent 3: Sentry  → Error monitoring check                   │
│  Agent 4: W&B     → Integration health verification          │
│                                                              │
│  ALL must pass ✅ before feature is marked COMPLETE          │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. ANTI-PATTERNS TO AVOID

These come from Google's "Hidden Technical Debt in ML Systems" paper and real-world ML engineering experience:

1. **Glue code** — Don't wrap third-party packages in massive adapters. Use thin wrappers with clear interfaces. If you're writing >50 lines to wrap a library call, you're doing it wrong.

2. **Pipeline jungles** — Don't add pipeline stages without considering the holistic DAG design. Every new stage must fit the existing architecture in `dag_builder.py`.

3. **Dead experimental codepaths** — Clean up feature flags after features stabilize. Don't leave `if config.experimental_feature:` branches rotting in the codebase.

4. **Configuration debt** — Every config field MUST have validation in `schema.py`. An unvalidated config field will eventually cause a production failure.

5. **Undeclared consumers** — If you change a function's output format, search for ALL callers with `grep -rn "function_name" src/`. Use mypy to catch type mismatches.

6. **Training-serving skew** — If a feature affects training, ensure the same logic applies at inference time. NEFTune should NOT be applied during inference.

7. **Skipping the walking skeleton** — Never jump straight to implementation. The skeleton validates that the architecture supports the feature.

8. **Testing after implementation** — Tests written after code tend to test implementation details rather than behavior. Write tests FIRST — they specify WHAT the code should do, not HOW.

---

## 12. BRANCHING & VERSIONING STRATEGY

- **Trunk-based development** with short-lived feature branches (< 2 days)
- **Branch naming**: `feature/[name]` (e.g., `feature/neftune-implementation`)
- **Config-driven feature flags**: New features DISABLED by default (`enabled: false`)
- **Semantic versioning**:
  - MAJOR (1.0 → 2.0): Breaking schema changes that invalidate existing configs
  - MINOR (1.0 → 1.1): New features (NEFTune, label smoothing, etc.)
  - PATCH (1.0.0 → 1.0.1): Bug fixes (gradient accumulation loss bug)
- **Strangler fig pattern**: Replace/upgrade existing components incrementally, never in big-bang rewrites

---

## 13. MEMORY FILE CONVENTIONS

### 13.1 MEMORY.md (Root — always kept updated)

```markdown
# MEMORY.md — llm-forge Project Memory

## Current Status
- Phase: 1 (Core Training Features)
- Last completed feature: [name] on [date]
- Test baseline: [X] passed, [Y] skipped, 0 failed
- Next feature: [name]

## Key Decisions
- [Date]: Chose [approach A] over [approach B] because [reason]

## Cross-References
- mcp_tools_and_methodology.md — MCP tool configs and methodology
- [feature]_research.md — Research findings for [feature]
- training_metrics.md — Historical training results
- known_issues.md — Known bugs and workarounds
```

### 13.2 Topic-Specific Memory Files

Each feature gets its own research file:

```markdown
# neftune_research.md

## Technique: NEFTune (Noisy Embedding Fine-Tuning)
## Paper: arxiv:2310.05914
## Status: COMPLETED on [date]

### What We Learned
- NEFTune adds uniform noise to embedding vectors during training
- Noise magnitude: alpha / sqrt(seq_len * hidden_dim)
- Applied only during training, NOT during inference

### Implementation Decisions
- Used forward hook on embedding layer (same as HuggingFace TRL)
- Alpha configurable via schema: neftune_alpha: 5.0
- Disabled by default (neftune_alpha: null)

### Known Gotchas
- Must be removed before saving model (or hooks persist in saved model)
- alpha > 15 can cause training instability
```

---

## 14. TROUBLESHOOTING COMMON MCP ISSUES

### Problem: MCP server shows "failed" status

```bash
# Check what's wrong
claude mcp list
# If a server shows "failed":

# 1. Verify the command exists
which npx        # Should return a path
which mcp-code-checker  # Should return a path

# 2. Check Node.js version
node --version   # Must be >= 18.0.0

# 3. Try running the command manually
npx -y @upstash/context7-mcp@latest   # Should start without errors

# 4. Remove and re-add the server
claude mcp remove context7
claude mcp add --scope user context7 -- npx -y @upstash/context7-mcp@latest
```

### Problem: "Connection closed" error on Windows

```bash
# Windows requires the cmd /c wrapper for npx-based servers
claude mcp add context7 -- cmd /c npx -y @upstash/context7-mcp@latest
```

### Problem: Context7 not finding a library

```
# Context7 needs the exact library ID. Ask Claude to resolve it first:
"Use Context7 to resolve the library ID for 'huggingface transformers'"
# Then use the resolved ID for documentation lookup
```

### Problem: mcp-code-checker can't find pytest

```bash
# Ensure pytest is installed in the SAME Python environment
pip install pytest pylint mypy --break-system-packages

# Or specify the Python executable explicitly
claude mcp add code-checker -- mcp-code-checker \
  --project-dir /path/to/llm-forge \
  --python-executable /path/to/venv/bin/python
```

### Problem: Too many MCP servers consuming context window

```
# Disable unused servers by @mentioning them in Claude Code:
# Or use /mcp command to toggle servers on/off

# Best practice: Only enable servers you're actively using
# Tier 1 (always on): Context7, code-checker
# Tier 2 (enable when needed): ArXiv, GitHub, SQLite
```

---

## 15. EXECUTION CHECKLIST TEMPLATE

Copy this template for EVERY new feature. Paste it into the feature's tracking issue or memory file:

```markdown
## Feature: [Name]
## Started: [Date]
## Branch: feature/[name]

### Step 1: Research Spike ⬜
- [ ] Deployed research agents (ArXiv, Context7, HuggingFace, Apify)
- [ ] ArXiv papers reviewed: [paper IDs]
- [ ] Context7 API verification done for: [libraries]
- [ ] Reference implementations checked: [HF TRL, Axolotl, etc.]
- [ ] Decision document saved to: [filename]_research.md
- [ ] Time spent: [X hours] (max 4 hours)

### Step 2: Contract-First Design ⬜
- [ ] Schema defined in schema.py (fields: [list them])
- [ ] Example YAML config written
- [ ] All 20+ existing configs validate (0 failures)
- [ ] Type hints added (mypy clean)

### Step 3: Walking Skeleton ⬜
- [ ] Empty stage created: src/llm_forge/[path]
- [ ] Wired into dag_builder.py
- [ ] Full pipeline passes end-to-end (0 failures)
- [ ] Smoke test written: tests/test_[name].py

### Step 4: Red-Green-Refactor ⬜
- [ ] Test list written ([X] tests planned)
- [ ] All tests RED first (confirmed failing)
- [ ] All tests GREEN (all passing)
- [ ] Code refactored (clean, documented)
- [ ] Linting passes (ruff clean)
- [ ] Total new tests: [X]

### Step 5: Quality Gates ⬜
- [ ] All tests pass: [X] passed, [Y] skipped, 0 failed
- [ ] All configs validate: [X]/[X] passed
- [ ] mypy: 0 errors
- [ ] ruff: 0 errors
- [ ] vulture: no new dead code
- [ ] Quality score: [X]/100 (≥80 required)

### Step 6: MCP Quality Audit ⬜
- [ ] Context7: All API signatures verified ✅
- [ ] Snyk: 0 HIGH/CRITICAL vulnerabilities ✅
- [ ] Sentry: 0 new issues ✅
- [ ] W&B: Integration verified ✅

### Step 7: Local Training ⬜ (skip if N/A)
- [ ] quickstart_tiny.yaml run: [PASS/FAIL]
- [ ] Loss: [start] → [end] (expected decrease)
- [ ] No NaN/Inf values
- [ ] Memory usage: [X] GB (reasonable)
- [ ] Metrics stored in SQLite: run_id=[X]

### Step 8: Documentation ⬜
- [ ] MEMORY.md updated with feature summary
- [ ] [feature]_research.md finalized
- [ ] Docstrings added to all new public functions
- [ ] README updated (if applicable)
- [ ] Commit message: feat: [description]

## COMPLETED: [Date] ✅
```

---

## QUICK REFERENCE: METHODOLOGY STEPS × TOOLS (One-Page Summary)

```
┌─────────────────────────┬──────────────────────────────────────────────────────────────────┐
│ Step                    │ MCP Tools Used                                                   │
├─────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ 1. Research Spike       │ ArXiv + Context7 + HuggingFace + Apify (rag-browser, arxiv,     │
│                         │ stackoverflow, github-scraper)                                    │
├─────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ 2. Contract-First       │ Context7 (Pydantic API) + mcp-code-checker (mypy)               │
├─────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ 3. Walking Skeleton     │ GitHub MCP (create branch) + mcp-code-checker (pytest smoke)     │
├─────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ 4. Red-Green-Refactor   │ mcp-code-checker (pytest every cycle) + mcp-server-analyzer     │
│                         │ (ruff) + Context7 (API verify) + PostToolUse hooks (auto-lint)   │
├─────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ 5. Quality Gates        │ mcp-code-checker (run_all_checks) + mcp-server-analyzer         │
│                         │ (vulture, analyze-code)                                          │
├─────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ 6. MCP Quality Audit    │ Context7 + Snyk + Sentry + W&B (all four — mandatory)           │
├─────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ 7. Local Training       │ SQLite MCP (metrics storage) + W&B (logging) +                  │
│                         │ mcp-system-monitor (GPU monitoring, if installed)                 │
├─────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ 8. Documentation        │ Manual — update MEMORY.md + docstrings + research files          │
└─────────────────────────┴──────────────────────────────────────────────────────────────────┘
```

---

*Last updated: March 2026*
*Methodology version: 2.0*
*Based on research from: Google ML Engineering, Meta FAIR, HuggingFace, Lightning AI, MosaicML, PyTorch, Martin Fowler, ThoughtWorks*
*MCP installation verified against: Claude Code docs, Context7 docs, GitHub MCP Server docs, mcp-code-checker GitHub*
