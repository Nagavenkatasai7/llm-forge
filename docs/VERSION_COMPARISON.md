# LLM Forge Version Comparison: v0.3.2 vs v0.4.0

## The Core Change

| Behavior | v0.3.2 (Old) | v0.4.0 (New) |
|----------|-------------|-------------|
| User says "convert my DOCX" | "Run this command: `textutil -convert txt file.docx`" | Converts the file itself, reports result |
| User says "install python-docx" | "Run: `pip install python-docx`" | Installs it automatically |
| User says "create training data" | "Create a file at data/train.jsonl with..." | Creates the file directly |
| User says "what's in my data file?" | "Open the file and check..." | Reads the file, shows contents |
| User says "download this FAQ page" | "Open your browser and save..." | Fetches the URL, extracts text |

**v0.3.2 was an advisor. v0.4.0 is an executor.**

---

## Feature Comparison Table

| Feature | v0.3.2 | v0.4.0 | Impact |
|---------|--------|--------|--------|
| **Shell command execution** | None — tells user to run commands | `run_command` tool executes commands directly | User never needs to open another terminal |
| **File reading** | Only scan_data (directory listing) | `read_file` reads any file's contents | Forge understands your data files |
| **File writing** | Only write_config (YAML only) | `write_file` creates any file type | Forge writes training data, scripts, etc. |
| **Document conversion** | None | `convert_document` (DOCX/PDF/HTML/MD → text) | User provides a DOCX, Forge converts it |
| **Package installation** | install_dependencies (llm-forge extras only) | `install_package` (any PyPI package) | Missing deps auto-installed |
| **Web access** | None | `fetch_url` downloads pages/files | Scrape FAQ sites for training data |
| **Permission system** | None | 3-tier (allow/prompt/block) | Safe by default, user controls access |
| **Security guards** | None | Blocked commands, directory restrictions | Can't rm -rf or sudo |
| **Total tools** | 24 | **30** (+6 execution tools) | Full system access |
| **Tests** | 1094 | **1139** (+45) | All passing |

---

## What Was Lacking in v0.3.2

1. **No execution capability** — The manager could only explain what to do, not do it
2. **No file access** — Couldn't read user's data files or create training data
3. **No shell access** — Couldn't run conversion commands, install tools, etc.
4. **No web access** — Couldn't fetch FAQ pages or download datasets
5. **No auto-install** — Missing packages required manual user intervention
6. **Every step required user action** — Back-and-forth for every operation

## What v0.4.0 Still Lacks (Future Work)

1. **Terminal input/output collision** — Typing while streaming still mixes characters (needs prompt_toolkit)
2. **Sandbox isolation** — Commands run in user's environment (no Seatbelt/bubblewrap yet)
3. **Autonomous multi-step workflows** — Forge does one tool at a time; could chain automatically
4. **Web scraping depth** — fetch_url gets one page; doesn't crawl entire sites
5. **Interactive permission prompts** — ASK_FIRST tools currently blocked unless auto_approve=True

---

## How to Test v0.4.0

### Install/Upgrade
```bash
# Fresh install:
curl -fsSL https://raw.githubusercontent.com/Nagavenkatasai7/llm-forge/main/install.sh | bash

# Or upgrade existing:
~/.llm-forge/venv/bin/pip install --upgrade llm-forge-new
```

### Test 1: File Reading (should work immediately)
```
$ llm-forge

You: read the file examples/data/sample_train.jsonl

Expected: Forge reads the file and shows you the contents with sample count,
format detection, etc. — NOT "open the file in your editor"
```

### Test 2: File Writing
```
You: create a file called test.txt with the text "hello world"

Expected: Forge creates the file and confirms — NOT "run: echo hello > test.txt"
```

### Test 3: Document Conversion
```
# First, put a DOCX file in your directory, then:
You: convert my-document.docx to text

Expected: Forge converts it (installs python-docx if needed) and shows preview
```

### Test 4: Command Execution
```
You: what Python version is installed on this machine?

Expected: Forge runs "python3 --version" and tells you the answer
```

### Test 5: Package Installation
```
You: install the pandas library

Expected: Forge runs pip install pandas and confirms
```

### Test 6: URL Fetching
```
You: fetch the content from https://example.com

Expected: Forge downloads the page and shows you the text content
```

### Test 7: Security (should be BLOCKED)
```
You: run sudo rm -rf /

Expected: BLOCKED. Forge refuses and explains why.
```

### Test 8: Full Workflow (the dream scenario)
```
You: I have a DOCX file at ~/Documents/faq.docx. Build me a chatbot from it.

Expected: Forge reads the DOCX → converts to text → extracts Q&A pairs →
writes training data → configures training → starts training
All without asking you to run any commands.
```

---

## Permission System

| Tool | Permission | What happens |
|------|-----------|-------------|
| `read_file` | Always allowed | Reads immediately |
| `convert_document` | Always allowed | Converts immediately |
| `fetch_url` | Always allowed | Fetches immediately |
| `run_command` | Ask first | Blocked unless auto_approve=True |
| `write_file` | Ask first | Blocked unless auto_approve=True |
| `install_package` | Ask first | Blocked unless auto_approve=True |

To enable auto-approve for all operations, the user can set it in their first session. The manager remembers the preference.

---

## Architecture

```
v0.3.2:
  User → Claude API → 24 ML tools → tells user what to do

v0.4.0:
  User → Claude API → 30 tools → DOES the work
                        ↓
            ┌──────────────────────┐
            │ 24 ML Tools          │ (train, eval, deploy, etc.)
            │ 6 Execution Tools    │ (bash, read, write, convert, install, fetch)
            │ Permission System    │ (allow/prompt/block)
            │ Security Guards      │ (blocked commands, directory restrictions)
            └──────────────────────┘
```
