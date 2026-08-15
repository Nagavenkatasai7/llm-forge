# Repository Audit — 2026-08-14

Full review of llm-forge (`main` @ `be7c5a6`), with fixes on branch
`fix/security-and-mac-training`. Every issue below is listed with its evidence
and status. Test suite: **1295 passed, 21 skipped** (was 1222 passed before this
work; +73 new tests).

---

## ⚠️ ACT ON THIS FIRST: revoke two API keys

`src/llm_forge/chat/api_keys.py` contained a **live Anthropic API key and a live
Google API key**, added in commits `c203d71` and `be7c5a6`. They were XOR'd with
a single byte and base64-encoded — the decode key and the decode function were
committed in the same file. Anyone who cloned this public repository could
recover both keys in three lines of Python.

**They have been removed from the code. That is not enough.** Removing a secret
from the working tree does not remove it from git history, and the repository
has been public. Both keys must be treated as compromised:

1. Revoke the Anthropic key: <https://console.anthropic.com/settings/keys>
2. Revoke the Google key: <https://aistudio.google.com/apikey>
3. Check both accounts' usage logs for activity you don't recognise

Purging git history was offered and declined in favour of revocation, which is
the decision that actually matters — a rewritten history does not un-publish a
key that was already fetched.

---

## Issues found and fixed

### Security

| # | Issue | Evidence | Status |
|---|---|---|---|
| 1 | Live Anthropic + Google keys committed, XOR+base64 obfuscated | `chat/api_keys.py:15-16`, commits `c203d71`, `be7c5a6` | **Fixed** — keys and all deobfuscation code deleted; resolution now via `ANTHROPIC_API_KEY` / `~/.llm-forge/.env`; regression test fails if any bundled credential or obfuscation helper reappears |
| 2 | Security audit claimed "No hardcoded secrets", falsified by later commits | `SECURITY_AUDIT_REPORT.md:13` | **Fixed** — correction block added at the top marking it CRITICAL |
| 3 | Installer advertised "AI services included — no API keys needed!" | `install.sh:133` | **Fixed** — prompts for the user's own key, saves to `~/.llm-forge/.env` at mode 0600, degrades to the offline wizard |

### Crashes

| # | Issue | Evidence | Status |
|---|---|---|---|
| 4 | `props.total_mem` is not a torch attribute — it's `total_memory`. Raised `AttributeError` on **every NVIDIA machine**, taking `estimate_training` down with it | `chat/tools.py` in both `_detect_available_vram` and `_detect_hardware` | **Fixed** — corrected in both; exception path now degrades to `(0.0, "cpu")` instead of propagating |
| 5 | `search_huggingface` silently returned zero results for **every agent call**. The tool schema advertised `"models"`/`"datasets"`; `research_tools.py` defaulted to `"model"`. The singular matched neither branch, so the function returned an empty list with no error | `chat/tools.py:1369` vs `agent_tools/research_tools.py:6` | **Fixed** — both forms normalise; unknown values now raise instead of returning empty |
| 6 | QLoRA on Apple Silicon built a `BitsAndBytesConfig` and died inside `from_pretrained` — minutes and a multi-gigabyte download into a run that could never work | `training/finetuner.py:317` | **Fixed** — new `training/preflight.py` blocks it at the `dag_builder` routing switch before anything downloads, naming the MLX alternative |

### Missing functionality

| # | Issue | Evidence | Status |
|---|---|---|---|
| 7 | `web_search` was a hardcoded `not_implemented` stub | `agent_tools/research_tools.py:24-30` | **Fixed** — runs through Anthropic's server-side `web_search` tool, reusing the key the session already needs. Returns source URLs |
| 8 | `read_url` was a hardcoded `not_implemented` stub | `agent_tools/research_tools.py:33-39` | **Fixed** — wired to the existing `execution.fetch_url` |
| 9 | Nothing assessed whether a model fits before recommending it — results ranked purely by download count | — | **Fixed** — `chat/discovery.py` reads real parameter counts from each repo's safetensors index and reports a per-method memory verdict for the detected machine |
| 10 | Nothing assessed whether a dataset's answers could be verified | — | **Fixed** — datasets now scored on benchmark registration, held-out splits, linked paper, annotation provenance, and labelled task type; results rank by that ahead of downloads |

### Wrong for this hardware

| # | Issue | Evidence | Status |
|---|---|---|---|
| 11 | Apple Silicon memory reported as 75% of installed RAM — oversubscribed. Exceeding a unified-memory budget swaps rather than OOMs, so a run gets pathologically slow instead of failing fast | `chat/tools.py:1830` | **Fixed** — `usable_unified_memory_gb()` reports total minus an OS reserve (24 GB → 18 GB), and the tool output says so explicitly |
| 12 | "Switch to QLoRA" given as the universal OOM fallback in four places. CUDA-specific and unactionable on Apple Silicon | `_estimate_training`, `_gpu_recommendation`, `system_prompt.py:67`, `knowledge_base.py:328` | **Fixed** — all four are backend-aware; the fallback is now whichever method actually fits the detected backend |
| 13 | Model-selection table in the knowledge base was CUDA-only, and it is appended verbatim to every system prompt — so it overrode tool output in practice | `knowledge_base.py:326-337` | **Fixed** — split into NVIDIA / Apple Silicon / CPU tables, with usable-memory figures on the Apple rows |
| 14 | `_estimate_training` kept its own copy of the memory arithmetic, separate from what search would quote | `chat/tools.py:1855-1886` | **Fixed** — delegates to `discovery.assess_fit`, so both quote the same number |

### Stale

| # | Issue | Evidence | Status |
|---|---|---|---|
| 15 | Claude model IDs a full generation stale (4.5/4.6) | `chat/engine.py:46-76` | **Fixed** — Claude 5 family, default `claude-opus-5` |
| 16 | Model catalogue duplicated in `engine.py` and `orchestrator.py` — **and already drifted**: engine defaulted to `sonnet-4.6`, orchestrator to `opus-4.6` | both files | **Fixed** — single source in `chat/claude_models.py`; old keys alias forward so existing usage keeps working |
| 17 | `max_tokens` sized for non-thinking models. Opus 5 thinks by default and `max_tokens` caps thinking + response *together*, so the orchestrator's 2048 would truncate mid tool-call | `orchestrator.py:240,304`, `engine.py:92` | **Fixed** — raised to 8192/16000; router runs at `effort: low`; `display: "summarized"` set so the UI isn't blank while the model reasons |
| 18 | Research agent's prompt still told it `web_search` returns "not-implemented" | `agents/research_agent.py:22-25` | **Fixed** — rewritten to lead with fit verdicts and ground-truth scores |
| 19 | Version mismatch: package said `2.1.0`, installer and git tags said `v3.0.0` — the v3.0.0 release never bumped it | `pyproject.toml:7`, `__init__.py:3` | **Fixed** — both at `3.0.1` |
| 20 | Installer accepted any Python ≥ 3.10, so a bare `python3` pointing at 3.14 would install and then fail on `import torch` (no wheels) | `install.sh:57` | **Fixed** — bounded to 3.10–3.13 |

### Interface

| # | Issue | Evidence | Status |
|---|---|---|---|
| 21 | Status bar showed model/provider/memory — nothing that bounds any decision | `chat/tui.py:49` | **Fixed** — now carries the hardware budget: `opus-5 · Apple M4 Pro · 18/24 GB usable (mps) · 12 insights` |
| 22 | No activity indicator. With thinking content omitted by default, there was nothing on screen between submitting and the first token — which reads as a hang | `chat/tui.py:219` | **Fixed** — live activity segment naming the running tool; cleared in every exit path including interrupt |
| 23 | `web_search` and `read_url` had no `_TOOL_LABELS` entries, so they rendered as raw identifiers | `chat/ui.py:47-80` | **Fixed** — labels plus result summaries that lead with the finding |
| 24 | README prerequisites were NVIDIA-only | `README.md:78` | **Fixed** — Apple Silicon section with the unified-memory explanation and a what-fits table |

---

## Deferred, with reasons

| Issue | Why deferred |
|---|---|
| 40 pre-existing `ruff` errors (23 unsorted imports, 9 unused imports, 3 empty f-strings) | Verified identical before and after this work — none introduced here. Cosmetic; fixing them would bloat the diff and bury the substantive changes. Run `ruff check --fix src tests` for the 37 auto-fixable ones |
| `chat/tools.py:27` — module-level state is not multi-session safe (existing `TODO`) | Real, but an architectural change to how sessions are scoped. Out of scope for a bug-fix pass; no user-visible symptom in single-session use |
| OpenAI fallback path uses `gpt-4o` | `engine.py:164`. A non-Anthropic provider path, untouched deliberately |
| Leaked keys remain in git history | Your decision to revoke rather than rewrite history — which is the correct priority. Revocation makes the historical copies inert |
| Docker base image CVEs, unpinned GitHub Actions | Pre-existing findings from the original audit, unrelated to this pass. Still valid; see the surviving sections of `SECURITY_AUDIT_REPORT.md` |

---

## Training on 24 GB of unified memory

Your machine — Apple M4 Pro, 24 GB unified, **~18 GB usable** after the OS
reserve. Produced by `discovery.assess_fit`, the same code the agent now quotes:

| Model size | Best method that fits | Needs | Notes |
|---|---|---|---|
| 360M | **full fine-tune** | ~5 GB | Every weight updated, comfortable headroom |
| 1.2B | **full fine-tune** (8-bit optimizer) | ~9.5 GB | Full fine-tune in bf16 + Adam needs 18.7 GB — just over |
| 3.2B | **LoRA** (bf16) | ~8.3 GB | Full fine-tune needs 48 GB |
| 8B | **MLX LoRA, 4-bit** | ~7.9 GB | bf16 LoRA needs 20.6 GB — over budget |
| 14B | **MLX LoRA, 4-bit** | ~13.7 GB | Practical ceiling for this machine |

**Recommended for a first complete model:** Llama-3.2-1B or Qwen2.5-1.5B with a
full fine-tune. Every weight updates, it fits with room to spare, and a run
finishes in a sensible time — which matters more than raw capability when the
goal is a working end-to-end result you can evaluate and iterate on.

Two things that are not available on this machine, and that the agent now
refuses rather than attempts:

- **`training.mode: qlora`** — bitsandbytes has no Metal backend. For 4-bit,
  set `mlx.enabled: true` and quantize with `mlx_lm.convert --hf-path <model> -q`.
- **FlashAttention-2, DeepSpeed, Megatron, 8-bit optimizers** — all CUDA-only.

Unified memory does not fail cleanly. Overshooting the budget swaps instead of
raising OOM, so the symptom is a run that gets ten times slower rather than one
that stops. That is why the budget is 18 GB rather than 24, and why it is
displayed permanently in the status bar.

---

## What changed

| Area | Files |
|---|---|
| Credentials | `chat/api_keys.py` (rewritten), `install.sh`, `SECURITY_AUDIT_REPORT.md` |
| Model catalogue | `chat/claude_models.py` (new), `chat/engine.py`, `chat/orchestrator.py`, `chat/memory.py`, `chat/slash_commands.py` |
| Discovery | `chat/discovery.py` (new), `chat/tools.py`, `agent_tools/research_tools.py`, `agents/research_agent.py` |
| Hardware & guardrails | `training/preflight.py` (new), `training/mac_utils.py`, `pipeline/dag_builder.py`, `chat/tools.py` |
| Prompts | `chat/system_prompt.py`, `chat/knowledge_base.py` |
| Interface | `chat/tui.py`, `chat/ui.py` |
| Tests | `test_chat/test_discovery.py`, `test_chat/test_status_bar.py`, `test_training/test_preflight.py` (all new), plus updates to `test_chat/test_orchestrator.py`, `test_chat/test_tui.py` |
