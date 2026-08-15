# Repository Audit — 2026-08-14

Full review of llm-forge (`main` @ `be7c5a6`), with fixes on branch
`fix/security-and-mac-training`. Every issue below is listed with its evidence
and status. Test suite: **1298 passed, 21 skipped** (was 1222 before this work; +76 new tests).

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

### Found in this pass's own work

| # | Issue | Evidence | Status |
|---|---|---|---|
| 25 | `assess_fit` offered `full_8bit_optim` on Apple Silicon, and made it the *recommended* method for a 1.2B model on 18 GB. An 8-bit optimizer means `training.optim: adamw_8bit`, a bitsandbytes optimizer — CUDA only — and MLX's optimizer set has no 8-bit variant. The same class of error this branch exists to fix, introduced in the code meant to prevent it | `chat/discovery.py` `METHOD_BACKENDS` | **Fixed** — restricted to `{"cuda"}`; both published fit tables recomputed from real parameter counts |
| 26 | Undefined `logger` in `_detect_available_vram`'s exception handler — would have raised `NameError` on the very path meant to degrade gracefully | `chat/tools.py:1896` | **Fixed** — caught by `ruff F821` before commit; branch forced and verified to return `(0.0, "cpu")` |
| 27 | `warn_if_tight` written but never called | `training/preflight.py` | **Fixed** — wired into the training stage |
| 28 | **The full-screen TUI was unreachable.** `launch_tui()` was defined and its docstring said "called by `llm-forge --tui`" — but no such flag existed and nothing called the function. The entire Textual interface was dead code | `chat/tui.py:467`, `cli.py:206` | **Fixed** — `--tui` flag added and verified end to end from a real install |
| 30 | **A THIRD embedded credential.** `nvidia_provider.py` carried an NVIDIA API key using the identical XOR+base64 scheme, described as a "community" key. My own regression guard missed it because it only scanned `api_keys.py` | `chat/nvidia_provider.py:18` | **Fixed** — key and deobfuscation removed; guard now scans the whole package for the markers *and* for long base64 literals |
| 31 | The test suite read the developer's real `~/.llm-forge/.env` and probed for a live local `ollama serve`, so results depended on the machine | `tests/conftest.py` | **Fixed** — autouse fixture isolates the credential file and stubs the network probe |
| 32 | `llm-forge wizard` referenced in the installer and error messages; no such subcommand exists (it is `setup`) | `install.sh`, `chat/api_keys.py:90` | **Fixed** |
| 33 | Installer printed "No API key configured" and "API key configured — ready to use" in the same run, because the first check only looked for an Anthropic key | `install.sh` | **Fixed** |
| 34 | **Session-killing tool-call bug.** Models that emit several sequential tool calls all reporting `index 0` had their arguments concatenated into `{...}{...}`; `json.loads` raised `Extra data`, escaped `send()`, and destroyed the whole turn ("Something went wrong") | `chat/ollama_provider.py` stream reassembly | **Fixed** — accumulation keys on index *and* id; reproduced from a live transcript and re-verified against the live model |
| 35 | `_parse_openai_response` used a bare `json.loads` on tool arguments, so any malformed payload was fatal rather than recoverable | `chat/engine.py:498` | **Fixed** — tolerant parser splits concatenated objects into separate calls, strips markdown fences, and flags unparseable payloads for the model instead of raising |
| 36 | A model omitting a required tool argument got back `{"error": "'path'"}` — a bare KeyError string naming neither the tool, the problem, nor the schema, so it could not self-correct | `chat/tools.py` dispatcher, 32 unguarded reads | **Fixed** — schema validation at the boundary. Pattern borrowed from grok-build's typed-input-as-source-of-truth approach |
| 37 | **`read_file` returned 715 KB of raw PDF binary as "content" with `status: ok`.** `.pdf`/`.docx` were absent from the binary list, and a ReportLab PDF has no null byte in its first 8 KB, so it passed both checks. The model was told the read succeeded and handed megabytes of `%PDF-1.4` noise | `chat/execution.py` `_is_binary_file` | **Fixed** — documents detected and redirected to `read_document`; binary detection now judges printable ratio after a permissive decode (valid-UTF-8 control blobs and latin-1 text were both misclassified) |
| 38 | No way to read a PDF's content at all. 3 of 15 real PDFs were image-only and extracted **zero characters** — a scan and two certificates | — | **Fixed** — new `chat/documents.py` with `read_document` / `read_folder`; pages with no extractable text are rendered and transcribed by a vision model |
| 39 | A 150-DPI scan rendered to 3686×5219 = a 23 MB PNG, which the API rejected with a bare `400 failed to read request body` — indistinguishable from a model fault | `chat/documents.py` | **Fixed** — long edge capped at 2000 px, JPEG re-encode above 4 MB (23 MB → 0.5 MB) |
| 40 | Vision intermittently returned an empty transcript on dense scans — the same page gave 0 chars, then 914 on retry — and an empty reply was recorded as a *successful* vision read, producing a silent zero | `chat/documents.py` | **Fixed** — 3 attempts, and a persistently empty page is reported as empty with a reason instead of looking transcribed |
| 29 | `launch_tui` skipped the API-key check the scrolling UI does, so with no key it would take over the terminal and *then* have a dead engine, with no way to answer the prompt | `chat/tui.py:465` | **Fixed** — checks first, prints setup instructions, exits 1 |

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
reserve. Figures below come from `discovery.assess_fit` against **real
parameter counts read from each repo's safetensors index** — not name-parsing,
not estimates:

| Base model | Params | Best method that fits | Needs | Full fine-tune would need |
|---|---|---|---|---|
| SmolLM2-360M | 0.36B | **full fine-tune** | 5.5 GB | 5.5 GB ✅ |
| Llama-3.2-1B-Instruct | 1.24B | **LoRA** | 3.2 GB | 18.6 GB ❌ (just over) |
| Qwen2.5-1.5B-Instruct | 1.54B | **LoRA** | 4.0 GB | 23.3 GB ❌ |
| Llama-3.2-3B-Instruct | 3.21B | **LoRA** | 8.3 GB | 48.4 GB ❌ |
| Llama-3.1-8B-Instruct | 8.03B | **MLX LoRA (4-bit)** | 7.9 GB | 120.9 GB ❌ |

**Note the 1.2B row.** A full fine-tune of Llama-3.2-1B needs ~18.6 GB against
an 18 GB budget — it misses by about half a gigabyte. Optimizer state is 12 of
the 16 bytes per parameter, so trimming sequence length or batch size does not
rescue it; the shortfall is structural. **SmolLM2-360M is the largest model on
this machine where every weight can be updated.**

**Recommended for a first complete model: Llama-3.2-3B-Instruct with LoRA**
(8.3 GB). It leaves 10 GB of headroom, it is a genuinely capable base, and LoRA
on a 3B beats a full fine-tune of a 360M on essentially any real task — the
base model's quality dominates. If you specifically want every weight updated,
use SmolLM2-360M. If you want the strongest possible result and don't mind the
extra setup, Llama-3.1-8B via MLX 4-bit LoRA fits in 7.9 GB.

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

## Verified vs. not verified

**Exercised in this environment:**

- Full test suite: 1298 passed, 21 skipped
- Live HuggingFace Hub queries — ground-truth scoring rates `openai/gsm8k` 5/6
  "strong" and `praneethd7/gsm8k_sycophancy` 0/6 "none"
- Real parameter counts pulled from safetensors indexes for the 24 GB table above
- Hardware detection on this Mac: `Apple M4 Pro · 18/24 GB usable (mps)`
- The TUI boots and renders headless via Textual's pilot
- The no-key path end to end: orchestrator raises `MissingAPIKeyError` →
  `_setup_api_key` raises `NoAPIKeyError` → offline wizard
- `_detect_available_vram`'s exception branch returns `(0.0, "cpu")` instead of
  raising, with the failure forced

**Not exercised — needs your key on first run:**

- `web_search`. The implementation follows the current server-side web-search
  tool contract, but there is no Anthropic key in this environment, so it has
  never made a live call. Its error path is tested; its success path is not.
- Any actual training run. The memory model, preflight gate, and routing are
  unit-tested, but no model has been downloaded or a step executed here.

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
