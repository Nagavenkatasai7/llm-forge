# LLM Forge: Input Handling & Context Management Problems

## Problem 1: Multi-Line Input Auto-Submits on Paste

### What Happens
When the user copies a multi-line message (like a terminal error output, stack trace, or log) and pastes it into the LLM Forge prompt, the **first newline character triggers Enter** and submits the incomplete message. The user can't paste 100 lines of error output and THEN type their question about it.

### Why It Happens
Python's built-in `input()` function treats `\n` (newline) as the submit key. When you paste text containing newlines from the clipboard, each newline triggers a submission. The first line gets submitted immediately, and the rest either gets lost or submitted as separate entries.

### Impact
- Users can't paste error messages, stack traces, or training logs for analysis
- Users can't paste multi-line code snippets
- Users can't compose multi-line messages

### How Claude Code Solves This
Claude Code uses a custom input handler (built with Ink/React for terminal) that:
1. **Detects paste vs. keystroke**: If multiple characters arrive within a very short time window (~10ms), it's a paste operation — don't submit on newlines within the paste
2. **Multi-line editing mode**: Shift+Enter or a specific key combo inserts a newline without submitting
3. **Bracket paste mode**: Modern terminals support "bracketed paste" (ANSI escape sequences `\e[200~` ... `\e[201~` that wrap pasted content). The terminal UI detects these markers and treats everything between them as a single paste, ignoring embedded newlines for submission

### Research Questions
1. How to implement **bracketed paste detection** in Python? (`prompt_toolkit` supports this natively)
2. How does `prompt_toolkit`'s `multiline=True` mode work? Can we detect when to auto-submit vs. continue?
3. How does `readline` handle paste in Python?
4. What terminal libraries support paste detection? (`prompt_toolkit`, `textual`, `blessed`, `urwid`)
5. Can we implement a heuristic: if >5 characters arrive in <50ms, treat as paste and buffer until a clean Enter?

### Potential Solutions
- **`prompt_toolkit` with `multiline` support** — The strongest option. Supports bracketed paste natively, multi-line editing, custom key bindings, tab completion, and syntax highlighting. Would replace `input()` entirely.
- **Custom `readline` wrapper** — Lower level, handles paste detection via timing heuristic
- **"Paste mode" toggle** — User types `/paste` to enter multi-line mode, then `---` on a line by itself to submit. Simple but not intuitive.
- **Automatic paste detection** — Monitor character arrival rate. If >5 chars arrive in <50ms, enter buffer mode. Display "pasting..." indicator. Submit when the user presses Enter after a pause.

---

## Problem 2: Context Confusion — Model Output Treated as Conversation

### What Happens
The user trains a model and it produces some output (e.g., a book-related response). The user then pastes this model output into LLM Forge to discuss it. But LLM Forge's Claude model **treats the pasted text as if it's part of the conversation** rather than understanding it's a **sample output from another model**.

For example:
```
User: [pastes model output about books]
Forge: [responds about books, as if the user is asking about books]
       [instead of understanding: "this is output from the trained model,
        the user wants to evaluate/discuss it"]
```

### Why It Happens
1. **No context delimiter**: There's nothing in the message to signal "this is model output, not my question"
2. **Conversation history confusion**: Long conversation histories accumulate context that can mislead the current interpretation
3. **System prompt doesn't instruct**: Claude doesn't know how to differentiate between user instructions and pasted content

### Impact
- Model evaluation becomes impossible through the chat interface
- Users can't share training results for analysis
- The assistant loses track of what the user actually wants

### How Claude Code Handles This
Claude Code has several mechanisms:
1. **Context awareness from tools**: When the user just ran training, Claude Code knows the context is "evaluating training results"
2. **File reading**: Instead of pasting, users can say "read the output" and Claude Code reads the file directly
3. **Session context**: The conversation history includes the tool calls (start_training, read_training_logs) so Claude knows they're in a training evaluation flow

### Research Questions
1. How to teach the LLM to distinguish between **user instructions** and **pasted content** from another source?
2. Can we add **context markers** in the system prompt? E.g., "If the user pastes long text without a question, ask: 'Is this output from your trained model? Want me to evaluate it?'"
3. Can we use **heuristic detection** to identify model output? (e.g., if the text looks like a model response — complete sentences, structured output — and the user just trained a model, assume it's model output)
4. Can we implement a **quoting mechanism** like `>` in Markdown to mark text as "not my words"?
5. How does the **conversation compaction** interact with this? When older context is summarized, does the model lose awareness of recent training actions?

### Potential Solutions

**Solution A: Smart Context Detection in System Prompt**
Add to the system prompt:
```
## Distinguishing User Intent from Pasted Content

When the user pastes a long block of text (>3 lines):
1. Check recent conversation context — did they just train a model?
   If yes, this is likely model output for evaluation.
2. Check if the text looks like model-generated content (complete
   sentences, structured responses, no questions).
3. If unsure, ask: "Is this output from your model? Want me to evaluate it?"

NEVER respond to pasted model output AS IF you are that model.
Always maintain your identity as LLM Forge and analyze the output
from an evaluator's perspective.
```

**Solution B: Explicit Paste Context via `/paste` Command**
```
User: /paste model-output
[pastes 100 lines]
---
What do you think of this output? It seems off-topic.

Forge: [knows the text is model output, evaluates it critically]
```

**Solution C: Automatic Heuristic Detection**
```python
def _classify_user_input(text, recent_tools):
    """Detect if the user is pasting model output vs. giving instructions."""
    # Did we just train or evaluate a model?
    recent_ml_action = any(t in recent_tools for t in
        ["start_training", "read_training_logs", "run_evaluation"])

    # Is the text long and looks like model output?
    lines = text.strip().split('\n')
    is_long = len(lines) > 5 or len(text) > 500
    has_no_question = '?' not in text

    if recent_ml_action and is_long and has_no_question:
        return "model_output"  # Wrap in context
    return "user_instruction"  # Treat normally
```

Then wrap the message before sending to Claude:
```python
if classification == "model_output":
    wrapped = (
        "The user has pasted the following OUTPUT from their trained model. "
        "They want you to evaluate and discuss it, NOT respond as if you "
        "are that model. Analyze the quality, relevance, and any issues:\n\n"
        f"--- MODEL OUTPUT ---\n{user_text}\n--- END MODEL OUTPUT ---\n\n"
        "What do you think of this output?"
    )
```

**Solution D: Combination (Recommended)**
1. `prompt_toolkit` for proper multi-line paste handling (Problem 1)
2. System prompt instructions for context awareness (Solution A)
3. Automatic heuristic wrapping for long pastes after training (Solution C)
4. `/paste` command as explicit fallback (Solution B)

---

## Priority

| Problem | Severity | Fix Complexity | Recommendation |
|---------|----------|---------------|----------------|
| Multi-line paste | High | Medium | Switch `input()` to `prompt_toolkit` session with bracketed paste support |
| Context confusion | High | Low (system prompt) + Medium (heuristic) | Add system prompt rules + auto-detection heuristic |

---

## Technical Requirements for prompt_toolkit Integration

```python
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

# Create a prompt session with multi-line support
bindings = KeyBindings()

@bindings.add('escape', 'enter')  # Esc+Enter for multi-line
def _(event):
    event.current_buffer.insert_text('\n')

session = PromptSession(
    message="You: ",
    multiline=False,  # Enter submits by default
    key_bindings=bindings,
    # Bracketed paste is enabled by default in prompt_toolkit
    # Pasted text with newlines is treated as a single input
    enable_open_in_editor=True,  # Ctrl+E opens full editor
    auto_suggest=AutoSuggestFromHistory(),
)

# Get input (handles paste, multi-line, history)
text = session.prompt()
```

Key `prompt_toolkit` features we need:
- **Bracketed paste**: Pasted text is treated as a single input regardless of newlines
- **History**: Up/Down arrow browses previous inputs
- **Auto-suggest**: Grey text shows previous similar inputs
- **Key bindings**: Esc+Enter for explicit newline, Enter to submit
- **Tab completion**: For `/` slash commands
- **Editor mode**: Ctrl+E opens $EDITOR for long messages

---

## What Success Looks Like

### Multi-line paste working:
```
You: Here's my model output:
     The quick brown fox jumped over the lazy dog.
     This is a test of the emergency broadcast system.
     ... (100 more lines pasted cleanly)
     What do you think?

Forge: This output looks like generic text, not domain-specific
       responses. Your model may need more training data focused
       on your target domain. Let me check the training config...
```

### Context-aware evaluation:
```
[after training completes]

You: [pastes model output about books]

Forge: I see you're testing your model's output. This response
       is about books, which doesn't match your finance training
       data. The model likely didn't learn enough from your
       domain data. Here are possible causes:

       1. **Dataset too small** — 500 samples may not be enough
       2. **Base model dominance** — Llama-3.2-1B's pre-training
          on general text is overpowering the fine-tuning
       3. **LoRA rank too low** — r=8 may need to be increased

       Want me to check the training logs for loss values?
```
