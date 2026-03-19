# Data Preparation Guide

Prepare, clean, and augment training data for llm-forge. This guide covers supported formats, data loading, the 7-stage cleaning pipeline, IFD scoring, and best practices for building high-quality training datasets.

---

## Supported Data Formats

llm-forge supports four conversation formats for training data. Choose the one that matches your dataset structure.

### Alpaca Format (default)

The standard instruction-following format with `instruction`, optional `input`, and `output` fields. This is the most common format for single-turn instruction tuning.

**JSON Schema:**

```json
{
  "instruction": "string (required) -- the user's instruction or question",
  "input": "string (optional) -- additional context or input data",
  "output": "string (required) -- the expected model response"
}
```

**Example -- without input:**

```json
{
  "instruction": "What is compound interest?",
  "input": "",
  "output": "Compound interest is interest calculated on both the initial principal and the accumulated interest from previous periods."
}
```

**Example -- with input:**

```json
{
  "instruction": "Summarize the following research abstract.",
  "input": "This study investigates the impact of transformer-based architectures on natural language understanding tasks. We fine-tuned BERT, RoBERTa, and DeBERTa on the GLUE benchmark...",
  "output": "The study compares BERT, RoBERTa, and DeBERTa on GLUE benchmark tasks. DeBERTa-v3-large achieved the best results..."
}
```

**Formatting template (with input):**

```
Below is an instruction that describes a task, paired with an input that provides
further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

**Formatting template (without input):**

```
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}
```

**YAML configuration:**

```yaml
data:
  train_path: "./data/train.jsonl"
  format: "alpaca"
  # These are the defaults; only specify if your columns differ:
  input_field: "instruction"
  output_field: "output"
  context_field: "input"
```

---

### ShareGPT Format

Multi-turn conversations with role-tagged turns. This is the format used for conversational fine-tuning and is required for `assistant_only_loss` to work correctly. llm-forge supports three layout variants.

**Variant 1: Classic ShareGPT (`conversations` with `from`/`value`)**

```json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "What is photosynthesis?"},
    {"from": "gpt", "value": "Photosynthesis is the process by which plants convert light energy into chemical energy..."},
    {"from": "human", "value": "Can you explain the light reactions?"},
    {"from": "gpt", "value": "The light reactions occur in the thylakoid membranes..."}
  ]
}
```

**Variant 2: OpenAI-style (`messages` with `role`/`content`)**

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is photosynthesis?"},
    {"role": "assistant", "content": "Photosynthesis is the process by which plants..."}
  ]
}
```

**Variant 3: Flat columns (`system`/`user`/`assistant`)**

Common in HuggingFace datasets like Finance-Instruct-500k:

```json
{
  "system": "You are a finance specialist.",
  "user": "What is a P/E ratio?",
  "assistant": "The Price-to-Earnings ratio is a valuation metric..."
}
```

**Role mappings:** `human`/`user` maps to User, `gpt`/`assistant` maps to Assistant, `system` maps to System.

**YAML configuration:**

```yaml
data:
  train_path: "Josephgflowers/Finance-Instruct-500k"
  format: "sharegpt"
  system_prompt: "You are a helpful assistant."  # Injected if no system turn present
```

**Output:** The preprocessor produces both a `text` column (flat text for backward compatibility) and a `messages` column (list of role/content dicts for TRL's chat-template pipeline and assistant-only loss masking).

---

### Completion Format

Raw text for continued pre-training or language modeling. The dataset must have a `text` field.

**JSON Schema:**

```json
{
  "text": "string (required) -- the raw text for language modeling"
}
```

**Example:**

```json
{"text": "The history of artificial intelligence began in the 1950s when Alan Turing published 'Computing Machinery and Intelligence', which proposed the Turing test as a measure of machine intelligence."}
```

```json
{"text": "In financial markets, a yield curve inversion occurs when short-term government bonds offer higher yields than long-term bonds, which historically has preceded economic recessions."}
```

**YAML configuration:**

```yaml
data:
  train_path: "./data/corpus/"
  format: "completion"
```

---

### Custom Format

Map arbitrary field names to the instruction/output schema. Use this when your dataset has non-standard column names.

**JSON Schema (your fields vary):**

```json
{
  "question": "string -- maps to input_field",
  "context": "string -- maps to context_field (optional)",
  "answer": "string -- maps to output_field"
}
```

**Example:**

```json
{
  "question": "What is the capital of France?",
  "context": "France is a country in Western Europe.",
  "answer": "Paris"
}
```

**YAML configuration:**

```yaml
data:
  train_path: "./data/qa_pairs.jsonl"
  format: "custom"
  input_field: "question"
  output_field: "answer"
  context_field: "context"
```

Setting `context_field: null` disables context injection:

```yaml
data:
  format: "custom"
  input_field: "prompt"
  output_field: "completion"
  context_field: null
```

---

## Data Loading

llm-forge's `DataLoader` automatically detects the source type and loads data accordingly.

### Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| JSONL | `.jsonl` | One JSON object per line. Recommended for large datasets. |
| JSON | `.json` | Array of objects, or dict-of-lists. |
| CSV | `.csv` | Auto-detected headers. |
| TSV | `.tsv` | Tab-delimited CSV variant. |
| Parquet | `.parquet` | Columnar format, efficient for large datasets. |
| Plain Text | `.txt` | Split on double newlines into paragraphs. |
| Markdown | `.md` | Treated the same as plain text. |
| PDF | `.pdf` | Extracted page by page (requires `pymupdf` via `pip install llm-forge[cleaning]`). |
| DOCX | `.docx` | Paragraph-level extraction (requires `python-docx` via `pip install llm-forge[cleaning]`). |
| HTML | `.html`, `.htm` | Main content extraction via `trafilatura` (requires `pip install llm-forge[cleaning]`). |

### Loading from Different Sources

**HuggingFace Hub:**

```yaml
data:
  train_path: "tatsu-lab/alpaca"
```

The dataset is downloaded automatically. If no `train` split exists, the first available split is used.

**Local file:**

```yaml
data:
  train_path: "./data/train.jsonl"
```

**Local directory (recursive):**

```yaml
data:
  train_path: "./data/corpus/"
```

All supported files in the directory and subdirectories are loaded and concatenated. Each record gets a `_source_file` metadata field.

**URL (download and auto-detect):**

```yaml
data:
  train_path: "https://example.com/dataset.jsonl"
```

The file is downloaded to a temporary location, loaded based on its extension, and cleaned up afterward.

**Separate evaluation set:**

```yaml
data:
  train_path: "./data/train.jsonl"
  eval_path: "./data/eval.jsonl"
```

If `eval_path` is omitted, a portion of the training data is split off automatically using `test_size`.

### Streaming Large Datasets

For datasets too large to fit in memory:

```yaml
data:
  train_path: "cerebras/SlimPajama-627B"
  streaming: true
```

### Limiting Sample Count

For debugging or quick experiments:

```yaml
data:
  max_samples: 1000
```

Samples are shuffled before truncation (with deterministic seed) to avoid bias toward early records.

---

## Chat Template Support

When the base model provides a chat template (Llama-3, Mistral, ChatML, etc.), llm-forge automatically uses it instead of the default formatting templates. This ensures training data matches the format the model was pre-trained with.

The system prompt is injected into every sample using the model's native template:

```yaml
data:
  system_prompt: "You are a helpful medical assistant."
```

For `assistant_only_loss` to work, the dataset must produce a `messages` column (list of `{role, content}` dicts). The ShareGPT format does this automatically. The Alpaca format builds messages internally via `_build_messages()`.

---

## Data Cleaning Pipeline

llm-forge includes a 7-stage data cleaning pipeline that processes training data before tokenization. Each stage is independently configurable and can be enabled or disabled.

### Pipeline Overview

```
Raw Data
  |
  v
[1] Unicode Fix        -- Repair encoding, remove invisible chars, NFC normalization
  |
  v
[2] Language Filter     -- Keep only specified languages (FastText lid.176.bin)
  |
  v
[3] Heuristic Filter    -- Rule-based quality filtering (word count, symbol ratio, etc.)
  |
  v
[4] Deduplication       -- Remove duplicates (exact SHA-256, fuzzy MinHash, semantic)
  |
  v
[5] Quality Classifier  -- Score and filter by quality (FastText + KenLM, heuristic fallback)
  |
  v
[6] PII Redaction       -- Detect and redact personal information (Microsoft Presidio)
  |
  v
[7] Toxicity Filter     -- Remove toxic content (Detoxify model)
  |
  v
Clean Data
```

### Running the Cleaning Pipeline

Via CLI:

```bash
llm-forge clean --config config.yaml
```

Via Python:

```python
from llm_forge.data.cleaning import CleaningPipeline

pipeline = CleaningPipeline(
    config={"unicode_fix": True, "language_filter": ["en"], "dedup_enabled": True}
)
cleaned_dataset, stats = pipeline.run(dataset)
print(stats.summary())
```

### Configuring Cleaning in YAML

The cleaning configuration lives under `data.cleaning`:

```yaml
data:
  train_path: "./data/train.jsonl"
  cleaning:
    enabled: true                         # Master switch
    quality_preset: "balanced"            # permissive / balanced / strict

    # Stage 1: Unicode
    unicode_fix: true

    # Stage 2: Language
    language_filter: ["en"]
    language_confidence_threshold: 0.65

    # Stage 3: Heuristic
    heuristic_filter: true
    min_word_count: 10
    max_word_count: 100000
    min_char_count: 20
    max_char_count: 5000000
    alpha_ratio_threshold: 0.6
    symbol_to_word_ratio: 0.1
    max_duplicate_line_fraction: 0.3
    max_duplicate_para_fraction: 0.3

    # Stage 4: Deduplication
    dedup_enabled: true
    dedup_tiers:
      - exact
      - fuzzy
    dedup_jaccard_threshold: 0.85
    dedup_num_perm: 128
    dedup_shingle_size: 5
    semantic_dedup_enabled: false
    semantic_dedup_threshold: 0.95
    semantic_dedup_model: "sentence-transformers/all-MiniLM-L6-v2"

    # Stage 6: PII
    pii_redaction: false
    pii_entities:
      - PERSON
      - EMAIL_ADDRESS
      - PHONE_NUMBER
      - CREDIT_CARD
      - US_SSN
      - IP_ADDRESS

    # Stage 7: Toxicity
    toxicity_filter: false
    toxicity_threshold: 0.8
```

To disable cleaning entirely:

```yaml
data:
  cleaning:
    enabled: false
```

---

### Stage 1: Unicode Fix

Repairs mojibake (encoding corruption), removes invisible control characters, and normalizes text to NFC Unicode form using the `ftfy` library.

- **Config key:** `unicode_fix` (default: `true`)
- **Effect:** Fixes corrupted text without removing samples
- **Dependency:** `ftfy` (included in `llm-forge[cleaning]`)

Examples of fixes:
- `â€™` becomes `'` (smart quote mojibake)
- `Ã©` becomes `e` (Latin encoding errors)
- Zero-width spaces and invisible characters are stripped

---

### Stage 2: Language Filter

Identifies the language of each document using FastText's language identification model (`lid.176.bin`) and keeps only documents matching the specified ISO-639 language codes.

- **Config key:** `language_filter` (default: `null` -- disabled)
- **Effect:** Removes documents not matching specified languages
- **Dependency:** `fasttext` (included in `llm-forge[cleaning]`)

```yaml
data:
  cleaning:
    language_filter: ["en"]                   # Keep only English
    language_confidence_threshold: 0.65       # Minimum detection confidence
```

Set `language_filter: null` or omit to skip this stage.

---

### Stage 3: Heuristic Filter

Applies rule-based quality filters inspired by Gopher (DeepMind), C4, and FineWeb heuristics. Documents that fail any check are removed.

- **Config key:** `heuristic_filter` (default: `true`)

Checks performed:
- Word count within configured min/max bounds
- Character count within configured min/max bounds
- Alphabetic character ratio above threshold (catches boilerplate/code dumps)
- Symbol-to-word ratio below threshold (catches URLs, HTML fragments)
- Within-document duplicate line fraction below threshold
- Within-document duplicate paragraph fraction below threshold

```yaml
data:
  cleaning:
    heuristic_filter: true
    min_word_count: 5
    max_word_count: 100000
    alpha_ratio_threshold: 0.6
    symbol_to_word_ratio: 0.1
    max_duplicate_line_fraction: 0.3
```

---

### Stage 4: Deduplication

Removes duplicate documents using up to three tiers of increasing cost and accuracy:

| Tier | Method | Speed | Catches |
|------|--------|-------|---------|
| `exact` | SHA-256 hash of normalized text | Very fast | Byte-identical duplicates |
| `fuzzy` | MinHash Locality-Sensitive Hashing (LSH) with Jaccard similarity | Fast | Near-duplicates (minor edits, whitespace changes) |
| `semantic` | Embedding-based cosine similarity | Slow | Paraphrases and rephrased content |

- **Config key:** `dedup_enabled` (default: `true`)
- **Dependency:** `datasketch` for fuzzy (included in `llm-forge[cleaning]`), `sentence-transformers` for semantic (included in `llm-forge[rag]`)

```yaml
data:
  cleaning:
    dedup_enabled: true
    dedup_tiers:
      - exact
      - fuzzy
    dedup_jaccard_threshold: 0.85       # Higher = stricter (fewer removed)
    dedup_num_perm: 128                 # More permutations = more accurate but slower
    dedup_shingle_size: 5               # N-gram size for MinHash signatures

    # Optional: semantic tier
    semantic_dedup_enabled: true
    semantic_dedup_threshold: 0.95
    semantic_dedup_model: "sentence-transformers/all-MiniLM-L6-v2"
```

---

### Stage 5: Quality Classification

Classifies documents by quality using a FastText classifier with KenLM perplexity scoring, falling back to heuristic methods when these models are unavailable.

- **Config key:** `quality_preset` (default: `"balanced"`)

| Preset | Behavior | Use Case |
|--------|----------|----------|
| `permissive` | Keeps most documents; only removes clearly bad data | Large-scale pre-training, when quantity matters |
| `balanced` | Moderate filtering; good quality/quantity trade-off | General instruction fine-tuning (default) |
| `strict` | Aggressive filtering; keeps only high-quality data | Domain-specific fine-tuning where quality is critical |

```yaml
data:
  cleaning:
    quality_preset: "strict"
```

---

### Stage 6: PII Redaction

Detects and redacts Personally Identifiable Information using Microsoft Presidio. Redacted entities are replaced with placeholder tags (e.g., `[PERSON]`, `[EMAIL_ADDRESS]`).

- **Config key:** `pii_redaction` (default: `false`)
- **Effect:** Replaces PII with anonymized placeholders (does **not** remove documents)
- **Dependency:** `presidio-analyzer`, `presidio-anonymizer` (included in `llm-forge[cleaning]`)

Supported entity types:
- `PERSON` -- Names
- `EMAIL_ADDRESS` -- Email addresses
- `PHONE_NUMBER` -- Phone numbers
- `CREDIT_CARD` -- Credit card numbers
- `US_SSN` -- Social Security numbers
- `IP_ADDRESS` -- IP addresses

```yaml
data:
  cleaning:
    pii_redaction: true
    pii_entities:
      - PERSON
      - EMAIL_ADDRESS
      - PHONE_NUMBER
```

---

### Stage 7: Toxicity Filter

Scores documents for toxicity using the Detoxify model and removes documents exceeding the configured threshold.

- **Config key:** `toxicity_filter` (default: `false`)
- **Effect:** Removes documents with toxicity scores above the threshold
- **Dependency:** `detoxify` (included in `llm-forge[cleaning]`)

```yaml
data:
  cleaning:
    toxicity_filter: true
    toxicity_threshold: 0.8       # Score 0-1, higher = more toxic
```

---

## Pipeline Statistics

After running the cleaning pipeline, llm-forge prints a detailed summary showing how many documents were removed at each stage:

```
============================================================
Data Cleaning Pipeline Summary
============================================================
Initial documents:  20,000
Final documents:    5,675
Total removed:      14,325 (71.6%)
Retention rate:     28.4%

Per-step breakdown:
  Unicode fix              20,000 remaining  [0.8s]
  Language filter          18,500 remaining (-1,500)  [2.1s]
  Heuristic filter         12,340 remaining (-6,160)  [1.5s]
  Deduplication             7,890 remaining (-4,450)  [6.3s]
  Quality filter            5,675 remaining (-2,215)  [3.9s]
  PII redaction             5,675 remaining  [8.2s]
  Toxicity filter           [SKIPPED]

Total pipeline time: 22.8s
Skipped steps: Toxicity filter
============================================================
```

---

## IFD Scoring for Data Quality

IFD (Instruction-Following Difficulty) is a data quality metric from Li et al. (NAACL 2024, arXiv:2308.12032) that measures how much an instruction helps the model generate the expected response.

**Formula:** `IFD(Q, A) = s(A|Q) / s(A)`

Where:
- `s(A|Q)` is the average per-token negative log-likelihood of the response given the full instruction+response
- `s(A)` is the same metric with only the response (no instruction)

A **high IFD** means the instruction did not help much -- these samples tend to be harder for the model and are often more valuable for training. By selecting top-IFD samples, you can train on a smaller but more informative subset.

### Configuration

```yaml
ifd:
  enabled: true
  select_ratio: 0.5         # Keep top 50% of samples by IFD score
  batch_size: 4              # Mini-batch size for forward passes
  max_length: 512            # Max sequence length for scoring
```

### Python API

```python
from llm_forge.data.ifd_scorer import IFDScorer

scorer = IFDScorer(max_length=512, batch_size=4)
result = scorer.score_dataset(
    dataset=train_dataset,
    model=model,
    tokenizer=tokenizer,
)

# result.scores contains per-sample IFD scores
# result.mean_ifd and result.median_ifd for summary statistics

# Select top-k by IFD score
filtered_dataset = scorer.filter_dataset(
    dataset=train_dataset,
    model=model,
    tokenizer=tokenizer,
    select_ratio=0.5,
)
```

---

## Dataset Mixing

llm-forge supports mixing multiple data sources with configurable ratios. This is useful for combining domain-specific data with general instruction data to prevent catastrophic forgetting.

### Python API

```python
from llm_forge.data.mixing import DataMixer

mixer = DataMixer(seed=42, temperature=2.0)

weights = mixer.compute_optimal_weights(
    datasets={"general": general_ds, "medical": medical_ds},
    method="temperature",     # proportional, equal, or temperature
)

mixed = mixer.mix_datasets(
    datasets={"general": general_ds, "medical": medical_ds},
    weights=weights,
    total_samples=50000,
)
```

### Weight Computation Methods

| Method | Description |
|--------|-------------|
| `proportional` | Weight proportional to dataset size |
| `equal` | Equal weight for each source |
| `temperature` | Size-scaled by a temperature parameter (higher temp = more uniform) |

---

## Synthetic Data Generation

llm-forge can generate synthetic instruction-response pairs from source documents.

### Running Synthetic Generation

```bash
llm-forge synthetic --config config.yaml
```

### Difficulty Tiers

Generated questions follow four cognitive difficulty levels:

| Tier | Name | Example |
|------|------|---------|
| L1 | Factual | "What is {topic}?" |
| L2 | Inferential | "What can be inferred about {topic}?" |
| L3 | Evaluative | "Evaluate the strengths and limitations of {topic}." |
| L4 | Counterfactual | "What would change if {topic} were different?" |

### Python API

```python
from llm_forge.data.synthetic.generator import SyntheticDataGenerator

gen = SyntheticDataGenerator(max_pairs_per_chunk=3, seed=42)

synthetic_dataset = gen.generate_from_dataset(
    dataset=source_dataset,
    text_field="text",
    num_samples=5000,
)
```

For higher quality, use a teacher model:

```python
gen = SyntheticDataGenerator(
    teacher_model="meta-llama/Llama-3.2-3B-Instruct",
    temperature_range=(0.3, 0.9),
)
gen.load_teacher_model("meta-llama/Llama-3.2-3B-Instruct")

synthetic_dataset = gen.generate_from_dataset(
    dataset=source_dataset,
    text_field="text",
    num_samples=10000,
)
```

Synthetic data is produced in Alpaca format with metadata fields:

```json
{
  "instruction": "Summarize the following text:\n\nThe transformer architecture...",
  "input": "",
  "output": "The transformer architecture introduced self-attention...",
  "_difficulty": "L1_factual",
  "_source": "heuristic"
}
```

---

## Format Conversion Examples

### CSV with non-standard columns

```yaml
data:
  train_path: "./data/qa_pairs.csv"
  format: "custom"
  input_field: "question"
  output_field: "answer"
  context_field: "context"
```

### Plain text directory for continued pre-training

```yaml
data:
  train_path: "./data/corpus/"
  format: "completion"
```

### HuggingFace dataset with non-standard field names

```yaml
data:
  train_path: "your-org/custom-dataset"
  format: "custom"
  input_field: "prompt"
  output_field: "completion"
  context_field: null
```

---

## Tips for Data Quality

1. **Start with quality over quantity.** A smaller, cleaner dataset often outperforms a larger noisy one. In the Finance Specialist v7 project, cleaning reduced 20K samples to 5,675 (72% removed) and achieved better results than training on the full uncleaned set.

2. **Enable deduplication.** Even curated datasets contain duplicates that cause overfitting. At minimum, use `exact` + `fuzzy` dedup tiers.

3. **Match the format to your use case.** Use `alpaca` for single-turn instruction following, `sharegpt` for multi-turn conversations, and `completion` for continued pre-training. Using the wrong format can cause the model to learn the wrong patterns.

4. **Use `assistant_only_loss: true` for instruction tuning.** This is the most important training setting for conversational models. Without it, the model trains on system and user tokens, which causes system-prompt regurgitation and loss inflation.

5. **Validate before training.** Run `llm-forge validate config.yaml` and `llm-forge clean --config config.yaml` to catch issues early. Check the cleaning summary for suspiciously high or low removal rates.

6. **Mix domain and general data.** Pure domain data causes catastrophic forgetting of general capabilities. A 40/40/20 split of general/domain/synthetic is a good starting point.

7. **Use the system prompt consistently.** If you plan to deploy with a system prompt, include it during training via `data.system_prompt` so the model learns to follow system-level instructions.

8. **Control sequence length.** Set `model.max_seq_length` to match your deployment needs. For single-turn Q&A, 512-1024 is sufficient. For multi-turn conversations, 2048 supports 4-6 turns. Longer sequences require more VRAM.

9. **Consider PII redaction for sensitive data.** Enable `pii_redaction: true` when training on data that may contain personal information, especially for publicly deployed models.

10. **Use IFD scoring for large datasets.** When you have more data than you need, IFD scoring selects the most informative samples. Training on the top 50% by IFD often matches or beats training on the full dataset.

---

## Next Steps

- [Configuration Reference](configuration.md) -- full YAML schema with all cleaning fields
- [Training Guide](training_guide.md) -- all training modes explained
- [Evaluation Guide](evaluation_guide.md) -- benchmark your trained model
- [Quickstart Guide](quickstart.md) -- get a training run going in minutes
