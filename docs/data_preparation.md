# Data Preparation Guide

Prepare, clean, and augment training data for llm-forge.

---

## Supported Data Formats

llm-forge supports loading data from a wide range of file formats, HuggingFace datasets, URLs, and directories.

### File Format Reference

| Format | Extension | Notes |
|--------|-----------|-------|
| JSONL | `.jsonl` | One JSON object per line. Recommended for large datasets. |
| JSON | `.json` | Array of objects, or dict-of-lists. |
| CSV | `.csv` | Auto-detected headers. |
| TSV | `.tsv` | Tab-delimited CSV variant. |
| Parquet | `.parquet` | Columnar format, efficient for large datasets. |
| Plain Text | `.txt` | Split on double newlines into paragraphs. |
| Markdown | `.md` | Treated the same as plain text. |
| PDF | `.pdf` | Extracted page by page (requires `pymupdf`). |
| DOCX | `.docx` | Paragraph-level extraction (requires `python-docx`). |
| HTML | `.html`, `.htm` | Main content extraction via `trafilatura`. |

### Loading from Different Sources

```yaml
# HuggingFace dataset
data:
  train_path: "tatsu-lab/alpaca"

# Local JSONL file
data:
  train_path: "./data/train.jsonl"

# Local directory (recursively loads all supported files)
data:
  train_path: "./data/corpus/"

# URL (downloads and auto-detects format)
data:
  train_path: "https://example.com/dataset.jsonl"
```

### Streaming Large Datasets

For datasets too large to fit in memory, enable streaming mode:

```yaml
data:
  train_path: "cerebras/SlimPajama-627B"
  streaming: true
```

---

## Dataset Conversation Formats

llm-forge supports four data formats. Choose the one that matches your dataset structure.

### Alpaca Format (default)

The standard instruction-following format with `instruction`, `input`, and `output` fields.

```json
{
  "instruction": "Summarize the following article.",
  "input": "The quick brown fox jumped over the lazy dog...",
  "output": "A fox jumped over a dog."
}
```

When the `input` field is empty, the template simplifies to instruction + response only.

**Template (with input):**

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

**Template (without input):**

```
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}
```

### ShareGPT Format

Multi-turn conversations with role-tagged turns. Supports both `from`/`value` and `role`/`content` key names.

```json
{
  "conversations": [
    {"from": "human", "value": "What is photosynthesis?"},
    {"from": "gpt", "value": "Photosynthesis is the process by which plants..."},
    {"from": "human", "value": "Can you explain the light reactions?"},
    {"from": "gpt", "value": "The light reactions occur in the thylakoid membranes..."}
  ]
}
```

Role mappings: `human`/`user` -> User, `gpt`/`assistant` -> Assistant, `system` -> System.

### Completion Format

Raw text for continued pre-training or language modelling. The dataset must have a `text` field.

```json
{
  "text": "The history of artificial intelligence began in the 1950s..."
}
```

### Custom Format

Map arbitrary field names to the instruction/output schema using the data config:

```yaml
data:
  format: "custom"
  input_field: "question"
  output_field: "answer"
  context_field: "context"
```

```json
{
  "question": "What is the capital of France?",
  "context": "France is a country in Western Europe.",
  "answer": "Paris"
}
```

---

## Chat Template Support

When the base model provides a chat template (e.g., Llama-3, Mistral, ChatML), llm-forge automatically uses it instead of the default formatting templates. This ensures the training data matches the format the model was pre-trained with.

```yaml
data:
  system_prompt: "You are a helpful medical assistant."
```

The system prompt is automatically injected into every sample using the model's native chat template.

---

## Format Conversion Examples

### Convert CSV to Alpaca JSONL

If your data is in CSV format with columns `question`, `context`, `answer`:

```yaml
data:
  train_path: "./data/qa_pairs.csv"
  format: "custom"
  input_field: "question"
  output_field: "answer"
  context_field: "context"
```

### Convert plain text for continued pre-training

```yaml
data:
  train_path: "./data/corpus/"    # directory of .txt files
  format: "completion"
```

### Use a HuggingFace dataset with non-standard field names

```yaml
data:
  train_path: "your-org/custom-dataset"
  format: "custom"
  input_field: "prompt"
  output_field: "completion"
  context_field: null
```

---

## Data Cleaning Pipeline

llm-forge includes a 7-stage data cleaning pipeline that processes your training data before tokenization. Each stage is independently configurable and can be enabled or disabled.

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

```bash
llm-forge clean --config config.yaml
```

### Stage 1: Unicode Fix

Repairs mojibake (encoding corruption), removes invisible control characters, and normalizes text to NFC Unicode form using the `ftfy` library.

- **Config key:** `unicode_fix` (default: `true`)
- **Effect:** Fixes corrupted text without removing samples
- **Dependency:** `ftfy` (included in `llm-forge[cleaning]`)

```yaml
data:
  cleaning:
    unicode_fix: true
```

### Stage 2: Language Filter

Identifies the language of each document using FastText's language identification model (`lid.176.bin`) and keeps only documents matching the specified ISO-639 language codes.

- **Config key:** `language_filter` (default: `null` -- disabled)
- **Effect:** Removes documents not matching specified languages
- **Dependency:** `fasttext` (included in `llm-forge[cleaning]`)

```yaml
data:
  cleaning:
    language_filter: ["en"]                   # Keep only English
    language_confidence_threshold: 0.65       # Minimum confidence score
```

### Stage 3: Heuristic Filter

Applies rule-based quality filters inspired by Gopher, C4, and FineWeb heuristics. Checks include word count bounds, character count bounds, alphabetic ratio, symbol-to-word ratio, and within-document duplicate line/paragraph detection.

- **Config key:** `heuristic_filter` (default: `true`)
- **Effect:** Removes low-quality documents that fail any check

```yaml
data:
  cleaning:
    heuristic_filter: true
    min_word_count: 5                         # Minimum words per document
    max_word_count: 100000                    # Maximum words per document
    min_char_count: 20                        # Minimum characters
    max_char_count: 5000000                   # Maximum characters
    alpha_ratio_threshold: 0.6                # Minimum alphabetic character fraction
    symbol_to_word_ratio: 0.1                 # Maximum symbol-to-word ratio
    max_duplicate_line_fraction: 0.3          # Maximum fraction of duplicate lines
    max_duplicate_para_fraction: 0.3          # Maximum fraction of duplicate paragraphs
```

### Stage 4: Deduplication

Removes duplicate documents using up to three tiers of increasing cost and accuracy:

1. **Exact** -- SHA-256 hash of normalized text. Fast and precise.
2. **Fuzzy** -- MinHash Locality-Sensitive Hashing (LSH) with Jaccard similarity. Catches near-duplicates.
3. **Semantic** -- Embedding-based cosine similarity. Catches paraphrases. (Optional, expensive.)

- **Config key:** `dedup_enabled` (default: `true`)
- **Effect:** Removes duplicate and near-duplicate documents

```yaml
data:
  cleaning:
    dedup_enabled: true
    dedup_tiers:
      - exact
      - fuzzy
    dedup_jaccard_threshold: 0.85             # Jaccard similarity threshold for fuzzy dedup
    dedup_num_perm: 128                       # MinHash permutations (higher = more accurate)
    dedup_shingle_size: 5                     # N-gram size for MinHash

    # Optional: semantic dedup (requires sentence-transformers)
    semantic_dedup_enabled: false
    semantic_dedup_threshold: 0.95
    semantic_dedup_model: "sentence-transformers/all-MiniLM-L6-v2"
```

### Stage 5: Quality Classification

Classifies documents by quality using a FastText classifier with KenLM perplexity scoring, falling back to heuristic methods when these models are unavailable. Documents below the quality threshold for the selected preset are removed.

- **Config key:** `quality_preset` (default: `"balanced"`)
- **Effect:** Removes documents classified as low quality

```yaml
data:
  cleaning:
    quality_preset: "balanced"    # "permissive", "balanced", or "strict"
```

### Stage 6: PII Redaction

Detects and redacts Personally Identifiable Information using Microsoft Presidio. Redacted entities are replaced with placeholder tags (e.g., `[PERSON]`, `[EMAIL_ADDRESS]`).

- **Config key:** `pii_redaction` (default: `false`)
- **Effect:** Replaces PII with anonymized placeholders (does not remove documents)
- **Dependency:** `presidio-analyzer`, `presidio-anonymizer` (included in `llm-forge[cleaning]`)

```yaml
data:
  cleaning:
    pii_redaction: true
    pii_entities:
      - PERSON
      - EMAIL_ADDRESS
      - PHONE_NUMBER
      - CREDIT_CARD
      - US_SSN
      - IP_ADDRESS
```

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

## Quality Presets

The `quality_preset` setting provides preconfigured strictness levels for the quality classification stage:

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

## Pipeline Statistics

After running the cleaning pipeline, llm-forge prints a detailed summary:

```
============================================================
Data Cleaning Pipeline Summary
============================================================
Initial documents:  50,000
Final documents:    38,245
Total removed:      11,755 (23.5%)
Retention rate:     76.5%

Per-step breakdown:
  Unicode fix              50,000 remaining  [1.2s]
  Language filter          48,123 remaining (-1,877)  [3.4s]
  Heuristic filter         44,567 remaining (-3,556)  [2.1s]
  Deduplication            41,234 remaining (-3,333)  [8.7s]
  Quality filter           38,245 remaining (-2,989)  [5.3s]
  PII redaction            38,245 remaining  [12.1s]
  Toxicity filter          [SKIPPED]

Total pipeline time: 32.8s
Skipped steps: Toxicity filter
============================================================
```

---

## Dataset Mixing

llm-forge supports mixing multiple data sources with configurable ratios, which is useful for combining domain-specific data with general instruction data.

### Default Mixing Weights

| Source | Default Weight |
|--------|---------------|
| `general` | 0.4 |
| `domain` | 0.4 |
| `synthetic` | 0.2 |

### Weight Computation Methods

| Method | Description |
|--------|-------------|
| `proportional` | Weight proportional to dataset size |
| `equal` | Equal weight for each source |
| `temperature` | Size-scaled by a temperature parameter (higher temp = more uniform) |

### Python API Example

```python
from llm_forge.data.mixing import DataMixer

mixer = DataMixer(seed=42, temperature=2.0)

weights = mixer.compute_optimal_weights(
    datasets={"general": general_ds, "medical": medical_ds},
    method="temperature",
)

mixed = mixer.mix_datasets(
    datasets={"general": general_ds, "medical": medical_ds},
    weights=weights,
    total_samples=50000,
)
```

### Upsampling and Downsampling

```python
# Upsample a small dataset to match a larger one
upsampled = mixer.upsample_dataset(small_ds, target_size=50000)

# Downsample a large dataset
downsampled = mixer.downsample_dataset(large_ds, target_size=10000)
```

---

## Synthetic Data Generation

llm-forge can generate synthetic instruction-response pairs from source documents, useful for augmenting domain-specific training data.

### Running Synthetic Generation

```bash
llm-forge synthetic --config config.yaml
```

### Difficulty Tiers

Generated questions are organized into four cognitive difficulty levels:

| Tier | Name | Example Prompt Pattern |
|------|------|----------------------|
| L1 | Factual | "What is {topic}?" / "Define {topic} based on the provided context." |
| L2 | Inferential | "What can be inferred about {topic}?" / "How does {topic} relate to the broader context?" |
| L3 | Evaluative | "Evaluate the strengths and limitations of {topic}." / "Compare and contrast the aspects of {topic}." |
| L4 | Counterfactual | "What would change if {topic} were different?" / "How would the outcome differ without {topic}?" |

### Heuristic Generation (no teacher model)

Without a teacher model, llm-forge uses template-based heuristic generation:

```python
from llm_forge.data.synthetic.generator import SyntheticDataGenerator

gen = SyntheticDataGenerator(
    max_pairs_per_chunk=3,
    seed=42,
)

synthetic_dataset = gen.generate_from_dataset(
    dataset=source_dataset,
    text_field="text",
    num_samples=5000,
)
```

### Teacher Model Generation

For higher-quality synthetic data, use a teacher model to generate instruction-response pairs:

```python
gen = SyntheticDataGenerator(
    teacher_model="meta-llama/Llama-3.2-3B-Instruct",
    temperature_range=(0.3, 0.9),
    max_pairs_per_chunk=3,
)

gen.load_teacher_model("meta-llama/Llama-3.2-3B-Instruct")

synthetic_dataset = gen.generate_from_dataset(
    dataset=source_dataset,
    text_field="text",
    num_samples=10000,
)
```

### Generating from Pre-Chunked Text

```python
chunks = ["Chunk 1 text...", "Chunk 2 text...", "Chunk 3 text..."]
topics = ["machine learning", "neural networks", "transformers"]

synthetic_dataset = gen.generate_from_chunks(
    chunks=chunks,
    topics=topics,
)
```

### Output Format

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

## Tips for High-Quality Training Data

1. **Start with quality over quantity.** A smaller, cleaner dataset often outperforms a larger noisy one. Use `quality_preset: "strict"` for domain-specific fine-tuning.

2. **Enable deduplication.** Even curated datasets often contain duplicates that cause the model to overfit on repeated patterns. At minimum, use `exact` + `fuzzy` dedup tiers.

3. **Match the format to your use case.** Use `alpaca` for instruction-following, `sharegpt` for multi-turn conversations, and `completion` for continued pre-training.

4. **Validate your data before training.** Run `llm-forge validate config.yaml` and `llm-forge clean --config config.yaml` to catch issues early.

5. **Mix domain and general data.** Pure domain data can cause "catastrophic forgetting" of general capabilities. A 40/40/20 split of general/domain/synthetic is a good starting point.

6. **Use the system prompt consistently.** If you plan to deploy with a system prompt, include it during training so the model learns to follow system-level instructions.

7. **Control sequence length.** Set `max_seq_length` in the model config to match your deployment requirements. Longer sequences require more VRAM and slow down training.

8. **Consider PII redaction for sensitive data.** Enable `pii_redaction: true` when training on data that may contain personal information, especially for models that will be deployed publicly.

---

## Next Steps

- [Configuration Reference](configuration.md) -- full YAML schema
- [Training Guide](training_guide.md) -- all training modes explained
- [Evaluation Guide](evaluation_guide.md) -- benchmark your trained model
