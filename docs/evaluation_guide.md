# Evaluation Guide

Benchmark, evaluate, and compare your fine-tuned models using llm-forge's evaluation system.

---

## Evaluation Overview

llm-forge provides three evaluation approaches:

1. **Standard Benchmarks** -- Run established LLM benchmarks (MMLU, HellaSwag, ARC, etc.) via EleutherAI's lm-evaluation-harness.
2. **Domain-Specific Evaluation** -- Evaluate on your own custom test sets with generation-based metrics.
3. **Perplexity Fallback** -- Basic perplexity evaluation when lm-eval is not installed.

---

## Quick Start

### Run benchmarks after training

```bash
llm-forge eval --config config.yaml --model-path ./outputs/my-lora
```

### Configure benchmarks in YAML

```yaml
evaluation:
  enabled: true
  benchmarks:
    - hellaswag
    - arc_easy
    - mmlu
  num_fewshot: 0
  batch_size: 8
  generate_report: true
```

---

## Standard Benchmarks

llm-forge integrates with EleutherAI's lm-evaluation-harness to run standardized benchmarks.

### Supported Benchmarks

| Benchmark | Task Name | Description | Default Few-Shot | Metric |
|-----------|-----------|-------------|-----------------|--------|
| MMLU | `mmlu` | Massive Multitask Language Understanding | 5 | `acc` |
| HellaSwag | `hellaswag` | Commonsense Natural Language Inference | 10 | `acc_norm` |
| ARC-Easy | `arc_easy` | AI2 Reasoning Challenge (Easy) | 25 | `acc_norm` |
| ARC-Challenge | `arc_challenge` | AI2 Reasoning Challenge (Challenge) | 25 | `acc_norm` |
| WinoGrande | `winogrande` | Winograd Schema Challenge at Scale | 5 | `acc` |
| TruthfulQA | `truthfulqa_mc2` | Measuring Truthfulness | 0 | `acc` |
| GSM8K | `gsm8k` | Grade School Math (8K problems) | 5 | `exact_match` |

### Task Aliases

For convenience, llm-forge supports shorthand aliases:

| Alias | Resolves To |
|-------|-------------|
| `arc` | `arc_challenge` |
| `truthfulqa` | `truthfulqa_mc2` |
| `gsm` | `gsm8k` |

### Running Specific Benchmarks

```yaml
evaluation:
  benchmarks:
    - mmlu
    - hellaswag
    - arc_challenge
    - winogrande
    - truthfulqa_mc2
    - gsm8k
  num_fewshot: 5           # Override default few-shot count for all tasks
  batch_size: 16           # Larger batch = faster evaluation
```

### Quick Testing with Limited Samples

Use `limit` to cap the number of evaluation samples per task for quick iteration:

```python
from llm_forge.evaluation.benchmarks import BenchmarkRunner

runner = BenchmarkRunner()
results = runner.run_benchmarks(
    model_path="./outputs/my-lora/merged",
    tasks=["hellaswag", "arc_easy"],
    limit=100,              # Only evaluate on 100 samples per task
)
```

### Installation

Standard benchmarks require the `eval` extra:

```bash
pip install llm-forge[eval]
```

This installs `lm-eval` (lm-evaluation-harness). Without it, llm-forge falls back to perplexity evaluation.

---

## Custom Domain Evaluation

For domain-specific evaluation, llm-forge provides the `DomainEvaluator` class that runs generation-based evaluation on your own test sets.

### Evaluation Dataset Format

Your evaluation dataset must be in JSONL or JSON format with at least an input field and an output (reference) field.

**JSONL format:**

```jsonl
{"input": "What is the capital of France?", "output": "Paris", "category": "geography"}
{"input": "What is 2+2?", "output": "4", "category": "math"}
{"input": "Who wrote Hamlet?", "output": "William Shakespeare", "category": "literature"}
```

**JSON format:**

```json
[
  {"input": "What is the capital of France?", "output": "Paris", "category": "geography"},
  {"input": "What is 2+2?", "output": "4", "category": "math"}
]
```

### Running Domain Evaluation

```python
from llm_forge.evaluation.domain_eval import DomainEvaluator

evaluator = DomainEvaluator(
    metrics=["exact_match", "f1", "accuracy"],
    input_field="input",
    output_field="output",
)

results = evaluator.evaluate(
    model=model,
    tokenizer=tokenizer,
    eval_dataset="./data/test_set.jsonl",
    max_new_tokens=256,
    batch_size=4,
    temperature=0.0,             # Greedy decoding for evaluation
    prompt_template="Question: {input}\nAnswer:",
    max_samples=500,
)

print(results["aggregate"])
# {'exact_match': 0.72, 'f1': 0.85, 'precision': 0.87, 'recall': 0.83, 'accuracy': 0.72}
```

### Custom Field Names

If your dataset uses different field names:

```python
evaluator = DomainEvaluator(
    input_field="question",
    output_field="answer",
)
```

### Post-Processing Predictions

Apply a custom post-processing function before computing metrics:

```python
def extract_answer(prediction: str) -> str:
    """Extract the final answer from model output."""
    if "Answer:" in prediction:
        return prediction.split("Answer:")[-1].strip()
    return prediction.strip()

results = evaluator.evaluate(
    model=model,
    tokenizer=tokenizer,
    eval_dataset="./data/test.jsonl",
    post_process_fn=extract_answer,
)
```

### Evaluating Pre-Computed Predictions

If you have already generated predictions (e.g., from a deployed API):

```python
results = evaluator.evaluate_predictions(
    predictions=["Paris", "4", "Shakespeare"],
    references=["Paris", "4", "William Shakespeare"],
    metrics=["exact_match", "f1"],
)
```

### Category Breakdown

If your evaluation data includes a `category` field in the metadata, llm-forge automatically computes per-category metrics:

```python
results = evaluator.evaluate(model, tokenizer, "test.jsonl")

for category, data in results["category_breakdown"].items():
    print(f"{category}: {data['count']} samples, metrics={data['metrics']}")
```

Output:

```
geography: 150 samples, metrics={'exact_match': 0.85, 'f1': 0.91, ...}
math: 200 samples, metrics={'exact_match': 0.78, 'f1': 0.82, ...}
literature: 150 samples, metrics={'exact_match': 0.63, 'f1': 0.79, ...}
```

---

## Available Metrics

### Text-Based Metrics

| Metric | Function | Description |
|--------|----------|-------------|
| Exact Match | `compute_exact_match` | Fraction of predictions that exactly match references (after normalization) |
| F1 | `compute_f1` | Token-level F1 score (SQuAD-style, macro-averaged) |
| Accuracy | `compute_accuracy` | Simple accuracy (works for strings and integer labels) |
| BLEU | `compute_bleu` | Corpus and sentence-level BLEU-1 through BLEU-4 |
| ROUGE | `compute_rouge` | ROUGE-1, ROUGE-2, ROUGE-L F1 scores |

### Model-Based Metrics

| Metric | Function | Description |
|--------|----------|-------------|
| Perplexity | `compute_perplexity` | Model perplexity over a set of text samples (lower is better) |

### Using the Metrics Computer Directly

```python
from llm_forge.evaluation.metrics import MetricsComputer

mc = MetricsComputer()

# Compute all text metrics at once
results = mc.compute_all(
    predictions=["The cat sat on the mat", "Paris is the capital"],
    references=["A cat sat on a mat", "Paris is the capital of France"],
    include=["exact_match", "f1", "bleu", "rouge"],
)

print(results)
# {
#   'exact_match': 0.0,
#   'f1': 0.82, 'precision': 0.85, 'recall': 0.80,
#   'bleu': 0.45, 'bleu_1': 0.72, 'bleu_2': 0.55, 'bleu_3': 0.41, 'bleu_4': 0.30,
#   'rouge1': 0.78, 'rouge2': 0.60, 'rougeL': 0.75,
# }
```

### Computing Perplexity

```python
perplexity = mc.compute_perplexity(
    model=model,
    tokenizer=tokenizer,
    texts=["Sample text 1", "Sample text 2"],
    max_length=2048,
    batch_size=4,
    device="cuda",
)
print(perplexity)
# {'perplexity': 12.34, 'avg_loss': 2.51, 'num_tokens': 256}
```

### Graceful Fallbacks

- **BLEU** falls back to a simple n-gram precision implementation when NLTK is not installed.
- **ROUGE** falls back to a longest-common-subsequence ROUGE-L implementation when `rouge-score` is not installed.
- Install the full evaluation suite with: `pip install llm-forge[eval]`

---

## Model Comparison

Compare the performance of a base model against your fine-tuned version:

```python
from llm_forge.evaluation.benchmarks import BenchmarkRunner

runner = BenchmarkRunner()

comparison = runner.compare_models(
    base_path="meta-llama/Llama-3.2-1B",
    finetuned_path="./outputs/my-lora/merged",
    tasks=["hellaswag", "arc_easy", "mmlu"],
    batch_size=8,
)

# Print comparison
for task, data in comparison["comparison"].items():
    if task.startswith("_"):
        continue
    print(
        f"{data['display_name']}: "
        f"base={data['base_score']:.4f} -> "
        f"finetuned={data['finetuned_score']:.4f} "
        f"(delta={data['delta']:+.4f}, {data['pct_change']:+.1f}%)"
    )

# Summary
summary = comparison["comparison"]["_summary"]
print(f"\nAverage delta: {summary['avg_delta']:+.4f}")
print(f"Improved: {summary['num_improved']}/{summary['total_tasks']} tasks")
```

Example output:

```
HellaSwag: base=0.4521 -> finetuned=0.4689 (delta=+0.0168, +3.7%)
ARC-Easy: base=0.6234 -> finetuned=0.6512 (delta=+0.0278, +4.5%)
MMLU: base=0.2891 -> finetuned=0.3102 (delta=+0.0211, +7.3%)

Average delta: +0.0219
Improved: 3/3 tasks
```

---

## Saving Evaluation Results

Save results to a JSON file for archival or further analysis:

```python
runner = BenchmarkRunner()
results = runner.run_benchmarks("./outputs/my-model", tasks=["hellaswag", "mmlu"])

runner.save_results(results, "./outputs/eval_results.json")
```

### Results Structure

```json
{
  "hellaswag": {
    "display_name": "HellaSwag",
    "score": 0.4689,
    "score_stderr": 0.0051,
    "metric": "acc_norm",
    "num_fewshot": 10,
    "raw": { ... }
  },
  "mmlu": {
    "display_name": "MMLU",
    "score": 0.3102,
    "score_stderr": 0.0042,
    "metric": "acc",
    "num_fewshot": 5,
    "raw": { ... }
  },
  "_aggregate": {
    "average_score": 0.3896,
    "num_tasks": 2
  },
  "_metadata": {
    "model_path": "./outputs/my-model",
    "tasks_requested": ["hellaswag", "mmlu"],
    "backend": "lm_eval",
    "elapsed_seconds": 342.51,
    "device": "cuda"
  }
}
```

---

## Interpreting Results

### Benchmark Score Ranges

Scores for common benchmarks vary significantly. Here are typical ranges for small models (1-3B parameters):

| Benchmark | Random Baseline | Typical 1B Model | Good 1B Fine-Tuned |
|-----------|----------------|-------------------|---------------------|
| MMLU | 25% (4-choice) | 25-35% | 35-45% |
| HellaSwag | 25% (4-choice) | 40-55% | 55-65% |
| ARC-Easy | 25% (4-choice) | 55-65% | 65-75% |
| ARC-Challenge | 25% (4-choice) | 30-40% | 40-50% |
| WinoGrande | 50% (binary) | 55-65% | 65-72% |
| TruthfulQA | ~25% | 30-40% | 40-50% |

### What to Look For

1. **Score improvement over base model** -- Even small improvements (1-3%) on hard benchmarks are meaningful.
2. **No degradation on other tasks** -- Fine-tuning for one domain should not destroy general capabilities.
3. **Domain-specific metrics** -- Custom eval metrics (F1, exact match) on your domain data are often more informative than general benchmarks.
4. **Perplexity** -- Lower perplexity on held-out data indicates better language modelling. Compare against the base model's perplexity on the same data.

### Common Pitfalls

- **Overfitting** -- High training accuracy but poor eval metrics suggests overfitting. Reduce epochs or increase regularization.
- **Data contamination** -- If your training data overlaps with benchmark test sets, scores will be artificially inflated.
- **Format mismatch** -- Benchmarks often expect specific output formats. Ensure your model generates outputs compatible with the evaluation harness.

---

## Evaluation Report Generation

When `generate_report: true` is set in the evaluation config, llm-forge produces a structured evaluation report alongside the JSON results.

```yaml
evaluation:
  enabled: true
  generate_report: true
```

---

## Listing Available Tasks

View all supported benchmark tasks:

```python
from llm_forge.evaluation.benchmarks import BenchmarkRunner

tasks = BenchmarkRunner.list_tasks()
for task in tasks:
    print(f"{task['name']:20s} {task['display_name']:20s} {task['description']}")
```

Output:

```
mmlu                 MMLU                 Massive Multitask Language Understanding
hellaswag            HellaSwag            Commonsense Natural Language Inference
arc_easy             ARC-Easy             AI2 Reasoning Challenge (Easy)
arc_challenge        ARC-Challenge        AI2 Reasoning Challenge (Challenge)
winogrande           WinoGrande           Winograd Schema Challenge at Scale
truthfulqa_mc2       TruthfulQA (MC2)     Measuring Truthfulness in Language Models
gsm8k                GSM8K                Grade School Math (8K)
```

You can also pass any valid lm-eval task name directly -- llm-forge will forward it to the harness.

---

## Perplexity Fallback Evaluation

When lm-eval is not installed, llm-forge automatically falls back to a perplexity-based evaluation using a small set of diverse evaluation texts:

```python
runner = BenchmarkRunner()
results = runner.run_benchmarks("./outputs/my-model", tasks=["hellaswag"])

# If lm-eval is not installed, results will contain:
# {
#   "perplexity_eval": {
#     "display_name": "Perplexity (fallback)",
#     "score": 15.23,
#     "avg_loss": 2.72,
#     "num_tokens": 312,
#     "metric": "perplexity",
#     "note": "Lower is better. lm-eval not installed; install with pip install 'llm-forge[eval]'."
#   }
# }
```

---

## Next Steps

- [Training Guide](training_guide.md) -- optimize your training pipeline
- [Deployment Guide](deployment.md) -- serve your evaluated model
- [Configuration Reference](configuration.md) -- evaluation config fields
