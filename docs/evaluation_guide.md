# Evaluation Guide

Benchmark, evaluate, and compare your fine-tuned models using llm-forge's multi-layered evaluation system: standard benchmarks, domain-specific evaluation, LLM-as-Judge, and knowledge retention probes.

---

## Table of Contents

1. [Evaluation Overview](#evaluation-overview)
2. [Quick Start](#quick-start)
3. [Standard Benchmarks via lm-evaluation-harness](#standard-benchmarks-via-lm-evaluation-harness)
4. [HTML Report Generation](#html-report-generation)
5. [Custom Domain Evaluation](#custom-domain-evaluation)
6. [LLM-as-Judge Evaluation](#llm-as-judge-evaluation)
7. [Knowledge Retention Probes](#knowledge-retention-probes)
8. [Model Comparison and Regression Detection](#model-comparison-and-regression-detection)
9. [Interpreting Benchmark Results](#interpreting-benchmark-results)
10. [Available Metrics](#available-metrics)
11. [Perplexity Fallback Evaluation](#perplexity-fallback-evaluation)

---

## Evaluation Overview

llm-forge provides five complementary evaluation approaches:

| Approach | Purpose | When to Use |
|----------|---------|-------------|
| **Standard Benchmarks** | Measure general capabilities (MMLU, GSM8K, etc.) | After every fine-tuning run to detect regressions |
| **Domain Evaluation** | Evaluate on your own test sets with generation metrics | When you have domain-specific test data |
| **LLM-as-Judge** | Use a stronger model to score outputs on criteria | When automated metrics are insufficient |
| **Knowledge Retention** | 100-question MCQ test across 10 knowledge domains | To detect catastrophic forgetting |
| **Perplexity Fallback** | Basic evaluation when lm-eval is not installed | Quick sanity check on any machine |

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

### Run from Python

```python
from llm_forge.evaluation.benchmarks import BenchmarkRunner

runner = BenchmarkRunner()
results = runner.run_benchmarks(
    model_path="./outputs/my-lora/merged",
    tasks=["hellaswag", "mmlu", "gsm8k"],
    batch_size=8,
)

# Save results
runner.save_results(results, "./eval_results.json")
```

---

## Standard Benchmarks via lm-evaluation-harness

llm-forge integrates with EleutherAI's lm-evaluation-harness to run standardized benchmarks.

### Installation

```bash
pip install llm-forge[eval]
```

This installs `lm-eval` (lm-evaluation-harness). Without it, llm-forge falls back to perplexity evaluation.

### Supported Benchmarks

| Benchmark | Task Name | Description | Default Few-Shot | Metric |
|-----------|-----------|-------------|-----------------|--------|
| MMLU | `mmlu` | Massive Multitask Language Understanding (57 subjects) | 5 | `acc` |
| HellaSwag | `hellaswag` | Commonsense Natural Language Inference | 10 | `acc_norm` |
| ARC-Easy | `arc_easy` | AI2 Reasoning Challenge (Easy) | 25 | `acc_norm` |
| ARC-Challenge | `arc_challenge` | AI2 Reasoning Challenge (Challenge) | 25 | `acc_norm` |
| WinoGrande | `winogrande` | Winograd Schema Challenge at Scale | 5 | `acc` |
| TruthfulQA | `truthfulqa_mc2` | Measuring Truthfulness | 0 | `acc` |
| GSM8K | `gsm8k` | Grade School Math (8K problems) | 5 | `exact_match` |
| IFEval | `ifeval` | Instruction-Following Evaluation (verifiable instructions) | 0 | `prompt_level_strict_acc` |

### Task Aliases

For convenience, llm-forge supports shorthand aliases:

| Alias | Resolves To |
|-------|-------------|
| `arc` | `arc_challenge` |
| `truthfulqa` | `truthfulqa_mc2` |
| `gsm` | `gsm8k` |
| `instruction_following` | `ifeval` |

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
    - ifeval
  num_fewshot: 5           # Override default few-shot count for all tasks
  batch_size: 16           # Larger batch = faster evaluation
```

### CLI Command

```bash
# Run default benchmarks (mmlu, hellaswag, arc_challenge, winogrande)
llm-forge eval --config config.yaml --model-path ./outputs/my-model

# Run specific benchmarks
llm-forge eval --config config.yaml --model-path ./outputs/my-model \
  --benchmarks mmlu gsm8k ifeval

# Quick test with limited samples
llm-forge eval --config config.yaml --model-path ./outputs/my-model \
  --limit 100
```

### Quick Testing with Limited Samples

Use `limit` to cap the number of evaluation samples per task for quick iteration:

```python
runner = BenchmarkRunner()
results = runner.run_benchmarks(
    model_path="./outputs/my-lora/merged",
    tasks=["hellaswag", "arc_easy"],
    limit=100,              # Only evaluate on 100 samples per task
)
```

### Chat Template Application

For instruction-tuned models, the benchmarks should be run with the chat template applied. This is critical for IFEval and other instruction-following benchmarks:

```python
results = runner.run_benchmarks(
    model_path="./outputs/my-model",
    tasks=["ifeval"],
    apply_chat_template=True,  # Default: True
)
```

### SLURM Batch Scripts for HPC

For running benchmarks on GPU clusters, see the `hopper_benchmark/` directory for example SBATCH scripts. Key points:

- Always set `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` in SBATCH
- Use `--exclude=gpu032,dgx003` and `--qos=gpu` on Hopper
- A100 80GB can run full benchmark suites; smaller GPUs may need `--limit`

---

## HTML Report Generation

When `generate_report: true` is set, llm-forge produces a self-contained HTML report with:

- **Quality report card**: Pass/fail verdict, letter grade (A+ through D), overall quality score (0-100)
- **Benchmark score table**: Per-task scores with visual bar charts and standard error
- **Model comparison table**: Side-by-side base vs fine-tuned with delta indicators
- **Training loss curve**: SVG bar chart with hover tooltips
- **Configuration summary**: Collapsible display of all training parameters
- **Sample outputs**: Model input/output pairs for manual inspection

### Quality Report Card Dimensions

The quality card evaluates three dimensions:

| Dimension | Weight | Pass Criteria |
|-----------|--------|---------------|
| Benchmark Performance | 40% | Average score > 25% |
| Regression Check | 30% | No benchmark drops > 2% vs base model |
| Training Stability | 30% | Loss decreased during training, no NaN/Inf |

### Generating a Report Programmatically

```python
from llm_forge.evaluation.report_generator import ReportGenerator

gen = ReportGenerator(title="Finance Specialist v7 Evaluation")
gen.generate_report(
    results=benchmark_results,
    config=training_config_dict,
    output_path="report.html",
    training_history=loss_history,
    sample_outputs=[
        {"input": "What is a P/E ratio?", "output": "The P/E ratio...", "reference": "..."},
    ],
    comparison=comparison_results,
)
```

### Quality Grades

| Grade | Overall Score | Meaning |
|-------|--------------|---------|
| A+ | 90+ | Excellent -- improved on most benchmarks, no regressions |
| A | 80-89 | Very good -- solid improvements, minor issues |
| B+ | 70-79 | Good -- some improvements, acceptable regressions |
| B | 60-69 | Acceptable -- marginal improvements |
| C | 50-59 | Neutral -- no significant change |
| D | <50 | Poor -- regressions detected, review needed |

---

## Custom Domain Evaluation

For domain-specific evaluation, llm-forge provides the `DomainEvaluator` class that runs generation-based evaluation on your own test sets.

### Evaluation Dataset Format

Your evaluation dataset must be in JSONL or JSON format:

```jsonl
{"input": "What is the capital of France?", "output": "Paris", "category": "geography"}
{"input": "What is 2+2?", "output": "4", "category": "math"}
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

### Post-Processing Predictions

Apply a custom post-processing function before computing metrics:

```python
def extract_answer(prediction: str) -> str:
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

### Category Breakdown

If your evaluation data includes a `category` field, llm-forge automatically computes per-category metrics:

```python
for category, data in results["category_breakdown"].items():
    print(f"{category}: {data['count']} samples, metrics={data['metrics']}")
```

---

## LLM-as-Judge Evaluation

Uses a language model to evaluate generated responses on criteria such as helpfulness, accuracy, relevance, and coherence. Based on the MT-Bench methodology (Zheng et al., NeurIPS 2023).

### Single-Score Evaluation

Score responses on a 1-10 scale for each criterion:

```python
from llm_forge.evaluation.llm_judge import LLMJudge

judge = LLMJudge(
    judge_model="meta-llama/Llama-3.2-3B-Instruct",  # Stronger model as judge
    criteria={
        "helpfulness": "How helpful and informative is the response?",
        "accuracy": "Is the information factually correct?",
        "coherence": "Is the response well-structured and logical?",
        "relevance": "Does the response stay on topic?",
    },
    max_new_tokens=256,
)

result = judge.evaluate(
    instructions=["Explain compound interest.", "What is a bond?"],
    responses=["Compound interest is...", "A bond is..."],
    criteria=["helpfulness", "accuracy"],  # Evaluate on subset of criteria
)

print(result.mean_scores)
# {'helpfulness': 7.5, 'accuracy': 8.0}
print(result.num_evaluated)
# 2
```

### Pairwise Comparison

Compare two model responses head-to-head:

```python
pairwise = judge.pairwise_compare(
    instruction="Explain the difference between stocks and bonds.",
    response_a="Stocks represent ownership...",   # Your fine-tuned model
    response_b="Stocks and bonds are...",         # Base model
    criterion="helpfulness",
)

print(f"Winner: {pairwise.winner}")   # "A", "B", or "tie"
print(f"Reason: {pairwise.reasoning}")
```

### Default Evaluation Criteria

| Criterion | Description |
|-----------|-------------|
| `helpfulness` | How helpful and informative is the response? Does it address the user's question? |
| `accuracy` | Is the information in the response factually correct? |
| `coherence` | Is the response well-structured, logical, and easy to understand? |
| `relevance` | Does the response stay on topic and address what was asked? |

### Custom Criteria

Define your own evaluation criteria:

```python
judge = LLMJudge(
    judge_model="path/to/judge-model",
    criteria={
        "financial_accuracy": "Does the response use correct financial terminology and calculations?",
        "regulatory_awareness": "Does the response mention relevant regulations when applicable?",
        "risk_disclosure": "Does the response include appropriate risk warnings?",
    },
)
```

---

## Knowledge Retention Probes

The `KnowledgeRetentionProber` provides a curated bank of 100 factual multiple-choice questions spanning 10 knowledge domains. Run probes on both the base model and the fine-tuned model to measure how much pre-existing knowledge was retained.

### Domains Covered

| Domain | Questions | Example Topic |
|--------|-----------|---------------|
| Science | 10 | Chemical symbols, speed of light |
| Mathematics | 10 | Pi, derivatives, factorials |
| Geography | 10 | Capitals, rivers, continents |
| History | 10 | World wars, presidents |
| Literature | 10 | Shakespeare, Orwell |
| Technology | 10 | CPU, HTML, binary |
| Biology | 10 | DNA, photosynthesis |
| Physics | 10 | Newton's laws, relativity |
| Chemistry | 10 | Periodic table, pH |
| General Knowledge | 10 | Leap years, currencies |

### Running Retention Probes

```python
from llm_forge.evaluation.retention_probes import KnowledgeRetentionProber

prober = KnowledgeRetentionProber()

# Evaluate base model
base_results = prober.evaluate(base_model, tokenizer)
print(f"Base model accuracy: {base_results['accuracy']:.1%}")

# Evaluate fine-tuned model
ft_results = prober.evaluate(finetuned_model, tokenizer)
print(f"Fine-tuned accuracy: {ft_results['accuracy']:.1%}")
```

### Comparing Retention

```python
comparison = KnowledgeRetentionProber.compare_retention(
    base_results=base_results,
    finetuned_results=ft_results,
)

print(f"Retention rate: {comparison['retention_rate']:.1%}")
print(f"Questions forgotten: {comparison['forgotten']}")
print(f"Questions gained: {comparison['gained']}")
print(f"Net change: {comparison['net_change']}")

# Per-domain breakdown
for domain, data in comparison['per_domain'].items():
    print(f"  {domain}: base={data['base_accuracy']:.1%} -> ft={data['finetuned_accuracy']:.1%} (delta={data['delta']:+.1%})")
```

### How Scoring Works

The prober uses log-probability scoring rather than generation. For each question, it:

1. Formats the question as a multiple-choice prompt
2. Feeds it to the model
3. Compares the log-probabilities of tokens "A", "B", "C", "D" at the last position
4. The highest-probability choice is the model's answer
5. Confidence is computed via softmax over the four choice scores

This is faster and more reliable than generating text and parsing the answer.

### Per-Domain Analysis

```python
# Evaluate specific domains only
math_results = prober.evaluate(
    model, tokenizer,
    domains=["math", "science"],
)
print(f"Math+Science accuracy: {math_results['accuracy']:.1%}")

# Get probes for a specific domain
math_probes = prober.get_probes_by_domain("math")
print(f"Number of math probes: {len(math_probes)}")
```

### Acceptable Retention Thresholds

| Retention Rate | Grade | Action |
|---------------|-------|--------|
| >95% | Excellent | No forgetting detected |
| 90-95% | Good | Minor forgetting, likely acceptable |
| 80-90% | Moderate | Review training hyperparameters |
| <80% | Concerning | Significant forgetting -- reduce LR, rank, or epochs |

---

## Model Comparison and Regression Detection

### Comparing Base vs Fine-Tuned

```python
runner = BenchmarkRunner()

comparison = runner.compare_models(
    base_path="meta-llama/Llama-3.2-1B",
    finetuned_path="./outputs/my-lora/merged",
    tasks=["hellaswag", "arc_easy", "mmlu", "gsm8k", "ifeval"],
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
```

### Regression Detection

```python
regression_report = BenchmarkRunner.check_regression(
    comparison=comparison["comparison"],
    threshold=-0.02,  # Flag any benchmark that dropped by >2%
)

print(f"Passed: {regression_report['passed']}")
print(f"Grade: {regression_report['grade']}")
print(f"Regressions: {len(regression_report['regressions'])}")
print(f"Improvements: {regression_report['num_improved']}")

if not regression_report['passed']:
    for reg in regression_report['regressions']:
        print(f"  REGRESSION: {reg['display_name']} dropped from "
              f"{reg['base_score']:.4f} to {reg['finetuned_score']:.4f} "
              f"(delta={reg['delta']:+.4f})")
```

### Regression Grades

| Grade | Average Delta | Meaning |
|-------|--------------|---------|
| A+ | > +20% | Massive improvements across all benchmarks |
| A | > +10% | Strong improvements |
| B+ | > +5% | Good improvements |
| B | > 0% | Marginal improvements |
| C | 0% | No change |
| D | < 0% | Regressions detected |

---

## Interpreting Benchmark Results

### Benchmark Score Ranges for Small Models (1-3B Parameters)

| Benchmark | Random Baseline | Typical 1B | Good 1B Fine-Tuned | Notes |
|-----------|----------------|-----------|---------------------|-------|
| MMLU | 25% (4-choice) | 25-35% | 35-45% | 57-subject average; some subjects easier |
| HellaSwag | 25% (4-choice) | 40-55% | 55-65% | Commonsense reasoning |
| ARC-Easy | 25% (4-choice) | 55-65% | 65-75% | Science questions (easy) |
| ARC-Challenge | 25% (4-choice) | 30-40% | 40-50% | Science questions (hard) |
| WinoGrande | 50% (binary) | 55-65% | 65-72% | Coreference resolution |
| TruthfulQA | ~25% | 30-40% | 40-50% | Measures truthfulness |
| GSM8K | 0% (exact match) | 20-35% | 30-45% | Math word problems |
| IFEval | 0% | 30-45% | 40-55% | Instruction-following |

### What Acceptable Regression Looks Like

Based on the finance-specialist v7 benchmarks (production-validated):

| Category | Acceptable Regression | Concerning Regression | Critical Regression |
|----------|----------------------|-----------------------|---------------------|
| Per-benchmark | < 2% drop | 2-5% drop | > 5% drop |
| MMLU (knowledge) | < 1% | 1-3% | > 3% |
| GSM8K (math) | < 3% | 3-10% | > 10% |
| IFEval (instruction) | < 3% | 3-10% | > 10% |
| Domain-specific | Improvement expected | No change | Any drop |

### Common Evaluation Pitfalls

**Data contamination**: If your training data overlaps with benchmark test sets, scores will be artificially inflated. Especially watch out with MMLU and ARC, which appear in many training datasets.

**Format mismatch**: Benchmarks expect specific output formats. For instruction-tuned models, always use `apply_chat_template=True` in the benchmark runner.

**Overfitting vs generalization**: High training accuracy but poor eval metrics suggests overfitting. Check that eval loss is not diverging from train loss.

**Small sample size**: Running benchmarks with `limit=100` gives a rough estimate but can differ from the full evaluation by 5-10%. Always run full benchmarks for production decisions.

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
| Perplexity | `compute_perplexity` | Model perplexity over text samples (lower is better) |

### Using the Metrics Computer Directly

```python
from llm_forge.evaluation.metrics import MetricsComputer

mc = MetricsComputer()

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

### Graceful Fallbacks

- **BLEU** falls back to a simple n-gram precision implementation when NLTK is not installed.
- **ROUGE** falls back to a longest-common-subsequence ROUGE-L implementation when `rouge-score` is not installed.
- Install the full evaluation suite with: `pip install llm-forge[eval]`

---

## Perplexity Fallback Evaluation

When lm-eval is not installed, llm-forge automatically falls back to a perplexity-based evaluation using a small set of diverse evaluation texts covering science, geography, math, history, and literature.

```python
runner = BenchmarkRunner()
results = runner.run_benchmarks("./outputs/my-model", tasks=["hellaswag"])

# If lm-eval is not installed:
# {
#   "perplexity_eval": {
#     "display_name": "Perplexity (fallback)",
#     "score": 15.23,
#     "avg_loss": 2.72,
#     "num_tokens": 312,
#     "metric": "perplexity",
#     "note": "Lower is better."
#   }
# }
```

Perplexity interpretation:

| Perplexity | Quality |
|-----------|---------|
| < 10 | Excellent (model very confident on evaluation texts) |
| 10-20 | Good |
| 20-50 | Moderate |
| 50-100 | Poor (model struggling with basic text) |
| > 100 | Very poor (model likely damaged by training) |

---

## Saving and Loading Results

### Save Results

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

## Listing Available Tasks

```python
from llm_forge.evaluation.benchmarks import BenchmarkRunner

tasks = BenchmarkRunner.list_tasks()
for task in tasks:
    print(f"{task['name']:20s} {task['display_name']:20s} {task['description']}")
```

You can also pass any valid lm-eval task name directly -- llm-forge will forward it to the harness. This includes MMLU subtasks like `mmlu_professional_accounting` or `mmlu_business_ethics`.

---

## Next Steps

- [Training Guide](training_guide.md) -- optimize your training pipeline
- [Deployment Guide](deployment.md) -- serve your evaluated model
- [Configuration Reference](configuration.md) -- evaluation config fields
