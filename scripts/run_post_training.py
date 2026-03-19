#!/usr/bin/env python3
"""Post-training pipeline: Evaluation + ITI + GGUF Export.

Runs the post-training stages that couldn't execute under torchrun's
distributed mode. Must be run as a single-process job (no torchrun).

Usage:
    PYTHONPATH=src python scripts/run_post_training.py \
        --model-path ~/llm-forge/outputs/finance-specialist-llama1b/merged \
        --output-dir ~/llm-forge/outputs/finance-specialist-llama1b

Stages:
    1. Evaluation — 7 benchmarks via lm-eval harness
    2. ITI Probing — Discover truthfulness directions in attention heads
    3. ITI Baking — Bake directions as o_proj biases (zero-cost at inference)
    4. GGUF Export — Q4_K_M quantization for Ollama/llama.cpp deployment
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def stage_banner(name: str, stage_num: int, total: int = 4) -> None:
    """Print a stage header banner."""
    print(f"\n{'=' * 60}")
    print(f"  Stage {stage_num}/{total}: {name}")
    print(f"{'=' * 60}\n")


def run_evaluation(model_path: str, output_dir: str) -> dict:
    """Run 7 evaluation benchmarks via lm-eval harness."""
    stage_banner("Evaluation Benchmarks", 1)

    from llm_forge.evaluation.benchmarks import BenchmarkRunner

    runner = BenchmarkRunner(device="cuda")

    tasks = [
        "hellaswag",
        "arc_easy",
        "mmlu",
        "truthfulqa_mc2",
        "ifeval",
        "winogrande",
        "gsm8k",
    ]

    print(f"Model:  {model_path}")
    print(f"Tasks:  {', '.join(tasks)}")
    print(f"Device: cuda")
    print(f"Batch:  8")
    print()

    start = time.time()
    results = runner.run_benchmarks(
        model_path=model_path,
        tasks=tasks,
        num_fewshot=0,
        batch_size=8,
        apply_chat_template=False,
    )
    elapsed = time.time() - start

    # Print results table
    print(f"\n{'─' * 50}")
    print(f"  {'Benchmark':<20} {'Score':>10} {'Metric':<20}")
    print(f"{'─' * 50}")
    for task in tasks:
        if task in results and isinstance(results[task], dict):
            score = results[task].get("score")
            metric = results[task].get("metric", "?")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            display = results[task].get("display_name", task)
            print(f"  {display:<20} {score_str:>10} {metric:<20}")
    if "_aggregate" in results:
        avg = results["_aggregate"].get("average_score", 0)
        print(f"{'─' * 50}")
        print(f"  {'AVERAGE':<20} {avg:.4f}")
    print(f"{'─' * 50}")
    print(f"  Completed in {elapsed:.1f}s")

    # Save results
    results_path = Path(output_dir) / "eval_results.json"
    runner.save_results(results, str(results_path))
    print(f"  Results saved: {results_path}")

    return results


def run_iti(model_path: str, output_dir: str) -> str:
    """Run ITI probing and baking, save the ITI-enhanced model."""
    stage_banner("ITI Anti-Hallucination", 2)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm_forge.evaluation.iti_prober import ITIProber
    from llm_forge.serving.iti_baker import ITIBaker

    iti_output = str(Path(output_dir) / "merged_iti")

    print(f"Model:    {model_path}")
    print(f"Dataset:  truthful_qa")
    print(f"Samples:  500")
    print(f"Heads:    48")
    print(f"Alpha:    15.0")
    print(f"Method:   center_of_mass")
    print(f"Output:   {iti_output}")
    print()

    # Load model (use float16 — bf16 causes ScalarType errors on MIG slices)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"  Model loaded: {model.config.model_type}, {model.num_parameters() / 1e9:.2f}B params")

    # Stage 2a: ITI Probing
    print("\n[2a] Probing attention heads for truthfulness directions...")
    start = time.time()

    prober = ITIProber(
        num_probing_samples=500,
        num_heads=48,
        method="center_of_mass",
    )

    # Create a simple config-like object for the prober
    class ITIConfig:
        probing_dataset = "truthful_qa"

    probe_result = prober.probe(model, tokenizer, ITIConfig())
    probe_elapsed = time.time() - start

    print(f"  Probing complete in {probe_elapsed:.1f}s")
    print(f"  Geometry: {probe_result.num_layers} layers, {probe_result.num_heads_per_layer} heads/layer, dim={probe_result.head_dim}")
    print(f"  Top heads selected: {len(probe_result.top_heads)}")

    # Show top 5 heads
    for i, (layer, head) in enumerate(probe_result.top_heads[:5]):
        acc = probe_result.probe_accuracies.get((layer, head), 0)
        print(f"    #{i + 1}: Layer {layer}, Head {head} (acc={acc:.3f})")

    # Stage 2b: ITI Baking
    print("\n[2b] Baking truthfulness directions into model weights...")
    start = time.time()

    baker = ITIBaker()
    model = baker.bake_interventions(
        model=model,
        directions=probe_result.directions,
        top_heads=probe_result.top_heads,
        alpha=15.0,
        sigmas=probe_result.sigmas,
    )
    bake_elapsed = time.time() - start
    print(f"  Baking complete in {bake_elapsed:.1f}s")

    # Save ITI-enhanced model
    print(f"\n  Saving ITI-enhanced model to {iti_output}...")
    os.makedirs(iti_output, exist_ok=True)
    model.save_pretrained(iti_output, safe_serialization=True)
    tokenizer.save_pretrained(iti_output)

    # Save probe results
    probe_info = {
        "num_layers": probe_result.num_layers,
        "num_heads_per_layer": probe_result.num_heads_per_layer,
        "head_dim": probe_result.head_dim,
        "num_top_heads": len(probe_result.top_heads),
        "top_heads": [(l, h) for l, h in probe_result.top_heads[:10]],
        "top_accuracies": {
            f"L{l}_H{h}": probe_result.probe_accuracies.get((l, h), 0)
            for l, h in probe_result.top_heads[:10]
        },
        "alpha": 15.0,
        "method": "center_of_mass",
        "probing_dataset": "truthful_qa",
        "num_probing_samples": 500,
    }
    probe_path = Path(output_dir) / "iti_probe_results.json"
    with open(probe_path, "w") as f:
        json.dump(probe_info, f, indent=2)
    print(f"  Probe results saved: {probe_path}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return iti_output


def run_gguf_export(model_path: str, output_dir: str) -> None:
    """Export model to GGUF Q4_K_M format."""
    stage_banner("GGUF Export (Q4_K_M)", 3)

    from llm_forge.serving.export import ModelExporter

    gguf_path = str(Path(output_dir) / "finance-specialist-llama1b-Q4_K_M.gguf")

    print(f"Model:        {model_path}")
    print(f"Output:       {gguf_path}")
    print(f"Quantization: Q4_K_M")
    print()

    start = time.time()
    try:
        result = ModelExporter.export_gguf(
            model_path=model_path,
            output_path=gguf_path,
            quantization="q4_k_m",
        )
        elapsed = time.time() - start
        size_mb = result.stat().st_size / (1024 * 1024) if result.exists() else 0
        print(f"  GGUF export complete in {elapsed:.1f}s")
        print(f"  File: {result}")
        print(f"  Size: {size_mb:.1f} MB")
    except (ImportError, RuntimeError) as e:
        print(f"  GGUF export SKIPPED: {e}")
        print("  Install llama-cpp-python or llama.cpp tools to enable GGUF export.")


def run_eval_iti_model(iti_model_path: str, output_dir: str) -> dict | None:
    """Optionally re-evaluate the ITI-enhanced model to measure improvement."""
    stage_banner("Evaluate ITI Model (optional)", 4)

    if not Path(iti_model_path).exists():
        print("  ITI model not found, skipping re-evaluation.")
        return None

    from llm_forge.evaluation.benchmarks import BenchmarkRunner

    runner = BenchmarkRunner(device="cuda")

    # Only run truthfulqa to measure ITI impact
    tasks = ["truthfulqa_mc2"]

    print(f"Model:  {iti_model_path}")
    print(f"Task:   TruthfulQA MC2 (measuring ITI impact)")
    print()

    start = time.time()
    results = runner.run_benchmarks(
        model_path=iti_model_path,
        tasks=tasks,
        num_fewshot=0,
        batch_size=8,
        apply_chat_template=False,
    )
    elapsed = time.time() - start

    if "truthfulqa_mc2" in results:
        score = results["truthfulqa_mc2"].get("score")
        print(f"  TruthfulQA MC2 (ITI): {score:.4f}" if score else "  TruthfulQA MC2 (ITI): N/A")

    # Save
    results_path = Path(output_dir) / "eval_results_iti.json"
    runner.save_results(results, str(results_path))
    print(f"  Results saved: {results_path}")
    print(f"  Completed in {elapsed:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Post-training pipeline")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to merged model directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation benchmarks",
    )
    parser.add_argument(
        "--skip-iti",
        action="store_true",
        help="Skip ITI probing and baking",
    )
    parser.add_argument(
        "--skip-gguf",
        action="store_true",
        help="Skip GGUF export",
    )
    parser.add_argument(
        "--skip-iti-eval",
        action="store_true",
        help="Skip re-evaluation of ITI model",
    )
    args = parser.parse_args()

    model_path = os.path.expandvars(os.path.expanduser(args.model_path))
    output_dir = os.path.expandvars(os.path.expanduser(args.output_dir))

    print("=" * 60)
    print("  llm-forge Post-Training Pipeline")
    print("=" * 60)
    print(f"  Model:      {model_path}")
    print(f"  Output:     {output_dir}")
    print(f"  Started:    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Validate model path
    if not Path(model_path).exists():
        print(f"\nERROR: Model path does not exist: {model_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    pipeline_start = time.time()

    # Stage 1: Evaluation
    eval_results = None
    if not args.skip_eval:
        try:
            eval_results = run_evaluation(model_path, output_dir)
        except Exception as e:
            print(f"\n  Evaluation FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[SKIP] Evaluation benchmarks")

    # Stage 2: ITI
    iti_model_path = None
    if not args.skip_iti:
        try:
            iti_model_path = run_iti(model_path, output_dir)
        except Exception as e:
            print(f"\n  ITI FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[SKIP] ITI probing/baking")

    # Stage 3: GGUF Export (use ITI model if available, else merged model)
    gguf_source = iti_model_path if iti_model_path and Path(iti_model_path).exists() else model_path
    if not args.skip_gguf:
        try:
            run_gguf_export(gguf_source, output_dir)
        except Exception as e:
            print(f"\n  GGUF export FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[SKIP] GGUF export")

    # Stage 4: Re-evaluate ITI model (TruthfulQA only)
    if not args.skip_iti_eval and iti_model_path:
        try:
            iti_eval = run_eval_iti_model(iti_model_path, output_dir)
            # Compare TruthfulQA scores
            if eval_results and iti_eval:
                base_tqa = eval_results.get("truthfulqa_mc2", {}).get("score")
                iti_tqa = iti_eval.get("truthfulqa_mc2", {}).get("score")
                if base_tqa is not None and iti_tqa is not None:
                    delta = iti_tqa - base_tqa
                    print(f"\n  TruthfulQA improvement from ITI: {delta:+.4f} ({base_tqa:.4f} -> {iti_tqa:.4f})")
        except Exception as e:
            print(f"\n  ITI evaluation FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print("  POST-TRAINING PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total time: {total_elapsed / 60:.1f} minutes")
    print(f"  Finished:   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # List output artifacts
    print(f"\n  Output artifacts:")
    out_path = Path(output_dir)
    for f in sorted(out_path.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {f.name:<45} {size_mb:>8.1f} MB")
        elif f.is_dir():
            # Count files and total size
            total = sum(ff.stat().st_size for ff in f.rglob("*") if ff.is_file())
            count = sum(1 for ff in f.rglob("*") if ff.is_file())
            print(f"    {f.name + '/':<45} {total / (1024 * 1024):>8.1f} MB ({count} files)")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
