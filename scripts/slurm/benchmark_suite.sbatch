#!/bin/bash
#SBATCH --job-name=llm-forge-bench
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# llm-forge Benchmark Suite — Runs all benchmark configs sequentially
# Tests multiple models, training modes, and evaluation
# Usage: sbatch scripts/slurm/benchmark_suite.sbatch
# Or run a specific config: sbatch scripts/slurm/benchmark_suite.sbatch configs/benchmark_smollm_135m.yaml

set -e

# Create output directories
mkdir -p $HOME/llm-forge/outputs

# Load only gnu10 for CUDA/GPU support — do NOT load system python
module load gnu10

# Activate Miniforge conda and the llm-forge environment
eval "$($HOME/miniforge/bin/conda shell.bash hook)"
conda activate llm-forge

# Print environment
echo "=========================================="
echo "  llm-forge Benchmark Suite"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
nvidia-smi
echo "=========================================="

cd $HOME/llm-forge

# If a specific config was passed as argument, run only that
if [ -n "$1" ]; then
    echo ""
    echo ">>> Running single config: $1"
    python -m llm_forge.cli validate "$1"
    python -m llm_forge.cli train --config "$1" --verbose
    echo ">>> Completed: $1"
    echo "End: $(date)"
    exit 0
fi

# Otherwise run the full benchmark suite (smallest to largest)
CONFIGS=(
    "configs/benchmark_smollm_135m.yaml"
    "configs/benchmark_smollm_360m.yaml"
    "configs/benchmark_tinyllama_1b.yaml"
    "configs/benchmark_qlora_phi2.yaml"
    "configs/benchmark_llama_1b_full.yaml"
)

PASSED=0
FAILED=0
RESULTS=""

for config in "${CONFIGS[@]}"; do
    echo ""
    echo "=========================================="
    echo ">>> Benchmark: $config"
    echo ">>> Start: $(date)"
    echo "=========================================="

    # Validate first
    if ! python -m llm_forge.cli validate "$config" 2>&1; then
        echo ">>> SKIP: Validation failed for $config"
        FAILED=$((FAILED + 1))
        RESULTS="$RESULTS\n  FAIL: $config (validation error)"
        continue
    fi

    # Run training
    if python -m llm_forge.cli train --config "$config" --verbose 2>&1; then
        echo ">>> PASS: $config"
        PASSED=$((PASSED + 1))
        RESULTS="$RESULTS\n  PASS: $config"
    else
        echo ">>> FAIL: $config"
        FAILED=$((FAILED + 1))
        RESULTS="$RESULTS\n  FAIL: $config"
    fi

    echo ">>> End: $(date)"
    echo ""
done

echo ""
echo "=========================================="
echo "  Benchmark Suite Results"
echo "=========================================="
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo -e "  Details:$RESULTS"
echo ""
echo "  Job End: $(date)"
echo "=========================================="
