#!/bin/bash
#SBATCH --job-name=llm-forge-demo
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# llm-forge Demo Training — Llama-3.2-1B LoRA on Alpaca
# Runs on GMU Hopper cluster (1x A100, ~30 min expected)
# Usage: sbatch scripts/slurm/train_demo.sbatch

set -e

# Create output directories
mkdir -p $HOME/llm-forge/outputs

# Load only gnu10 for CUDA/GPU support — do NOT load system python
module load gnu10

# Activate Miniforge conda and the llm-forge environment
eval "$($HOME/miniforge/bin/conda shell.bash hook)"
conda activate llm-forge

# Change to llm-forge directory
cd $HOME/llm-forge

# Print environment info
echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python) ($(python --version))"
echo "Config: configs/demo_lora_llama.yaml"
echo "Start: $(date)"
nvidia-smi
echo "================"

# Validate config first
python -m llm_forge.cli validate configs/demo_lora_llama.yaml

# Run training
python -m llm_forge.cli train --config configs/demo_lora_llama.yaml --verbose

echo "=== Training Complete ==="
echo "End: $(date)"
echo "Outputs: ./outputs/demo-lora-llama3.2-1b/"
