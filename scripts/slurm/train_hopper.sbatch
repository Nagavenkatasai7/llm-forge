#!/bin/bash
#SBATCH --job-name=llm-forge-train
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# llm-forge Training Script for GMU Hopper Cluster
# Usage: sbatch train_hopper.sbatch configs/example_lora.yaml

set -e

# Create log directory
mkdir -p logs

# Load modules
module load cuda/12.2
module load anaconda3

# Activate environment
conda activate llm-forge

# Print environment info
echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: $1"
nvidia-smi
echo "================"

# Validate config first
python -m llm_forge.cli validate "$1"

# Run training
python -m llm_forge.cli train --config "$1" --verbose

echo "=== Training Complete ==="
