#!/bin/bash
#SBATCH --job-name=llm-forge-distributed
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# llm-forge Multi-Node Distributed Training for GMU Hopper Cluster
# Usage: sbatch train_multi_node.sbatch configs/example_lora.yaml

set -e

mkdir -p logs

module load cuda/12.2
module load anaconda3
conda activate llm-forge

# Get master node address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

echo "=== Distributed Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Config: $1"
echo "============================"

# Launch distributed training
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m llm_forge.cli train --config "$1" --verbose

echo "=== Distributed Training Complete ==="
