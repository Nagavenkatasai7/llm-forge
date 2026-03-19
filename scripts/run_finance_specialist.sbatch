#!/bin/bash
# ============================================================================
# SLURM Job: Finance Specialist Pipeline — 2x A100.80gb
# ============================================================================
# Runs the full 12-stage llm-forge DAG pipeline:
#   data_loading -> cleaning -> preprocessing -> refusal_augmentation ->
#   ifd_scoring -> training -> alignment -> iti_probing -> iti_baking ->
#   model_merging -> evaluation -> export
#
# Model:   meta-llama/Llama-3.2-1B + LoRA r=64 (DoRA + RSLoRA)
# Dataset: Josephgflowers/Finance-Instruct-500k (518K samples)
# Target:  2x NVIDIA A100 80GB on Hopper HPC (reduced from 4 — cluster fully loaded)
# ============================================================================

#SBATCH --job-name=finance-specialist
#SBATCH --partition=contrib-gpuq
#SBATCH --account=cs678fl25
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --exclude=gpu032,dgx003
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nchennu@gmu.edu

set -euo pipefail

# ============================================================================
# Environment Setup
# ============================================================================
echo "=============================================="
echo "Job:       $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node:      $(hostname)"
echo "GPUs:      $SLURM_GPUS_ON_NODE"
echo "Started:   $(date)"
echo "=============================================="

# Activate conda (NEVER use module load python — it overrides conda)
eval "$($HOME/miniforge/bin/conda shell.bash hook)"
conda activate llm-forge

# Verify critical packages
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({mem / 1e9:.0f} GB)')
import transformers; print(f'transformers {transformers.__version__}')
import peft; print(f'peft {peft.__version__}')
import trl; print(f'trl {trl.__version__}')
"

# ============================================================================
# Config and Paths
# ============================================================================
PROJECT_DIR="$HOME/llm-forge"
CONFIG="$PROJECT_DIR/configs/finance_specialist_h100.yaml"
OUTPUT_DIR="$HOME/llm-forge/outputs/finance-specialist-llama1b"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Copy config to output for reproducibility
cp "$CONFIG" "$OUTPUT_DIR/config_snapshot.yaml"

echo ""
echo "Config:     $CONFIG"
echo "Output:     $OUTPUT_DIR"
echo "Log dir:    $LOG_DIR"
echo ""

# ============================================================================
# NCCL / Distributed Configuration
# ============================================================================
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# HuggingFace cache and auth (use shared storage)
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || echo "")

# W&B config — disabled (no API key on cluster; set WANDB_MODE=online when ready)
export WANDB_MODE="disabled"
export WANDB_PROJECT="finance-specialist"
export WANDB_RUN_GROUP="llama-1b-lora64-dora"

echo "NCCL config:"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"
echo "  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo ""

# ============================================================================
# Stage 0: Pre-flight Checks
# ============================================================================
echo "[Stage 0] Pre-flight checks..."

# Verify config validates
cd "$PROJECT_DIR"
PYTHONPATH=src python -c "
from llm_forge.config.schema import LLMForgeConfig
import yaml
with open('$CONFIG') as f:
    data = yaml.safe_load(f)
config = LLMForgeConfig(**data)
print(f'  Config validated: {config.model.name}')
print(f'  LoRA r={config.lora.r}, DoRA={config.lora.use_dora}, RSLoRA={config.lora.use_rslora}')
print(f'  Effective batch size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps * config.distributed.num_gpus}')
print(f'  Pipeline stages: cleaning={config.data.cleaning.enabled}, refusal={config.refusal.enabled}, ifd={config.ifd.enabled}, iti={config.iti.enabled}')
"

echo "[Stage 0] Pre-flight PASSED"
echo ""

# ============================================================================
# Stage 1-12: Full Pipeline via torchrun
# ============================================================================
echo "[Stages 1-12] Launching full pipeline with torchrun..."
echo "  Start time: $(date)"

# Ensure HF token is available to all child processes
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || echo "")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
echo "HF_TOKEN set: ${HF_TOKEN:0:8}..."

# Run the full 12-stage DAG pipeline
cd "$PROJECT_DIR"
PYTHONPATH=src HF_TOKEN="$HF_TOKEN" torchrun \
    --nproc_per_node=2 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m llm_forge.cli train \
    --config "$CONFIG" \
    --no-auto-optimize \
    --verbose \
    2>&1 | tee "$LOG_DIR/pipeline_full.log"

TRAIN_EXIT=$?

echo ""
echo "Pipeline exit code: $TRAIN_EXIT"
echo "End time: $(date)"

# ============================================================================
# Post-Training Summary
# ============================================================================
echo ""
echo "=============================================="
echo "POST-TRAINING SUMMARY"
echo "=============================================="

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "STATUS: SUCCESS"

    # List output artifacts
    echo ""
    echo "Output artifacts:"
    ls -lh "$OUTPUT_DIR/" 2>/dev/null || echo "  (no output dir found)"

    # Check for GGUF export
    if ls "$OUTPUT_DIR"/*.gguf 2>/dev/null; then
        echo ""
        echo "GGUF file(s) exported:"
        ls -lh "$OUTPUT_DIR"/*.gguf
    fi

    # Run results analyzer if available
    if [ -f "$PROJECT_DIR/scripts/analyze_results.py" ]; then
        echo ""
        echo "Running results analyzer..."
        PYTHONPATH=src python "$PROJECT_DIR/scripts/analyze_results.py" \
            --output-dir "$OUTPUT_DIR" \
            --log-file "$LOG_DIR/pipeline_full.log" \
            2>&1 | tee "$LOG_DIR/analysis_report.txt"
    fi
else
    echo "STATUS: FAILED (exit code $TRAIN_EXIT)"
    echo ""
    echo "Last 50 lines of log:"
    tail -50 "$LOG_DIR/pipeline_full.log"
fi

echo ""
echo "=============================================="
echo "Job completed: $(date)"
echo "Wall time: $SECONDS seconds"
echo "=============================================="
