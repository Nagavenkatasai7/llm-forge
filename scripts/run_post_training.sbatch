#!/bin/bash
# ============================================================================
# SLURM Job: Post-Training Pipeline — 1x A100 MIG 3g.40gb
# ============================================================================
# Runs the post-training stages that couldn't execute under torchrun:
#   1. Evaluation — 7 benchmarks (hellaswag, arc_easy, mmlu, truthfulqa_mc2,
#                                  ifeval, winogrande, gsm8k)
#   2. ITI Probing — Discover truthfulness directions in 48 attention heads
#   3. ITI Baking — Bake directions as o_proj biases (zero inference cost)
#   4. GGUF Export — Q4_K_M quantization for Ollama/llama.cpp deployment
#
# Model:  Merged LoRA (Llama-3.2-1B + Finance-Instruct-500k)
# Target: 1x MIG 3g.40gb slice (single-process, no torchrun)
#         40GB VRAM is plenty for 1.3B model inference
# ============================================================================

#SBATCH --job-name=finance-post-train
#SBATCH --partition=contrib-gpuq
#SBATCH --account=cs678fl25
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --exclude=gpu032,dgx003
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nchennu@gmu.edu

set -euo pipefail

# Force unbuffered Python output so SLURM .out file updates in real-time
export PYTHONUNBUFFERED=1

# ============================================================================
# Environment Setup
# ============================================================================
echo "=============================================="
echo "Job:       $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node:      $(hostname)"
echo "GPUs:      ${SLURM_GPUS_ON_NODE:-1}"
echo "Started:   $(date)"
echo "=============================================="

# Activate conda (NEVER use module load python — it overrides conda)
eval "$($HOME/miniforge/bin/conda shell.bash hook)"
conda activate llm-forge

# Fix CXXABI version mismatch: use conda's libstdc++ instead of system GCC 9.3.0
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

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
import lm_eval; print(f'lm-eval {lm_eval.__version__}')
import transformers; print(f'transformers {transformers.__version__}')
"

# ============================================================================
# Config and Paths
# ============================================================================
PROJECT_DIR="$HOME/llm-forge"
MODEL_PATH="$HOME/llm-forge/outputs/finance-specialist-llama1b/merged"
OUTPUT_DIR="$HOME/llm-forge/outputs/finance-specialist-llama1b"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$LOG_DIR"

echo ""
echo "Model:      $MODEL_PATH"
echo "Output:     $OUTPUT_DIR"
echo ""

# Verify model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

echo "Model files:"
ls -lh "$MODEL_PATH/"
echo ""

# ============================================================================
# HuggingFace + W&B Config
# ============================================================================
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || echo "")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

# W&B disabled (no API key on cluster)
export WANDB_MODE="disabled"

# ============================================================================
# GGUF export tools (pre-built on login node — DO NOT pip install llama.cpp reqs)
# ============================================================================
echo "Checking GGUF export tools..."
if [ -f "$HOME/llama.cpp/convert_hf_to_gguf.py" ]; then
    export PATH="$HOME/llama.cpp:$HOME/llama.cpp/build/bin:$PATH"
    echo "  convert_hf_to_gguf.py: OK"
    if which llama-quantize >/dev/null 2>&1; then
        echo "  llama-quantize: OK"
    else
        echo "  llama-quantize: NOT FOUND (will try f16 GGUF only)"
    fi
else
    echo "  llama.cpp tools not found — GGUF export will be skipped"
fi
echo ""

# ============================================================================
# Run Post-Training Pipeline
# ============================================================================
echo "Launching post-training pipeline..."
echo "  Start time: $(date)"
echo ""

cd "$PROJECT_DIR"
PYTHONPATH=src python scripts/run_post_training.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --skip-gguf \
    2>&1 | tee "$LOG_DIR/post_training.log"

PIPELINE_EXIT=$?

# ============================================================================
# GGUF Export (run directly from shell for reliable LD_LIBRARY_PATH)
# ============================================================================
echo ""
echo "=============================================="
echo "  GGUF Export (Q4_K_M) — Direct Shell"
echo "=============================================="

GGUF_SOURCE="$OUTPUT_DIR/merged_iti"
if [ ! -d "$GGUF_SOURCE" ]; then
    GGUF_SOURCE="$MODEL_PATH"
fi
GGUF_F16="$OUTPUT_DIR/finance-specialist-llama1b-f16.gguf"
GGUF_Q4="$OUTPUT_DIR/finance-specialist-llama1b-Q4_K_M.gguf"

echo "Source:  $GGUF_SOURCE"
echo "F16:     $GGUF_F16"
echo "Q4_K_M:  $GGUF_Q4"
echo ""

if [ -f "$HOME/llama.cpp/convert_hf_to_gguf.py" ]; then
    echo "[Step 1] Converting HF model to f16 GGUF..."
    python "$HOME/llama.cpp/convert_hf_to_gguf.py" \
        "$GGUF_SOURCE" \
        --outfile "$GGUF_F16" \
        --outtype f16 2>&1

    if [ $? -eq 0 ] && [ -f "$GGUF_F16" ]; then
        F16_SIZE=$(du -h "$GGUF_F16" | cut -f1)
        echo "  f16 GGUF created: $GGUF_F16 ($F16_SIZE)"

        if [ -x "$HOME/llama.cpp/build/bin/llama-quantize" ]; then
            echo "[Step 2] Quantizing to Q4_K_M..."
            "$HOME/llama.cpp/build/bin/llama-quantize" \
                "$GGUF_F16" "$GGUF_Q4" Q4_K_M 2>&1

            if [ $? -eq 0 ] && [ -f "$GGUF_Q4" ]; then
                Q4_SIZE=$(du -h "$GGUF_Q4" | cut -f1)
                echo "  Q4_K_M GGUF created: $GGUF_Q4 ($Q4_SIZE)"
                # Remove intermediate f16
                rm -f "$GGUF_F16"
                echo "  Removed intermediate f16 GGUF"
            else
                echo "  Quantization failed — keeping f16 GGUF"
            fi
        else
            echo "  llama-quantize not available — keeping f16 GGUF"
        fi
    else
        echo "  f16 conversion failed"
    fi
else
    echo "  GGUF export skipped (no convert script)"
fi

echo ""
echo "=============================================="
echo "Pipeline exit code: $PIPELINE_EXIT"
echo "End time: $(date)"
echo "Wall time: $SECONDS seconds"
echo "=============================================="

# ============================================================================
# Final Artifact Summary
# ============================================================================
echo ""
echo "Output directory contents:"
ls -lhR "$OUTPUT_DIR/" 2>/dev/null | head -60

exit $PIPELINE_EXIT
