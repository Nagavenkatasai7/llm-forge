#!/bin/bash
# ============================================================================
# SLURM Job: Finance Specialist v7 — Knowledge-Preserving Fine-Tuning
# ============================================================================
# Changes from v6 (which FAILED benchmarks — DO NOT DEPLOY v6):
#   1. LoRA r=8, alpha=16 (was r=16, alpha=32) — less perturbation
#   2. target_modules: attention-only q/k/v/o_proj (was all-linear)
#      → Preserves MLP reasoning layers (GSM8K dropped 27% with all-linear)
#   3. learning_rate: 1e-5 (was 5e-5) — 5x lower
#   4. NEFTune disabled (was alpha=5.0 — amplified forgetting on 5K)
#   5. label_smoothing: 0.0 (hurts MC accuracy on small data)
#   6. Data cleaning enabled (filter noise from Finance-Instruct-500k)
#   7. max_samples: 20000 (was 5000) — more data with cleaning
#   8. use_rslora: false (was causing 4x LR boost)
#   9. Single GPU (1B model doesn't need distributed)
#
# Model:   unsloth/Llama-3.2-1B-Instruct + LoRA r=8
# Dataset: Josephgflowers/Finance-Instruct-500k (20K cleaned)
# Target:  1x NVIDIA A100 80GB on Hopper HPC
# ============================================================================

#SBATCH --job-name=finance-v7
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
echo "GPUs:      ${SLURM_GPUS_ON_NODE:-1}"
echo "Started:   $(date)"
echo "=============================================="

# Activate conda (NEVER use module load python)
eval "$($HOME/miniforge/bin/conda shell.bash hook)"
conda activate llm-forge

# CXXABI compat for conda-installed libraries
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

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
CONFIG="$PROJECT_DIR/configs/finance_specialist_v7_h100.yaml"
OUTPUT_DIR="/scratch/nchennu/models/finance-specialist-v7"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Copy config to output for reproducibility
cp "$CONFIG" "$OUTPUT_DIR/config_snapshot.yaml"

echo ""
echo "Config:     $CONFIG"
echo "Output:     $OUTPUT_DIR"
echo "Log dir:    $LOG_DIR"
echo ""

# HuggingFace cache on scratch (1.9 PB filesystem)
export HF_HOME="/scratch/nchennu/.cache/huggingface"
export HF_DATASETS_CACHE="/scratch/nchennu/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/scratch/nchennu/.cache/huggingface/hub"
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || echo "")

# W&B disabled
export WANDB_MODE="disabled"

# Tokenizer parallelism off (avoids fork warnings)
export TOKENIZERS_PARALLELISM=false

# CUDA memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# Stage 0: Pre-flight Checks
# ============================================================================
echo "[Stage 0] Pre-flight checks..."
cd "$PROJECT_DIR"
PYTHONPATH=src python -c "
from llm_forge.config.schema import LLMForgeConfig
import yaml
with open('$CONFIG') as f:
    data = yaml.safe_load(f)
config = LLMForgeConfig(**data)
print(f'  Model:            {config.model.name}')
print(f'  LoRA:             r={config.lora.r}, alpha={config.lora.alpha}')
print(f'  target_modules:   {config.lora.target_modules}')
print(f'  use_rslora:       {config.lora.use_rslora}')
print(f'  LR:               {config.training.learning_rate}')
print(f'  Epochs:           {config.training.num_epochs}')
print(f'  Completion-only:  {config.training.completion_only_loss}')
print(f'  Assistant-only:   {config.training.assistant_only_loss}')
print(f'  NEFTune alpha:    {config.training.neftune_noise_alpha}')
print(f'  Data cleaning:    {config.data.cleaning.enabled}')
print(f'  Max samples:      {config.data.max_samples}')
eff = config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps
print(f'  Effective batch:  {eff}')
"
echo "[Stage 0] PASSED"
echo ""

# ============================================================================
# Training (single GPU — 1B model fits easily on A100 80GB)
# ============================================================================
echo "=============================================="
echo "[Training] Finance Specialist v7"
echo "  Start time: $(date)"
echo "=============================================="

cd "$PROJECT_DIR"
# --no-auto-optimize: CRITICAL — the auto-optimizer overrides our carefully tuned
# batch_size/gradient_checkpointing settings for A100 80GB (sets batch=16,
# checkpointing=False), causing OOM. We want our config values used as-is.
PYTHONPATH=src python -m llm_forge.cli train \
    --config "$CONFIG" \
    --no-auto-optimize \
    --verbose \
    2>&1 | tee "$LOG_DIR/training.log"

TRAIN_EXIT=$?
echo ""
echo "Training exit code: $TRAIN_EXIT"
echo "Training end time:  $(date)"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "TRAINING FAILED — aborting pipeline"
    tail -50 "$LOG_DIR/training.log"
    exit $TRAIN_EXIT
fi

# ============================================================================
# Post-Training: Interactive Validation
# ============================================================================
echo ""
echo "=============================================="
echo "[Post-Training] Model Validation"
echo "  Start time: $(date)"
echo "=============================================="

MERGED_DIR="$OUTPUT_DIR/merged"
if [ ! -d "$MERGED_DIR" ]; then
    MERGED_DIR="$OUTPUT_DIR"
fi
echo "Model path: $MERGED_DIR"

PYTHONPATH=src python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '$MERGED_DIR'
print(f'Loading model from {model_path}...')
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

prompts = [
    'What is a bond yield curve inversion?',
    'Explain the difference between a stock split and a reverse stock split.',
    'What are the key risks of investing in high-yield bonds?',
]

pass_count = 0
for i, prompt in enumerate(prompts):
    messages = [
        {'role': 'system', 'content': 'You are a finance specialist AI assistant.'},
        {'role': 'user', 'content': prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9,
                             repetition_penalty=1.1, do_sample=True)
    response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    print(f'  Q{i+1}: {prompt}')
    print(f'  A{i+1}: {response[:300]}')
    issues = []
    if len(response.strip()) < 20: issues.append('TOO_SHORT')
    if 'you are' in response.lower()[:100] and 'system' in response.lower()[:50]: issues.append('SYSTEM_REGURGITATION')
    if response.count(response[:30]) > 2: issues.append('REPETITION')
    status = 'FAIL: ' + ','.join(issues) if issues else 'PASS'
    print(f'  [{status}]')
    print()
    if not issues: pass_count += 1

print(f'Validation: {pass_count}/3 passed')
if pass_count < 2:
    print('WARNING: Model quality may be poor.')
" 2>&1 | tee "$LOG_DIR/validation.log"

# ============================================================================
# Copy merged model to standard location for benchmarking
# ============================================================================
echo ""
echo "[Post-Training] Saving model for benchmarking..."
BENCH_MODEL="/scratch/nchennu/models/finance-specialist-v7/merged"
if [ -d "$MERGED_DIR" ] && [ "$MERGED_DIR" != "$BENCH_MODEL" ]; then
    mkdir -p "$BENCH_MODEL"
    cp -r "$MERGED_DIR"/* "$BENCH_MODEL"/ 2>/dev/null || true
    echo "Model copied to: $BENCH_MODEL"
else
    echo "Model already at: $BENCH_MODEL"
fi

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=============================================="
echo "TRAINING COMPLETE"
echo "=============================================="
echo "  Training log:   $LOG_DIR/training.log"
echo "  Validation log: $LOG_DIR/validation.log"
echo "  Merged model:   $BENCH_MODEL"
echo "  Wall time:      $SECONDS seconds"
echo "  Completed:      $(date)"
echo ""
echo "  NEXT STEPS:"
echo "    1. Review training log for loss curves"
echo "    2. Run benchmark suite: cd ~/hopper_benchmark && bash submit_all_v7.sh"
echo "    3. Compare v7 vs base: python ~/hopper_benchmark/collect_results.py"
echo "=============================================="
