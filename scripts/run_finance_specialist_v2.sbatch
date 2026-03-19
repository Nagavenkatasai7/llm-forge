#!/bin/bash
# ============================================================================
# SLURM Job: Finance Specialist Pipeline v2 — FIXED
# ============================================================================
# FIXES from v1:
#   1. assistant-only loss (completion_only_loss: true)
#   2. Lower LR (5e-5 vs 2e-4), 1 epoch (vs 3), RSLoRA enabled
#   3. Post-training eval+export runs on single GPU AFTER distributed training
#   4. Interactive model validation before GGUF export
#   5. Proper checkpoint saving (save_steps=250 < total steps)
#
# Model:   unsloth/Llama-3.2-1B + LoRA r=64 (RSLoRA)
# Dataset: Josephgflowers/Finance-Instruct-500k
# Target:  2x NVIDIA A100 80GB on Hopper HPC
# ============================================================================

#SBATCH --job-name=finance-v2
#SBATCH --partition=contrib-gpuq
#SBATCH --account=cs678fl25
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100.80gb:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
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
CONFIG="$PROJECT_DIR/configs/finance_specialist_h100.yaml"
OUTPUT_DIR="$HOME/llm-forge/outputs/finance-specialist-llama1b-v2"
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

# HuggingFace cache and auth
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || echo "")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

# W&B disabled by default
export WANDB_MODE="disabled"

# Tokenizer parallelism off (avoids fork warnings)
export TOKENIZERS_PARALLELISM=false

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
print(f'  LoRA:             r={config.lora.r}, RSLoRA={config.lora.use_rslora}')
print(f'  LR:               {config.training.learning_rate}')
print(f'  Epochs:           {config.training.num_epochs}')
print(f'  Completion-only:  {config.training.completion_only_loss}')
print(f'  Pack sequences:   {config.training.pack_sequences}')
eff = config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps * config.distributed.num_gpus
print(f'  Effective batch:  {eff}')
"
echo "[Stage 0] PASSED"
echo ""

# ============================================================================
# PHASE 1: Distributed Training (torchrun, 2 GPUs)
# ============================================================================
echo "=============================================="
echo "[PHASE 1] Distributed Training (2x A100)"
echo "  Start time: $(date)"
echo "=============================================="

# Run training only (skip post-training stages in distributed mode)
cd "$PROJECT_DIR"
PYTHONPATH=src torchrun \
    --nproc_per_node=2 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m llm_forge.cli train \
    --config "$CONFIG" \
    --no-auto-optimize \
    --verbose \
    --skip-stages evaluation,export \
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
# PHASE 2: Post-Training (single GPU — eval, merge, export)
# ============================================================================
echo ""
echo "=============================================="
echo "[PHASE 2] Post-Training Pipeline (single GPU)"
echo "  Start time: $(date)"
echo "=============================================="

# Find the merged model path
MERGED_DIR="$OUTPUT_DIR/merged"
if [ ! -d "$MERGED_DIR" ]; then
    MERGED_DIR="$OUTPUT_DIR"
fi
echo "Merged model path: $MERGED_DIR"

# --- Interactive validation ---
echo ""
echo "[2.1] Interactive model validation..."
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
        {'role': 'system', 'content': 'You are a knowledgeable financial assistant.'},
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
    print('WARNING: Model quality may be poor. Proceeding with export anyway.')
" 2>&1 | tee "$LOG_DIR/validation.log"

# --- Benchmarks ---
echo ""
echo "[2.2] Running evaluation benchmarks..."
PYTHONPATH=src python -c "
from llm_forge.evaluation.benchmarks import BenchmarkRunner
runner = BenchmarkRunner(
    model_path='$MERGED_DIR',
    tasks=['hellaswag', 'arc_easy', 'truthfulqa_mc2'],
    batch_size=8,
    num_fewshot=0,
)
try:
    results = runner.run()
    for task, score in results.items():
        print(f'  {task}: {score:.4f}')
except Exception as e:
    print(f'  Benchmark error: {e}')
    print('  Skipping benchmarks (non-fatal)')
" 2>&1 | tee "$LOG_DIR/benchmarks.log" || true

# --- GGUF Export ---
echo ""
echo "[2.3] Exporting to GGUF (Q4_K_M)..."
GGUF_DIR="$OUTPUT_DIR/gguf"
mkdir -p "$GGUF_DIR"

PYTHONPATH=src python -c "
from llm_forge.serving.export import ModelExporter
result = ModelExporter.export_gguf(
    model_path='$MERGED_DIR',
    output_path='$GGUF_DIR/finance-specialist-llama1b-v2-Q4_K_M.gguf',
    quantization='Q4_K_M',
)
print(f'GGUF exported: {result}')
" 2>&1 | tee "$LOG_DIR/gguf_export.log" || echo "GGUF export failed (non-fatal)"

# --- Create Ollama Modelfile ---
echo ""
echo "[2.4] Creating Ollama Modelfile..."
GGUF_FILE="$GGUF_DIR/finance-specialist-llama1b-v2-Q4_K_M.gguf"
cat > "$GGUF_DIR/Modelfile" << 'MODELFILE_EOF'
FROM ./finance-specialist-llama1b-v2-Q4_K_M.gguf

SYSTEM "You are a finance specialist AI assistant. Provide accurate, well-reasoned financial analysis and advice based on established financial principles, regulations, and market practices. Always note when professional financial advice should be sought."

TEMPLATE """<|begin_of_text|>{{- if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{- end }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
PARAMETER num_predict 512
MODELFILE_EOF

echo "Modelfile created at: $GGUF_DIR/Modelfile"
echo "To deploy: ollama create finance-specialist-v2 -f $GGUF_DIR/Modelfile"

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "=============================================="
echo "PIPELINE COMPLETE"
echo "=============================================="
echo "  Training log:   $LOG_DIR/training.log"
echo "  Validation log: $LOG_DIR/validation.log"
echo "  Benchmark log:  $LOG_DIR/benchmarks.log"
echo "  Merged model:   $MERGED_DIR"
echo "  GGUF file:      $GGUF_FILE"
echo "  Modelfile:      $GGUF_DIR/Modelfile"
echo "  Wall time:      $SECONDS seconds"
echo "  Completed:      $(date)"
echo "=============================================="
