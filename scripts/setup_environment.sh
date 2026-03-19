#!/bin/bash
# llm-forge One-Command Setup
# Works on: Linux, macOS, HPC clusters (with conda)
# Usage: bash scripts/setup_environment.sh
#
# What this does:
#   1. Checks Python version (3.10+ required)
#   2. Creates a virtual environment
#   3. Detects GPU and installs appropriate dependencies
#   4. Validates the installation
#   5. Prints next steps

set -e

echo ""
echo "============================================"
echo "  llm-forge Setup"
echo "============================================"
echo ""

# --- Detect environment ---
IS_HPC=false
if command -v sinfo &> /dev/null || command -v sbatch &> /dev/null; then
    IS_HPC=true
    echo "  Detected: HPC cluster (SLURM)"
elif [[ "$(uname)" == "Darwin" ]]; then
    echo "  Detected: macOS"
else
    echo "  Detected: Linux"
fi

# --- Check Python ---
echo ""
echo "[1/5] Checking Python..."
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v $cmd &> /dev/null; then
        PY_VER=$($cmd --version 2>&1 | awk '{print $2}')
        PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 10 ] && [ "$PY_MINOR" -le 12 ]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "  ERROR: Python 3.10-3.12 is required."
    echo "  Install: https://www.python.org/downloads/"
    if [ "$IS_HPC" = true ]; then
        echo "  Or try: module load python"
    fi
    exit 1
fi
echo "  Python: $($PYTHON_CMD --version)"

# --- Create environment ---
echo ""
echo "[2/5] Setting up environment..."

if [ "$IS_HPC" = true ] && command -v conda &> /dev/null; then
    # HPC: use conda
    ENV_NAME="llm-forge"
    if conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
        echo "  Conda env '$ENV_NAME' already exists, activating..."
    else
        echo "  Creating conda env '$ENV_NAME'..."
        conda create -n "$ENV_NAME" python=3.11 -y -q
    fi
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    echo "  Using conda env: $ENV_NAME"
else
    # Local: use venv
    if [ ! -d ".venv" ]; then
        echo "  Creating virtual environment..."
        $PYTHON_CMD -m venv .venv
    fi
    source .venv/bin/activate
    echo "  Using venv: .venv/"
fi

pip install --upgrade pip -q

# --- Detect GPU and install ---
echo ""
echo "[3/5] Detecting hardware and installing..."

if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU: $GPU_INFO"
    echo "  Installing with full GPU support..."
    pip install -e ".[all]" -q 2>&1 | tail -3
else
    echo "  No NVIDIA GPU detected (CPU-only mode)"
    echo "  Installing core dependencies..."
    pip install -e ".[dev,eval]" -q 2>&1 | tail -3
fi

# --- Validate installation ---
echo ""
echo "[4/5] Validating installation..."

$PYTHON_CMD -c "
import llm_forge
print(f'  llm-forge: {llm_forge.__version__}')
import torch
print(f'  PyTorch: {torch.__version__}')
cuda = 'CUDA ' + torch.version.cuda if torch.cuda.is_available() else 'CPU only'
print(f'  Device: {cuda}')
import transformers, peft, trl, accelerate, datasets
print('  Training deps: OK')
" 2>&1

# --- Validate demo config ---
echo ""
echo "[5/5] Checking demo config..."
if [ -f "configs/demo_lora_llama.yaml" ]; then
    $PYTHON_CMD -m llm_forge.cli validate configs/demo_lora_llama.yaml 2>&1 | grep -E "(valid|Error)" | head -1
else
    echo "  No demo config found (OK if installing from PyPI)"
fi

# --- Done ---
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
if [ "$IS_HPC" = true ]; then
    echo "  Activate: conda activate llm-forge"
else
    echo "  Activate: source .venv/bin/activate"
fi
echo ""
echo "  Quick start:"
echo "    llm-forge train --config configs/demo_lora_llama.yaml"
echo ""
echo "  Or create your own config:"
echo "    llm-forge init --template lora"
echo "    llm-forge validate config.yaml"
echo "    llm-forge train --config config.yaml"
echo ""
