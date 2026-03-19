#!/bin/bash
# llm-forge Demo Setup for GMU Hopper Cluster
# One-shot script: install miniforge, create env, clone repo, install deps, submit job
# Usage: bash setup_hopper_demo.sh
#
# Prerequisites:
#   - SSH into Hopper: ssh NetID@hopper.orc.gmu.edu
#   - Off-campus requires GMU VPN or Duo MFA
#   - Accept Llama license: https://huggingface.co/meta-llama/Llama-3.2-1B

set -e

echo "============================================"
echo "  llm-forge Demo Setup — GMU Hopper Cluster"
echo "============================================"
echo ""

# --- Configuration ---
REPO_URL="https://github.com/Nagavenkatasai7/llm-forge.git"
INSTALL_DIR="$HOME/llm-forge"
CONDA_ENV="llm-forge"
PYTHON_VERSION="3.11"

# --- Step 1: Install Miniforge (if not present) ---
echo "[1/7] Setting up Miniforge..."
if [ ! -d "$HOME/miniforge" ]; then
    echo "  Downloading Miniforge..."
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    echo "  Installing Miniforge to $HOME/miniforge..."
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge"
    rm /tmp/miniforge.sh
    echo "  Miniforge installed"
else
    echo "  Miniforge already installed"
fi
source "$HOME/miniforge/bin/activate"
echo "  Conda: $(conda --version)"

# --- Step 2: Clone or update repo ---
echo ""
echo "[2/7] Setting up repository..."
if [ -d "$INSTALL_DIR" ]; then
    echo "  Directory exists, pulling latest..."
    cd "$INSTALL_DIR"
    git pull origin main
else
    echo "  Cloning repository..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# --- Step 3: Create conda environment ---
echo ""
echo "[3/7] Setting up conda environment..."
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "  Environment '${CONDA_ENV}' already exists, activating..."
else
    echo "  Creating environment '${CONDA_ENV}' with Python ${PYTHON_VERSION}..."
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
fi
source "$HOME/miniforge/bin/activate" "$CONDA_ENV"
echo "  Python: $(python --version)"

# --- Step 4: Install llm-forge ---
echo ""
echo "[4/7] Installing llm-forge with all dependencies..."
pip install --upgrade pip
pip install -e ".[all]"
echo "  llm-forge installed successfully"

# --- Step 5: Verify GPU access ---
echo ""
echo "[5/7] Verifying GPU access..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  NOTE: No GPU on login node (expected — GPUs are on compute nodes)')
"

# --- Step 6: Check HuggingFace token ---
echo ""
echo "[6/7] Checking HuggingFace authentication..."
if [ -z "$HF_TOKEN" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
    echo "  WARNING: No HF token found."
    echo "  Llama models require accepting the license at:"
    echo "    https://huggingface.co/meta-llama/Llama-3.2-1B"
    echo "  Then run: huggingface-cli login"
    echo ""
    read -p "  Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "  Setup paused. Run 'huggingface-cli login' and re-run this script."
        exit 1
    fi
else
    echo "  HuggingFace token found"
fi

# --- Step 7: Submit training job ---
echo ""
echo "[7/7] Submitting SLURM job..."
mkdir -p /scratch/$USER/logs
cd "$INSTALL_DIR"

# Validate config before submitting
echo "  Validating config..."
python -m llm_forge.cli validate configs/demo_lora_llama.yaml

echo "  Submitting job..."
JOB_ID=$(sbatch --parsable scripts/slurm/train_demo.sbatch)
echo ""
echo "============================================"
echo "  Job submitted successfully!"
echo "  Job ID: $JOB_ID"
echo "  Monitor: tail -f /scratch/$USER/llm-forge-demo_${JOB_ID}.out"
echo "  Status:  squeue -j $JOB_ID"
echo "  Cancel:  scancel $JOB_ID"
echo "============================================"
