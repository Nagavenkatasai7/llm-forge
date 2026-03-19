#!/bin/bash
# llm-forge — Deploy to GMU Hopper
# Requires active SSH master connection (run hopper_connect.sh first)
# Usage: bash scripts/deploy_to_hopper.sh

set -e

REMOTE="hopper"
REMOTE_USER="Spotluru"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/home/${REMOTE_USER}/llm-forge"
CONFIG="configs/demo_lora_llama.yaml"
SBATCH="scripts/slurm/train_demo.sbatch"

echo "============================================"
echo "  llm-forge → Hopper Deployment"
echo "============================================"
echo ""

# Check master connection
echo "[1/5] Checking SSH connection..."
if ! ssh -O check "$REMOTE" 2>/dev/null; then
    echo "  ERROR: No SSH master connection. Run hopper_connect.sh first."
    exit 1
fi
echo "  Connected"
echo ""

# Sync code
echo "[2/5] Syncing code..."
rsync -avz --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.eggs' \
    --exclude '*.egg-info' \
    --exclude '.mypy_cache' \
    --exclude '.pytest_cache' \
    --exclude '.ruff_cache' \
    --exclude 'outputs/' \
    --exclude 'wandb/' \
    --exclude '.venv' \
    "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR}/"
echo "  Synced to ${REMOTE}:${REMOTE_DIR}"
echo ""

# Install dependencies
echo "[3/5] Installing dependencies..."
ssh "$REMOTE" bash -l <<'EOF'
set -e
source "$HOME/miniforge/bin/activate" llm-forge
cd "$HOME/llm-forge"
pip install --upgrade pip -q
pip install -e ".[all]" -q
echo "  Done"
EOF
echo ""

# Setup scratch + validate
echo "[4/5] Setting up scratch and validating config..."
ssh "$REMOTE" bash -l <<EOF
set -e
source "\$HOME/miniforge/bin/activate" llm-forge
cd "\$HOME/llm-forge"
mkdir -p /scratch/\$USER/logs /scratch/\$USER/hf_cache
export HF_HOME="/scratch/\$USER/hf_cache"
python -m llm_forge.cli validate ${CONFIG}
echo "  Config valid"
EOF
echo ""

# Submit SLURM job
echo "[5/5] Submitting SLURM job..."
JOB_ID=$(ssh "$REMOTE" bash -l <<EOF
source "\$HOME/miniforge/bin/activate" llm-forge
cd "\$HOME/llm-forge"
sbatch --parsable ${SBATCH}
EOF
)

echo ""
echo "============================================"
echo "  Deployed! Job ID: ${JOB_ID}"
echo "============================================"
echo "  Monitor:  ssh hopper tail -f /scratch/${REMOTE_USER}/llm-forge-demo_${JOB_ID}.out"
echo "  Status:   ssh hopper squeue -u ${REMOTE_USER}"
echo "  Cancel:   ssh hopper scancel ${JOB_ID}"
echo ""
