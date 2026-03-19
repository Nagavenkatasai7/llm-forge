#!/bin/bash
# ============================================================================
# Deploy Finance Specialist v7 to Hopper HPC
# ============================================================================
# Usage: bash scripts/deploy_v7.sh
#
# This script:
#   1. Syncs code to Hopper (via ControlMaster SSH socket)
#   2. Syncs benchmark scripts
#   3. Submits the training SBATCH job
#   4. Shows you how to monitor
#
# Prerequisites:
#   - SSH ControlMaster to hopper must be active
#     Run: ssh hopper   (login with password + Duo 2FA)
#     Then open a new terminal and run this script.
# ============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Project: $PROJECT_DIR"
echo ""

# Step 1: Check SSH connection
echo "[1/4] Checking SSH connection to Hopper..."
if ! ssh -o ConnectTimeout=5 hopper "echo 'Connected to $(hostname)'" 2>/dev/null; then
    echo "ERROR: Cannot connect to Hopper."
    echo "Open a terminal and run: ssh hopper"
    echo "Complete password + Duo 2FA, then try again."
    exit 1
fi
echo ""

# Step 2: Sync code
echo "[2/4] Syncing code to Hopper..."
rsync -avz --delete \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='outputs/' \
    --exclude='eval_results/' \
    --exclude='.git' \
    --exclude='*.egg-info' \
    --exclude='.DS_Store' \
    "$PROJECT_DIR/" hopper:~/llm-forge/
echo ""

# Step 3: Sync benchmark scripts
echo "[3/4] Syncing benchmark scripts..."
rsync -avz \
    "$PROJECT_DIR/hopper_benchmark/" hopper:~/hopper_benchmark/
echo ""

# Step 4: Submit training job
echo "[4/4] Submitting training job..."
JOB_ID=$(ssh hopper "cd ~/llm-forge && mkdir -p logs && sbatch scripts/run_finance_specialist_v7.sbatch" | awk '{print $4}')
echo "Training job submitted: $JOB_ID"
echo ""

echo "=============================================="
echo "  DEPLOYMENT COMPLETE"
echo "=============================================="
echo ""
echo "  Monitor training:"
echo "    ssh hopper 'squeue -u nchennu'"
echo "    ssh hopper 'tail -f ~/llm-forge/finance-v7-${JOB_ID}.out'"
echo ""
echo "  After training completes:"
echo "    ssh hopper 'cd ~/hopper_benchmark && bash submit_all_v7.sh'"
echo ""
echo "  After benchmarks complete:"
echo "    ssh hopper 'source ~/miniforge/bin/activate llm-forge && python ~/hopper_benchmark/collect_results_v7.py'"
echo ""
echo "  Sync results back to Mac:"
echo "    rsync -avz hopper:/scratch/nchennu/eval_results/v7_* $PROJECT_DIR/eval_results/"
echo "=============================================="
