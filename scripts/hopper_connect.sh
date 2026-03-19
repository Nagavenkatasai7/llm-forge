#!/bin/bash
# Establish SSH master connection to GMU Hopper
# Usage: bash scripts/hopper_connect.sh

REMOTE="hopper"

if ssh -O check "$REMOTE" 2>/dev/null; then
    echo "Already connected to Hopper."
    echo "  ssh hopper              # interactive shell"
    echo "  bash scripts/deploy_to_hopper.sh  # deploy"
else
    echo "Connecting to Hopper (password + Duo)..."
    expect "$(dirname "$0")/hopper_ssh.exp"
    if ssh -O check "$REMOTE" 2>/dev/null; then
        echo "Connected! Session persists for 4 hours."
    else
        echo "Connection failed. Check password or Duo approval."
        exit 1
    fi
fi
