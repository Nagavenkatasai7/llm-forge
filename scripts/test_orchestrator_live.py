#!/usr/bin/env python3
"""Live end-to-end test of the multi-agent orchestrator.

Runs a complete workflow:
  1. Data Agent scans training data
  2. Config Agent generates/validates config
  3. Training Agent launches training
  4. Training Agent monitors progress

Uses SmolLM2-135M (tiny) + 21 sample dataset = ~2-5 min on M4 Pro.
"""

import os
import sys
import time

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_forge.chat.orchestrator import OrchestratorEngine

GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

if not GOOGLE_KEY or not ANTHROPIC_KEY:
    print("Set ANTHROPIC_API_KEY and GOOGLE_API_KEY first")
    sys.exit(1)


def print_step(n, title):
    print(f"\n{'='*60}")
    print(f"  STEP {n}: {title}")
    print(f"{'='*60}\n")


def main():
    print("Creating OrchestratorEngine (Claude Opus 4.6 + Gemini agents)...")
    engine = OrchestratorEngine(
        project_dir=".",
        gemini_api_key=GOOGLE_KEY,
    )
    print(f"Ready. Model: {engine.model_key}\n")

    # Step 1: Scan the training data
    print_step(1, "SCAN TRAINING DATA")
    response = engine.send(
        "Scan the training data at ./examples/data/sample_train.jsonl "
        "and tell me the format, sample count, and show me 2 examples."
    )
    print(response[:1500])

    # Step 2: Detect hardware
    print_step(2, "DETECT HARDWARE")
    response = engine.send("Detect my hardware — GPU, RAM, CPU.")
    print(response[:1000])

    # Step 3: Start training with existing config
    print_step(3, "START TRAINING")
    response = engine.send(
        "Start training using the config at configs/quickstart_tiny.yaml. "
        "This uses SmolLM2-135M with LoRA on the sample data."
    )
    print(response[:1500])

    # Step 4: Monitor training (poll every 15 seconds)
    print_step(4, "MONITOR TRAINING")
    for i in range(6):
        time.sleep(15)
        print(f"\n--- Check {i+1}/6 (t={15*(i+1)}s) ---")
        response = engine.send("Check training status. Show current loss and progress.")
        print(response[:800])
        if "complete" in response.lower() or "finished" in response.lower():
            print("\nTraining completed!")
            break

    # Step 5: Final status
    print_step(5, "FINAL STATUS")
    response = engine.send(
        "Training should be done. Read the training logs from ./outputs/quickstart-tiny/ "
        "and give me the final loss and training summary."
    )
    print(response[:1500])

    print(f"\n{'='*60}")
    print("  END-TO-END TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
