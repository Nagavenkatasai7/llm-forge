#!/bin/bash
# ============================================================================
# Persistent Job Monitor — Runs on Hopper via nohup
# ============================================================================
# Checks job status every 5 minutes and appends to a log file.
# Captures: job state, GPU utilization, training loss, stage progress.
#
# Usage:
#   nohup bash ~/llm-forge/scripts/monitor_job.sh 6231073 &
#   tail -f ~/llm-forge/outputs/finance-specialist-llama1b/monitor.log
# ============================================================================

JOB_ID="${1:?Usage: monitor_job.sh <JOB_ID>}"
OUTPUT_DIR="$HOME/llm-forge/outputs/finance-specialist-llama1b"
LOG_FILE="$OUTPUT_DIR/monitor.log"
CHECK_INTERVAL=300  # 5 minutes

mkdir -p "$OUTPUT_DIR"

echo "$(date '+%Y-%m-%d %H:%M:%S') [MONITOR] Starting monitor for job $JOB_ID" >> "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') [MONITOR] Check interval: ${CHECK_INTERVAL}s" >> "$LOG_FILE"

while true; do
    TS=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if job still exists in queue
    JOB_STATE=$(squeue -j "$JOB_ID" --noheader --format="%T" 2>/dev/null)

    if [ -z "$JOB_STATE" ]; then
        # Job no longer in queue — check sacct for final status
        FINAL_STATE=$(sacct -j "$JOB_ID" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
        echo "$TS [MONITOR] Job $JOB_ID no longer in queue. Final state: $FINAL_STATE" >> "$LOG_FILE"

        # Capture final metrics
        echo "$TS [MONITOR] === FINAL JOB ACCOUNTING ===" >> "$LOG_FILE"
        sacct -j "$JOB_ID" --format=JobID,JobName,Elapsed,State,ExitCode,MaxRSS,MaxVMSize,AllocTRES 2>/dev/null >> "$LOG_FILE"

        # Capture last 100 lines of job output
        JOB_OUT="$HOME/llm-forge/finance-specialist-${JOB_ID}.out"
        if [ -f "$JOB_OUT" ]; then
            echo "$TS [MONITOR] === LAST 100 LINES OF OUTPUT ===" >> "$LOG_FILE"
            tail -100 "$JOB_OUT" >> "$LOG_FILE"
        fi

        echo "$TS [MONITOR] Monitor exiting." >> "$LOG_FILE"
        break
    fi

    # Job still exists — record state
    JOB_INFO=$(squeue -j "$JOB_ID" --noheader --format="%T %M %N %r" 2>/dev/null)
    echo "$TS [STATUS] $JOB_INFO" >> "$LOG_FILE"

    if [ "$JOB_STATE" = "RUNNING" ]; then
        NODE=$(squeue -j "$JOB_ID" --noheader --format="%N" 2>/dev/null)

        # GPU utilization (if nvidia-smi is accessible)
        GPU_UTIL=$(ssh "$NODE" 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits' 2>/dev/null)
        if [ -n "$GPU_UTIL" ]; then
            echo "$TS [GPU] $GPU_UTIL" >> "$LOG_FILE"
        fi

        # Latest training loss from job output
        JOB_OUT="$HOME/llm-forge/finance-specialist-${JOB_ID}.out"
        if [ -f "$JOB_OUT" ]; then
            # Get latest loss line
            LAST_LOSS=$(grep "'loss'" "$JOB_OUT" 2>/dev/null | tail -1)
            if [ -n "$LAST_LOSS" ]; then
                echo "$TS [LOSS] $LAST_LOSS" >> "$LOG_FILE"
            fi

            # Get latest stage indicator
            LAST_STAGE=$(grep -E "\[Stage" "$JOB_OUT" 2>/dev/null | tail -1)
            if [ -n "$LAST_STAGE" ]; then
                echo "$TS [STAGE] $LAST_STAGE" >> "$LOG_FILE"
            fi

            # Count errors and warnings
            ERR_COUNT=$(grep -ci "error" "$JOB_OUT" 2>/dev/null || echo 0)
            WARN_COUNT=$(grep -ci "warning" "$JOB_OUT" 2>/dev/null || echo 0)
            LINES=$(wc -l < "$JOB_OUT" 2>/dev/null || echo 0)
            echo "$TS [LOG] lines=$LINES errors=$ERR_COUNT warnings=$WARN_COUNT" >> "$LOG_FILE"
        fi
    fi

    sleep "$CHECK_INTERVAL"
done
