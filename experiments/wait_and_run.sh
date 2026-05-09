#!/bin/bash
# =============================================================================
# Wait for GPU to be free, then run Sokoban experiments
# Usage: nohup bash experiments/wait_and_run.sh &
# =============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/logging_utils.sh"

echo "[$(date)] Waiting for GPU to be free..."

while true; do
    # Check if any python/ray processes are using GPUs
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "$GPU_PROCS" -eq "0" ]; then
        echo "[$(date)] GPU is free! Starting experiments..."
        break
    fi
    echo "[$(date)] GPU still in use ($GPU_PROCS processes). Waiting 60s..."
    sleep 60
done

# Small delay to ensure cleanup
sleep 10

# Run all experiments
LOG_FILE="$(make_log_path sokoban "${GPU_COUNT:-2}" master_run)"
announce_log_path "$LOG_FILE"
bash experiments/run_all_experiments.sh 2>&1 | tee "$LOG_FILE"

echo "[$(date)] All experiments completed!"
