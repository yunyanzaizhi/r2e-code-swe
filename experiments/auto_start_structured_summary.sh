#!/usr/bin/env bash
# =============================================================================
# Auto-start Structured Summary after Recent Window finishes
# Usage: nohup bash experiments/auto_start_structured_summary.sh &
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

source experiments/logging_utils.sh

GPU_COUNT="${GPU_COUNT:-2}"
GPU_BUCKET="$(log_gpu_bucket "$GPU_COUNT")"
LOG_ROOT="${EXPERIMENT_LOG_ROOT:-experiments/logs}"
SS_SCRIPT="${PROJECT_ROOT}/experiments/run_sokoban_structured_summary.sh"
WATCHDOG_LOG="$(make_log_path sokoban "$GPU_COUNT" structured_summary_watchdog)"

latest_log_for() {
    local pattern="$1"
    local dir="${LOG_ROOT}/sokoban/${GPU_BUCKET}"
    if [[ ! -d "$dir" ]]; then
        return 0
    fi
    find "$dir" -maxdepth 1 -type f -name "$pattern" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

RW_LOG="${RECENT_WINDOW_LOG:-$(latest_log_for 'recent_window_k3*.log')}"

announce_log_path "$WATCHDOG_LOG"
exec > >(tee -a "$WATCHDOG_LOG") 2>&1

echo "[$(date)] Watchdog started. Monitoring Recent Window training..."
echo "[$(date)] Project root: ${PROJECT_ROOT}"
echo "[$(date)] GPU bucket: ${GPU_BUCKET}"
echo "[$(date)] Recent Window log: ${RW_LOG:-not found yet}"
echo "[$(date)] Waiting for Recent Window to reach 50/50 or process to exit..."

while true; do
    # Check if training process is still running
    RW_PID=$(pgrep -f "experiment_name=recent_window" 2>/dev/null | head -1)

    if [ -z "$RW_PID" ]; then
        # Process not found — check if it completed successfully
        if [[ -n "$RW_LOG" ]] && { grep -q "100%.*50/50" "$RW_LOG" 2>/dev/null || grep -q "Training Progress: 100%" "$RW_LOG" 2>/dev/null; }; then
            echo "[$(date)] Recent Window completed successfully (50/50)!"
        else
            echo "[$(date)] Recent Window process ended (may have finished or crashed)."
            echo "[$(date)] Last progress line:"
            if [[ -n "$RW_LOG" ]]; then
                grep "Training Progress" "$RW_LOG" 2>/dev/null | tail -1
            fi
        fi
        break
    fi

    # Also check if log indicates completion
    if [[ -n "$RW_LOG" ]] && grep -q "100%.*50/50" "$RW_LOG" 2>/dev/null; then
        echo "[$(date)] Recent Window reached 50/50 in log!"
        break
    fi

    # Print progress every check
    if [[ -z "$RW_LOG" ]]; then
        RW_LOG="$(latest_log_for 'recent_window_k3*.log')"
    fi
    PROGRESS=""
    if [[ -n "$RW_LOG" ]]; then
        PROGRESS=$(grep "Training Progress" "$RW_LOG" 2>/dev/null | tail -1 || true)
    fi
    echo "[$(date)] Still running (PID=$RW_PID). $PROGRESS"

    sleep 120  # Check every 2 minutes
done

echo ""
echo "[$(date)] ============================================"
echo "[$(date)] Starting Structured Summary training..."
echo "[$(date)] ============================================"

# Wait a moment for GPU memory to be fully released
sleep 30

# Kill any remaining Ray processes from previous run
ray stop --force 2>/dev/null
sleep 10

# Start Structured Summary
source .venv/bin/activate
nohup bash "$SS_SCRIPT" vllm >/dev/null 2>&1 &
SS_PID=$!
echo "[$(date)] Structured Summary started with PID=$SS_PID"
echo "[$(date)] Structured Summary script writes its own log under ${LOG_ROOT}/sokoban/${GPU_BUCKET}/"

# Wait a bit and verify it's running
sleep 60
if kill -0 $SS_PID 2>/dev/null; then
    echo "[$(date)] Structured Summary is running successfully."
else
    echo "[$(date)] WARNING: Structured Summary may have failed to start. Check logs."
fi

echo "[$(date)] Watchdog done."
