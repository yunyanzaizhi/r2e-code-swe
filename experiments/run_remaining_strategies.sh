#!/bin/bash
# =============================================================================
# 等待 Full History 完成后，依次运行 Recent Window 和 Structured Summary
# =============================================================================
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/logging_utils.sh"

FULL_HISTORY_PID=$1
if [ -z "$FULL_HISTORY_PID" ]; then
    echo "Usage: $0 <full_history_training_pid>"
    exit 1
fi

echo "[$(date)] Waiting for Full History training (PID $FULL_HISTORY_PID) to finish..."

# 等待 Full History 完成
while kill -0 "$FULL_HISTORY_PID" 2>/dev/null; do
    sleep 60
done

echo "[$(date)] Full History training finished!"
sleep 10

# 清理 Ray
ray stop --force 2>/dev/null
sleep 5

# =============================================
# 1. Recent Window K=3
# =============================================
echo "[$(date)] Starting Recent Window K=3..."
WRAPPER_LOG_FILE="$(make_log_path sokoban "${GPU_COUNT:-2}" recent_window_k3_wrapper)"
announce_log_path "$WRAPPER_LOG_FILE"
bash experiments/run_sokoban_recent_window.sh vllm 3 2>&1 | tee "$WRAPPER_LOG_FILE"

echo "[$(date)] Recent Window K=3 finished!"
sleep 10
ray stop --force 2>/dev/null
sleep 5

# =============================================
# 2. Structured Summary
# =============================================
echo "[$(date)] Starting Structured Summary..."
WRAPPER_LOG_FILE="$(make_log_path sokoban "${GPU_COUNT:-2}" structured_summary_wrapper)"
announce_log_path "$WRAPPER_LOG_FILE"
bash experiments/run_sokoban_structured_summary.sh vllm 2>&1 | tee "$WRAPPER_LOG_FILE"

echo "[$(date)] Structured Summary finished!"
echo "[$(date)] All experiments complete!"
