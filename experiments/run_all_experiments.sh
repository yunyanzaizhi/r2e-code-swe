#!/bin/bash
# =============================================================================
# Master Experiment Runner for Sokoban History Management Comparison
# Runs all three strategies sequentially and collects results
# =============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/experiments/logging_utils.sh"

mkdir -p "${EXPERIMENT_LOG_ROOT:-experiments/logs}"
mkdir -p experiments/results

echo "============================================"
echo "Sokoban History Management Experiment Suite"
echo "============================================"
echo "Start time: $(date)"
echo ""

# Record GPU status before experiments
echo "=== GPU Status ===" | tee experiments/results/gpu_status.txt
nvidia-smi | tee -a experiments/results/gpu_status.txt
echo ""

# ============================================
# Experiment 1: Full History (Baseline)
# ============================================
echo ">>> [1/4] Running Full History Baseline..."
echo "Start: $(date)" | tee experiments/results/timing.txt
START_TIME=$(date +%s)

bash experiments/run_sokoban_full_history.sh

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Full History: ${ELAPSED}s" | tee -a experiments/results/timing.txt
echo "<<< Full History completed in ${ELAPSED}s"
echo ""

# ============================================
# Experiment 2: Recent Window K=3
# ============================================
echo ">>> [2/4] Running Recent Window K=3..."
START_TIME=$(date +%s)

bash experiments/run_sokoban_recent_window.sh vllm 3

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Recent Window K=3: ${ELAPSED}s" | tee -a experiments/results/timing.txt
echo "<<< Recent Window K=3 completed in ${ELAPSED}s"
echo ""

# ============================================
# Experiment 3: Recent Window K=5
# ============================================
echo ">>> [3/4] Running Recent Window K=5..."
START_TIME=$(date +%s)

bash experiments/run_sokoban_recent_window.sh vllm 5

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Recent Window K=5: ${ELAPSED}s" | tee -a experiments/results/timing.txt
echo "<<< Recent Window K=5 completed in ${ELAPSED}s"
echo ""

# ============================================
# Experiment 4: Structured Summary 
# ============================================
echo ">>> [4/4] Running Structured Summary..."
START_TIME=$(date +%s)

bash experiments/run_sokoban_structured_summary.sh

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Structured Summary: ${ELAPSED}s" | tee -a experiments/results/timing.txt
echo "<<< Structured Summary completed in ${ELAPSED}s"
echo ""

# ============================================
# Collect Results
# ============================================
echo "============================================"
echo "All experiments completed!"
echo "End time: $(date)"
echo "============================================"

# Record final GPU status
echo "=== Final GPU Status ===" | tee -a experiments/results/gpu_status.txt
nvidia-smi | tee -a experiments/results/gpu_status.txt

echo ""
echo "Logs saved in ${EXPERIMENT_LOG_ROOT:-experiments/logs}/<environment>/<gpu_bucket>/"
echo "Results saved in experiments/results/"
echo "Timing: $(cat experiments/results/timing.txt)"
