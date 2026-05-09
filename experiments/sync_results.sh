#!/usr/bin/env bash
# =============================================================================
# Sync experiment results from remote server to local
# Usage: bash experiments/sync_results.sh
# =============================================================================
set -e
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-lab-server-1:/home/caiting/verl-agent-exp-copy-from-lab-server-20260505}"

echo "Syncing results from remote server..."

# Sync logs
mkdir -p "$LOCAL_DIR/experiments/logs"
rsync -avz "$REMOTE_DIR/experiments/logs/" "$LOCAL_DIR/experiments/logs/"

# Sync results
mkdir -p "$LOCAL_DIR/experiments/results"
rsync -avz "$REMOTE_DIR/experiments/results/" "$LOCAL_DIR/experiments/results/" 2>/dev/null || true

# Sync checkpoints (metrics only, not model weights)
rsync -avz --include='*/' --include='*.json' --include='*.csv' --include='*.txt' --include='*.log' --exclude='*.pt' --exclude='*.bin' --exclude='*.safetensors' \
  "$REMOTE_DIR/checkpoints/" "$LOCAL_DIR/experiments/checkpoints/" 2>/dev/null || true

echo "Sync complete!"
echo "Logs: $LOCAL_DIR/experiments/logs/"
echo "Results: $LOCAL_DIR/experiments/results/"
