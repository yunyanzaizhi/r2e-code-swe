#!/usr/bin/env bash
# =============================================================================
# Setup script for verl-agent experiment environment on lab-server
# Run this once to prepare the experiment environment
# =============================================================================
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="${PROJ_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Create virtual environment if not exists
if [ ! -d "$PROJ_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    cd $PROJ_DIR
    python3.10 -m venv .venv || python3 -m venv .venv
fi

source $PROJ_DIR/.venv/bin/activate

# Install the project and dependencies
cd $PROJ_DIR
pip install -e . 2>&1 | tail -5

# Install Sokoban dependencies
pip install gym==0.26.2 gym-sokoban matplotlib 2>&1 | tail -3

# Create logs directory
mkdir -p experiments/logs
mkdir -p experiments/results

# Verify installation
python3 -c "
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())
import verl
print('verl imported OK')
"

echo "Setup complete!"
