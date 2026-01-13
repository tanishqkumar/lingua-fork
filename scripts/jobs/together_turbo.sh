#!/bin/bash
#SBATCH --job-name=ssd
#SBATCH --output=%x_%j.log
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00

# Together AI mk-turbo cluster job script
# Max: 1 node, 8x H100 per node
# Package manager: uv (for reproducibility)

set -e

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start: $(date)"

# Set cluster environment
export LINGUA_CLUSTER=mk-turbo

# Ensure ~/.local/bin is in PATH (for uv)
export PATH="$HOME/.local/bin:$PATH"

# WandB API key for logging
export WANDB_API_KEY="cf165ee00e06cddeca6a6b9080ec27ca55962aad"

# Create output directories (only accessible from compute nodes)
mkdir -p /data/tkumar/lingua-out/logs
mkdir -p /data/tkumar/datasets

# IMPORTANT: Put .venv on /data to avoid filling up small root partition
# /data has much more space than /home on mk-turbo
VENV_DIR="/data/tkumar/lingua-venv"
mkdir -p "$VENV_DIR"

# Navigate to repo
cd ~/lingua-fork

# Symlink .venv to /data if not already done
if [ ! -L .venv ] || [ "$(readlink .venv)" != "$VENV_DIR" ]; then
    rm -rf .venv
    ln -sf "$VENV_DIR" .venv
    echo "Symlinked .venv -> $VENV_DIR"
fi

# Install uv if not available (ignore errors from race conditions)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || true
    # Wait a moment for any concurrent installs to complete
    sleep 2
fi

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found after installation attempt"
    exit 1
fi

# Sync dependencies (will use symlinked .venv on /data)
uv sync

# Get number of GPUs
NGPUS=${SLURM_GPUS_ON_NODE:-8}

# Default config if not provided
CONFIG=${CONFIG:-apps/main/configs/debug.yaml}
EXPERIMENT=${EXPERIMENT:-default}

# Build run name: experiment_cluster_jobid (e.g., lr_sweep_mk-turbo_12345)
RUN_NAME="${EXPERIMENT}_mk-turbo_${SLURM_JOB_ID}"

echo "=== Training Config ==="
echo "Config: $CONFIG"
echo "Experiment: $EXPERIMENT"
echo "Run Name: $RUN_NAME"
echo "GPUs: $NGPUS"
echo "Extra Args: $EXTRA_ARGS"

# Run training with uv run (checkpointing disabled to save disk space)
uv run torchrun \
    --nproc_per_node=$NGPUS \
    --standalone \
    -m apps.main.train \
    config=$CONFIG \
    name=$RUN_NAME \
    checkpoint.dump.every=-1 \
    checkpoint.eval.every=-1 \
    logging.wandb.project=tanishqbot \
    logging.wandb.name=$RUN_NAME \
    logging.wandb.tags="[together,mk-turbo,H100,$EXPERIMENT,job_$SLURM_JOB_ID]" \
    logging.wandb.group="$EXPERIMENT/mk-turbo" \
    $EXTRA_ARGS

echo "=== Done ==="
echo "End: $(date)"
