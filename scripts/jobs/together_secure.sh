#!/bin/bash
#SBATCH --job-name=ssd
#SBATCH --output=/data/tkumar/lingua-out/logs/%x_%j.log
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00

# Together AI research-secure cluster job script
# Max: 4 nodes, 8x H100 per node
# Package manager: uv (for reproducibility)

set -e

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start: $(date)"

# Set cluster environment
export LINGUA_CLUSTER=research-secure

# Ensure ~/.local/bin is in PATH (for uv)
export PATH="$HOME/.local/bin:$PATH"

# WandB API key for logging
export WANDB_API_KEY="cf165ee00e06cddeca6a6b9080ec27ca55962aad"

# Create output directories (only accessible from compute nodes)
mkdir -p /data/tkumar/lingua-out/logs
mkdir -p /data/tkumar/datasets

# Put .venv on /data to avoid filling up home partition
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

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Sync dependencies
uv sync

# Get number of GPUs
NGPUS=${SLURM_GPUS_ON_NODE:-8}

# Default config if not provided
CONFIG=${CONFIG:-apps/main/configs/debug.yaml}
EXPERIMENT=${EXPERIMENT:-default}

# Build run name: experiment_cluster_jobid (e.g., lr_sweep_research-secure_12345)
RUN_NAME="${EXPERIMENT}_research-secure_${SLURM_JOB_ID}"

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
    logging.wandb.tags="[together,research-secure,H100,$EXPERIMENT,job_$SLURM_JOB_ID]" \
    logging.wandb.group="$EXPERIMENT/research-secure" \
    $EXTRA_ARGS

echo "=== Done ==="
echo "End: $(date)"
