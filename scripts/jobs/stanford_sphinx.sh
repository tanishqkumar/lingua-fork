#!/bin/bash
#SBATCH --job-name=lingua
#SBATCH --output=/juice5b/scr5b/tanishq/lingua-out/logs/%x_%j.log
#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4:00:00

# Stanford sphinx cluster job script (A100s)
# Max: 2 nodes, 8x A100 per node

set -e

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start: $(date)"

# Set cluster environment
export LINGUA_CLUSTER=sphinx

# Ensure ~/.local/bin is in PATH (for uv)
export PATH="$HOME/.local/bin:$PATH"

# WandB API key for logging
export WANDB_API_KEY="cf165ee00e06cddeca6a6b9080ec27ca55962aad"

# Create output directories
mkdir -p /juice5b/scr5b/tanishq/lingua-out/logs

# Navigate to repo
cd ~/lingua-fork

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || true
    sleep 2
fi

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found"
    exit 1
fi

# Sync dependencies
uv sync

# Get number of GPUs
NGPUS=${SLURM_GPUS_ON_NODE:-8}

# Default config if not provided
CONFIG=${CONFIG:-apps/main/configs/debug.yaml}
EXPERIMENT=${EXPERIMENT:-default}

# Build run name: experiment_cluster_jobid (e.g., lr_sweep_sphinx_12345)
RUN_NAME="${EXPERIMENT}_sphinx_${SLURM_JOB_ID}"

echo "=== Training Config ==="
echo "Config: $CONFIG"
echo "Experiment: $EXPERIMENT"
echo "Run Name: $RUN_NAME"
echo "GPUs: $NGPUS"
echo "Extra Args: $EXTRA_ARGS"

# Run training with uv (checkpointing disabled to save disk space)
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
    logging.wandb.tags="[stanford,sphinx,A100,$EXPERIMENT,job_$SLURM_JOB_ID]" \
    logging.wandb.group="$EXPERIMENT/sphinx" \
    $EXTRA_ARGS

echo "=== Done ==="
echo "End: $(date)"
