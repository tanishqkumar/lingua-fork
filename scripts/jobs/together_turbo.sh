#!/bin/bash
#SBATCH --job-name=ssd
#SBATCH --output=/data/tkumar/lingua-out/logs/%x_%j.log
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

# Navigate to repo
cd ~/lingua-fork

# Use uv for reproducible environments
uv sync

# Get number of GPUs
NGPUS=${SLURM_GPUS_ON_NODE:-8}

# Default config if not provided
CONFIG=${CONFIG:-apps/main/configs/debug.yaml}
EXPERIMENT=${EXPERIMENT:-default}

echo "=== Training Config ==="
echo "Config: $CONFIG"
echo "Experiment: $EXPERIMENT"
echo "GPUs: $NGPUS"

# Run training with uv run
uv run torchrun \
    --nproc_per_node=$NGPUS \
    --standalone \
    -m apps.main.train \
    config=$CONFIG \
    logging.wandb.project=lingua-fork \
    logging.wandb.tags="[together,mk-turbo,H100,$EXPERIMENT]" \
    logging.wandb.group="$EXPERIMENT/together/mk-turbo" \
    $EXTRA_ARGS

echo "=== Done ==="
echo "End: $(date)"
