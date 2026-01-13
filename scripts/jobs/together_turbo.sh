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

# Build run name: experiment_cluster_jobid (e.g., lr_sweep_mk-turbo_12345)
RUN_NAME="${EXPERIMENT}_mk-turbo_${SLURM_JOB_ID}"

echo "=== Training Config ==="
echo "Config: $CONFIG"
echo "Experiment: $EXPERIMENT"
echo "Run Name: $RUN_NAME"
echo "GPUs: $NGPUS"
echo "Extra Args: $EXTRA_ARGS"

# Run training with uv run
uv run torchrun \
    --nproc_per_node=$NGPUS \
    --standalone \
    -m apps.main.train \
    config=$CONFIG \
    name=$RUN_NAME \
    logging.wandb.project=tanishqbot \
    logging.wandb.name=$RUN_NAME \
    logging.wandb.tags="[together,mk-turbo,H100,$EXPERIMENT,job_$SLURM_JOB_ID]" \
    logging.wandb.group="$EXPERIMENT/mk-turbo" \
    $EXTRA_ARGS

echo "=== Done ==="
echo "End: $(date)"
