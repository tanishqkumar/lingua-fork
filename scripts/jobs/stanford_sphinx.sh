#!/bin/bash
#SBATCH --job-name=lingua
#SBATCH --output=/juice5b/scr5b/tanishq/lingua-out/logs/%x_%j.log
#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00

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

# Navigate to repo
cd ~/lingua-fork

# Get number of GPUs
NGPUS=${SLURM_GPUS_ON_NODE:-8}

# Default config if not provided
CONFIG=${CONFIG:-apps/main/configs/debug.yaml}
EXPERIMENT=${EXPERIMENT:-default}

echo "=== Training Config ==="
echo "Config: $CONFIG"
echo "Experiment: $EXPERIMENT"
echo "GPUs: $NGPUS"

# Run training with uv
uv run torchrun \
    --nproc_per_node=$NGPUS \
    --standalone \
    -m apps.main.train \
    config=$CONFIG \
    logging.wandb.project=lingua-fork \
    logging.wandb.tags="[stanford,sphinx,A100,$EXPERIMENT]" \
    logging.wandb.group="$EXPERIMENT/stanford/sphinx" \
    $EXTRA_ARGS

echo "=== Done ==="
echo "End: $(date)"
