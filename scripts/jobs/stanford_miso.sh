#!/bin/bash
#SBATCH --job-name=lingua
#SBATCH --output=/juice5b/scr5b/tanishq/lingua-out/logs/%x_%j.log
#SBATCH --account=nlp
#SBATCH --partition=miso
#SBATCH --nodes=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00

# Stanford miso cluster job script (H200s)
# Max: 2 nodes, 8x H200 per node
# NOTE: miso requires using all 8 GPUs per node

set -e

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: 8 (miso requires all 8)"
echo "Start: $(date)"

# Set cluster environment
export LINGUA_CLUSTER=miso

# Navigate to repo
cd ~/lingua-fork

# miso always uses 8 GPUs
NGPUS=8

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
    logging.wandb.tags="[stanford,miso,H200,$EXPERIMENT]" \
    logging.wandb.group="$EXPERIMENT/stanford/miso" \
    $EXTRA_ARGS

echo "=== Done ==="
echo "End: $(date)"
