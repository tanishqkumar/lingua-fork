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
# Max: 1 node, 8x H200 per node
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

# Build run name: experiment_cluster_jobid (e.g., lr_sweep_miso_12345)
RUN_NAME="${EXPERIMENT}_miso_${SLURM_JOB_ID}"

echo "=== Training Config ==="
echo "Config: $CONFIG"
echo "Experiment: $EXPERIMENT"
echo "Run Name: $RUN_NAME"
echo "GPUs: $NGPUS"
echo "Extra Args: $EXTRA_ARGS"

# Run training with uv
uv run torchrun \
    --nproc_per_node=$NGPUS \
    --standalone \
    -m apps.main.train \
    config=$CONFIG \
    name=$RUN_NAME \
    logging.wandb.project=tanishqbot \
    logging.wandb.name=$RUN_NAME \
    logging.wandb.tags="[stanford,miso,H200,$EXPERIMENT,job_$SLURM_JOB_ID]" \
    logging.wandb.group="$EXPERIMENT/miso" \
    $EXTRA_ARGS

echo "=== Done ==="
echo "End: $(date)"
