#!/bin/bash
# Launch all 4 scaling experiment runs
# Usage: ./scripts/launch_scaling.sh [cluster]
# Cluster: research-secure (default), mk-turbo, sphinx, miso

CLUSTER=${1:-research-secure}

echo "=== Launching Scaling Experiment ==="
echo "Cluster: $CLUSTER"
echo "Models: 30M, 60M, 120M, 240M non-embedding params"
echo "Training: 2B tokens, lr=1e-3, batch=256k tokens"
echo ""

# Determine job script based on cluster
case $CLUSTER in
    research-secure)
        JOB_SCRIPT="scripts/jobs/together_secure.sh"
        ;;
    mk-turbo)
        JOB_SCRIPT="scripts/jobs/together_turbo.sh"
        ;;
    sphinx)
        JOB_SCRIPT="scripts/jobs/stanford_sphinx.sh"
        ;;
    miso)
        JOB_SCRIPT="scripts/jobs/stanford_miso.sh"
        ;;
    *)
        echo "Unknown cluster: $CLUSTER"
        echo "Available: research-secure, mk-turbo, sphinx, miso"
        exit 1
        ;;
esac

# Launch all 4 model sizes
SIZES="30m 60m 120m 240m"

for SIZE in $SIZES; do
    echo "Submitting scale_${SIZE}..."
    EXPERIMENT="scaling_${SIZE}" \
    CONFIG="apps/main/configs/scaling/scale_${SIZE}.yaml" \
    sbatch $JOB_SCRIPT
    sleep 1  # Small delay between submissions
done

echo ""
echo "=== All jobs submitted ==="
echo "Monitor with: squeue -u \$USER"
echo "WandB: https://wandb.ai/tk07-stanford-university/lingua-fork"
