#!/bin/bash
# Launch all 4 scaling experiment runs
# Usage: ./scripts/launch_scaling.sh [cluster]
# Cluster: research-secure (default), mk-turbo, sphinx, miso
#
# Models (6 layers fixed depth, width varies):
#   - 30M:  dim=640,  n_layers=6, n_heads=10 (30.5M non-embed params)
#   - 60M:  dim=896,  n_layers=6, n_heads=14 (60.6M non-embed params)
#   - 120M: dim=1280, n_layers=6, n_heads=20 (121.9M non-embed params)
#   - 240M: dim=1792, n_layers=6, n_heads=28 (234.0M non-embed params)
#
# Training: 2B tokens, lr=1e-3, batch=256k tokens (seq_len=1024), 7629 steps

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$SCRIPT_DIR/.."
cd "$ROOT_DIR"

CLUSTER=${1:-research-secure}

echo "=== Launching Scaling Experiment ==="
echo "Cluster: $CLUSTER"
echo "Models: 30M, 60M, 120M, 240M non-embedding params"
echo "Training: 2B tokens, lr=1e-3, batch=256k tokens, 7629 steps"
echo ""

# Determine job script and SSH host based on cluster
case $CLUSTER in
    research-secure)
        JOB_SCRIPT="scripts/jobs/together_secure.sh"
        SSH_HOST="research-secure-hn"
        SSH_CMD="ssh"
        ;;
    mk-turbo)
        JOB_SCRIPT="scripts/jobs/together_turbo.sh"
        SSH_HOST="mk-turbo-hn"
        SSH_CMD="ssh"
        ;;
    sphinx)
        JOB_SCRIPT="scripts/jobs/stanford_sphinx.sh"
        SSH_HOST="tanishq@sc.stanford.edu"
        SSH_CMD="sshpass -p 'december1972' ssh -o StrictHostKeyChecking=accept-new"
        ;;
    miso)
        JOB_SCRIPT="scripts/jobs/stanford_miso.sh"
        SSH_HOST="tanishq@sc.stanford.edu"
        SSH_CMD="sshpass -p 'december1972' ssh -o StrictHostKeyChecking=accept-new"
        ;;
    *)
        echo "Unknown cluster: $CLUSTER"
        echo "Available: research-secure, mk-turbo, sphinx, miso"
        exit 1
        ;;
esac

# Launch all 4 model sizes
SIZES="30m 60m 120m 240m"

echo "Submitting jobs to $CLUSTER via $SSH_HOST..."
echo ""

for SIZE in $SIZES; do
    echo "Submitting scale_${SIZE}..."

    CONFIG="apps/main/configs/scaling/scale_${SIZE}.yaml"
    EXPERIMENT="scaling_${SIZE}"

    # Submit job via SSH
    $SSH_CMD $SSH_HOST "cd ~/lingua-fork && export CONFIG='$CONFIG' && export EXPERIMENT='$EXPERIMENT' && sbatch $JOB_SCRIPT"

    sleep 1  # Small delay between submissions
done

echo ""
echo "=== All 4 jobs submitted ==="
echo "Monitor with: ./cluster.sh status"
echo "WandB: https://wandb.ai/tk07-stanford-university/tanishqbot"
echo ""
echo "Expected run time on 8x H100 (estimate):"
echo "  30M:  ~15 min"
echo "  60M:  ~20 min"
echo "  120M: ~30 min"
echo "  240M: ~50 min"
