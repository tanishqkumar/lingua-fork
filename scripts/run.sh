#!/bin/bash
# Unified training launcher for lingua-fork
# Supports both interactive and batch modes across all clusters
#
# Usage:
#   Interactive (single GPU debug):
#     ./scripts/run.sh --interactive --config apps/main/configs/debug.yaml
#
#   Batch job:
#     ./scripts/run.sh --partition sphinx --config apps/main/configs/base.yaml --experiment lr_sweep
#
#   With extra args:
#     ./scripts/run.sh --partition research-secure --config apps/main/configs/debug.yaml \
#         --experiment test -- steps=500 model.dim=768

set -e

# Defaults
PARTITION=""
CONFIG="apps/main/configs/debug.yaml"
EXPERIMENT="default"
INTERACTIVE=false
NGPUS=""
NODES=1
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --partition|-p)
            PARTITION="$2"
            shift 2
            ;;
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --experiment|-e)
            EXPERIMENT="$2"
            shift 2
            ;;
        --interactive|-i)
            INTERACTIVE=true
            shift
            ;;
        --gpus|-g)
            NGPUS="$2"
            shift 2
            ;;
        --nodes|-n)
            NODES="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS="$*"
            break
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Detect cluster if not specified
if [ -z "$PARTITION" ]; then
    HOSTNAME=$(hostname)
    if [[ "$HOSTNAME" == *"research-secure"* ]]; then
        PARTITION="research-secure"
    elif [[ "$HOSTNAME" == *"mk-turbo"* ]]; then
        PARTITION="mk-turbo"
    elif [[ "$HOSTNAME" == *"rice"* ]] || [[ "$HOSTNAME" == *"wheat"* ]] || [[ "$HOSTNAME" == *"stanford"* ]]; then
        # Stanford login node - default to sphinx
        PARTITION="sphinx"
    else
        echo "Could not detect cluster. Please specify with --partition"
        echo "Available: research-secure, mk-turbo, sphinx, miso"
        exit 1
    fi
fi

echo "=== Lingua Training Launcher ==="
echo "Partition: $PARTITION"
echo "Config: $CONFIG"
echo "Experiment: $EXPERIMENT"
echo "Interactive: $INTERACTIVE"

# Set environment variable
export LINGUA_CLUSTER="$PARTITION"

if [ "$INTERACTIVE" = true ]; then
    # Interactive mode - run directly
    echo "Running in interactive mode..."

    # Determine GPU count
    if [ -z "$NGPUS" ]; then
        NGPUS=1
    fi

    # Determine how to run based on cluster
    case $PARTITION in
        research-secure|mk-turbo)
            source /data/tkumar/miniconda3/bin/activate 2>/dev/null || true
            torchrun \
                --nproc_per_node=$NGPUS \
                --standalone \
                -m apps.main.train \
                config=$CONFIG \
                $EXTRA_ARGS
            ;;
        sphinx|miso)
            uv run torchrun \
                --nproc_per_node=$NGPUS \
                --standalone \
                -m apps.main.train \
                config=$CONFIG \
                $EXTRA_ARGS
            ;;
    esac
else
    # Batch mode - submit SLURM job
    echo "Submitting batch job..."

    # Select job script based on partition
    case $PARTITION in
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
            echo "Unknown partition: $PARTITION"
            exit 1
            ;;
    esac

    # Create log directory
    case $PARTITION in
        research-secure|mk-turbo)
            mkdir -p /data/tkumar/lingua-out/logs 2>/dev/null || true
            ;;
        sphinx|miso)
            mkdir -p /juice5b/scr5b/tanishq/lingua-out/logs 2>/dev/null || true
            ;;
    esac

    # Submit job with environment variables
    export CONFIG EXPERIMENT EXTRA_ARGS

    if [ "$NODES" -gt 1 ]; then
        sbatch --nodes=$NODES $JOB_SCRIPT
    else
        sbatch $JOB_SCRIPT
    fi
fi

echo "=== Done ==="
