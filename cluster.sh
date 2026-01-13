#!/bin/bash
# cluster.sh - Multi-cluster management script for lingua-fork
#
# Usage:
#   ./cluster.sh rsync           - Sync code to ALL clusters
#   ./cluster.sh best            - Find cluster with most idle nodes (ALWAYS RUN FIRST!)
#   ./cluster.sh submit <cluster> - Submit job to specified cluster
#   ./cluster.sh status          - Show queue status on all clusters
#   ./cluster.sh ssh <cluster>   - SSH to cluster login node

set -e

# Stanford password
STANFORD_PASS="december1972"

# All clusters
ALL_CLUSTERS="research-secure mk-turbo sphinx miso"

# Helper functions (bash 3.x compatible - no associative arrays)
get_ssh_host() {
    case "$1" in
        research-secure) echo "research-secure-hn" ;;
        mk-turbo) echo "mk-turbo-hn" ;;
        sphinx|miso) echo "tanishq@sc.stanford.edu" ;;
    esac
}

get_partition() {
    case "$1" in
        research-secure|mk-turbo) echo "batch" ;;
        sphinx) echo "sphinx" ;;
        miso) echo "miso" ;;
    esac
}

get_max_nodes() {
    case "$1" in
        research-secure|mk-turbo) echo "4" ;;
        sphinx) echo "2" ;;
        miso) echo "1" ;;
    esac
}

get_job_script() {
    case "$1" in
        research-secure) echo "scripts/jobs/together_secure.sh" ;;
        mk-turbo) echo "scripts/jobs/together_turbo.sh" ;;
        sphinx) echo "scripts/jobs/stanford_sphinx.sh" ;;
        miso) echo "scripts/jobs/stanford_miso.sh" ;;
    esac
}

is_stanford_cluster() {
    case "$1" in
        sphinx|miso) return 0 ;;
        *) return 1 ;;
    esac
}

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory (for rsync source)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ssh_cmd() {
    local cluster="$1"
    shift
    local host=$(get_ssh_host "$cluster")

    if is_stanford_cluster "$cluster"; then
        sshpass -p "$STANFORD_PASS" ssh -o StrictHostKeyChecking=accept-new "$host" "$@"
    else
        ssh "$host" "$@"
    fi
}

rsync_to_cluster() {
    local cluster="$1"
    local host=$(get_ssh_host "$cluster")

    echo -e "${BLUE}Syncing to $cluster ($host)...${NC}"

    if is_stanford_cluster "$cluster"; then
        sshpass -p "$STANFORD_PASS" rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' --exclude '*.pyc' \
            "$SCRIPT_DIR/" "$host:~/lingua-fork/" 2>/dev/null
    else
        rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' --exclude '*.pyc' \
            "$SCRIPT_DIR/" "$host:~/lingua-fork/" 2>/dev/null
    fi

    echo -e "${GREEN}Done syncing to $cluster${NC}"
}

get_idle_nodes() {
    local cluster="$1"
    local partition=$(get_partition "$cluster")
    local host=$(get_ssh_host "$cluster")

    # Get idle nodes count using sinfo
    if is_stanford_cluster "$cluster"; then
        idle=$(sshpass -p "$STANFORD_PASS" ssh -o StrictHostKeyChecking=accept-new "$host" \
            "sinfo -p $partition -h -o '%A' 2>/dev/null | cut -d'/' -f2" 2>/dev/null || echo "0")
    else
        idle=$(ssh "$host" "sinfo -p $partition -h -o '%A' 2>/dev/null | cut -d'/' -f2" 2>/dev/null || echo "0")
    fi

    # Clean up the output
    idle=$(echo "$idle" | tr -d '[:space:]')
    [ -z "$idle" ] && idle="0"

    echo "$idle"
}

get_queue_status() {
    local cluster="$1"
    local host=$(get_ssh_host "$cluster")

    echo -e "${BLUE}=== $cluster ===${NC}"

    if is_stanford_cluster "$cluster"; then
        sshpass -p "$STANFORD_PASS" ssh -o StrictHostKeyChecking=accept-new "$host" \
            "squeue -u tanishq 2>/dev/null || echo 'No jobs'" 2>/dev/null
    else
        ssh "$host" "squeue -u tkumar 2>/dev/null || echo 'No jobs'" 2>/dev/null
    fi
    echo ""
}

find_best_cluster() {
    local best_cluster=""
    local best_idle=0

    echo -e "${YELLOW}Checking idle nodes on all clusters...${NC}" >&2

    for cluster in $ALL_CLUSTERS; do
        idle=$(get_idle_nodes "$cluster")
        max=$(get_max_nodes "$cluster")
        echo -e "  $cluster: ${GREEN}$idle${NC}/${max} idle" >&2

        if [ "$idle" -gt "$best_idle" ]; then
            best_idle="$idle"
            best_cluster="$cluster"
        fi
    done

    if [ -z "$best_cluster" ] || [ "$best_idle" -eq 0 ]; then
        echo -e "${RED}No idle nodes found on any cluster!${NC}" >&2
        echo ""
    else
        echo -e "${GREEN}Best cluster: $best_cluster with $best_idle idle nodes${NC}" >&2
        echo "$best_cluster"
    fi
}

submit_job() {
    local cluster="$1"
    local host=$(get_ssh_host "$cluster")
    local job_script=$(get_job_script "$cluster")

    echo -e "${BLUE}Submitting to $cluster...${NC}"

    # Pass environment variables for config/experiment/extra_args
    local env_vars=""
    [ -n "$CONFIG" ] && env_vars="$env_vars export CONFIG='$CONFIG';"
    [ -n "$EXPERIMENT" ] && env_vars="$env_vars export EXPERIMENT='$EXPERIMENT';"
    [ -n "$EXTRA_ARGS" ] && env_vars="$env_vars export EXTRA_ARGS='$EXTRA_ARGS';"

    if is_stanford_cluster "$cluster"; then
        sshpass -p "$STANFORD_PASS" ssh -o StrictHostKeyChecking=accept-new "$host" \
            "cd ~/lingua-fork && $env_vars sbatch $job_script"
    else
        ssh "$host" "cd ~/lingua-fork && $env_vars sbatch $job_script"
    fi
}

case "$1" in
    rsync)
        echo -e "${YELLOW}Syncing to all clusters...${NC}"
        for cluster in $ALL_CLUSTERS; do
            rsync_to_cluster "$cluster" &
        done
        wait
        echo -e "${GREEN}All clusters synced!${NC}"
        ;;

    best)
        find_best_cluster
        ;;

    submit)
        if [ -z "$2" ]; then
            echo "Usage: ./cluster.sh submit <cluster>"
            echo "Clusters: $ALL_CLUSTERS"
            exit 1
        fi
        submit_job "$2"
        ;;

    status)
        for cluster in $ALL_CLUSTERS; do
            get_queue_status "$cluster"
        done
        ;;

    ssh)
        if [ -z "$2" ]; then
            echo "Usage: ./cluster.sh ssh <cluster>"
            echo "Clusters: $ALL_CLUSTERS"
            exit 1
        fi
        cluster="$2"
        host=$(get_ssh_host "$cluster")
        if is_stanford_cluster "$cluster"; then
            sshpass -p "$STANFORD_PASS" ssh -o StrictHostKeyChecking=accept-new "$host"
        else
            ssh "$host"
        fi
        ;;

    *)
        echo "Usage: ./cluster.sh <command>"
        echo ""
        echo "Commands:"
        echo "  rsync           - Sync code to ALL clusters"
        echo "  best            - Find cluster with most idle nodes"
        echo "  submit <cluster> - Submit job to specified cluster"
        echo "  status          - Show queue status on all clusters"
        echo "  ssh <cluster>   - SSH to cluster login node"
        echo ""
        echo "Clusters: $ALL_CLUSTERS"
        echo ""
        echo "Environment variables for submit:"
        echo "  CONFIG=<path>       - Config file (default: apps/main/configs/debug.yaml)"
        echo "  EXPERIMENT=<name>   - Experiment name for wandb grouping"
        echo "  EXTRA_ARGS=<args>   - Additional CLI args (OmegaConf format)"
        ;;
esac
