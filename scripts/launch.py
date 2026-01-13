#!/usr/bin/env python3
"""
Multi-cluster training launcher for lingua-fork.

This script:
1. Detects the current cluster or uses the specified one
2. Resolves paths (data_root, dump_base) for the cluster
3. Sets up WandB tags/group based on experiment name and cluster
4. Launches training with torchrun

Usage:
    # Auto-detect cluster and launch
    python scripts/launch.py --config apps/main/configs/debug.yaml --experiment "lr_sweep"

    # Specify cluster explicitly
    python scripts/launch.py --config apps/main/configs/debug.yaml --cluster research-secure

    # Override specific settings
    python scripts/launch.py --config apps/main/configs/debug.yaml --experiment "test" steps=500 model.dim=512
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lingua.cluster import (
    CLUSTER_CONFIGS,
    LAUNCH_PRIORITY,
    detect_cluster,
    get_cluster_config,
    get_wandb_tags,
    get_wandb_group,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-cluster training launcher")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        choices=list(CLUSTER_CONFIGS.keys()),
        default=None,
        help="Target cluster (auto-detected if not specified)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name for WandB grouping (e.g., 'lr_sweep', 'arch_ablation')",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="Number of processes (GPUs). Defaults to cluster's gpus_per_node.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity (team/username)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="lingua-fork",
        help="WandB project name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    # All remaining args are passed to train.py as OmegaConf overrides
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in OmegaConf format (e.g., steps=1000 model.dim=512)",
    )

    return parser.parse_args()


def build_command(args):
    """Build the torchrun command with all necessary arguments."""
    # Detect or use specified cluster
    cluster_name = args.cluster or detect_cluster()
    if cluster_name is None:
        print("ERROR: Could not detect cluster. Please specify with --cluster")
        print(f"Available clusters: {list(CLUSTER_CONFIGS.keys())}")
        sys.exit(1)

    config = get_cluster_config(cluster_name)
    print(f"Using cluster: {config.name} ({config.cluster})")
    print(f"  GPU Type: {config.gpu_type}")
    print(f"  Data Root: {config.data_root}")
    print(f"  Dump Base: {config.dump_base}")

    # Determine number of processes
    nproc = args.nproc or config.gpus_per_node
    if nproc > config.gpus_per_node * config.max_nodes:
        print(f"WARNING: Requested {nproc} GPUs but cluster max is {config.gpus_per_node * config.max_nodes}")

    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--standalone",
        "-m", "apps.main.train",
        f"config={args.config}",
    ]

    # Add cluster-specific path overrides
    cmd.append(f"data.root_dir={config.data_root}")
    cmd.append(f"dump_base={config.dump_base}")

    # Add WandB configuration
    if not args.no_wandb:
        tags = get_wandb_tags(cluster_name, args.experiment)
        tags_str = "[" + ",".join(tags) + "]"
        cmd.append(f"logging.wandb.tags={tags_str}")
        cmd.append(f"logging.wandb.project={args.wandb_project}")

        if args.experiment:
            group = get_wandb_group(args.experiment, cluster_name)
            cmd.append(f"logging.wandb.group={group}")

        if args.wandb_entity:
            cmd.append(f"logging.wandb.entity={args.wandb_entity}")
    else:
        cmd.append("logging.wandb=null")

    # Add any user-provided overrides
    cmd.extend(args.overrides)

    return cmd


def main():
    args = parse_args()

    cmd = build_command(args)

    print("\nCommand:")
    print("  " + " \\\n    ".join(cmd))
    print()

    if args.dry_run:
        print("(dry run - not executing)")
        return

    # Execute
    env = os.environ.copy()
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
