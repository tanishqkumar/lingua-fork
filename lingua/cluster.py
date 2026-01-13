# Copyright (c) Meta Platforms, Inc. and affiliates.
# Cluster detection and configuration for multi-cluster training

import os
import socket
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger()


@dataclass
class ClusterConfig:
    """Configuration for a specific cluster/partition."""
    name: str                    # e.g., "research-secure", "mk-turbo", "sphinx", "miso"
    cluster: str                 # e.g., "together", "stanford"
    data_root: str               # Root directory for datasets
    dump_base: str               # Base directory for outputs
    max_nodes: int               # Maximum nodes we can use
    gpus_per_node: int           # GPUs per node
    gpu_type: str                # e.g., "H100", "H200", "A100"
    partition: Optional[str]     # SLURM partition name (if applicable)
    account: Optional[str]       # SLURM account (if applicable)


# Cluster configurations
CLUSTER_CONFIGS: Dict[str, ClusterConfig] = {
    # Together AI clusters
    "research-secure": ClusterConfig(
        name="research-secure",
        cluster="together",
        data_root="/data/tkumar/datasets",
        dump_base="/data/tkumar/lingua-out",
        max_nodes=4,
        gpus_per_node=8,
        gpu_type="H100",
        partition="batch",
        account=None,
    ),
    "mk-turbo": ClusterConfig(
        name="mk-turbo",
        cluster="together",
        data_root="/data/tkumar/datasets",
        dump_base="/data/tkumar/lingua-out",
        max_nodes=1,
        gpus_per_node=8,
        gpu_type="H100",
        partition="batch",
        account=None,
    ),
    # Stanford clusters
    "sphinx": ClusterConfig(
        name="sphinx",
        cluster="stanford",
        data_root="/juice5/scr5/nlp/data/huggingface/lingua-data",
        dump_base="/juice5b/scr5b/tanishq/lingua-out",
        max_nodes=2,
        gpus_per_node=8,
        gpu_type="A100",
        partition="sphinx",
        account="nlp",
    ),
    "miso": ClusterConfig(
        name="miso",
        cluster="stanford",
        data_root="/juice5/scr5/nlp/data/huggingface/lingua-data",
        dump_base="/juice5b/scr5b/tanishq/lingua-out",
        max_nodes=2,
        gpus_per_node=8,
        gpu_type="H200",
        partition="miso",
        account="nlp",
    ),
}

# Priority order for launching (descending priority)
LAUNCH_PRIORITY = ["research-secure", "mk-turbo", "sphinx", "miso"]


def detect_cluster() -> Optional[str]:
    """Detect which cluster we're running on based on hostname and environment."""
    hostname = socket.gethostname().lower()

    # Together AI clusters
    if "research-secure" in hostname or hostname.startswith("research-secure"):
        return "research-secure"
    if "mk-turbo" in hostname or hostname.startswith("mk-turbo"):
        return "mk-turbo"

    # Stanford clusters - check SLURM partition or hostname
    slurm_partition = os.environ.get("SLURM_JOB_PARTITION", "").lower()
    if slurm_partition == "sphinx" or "sphinx" in hostname:
        return "sphinx"
    if slurm_partition == "miso" or "miso" in hostname:
        return "miso"

    # Check for Stanford login node
    if "sc.stanford.edu" in hostname or hostname.startswith("rice") or hostname.startswith("wheat"):
        # On Stanford login node, check environment hint
        cluster_hint = os.environ.get("LINGUA_CLUSTER", "").lower()
        if cluster_hint in CLUSTER_CONFIGS:
            return cluster_hint
        # Default to sphinx for Stanford
        return "sphinx"

    # Check environment variable override
    cluster_override = os.environ.get("LINGUA_CLUSTER", "").lower()
    if cluster_override in CLUSTER_CONFIGS:
        return cluster_override

    return None


def get_cluster_config(cluster_name: Optional[str] = None) -> ClusterConfig:
    """Get configuration for a cluster. Auto-detects if not specified."""
    if cluster_name is None:
        cluster_name = detect_cluster()

    if cluster_name is None:
        raise RuntimeError(
            "Could not detect cluster. Set LINGUA_CLUSTER environment variable "
            f"to one of: {list(CLUSTER_CONFIGS.keys())}"
        )

    if cluster_name not in CLUSTER_CONFIGS:
        raise ValueError(
            f"Unknown cluster: {cluster_name}. "
            f"Available: {list(CLUSTER_CONFIGS.keys())}"
        )

    return CLUSTER_CONFIGS[cluster_name]


def get_data_root(cluster_name: Optional[str] = None) -> str:
    """Get the data root directory for the current/specified cluster."""
    config = get_cluster_config(cluster_name)
    return config.data_root


def get_dump_base(cluster_name: Optional[str] = None) -> str:
    """Get the output base directory for the current/specified cluster."""
    config = get_cluster_config(cluster_name)
    return config.dump_base


def resolve_paths_for_cluster(
    data_root: Optional[str] = None,
    dump_base: Optional[str] = None,
    dump_dir: Optional[str] = None,
    cluster_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Resolve paths for the current cluster.

    If paths are provided and look like absolute paths for a different cluster,
    they will be remapped to the current cluster's paths.
    """
    config = get_cluster_config(cluster_name)

    result = {
        "data_root": data_root or config.data_root,
        "dump_base": dump_base or config.dump_base,
    }

    # Remap Stanford paths to Together paths and vice versa
    stanford_data = "/juice5/scr5/nlp/data/huggingface/lingua-data"
    together_data = "/data/tkumar/datasets"
    stanford_dump = "/juice5b/scr5b/tanishq/lingua-out"
    together_dump = "/data/tkumar/lingua-out"

    if config.cluster == "together":
        # Remap Stanford paths to Together
        if result["data_root"].startswith("/juice"):
            result["data_root"] = result["data_root"].replace(stanford_data, together_data)
        if result["dump_base"] and result["dump_base"].startswith("/juice"):
            result["dump_base"] = result["dump_base"].replace(stanford_dump, together_dump)
        if dump_dir and dump_dir.startswith("/juice"):
            result["dump_dir"] = dump_dir.replace(stanford_dump, together_dump)
    elif config.cluster == "stanford":
        # Remap Together paths to Stanford
        if result["data_root"].startswith("/data/tkumar"):
            result["data_root"] = result["data_root"].replace(together_data, stanford_data)
        if result["dump_base"] and result["dump_base"].startswith("/data/tkumar"):
            result["dump_base"] = result["dump_base"].replace(together_dump, stanford_dump)
        if dump_dir and dump_dir.startswith("/data/tkumar"):
            result["dump_dir"] = dump_dir.replace(together_dump, stanford_dump)

    return result


def get_wandb_tags(
    cluster_name: Optional[str] = None,
    experiment: Optional[str] = None,
    extra_tags: Optional[list] = None,
) -> list:
    """Generate WandB tags for the current run."""
    config = get_cluster_config(cluster_name)

    tags = [
        config.cluster,           # "together" or "stanford"
        config.name,              # "research-secure", "mk-turbo", "sphinx", "miso"
        config.gpu_type,          # "H100", "H200", "A100"
    ]

    if experiment:
        tags.append(experiment)

    if extra_tags:
        tags.extend(extra_tags)

    return tags


def get_wandb_group(
    experiment: str,
    cluster_name: Optional[str] = None,
) -> str:
    """Generate WandB group name for experiment tracking."""
    config = get_cluster_config(cluster_name)
    return f"{experiment}/{config.cluster}/{config.name}"


def print_cluster_info(cluster_name: Optional[str] = None):
    """Print information about the current cluster configuration."""
    try:
        config = get_cluster_config(cluster_name)
        logger.info(f"Cluster: {config.name} ({config.cluster})")
        logger.info(f"GPU Type: {config.gpu_type} x {config.gpus_per_node} per node")
        logger.info(f"Max Nodes: {config.max_nodes}")
        logger.info(f"Data Root: {config.data_root}")
        logger.info(f"Dump Base: {config.dump_base}")
    except Exception as e:
        logger.warning(f"Could not detect cluster: {e}")


if __name__ == "__main__":
    # Test cluster detection
    logging.basicConfig(level=logging.INFO)
    print(f"Detected cluster: {detect_cluster()}")
    print_cluster_info()
