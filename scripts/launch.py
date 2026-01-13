#!/usr/bin/env python3
"""
Smart multi-cluster launcher that checks for idle nodes before submitting.

Checks clusters in priority order and submits to the first one with idle nodes.

Usage:
    # Check availability and submit to best cluster
    python scripts/launch.py -c apps/main/configs/debug.yaml -e my_experiment

    # Dry run - just check availability
    python scripts/launch.py --dry-run

    # Force specific cluster (skip idle check)
    python scripts/launch.py --cluster research-secure -c apps/main/configs/debug.yaml

    # Pass extra training args
    python scripts/launch.py -c apps/main/configs/debug.yaml -e test -x "steps=500 model.dim=512"
"""

import subprocess
import sys
import argparse

# Cluster configs in priority order: (name, ssh_host, ssh_pass, partition, job_script)
CLUSTERS = [
    ("research-secure", "research-secure-hn", None, "batch", "scripts/jobs/together_secure.sh"),
    ("mk-turbo", "mk-turbo-hn", None, "batch", "scripts/jobs/together_turbo.sh"),
    ("sphinx", "tanishq@sc.stanford.edu", "december1972", "sphinx", "scripts/jobs/stanford_sphinx.sh"),
    ("miso", "tanishq@sc.stanford.edu", "december1972", "miso", "scripts/jobs/stanford_miso.sh"),
]


def ssh_cmd(host: str, cmd: str, password: str = None, timeout: int = 15) -> tuple[int, str]:
    """Run SSH command, return (returncode, output)."""
    if password:
        full_cmd = ["sshpass", "-p", password, "ssh", "-o", "StrictHostKeyChecking=no", host, cmd]
    else:
        full_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", host, cmd]

    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return 1, "timeout"
    except Exception as e:
        return 1, str(e)


def check_idle_nodes(host: str, partition: str, password: str = None) -> int:
    """Check how many nodes are idle on a partition. Returns count of idle nodes."""
    cmd = f"sinfo -p {partition} -t idle -h -o '%D' 2>/dev/null | head -1"
    code, output = ssh_cmd(host, cmd, password)

    if code != 0:
        return 0

    try:
        # Handle empty output
        output = output.strip()
        if not output:
            return 0
        return int(output)
    except (ValueError, AttributeError):
        return 0


def submit_job(host: str, job_script: str, password: str = None,
               config: str = None, experiment: str = None, extra_args: str = None) -> tuple[bool, str]:
    """Submit a job to the cluster. Returns (success, message)."""
    env_parts = []
    if config:
        env_parts.append(f"CONFIG={config}")
    if experiment:
        env_parts.append(f"EXPERIMENT={experiment}")
    if extra_args:
        env_parts.append(f"EXTRA_ARGS='{extra_args}'")

    env_str = " ".join(env_parts) + " " if env_parts else ""
    cmd = f"cd ~/lingua-fork && {env_str}sbatch {job_script}"

    code, output = ssh_cmd(host, cmd, password, timeout=30)
    return code == 0, output.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Smart launcher - checks idle nodes before submitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "-c", default="apps/main/configs/debug.yaml",
                        help="Config file path")
    parser.add_argument("--experiment", "-e", default="default",
                        help="Experiment name for WandB grouping")
    parser.add_argument("--extra", "-x", default="",
                        help="Extra args passed to training (e.g., 'steps=500 model.dim=512')")
    parser.add_argument("--cluster", choices=[c[0] for c in CLUSTERS],
                        help="Force specific cluster (skips idle check)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Check availability without submitting")
    args = parser.parse_args()

    print("Checking cluster availability (priority order)...")
    print("-" * 60)

    selected = None
    results = []

    for name, host, password, partition, job_script in CLUSTERS:
        # If user specified a cluster, only check that one
        if args.cluster and args.cluster != name:
            continue

        idle = check_idle_nodes(host, partition, password)
        status = f"{idle} idle node(s)" if idle > 0 else "no idle nodes"

        is_selected = False
        if idle > 0 and selected is None:
            selected = (name, host, password, partition, job_script)
            is_selected = True

        results.append((name, idle, is_selected))

    for name, idle, is_selected in results:
        marker = " <-- SELECTED" if is_selected else ""
        status = f"{idle} idle" if idle > 0 else "busy"
        print(f"  {name:20} {status}{marker}")

    print("-" * 60)

    if selected is None:
        if args.cluster:
            # User forced a specific cluster, use it even if busy
            for name, host, password, partition, job_script in CLUSTERS:
                if name == args.cluster:
                    selected = (name, host, password, partition, job_script)
                    print(f"Warning: {args.cluster} has no idle nodes, submitting anyway...")
                    break
        else:
            # No idle nodes anywhere - fall back to priority order
            print("Warning: No idle nodes found, using priority order...")
            selected = CLUSTERS[0]  # First in priority order

    name, host, password, partition, job_script = selected

    if args.dry_run:
        print(f"Would submit to: {name}")
        print(f"  Config: {args.config}")
        print(f"  Experiment: {args.experiment}")
        if args.extra:
            print(f"  Extra args: {args.extra}")
        sys.exit(0)

    print(f"Submitting to {name}...")
    print(f"  Config: {args.config}")
    print(f"  Experiment: {args.experiment}")

    success, output = submit_job(
        host, job_script, password,
        config=args.config,
        experiment=args.experiment,
        extra_args=args.extra if args.extra else None
    )

    if success:
        print(f"Success: {output}")
    else:
        print(f"Failed: {output}")
        sys.exit(1)


if __name__ == "__main__":
    main()
