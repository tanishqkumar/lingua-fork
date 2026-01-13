# CLAUDE.md - lingua-fork Setup Guide for Stanford Cluster

This document provides guidance for Claude (AI assistant) and future agents when working with this lingua-fork codebase on the Stanford cluster.

## Quick Start

### Prerequisites
- Access to Stanford cluster (sc.stanford.edu)
- Access to miso or sphinx partition
- Account: `nlp` for sphinx, `miso` for miso partition

### Environment Setup

The Python virtual environment is stored on shared storage to avoid AFS quota issues:
```bash
# The venv lives at:
/juice5b/scr5b/tanishq/lingua-venv

# Symlinked from:
~/lingua-fork/.venv -> /juice5b/scr5b/tanishq/lingua-venv
```

### Running Training

**Single GPU (sphinx partition - A100):**
```bash
sbatch --job-name=sphinx_train --account=nlp --partition=sphinx --gres=gpu:1 \
  --time=00:30:00 --mem=48G \
  --output=/juice5b/scr5b/tanishq/lingua-logs/sphinx_%j.out \
  --error=/juice5b/scr5b/tanishq/lingua-logs/sphinx_%j.err \
  --wrap="cd ~/lingua-fork && uv run torchrun --nproc_per_node=1 --standalone -m apps.main.train config=/juice5b/scr5b/tanishq/lingua-configs/sphinx_test.yaml"
```

**8 GPU (miso partition - H200):**
```bash
sbatch --job-name=miso_train --account=miso --partition=miso --gpus-per-task=8 \
  --ntasks=1 --time=00:30:00 --mem=200G \
  --output=/juice5b/scr5b/tanishq/lingua-logs/miso_%j.out \
  --error=/juice5b/scr5b/tanishq/lingua-logs/miso_%j.err \
  --wrap="cd ~/lingua-fork && uv run torchrun --nproc_per_node=8 --standalone -m apps.main.train config=/juice5b/scr5b/tanishq/lingua-configs/miso_test.yaml"
```

## Key Paths

| Path | Description |
|------|-------------|
| `~/lingua-fork` | Main repo (on AFS/sailhome) |
| `/juice5b/scr5b/tanishq/lingua-venv` | Python venv (shared storage) |
| `/juice5b/scr5b/tanishq/lingua-configs/` | Config files |
| `/juice5b/scr5b/tanishq/lingua-logs/` | Job logs |
| `/juice5b/scr5b/tanishq/lingua-out/` | Training outputs/checkpoints |
| `/juice5/scr5/nlp/data/huggingface/lingua-data/` | Training data |

## Patches Applied

### 1. distributed.py - SLURM Hang Fix
**File:** `lingua/distributed.py` line 257

The original code had `mp.set_start_method` and `mp.Manager()` calls that hang in SLURM:
```python
# ORIGINAL (hangs in SLURM):
mp.set_start_method(dist_args.spawn_method)
with mp.Manager():
    pass

# PATCHED:
# PATCHED: Removed mp.set_start_method and mp.Manager to fix SLURM hang
```

### 2. train.py - Wandb Null Check
**File:** `apps/main/train.py` line 495

Added null check before accessing wandb.id:
```python
# PATCHED:
if args.logging.wandb is not None:
    args.logging.wandb.id = train_state.wandb_id
```

### 3. train.py - Eval Null Check  
**File:** `apps/main/train.py` (end of train function)

Added null check before calling launch_eval:
```python
# PATCHED:
if args.eval is not None:
    launch_eval(...)
```

## Config Structure

Example minimal config for testing:
```yaml
dump_dir: /juice5b/scr5b/tanishq/lingua-out/test_run
name: test_50M
steps: 500
seed: 42

optim:
  lr: 3.0e-4
  warmup: 100
  lr_min_ratio: 0.1
  clip: 1.0

distributed:
  fsdp_type: no_shard
  compile: false
  model_dtype: bf16
  tp_size: 1

model:
  dim: 512
  n_layers: 8
  n_heads: 8

data:
  root_dir: /juice5/scr5/nlp/data/huggingface/lingua-data/
  sources:
    fineweb_edu_10bt_shuffled: 100.0
  batch_size: 8
  seq_len: 2048
  tokenizer:
    name: bytes

logging:
  freq: 10

checkpoint:
  dump:
    every: 5000
    keep: 1
  eval:
    every: 5000
    keep: 1

eval: null
```

## Verified Working

- **Sphinx partition (A100):** 500 steps completed, loss decreased from ~3.8 to ~1.5
- **Throughput:** ~200-500K tokens/sec on single A100
- **Memory usage:** ~10% for 27M parameter model

## Common Issues

1. **AFS slow/hangs:** Use /juice5b for data and logs, not AFS home
2. **Disk quota:** Keep venv and data on /juice5b, only code on AFS
3. **Job output not visible:** Output goes to compute node /tmp - use shared paths
4. **Config errors:** Use `warmup` not `warmup_steps`, checkpoint.dump.every needs int (not null)
5. **NCCL hang:** Ensure the distributed.py patch is applied

## Useful Commands

```bash
# Check job queue
squeue -u tanishq

# Cancel jobs
scancel <job_id>

# Check logs in real-time
tail -f /juice5b/scr5b/tanishq/lingua-logs/sphinx_<job_id>.out

# Check GPU availability
sinfo -p sphinx
sinfo -p miso
```
