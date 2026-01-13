# CLAUDE.md - lingua-fork Setup Guide for Stanford Cluster

This document provides guidance for Claude (AI assistant) and future agents when working with this lingua-fork codebase on the Stanford cluster.

## Training Time Estimates

**Configuration**: batch_size=8 per GPU, seq_len=2048 → 16K tokens/step/GPU

| Model Size | Params | GPUs | Tokens/Step | 500 Steps Time | Notes |
|------------|--------|------|-------------|----------------|-------|
| Small | 27.5M | 1x A100 | 16K | ~1 min | Verified |
| Small | 27.5M | 8x H200 | 131K | ~2.5 min | Verified |
| 40M | ~40M | 8x H200 | 131K | ~3.5 min | Estimated |
| 80M | ~80M | 8x H200 | 131K | ~7 min | Estimated |
| 160M | ~160M | 8x H200 | 131K | ~14 min | Estimated |

**To get 128K+ tokens/step**: Use 8 GPUs with batch_size=8 (8 × 8 × 2048 = 131K tokens/step)

**Observed throughput**:
- Single A100: ~230K tokens/sec, loss 3.78 → 1.54 in 500 steps
- 8x H200: ~75K tokens/sec/GPU, loss 4.11 → 1.36 in 500 steps

## Quick Start

### Running Training

**Single GPU (sphinx partition - A100):**
```bash
sbatch --job-name=train --account=nlp --partition=sphinx --gres=gpu:1 \
  --time=00:30:00 --mem=48G \
  --output=/juice5b/scr5b/tanishq/lingua-logs/%j.out \
  --error=/juice5b/scr5b/tanishq/lingua-logs/%j.err \
  --wrap="cd ~/lingua-fork && uv run torchrun --nproc_per_node=1 --standalone -m apps.main.train config=/juice5b/scr5b/tanishq/lingua-configs/sphinx_test.yaml"
```

**8 GPU (miso partition - H200):**
```bash
sbatch --job-name=train --account=miso --partition=miso --gpus-per-task=8 \
  --ntasks=1 --time=00:30:00 --mem=200G \
  --output=/juice5b/scr5b/tanishq/lingua-logs/%j.out \
  --error=/juice5b/scr5b/tanishq/lingua-logs/%j.err \
  --wrap="cd ~/lingua-fork && uv run torchrun --nproc_per_node=8 --standalone -m apps.main.train config=/juice5b/scr5b/tanishq/lingua-configs/miso_test.yaml"
```

## Key Paths

| Path | Description |
|------|-------------|
| `~/lingua-fork` | Main repo (on AFS) |
| `/juice5b/scr5b/tanishq/lingua-venv` | Python venv (symlinked from .venv) |
| `/juice5b/scr5b/tanishq/lingua-configs/` | Config files |
| `/juice5b/scr5b/tanishq/lingua-logs/` | Job logs |
| `/juice5b/scr5b/tanishq/lingua-out/` | Training outputs |
| `/juice5/scr5/nlp/data/huggingface/lingua-data/` | Training data |

## Patches Applied

### 1. distributed.py - SLURM Hang Fix (line 257)
```python
# Removed mp.set_start_method and mp.Manager() that hang in SLURM
```

### 2. train.py - Wandb/Eval Null Checks
```python
if args.logging.wandb is not None:
    args.logging.wandb.id = train_state.wandb_id

if args.eval is not None:
    launch_eval(...)
```

## Example Config

```yaml
dump_dir: /juice5b/scr5b/tanishq/lingua-out/my_run
name: my_run
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
  dim: 512      # 27.5M params
  n_layers: 8
  n_heads: 8
  # dim: 768, n_layers: 12 → ~80M params
  # dim: 1024, n_layers: 16 → ~160M params

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

## Useful Commands

```bash
squeue -u tanishq              # Check job queue
scancel <job_id>               # Cancel job
sinfo -p sphinx                # Check sphinx availability
sinfo -p miso                  # Check miso availability
```
