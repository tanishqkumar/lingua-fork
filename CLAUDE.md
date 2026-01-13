# CLAUDE.md - lingua-fork Setup Guide for Stanford Cluster

## Training Throughput (Measured)

| Setup | Model | Batch/GPU | Total Batch | Tokens/Step | Throughput | 500 Steps |
|-------|-------|-----------|-------------|-------------|------------|-----------|
| 1x A100 (sphinx) | 27.5M | 8 | 8 | 16K | ~140K tok/s | ~58s |
| 8x H200 (miso) | 27.5M | 8 | 64 | 131K | ~460K tok/s | ~142s |

**Note**: Miso processed 8x more tokens per step, so despite taking longer wall-clock time, it has ~3.3x higher throughput.

## Quick Start

**Single GPU (sphinx - A100):**
```bash
sbatch --account=nlp --partition=sphinx --gres=gpu:1 --time=00:30:00 --mem=48G \
  --output=/juice5b/scr5b/tanishq/lingua-logs/%j.out \
  --error=/juice5b/scr5b/tanishq/lingua-logs/%j.err \
  --wrap="cd ~/lingua-fork && uv run torchrun --nproc_per_node=1 --standalone -m apps.main.train config=/juice5b/scr5b/tanishq/lingua-configs/sphinx_test.yaml"
```

**8 GPU (miso - H200):**
```bash
sbatch --account=miso --partition=miso --gpus-per-task=8 --ntasks=1 --time=00:30:00 --mem=200G \
  --output=/juice5b/scr5b/tanishq/lingua-logs/%j.out \
  --error=/juice5b/scr5b/tanishq/lingua-logs/%j.err \
  --wrap="cd ~/lingua-fork && uv run torchrun --nproc_per_node=8 --standalone -m apps.main.train config=/juice5b/scr5b/tanishq/lingua-configs/miso_test.yaml"
```

## Key Paths

| Path | Description |
|------|-------------|
| `~/lingua-fork` | Code (AFS) |
| `/juice5b/scr5b/tanishq/lingua-venv` | Python venv (symlinked from .venv) |
| `/juice5b/scr5b/tanishq/lingua-configs/` | Config files |
| `/juice5b/scr5b/tanishq/lingua-logs/` | Job logs |
| `/juice5b/scr5b/tanishq/lingua-out/` | Checkpoints |
| `/juice5/scr5/nlp/data/huggingface/lingua-data/` | Training data |

## Patches Applied

1. **lingua/distributed.py:257** - Removed `mp.set_start_method` and `mp.Manager()` that hang in SLURM
2. **apps/main/train.py** - Added null checks for `wandb.id` and `eval`

## Example Config

```yaml
dump_dir: /juice5b/scr5b/tanishq/lingua-out/my_run
name: my_run
steps: 500

optim:
  lr: 3.0e-4
  warmup: 100
  clip: 1.0

distributed:
  fsdp_type: no_shard
  compile: false
  model_dtype: bf16

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

checkpoint:
  dump:
    every: 5000
    keep: 1
  eval:
    every: 5000
    keep: 1

eval: null
```
