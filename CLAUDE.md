# CLAUDE.md - lingua-fork Setup Guide for Stanford Cluster

## Setup (One-Time)

```bash
# 1. Clone repo (patches already included)
git clone https://github.com/tanishqkumar/lingua-fork.git
cd lingua-fork

# 2. Install dependencies
uv sync

# 3. Create config file (see example below)
# Data is at: /juice5/scr5/nlp/data/huggingface/lingua-data/
```

That is it. The patches for SLURM compatibility are already in the repo.

## Running Training

**sphinx (1-8x A100):**
```bash
sbatch --account=nlp --partition=sphinx --gres=gpu:1 --time=01:00:00 --mem=48G \
  --wrap="cd ~/lingua-fork && uv run torchrun --nproc_per_node=1 --standalone -m apps.main.train config=my_config.yaml"
```

**miso (8x H200 only - must use all 8):**
```bash
sbatch --account=miso --partition=miso --gpus-per-task=8 --ntasks=1 --time=01:00:00 --mem=200G \
  --wrap="cd ~/lingua-fork && uv run torchrun --nproc_per_node=8 --standalone -m apps.main.train config=my_config.yaml"
```

## GPU Allocation Rules

- **sphinx**: Can request 1-8 GPUs with `--gres=gpu:N`
- **miso**: Must use all 8 GPUs on a node - no partial allocations

## Training Throughput (Measured)

| Setup | Tokens/Step | Throughput | 500 Steps |
|-------|-------------|------------|-----------|
| 1x A100 (sphinx) | 16K | ~140K tok/s | ~58s |
| 8x H200 (miso) | 131K | ~460K tok/s | ~142s |

## Example Config

```yaml
dump_dir: /juice5b/scr5b/YOUR_USERNAME/lingua-out/my_run
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

## Patches Included

1. **lingua/distributed.py:257** - Removed `mp.set_start_method` and `mp.Manager()` that hang in SLURM
2. **apps/main/train.py** - Added null checks for `wandb.id` and `eval`
