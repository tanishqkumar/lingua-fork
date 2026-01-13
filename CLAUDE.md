# CLAUDE.md - lingua-fork Setup Guide for Stanford Cluster

## Setup

```bash
git clone https://github.com/tanishqkumar/lingua-fork.git
cd lingua-fork
uv sync
```

## Running Training

**Interactive (see loss live):**
```bash
# Get a GPU node first
srun --account=nlp --partition=sphinx --gres=gpu:1 --time=01:00:00 --mem=48G --pty bash

# Then run training
cd ~/lingua-fork
uv run torchrun --nproc_per_node=1 --standalone -m apps.main.train config=my_config.yaml
```

**Batch job:**
```bash
sbatch --account=nlp --partition=sphinx --gres=gpu:1 --time=01:00:00 --mem=48G \
  --wrap="cd ~/lingua-fork && uv run torchrun --nproc_per_node=1 --standalone -m apps.main.train config=my_config.yaml"
```

**Argument overrides (OmegaConf dot notation):**
```bash
uv run torchrun --nproc_per_node=1 --standalone -m apps.main.train \
  config=my_config.yaml \
  steps=1000 \
  model.dim=768 \
  optim.lr=1e-4 \
  data.batch_size=16
```

## GPU Allocation

- **sphinx**: 1-8 GPUs with `--gres=gpu:N`
- **miso**: Must use all 8 GPUs (`--gpus-per-task=8 --nproc_per_node=8`)

## Throughput (Measured)

| Setup | Tokens/Step | Throughput | 500 Steps |
|-------|-------------|------------|-----------|
| 1x A100 | 16K | ~140K tok/s | ~58s |
| 8x H200 | 131K | ~460K tok/s | ~142s |

## Example Config

```yaml
dump_dir: /juice5b/scr5b/YOUR_USER/out
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

1. `lingua/distributed.py:257` - Removed SLURM-hanging mp.Manager()
2. `apps/main/train.py` - Added null checks for wandb/eval
