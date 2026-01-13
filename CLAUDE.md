# CLAUDE.md - lingua-fork Setup Guide for Stanford Cluster

## SSH Access (IMPORTANT - READ FIRST)
```
Host: sc.stanford.edu
User: tanishq
Password: december1972
Clusters: miso (H200s), sphinx (A100s)
```

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
uv run torchrun --nproc_per_node=1 --standalone -m apps.main.train config=apps/main/configs/debug.yaml
```

**Argument overrides (OmegaConf dot notation):**
```bash
uv run torchrun --nproc_per_node=1 --standalone -m apps.main.train \
  config=apps/main/configs/debug.yaml \
  steps=1000 \
  model.dim=768 \
  optim.lr=1e-4 \
  data.batch_size=16
```

## GPU Allocation

- **sphinx**: 1-8 GPUs with `--gres=gpu:N`
- **miso**: Must use all 8 GPUs (`--gpus-per-task=8 --nproc_per_node=8`)

## Registry Pattern (Extensibility)

The codebase uses registries for easy ablation of architectural components:

**Optimizers** (`lingua/optim.py`):
```python
from lingua.optim import OPTIMIZER_REGISTRY, register_optimizer
# Available: "adamw", "sgd"
# Add custom: register_optimizer("my_opt", my_builder_fn)
```

**Schedulers** (`lingua/optim.py`):
```python
from lingua.optim import SCHEDULER_REGISTRY, register_scheduler
# Available: "constant", "linear", "cosine", "inv_sqrt", "wsd"
```

**Positional Embeddings** (`lingua/transformer.py`):
```python
from lingua.transformer import POSEMBED_REGISTRY, register_posembed
# Available: "rope", "none"
```

**Normalization** (`lingua/transformer.py`):
```python
from lingua.transformer import NORM_REGISTRY, register_norm
# Available: "rmsnorm", "layernorm"
```

**Activations** (`lingua/transformer.py`):
```python
from lingua.transformer import ACTIVATION_REGISTRY, register_activation
# Available: "silu", "gelu", "relu", "tanh", "sigmoid"
```

## Config Example

```yaml
dump_dir: /juice5b/scr5b/YOUR_USER/out
name: my_run
steps: 500

optim:
  optimizer: adamw  # from OPTIMIZER_REGISTRY
  scheduler: cosine  # from SCHEDULER_REGISTRY
  lr: 3.0e-4
  warmup: 100

model:
  dim: 512
  n_layers: 8
  n_heads: 8
  norm_type: rmsnorm  # from NORM_REGISTRY
  activation: silu     # from ACTIVATION_REGISTRY
  pos_embed_type: rope # from POSEMBED_REGISTRY

distributed:
  fsdp_type: no_shard
  compile: false
  model_dtype: bf16

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
  eval:
    every: 5000

eval: null
```

## Throughput (Measured)

| Setup | Tokens/Step | Throughput | 500 Steps |
|-------|-------------|------------|-----------|
| 1x A100 | 16K | ~140K tok/s | ~58s |
| 8x H200 | 131K | ~460K tok/s | ~142s |
