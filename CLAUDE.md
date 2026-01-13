# CLAUDE.md - lingua-fork Multi-Cluster Guide

## Quick Reference - Cluster Access

| Cluster | SSH Host | Environment | Data Path | Max Nodes | GPU |
|---------|----------|-------------|-----------|-----------|-----|
| **research-secure** | `research-secure-hn` | `uv` | `/data/tkumar/datasets` | 4 | 8x H100 |
| **mk-turbo** | `mk-turbo-hn` | `uv` | `/data/tkumar/datasets` | 4 | 8x H100 |
| **sphinx** | `tanishq@sc.stanford.edu` | `uv` | `/juice5/scr5/nlp/data/huggingface/lingua-data` | 2 | 8x A100 |
| **miso** | `tanishq@sc.stanford.edu` | `uv` | `/juice5/scr5/nlp/data/huggingface/lingua-data` | 1 | 8x H200 |

**Priority Order**: research-secure > mk-turbo > sphinx > miso

**Note on configs**: Default configs are tuned for H100/H200. On A100 (sphinx), batch_size is automatically halved and grad_acc_steps doubled to maintain equivalent effective batch size with lower memory.

---

## SSH Setup

### Together AI (research-secure, mk-turbo)
```bash
# ~/.ssh/config
Host research-secure-hn
    HostName research-secure-hn.cloud.together.ai
    User tkumar
    IdentityFile ~/.ssh/id_rsa-together

Host mk-turbo-hn
    HostName mk-turbo-hn.cloud.together.ai
    User tkumar
    IdentityFile ~/.ssh/id_rsa-together
```

### Stanford (sphinx, miso)
```
Host: sc.stanford.edu
User: tanishq
Password: december1972
Partitions: sphinx (A100s), miso (H200s)
```

---

## Environment Setup (All Clusters)

**All clusters use `uv` for reproducible package management.**

```bash
# First time setup (on any cluster)
cd ~/lingua-fork
uv sync  # Creates .venv and installs from uv.lock

# Running commands
uv run torchrun ...  # Uses the synced environment
```

### Data Paths by Cluster

| Cluster | Data Root | Output Dir |
|---------|-----------|------------|
| Together (both) | `/data/tkumar/datasets` | `/data/tkumar/lingua-out` |
| Stanford (both) | `/juice5/scr5/nlp/data/huggingface/lingua-data` | `/juice5b/scr5b/tanishq/lingua-out` |

**Note**: On Together clusters, `/data` is only accessible from compute nodes, not the login node.

### Known Issues

**xformers 0.0.31 AttributeError**: When using xformers with activation recomputation:
```
AttributeError: '_OpNamespace' 'xformers' object has no attribute 'efficient_attention_forward_cutlass'
```
This is commented out in `lingua/distributed.py` - we don't use xformers activation recomputation.

---

## Workflow: Local Edit → Cluster Run

### 1. Clone locally (ground truth)
```bash
git clone https://github.com/tanishqkumar/lingua-fork.git
cd lingua-fork
```

### 2. Make edits locally

### 3. Rsync to target cluster
```bash
# To Together AI
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
    . research-secure-hn:~/lingua-fork/

# To Stanford
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
    . tanishq@sc.stanford.edu:~/lingua-fork/
```

### 4. Launch training (priority order)
```bash
# Try research-secure first (max 4 nodes)
ssh research-secure-hn
sbatch scripts/jobs/together_secure.sh

# If busy, try mk-turbo (max 1 node)
ssh mk-turbo-hn
sbatch scripts/jobs/together_turbo.sh

# If Together is full, try Stanford sphinx (max 2 nodes)
ssh tanishq@sc.stanford.edu
sbatch scripts/jobs/stanford_sphinx.sh

# Last resort: Stanford miso (max 2 nodes)
sbatch scripts/jobs/stanford_miso.sh
```

---

## Running Training

### Interactive (debug, see loss live)
```bash
# Together AI - research-secure
ssh research-secure-hn
srun --partition=batch --gpus-per-node=1 --time=01:00:00 --mem=48G --pty bash
cd ~/lingua-fork
uv run torchrun --nproc_per_node=1 --standalone -m apps.main.train config=apps/main/configs/debug.yaml

# Stanford - sphinx
ssh tanishq@sc.stanford.edu  # password: december1972
srun --account=nlp --partition=sphinx --gres=gpu:1 --time=01:00:00 --mem=48G --pty bash
cd ~/lingua-fork
uv run torchrun --nproc_per_node=1 --standalone -m apps.main.train config=apps/main/configs/debug.yaml
```

### Batch Jobs (with WandB tracking)
```bash
# Set experiment name for WandB grouping
export EXPERIMENT="lr_sweep"
export CONFIG="apps/main/configs/base.yaml"
export EXTRA_ARGS="steps=1000 optim.lr=1e-4"

# Submit to appropriate cluster
sbatch scripts/jobs/together_secure.sh   # Job name: ssd
sbatch scripts/jobs/together_turbo.sh    # Job name: ssd
sbatch scripts/jobs/stanford_sphinx.sh   # Job name: lingua
sbatch scripts/jobs/stanford_miso.sh     # Job name: lingua
```

### Config Overrides (OmegaConf dot notation)
```bash
torchrun --nproc_per_node=1 --standalone -m apps.main.train \
    config=apps/main/configs/debug.yaml \
    steps=1000 \
    model.dim=768 \
    optim.lr=1e-4 \
    data.batch_size=16
```

---

## WandB Integration

**Project**: `tk07-stanford-university/tanishqbot`

All runs log to WandB with automatic naming, tagging, and grouping:

- **Run Name**: `{experiment}_{cluster}_{slurm_job_id}` (e.g., `lr_sweep_sphinx_12345`)
- **Tags**: `[provider, cluster, gpu_type, experiment, job_{slurm_job_id}]`
  - Example: `[stanford, sphinx, A100, lr_sweep, job_12345]`
- **Group**: `{experiment}/{cluster}` (e.g., `lr_sweep/sphinx`)

This makes it easy to:
1. Filter by experiment name to see all runs in a sweep
2. Filter by cluster to compare performance across hardware
3. Track back to SLURM job ID for debugging

To disable WandB:
```bash
torchrun ... logging.wandb=null
```

---

## Registry Pattern (Extensibility)

The codebase uses registries for easy ablation:

**Optimizers** (`lingua/optim.py`):
```python
from lingua.optim import OPTIMIZER_REGISTRY, register_optimizer
# Available: "adamw", "sgd"
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

---

## Config Example

```yaml
dump_base: null  # Auto-resolved by cluster detection
name: my_experiment
steps: 1000

optim:
  optimizer: adamw
  scheduler: cosine
  lr: 3e-4
  warmup: 100

model:
  dim: 768
  n_layers: 12
  n_heads: 12
  norm_type: rmsnorm
  activation: silu
  pos_embed_type: rope

distributed:
  fsdp_type: full_shard
  compile: false
  model_dtype: bf16

data:
  root_dir: null  # Auto-resolved by cluster detection
  sources:
    fineweb_edu_10bt_shuffled: 100.0
  batch_size: 8
  seq_len: 2048
  tokenizer:
    name: bytes

logging:
  freq: 10
  wandb:
    project: tanishqbot
```

---

## Dataset Locations

**FineWeb-Edu 10BT** (train: 9.67M samples, val: 10K samples):

| Cluster | Train Path | Val Path |
|---------|------------|----------|
| Together (both) | `/data/tkumar/datasets/fineweb_edu_10bt_shuffled/fineweb_edu_10bt.chunk.00.jsonl` | `.../fineweb_edu_10bt.val.jsonl` |
| Stanford (both) | `/juice5/scr5/nlp/data/huggingface/lingua-data/fineweb_edu_10bt_shuffled/fineweb_edu_10bt.chunk.00.jsonl` | `.../fineweb_edu_10bt.val.jsonl` |

**Note**: Validation files are identical (MD5: `76da820d4118c0c6606a4d144667d540`) across all clusters.

---

## Throughput Reference

| Cluster | GPUs | Tokens/Step | Throughput | 500 Steps |
|---------|------|-------------|------------|-----------|
| research-secure | 1x H100 | 16K | ~180K tok/s | ~45s |
| mk-turbo | 1x H100 | 16K | ~180K tok/s | ~45s |
| sphinx | 1x A100 | 16K | ~140K tok/s | ~58s |
| miso | 8x H200 | 131K | ~460K tok/s | ~142s |

---

## Troubleshooting

**"Could not detect cluster"**: Set `LINGUA_CLUSTER` environment variable:
```bash
export LINGUA_CLUSTER=research-secure  # or mk-turbo, sphinx, miso
```

**mk-turbo /data not found**: The `/data` filesystem is only mounted on compute nodes, not the login node. Use `srun` to access it.

**mk-turbo disk full (58GB root partition)**: The login node has a small root partition. Always symlink .venv to /data:
```bash
mkdir -p /data/tkumar/lingua-venv
ln -sf /data/tkumar/lingua-venv .venv
uv sync
```

**FSDP full_shard RuntimeError** (`get_group_info: no group info`): PyTorch 2.7.1 has a bug with `full_shard` FSDP. Use `fsdp_type: no_shard` in configs.

**Checkpoint spam even with every=-1**: Fixed in train.py - Python `x % -1 == 0` is always True. The `every_n_steps` function now returns False for freq <= 0.

**wandb "api_key not configured (no-tty)"**: Job scripts need:
```bash
export WANDB_API_KEY="your_api_key_here"
```

**Stanford auth**: Use password `december1972` for tanishq@sc.stanford.edu

---

## Logging Configuration

**Training loss**: Logged every `logging.freq` steps as `loss/out`

**Validation loss**: Computed every `logging.val_loss_every` steps (default: 100) over `logging.val_loss_batches` batches (default: 10). Logged as `loss/val`.

**Initial loss**: Computed before training starts at step 0, logged as `loss/init`. This gives you the loss at initialization.

To disable validation loss:
```yaml
logging:
  val_loss_every: 0  # 0 disables validation loss
```

To disable checkpointing entirely:
```yaml
checkpoint:
  dump:
    every: -1  # -1 disables checkpoint saving
  eval:
    every: -1  # -1 disables eval checkpoints
```
