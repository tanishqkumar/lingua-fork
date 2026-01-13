# CLAUDE.md - lingua-fork Multi-Cluster Guide

## Quick Reference - Cluster Access

| Cluster | SSH Host | Environment | Data Path | Max Nodes | GPU |
|---------|----------|-------------|-----------|-----------|-----|
| **research-secure** | `research-secure-hn` | `uv` | `/data/tkumar/datasets` | 4 | 8x H100 |
| **mk-turbo** | `mk-turbo-hn` | `uv` | `/data/tkumar/datasets` | 1 | 8x H100 |
| **sphinx** | `tanishq@sc.stanford.edu` | `uv` | `/juice5/scr5/nlp/data/huggingface/lingua-data` | 2 | 8x A100 |
| **miso** | `tanishq@sc.stanford.edu` | `uv` | `/juice5/scr5/nlp/data/huggingface/lingua-data` | 1 | 8x H100 |

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
Partitions: sphinx (A100s), miso (H100s)
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

## Smart Launcher (Recommended)

The smart launcher checks for idle nodes across all clusters and submits to the first available one:

```bash
# Check availability and submit to best cluster
python scripts/launch.py -c apps/main/configs/debug.yaml -e my_experiment

# Just check availability (dry run)
python scripts/launch.py --dry-run

# Force a specific cluster
python scripts/launch.py --cluster sphinx -c apps/main/configs/debug.yaml

# Pass extra training args
python scripts/launch.py -c apps/main/configs/debug.yaml -e test -x "steps=500 model.dim=512"
```

Output example:
```
Checking cluster availability (priority order)...
------------------------------------------------------------
  research-secure      1 idle <-- SELECTED
  mk-turbo             7 idle
  sphinx               busy
  miso                 busy
------------------------------------------------------------
Submitting to research-secure...
Success: Submitted batch job 12345
```

---

## Manual Workflow: Local Edit → Cluster Run

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

### 4. Launch training
```bash
# Together AI
ssh research-secure-hn
sbatch scripts/jobs/together_secure.sh

# Stanford
ssh tanishq@sc.stanford.edu
sbatch scripts/jobs/stanford_sphinx.sh
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

**Tokenizers** (`lingua/tokenizer.py`):
```yaml
tokenizer:
  name: cl100k  # RECOMMENDED: Fast tiktoken BPE (GPT-4 style)
  # Other options:
  # name: bytes           # 258 vocab, unrealistically low loss (~1.0-1.5)
  # name: hf              # HuggingFace (slow for Llama)
  #   path: meta-llama/Llama-3.2-1B
  # name: tiktoken        # Custom tiktoken file
  #   path: path/to/file.tiktoken
  # name: sp              # SentencePiece
  #   path: path/to/model.model
```

| Tokenizer | Vocab Size | Expected Loss | Speed | Notes |
|-----------|------------|---------------|-------|-------|
| **cl100k** | 100,277 | 3.0-4.0 | Fast | GPT-4 style, recommended |
| bytes | 258 | 1.0-1.5 | Fast | Misleadingly low loss |
| hf (Llama) | 128,256 | 3.0-4.0 | Slow | use_fast=False by default |

**Important**: Always set `model.vocab_size` to match your tokenizer!
- cl100k: `vocab_size: 100277`
- bytes: `vocab_size: 258` (or omit, auto-detected)
- Llama 3: `vocab_size: 128256`

---

## Default Config (`apps/main/configs/default.yaml`)

**Optimized for fast iteration**: 34M params, 1B tokens, ~11 min on 8x H100.

```yaml
dump_base: null  # Auto-resolved by cluster detection
name: "default"
steps: 4000  # 1B tokens / 256k tokens per step

optim:
  optimizer: adamw
  scheduler: cosine
  lr: 3e-3  # Optimal from LR sweep
  warmup: 100

model:
  dim: 832
  n_layers: 4
  n_heads: 13  # head_dim = 64
  vocab_size: 100277  # cl100k_base

distributed:
  fsdp_type: no_shard  # faster for small models
  compile: true  # torch.compile for speed
  model_dtype: bf16

data:
  batch_size: 32  # per GPU
  seq_len: 1024
  tokenizer:
    name: cl100k
```

**Performance** (8x H100):
- Throughput: ~1.5M tok/s (183k/GPU)
- MFU: ~13% (small model = memory-bandwidth bound)
- Time: ~11 min for 1B tokens

---

## Dataset Locations

**FineWeb-Edu 10BT** (train: 9.67M samples, val: 10K samples):

| Cluster | Train Path | Val Path |
|---------|------------|----------|
| Together (both) | `/data/tkumar/datasets/fineweb_edu_10bt_shuffled/fineweb_edu_10bt.chunk.00.jsonl` | `.../fineweb_edu_10bt.val.jsonl` |
| Stanford (both) | `/juice5/scr5/nlp/data/huggingface/lingua-data/fineweb_edu_10bt_shuffled/fineweb_edu_10bt.chunk.00.jsonl` | `.../fineweb_edu_10bt.val.jsonl` |

**Note**: Validation files are identical (MD5: `76da820d4118c0c6606a4d144667d540`) across all clusters.

---

## Throughput Reference (default.yaml: 34M model, 256k batch)

| Cluster | GPUs | Throughput | 1B tokens |
|---------|------|------------|-----------|
| mk-turbo | 8x H100 | ~1.5M tok/s | ~11 min |
| research-secure | 8x H100 | ~1.5M tok/s | ~11 min |
| sphinx | 8x A100 | ~1.0M tok/s | ~17 min |
| miso | 8x H200 | ~1.8M tok/s | ~9 min |

**Bad nodes** (excluded in job scripts):
- mk-turbo-02: Slow performance (~3x slower)
- mk-turbo-08: GPU failures

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
