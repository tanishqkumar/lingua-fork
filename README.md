# lingua-fork

Multi-cluster LLM pretraining framework. Fork of [Meta's lingua](https://github.com/facebookresearch/lingua).

## Supported Clusters

| Cluster | GPU | Max Nodes |
|---------|-----|-----------|
| Together research-secure | 8x H100 | 4 |
| Together mk-turbo | 8x H100 | 1 |
| Stanford sphinx | 8x A100 | 2 |
| Stanford miso | 8x H100 | 1 |

## Quick Start

```bash
# Clone locally
git clone https://github.com/tanishqkumar/lingua-fork.git
cd lingua-fork

# Sync to cluster
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
    . research-secure-hn:~/lingua-fork/

# SSH and run
ssh research-secure-hn
cd ~/lingua-fork
uv sync
sbatch scripts/jobs/together_secure.sh
```

## Running Training

**Interactive (debug):**
```bash
srun --partition=batch --gpus-per-node=1 --time=01:00:00 --mem=48G --pty bash
cd ~/lingua-fork
uv run torchrun --nproc_per_node=1 --standalone -m apps.main.train config=apps/main/configs/debug.yaml
```

**Batch job:**
```bash
export EXPERIMENT="my_experiment"
export CONFIG="apps/main/configs/base.yaml"
sbatch scripts/jobs/together_secure.sh
```

## Config Notes

- Default configs are tuned for **H100/H200**
- On **A100** (sphinx): batch_size is automatically halved and grad_acc_steps doubled
- Paths are auto-resolved based on detected cluster

## WandB Integration

**Project**: `tk07-stanford-university/tanishqbot`

Runs are automatically named and tagged for easy filtering:
- **Run Name**: `{experiment}_{cluster}_{job_id}` (e.g., `lr_sweep_sphinx_12345`)
- **Tags**: `[provider, cluster, gpu_type, experiment, job_id]`
- **Group**: `{experiment}/{cluster}`

## Registry Pattern

Easily swap components via config:

```yaml
optim:
  optimizer: adamw   # adamw, sgd
  scheduler: cosine  # constant, linear, cosine, inv_sqrt, wsd

model:
  norm_type: rmsnorm    # rmsnorm, layernorm
  activation: silu      # silu, gelu, relu
  pos_embed_type: rope  # rope, none
```

## Known Issues

**xformers 0.0.31**: `AttributeError: 'xformers' object has no attribute 'efficient_attention_forward_cutlass'` - commented out in `lingua/distributed.py`, not needed for training.

## See Also

- [CLAUDE.md](CLAUDE.md) - Detailed cluster setup and SSH credentials
