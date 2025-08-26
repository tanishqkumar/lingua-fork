#!/bin/bash

#SBATCH --account=miso
#SBATCH --job-name=LM8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=2438G
#SBATCH --partition=miso
#SBATCH --output=out/log.out
#SBATCH --error=out/log.err
#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
#SBATCH --distribution=block
#SBATCH --mail-type=NONE

mkdir -p out/logs

export OMP_NUM_THREADS=1
export LAUNCH_WITH="SBATCH"
srun -o out/logs/log_${SLURM_JOB_ID}.out -e out/logs/log_${SLURM_JOB_ID}.err -n 1 -N 1 uv run torchrun --nproc_per_node=8 -m apps.main.train config=$CONFIG_FILE $EXTRA_ARGS
