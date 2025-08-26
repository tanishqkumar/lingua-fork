#!/bin/bash

#SBATCH --account=nlp
#SBATCH --job-name=Lo-LM1gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=sc-loprio
#SBATCH --mem=64G
#SBATCH --output=out/log.out
#SBATCH --error=out/log.err
#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
#SBATCH --distribution=block
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --constraint=[48G|80G|141G]

mkdir -p out/logs

export OMP_NUM_THREADS=1
export LAUNCH_WITH="SBATCH"
srun -o out/logs/log_${SLURM_JOB_ID}.out -e out/logs/log_${SLURM_JOB_ID}.err -n 1 -N 1 uv run python -u -m apps.main.train config=$CONFIG_FILE $EXTRA_ARGS
