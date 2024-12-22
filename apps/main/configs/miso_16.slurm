#!/bin/bash

#SBATCH --account=miso
#SBATCH --job-name=LM16gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=2438G
#SBATCH --partition=miso
#SBATCH --output=out/log.out
#SBATCH --error=out/log.err
#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
#SBATCH --distribution=block
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --propagate=NONE


# Multinode env vars we need
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo ULIMIT-L: $(ulimit -l)
echo Node IP: $head_node_ip
echo Head Node: $head_node
echo Node list: $nodes


# Mimic the effect of "conda init", which doesn't work for scripts
eval "$($CONDA_PATH shell.bash hook)"
source activate $CONDA_ENV_PATH

mkdir -p out/logs

export OMP_NUM_THREADS=18
export LAUNCH_WITH="SBATCH"

srun -o out/logs/log_${SLURM_JOB_ID}.out -e out/logs/log_${SLURM_JOB_ID}.err  torchrun --nnodes 2 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 --nproc_per_node=8 -m apps.main.train config=$CONFIG_FILE $EXTRA_ARGS
