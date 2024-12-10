#!/bin/bash

#SBATCH --account=nlp
#SBATCH --job-name=lingua-LM1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=sphinx
#SBATCH --mem=64G
#SBATCH --output=out/log.out
#SBATCH --error=out/log.err
#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
#SBATCH --distribution=block
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --constraint=ampere

# Mimic the effect of "conda init", which doesn't work for scripts
eval "$($CONDA_PATH shell.bash hook)"
source activate $CONDA_ENV_PATH

# Parse the dump_dir and set it dynamically.
if [[ -z "$DUMP_DIR" ]]; then   
    export OUT_PATH=$(grep '^[[:space:]]*dump_dir:' $CONFIG_FILE | awk -F': ' '{print $2}' | tr -d '"' | tr -d "'")
    export DUMP_DIR=$BASE_DIR/$OUT_PATH
fi

mkdir -p $DUMP_DIR

export OMP_NUM_THREADS=1
export LAUNCH_WITH="SBATCH"
srun -o $DUMP_DIR/log.out -e $DUMP_DIR/log.err -n 1 -N 1 python -u -m apps.main.train config=$CONFIG_FILE $EXTRA_ARGS
