#!/bin/bash

# Setup common environment variables
export BASE_DIR=/path/to/lingua-fork
export CONFIG_FILE=apps/main/configs/llama_1B_8H100_cpt.yaml

# Email user
export EMAIL=nband@stanford.edu

# Learning rates to sweep
learning_rates=(1e-5 5e-5 1e-4 5e-4 1e-3 5e-3)

# Launch jobs for each learning rate
for lr in "${learning_rates[@]}"; do
    # Modify the dump directory to include the learning rate
    export EXTRA_ARGS="dump_dir=out/llama_1B_8H100_cpt_lr${lr} \
                        name=llama_1B_8H100_cpt_lr${lr} \
                        logging.wandb.name=llama_1B_8H100_cpt_lr${lr} \
                        optim.lr=${lr}"
    export DUMP_DIR=$BASE_DIR/out/llama_1B_8H100_cpt_lr${lr}
    
    sbatch --mail-user=$EMAIL \
           --export=ALL,BASE_DIR,CONFIG_FILE,EXTRA_ARGS,DUMP_DIR \
           apps/main/configs/miso_8.slurm
    sleep 1
done