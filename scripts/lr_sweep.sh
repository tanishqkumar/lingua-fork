#!/bin/bash

# Setup common environment variables
export BASE_DIR=/juice5/scr5/thashim/lingua-test/lingua-fork
export CONFIG_FILE=apps/main/configs/llama_280M_8H200.yaml
export EMAIL=thashim@stanford.edu

# Learning rates to sweep
learning_rates=(5e-3 1e-2 5e-2)

# Launch jobs for each learning rate
for lr in "${learning_rates[@]}"; do
    # Modify the dump directory to include the learning rate
    export EXTRA_ARGS="dump_dir=out/llama_280M_8H200_lr${lr} \
                        name=llama_280M_8H200_lr${lr} \
                        logging.wandb.name=llama_280M_8H200_lr${lr} \
                        optim.lr=${lr}"
    export DUMP_DIR=$BASE_DIR/out/llama_280M_8H200_lr${lr}
    
    sbatch --mail-user=$EMAIL \
           --export=ALL,BASE_DIR,CONFIG_FILE,EXTRA_ARGS,DUMP_DIR \
           apps/main/configs/miso_8.slurm
    sleep 1
done