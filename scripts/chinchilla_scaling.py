#!/usr/bin/env python3
import os
import subprocess
import time

# Setup common environment variables
base_dir = "/juice5/scr5/thashim/lingua-test/lingua-fork"
config_file = "apps/main/configs/llama_8H200_chinchilla.yaml"
email = "thashim@stanford.edu"

# Learning rates to sweep
learning_rates = ["3e-3", "1e-3", "5e-4"] # LRs taken rougly from https://huggingface.co/EleutherAI/pythia-6.9b
model_depths = [24, 16, 12, 8, 4]
model_aspect = 96 # 96 dim per depth https://arxiv.org/pdf/2001.08361. we dont worry about powers of two stuff since we'll torch compile it and it'll pad out the matrices..
model_head_dim = 32 # same cite as above.
replicate_count = 2

#flop_target = "52.41e18" # 4 hours.
flop_target = "26.205e18" # 2 hours.

# Launch jobs for each learning rate
for replicate in range(replicate_count):    
    for lr in learning_rates:
        for depth in model_depths:
            batch_target = 16
            # Modify the dump directory to include the learning rate
            extra_args = (f"name=llama_8H200_chinchilla_flops_{flop_target}_lr{lr}_depth{depth}_replicate{replicate} "
                 f"optim.lr={lr} "
                 f"model.dim={model_aspect * depth} "
                 f"model.n_heads={model_aspect * depth // model_head_dim} "
                 f"model.n_layers={depth} "
                 f"data.batch_size={batch_target} "
                 f"total_flops={flop_target} "
                 f"seed={replicate}")
    
            # Set environment variables for the subprocess
            env = os.environ.copy()
            env.update({
                "BASE_DIR": base_dir,
                "CONFIG_FILE": config_file,
                "EXTRA_ARGS": extra_args,
            })
            
            # Launch sbatch command
            cmd = [
                "sbatch",
                f"--mail-user={email}",
                "--export=ALL,BASE_DIR,CONFIG_FILE,EXTRA_ARGS",
                "--time=4:00:00",
                "apps/main/configs/miso_8_silent.slurm"
            ]
            
            subprocess.run(cmd, env=env)
            time.sleep(1)