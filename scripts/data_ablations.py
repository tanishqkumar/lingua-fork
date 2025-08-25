#!/usr/bin/env python3
import os
import subprocess
import time

def launch_job(extra_args):
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

# Setup common environment variables
base_dir = "/juice5/scr5/thashim/lingua-test/lingua-fork"
config_file = "apps/main/configs/llama_8H200_chinchilla.yaml"
email = "thashim@stanford.edu"

#default hpars
params_sets = [
    #"data.sources": ["cosmopedia_v2_shuffled", "dclm_baseline_1_0_shuffled"]
    #"data.sources": ["dclm_baseline_1_0_shuffled"]
    ['smollm_mix',{"data.sources.cosmopedia_v2_shuffled": 30.0, "data.sources.fineweb_edu_shuffled": 70.0}],
    ['synth70_fine30',{"data.sources.cosmopedia_v2_shuffled": 70.0, "data.sources.fineweb_edu_shuffled": 30.0}],
]
replicate_count = 3

#flop_target = "52.41e18" # 4 hours.
flop_target = "26.205e18" # 2 hours.

# Launch jobs for each learning rate
for replicate in range(replicate_count):    
    for params_set in params_sets:
        override_strings = []
        for key, value in params_set[1].items():
            override_strings.append(f"{key}={value}")
        override_strings = " ".join(override_strings)
        param_set_name = params_set[0]
        # Modify the dump directory to include the learning rate
        extra_args = (f"name=data_ablation_{param_set_name}_rep{replicate} "
            f"logging.wandb.project=2h_data_ablation "
            f"data.sources.fineweb_edu_shuffled=0.0 "
            f"{override_strings} "
            f"total_flops={flop_target} "
            f"seed={replicate} "
            f"model.seed={replicate}")
        launch_job(extra_args)

exit(0)

# Launch 4 baseline jobs
for replicate in range(3):
    extra_args = (f"name=baseline_rep{replicate} "
                f"logging.wandb.project=2h_data_ablation "
                f"total_flops={flop_target} "
                f"seed={replicate} "
                f"model.seed={replicate}")
    launch_job(extra_args)


            