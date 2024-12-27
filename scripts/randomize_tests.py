#!/usr/bin/env python3
import os
import subprocess
import time

def launch_job(extra_args, is_silent=True):
    # Set environment variables for the subprocess
    env = os.environ.copy()
    env.update({
        "BASE_DIR": base_dir,
        "CONDA_PATH": conda_path,
        "CONDA_ENV_PATH": conda_env_path,
        "CONFIG_FILE": config_file,
        "EXTRA_ARGS": extra_args,
    })
    
    if is_silent:
        slurm_config = "apps/main/configs/miso_8_silent.slurm"
    else:
        slurm_config = "apps/main/configs/miso_8.slurm"

    # Launch sbatch command
    cmd = [
        "sbatch",
        f"--mail-user={email}",
        "--export=ALL,BASE_DIR,CONDA_PATH,CONDA_ENV_PATH,CONFIG_FILE,EXTRA_ARGS",
        "--time=4:00:00",
        slurm_config
    ]

    
    subprocess.run(cmd, env=env)
    time.sleep(1)

# Setup common environment variables
base_dir = "/juice5/scr5/thashim/lingua-test/lingua-fork"
conda_path = "/juice5/scr5/thashim/miniconda3/bin/conda"
conda_env_path = "/juice5/scr5/thashim/miniconda3/envs/lingua_241127"
config_file = "apps/main/configs/llama_8H200_chinchilla.yaml"
email = "thashim@stanford.edu"


#python -u -m apps.main.train config=apps/main/configs/llama_8H200_chinchilla.yaml name=data_randomization_tests_v0 data.randomize_rate=0.5 logging.wandb.project=data_randomization_tests logging.wandb.group=data_randomization_tests_v0 total_flops=26.205e18 seed=0 model.seed=0
project_name = "data_randomization_tests_v1"
flop_target = "26.205e18"
rand_rate_list = [0.0, 0.0001,0.0005, 0.001,0.005, 0.01, 0.02, 0.035, 0.05, 0.1, 0.2]
params_sets = [
    [f"{random}_rd",{"data.randomize_rate": random,"eval.wipe_ckpt": "true"}] for random in rand_rate_list
]
replicate_start = 0
replicate_count = 3
debug_mode = False

# Launch jobs for each learning rate
for replicate in range(replicate_count):   
    replicate_id = replicate + replicate_start
    for params_set in params_sets:
        override_strings = []
        for key, value in params_set[1].items():
            override_strings.append(f"{key}={value}")
        override_strings = " ".join(override_strings)
        param_set_name = params_set[0]
        # Modify the dump directory to include the learning rate
        extra_args = (f"name={param_set_name}_rep{replicate_id} "
            f"logging.wandb.project={project_name} "
            f"logging.wandb.group={param_set_name} "
            f"{override_strings} "
            f"total_flops={flop_target} "
            f"seed={replicate_id} "
            f"model.seed={replicate_id}")
        
        # make the last job and any debug jobs non-silent
        if debug_mode or(replicate == replicate_count-1 and params_set == params_sets[-1]):
            launch_job(extra_args, is_silent=False)
        else:
            launch_job(extra_args, is_silent=True)
        if debug_mode:
            exit(0)

exit(0)

# Launch 4 baseline jobs
for replicate in range(3):
    extra_args = (f"name=baseline_rep{replicate} "
                f"logging.wandb.project={project_name} "
                f"total_flops={flop_target} "
                f"seed={replicate} "
                f"model.seed={replicate}")
    launch_job(extra_args)


            