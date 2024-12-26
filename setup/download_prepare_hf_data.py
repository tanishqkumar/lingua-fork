# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import time
import subprocess
import requests
from huggingface_hub import snapshot_download

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def download_dataset(repo_id, local_dir, allow_patterns):
    print(f"Downloading dataset from {repo_id}...")
    max_retries = 5
    retry_delay = 10  # seconds
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                resume_download=True,
                max_workers=16, # Don't hesitate to increase this number to lower the download time
            )
            break
        except requests.exceptions.ReadTimeout:
            if attempt < max_retries - 1:
                print(f"Timeout occurred. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise
    print(f"Dataset downloaded to {local_dir}")


def parquet_to_jsonl(dataset, work_dir, src_dir, tgt_dir, ntasks=64):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                src_dir,
                file_progress=True,
                doc_progress=True,
                glob_pattern="**/*.parquet",
            ),
            JsonlWriter(
                tgt_dir,
                output_filename=dataset + ".chunk.${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=ntasks,
        logging_dir=os.path.join(work_dir, "datatrove"),
    )
    pipeline_exec.run()


def setup_terashuf(work_dir):
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir


def main(dataset, memory, data_dir, final_data_dir=None, clear_work_dir=False, seed=42, nchunks=32):
    # Configuration
    repo_id = {
        "fineweb_edu": "HuggingFaceFW/fineweb-edu",
        "fineweb_edu_10bt": "HuggingFaceFW/fineweb-edu",
        "dclm_baseline_1.0": "mlfoundations/dclm-baseline-1.0",
        "dclm_baseline_1.0_10prct": "mlfoundations/dclm-baseline-1.0",
        "dclm_pool_1b_1x": "mlfoundations/dclm-pool-1b-1x",
        "cosmopedia_v2": "HuggingFaceTB/smollm-corpus",
        "python_edu": "HuggingFaceTB/smollm-corpus",
    }[dataset]
    src_dir = f"{data_dir}/{dataset}"
    out_dir = f"{src_dir}_shuffled"
    os.makedirs(out_dir, exist_ok=True)
    work_dir = src_dir  # Directory of this Python file
    prefix = f"{dataset}.chunk."
    orig_extension = {
        "fineweb_edu": ".jsonl",
        "fineweb_edu_10bt": ".jsonl",
        "dclm_baseline_1.0": ".jsonl.zst",
        "dclm_baseline_1.0_10prct": ".jsonl.zst",
        "dclm_pool_1b_1x": ".jsonl.zst",
        "cosmopedia_v2": ".jsonl",
        "python_edu": ".jsonl",
    }[dataset]
    cat_command = {
        "fineweb_edu": "cat {}",
        "fineweb_edu_10bt": "cat {}",
        "dclm_baseline_1.0": "zstdcat {} && echo",
        "dclm_baseline_1.0_10prct": "zstdcat {} && echo",
        "dclm_pool_1b_1x": "zstdcat {} && echo",
        "cosmopedia_v2": "cat {}",
        "python_edu": "cat {}",
    }[dataset]
    allow_patterns = {
        "fineweb_edu": None,
        "fineweb_edu_10bt": "sample/10BT/*",
        "dclm_baseline_1.0": "*.jsonl.zst",
        "dclm_baseline_1.0_10prct": "global-shard_01_of_10/*.jsonl.zst",
        "dclm_pool_1b_1x": "*.jsonl.zst",
        "cosmopedia_v2": "cosmopedia-v2/*",
        "python_edu": "python-edu/*",
    }[dataset]
    suffix = ".jsonl"
    k_validation = 10000  # Number of lines to take from each chunk for validation

    # Setup terashuf
    terashuf_dir = setup_terashuf(work_dir)

    # Download dataset
    download_dataset(repo_id, src_dir, allow_patterns)

    if "fineweb" in dataset or "cosmopedia" in dataset:
        parquet_to_jsonl(dataset, work_dir, src_dir, src_dir)

    # Set up environment variables
    os.environ["MEMORY"] = f"{memory}"
    os.environ["SEED"] = f"{seed}"

    # Run the original shuffling and splitting command
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    run_command(
        f"ulimit -n 100000 && "
        f"find {src_dir} -type f -name '*{orig_extension}' -print0 | xargs -0 -I {{}} sh -c '{cat_command}' | {terashuf_executable} | "
        f"split -n r/{nchunks} -d --suffix-length 2 --additional-suffix {suffix} - {out_dir}/{prefix}"
        "; trap 'echo \"Caught signal 13, exiting with code 1\"; exit 1' PIPE;"
    )

    # Create validation set and remove lines from chunks
    validation_file = f"{out_dir}/{dataset}.val{suffix}"
    for i in range(nchunks):
        chunk_file = f"{out_dir}/{prefix}{i:02d}{suffix}"
        run_command(f"head -n {k_validation} {chunk_file} >> {validation_file}")
        run_command(f"sed -i '1,{k_validation}d' {chunk_file}")

    print(f"Processing completed in work directory {work_dir}")

    # Handle final data directory transfer if specified
    if final_data_dir:
        final_out_dir = os.path.join(final_data_dir, f"{dataset}_shuffled")
        
        print(f"Copying shuffled data to final destination: {final_out_dir}")
        os.makedirs(final_data_dir, exist_ok=True)
        
        # Use rsync with these flags:
        # -a: archive mode (preserves permissions, timestamps, etc.)
        # -v: verbose
        # -P: shows progress and allows partial transfer resumption
        # --delete: remove files in dest that aren't in source
        run_command(f"rsync -avP --delete {out_dir}/ {final_out_dir}")
        
        # Clear work directories if requested
        if clear_work_dir:
            print("Clearing work directories...")
            run_command(f"rm -rf {src_dir}")
            run_command(f"rm -rf {out_dir}")
            
    print("All tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("memory", type=float, default=8)
    parser.add_argument("--tmp_dir", type=str, default="/tmp")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nchunks", type=int, default=32)
    parser.add_argument(
        "--final_data_dir", 
        type=str, 
        default=None, 
        help="Final destination for the shuffled data, e.g., a slower, large disk."
    )
    parser.add_argument(
        "--clear_work_dir_after_transfer_to_final", 
        action="store_true",
        help="Clear work directories after transferring to final location."
    )

    args = parser.parse_args()

    os.environ["TMPDIR"] = args.tmp_dir
    main(
        args.dataset, 
        args.memory, 
        args.data_dir, 
        args.final_data_dir, 
        args.clear_work_dir_after_transfer_to_final,
        args.seed,
        args.nchunks
    )