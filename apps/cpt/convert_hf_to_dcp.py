import argparse
import re
from pathlib import Path
import os

import torch
from torch.distributed.checkpoint.format_utils import torch_save_to_dcp
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download


def convert_hf_to_dcp(hf_model_name: str, output_dir: str, dtype: str = "float32"):
    """
    Downloads a HuggingFace model and converts it to DCP format for Lingua.
    Args:
        hf_model_name: Name/path of the HuggingFace model
        output_dir: Directory to save the converted checkpoint
        dtype: Data type to load the model in ("float16", "bfloat16", or "float32")
    """
    
    hf_hub_download(repo_id=hf_model_name, filename="original/consolidated.00.pth", local_dir=output_dir)

    consolidated_path = os.path.join(output_dir, "consolidated/consolidated.pth")
    os.makedirs(os.path.dirname(consolidated_path), exist_ok=True)
    os.rename(os.path.join(output_dir, "original/consolidated.00.pth"), consolidated_path)
    os.rmdir(os.path.join(output_dir, "original"))

    # Convert to DCP format
    print("Converting to DCP format...")
    torch_save_to_dcp(consolidated_path, output_dir)

    # Write the tokenizer to the output directory
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to DCP format"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model name/path"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for DCP checkpoint"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model loading",
    )

    args = parser.parse_args()
    convert_hf_to_dcp(args.model, args.output, args.dtype)