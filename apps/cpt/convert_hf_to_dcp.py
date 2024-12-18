import argparse
import re
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import torch_save_to_dcp
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_hf_to_dcp(hf_model_name: str, output_dir: str, dtype: str = "float32"):
    """
    Downloads a HuggingFace model and converts it to DCP format for Lingua.

    Args:
        hf_model_name: Name/path of the HuggingFace model
        output_dir: Directory to save the converted checkpoint
        dtype: Data type to load the model in ("float16", "bfloat16", or "float32")
    """
    print(f"Converting {hf_model_name} to DCP format...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]

    # Download and load the model
    print("Loading model and its config from HuggingFace...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name, torch_dtype=torch_dtype, trust_remote_code=True
    )

    # Create temporary PyTorch checkpoint
    temp_checkpoint = output_path / "temp_checkpoint.pt"
    print(f"Saving temporary checkpoint to {temp_checkpoint}...")
    state_dict = model.state_dict()

    rename_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.o_proj.bias": "layers.{}.attention.wo.bias",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    final_state_dict = {}
    for key, value in state_dict.items():
        if "layers" in key:
            abstract_key = re.sub(r"(\d+)", "{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = rename_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = rename_map[key]

        final_state_dict[new_key] = value

    torch.save(final_state_dict, temp_checkpoint)

    # Convert to DCP format
    print("Converting to DCP format...")
    torch_save_to_dcp(str(temp_checkpoint), str(output_path))

    # Clean up temporary file
    temp_checkpoint.unlink()
    print(f"Conversion complete! DCP checkpoint saved to: {output_path}")

    # Write the tokenizer to the output directory
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    tokenizer.save_pretrained(output_path)


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
