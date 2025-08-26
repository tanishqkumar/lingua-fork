#!/bin/bash

# Default values
DTYPE="float32"
MODELS=(meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B meta-llama/Llama-3.1-8B)

for MODEL in "${MODELS[@]}"; do
    uv run python apps/cpt/convert_hf_to_dcp.py \
        --model "$MODEL" \
        --output "./out/$MODEL" \
        --dtype "$DTYPE"
done