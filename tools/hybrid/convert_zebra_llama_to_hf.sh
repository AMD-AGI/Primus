#!/bin/bash
python tools/hybrid/convert_zebra_llama_to_hf.py \
    --checkpoint-path output/zebra_mamba_1B_hybrid-pretrain/iter_0020000 \
    --output-dir output/zebra_mamba_1B_hybrid_hf_iter_0020000 \
