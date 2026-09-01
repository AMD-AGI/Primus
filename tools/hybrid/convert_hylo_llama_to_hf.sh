#!/bin/bash
python tools/hybrid/convert_hylo_llama_to_hf.py \
    --checkpoint-path output/hylo_llama_mamba_1B_BF16-pretrain/iter_0020000 \
    --output-dir output/hylo_mamba_1B_hybrid_hf_iter_0020000 \
