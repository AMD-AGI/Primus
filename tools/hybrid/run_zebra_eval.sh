#!/bin/bash
python3 tools/hybrid/lm_harness_eval.py --model zebra_llama \
    --model_args pretrained=output/zebra_mamba_1B_hybrid_hf_iter_0200000,dtype=bfloat16  \
    --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa,race,openbookqa \
    --batch_size 32
