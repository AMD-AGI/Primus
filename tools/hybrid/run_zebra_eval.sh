#!/bin/bash
# lm_harness_eval.py takes --model_path / --tokenizer / --dtype; it does not
# accept lm-eval's --model / --model_args. The converted checkpoint carries no
# tokenizer files, so --tokenizer must point at the base model.
python3 tools/hybrid/lm_harness_eval.py \
    --model_path output/zebra_llama_kda_1B_hf_iter_0028000 \
    --tokenizer meta-llama/Llama-3.2-1B \
    --dtype bfloat16 \
    --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa,race,openbookqa \
    --batch_size 32
