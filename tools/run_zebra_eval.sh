python3 tools/lm_harness_eval.py --model zebra_llama \
    --model_args pretrained=output/zebra_llama_1B_hf_iter_0200000,dtype=bfloat16  \
    --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa,race,openbookqa \
    --batch_size 32
