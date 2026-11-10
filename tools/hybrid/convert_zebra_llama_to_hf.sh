#!/bin/bash
# This converter targets the KDA + MLA architecture implemented by
# modeling_zebra_llama.py. Mamba2 and GDN hybrids have their own converters,
# convert_mamba_hybrid_to_fla_hf.py and convert_gdn_hybrid_to_fla_hf.py.
python tools/hybrid/convert_zebra_llama_to_hf.py \
    --checkpoint-path output/zebra_llama_kda_1B_BF16-pretrain/iter_0028000 \
    --output-dir output/zebra_llama_kda_1B_hf_iter_0028000
