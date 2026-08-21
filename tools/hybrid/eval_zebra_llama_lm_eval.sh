#!/bin/bash

args=("$@")
for arg in "${args[@]}"; do
    eval "$arg"
done

echo "model_path:   ${model_path:=output/kda_1B_pure_hf}"
echo "tokenizer:    ${tokenizer:=meta-llama/Llama-3.2-1B}"
echo "batch_size:   ${batch_size:=auto}"
echo "num_fewshot:  ${num_fewshot:=0}"
echo "output_path:  ${output_path:=eval_results}"
echo "device:       ${device:=cuda}"

TASKS="arc_easy,arc_challenge,hellaswag,mmlu,openbookqa,piqa,race,winogrande"

pip install lm-eval --quiet 2>/dev/null

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.." || exit 1

python -c "
import sys; sys.path.insert(0, 'tools/hybrid')
from modeling_zebra_llama import ZebraLlamaForCausalLM, ZebraLlamaConfig
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register('zebra_llama', ZebraLlamaConfig)
AutoModelForCausalLM.register(ZebraLlamaConfig, ZebraLlamaForCausalLM)
import lm_eval
results = lm_eval.simple_evaluate(
    model='hf',
    model_args='pretrained=${model_path},tokenizer=${tokenizer},trust_remote_code=True',
    tasks='${TASKS}'.split(','),
    batch_size='${batch_size}',
    num_fewshot=${num_fewshot} if '${num_fewshot}' != 'None' else None,
    device='${device}',
    log_samples=True,
)
from lm_eval.utils import make_table
print(make_table(results))
import json, os
os.makedirs('${output_path}', exist_ok=True)
with open('${output_path}/results.json', 'w') as f:
    json.dump(results['results'], f, indent=2, default=str)
print(f\"Results saved to ${output_path}/results.json\")
"
