#!/usr/bin/env python3
"""
lm-eval wrapper for KDA models (Kimi Delta Attention).

Importing `fla` registers KDAConfig / KDAForCausalLM with transformers'
AutoConfig / AutoModel, which lm-eval's --model hf path relies on.
Without this import, AutoConfig.from_pretrained fails with
'model type kda not recognized'.

Usage (same CLI as lm_eval, just swap the command):

    python tools/eval_kda_lm_eval.py \
        --model hf \
        --model_args pretrained=output/kda_pure_300M_fla_hf,dtype=bfloat16,trust_remote_code=True,tokenizer=meta-llama/Llama-3.2-1B \
        --tasks arc_easy,arc_challenge,hellaswag,openbookqa,piqa,winogrande,mmlu,race \
        --batch_size auto \
        --output_path output/kda_pure_300M_eval_results
"""
import fla  # noqa: F401  — registers KDA with AutoConfig/AutoModel

# Monkey-patch FLA model classes to accept **kwargs (e.g. 'dtype')
# that newer transformers (>=4.55) passes internally during from_pretrained
from fla.models.kda import KDAForCausalLM, KDAModel

_orig_causal_init = KDAForCausalLM.__init__
_orig_model_init = KDAModel.__init__


def _patched_causal_init(self, config, *args, **kwargs):
    kwargs.pop('dtype', None)
    return _orig_causal_init(self, config, *args, **kwargs)


def _patched_model_init(self, config, *args, **kwargs):
    kwargs.pop('dtype', None)
    return _orig_model_init(self, config, *args, **kwargs)


KDAForCausalLM.__init__ = _patched_causal_init
KDAModel.__init__ = _patched_model_init

from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
