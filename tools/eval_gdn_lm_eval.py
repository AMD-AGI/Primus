#!/usr/bin/env python3
"""
lm-eval wrapper for GDN models converted to FLA HuggingFace format.

Importing `fla` registers GatedDeltaNetConfig / GatedDeltaNetForCausalLM
with transformers' AutoConfig / AutoModel, which lm-eval's --model hf
path relies on. Without this import, AutoConfig.from_pretrained fails
with 'model type gated_deltanet not recognized'.

Usage (same CLI as lm_eval, just swap the command):

    python tools/eval_gdn_lm_eval.py \
        --model hf \
        --model_args pretrained=output/gdn_pure_1B_fla_hf,dtype=bfloat16,trust_remote_code=True \
        --tasks arc_easy,arc_challenge,hellaswag,mmlu,openbookqa,piqa,race,winogrande \
        --batch_size auto \
        --output_path eval_results/gdn_pure_1B
"""
import fla  # noqa: F401  — registers GatedDeltaNet with AutoConfig/AutoModel

# Monkey-patch FLA model classes to accept **kwargs (e.g. 'dtype')
# that newer transformers (>=4.55) passes internally during from_pretrained
from fla.models.gated_deltanet import GatedDeltaNetForCausalLM, GatedDeltaNetModel

_orig_causal_init = GatedDeltaNetForCausalLM.__init__
_orig_model_init = GatedDeltaNetModel.__init__

def _patched_causal_init(self, config, *args, **kwargs):
    kwargs.pop('dtype', None)
    return _orig_causal_init(self, config, *args, **kwargs)

def _patched_model_init(self, config, *args, **kwargs):
    kwargs.pop('dtype', None)
    return _orig_model_init(self, config, *args, **kwargs)

GatedDeltaNetForCausalLM.__init__ = _patched_causal_init
GatedDeltaNetModel.__init__ = _patched_model_init

from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
