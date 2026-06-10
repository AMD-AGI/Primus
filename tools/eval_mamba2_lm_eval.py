#!/usr/bin/env python3
"""
lm-eval wrapper for Mamba2 hybrid models in FLA HuggingFace format.

Adds the same three workarounds we needed for GDN, plus one extra:

  1. Import `fla` so Mamba2Config / Mamba2ForCausalLM register with
     transformers' AutoConfig / AutoModel.
  2. Monkey-patch Mamba2ForCausalLM / Mamba2Model __init__ to swallow
     the `dtype=...` kwarg that newer transformers passes internally.
  3. Force `residual_in_fp32=False` on the loaded config — FLA's Mamba2
     mixer in_proj is bf16; with residual_in_fp32=True the residual comes
     back from mixer_norm in fp32 and the next mixer's in_proj F.linear
     fails with "expected mat1 and mat2 to have the same dtype".
  4. Default `trust_remote_code=True` so the Primus-converted checkpoint
     auto-loads via our custom Mamba2FullMlpForCausalLM class (saved as
     `modeling_mamba2_full_mlp.py` next to the checkpoint).

Usage (same CLI as lm_eval, just swap the command):

    python tools/eval_mamba2_lm_eval.py \
        --model hf \
        --model_args pretrained=output/mamba_hybrid_300M_fla_hf,dtype=bfloat16,trust_remote_code=True \
        --tasks arc_easy,arc_challenge,hellaswag,mmlu,openbookqa,piqa,race,winogrande \
        --batch_size auto \
        --output_path eval_results/mamba_hybrid_300M_primus
"""
import os
import fla  # noqa: F401  — registers Mamba2 with AutoConfig/AutoModel

# 1. dtype-kwarg patch (same trick as eval_gdn_lm_eval.py)
from fla.models.mamba2 import Mamba2ForCausalLM, Mamba2Model

_orig_causal_init = Mamba2ForCausalLM.__init__
_orig_model_init = Mamba2Model.__init__


def _patched_causal_init(self, config, *args, **kwargs):
    kwargs.pop("dtype", None)
    return _orig_causal_init(self, config, *args, **kwargs)


def _patched_model_init(self, config, *args, **kwargs):
    kwargs.pop("dtype", None)
    return _orig_model_init(self, config, *args, **kwargs)


Mamba2ForCausalLM.__init__ = _patched_causal_init
Mamba2Model.__init__ = _patched_model_init


# 2. residual_in_fp32=False patch — most reliable place is in the model's
#    __init__: override config.residual_in_fp32 before super().__init__ uses it.
def _make_patch(orig):
    def _patched(self, config, *args, **kwargs):
        kwargs.pop("dtype", None)
        if getattr(config, "residual_in_fp32", False):
            config.residual_in_fp32 = False
            print(f"[eval_mamba2_lm_eval] forced residual_in_fp32=False on {type(self).__name__}")
        return orig(self, config, *args, **kwargs)
    return _patched


Mamba2ForCausalLM.__init__ = _make_patch(_orig_causal_init)
Mamba2Model.__init__ = _make_patch(_orig_model_init)

# Also patch Mamba2Block.__init__ so the per-block `self.residual_in_fp32`
# attribute is set from the (now-patched) config, not from the original
# True value.  Block is constructed inside Mamba2Model.__init__.
from fla.models.mamba2.modeling_mamba2 import Mamba2Block as _Mamba2Block

_orig_block_init = _Mamba2Block.__init__


def _patched_block_init(self, config, layer_idx):
    if getattr(config, "residual_in_fp32", False):
        config.residual_in_fp32 = False
    return _orig_block_init(self, config, layer_idx)


_Mamba2Block.__init__ = _patched_block_init


# 3. Hand off to lm-eval CLI
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
