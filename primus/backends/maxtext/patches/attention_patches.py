###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText attention patches (ROCm/maxtext release/v26.10).

- ``maxtext.attention.synthetic_causal_mask`` (0aaedc18c): with
  ``dataset_type: synthetic`` the TE (cudnn_flash_te) attention path uses a plain
  causal mask instead of building the S x S mask. Synthetic batches are unpadded
  and single-segment, so causal is exact; sliding windows still reach the kernel
  through ``window_size``.
- ``maxtext.models.query_pre_attn_scalar`` (ba34cb9e2): pass
  ``query_pre_attn_scalar=head_dim**-0.5`` to ``Attention`` in the Gemma, Gemma2,
  Llama2, Mistral and Mixtral decoder layers.
"""

import importlib

from primus.backends.maxtext.patches._source_patch_utils import (
    any_in_source,
    rewrite_source,
)
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_MASK_ANCHOR = '    elif model_mode == MODEL_MODE_PREFILL and self.config.attention_kernel == "cudnn":\n'
_MASK_NEW = (
    '    elif self.config.dataset_type == "synthetic":\n'
    "      # Synthetic batches are unpadded and single-segment (the pipeline hands out an all-ones\n"
    "      # inputs_segmentation), so a causal mask is already exact and the S x S mask built below can\n"
    "      # be skipped. Unlike context parallelism this places no limit on the attention type: a\n"
    "      # sliding window reaches the kernel as window_size rather than through attn_mask.\n"
    "      attn_mask = None\n"
    "      dummy_attn_mask = None\n"
    '      mask_type = "causal"\n' + _MASK_ANCHOR
)


@register_patch(
    patch_id="maxtext.attention.synthetic_causal_mask",
    backend="maxtext",
    phase="setup",
    description="Use a causal mask (no S x S mask) in TE attention when dataset_type is synthetic",
    condition=lambda ctx: True,
)
def patch_synthetic_causal_mask(ctx: PatchContext) -> None:
    del ctx
    prefix = "[Patch:maxtext.attention.synthetic_causal_mask]"
    try:
        from maxtext.layers import attention_op
    except ImportError as e:
        warning_rank_0(f"{prefix} maxtext.layers.attention_op not importable; skipped: {e}")
        return
    owner = attention_op.AttentionOp
    if any_in_source(owner, "cudnn_flash_attention", ['elif self.config.dataset_type == "synthetic":']):
        log_rank_0(f"{prefix} AttentionOp.cudnn_flash_attention already special-cases synthetic data.")
        return
    rewrite_source(owner, "cudnn_flash_attention", [(_MASK_ANCHOR, _MASK_NEW)])
    log_rank_0(f"{prefix} Synthetic data uses a causal mask in cudnn_flash_attention.")


_SCALAR_LINE = "        query_pre_attn_scalar=(config.head_dim**-0.5),\n"
_RAGGED_ANCHOR = "        ragged_block_size=config.ragged_block_size,\n"
_SOFTCAP_ANCHOR = "        attn_logits_soft_cap=config.attn_logits_soft_cap,\n"

# module, decoder layer class, anchor the new kwarg follows, number of Attention(...) calls
_DECODER_LAYERS = (
    ("maxtext.models.gemma", "GemmaDecoderLayer", _RAGGED_ANCHOR, 1),
    ("maxtext.models.gemma2", "Gemma2DecoderLayer", _SOFTCAP_ANCHOR, 2),
    ("maxtext.models.llama2", "LlamaDecoderLayer", _RAGGED_ANCHOR, 1),
    ("maxtext.models.mistral", "MistralDecoderLayer", _RAGGED_ANCHOR, 1),
    ("maxtext.models.mixtral", "MixtralDecoderLayer", _RAGGED_ANCHOR, 1),
)


@register_patch(
    patch_id="maxtext.models.query_pre_attn_scalar",
    backend="maxtext",
    phase="setup",
    description="Pass query_pre_attn_scalar=head_dim**-0.5 in the Gemma/Gemma2/Llama2/Mistral/Mixtral decoder layers",
    condition=lambda ctx: True,
)
def patch_query_pre_attn_scalar(ctx: PatchContext) -> None:
    del ctx
    prefix = "[Patch:maxtext.models.query_pre_attn_scalar]"
    patched = []
    for module_name, class_name, anchor, count in _DECODER_LAYERS:
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            warning_rank_0(f"{prefix} {module_name} not importable; skipped: {e}")
            continue
        owner = getattr(module, class_name)
        if any_in_source(owner, "__init__", ["query_pre_attn_scalar"]):
            log_rank_0(f"{prefix} {class_name} already passes query_pre_attn_scalar; skipped.")
            continue
        rewrite_source(owner, "__init__", [(anchor, anchor + _SCALAR_LINE, count)])
        patched.append(class_name)
    log_rank_0(f"{prefix} Patched: {patched}")
