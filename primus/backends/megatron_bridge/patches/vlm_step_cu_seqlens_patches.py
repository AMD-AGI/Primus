###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron-Bridge ``vlm_step`` packed-sequence padding patch.

``vlm_step.get_batch`` pads tokens to a multiple of 128 (or ``seq_length`` with
PP) but leaves ``cu_seqlens`` untouched, so packed batches built with
``pad_to_max_length=False`` fail in THD RoPE. This registers the padded tail as
an extra segment; padded positions have ``loss_mask == 0``.

Only applied while ``get_batch`` still has the padding block; remove once the
submodule includes NVIDIA-NeMo/Megatron-Bridge#4315.
"""

from __future__ import annotations

import functools
import inspect
from typing import Optional

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCHED_ATTR = "_primus_vlm_step_cu_seqlens_patched"


def _vlm_step_pads_without_cu_seqlens(_ctx: PatchContext) -> bool:
    try:
        import megatron.bridge.training.vlm_step as vlm_step

        source = inspect.getsource(vlm_step.get_batch)
    except (ImportError, OSError, TypeError):
        return False
    return "_ceil_to_mult" in source


def extend_cu_seqlens_to(
    cu_seqlens: Optional[torch.Tensor],
    cu_seqlens_argmin: Optional[torch.Tensor],
    max_seqlen: Optional[torch.Tensor],
    padded_len: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extend a ``-1``-padded ``[1, N]`` cu_seqlens so its segments end at ``padded_len``.

    ``cu_seqlens_argmin`` (host tensor) is the number of valid entries.
    """
    if cu_seqlens is None or cu_seqlens.size(0) != 1:
        return cu_seqlens, cu_seqlens_argmin, max_seqlen

    if cu_seqlens_argmin is not None:
        num_valid = int(cu_seqlens_argmin.reshape(-1)[0].item())
    else:
        sentinel = (cu_seqlens[0] == -1).nonzero()
        num_valid = int(sentinel[0].item()) if sentinel.numel() else cu_seqlens.size(1)
    if num_valid == 0:
        return cu_seqlens, cu_seqlens_argmin, max_seqlen
    packed_len = int(cu_seqlens[0, num_valid - 1].item())
    if packed_len >= padded_len:
        return cu_seqlens, cu_seqlens_argmin, max_seqlen

    if num_valid >= cu_seqlens.size(1):
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (0, 1), value=-1)
    else:
        cu_seqlens = cu_seqlens.clone()
    cu_seqlens[0, num_valid] = padded_len

    if cu_seqlens_argmin is not None:
        cu_seqlens_argmin = torch.full_like(cu_seqlens_argmin, num_valid + 1)
    if max_seqlen is not None:
        max_seqlen = torch.clamp(max_seqlen, min=padded_len - packed_len)
    return cu_seqlens, cu_seqlens_argmin, max_seqlen


def wrap_vlm_get_batch(orig_get_batch):
    warned_cp = False

    @functools.wraps(orig_get_batch)
    def get_batch(data_iterator, cfg, use_mtp: bool = False, *, pg_collection):
        from megatron.core.utils import get_pg_size

        nonlocal warned_cp
        outputs = orig_get_batch(data_iterator, cfg, use_mtp, pg_collection=pg_collection)
        tokens, labels, loss_mask, attention_mask, position_ids, cu_seqlens, argmin, max_seqlen, visual = (
            outputs
        )
        if tokens is None or cu_seqlens is None:
            return outputs
        # Under CP tokens are sliced per rank and no longer match cu_seqlens.
        if get_pg_size(getattr(pg_collection, "cp", None)) > 1:
            if not warned_cp:
                warning_rank_0(
                    "[Patch:megatron_bridge.vlm_step_cu_seqlens] Not applied with context parallelism; "
                    "set pad_to_max_length=True for packed VLM fine-tuning."
                )
                warned_cp = True
            return outputs

        cu_seqlens, argmin, max_seqlen = extend_cu_seqlens_to(cu_seqlens, argmin, max_seqlen, tokens.size(1))
        return tokens, labels, loss_mask, attention_mask, position_ids, cu_seqlens, argmin, max_seqlen, visual

    setattr(get_batch, _PATCHED_ATTR, True)
    return get_batch


@register_patch(
    "megatron_bridge.train.vlm_step_cu_seqlens",
    backend="megatron_bridge",
    phase="before_train",
    condition=_vlm_step_pads_without_cu_seqlens,
    description="Keep packed cu_seqlens in sync with vlm_step padding (Megatron-Bridge#4315).",
)
def patch_vlm_step_cu_seqlens(_ctx: PatchContext) -> None:
    import megatron.bridge.training.vlm_step as vlm_step

    if getattr(vlm_step.get_batch, _PATCHED_ATTR, False):
        return
    vlm_step.get_batch = wrap_vlm_get_batch(vlm_step.get_batch)
    log_rank_0("[Patch:megatron_bridge.vlm_step_cu_seqlens] Synced packed cu_seqlens with vlm_step padding")
