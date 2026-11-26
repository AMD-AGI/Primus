###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in FlyDSL fused packed-QKV RoPE.

``PRIMUS_TURBO_FLYDSL_ROPE=1`` replaces Megatron's TE
``apply_fused_qkv_rotary_pos_emb`` (used when ``fused_single_qkv_rope`` is set)
with Primus-Turbo ``fused_qkv_rope``, whose forward and backward are both
FlyDSL kernels. It takes the Llama call shape: SBHD, BF16, fp32 freqs,
non-interleaved, no context parallelism, one-head K/V splits of width 128.
Every other call goes to the TE function it replaced, with a warning on the
first one, so a run that asked for FlyDSL RoPE does not train on TE unnoticed.
"""

from __future__ import annotations

import torch

from primus.backends.megatron.patches.turbo.fused_residual_norm_patches import (
    _env_truthy,
)
from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_KEY = "megatron.turbo.fused_qkv_rope"
_INSTALLED_ATTR = "_primus_turbo_fused_qkv_rope_installed"


def _enabled(ctx: PatchContext) -> bool:
    return _env_truthy("PRIMUS_TURBO_FLYDSL_ROPE") and is_primus_turbo_can_patch(ctx)


def unsupported_reason(
    qkv,
    q_freqs,
    k_freqs,
    qkv_split_arg_list,
    tensor_format,
    start_positions,
    interleaved,
    cu_seqlens,
    cp_size,
):
    """Why FlyDSL cannot take this call, or None when it can."""
    from primus_turbo.pytorch.kernels.rope.rope_impl import rope_shape_error

    if tensor_format != "sbhd" or interleaved or start_positions is not None or cu_seqlens is not None:
        return (
            f"format={tensor_format} interleaved={interleaved} "
            f"start_positions={start_positions is not None} cu_seqlens={cu_seqlens is not None}"
        )
    if cp_size != 1:
        return f"cp_size={cp_size}"
    if q_freqs.dtype != torch.float32 or k_freqs.dtype != torch.float32:
        return f"freqs dtype {q_freqs.dtype}/{k_freqs.dtype}"
    return rope_shape_error(qkv, q_freqs, k_freqs, qkv_split_arg_list)


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Route Megatron fused packed-QKV RoPE through Primus-Turbo FlyDSL fused_qkv_rope",
    condition=_enabled,
    priority=70,
)
def patch_fused_qkv_rope(_ctx: PatchContext) -> None:
    import megatron.core.transformer.attention as attention_module
    from primus_turbo.pytorch.ops.rope import fused_qkv_rope

    if getattr(attention_module, _INSTALLED_ATTR, False):
        return
    te_fn = getattr(attention_module, "apply_fused_qkv_rotary_pos_emb", None)
    warned = []

    def _apply_fused_qkv_rotary_pos_emb(
        qkv,
        q_freqs,
        k_freqs,
        qkv_split_arg_list,
        tensor_format="sbhd",
        start_positions=None,
        interleaved=False,
        cu_seqlens=None,
        cp_size=1,
        cp_rank=0,
    ):
        why = unsupported_reason(
            qkv,
            q_freqs,
            k_freqs,
            qkv_split_arg_list,
            tensor_format,
            start_positions,
            interleaved,
            cu_seqlens,
            cp_size,
        )
        if why is None:
            return fused_qkv_rope(qkv, q_freqs.contiguous(), k_freqs.contiguous(), list(qkv_split_arg_list))
        if te_fn is None:
            raise RuntimeError(
                f"[Patch:{_PATCH_KEY}] FlyDSL RoPE cannot take this call ({why}) and TE has none"
            )
        if not warned:
            warned.append(True)
            warning_rank_0(f"[Patch:{_PATCH_KEY}] TE fused QKV RoPE fallback: {why}")
        return te_fn(
            qkv,
            q_freqs,
            k_freqs,
            qkv_split_arg_list,
            tensor_format=tensor_format,
            start_positions=start_positions,
            interleaved=interleaved,
            cu_seqlens=cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )

    attention_module.apply_fused_qkv_rotary_pos_emb = _apply_fused_qkv_rotary_pos_emb
    # Megatron asserts this when fused_single_qkv_rope is set; the TE import is
    # no longer what provides the kernel.
    attention_module.HAVE_FUSED_QKV_ROPE = True
    setattr(attention_module, _INSTALLED_ATTR, True)
    log_rank_0(f"[Patch:{_PATCH_KEY}] Megatron fused QKV RoPE routed through Primus-Turbo FlyDSL")
