###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in FlyDSL fused packed-QKV RoPE.

``PRIMUS_TURBO_FLYDSL_ROPE=1`` replaces Megatron's TE
``apply_fused_qkv_rotary_pos_emb`` (used when ``fused_single_qkv_rope`` is set)
with Primus-Turbo ``fused_qkv_rope``, whose forward and backward are both
FlyDSL kernels. The replacement covers the Llama call shape: SBHD, BF16,
non-interleaved, no context parallelism, one-head K/V splits of width 128.
Other calls raise instead of falling back to TE, so a run that asked for
FlyDSL RoPE cannot silently train on the TE kernel.
"""

from __future__ import annotations

import os

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.turbo.fused_qkv_rope"
_INSTALLED_ATTR = "_primus_turbo_fused_qkv_rope_installed"


def _enabled(ctx: PatchContext) -> bool:
    value = os.environ.get("PRIMUS_TURBO_FLYDSL_ROPE", "0").strip().lower()
    return value in ("1", "true", "yes", "on") and is_primus_turbo_can_patch(ctx)


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
        if (
            tensor_format != "sbhd"
            or start_positions is not None
            or interleaved
            or cu_seqlens is not None
            or cp_size != 1
        ):
            raise RuntimeError(
                f"[Patch:{_PATCH_KEY}] FlyDSL RoPE has no TE fallback: "
                f"format={tensor_format} interleaved={interleaved} cp_size={cp_size} "
                f"start_positions={start_positions is not None} cu_seqlens={cu_seqlens is not None}"
            )
        return fused_qkv_rope(qkv, q_freqs, k_freqs, qkv_split_arg_list)

    attention_module.apply_fused_qkv_rotary_pos_emb = _apply_fused_qkv_rotary_pos_emb
    # Megatron asserts this when fused_single_qkv_rope is set; the TE import is
    # no longer what provides the kernel.
    attention_module.HAVE_FUSED_QKV_ROPE = True
    setattr(attention_module, _INSTALLED_ATTR, True)
    log_rank_0(f"[Patch:{_PATCH_KEY}] Megatron fused QKV RoPE routed through Primus-Turbo FlyDSL")
