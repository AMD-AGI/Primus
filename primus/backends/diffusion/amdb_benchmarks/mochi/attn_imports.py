###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# SDPA stand-in for the attention helpers the official Mochi DiT imports from
# genmo.lib.attn_imports. Flash-attn is used when it is installed.

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F


def sdpa_attn_ctx(training: bool = False):
    del training
    return contextlib.nullcontext()


def sage_attn(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("SageAttention is not bundled. Use attention_mode='sdpa' or 'flash'.")


def flash_varlen_attn(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p: float = 0.0,
    softmax_scale=None,
):
    from primus.backends.diffusion.amdb_benchmarks.flops import (
        flop_counter_active,
        mark_attention_flops,
        varlen_qk_pairs,
    )
    from primus.backends.diffusion.attention import get_attention_backend
    from primus.backends.diffusion.attention.aiter import (
        AITER_FLASH_ATTN_AVAILABLE,
        aiter,
    )

    def mark(out):
        if not flop_counter_active():
            return out
        # Diffusers Mochi runs SDPA per sample over valid tokens only.
        return mark_attention_flops(
            out,
            heads=q.size(-2),
            head_dim=q.size(-1),
            v_head_dim=v.size(-1),
            qk_pairs=varlen_qk_pairs(cu_seqlens_q, cu_seqlens_k),
        )

    if get_attention_backend() == "flash_attn_aiter":
        if not AITER_FLASH_ATTN_AVAILABLE or not q.is_cuda or cu_seqlens_q is None:
            raise RuntimeError("Mochi flash attention requires AITER and packed CUDA sequences.")
        out = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            return_lse=torch.is_grad_enabled(),
        )
        return mark(out[0] if isinstance(out, tuple) else out)
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        flash_attn_varlen_func = None
    if flash_attn_varlen_func is not None and q.is_cuda and cu_seqlens_q is not None:
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
        )
        return mark(out)
    # Packed tokens [total, heads, dim]. Official SDPA training uses batch size 1.
    query = q.unsqueeze(0).transpose(1, 2)
    key = k.unsqueeze(0).transpose(1, 2)
    value = v.unsqueeze(0).transpose(1, 2)
    out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p)
    return out.transpose(1, 2).squeeze(0)
