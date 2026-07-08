###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import torch

from ._flash_common import run_flash_attention_backend

AITER_FLASH_ATTN_AVAILABLE = False
aiter = None

try:
    import aiter as _aiter  # type: ignore

    if hasattr(_aiter, "flash_attn_func") and hasattr(_aiter, "flash_attn_varlen_func"):
        aiter = _aiter
        AITER_FLASH_ATTN_AVAILABLE = True
except Exception:
    AITER_FLASH_ATTN_AVAILABLE = False


def _normalize_window_size(window_size: tuple[int, ...]) -> tuple[int, int, int]:
    if len(window_size) == 2:
        left, right = window_size
        return (left, right, 0)
    if len(window_size) == 3:
        left, right, sink = window_size
        return (left, right, sink)
    raise ValueError(f"window_size must have 2 or 3 items, got {window_size}")


def aiter_flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    assert AITER_FLASH_ATTN_AVAILABLE and aiter is not None

    need_lse = torch.is_grad_enabled()
    return run_flash_attention_backend(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        fixed_attention=aiter.flash_attn_func,
        varlen_attention=aiter.flash_attn_varlen_func,
        window_size_adapter=_normalize_window_size,
        fixed_extra_kwargs={"return_lse": need_lse},
        varlen_extra_kwargs={"return_lse": need_lse},
    )
