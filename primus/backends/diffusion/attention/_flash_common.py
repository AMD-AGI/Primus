###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from collections.abc import Callable

import torch


def _unwrap_output(x: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return x[0] if isinstance(x, tuple) else x


def run_flash_attention_backend(
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
    *,
    fixed_attention: Callable[..., torch.Tensor | tuple[torch.Tensor, ...]],
    varlen_attention: Callable[..., torch.Tensor | tuple[torch.Tensor, ...]],
    window_size_adapter: Callable[[tuple[int, ...]], tuple[int, ...]] | None = None,
    fixed_extra_kwargs: dict | None = None,
    varlen_extra_kwargs: dict | None = None,
):
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    call_window_size = window_size_adapter(window_size) if window_size_adapter is not None else window_size
    fixed_extra_kwargs = fixed_extra_kwargs or {}
    varlen_extra_kwargs = varlen_extra_kwargs or {}

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None and k_lens is None:
        qh = half(q)
        kh = half(k)
        vh = half(v)
        qh = qh.to(vh.dtype)
        kh = kh.to(vh.dtype)
        if q_scale is not None:
            qh = qh * q_scale

        x = fixed_attention(
            qh,
            kh,
            vh,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=call_window_size,
            deterministic=deterministic,
            **fixed_extra_kwargs,
        )
        return _unwrap_output(x).type(out_dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens, strict=False)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens, strict=False)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens, strict=False)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale

    cu_seqlens_q = (
        torch.cat([q_lens.new_zeros([1]), q_lens])
        .cumsum(0, dtype=torch.int32)
        .to(q.device, non_blocking=True)
    )
    cu_seqlens_k = (
        torch.cat([k_lens.new_zeros([1]), k_lens])
        .cumsum(0, dtype=torch.int32)
        .to(q.device, non_blocking=True)
    )
    max_sq = int(q_lens.max())
    max_sk = int(k_lens.max())

    import os

    import torch.distributed as dist

    if os.environ.get("OMNIFLOW_DEBUG_ATTN") == "1" and (not dist.is_initialized() or dist.get_rank() == 0):
        print(f"[DEBUG ATTN] q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
        print(f"[DEBUG ATTN] q_lens={q_lens}, k_lens={k_lens}")
        print(f"[DEBUG ATTN] cu_seqlens_q={cu_seqlens_q}, cu_seqlens_k={cu_seqlens_k}")
        print(f"[DEBUG ATTN] max_seqlen_q={max_sq} (lq={lq}), max_seqlen_k={max_sk} (lk={lk})")
        print(f"[DEBUG ATTN] q.dtype={q.dtype}, causal={causal}, window_size={call_window_size}")

    x = varlen_attention(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_sq,
        max_seqlen_k=max_sk,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=call_window_size,
        deterministic=deterministic,
        **varlen_extra_kwargs,
    )
    # Pad output back to (b, lq) shape to match the input padded layout
    out = _unwrap_output(x)
    if max_sq == lq:
        return out.unflatten(0, (b, lq)).type(out_dtype)
    # Variable-length: need to scatter back into padded tensor
    result = q.new_zeros(b, lq, *out.shape[1:])
    offset = 0
    for i in range(b):
        sl = int(q_lens[i])
        result[i, :sl] = out[offset : offset + sl]
        offset += sl
    return result.type(out_dtype)
