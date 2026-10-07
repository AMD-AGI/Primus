###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Route World Mirror attention through aiter when that package is installed.

The amd_dev checkout calls PyTorch SDPA. This wraps that call inside the
World Mirror attention module. Float32 queries, keys, and values are cast to
bf16 for aiter and the result is cast back. Dropout and head dims above 256
stay on SDPA. If aiter is missing, SDPA is used and a warning is logged once.
"""

from __future__ import annotations

import logging
import os

_SKIPPED = set()


def _skip(reason: str) -> None:
    if reason in _SKIPPED:
        return
    _SKIPPED.add(reason)
    logging.getLogger(__name__).warning(
        "WORLD_MIRROR_ATTN_BACKEND=aiter fell back to SDPA: %s",
        reason,
    )


def _aiter_attention(q, k, v):
    import torch

    orig_dtype = q.dtype
    if orig_dtype not in (torch.float16, torch.bfloat16):
        if "bf16-cast" not in _SKIPPED:
            _SKIPPED.add("bf16-cast")
            logging.getLogger(__name__).info(
                "WORLD_MIRROR_ATTN_BACKEND=aiter casting %s attention inputs to bf16",
                orig_dtype,
            )
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    if q.shape[-1] > 256:
        raise RuntimeError(f"aiter flash attention supports head_dim <= 256, got {q.shape[-1]}")
    try:
        import aiter
    except Exception as exc:
        raise RuntimeError("aiter.flash_attn_func is not installed") from exc
    if not hasattr(aiter, "flash_attn_func"):
        raise RuntimeError("aiter.flash_attn_func is not installed")

    q_b = q.transpose(1, 2).contiguous()
    k_b = k.transpose(1, 2).contiguous()
    v_b = v.transpose(1, 2).contiguous()
    kwargs = {
        "dropout_p": 0.0,
        "softmax_scale": None,
        "causal": False,
        "window_size": (-1, -1),
    }
    flash = aiter.flash_attn_func
    if torch.is_grad_enabled():
        try:
            out = flash(q_b, k_b, v_b, return_lse=True, **kwargs)
        except TypeError:
            out = flash(q_b, k_b, v_b, **kwargs)
    else:
        out = flash(q_b, k_b, v_b, **kwargs)
    if isinstance(out, tuple):
        out = out[0]
    out = out.transpose(1, 2).contiguous()
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    return out


def install_aiter_attention() -> None:
    """Patch World Mirror's attention module. Import World Mirror first."""
    backend = os.environ.get("WORLD_MIRROR_ATTN_BACKEND", "sdpa").strip().lower()
    if backend != "aiter":
        return

    import src.models.layers.attention as attention

    if getattr(attention, "_primus_aiter_installed", False):
        return

    original = attention.F.scaled_dot_product_attention

    def _scaled_dot_product_attention(q, k, v, dropout_p=0.0, **kwargs):
        if dropout_p == 0.0:
            try:
                return _aiter_attention(q, k, v)
            except Exception as exc:
                _skip(str(exc))
        return original(q, k, v, dropout_p=dropout_p, **kwargs)

    attention.F.scaled_dot_product_attention = _scaled_dot_product_attention
    attention._primus_aiter_installed = True
