"""FlyDSL QKV split + RoPE in place of TE's ``apply_fused_qkv_rotary_pos_emb``.

With ``fused_single_qkv_rope`` on, Megatron's attention applies RoPE to the packed QKV
projection ``[S, B, NG, (NPG + 2) * D]`` in one TE call and gets q, k, v back. Turbo's
``fused_qkv_rope`` takes the same arguments and returns the same tensors (q, k within one
bf16 ulp of TE, v a copy). At the 70B shape its kernels take 0.168 ms per layer fwd+bwd
against TE's 0.278 ms.

Megatron binds the TE function into its own module namespace at import, so that binding
is the one replaced. Any call Turbo cannot take (extra kwargs such as cu_seqlens or
interleaved, non-fp32 freqs, a shape ``rope_shape_error`` rejects) goes to TE.
"""

from __future__ import annotations

import os

import torch

_STATE = {"installed": False, "flydsl_calls": 0, "te_calls": 0}


def _log(msg: str) -> None:
    if os.environ.get("TURBO_DEBUG", "0") in ("1", "true", "True"):
        try:
            with open("/results/turbo_decide.log", "a") as fh:
                fh.write(msg + "\n")
        except Exception:
            pass


def install() -> bool:
    if _STATE["installed"]:
        return True
    import megatron.core.transformer.attention as attn_mod
    from primus_turbo.pytorch.kernels.rope.rope_impl import rope_shape_error
    from primus_turbo.pytorch.ops import fused_qkv_rope

    te_fn = getattr(attn_mod, "apply_fused_qkv_rotary_pos_emb", None)
    if te_fn is None:
        raise RuntimeError(
            "PRIMUS_TURBO_FLYDSL_ROPE=1, but megatron.core.transformer.attention has no "
            "apply_fused_qkv_rotary_pos_emb (TE fused QKV RoPE unavailable)"
        )

    def apply_fused_qkv_rotary_pos_emb(qkv, q_freqs, k_freqs, qkv_split_arg_list, *args, **kwargs):
        why = None
        if args or kwargs:
            why = f"extra arguments {list(kwargs) or len(args)}"
        elif q_freqs.dtype != torch.float32 or k_freqs.dtype != torch.float32:
            why = f"freqs dtype {q_freqs.dtype}"
        else:
            why = rope_shape_error(qkv, q_freqs, k_freqs, qkv_split_arg_list)
        if why is not None:
            _STATE["te_calls"] += 1
            if _STATE["te_calls"] == 1:
                _log(f"[FlyDSLRoPE] TE fallback: {why}")
            return te_fn(qkv, q_freqs, k_freqs, qkv_split_arg_list, *args, **kwargs)
        _STATE["flydsl_calls"] += 1
        if _STATE["flydsl_calls"] == 1:
            _log(
                f"[FlyDSLRoPE] fused_qkv_rope qkv={tuple(qkv.shape)} "
                f"split={list(qkv_split_arg_list)}"
            )
        return fused_qkv_rope(
            qkv, q_freqs.contiguous(), k_freqs.contiguous(), list(qkv_split_arg_list)
        )

    apply_fused_qkv_rotary_pos_emb._te_fn = te_fn
    attn_mod.apply_fused_qkv_rotary_pos_emb = apply_fused_qkv_rotary_pos_emb
    _STATE["installed"] = True
    return True
