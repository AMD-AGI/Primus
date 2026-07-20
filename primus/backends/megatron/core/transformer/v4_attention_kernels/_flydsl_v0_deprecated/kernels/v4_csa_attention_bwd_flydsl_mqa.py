"""V4 CSA attention backward FlyDSL launcher (Phase B STEP 3a + 3b).

Env knobs:
  V4_FLYDSL_CSA_BWD_FLY_DQ   default 0 (1 = FlyDSL dq, 0 = Triton dq)
  V4_FLYDSL_CSA_BWD_FLY_DKV  default 0 (1 = FlyDSL dk_local/dv_local/
                              dgathered/dsink, 0 = Triton)
  V4_FLYDSL_BWD_VERBOSE      default 0

Wiring:
  knob_dq=0, knob_dkv=0   -> all Triton.
  knob_dq=1, knob_dkv=0   -> FlyDSL dq-only; Triton emits all others.
  knob_dq=0, knob_dkv=1   -> Triton dq; FlyDSL emits dk/dv/dgathered/dsink.
                            (rare path; uses the FULL FlyDSL kernel and
                            discards its dq.)
  knob_dq=1, knob_dkv=1   -> ONE FlyDSL launch produces all 5 grads
                            (no Triton call needed; the cheapest path
                            when both are on).

The launcher always emits a provenance line so the harness can prove the
expected backend was used.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Optional, Tuple

import torch

_FLYDSL_SRC = "/workspace/FlyDSL-amd"
if _FLYDSL_SRC not in sys.path:
    sys.path.insert(0, _FLYDSL_SRC)

os.environ.setdefault("FLYDSL_WAVES_PER_EU", "2")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if "/workspace/Primus" not in sys.path:
    sys.path.insert(0, "/workspace/Primus")

import triton  # noqa: E402

from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v0_deprecated.v4_csa_attention_bwd import (  # noqa: E402
    _launch_v4_csa_attention_bwd,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v1.v4_attention_bwd import (  # noqa: E402
    _v4_attention_bwd_preprocess_kernel,
)

_DQ_KERNEL_CACHE = {}
_DQ_KERNEL_LOCK = threading.Lock()
_FULL_KERNEL_CACHE = {}
_FULL_KERNEL_LOCK = threading.Lock()


def _get_fly_csa_dq(num_heads, head_dim, swa_window, dtype_str, has_sink, has_sparse, block_n, block_k):
    key = (num_heads, head_dim, swa_window, dtype_str, has_sink, has_sparse, block_n, block_k)
    with _DQ_KERNEL_LOCK:
        if key in _DQ_KERNEL_CACHE:
            return _DQ_KERNEL_CACHE[key]
        from v4_csa_bwd_dq_kernel import build_v4_csa_bwd_dq_module

        launch = build_v4_csa_bwd_dq_module(
            num_heads=num_heads,
            head_dim=head_dim,
            swa_window=swa_window,
            dtype_str=dtype_str,
            has_sink=has_sink,
            has_sparse=has_sparse,
            block_n=block_n,
            block_k=block_k,
            mqa_kv=True,
        )
        _DQ_KERNEL_CACHE[key] = launch
        return launch


def _get_fly_csa_full(num_heads, head_dim, swa_window, dtype_str, has_sink, has_sparse, block_n, block_k):
    key = (num_heads, head_dim, swa_window, dtype_str, has_sink, has_sparse, block_n, block_k)
    with _FULL_KERNEL_LOCK:
        if key in _FULL_KERNEL_CACHE:
            return _FULL_KERNEL_CACHE[key]
        from v4_csa_bwd_full_kernel import build_v4_csa_bwd_full_module

        launch = build_v4_csa_bwd_full_module(
            num_heads=num_heads,
            head_dim=head_dim,
            swa_window=swa_window,
            dtype_str=dtype_str,
            has_sink=has_sink,
            has_sparse=has_sparse,
            block_n=block_n,
            block_k=block_k,
            mqa_kv=True,
        )
        _FULL_KERNEL_CACHE[key] = launch
        return launch


def _preprocess_deltas(out, dout, B, HQ, Sq, D):
    BLOCK_M_PRE = 64
    d_buf = torch.empty((B, HQ, Sq), device=out.device, dtype=torch.float32)
    pre_grid = (triton.cdiv(Sq, BLOCK_M_PRE), B * HQ)
    _v4_attention_bwd_preprocess_kernel[pre_grid](
        out,
        dout,
        d_buf,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dout.stride(3),
        d_buf.stride(0),
        d_buf.stride(1),
        d_buf.stride(2),
        Sq,
        HEAD=HQ,
        BLOCK_M=BLOCK_M_PRE,
        BLOCK_DMODEL=D,
        num_warps=4,
        num_stages=1,
    )
    return d_buf


def flydsl_v4_csa_attention_bwd(
    q: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    gathered: torch.Tensor,
    sparse_mask: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    *,
    sink: Optional[torch.Tensor],
    swa_window: int,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if q.dim() != 4 or k_local.dim() != 4 or v_local.dim() != 4:
        raise ValueError("rank-4 q / k_local / v_local required")
    if dout.shape != out.shape or out.shape != q.shape:
        raise ValueError(
            f"shape mismatch: q={tuple(q.shape)} out={tuple(out.shape)} dout={tuple(dout.shape)}"
        )
    B, HQ, Sq, D = q.shape
    K_topk = gathered.shape[2]
    has_sink = sink is not None

    use_fly_dq = os.getenv("V4_FLYDSL_CSA_BWD_FLY_DQ", "0") == "1"
    use_fly_dkv = os.getenv("V4_FLYDSL_CSA_BWD_FLY_DKV", "0") == "1"
    verbose = os.getenv("V4_FLYDSL_BWD_VERBOSE", "0") == "1"

    dq_tag = "FlyDSL" if use_fly_dq else "Triton"
    dkv_tag = "FlyDSL" if use_fly_dkv else "Triton"
    dsink_tag = (
        "FlyDSL" if (use_fly_dkv and has_sink) else ("FlyDSL" if (use_fly_dq and has_sink) else "Triton")
    )
    print(
        f"[v4_csa_bwd] provenance: dq={dq_tag} dk_local={dkv_tag} dv_local={dkv_tag} "
        f"dgathered={dkv_tag} dsink={dsink_tag} "
        f"B={B} HQ={HQ} Sq={Sq} K_topk={K_topk} D={D} swa={swa_window} "
        f"has_sink={has_sink}",
        flush=True,
    )

    # ============================================================
    # Path 1: nothing FlyDSL -> direct Triton pass-through.
    # ============================================================
    if not use_fly_dq and not use_fly_dkv:
        return _launch_v4_csa_attention_bwd(
            q,
            k_local,
            v_local,
            gathered,
            sparse_mask,
            out,
            dout,
            lse,
            sink=sink,
            swa_window=int(swa_window),
            scale=float(scale),
        )

    # ============================================================
    # Validate the FlyDSL kernel's preconditions.
    # ============================================================
    if q.dtype != torch.bfloat16:
        raise NotImplementedError(f"FlyDSL CSA bwd supports bf16 only; got {q.dtype}")
    if D % 64 != 0:
        raise NotImplementedError(f"FlyDSL CSA bwd requires D % 64 == 0; got D={D}")

    # MQA view -- the Triton wrapper expands MQA->MHA by .expand().contiguous(),
    # so head 0 has the original values.
    k_mqa = k_local[:, :1, :, :].contiguous()
    v_mqa = v_local[:, :1, :, :].contiguous()

    q_c = q.contiguous()
    dout_c = dout.contiguous()
    lse_c = lse.contiguous()
    gathered_c = gathered.contiguous()
    sparse_mask_c = sparse_mask.contiguous()

    d_buf = _preprocess_deltas(out, dout_c, B, HQ, Sq, D)

    if has_sink:
        sink_arg = sink.to(torch.float32) if sink.dtype != torch.float32 else sink.contiguous()
    else:
        sink_arg = torch.zeros((HQ,), device=q.device, dtype=torch.float32)

    block_n = 32
    block_k = 32
    dtype_str = "bf16"
    has_sparse = K_topk > 0

    # ============================================================
    # Path 2: BOTH knobs on -- one FlyDSL full-kernel launch.
    # ============================================================
    if use_fly_dq and use_fly_dkv:
        dq_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
        # dk_local / dv_local are stored at [B, HQ, Sq, D] (MHA-shape buffer
        # matching Triton's atomic-add target). Caller is free to reduce.
        dkl_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
        dvl_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
        dgathered_fp32 = torch.zeros((B, Sq, K_topk, D), device=q.device, dtype=torch.float32)
        dsink_fp32 = torch.zeros((HQ,), device=q.device, dtype=torch.float32)

        launch = _get_fly_csa_full(
            num_heads=HQ,
            head_dim=D,
            swa_window=int(swa_window),
            dtype_str=dtype_str,
            has_sink=has_sink,
            has_sparse=has_sparse,
            block_n=block_n,
            block_k=block_k,
        )

        if verbose:
            print(
                f"[v4_csa_bwd] FlyDSL FULL launch: B={B} HQ={HQ} Sq={Sq} K_topk={K_topk} "
                f"D={D} swa={swa_window} has_sink={has_sink} has_sparse={has_sparse}",
                flush=True,
            )

        launch(
            q_c,
            k_mqa,
            v_mqa,
            gathered_c,
            sparse_mask_c,
            dout_c,
            lse_c,
            d_buf,
            sink_arg,
            dq_fp32,
            dkl_fp32,
            dvl_fp32,
            dgathered_fp32,
            dsink_fp32,
            int(B),
            int(Sq),
            int(K_topk),
        )

        dq_out = dq_fp32.to(q.dtype)
        # dk_local / dv_local return shape [B, HQ, Sq, D] to match the Triton
        # output (it returns the broadcast-MHA buffer; bwd_modes._run_csa_pair
        # does the autograd reduction itself when the leaves were created
        # via .expand().contiguous()).
        dkl_out = dkl_fp32.to(k_local.dtype)
        dvl_out = dvl_fp32.to(v_local.dtype)
        dg_out = dgathered_fp32.to(gathered.dtype)
        dsink_out = dsink_fp32.to(sink.dtype) if has_sink else None
        return dq_out, dkl_out, dvl_out, dg_out, dsink_out

    # ============================================================
    # Path 3: partial overlap with Triton.
    # ============================================================
    # For partial paths we always run the Triton kernel and selectively
    # override its outputs with FlyDSL values.
    triton_out = _launch_v4_csa_attention_bwd(
        q,
        k_local,
        v_local,
        gathered,
        sparse_mask,
        out,
        dout,
        lse,
        sink=sink,
        swa_window=int(swa_window),
        scale=float(scale),
    )
    dq_t, dkl_t, dvl_t, dg_t, dsink_t = triton_out

    if use_fly_dq:
        # Override dq (+ dsink for sink case) with FlyDSL dq-only kernel.
        dq_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
        dsink_fp32 = torch.zeros((HQ,), device=q.device, dtype=torch.float32)
        launch = _get_fly_csa_dq(
            num_heads=HQ,
            head_dim=D,
            swa_window=int(swa_window),
            dtype_str=dtype_str,
            has_sink=has_sink,
            has_sparse=has_sparse,
            block_n=block_n,
            block_k=block_k,
        )
        if verbose:
            print(
                f"[v4_csa_bwd] FlyDSL dq launch: B={B} HQ={HQ} Sq={Sq} K_topk={K_topk} "
                f"D={D} swa={swa_window} has_sink={has_sink} has_sparse={has_sparse}",
                flush=True,
            )
        launch(
            q_c,
            k_mqa,
            v_mqa,
            gathered_c,
            sparse_mask_c,
            dout_c,
            lse_c,
            d_buf,
            sink_arg,
            dq_fp32,
            dsink_fp32,
            int(B),
            int(Sq),
            int(K_topk),
        )
        dq_out = dq_fp32.to(q.dtype)
        dsink_out = dsink_fp32.to(sink.dtype) if has_sink else None
        return dq_out, dkl_t, dvl_t, dg_t, dsink_out

    if use_fly_dkv:
        # Run the FULL kernel; we keep dk/dv/dgathered/dsink from FlyDSL and
        # discard its dq (Triton's dq is the reference here).
        dq_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
        dkl_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
        dvl_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
        dgathered_fp32 = torch.zeros((B, Sq, K_topk, D), device=q.device, dtype=torch.float32)
        dsink_fp32 = torch.zeros((HQ,), device=q.device, dtype=torch.float32)

        launch = _get_fly_csa_full(
            num_heads=HQ,
            head_dim=D,
            swa_window=int(swa_window),
            dtype_str=dtype_str,
            has_sink=has_sink,
            has_sparse=has_sparse,
            block_n=block_n,
            block_k=block_k,
        )

        if verbose:
            print(
                f"[v4_csa_bwd] FlyDSL DKV launch (dq discarded): B={B} HQ={HQ} Sq={Sq} K_topk={K_topk} "
                f"D={D} swa={swa_window} has_sink={has_sink} has_sparse={has_sparse}",
                flush=True,
            )

        launch(
            q_c,
            k_mqa,
            v_mqa,
            gathered_c,
            sparse_mask_c,
            dout_c,
            lse_c,
            d_buf,
            sink_arg,
            dq_fp32,
            dkl_fp32,
            dvl_fp32,
            dgathered_fp32,
            dsink_fp32,
            int(B),
            int(Sq),
            int(K_topk),
        )

        dkl_out = dkl_fp32.to(k_local.dtype)
        dvl_out = dvl_fp32.to(v_local.dtype)
        dg_out = dgathered_fp32.to(gathered.dtype)
        dsink_out = dsink_fp32.to(sink.dtype) if has_sink else None
        return dq_t, dkl_out, dvl_out, dg_out, dsink_out

    raise RuntimeError("unreachable")


__all__ = ["flydsl_v4_csa_attention_bwd"]
