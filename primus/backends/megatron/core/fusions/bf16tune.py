# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""bf16tune line: kernel choice for the BF16 GEMMs that the int TunableOp CSV leaves on hipBLASLt's untuned defaults
(Bias_HA_S_SAV ... CD1_1, PostGSU split-K): grouped o_a wgrad, CSA / HCA compressor wkv_gate fwd / dgrad / wgrad,
indexer compressor (4096 -> 512) and indexer q/w projection (4096 -> 192) forwards.

PRIMUS_BF16TUNE=1 (default 0; off = int byte-for-byte):
  * opt3_best.env points TunableOp at tunableop_dsv4_gfx1250_bf16tune_.csv (int bf16 CSV + tuned entries for these
    shapes), so shapes whose winner is a tuned hipBLASLt / rocBLAS solution need no code here;
  * the call sites below ask ``bt_mm`` first; it runs the shape's Gluon winner (bf16g2_mm, or deterministic split-K
    bf16g2 + fixed-order fp32 reduce) and returns None for every other shape (caller keeps its torch call).
Numerics (user decision 2026-11-22): these GEMMs may change K accumulation order (deterministic, no atomics); the
e2e loss gate decides. PRIMUS_BF16TUNE_PLAN=name=spec,... overrides the table (spec: g2:<tile> | sk<S>:<tile> | off).
"""
from __future__ import annotations

import os

import torch

_ON = os.environ.get("PRIMUS_BF16TUNE", "0") == "1"

# name: (ndim, M, N, K, A K-contiguous, B N-contiguous) -- training layouts at S=4096, H=4096
_SHAPES = {
    "oa_wgrad": (3, 1024, 4096, 4096, False, True),  # per group, G=8 batched
    "csa_fwd": (2, 4096, 2048, 4096, True, False),
    "csa_dgrad": (2, 4096, 4096, 2048, True, True),
    "csa_wgrad": (2, 2048, 4096, 4096, False, True),
    "hca_fwd": (2, 4096, 1024, 4096, True, False),
    "hca_dgrad": (2, 4096, 4096, 1024, True, True),
    "hca_wgrad": (2, 1024, 4096, 4096, False, True),
    "idx512_fwd": (2, 4096, 512, 4096, True, False),
    "q192_fwd": (2, 4096, 192, 4096, True, False),
}
# winners from ~/dsv41/work/bf16tune (microbench, L2 flushed, after a GEMM, median of 30); "off" = torch (TunableOp)
_PLAN = {
    "hca_fwd": "g2:256x128",
    "hca_dgrad": "g2:256x256",
    "idx512_fwd": "sk4:256x128",
    "q192_fwd": "sk8:128x256",  # bit-identical to the int GSU8 kernel
}


def _parse_plan():
    plan = dict(_PLAN)
    for item in filter(None, os.environ.get("PRIMUS_BF16TUNE_PLAN", "").split(",")):
        k, v = item.split("=")
        plan[k.strip()] = v.strip()
    by_shape = {}
    for name, spec in plan.items():
        if spec != "off" and name in _SHAPES:
            kind, tile = spec.split(":")[:2]
            splits = 1 if kind == "g2" else int(kind[2:])
            by_shape[_SHAPES[name]] = (name, splits, tile)
    return by_shape


_STATE = {}


def _impl():
    if "impl" not in _STATE:
        try:
            from primus_turbo.gluon.gemm.bf16g2_gemm import bf16g2_mm
            from primus_turbo.gluon.gemm.bf16g2_splitk import bf16g2_splitk_mm

            _STATE["impl"] = (bf16g2_mm, bf16g2_splitk_mm)
            _STATE["plan"] = _parse_plan()
            print(f"[bf16tune] Gluon GEMM plan: {sorted(v[0] + '=' + str(v[1:]) for v in _STATE['plan'].values())}",
                  flush=True)
        except Exception as ex:  # noqa: BLE001
            _STATE["impl"] = None
            print(f"[bf16tune] Gluon bf16g2 unavailable ({type(ex).__name__}: {ex}); hipBLASLt / TunableOp only",
                  flush=True)
    return _STATE["impl"]


def enabled() -> bool:
    return _ON


def bt_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None):
    """out = a @ b (2-D, or 3-D batched like torch.bmm) through the shape's Gluon winner; None if not planned."""
    if not _ON or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        return None
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        return None  # only called inside autograd Functions / on frozen weights; never bypass autograd
    impl = _impl()
    if impl is None:
        return None
    M, K = a.shape[-2:]
    N = b.shape[-1]
    if a.stride(-1) != 1 and a.stride(-2) != 1 or b.stride(-1) != 1 and b.stride(-2) != 1:
        return None
    key = (a.dim(), M, N, K, a.stride(-1) == 1, b.stride(-1) == 1)
    hit = _STATE["plan"].get(key)
    if hit is None:
        return None
    name, splits, tile = hit
    shape = tuple(a.shape[:-1]) + (N,)
    if out is None:
        out = torch.empty(shape, device=a.device, dtype=torch.bfloat16)
    elif tuple(out.shape) != shape or out.dtype != torch.bfloat16 or out.stride(-1) != 1:
        return None
    if K % (128 * splits) or K // splits < 384:
        return None
    g2, sk = impl
    if a.dim() == 3:
        if splits != 1:
            return None
        return g2(a, b, out, tile=tile)
    return sk(a, b, out, splits, tile=tile)


def bt_linear(x: torch.Tensor, w: torch.Tensor, bias=None):
    """y = x @ w^T for an nn.Linear without bias through bt_mm; None if not planned (caller runs the module)."""
    if not _ON or bias is not None or x.dim() < 2 or not x.is_contiguous() or w.dim() != 2:
        return None
    y = bt_mm(x.reshape(-1, x.shape[-1]), w.t())
    return None if y is None else y.view(*x.shape[:-1], w.shape[0])
