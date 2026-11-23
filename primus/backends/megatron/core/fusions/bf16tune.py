###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Kernel choice for the BF16 GEMMs that the TunableOp CSV leaves on hipBLASLt's untuned defaults
(Bias_HA_S_SAV ... CD1_1, PostGSU split-K): the grouped o_a wgrad, the CSA / HCA compressor wkv_gate
fwd / dgrad / wgrad, and the indexer compressor (4096 -> 512) and q / weight projection (4096 -> 192) forwards.

Each planned shape runs a Gluon kernel that reproduces hipBLASLt's fixed fp32 K-partial order bit for bit:
a deterministic split-K ``bf16g2`` (``sk<S>``: the GSU4 / GSU8 kernels) or ``bf16rot`` (``rot``: bf16g2 with
hipBLASLt's StaggerU K start). ``bt_mm`` returns None for every other shape, so callers keep their torch call.

The bit-equality is keyed to the hipBLASLt solutions the image's TunableOp CSV selects; rerun the bf16grp tests
after an image change. Needs the Gluon gate (``dsv4_features.gluon_ok``).
"""
from __future__ import annotations

import torch

from primus.backends.megatron.core.dsv4_features import fast_enabled, gluon_ok
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

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
# spec: g2:<tile>[:t<pad>] = bf16g2_mm; sk<S>:<tile>[:t<pad>] = split-K S; rot:<tile>[:t<pad>] = bf16rot_mm;
# t16 = 16-element LDS row pad for MN-major operand tiles.
_PLAN = {
    "hca_fwd": "sk4:256x256",  # GSU4
    "hca_wgrad": "sk4:256x256:t16",  # GSU4
    "q192_fwd": "sk8:128x256",  # GSU8
    "csa_fwd": "rot:256x128",
    "csa_dgrad": "rot:256x256:t16",
    "csa_wgrad": "rot:256x128s3:t16",
    "hca_dgrad": "rot:256x256:t16",
    "oa_wgrad": "rot:256x256:t16",
}


def _parse_plan():
    by_shape = {}
    for name, spec in _PLAN.items():
        f = spec.split(":")
        kind, tile = f[0], f[1]
        splits = 1 if kind == "g2" else (0 if kind == "rot" else int(kind[2:]))  # 0: bf16rot (StaggerU order)
        kw = {"tpad": int(t[1:]) for t in f[2:] if t.startswith("t")}
        by_shape[_SHAPES[name]] = (name, splits, tile, kw)
    return by_shape


_STATE = {}


def _impl():
    if "impl" not in _STATE:
        try:
            from primus_turbo.gluon.gemm.gemm_bf16_grouped_kernel import bf16rot_mm
            from primus_turbo.gluon.gemm.gemm_bf16_kernel import bf16g2_mm
            from primus_turbo.gluon.gemm.gemm_bf16_splitk_kernel import bf16g2_splitk_mm

            _STATE["impl"] = (bf16g2_mm, bf16g2_splitk_mm)
            _STATE["rot"] = bf16rot_mm
            _STATE["plan"] = _parse_plan()
            log_rank_0(
                f"[bf16tune] Gluon GEMM plan: {sorted(v[0] + '=' + str(v[1:]) for v in _STATE['plan'].values())}"
            )
        except Exception as ex:  # noqa: BLE001
            _STATE["impl"] = None
            warning_rank_0(
                f"[bf16tune] Gluon bf16g2 unavailable ({type(ex).__name__}: {ex}); hipBLASLt / TunableOp only"
            )
    return _STATE["impl"]


def enabled() -> bool:
    return fast_enabled() and gluon_ok()


def bt_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None):
    """out = a @ b (2-D, or 3-D batched like torch.bmm) through the shape's Gluon winner; None if not planned."""
    if not enabled() or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
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
    name, splits, tile, kw = hit
    shape = tuple(a.shape[:-1]) + (N,)
    if out is None:
        out = torch.empty(shape, device=a.device, dtype=torch.bfloat16)
    elif tuple(out.shape) != shape or out.dtype != torch.bfloat16 or out.stride(-1) != 1:
        return None
    if splits == 0:  # bf16grp: bf16rot_mm (bf16g2 + hipBLASLt StaggerU K start)
        rot = _STATE.get("rot")
        if rot is None or K % 128 or K < 384:
            return None
        return rot(a, b, out, tile=tile, **kw)
    if K % (128 * splits) or K // splits < 384:
        return None
    g2, sk = impl
    if a.dim() == 3:
        if splits != 1:
            return None
        return g2(a, b, out, tile=tile, **kw)
    return sk(a, b, out, splits, tile=tile, **kw)


def bt_linear(x: torch.Tensor, w: torch.Tensor, bias=None):
    """y = x @ w^T for an nn.Linear without bias through bt_mm; None if not planned (caller runs the module)."""
    if not enabled() or bias is not None or x.dim() < 2 or not x.is_contiguous() or w.dim() != 2:
        return None
    y = bt_mm(x.reshape(-1, x.shape[-1]), w.t())
    return None if y is None else y.view(*x.shape[:-1], w.shape[0])
