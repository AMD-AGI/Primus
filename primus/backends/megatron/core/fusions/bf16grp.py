# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""bf16grp line: grouped launches for the small same-phase BF16 GEMMs on the critical path.

PRIMUS_BF16GRP_IDX=1 (default 0; off = int byte-for-byte): in CSA layers the frozen indexer runs two independent
BF16 GEMMs on the same input -- its compressor's wkv_gate (4096 -> 512) and the fused q / weight projection w_dq_w
(4096 -> 192). hipBLASLt runs them as 4 serial launches on the indexer branch (default GSU5 / GSU8 solutions + 2
PostGSU reduces, ~75-90 us per layer in-step), and that branch is the critical path of the CSA forward (the q / kv
path on the other graph stream finishes ~70-150 us earlier). bf16grp_mm runs both as ONE grouped split-K GEMM
launch + ONE fixed-order reduce launch with the same K slices (5 x 832 / 8 x 512), so both outputs are bit-identical
to the int kernels (checked on the training shapes; see ~/dsv41/work/bf16grp/ledger.md).
Only for shapes whose hipBLASLt slice plan is known (_PLAN, keyed on (M, N, K)); anything else keeps the torch calls.
"""
from __future__ import annotations

import os

import torch

_IDX = os.environ.get("PRIMUS_BF16GRP_IDX", "0") == "1"
_TILE = os.environ.get("PRIMUS_BF16GRP_TILE", "128x256")

# (M, N, K) -> K slice length of the int hipBLASLt solution (GSU S = ceil(K / L)), bf16 in / out, beta 0, no bias
_PLAN = {
    (4096, 512, 4096): 832,  # Cijk_..._MT256x176x64 GSU5 + PostGSU (indexer compressor wkv_gate)
    (4096, 192, 4096): 512,  # Cijk_..._MT192x128x64 GSU8 + PostGSU (indexer w_dq_w)
}
_STATE = {}


def _impl():
    if "mm" not in _STATE:
        try:
            from primus_turbo.gluon.gemm.bf16grp_gemm import bf16grp_mm, grp_supported

            _STATE["mm"] = (bf16grp_mm, grp_supported)
            print(f"[bf16grp] grouped indexer GEMMs on (tile {_TILE})", flush=True)
        except Exception as ex:  # noqa: BLE001
            _STATE["mm"] = None
            print(f"[bf16grp] Gluon bf16grp unavailable ({type(ex).__name__}: {ex}); torch GEMMs", flush=True)
    return _STATE["mm"]


def idx_enabled() -> bool:
    return _IDX


def _frozen_linear_w(lin):
    w = getattr(lin, "weight", None)
    b = getattr(lin, "bias", None)
    if not isinstance(w, torch.Tensor) or w.requires_grad or w.dtype != torch.bfloat16 or w.dim() != 2:
        return None
    if isinstance(b, torch.Tensor) and b.numel() > 0:
        return None
    if w.stride(1) != 1:
        return None
    return w


def indexer_pair(indexer, hidden: torch.Tensor):
    """(compressor proj [B, S, Nc], dqw [B, S, Nq]) for the frozen indexer through one grouped launch, or None.

    Caller contract: the compressor proj goes to indexer.indexer_compressor (via _bf16grp_proj), dqw replaces
    indexer.w_dq_w(hidden). Same bits as the two torch linears (int TunableOp file) on the planned shapes."""
    if not _IDX or hidden.dim() != 3 or hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
        return None
    if hidden.requires_grad and torch.is_grad_enabled():
        return None
    comp = getattr(indexer, "indexer_compressor", None)
    lin_q = getattr(indexer, "w_dq_w", None) if getattr(indexer, "_fuse_qw_proj", False) else None
    if comp is None or lin_q is None or not getattr(comp, "_fuse_proj", False):
        return None
    wc = _frozen_linear_w(getattr(comp, "wkv_gate", None))
    wq = _frozen_linear_w(lin_q)
    if wc is None or wq is None:
        return None
    B, S, D = hidden.shape
    M = B * S
    Lc = _PLAN.get((M, wc.shape[0], D))
    Lq = _PLAN.get((M, wq.shape[0], D))
    if Lc is None or Lq is None or wc.shape[1] != D or wq.shape[1] != D:
        return None
    impl = _impl()
    if impl is None:
        return None
    mm, ok = impl
    h2 = hidden.view(M, D)
    probs = [(h2, wc, Lc), (h2, wq, Lq)]
    if not ok(probs, _TILE):
        return None
    yc, yq = mm(probs, tile=_TILE)
    return yc.view(B, S, -1), yq.view(B, S, -1)


# ---- PRIMUS_BF16GRP_PAIR: compressor backward dgrad + wgrad in one launch (bf16rot2_mm, StaggerU order) ----
_PAIR = os.environ.get("PRIMUS_BF16GRP_PAIR", "0") == "1"
# (T, n, H) -> (wgrad tile, tpad, dgrad tile, tpad); shapes whose int kernels are untuned SAV DepthU-128 (StaggerU)
_PAIR_PLAN = {
    (4096, 2048, 4096): ("256x128s3", 16, "256x256", 16),  # CSA compressor wkv_gate
}


def compressor_bwd_pair(g2: torch.Tensor, w: torch.Tensor, x2: torch.Tensor, mg: torch.Tensor):
    """dX = g2 @ w (returned) and main_grad = g2^T @ x2 (beta 0, written into mg) in ONE launch, bit-identical to the
    int hipBLASLt pair (untuned SAV kernels = sequential K with a StaggerU start). None if not planned."""
    if not _PAIR or g2.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or x2.dtype != torch.bfloat16:
        return None
    if g2.dim() != 2 or w.dim() != 2 or x2.dim() != 2 or mg.dtype != torch.bfloat16 or not mg.is_contiguous():
        return None
    if g2.stride(1) != 1 or w.stride(1) != 1 or x2.stride(1) != 1:
        return None
    T, n = g2.shape
    H = w.shape[1]
    plan = _PAIR_PLAN.get((T, n, H))
    if plan is None or tuple(w.shape) != (n, H) or tuple(x2.shape) != (T, H) or tuple(mg.shape) != (n, H):
        return None
    if "pair" not in _STATE:
        try:
            from primus_turbo.gluon.gemm.bf16grp_gemm import bf16rot2_mm

            _STATE["pair"] = bf16rot2_mm
            print("[bf16grp] compressor dgrad + wgrad in one launch (bf16rot2)", flush=True)
        except Exception as ex:  # noqa: BLE001
            _STATE["pair"] = None
            print(f"[bf16grp] bf16rot2 unavailable ({type(ex).__name__}: {ex})", flush=True)
    fn = _STATE["pair"]
    if fn is None:
        return None
    wt, wp, dt, dp = plan
    dx = torch.empty(T, H, device=g2.device, dtype=torch.bfloat16)
    fn((g2.t(), x2, mg, wt, wp), (g2, w, dx, dt, dp))
    return dx
