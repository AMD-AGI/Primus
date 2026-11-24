"""Row gather for the EC dedup expand forward (DLRM_HSTU_FP8_EC_GATHER_V2).

``torch.gather(table, 0, idx.unsqueeze(1).expand(-1, W))`` on the hoisted EC
keys lowers to PyTorch's ``vectorized_gather_kernel<16, long>``: ONE 64-lane
workgroup per output row (1 KiB at W=512 bf16) -- load the row's index, load
the row, store it -- and the launch is split into 2 (item, [2.37M, 512]) or 4
(cat-fused artist/album, [4.74M, 512]) sub-launches because the output byte
offset exceeds int32. In the ship10 1n trace that is 6 launches of ~0.31-0.34
ms writing 7.28 GB in 1.93 ms = 3.8 TB/s, while a plain 2 GiB fill on the same
GPU writes at ~6.6 TB/s: each wave has a two-deep dependent chain (index, then
row) for 1 KiB of payload.

``expand_rows`` moves ROWS rows per program (all indices in one load, all row
loads in flight, then the stores) in a single launch with 64-bit row math, and
by default streams the output with non-temporal stores (the ~7 GB output does
not stay in cache anyway). It is a pure copy: the output bytes are
torch.gather's for in-range indices. Unlike the aten kernel, out-of-range
indices are NOT checked; the EC callers pass torchrec dedup reverse indices /
merged member indices, which are in range by construction.

Measured (job 63410, gfx950, trace shapes U=78282, W=512 bf16; ids: lognormal sigma 1.56
row popularity fitted to a real-run EC segment-length histogram, and generic u**3;
probes/iter64/gatherv2): raw-bit (int16 view) equal to torch.gather for every config and
edge case (N in {0, 1, 31, 33, 1000}, W in {128, 512, 600}, bf16 and fp32); cat 1.569 ->
1.084 ms and item 0.799 -> 0.575 ms with ROWS=32, WARPS=4, NT=1 (the defaults).

Gate: DLRM_HSTU_FP8_EC_GATHER_V2, default OFF in code (the gin binding
apply_env_bootstrap.DLRM_HSTU_FP8_EC_GATHER_V2 turns it on; "0" is the kill switch; read
once at import; forwarded by launch_slurm.sh's DLRM_HSTU_FP8_ prefix rule).
Knobs: ..._ROWS (32; 64 with EC_GATHER_WIDE), ..._WARPS (4; 16 with
EC_GATHER_WIDE), ..._NT (1 = non-temporal stores), ..._MIN (100000: smaller
index tensors stay on torch.gather). Arming line (rank log, once): "DLRMV4
armed OK: ec gather v2 -- ...".

DLRM_HSTU_FP8_EC_GATHER_WIDE (rccl iter 72, track ec): default OFF in code (the
gin binding apply_env_bootstrap.DLRM_HSTU_FP8_EC_GATHER_WIDE turns it on; "0"
is the kill switch; read once at import; acts only where EC_GATHER_V2 is on).
The same kernel launches 64 rows per program on 16 warps instead of 32 rows on
4 warps: each wave moves 4 rows (one 1 KiB row per load and per store
instruction at W=512 bf16) instead of 8, half as many workgroups get
dispatched for the same rows, 46 instead of 61 VGPRs (occupancy 8 both). Same
bytes, same pure row copy (raw-bit equal to torch.gather; no arithmetic). An
explicit ..._ROWS / ..._WARPS env value still wins. MI350X lab (rccl iter 72,
probes/iter72/ec and probes/iter72/m4, paired, 30 rounds): faster at every shape tried,
but the size depends on the ids' row locality -- about 1% with synthetic rand^3 ids
through torchrec's dedup code, 10-30% with make_ids_runs session ids (sigma 1.56 as
above, run means 8 / 3 calibrated to in-model times). Both id laws: size-distribution
class, training_rules L67-68 (main ruling 2026-10-13), not re-derived; yambda_5b.gin
(24), (49)(d). Arming line (rank log, once): "DLRMV4 armed OK: ec gather wide -- ...".
"""
import logging
import os

import torch
import triton
import triton.language as tl

logger: logging.Logger = logging.getLogger(__name__)

_ENV = "DLRM_HSTU_FP8_EC_GATHER_V2"
_LOGGED: bool = False


def _knob(name: str, default: int) -> int:
    return int(os.environ.get(f"{_ENV}_{name}", str(default)))


_ENABLED: bool = os.environ.get(_ENV, "").strip().lower() in (
    "1", "true", "on", "yes")
_WIDE_ENV = "DLRM_HSTU_FP8_EC_GATHER_WIDE"
_WIDE: bool = os.environ.get(_WIDE_ENV, "").strip().lower() in (
    "1", "true", "on", "yes")
_ROWS: int = _knob("ROWS", 64 if _WIDE else 32)
_WARPS: int = _knob("WARPS", 16 if _WIDE else 4)
_NT: bool = bool(_knob("NT", 1))
_MIN: int = _knob("MIN", 100_000)


def enabled() -> bool:
    return _ENABLED


def min_rows() -> int:
    return _MIN


@triton.jit
def _expand_rows(src_ptr, idx_ptr, out_ptr, N, W, stride_s, stride_o,
                 ROWS: tl.constexpr, BLOCK_W: tl.constexpr, NCB: tl.constexpr,
                 EVEN_W: tl.constexpr, NT_STORE: tl.constexpr):
    pid = tl.program_id(0)
    rb = pid // NCB
    cb = pid - rb * NCB
    r0 = rb.to(tl.int64) * ROWS
    rows = r0 + tl.arange(0, ROWS)
    cols = cb * BLOCK_W + tl.arange(0, BLOCK_W)
    if r0 + ROWS <= N:
        idx = tl.load(idx_ptr + rows)
        sp = src_ptr + idx[:, None] * stride_s + cols[None, :]
        op = out_ptr + rows[:, None] * stride_o + cols[None, :]
        if EVEN_W:
            v = tl.load(sp)
            if NT_STORE:
                tl.store(op, v, cache_modifier=".cs")
            else:
                tl.store(op, v)
        else:
            cm = (cols < W)[None, :]
            v = tl.load(sp, mask=cm)
            tl.store(op, v, mask=cm)
    else:
        rm = rows < N
        idx = tl.load(idx_ptr + rows, mask=rm, other=0)
        m = rm[:, None] & (cols < W)[None, :]
        v = tl.load(src_ptr + idx[:, None] * stride_s + cols[None, :], mask=m)
        tl.store(out_ptr + rows[:, None] * stride_o + cols[None, :], v, mask=m)


def supported(src: torch.Tensor, idx: torch.Tensor) -> bool:
    return (
        src.dim() == 2 and src.stride(1) == 1 and src.is_cuda
        and idx.dim() == 1 and idx.is_contiguous() and idx.device == src.device
        and idx.dtype in (torch.int32, torch.int64)
    )


def expand_rows(src: torch.Tensor, idx: torch.Tensor, rows=None, num_warps=None,
                nt=None) -> torch.Tensor:
    """out[i] = src[idx[i]] for a 2-D row-major ``src`` and 1-D ``idx``."""
    global _LOGGED
    rows = rows or _ROWS
    num_warps = num_warps or _WARPS
    nt = _NT if nt is None else bool(nt)
    n = idx.numel()
    w = src.size(1)
    out = torch.empty((n, w), dtype=src.dtype, device=src.device)
    if n == 0 or w == 0:
        return out
    block_w = min(triton.next_power_of_2(w), 512)
    ncb = triton.cdiv(w, block_w)
    grid = (triton.cdiv(n, rows) * ncb,)
    _expand_rows[grid](src, idx, out, n, w, src.stride(0), out.stride(0),
                       ROWS=rows, BLOCK_W=block_w, NCB=ncb, EVEN_W=(w % block_w == 0),
                       NT_STORE=nt, num_warps=num_warps)
    if not _LOGGED:
        _LOGGED = True
        logger.info(
            "DLRMV4 armed OK: ec gather v2 -- expand [%d, %d] %s with %d rows/program, "
            "%d warps, nt_store=%s (kill switch %s=0)", n, w, src.dtype, rows,
            num_warps, nt, _ENV,
        )
        if _WIDE:
            logger.info(
                "DLRMV4 armed OK: ec gather wide -- %d rows/program on %d warps "
                "(was 32 x 4; same kernel and bytes; kill switch %s=0)",
                rows, num_warps, _WIDE_ENV,
            )
    return out
