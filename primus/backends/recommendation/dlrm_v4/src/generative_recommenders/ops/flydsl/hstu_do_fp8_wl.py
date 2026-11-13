"""C8 dO block quantizer, work-list form (DLRM_HSTU_FP8_DOBLK_WL).

Bitwise drop-in for `hstu_do_fp8.quantize_do_e4m3_blk` (Arbor X4C19 item
(48)(a); iter39 track blk, integrated iter40; the "retuned standalone
quantizer" direction came from a gpt-6 design critique): same signature,
same outputs BYTE FOR BYTE
(e4m3 dO and E8 = 127 - e_g), same group rule
    G(p) = (p // 64, (p % 16) // 8)   (64-token tiles, per sequence and head)
and the same per-element math (fp32 upcast, |x| group max, `_grp_exp` from the
fp32 bit pattern -- imported from hstu_do_fp8, not copied -- x * 2**e in fp32,
cast to float8e4nv). Max is exact and order independent, so only the work
decomposition differs:

1. Work list instead of the (B, cdiv(N, 64)) grid. A tiny kernel builds a
   SEQUENCE-MAJOR list of the non-empty (sequence, 64-token tile) items on the
   device (no host sync); the quantizer runs one program per item. In-model
   ~44% of the stock grid's programs exited immediately, and consecutive
   stock programs strode across 1024 sequences.
2. Per-tile scalar base pointer + int32 in-tile offsets, so a dO larger than
   2 GiB (the in-model case, ~2.3 M tokens x 1 KiB) still gets buffer loads
   with out-of-range masking instead of exec-branched 64-bit global loads.
3. An empty side-effecting inline asm on the loaded tile, so LLVM cannot hoist
   the bf16 -> f32 upcast between the tile's loads (it otherwise inserts an
   s_waitcnt vmcnt(0) after the first one or two loads).
4. One joined [64, H, 2] cross-warp max for both token groups (2 barriers per
   tile instead of 7).
5. num_warps=16; complete 64-row tiles take an unmasked path.

Coverage is identical to the stock grid: tiles t < cdiv(N, 64) of every
sequence (a sequence longer than N keeps its tail rows untouched, exactly as
in the stock kernel). Each build program scans all B offsets, so the
dispatcher keeps the grid kernel for B > WL_MAX_B (in-model B = 1024 per
rank; at B = 16384 the build kernel already costs more than it saves).

Measured (Arbor X4C19 iter39 blk, lab microbench, L2/MALL flushed, in-model
lengths): 0.785-0.802 -> 0.648-0.661 ms/call (-17.5..-19.5%, 30/30 rounds on
two MI350X GPUs); independent verifier -17.4% (12/12), warm mode -15.6%;
e4m3/E8 bytes identical on every pool and edge case incl. whole-group
NaN/Inf, signalling NaN and a 4.29 GB dO.
"""

import torch
import triton
import triton.language as tl

from generative_recommenders.ops.flydsl.hstu_do_fp8 import _grp_exp

# Largest batch (sequences per call) served by the work list; above it the
# dispatcher in hstu_do_fp8 keeps the grid kernel (same bytes either way).
WL_MAX_B: int = 8192


# B and cap are only used in masks/compares: not specialized, so a cap that
# changes per step (cdiv(T + 63 B, 64)) never JITs a second variant mid-run.
@triton.jit(do_not_specialize=["B", "cap"])
def _blk_wl_build_kernel(offs_ptr, wl_ptr, cnt_ptr, B, cap,
                         BP2: tl.constexpr):
    # Sequence-major work list, one program per tile index t (grid
    # cdiv(N, 64), the stock grid's y extent, so coverage is identical).
    # Entry = (first token row << 7) | valid rows (1..64).
    t = tl.program_id(0)
    NT = tl.num_programs(0)
    i = tl.arange(0, BP2)
    m = i < B
    st = tl.load(offs_ptr + i, mask=m, other=0)
    L = tl.load(offs_ptr + i + 1, mask=m, other=0) - st
    nt = tl.where(m, tl.minimum(tl.maximum((L + 63) // 64, 0), NT), 0)
    pos = (tl.cumsum(nt, axis=0) - nt) + t
    p0 = t * 64
    ent = ((st + p0) << 7) | tl.minimum(L - p0, 64)
    tl.store(wl_ptr + pos, ent, mask=(nt > t) & (pos < cap))
    if t == 0:
        tl.store(cnt_ptr, tl.minimum(tl.sum(nt, axis=0), cap).to(tl.int32))


@triton.jit
def _blk_wl_tile(x_ptr, y_ptr, e8_ptr, tok0, nv, H: tl.constexpr,
                 D: tl.constexpr, MASK: tl.constexpr):
    rows = tl.arange(0, 64)
    cols = tl.arange(0, H * D)
    valid = rows < nv
    off = rows[:, None] * (H * D) + cols[None, :]                # int32
    xb = x_ptr + tok0 * (H * D)
    if MASK:
        x = tl.load(xb + off, mask=valid[:, None], other=0.0)
    else:
        x = tl.load(xb + off)
    # Scheduling fence: no instruction, but LLVM must keep it after the
    # tile's loads, so all loads issue before the first use.
    if x.dtype == tl.float32:
        x = tl.inline_asm_elementwise("", "=v,0", [x], dtype=x.dtype,
                                      is_pure=False, pack=1)
    else:
        x = tl.inline_asm_elementwise("", "=v,0", [x], dtype=x.dtype,
                                      is_pure=False, pack=2)
    x3 = tl.reshape(x.to(tl.float32), (64, H, D))
    ram = tl.max(tl.abs(x3), axis=2)                             # [64, H]
    sub0 = ((rows % 16) // 8) == 0                               # [64]
    both = tl.join(tl.where(sub0[:, None], ram, 0.0),
                   tl.where(sub0[:, None], 0.0, ram))             # [64, H, 2]
    a0, a1 = tl.split(tl.max(both, axis=0))                      # [H] each
    e0 = _grp_exp(a0)
    e1 = _grp_exp(a1)
    e_row = tl.where(sub0[:, None], e0[None, :], e1[None, :])
    # 2**e exactly, from the exponent field (e in [-60, 60]).
    s = ((e_row + 127) << 23).to(tl.float32, bitcast=True)       # [64, H]
    y = tl.reshape(x3 * s[:, :, None], (64, H * D)).to(tl.float8e4nv)
    hh = tl.arange(0, H)
    eoff = rows[:, None] * H + hh[None, :]
    e8 = (127 - e_row).to(tl.uint8)
    if MASK:
        tl.store((y_ptr + tok0 * (H * D)) + off, y, mask=valid[:, None])
        tl.store((e8_ptr + tok0 * H) + eoff, e8, mask=valid[:, None])
    else:
        tl.store((y_ptr + tok0 * (H * D)) + off, y)
        tl.store((e8_ptr + tok0 * H) + eoff, e8)


@triton.jit
def _blk_wl_kernel(x_ptr, y_ptr, e8_ptr, wl_ptr, cnt_ptr, H: tl.constexpr,
                   D: tl.constexpr, FULL: tl.constexpr):
    w = tl.program_id(0)
    cnt = tl.load(cnt_ptr)
    e = tl.load(wl_ptr + w)              # w < cap: always in bounds
    if w < cnt:
        tok0 = e >> 7
        nv = e & 127
        if FULL:
            if nv >= 64:
                _blk_wl_tile(x_ptr, y_ptr, e8_ptr, tok0, nv, H, D, False)
            else:
                _blk_wl_tile(x_ptr, y_ptr, e8_ptr, tok0, nv, H, D, True)
        else:
            _blk_wl_tile(x_ptr, y_ptr, e8_ptr, tok0, nv, H, D, True)


def quantize_do_e4m3_blk_wl(dout: torch.Tensor, seq_offsets: torch.Tensor,
                            N: int, full: bool = True, num_warps: int = 16,
                            out=None):
    """Work-list form of `hstu_do_fp8.quantize_do_e4m3_blk` (bitwise).

    dout [T, H, D] bf16/f32, seq_offsets [B+1] (any int dtype, device),
    N >= every sequence length (host int, no sync). Returns (dO8 [T, H, D]
    e4m3, E8 [T, H] uint8 = 127 - e_g). `out` (test hook) = caller-owned
    (dO8, E8) output buffers.
    """
    x = dout.contiguous()
    T, H, D = x.shape
    assert (H * D) & (H * D - 1) == 0 and H & (H - 1) == 0, (H, D)
    if out is None:
        y = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
        e8 = torch.empty((T, H), dtype=torch.uint8, device=x.device)
    else:
        y, e8 = out
    offs = seq_offsets.to(device=x.device, dtype=torch.int64).contiguous()
    B = offs.numel() - 1
    NT = triton.cdiv(int(N), 64)
    # NT == 0: the stock grid (B, 0) launches nothing; so does this (the
    # build kernel would not write cnt).
    if B > 0 and T > 0 and NT > 0:
        cap = triton.cdiv(T + 63 * B, 64)    # >= sum_b cdiv(L_b, 64)
        wl = torch.empty(cap, dtype=torch.int64, device=x.device)
        cnt = torch.empty(1, dtype=torch.int32, device=x.device)
        _blk_wl_build_kernel[(NT,)](
            offs, wl, cnt, B, cap, BP2=triton.next_power_of_2(max(B, 2)),
            num_warps=4)
        _blk_wl_kernel[(cap,)](x, y, e8, wl, cnt, H=H, D=D, FULL=bool(full),
                               num_warps=num_warps)
    return y, e8
