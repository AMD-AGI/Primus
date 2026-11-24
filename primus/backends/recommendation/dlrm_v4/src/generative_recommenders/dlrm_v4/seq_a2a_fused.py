"""Fused passes of the sequence-embedding all-to-all (Triton), bitwise equal to stock.

Used by ``seq_a2a_rowperm.py`` (switch ``DLRM_HSTU_FP8_SEQ_A2A_FUSED``; see that
module for the switches, eligibility and the VERIFY machinery). This module is
pure math: no process group and no torchrec/fbgemm import.

Kernels (names start with ``_seq_a2a_`` so traces and bench EVIDENCE_KERNELS can
name them):

* ``_seq_a2a_send_fwd_kernel`` (K1, forward, before the a2a). For every wire row
  ``o``: ``wire[o, :] = clamp16(fp16_rne(emb[fidx[o], :]))``. The same launch
  also writes the permuted D-scaled lengths ``tok_lengths.view(n_blocks, -1)[recat] * D``
  (``a2ai.permuted_lengths_after_sparse_data_all2all``).
  Replaces: the lengths ``mul`` and ``index_select``, ``permute_2D_sparse_data``
  over an arange (row index), the encode ``.half()``, ``clamp_`` and the wire gather.
* ``_seq_a2a_send_bwd_kernel`` (K2, backward, after the a2a). For every source
  row ``q``: ``grad[q, :] = float(wire[bidx[q], :]) * 2**-k`` (``W = 2**k``; no
  multiply when GRADIENT_DIVISION is off). Replaces: ``permute_2D_sparse_data``
  over an arange, the wire gather, the decode ``.float()`` and ``div_(W)``.
* ``_seq_a2a_recv_fwd_kernel`` (K3', forward, after the a2a wait; receive tier 1).
  For every output row ``u``: ``out[u, :] = float(wire[perm[u], :])`` with the
  row-wise unbucketize permutation ``perm`` handed over by
  ``unbucketize_gather.py``, or ``out[u, :] = float(wire[u, :])`` without one.
  Replaces: the codec decode ``.float()`` and the unbucketize
  ``index_select`` (``_PermuteRows`` forward).
* ``_seq_a2a_recv_bwd_kernel`` (K4, backward, before the backward a2a). For every
  gradient row ``u``: ``wire_bwd[perm[u], :] = clamp16(fp16_rne(grad[u, :]))``
  (a row SCATTER; ``perm`` is a bijection, so every wire row is written exactly
  once), or ``wire_bwd[u, :] = ...`` without a permutation. Replaces: the
  ``_PermuteRows`` backward (inverse-permutation ``index_copy_`` + row gather)
  and the codec encode ``.half()`` + ``clamp_``.
* ``_seq_a2a_rowmap_{fwd,bwd}_kernel``: test/diagnostic only. They write the
  source row index of every output row with the same index helpers as K1/K2.

Index math. ``blk[k]`` = rows of input block k (``aten::sum`` over
``tok_lengths.view(n_blocks, -1)``, the one prologue launch, kept by the caller).
Every program recomputes, redundantly and in registers, the block tables
(n_blocks <= 1024 entries; production <= 449):

* forward:  ``plen = blk[recat]``, ``out_start = excl_scan(plen)``,
  ``in_start = excl_scan(blk)``; for wire row o in block p:
  ``fidx[o] = in_start[recat[p]] + (o - out_start[p])``.
* backward: the wire blocks have sizes ``plen`` and starts ``out_start``; the
  gradient block j is wire block ``bwd_recat[j]``, so its size is
  ``plen[bwd_recat[j]]`` and ``bstart = excl_scan(plen[bwd_recat])``; for grad
  row q in block j: ``bidx[q] = out_start[bwd_recat[j]] + (q - bstart[j])``.

These are exactly the values ``permute_2D_sparse_data(recat, blk[n,1], arange)``
and ``permute_2D_sparse_data(bwd_recat, blk[recat][n,1], arange)`` write (the
shipped ``row_index``); no assumption about ``bwd_recat`` being the inverse of
``recat`` is made (torchrec builds it as the inverse; the tests check both).
The block p of a row o is found with upper-bound semantics:
``p = #{k < n_blocks : out_start[k] <= o} - 1``, i.e. the LAST block whose start
is <= o. Empty blocks share their start with the next block, so they are never
selected for a row (a "first block with start <= o" search would pick them).
The search is a [BLOCK_R, BLOCK_N] compare + reduce per row tile (no binary
search, no LDS gathers), and is done per row, so tiles that straddle block
boundaries are exact.

Bitwise argument (also see the tests and gpu_check.py):

* Each output element is a copy of exactly one input element, and every
  transform (encode, decode, scale) is elementwise, so gather-then-transform
  equals transform-then-gather.
* fp32 -> fp16: ``x.to(tl.float16)`` is LLVM ``fptrunc`` (RNE), the same as ATen's
  ``float16_copy_kernel_cuda`` (``c10::Half(float)`` -> ``__float2half`` ->
  ``static_cast<_Float16>``); on gfx950 both select ``v_cvt[_pk]_f16_f32``.
* ``clamp_(-65504, 65504)`` on a half tensor (ATen ``launch_clamp_scalar``):
  NaN is returned unchanged (``if (_isnan(v)) return v``); any other half v is
  ``min(max(float(v), lo), hi)`` converted back. Every finite half lies in
  [-65504, 65504] and round-trips exactly (incl. -0.0 and subnormals), so the
  only non-NaN values that change are +inf -> 0x7BFF and -inf -> 0xFBFF. K1
  implements exactly that as an integer pattern replacement on the fp16 bits.
  The stock fbgemm order (``clamp`` in fp32, then ``.half()``) gives the same
  bits: |x| in (65504, 65520) rounds to 65504, |x| >= 65520 and inf saturate to
  65504 either way, NaN passes both.
* fp16 -> fp32 is exact (``.float()``).
* GRADIENT_DIVISION, W = 2**k: ATen's ``div_`` by a CPU scalar computes either
  ``x * (1.0f / W)`` (MulFunctor) or ``x / W`` (DivFunctor, correctly rounded).
  For a power of two both equal the exact value ``x * 2**-k`` whenever it is a
  normal fp32: the inputs are decoded fp16 values (|x| >= 2**-24 or 0), so
  |x / W| >= 2**-24-k is normal for k <= 101. ±inf stays ±inf, ±0 stays ±0, and
  NaN is propagated as a quiet NaN by both instructions (checked exhaustively
  over all 65,536 fp16 patterns in gpu_check.py). K2 multiplies by the exact
  constant 2**-k (``scale_op="div"`` selects an IEEE division instead, for the
  GPU check).

Receive side (K3', K4), bitwise argument:

* K3' forward: stock is ``index_select(decode(wire).view(-1, D), 0, perm)``. Each
  output element is one decoded wire element (exact fp16 -> fp32 upcast), and
  a row copy changes no bits, so ``float(wire[perm[u]])`` is the same value for
  ANY in-range ``perm`` (duplicates included). Without a permutation it is the
  decode itself.
* K4 backward: stock (``_PermuteRows`` gather mode) is ``inv[perm[u]] = u``,
  ``dx = grad[inv]``, then ``encode(dx)``, so ``wire[j] = enc(grad[inv[j]])``. For
  ``j = perm[u]``, ``inv[j] = u``: ``wire[perm[u]] = enc(grad[u])``, and because
  ``perm`` is a bijection every ``j`` is some ``perm[u]``. The scatter mode
  (``dx.index_copy_(0, perm, grad)``) writes the same values. ``enc`` is K1's
  element op (``_seq_a2a_enc16``: RNE cast, then the NaN-passing clamp),
  verified against both fp16 encodes over all 2**32 fp32 patterns.
* K4 equals ONLY the ``_PermuteRows`` backward. The stock ``index_select``
  backward (``zeros(...).index_add_(0, perm, g)``) adds the gradient to +0.0,
  which turns -0.0 into +0.0 (and quiets signaling NaNs). The caller therefore
  engages K4 only where ``unbucketize_gather._unbucketize`` would have chosen
  ``_PermuteRows`` (int64 permutation, autograd recording, bijectivity check
  passed, module not disabled); everywhere else it runs the stock ops.
* Out-of-range ``perm`` entries (impossible for a checked bijection; stock
  ``index_select`` would raise a device-side assert): K3' writes NaN rows and
  K4 skips the store, so neither reads or writes out of bounds.
"""

import os
from typing import Dict, Optional, Tuple

import torch

try:  # Triton is optional at import time; callers check available().
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAVE_TRITON = False

# Largest block count handled by the per-program tables (BLOCK_N is the next
# power of two). Production: 16 (1n TW), 224 (2n), 448 (4n) blocks.
MAX_BLOCKS = 1024
# Default tiling (gpu_check.py sweeps these; see build/README.md). Picked from the
# iter-34 MI350X sweep (comm probes/iter34/gpucheck/gpu_check_sweep.json; rows 40k /
# 335k, n_blocks 24 / 352): one tile per program (no persistent loop) was 8-14%
# faster than 8 programs per CU for both kernels; K1 best at 8 rows x 512 (4 warps),
# K2 at 16 rows x 512 (4 warps). Tiling never changes a value: every output
# element is computed by the same per-element expression whatever the tile.
DEFAULT_TILE_ELEMS = 4096  # K1: BLOCK_R * BLOCK_D per tile
DEFAULT_TILE_ELEMS_BWD = 8192  # K2
DEFAULT_NUM_WARPS = 4
DEFAULT_PROGS_PER_SM = 1 << 16  # programs = min(tiles, SMs * this): one tile each
# Receive kernels (K3', K4): one tile per program, BLOCK_R rows x BLOCK_D cols.
# Picked from the iter-35 MI350X tile sweep (comm probes/iter35/recv/README.md,
# build/gpu_check_bits_sweep.json): at D=512 two rows per program was best for
# both kernels (K3' 2 warps, K4 4 warps; 4-6% over 8 rows x 4 warps); at D=128
# the same 1024-element tile gives 8 rows, also at the top. Tiling never
# changes a value.
DEFAULT_TILE_ELEMS_RECV_FWD = 1024  # K3'
DEFAULT_TILE_ELEMS_RECV_BWD = 1024  # K4
DEFAULT_NUM_WARPS_RECV_FWD = 2
DEFAULT_NUM_WARPS_RECV_BWD = 4
_MAX_BLOCK_R = 32
_MAX_BLOCK_D = 512
_MAX_BLOCK_B = 1024

_num_sms_cache: Dict[int, int] = {}


def available() -> bool:
    return _HAVE_TRITON


def interpret_mode() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0").strip() == "1"


def device_ok(t: torch.Tensor) -> bool:
    """Compiled kernels need a GPU tensor; CPU tensors only under TRITON_INTERPRET=1."""
    return t.is_cuda or (t.device.type == "cpu" and interpret_mode())


def is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _num_sms(device: torch.device) -> int:
    if device.type != "cuda":
        return 4
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _num_sms_cache.get(idx)
    if n is None:
        n = int(torch.cuda.get_device_properties(idx).multi_processor_count)
        _num_sms_cache[idx] = n
    return n


def _tiling(
    rows: int,
    D: int,
    n_blocks: int,
    device: torch.device,
    block_r: Optional[int],
    progs_per_sm: Optional[int],
    tile_elems: int = DEFAULT_TILE_ELEMS,
) -> Tuple[int, int, int, int, int]:
    """(BLOCK_N, BLOCK_D, BLOCK_R, tiles_per_prog, grid)."""
    block_n = max(16, _next_pow2(n_blocks))
    block_d = min(_MAX_BLOCK_D, _next_pow2(D))
    if block_r is None:
        block_r = max(1, min(_MAX_BLOCK_R, tile_elems // block_d))
    n_tiles = max(1, _cdiv(rows, block_r))
    max_progs = max(1, _num_sms(device) * (progs_per_sm or DEFAULT_PROGS_PER_SM))
    tiles_per_prog = max(1, _cdiv(n_tiles, max_progs))
    grid = max(1, _cdiv(n_tiles, tiles_per_prog))
    return block_n, block_d, block_r, tiles_per_prog, grid


class _DeviceGuard:
    """Launch on the tensors' device (Triton launches on the current device)."""

    def __init__(self, t: torch.Tensor) -> None:
        self._ctx = (
            torch.cuda.device(t.device)
            if t.is_cuda and t.device.index != torch.cuda.current_device()
            else None
        )

    def __enter__(self) -> None:
        if self._ctx is not None:
            self._ctx.__enter__()

    def __exit__(self, *exc) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(*exc)


if _HAVE_TRITON:
    # fp16 bit patterns as int16 (globals used in @jit code must be constexpr).
    _FP16_POS_INF = tl.constexpr(0x7C00)
    _FP16_POS_MAX = tl.constexpr(0x7BFF)
    _FP16_NEG_INF = tl.constexpr(-0x400)  # 0xFC00
    _FP16_NEG_MAX = tl.constexpr(-0x401)  # 0xFBFF

    @triton.jit
    def _seq_a2a_enc16(xv):
        """fp32 -> fp16 bit pattern: the fp16 codec encode (K1 and K4 share it).

        RNE cast, then clamp_(+-65504) == {+inf -> 0x7BFF, -inf -> 0xFBFF}; NaN
        bits untouched. The caller stores ``.to(tl.int16)`` of the result.
        """
        hb = xv.to(tl.float16).to(tl.int16, bitcast=True)
        hb = tl.where(hb == _FP16_POS_INF, _FP16_POS_MAX, hb)
        hb = tl.where(hb == _FP16_NEG_INF, _FP16_NEG_MAX, hb)
        return hb

    @triton.jit
    def _seq_a2a_ld_idx(ptr, idx, ok, n_blocks):
        """Load block indices ptr[idx] as int32; out-of-range values -> (0, False)."""
        v = tl.load(ptr + idx, mask=ok, other=0).to(tl.int32)
        ok2 = ok & (v >= 0) & (v < n_blocks)
        return tl.where(ok2, v, 0), ok2

    @triton.jit
    def _seq_a2a_fwd_tables(blk_ptr, recat_ptr, n_blocks, BLOCK_N: tl.constexpr):
        """Per-program forward tables over the n_blocks blocks (int32)."""
        k = tl.arange(0, BLOCK_N)
        kmask = k < n_blocks
        rec, rok = _seq_a2a_ld_idx(recat_ptr, k, kmask, n_blocks)
        plen = tl.load(blk_ptr + rec, mask=rok, other=0).to(tl.int32)  # blk[recat]
        ostart = tl.cumsum(plen, axis=0) - plen  # wire block starts
        blkk = tl.load(blk_ptr + k, mask=kmask, other=0).to(tl.int32)
        return k, kmask, ostart, blkk

    @triton.jit
    def _seq_a2a_fwd_src(o, k, kmask, ostart, blkk, recat_ptr, n_blocks):
        """Stock fidx[o] for a vector of wire rows o."""
        # Upper bound: p = last block whose start <= o (skips empty blocks).
        le = (ostart[None, :] <= o[:, None]) & kmask[None, :]
        p = tl.sum(le.to(tl.int32), axis=1) - 1
        os_p = tl.max(tl.where(le, ostart[None, :], 0), axis=1)  # out_start[p]
        rp, _ = _seq_a2a_ld_idx(recat_ptr, p, p >= 0, n_blocks)
        ins = tl.sum(
            tl.where((k[None, :] < rp[:, None]) & kmask[None, :], blkk[None, :], 0),
            axis=1,
        )  # in_start[recat[p]]
        return ins + (o - os_p)

    @triton.jit
    def _seq_a2a_bwd_tables(
        blk_ptr, recat_ptr, bwd_recat_ptr, n_blocks, BLOCK_N: tl.constexpr
    ):
        """Per-program backward tables (int32)."""
        k = tl.arange(0, BLOCK_N)
        kmask = k < n_blocks
        rec, rok = _seq_a2a_ld_idx(recat_ptr, k, kmask, n_blocks)
        plen = tl.load(blk_ptr + rec, mask=rok, other=0).to(tl.int32)  # wire block sizes
        brec, bok = _seq_a2a_ld_idx(bwd_recat_ptr, k, kmask, n_blocks)
        rr, rrok = _seq_a2a_ld_idx(recat_ptr, brec, bok, n_blocks)
        bl = tl.load(blk_ptr + rr, mask=rrok, other=0).to(tl.int32)  # plen[bwd_recat]
        bstart = tl.cumsum(bl, axis=0) - bl  # grad block starts
        return k, kmask, plen, bstart

    @triton.jit
    def _seq_a2a_bwd_src(q, k, kmask, plen, bstart, bwd_recat_ptr, n_blocks):
        """Stock bidx[q] for a vector of grad (source-order) rows q."""
        le = (bstart[None, :] <= q[:, None]) & kmask[None, :]
        j = tl.sum(le.to(tl.int32), axis=1) - 1
        bs_j = tl.max(tl.where(le, bstart[None, :], 0), axis=1)  # bstart[j]
        m, _ = _seq_a2a_ld_idx(bwd_recat_ptr, j, j >= 0, n_blocks)
        wst = tl.sum(
            tl.where((k[None, :] < m[:, None]) & kmask[None, :], plen[None, :], 0),
            axis=1,
        )  # out_start[bwd_recat[j]]
        return wst + (q - bs_j)

    # Runtime ints change every step (rows) or per call; do not specialize on them
    # (%16 / ==1), so a new value never triggers a recompile mid-run. The hot
    # address math uses the constexpr D, so vectorization does not depend on them.
    @triton.jit(do_not_specialize=["rows", "n_blocks", "B", "tiles_per_prog"])
    def _seq_a2a_send_fwd_kernel(
        x_ptr,  # fp32 [rows, D] (TBE output)
        wire_ptr,  # int16 view of the fp16 wire [rows, D]
        blk_ptr,  # int64 [n_blocks] rows per input block
        recat_ptr,  # int32/int64 [n_blocks]
        tok_ptr,  # token lengths [n_blocks, B]
        plen_ptr,  # out: permuted D-scaled lengths [n_blocks, B]
        rows,
        n_blocks,
        B,
        tiles_per_prog,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        # (1) permuted D-scaled lengths: plen[i, :] = tok[recat[i], :] * D.
        cb = tl.arange(0, BLOCK_B)
        for i in range(pid, n_blocks, nprog):
            r = tl.load(recat_ptr + i).to(tl.int32)
            rok = (r >= 0) & (r < n_blocks)
            r = tl.where(rok, r, 0)
            for b0 in range(0, B, BLOCK_B):
                bb = b0 + cb
                bm = bb < B
                t = tl.load(tok_ptr + r * B + bb, mask=bm & rok, other=0)
                tl.store(plen_ptr + i * B + bb, t * D, mask=bm)
        # (2) wire rows.
        k, kmask, ostart, blkk = _seq_a2a_fwd_tables(blk_ptr, recat_ptr, n_blocks, BLOCK_N)
        rr = tl.arange(0, BLOCK_R)
        cc = tl.arange(0, BLOCK_D)
        for t in range(tiles_per_prog):
            o = (pid * tiles_per_prog + t) * BLOCK_R + rr
            omask = o < rows
            src = _seq_a2a_fwd_src(o, k, kmask, ostart, blkk, recat_ptr, n_blocks)
            lmask = omask & (src >= 0) & (src < rows)
            s_off = src.to(tl.int64) * D
            o_off = o.to(tl.int64) * D
            for c0 in tl.static_range(0, D, BLOCK_D):
                cols = c0 + cc
                if c0 + BLOCK_D <= D:
                    lm = lmask[:, None]
                    sm = omask[:, None]
                else:
                    cm = cols < D
                    lm = lmask[:, None] & cm[None, :]
                    sm = omask[:, None] & cm[None, :]
                xv = tl.load(x_ptr + (s_off[:, None] + cols[None, :]), mask=lm, other=0.0)
                # RNE cast, then clamp_(+-65504) == {+inf -> 0x7BFF, -inf -> 0xFBFF},
                # NaN bits untouched.
                hb = _seq_a2a_enc16(xv)
                tl.store(
                    wire_ptr + (o_off[:, None] + cols[None, :]), hb.to(tl.int16), mask=sm
                )

    @triton.jit(do_not_specialize=["rows", "n_blocks", "tiles_per_prog"])
    def _seq_a2a_send_bwd_kernel(
        wire_ptr,  # fp16 [rows, D] (backward wire, permuted order)
        g_ptr,  # out: fp32 [rows, D] (source order)
        blk_ptr,
        recat_ptr,
        bwd_recat_ptr,
        rows,
        n_blocks,
        tiles_per_prog,
        scale,  # fp32: 2**-k (mul) or W (div)
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
        APPLY_SCALE: tl.constexpr,
        SCALE_DIV: tl.constexpr,
    ):
        pid = tl.program_id(0)
        k, kmask, plen, bstart = _seq_a2a_bwd_tables(
            blk_ptr, recat_ptr, bwd_recat_ptr, n_blocks, BLOCK_N
        )
        rr = tl.arange(0, BLOCK_R)
        cc = tl.arange(0, BLOCK_D)
        for t in range(tiles_per_prog):
            q = (pid * tiles_per_prog + t) * BLOCK_R + rr
            qmask = q < rows
            src = _seq_a2a_bwd_src(q, k, kmask, plen, bstart, bwd_recat_ptr, n_blocks)
            lmask = qmask & (src >= 0) & (src < rows)
            s_off = src.to(tl.int64) * D
            q_off = q.to(tl.int64) * D
            for c0 in tl.static_range(0, D, BLOCK_D):
                cols = c0 + cc
                if c0 + BLOCK_D <= D:
                    lm = lmask[:, None]
                    sm = qmask[:, None]
                else:
                    cm = cols < D
                    lm = lmask[:, None] & cm[None, :]
                    sm = qmask[:, None] & cm[None, :]
                w = tl.load(wire_ptr + (s_off[:, None] + cols[None, :]), mask=lm, other=0.0)
                g = w.to(tl.float32)
                if APPLY_SCALE:
                    if SCALE_DIV:
                        g = tl.math.div_rn(g, scale)
                    else:
                        g = g * scale
                tl.store(g_ptr + (q_off[:, None] + cols[None, :]), g, mask=sm)

    # U (received rows) changes every step: never specialize on it (no mid-run
    # re-JIT). The hot address math uses the constexpr D.
    @triton.jit(do_not_specialize=["U"])
    def _seq_a2a_recv_fwd_kernel(
        wire_ptr,  # fp16 [U, D]: the received forward wire (a2a output order)
        perm_ptr,  # int64/int32 [U]: unbucketize permutation (read iff HAS_PERM)
        out_ptr,  # out: fp32 [U, D]
        U,
        D: tl.constexpr,
        HAS_PERM: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rr = tl.arange(0, BLOCK_R)
        cc = tl.arange(0, BLOCK_D)
        u = pid * BLOCK_R + rr
        umask = u < U
        if HAS_PERM:
            src = tl.load(perm_ptr + u, mask=umask, other=0).to(tl.int64)
            lmask = umask & (src >= 0) & (src < U)
        else:
            src = u.to(tl.int64)
            lmask = umask
        s_off = src * D
        o_off = u.to(tl.int64) * D
        for c0 in tl.static_range(0, D, BLOCK_D):
            cols = c0 + cc
            if c0 + BLOCK_D <= D:
                lm = lmask[:, None]
                sm = umask[:, None]
            else:
                cm = cols < D
                lm = lmask[:, None] & cm[None, :]
                sm = umask[:, None] & cm[None, :]
            # Out-of-range perm entries (never for a checked bijection) read NaN.
            # Cache hints (value-neutral): the wire is read exactly once (".cg",
            # do not keep it in L2); the output is read next by the EC expand,
            # written through (".wt"; a non-temporal store pushed it out of MALL
            # and slowed that consumer, iter-35 cm_experiment2).
            w = tl.load(
                wire_ptr + (s_off[:, None] + cols[None, :]),
                mask=lm,
                other=float("nan"),
                cache_modifier=".cg",
            )
            tl.store(
                out_ptr + (o_off[:, None] + cols[None, :]),
                w.to(tl.float32),
                mask=sm,
                cache_modifier=".wt",
            )

    @triton.jit(do_not_specialize=["U"])
    def _seq_a2a_recv_bwd_kernel(
        g_ptr,  # fp32 [U, D] contiguous: gradient of the (permuted) wait output
        perm_ptr,  # int64/int32 [U] bijection (read iff HAS_PERM)
        wire_ptr,  # out: int16 view of the fp16 backward wire [U, D]
        U,
        D: tl.constexpr,
        HAS_PERM: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rr = tl.arange(0, BLOCK_R)
        cc = tl.arange(0, BLOCK_D)
        u = pid * BLOCK_R + rr
        umask = u < U
        if HAS_PERM:
            dst = tl.load(perm_ptr + u, mask=umask, other=0).to(tl.int64)
            smask = umask & (dst >= 0) & (dst < U)  # never write out of bounds
        else:
            dst = u.to(tl.int64)
            smask = umask
        g_off = u.to(tl.int64) * D
        d_off = dst * D
        for c0 in tl.static_range(0, D, BLOCK_D):
            cols = c0 + cc
            if c0 + BLOCK_D <= D:
                lm = umask[:, None]
                sm = smask[:, None]
            else:
                cm = cols < D
                lm = umask[:, None] & cm[None, :]
                sm = smask[:, None] & cm[None, :]
            # Cache hints (value-neutral): the gradient is read exactly once
            # (".cg"); the wire is read next by the backward all-to-all, written
            # through (".wt", best with that read in cm_experiment2).
            xv = tl.load(
                g_ptr + (g_off[:, None] + cols[None, :]), mask=lm, other=0.0, cache_modifier=".cg"
            )
            hb = _seq_a2a_enc16(xv)
            tl.store(
                wire_ptr + (d_off[:, None] + cols[None, :]),
                hb.to(tl.int16),
                mask=sm,
                cache_modifier=".wt",
            )

    @triton.jit(do_not_specialize=["rows", "n_blocks", "tiles_per_prog"])
    def _seq_a2a_rowmap_fwd_kernel(
        out_ptr, blk_ptr, recat_ptr, rows, n_blocks, tiles_per_prog,
        BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        k, kmask, ostart, blkk = _seq_a2a_fwd_tables(blk_ptr, recat_ptr, n_blocks, BLOCK_N)
        rr = tl.arange(0, BLOCK_R)
        for t in range(tiles_per_prog):
            o = (pid * tiles_per_prog + t) * BLOCK_R + rr
            src = _seq_a2a_fwd_src(o, k, kmask, ostart, blkk, recat_ptr, n_blocks)
            tl.store(out_ptr + o, src.to(tl.int64), mask=o < rows)

    @triton.jit(do_not_specialize=["rows", "n_blocks", "tiles_per_prog"])
    def _seq_a2a_rowmap_bwd_kernel(
        out_ptr, blk_ptr, recat_ptr, bwd_recat_ptr, rows, n_blocks, tiles_per_prog,
        BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        k, kmask, plen, bstart = _seq_a2a_bwd_tables(
            blk_ptr, recat_ptr, bwd_recat_ptr, n_blocks, BLOCK_N
        )
        rr = tl.arange(0, BLOCK_R)
        for t in range(tiles_per_prog):
            q = (pid * tiles_per_prog + t) * BLOCK_R + rr
            src = _seq_a2a_bwd_src(q, k, kmask, plen, bstart, bwd_recat_ptr, n_blocks)
            tl.store(out_ptr + q, src.to(tl.int64), mask=q < rows)


# --------------------------------------------------------------------------
# Launchers
# --------------------------------------------------------------------------
def _check_common(blk: torch.Tensor, recat: torch.Tensor, n_blocks: int) -> None:
    if not _HAVE_TRITON:
        raise RuntimeError("seq_a2a_fused: triton is not available")
    if not (1 <= n_blocks <= MAX_BLOCKS):
        raise ValueError(f"seq_a2a_fused: n_blocks={n_blocks} not in [1, {MAX_BLOCKS}]")
    if blk.dtype != torch.int64 or blk.numel() != n_blocks or not blk.is_contiguous():
        raise ValueError("seq_a2a_fused: blk must be contiguous int64 [n_blocks]")
    if recat.dtype not in (torch.int32, torch.int64) or recat.numel() != n_blocks:
        raise ValueError("seq_a2a_fused: recat must be int32/int64 [n_blocks]")


def send_fwd(
    emb: torch.Tensor,
    blk: torch.Tensor,
    recat: torch.Tensor,
    tok_lengths: torch.Tensor,
    n_blocks: int,
    D: int,
    *,
    block_r: Optional[int] = None,
    num_warps: Optional[int] = None,
    progs_per_sm: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """K1. Returns (permuted D-scaled lengths [n_blocks, B], fp16 wire 1-D [rows*D]).

    Bitwise equal to ``index_select((tok_lengths*D).view(n_blocks,-1), 0, recat)``
    and to ``encode(emb.view(-1)).view(rows, D)[fidx].view(-1)`` with the fp16
    clamp codec (stock fbgemm or the lowmem cast-then-clamp_ patch).
    """
    _check_common(blk, recat, n_blocks)
    if emb.dtype != torch.float32 or not emb.is_contiguous() or emb.numel() % D:
        raise ValueError("seq_a2a_fused.send_fwd: emb must be contiguous fp32, numel % D == 0")
    if not tok_lengths.is_contiguous() or tok_lengths.numel() % n_blocks:
        raise ValueError("seq_a2a_fused.send_fwd: tok_lengths must be contiguous, numel % n_blocks == 0")
    rows = emb.numel() // D
    bsz = tok_lengths.numel() // n_blocks
    wire = torch.empty(rows * D, dtype=torch.float16, device=emb.device)
    plen = torch.empty((n_blocks, bsz), dtype=tok_lengths.dtype, device=tok_lengths.device)
    block_n, block_d, block_r, tpp, grid = _tiling(
        rows, D, n_blocks, emb.device, block_r, progs_per_sm
    )
    block_b = min(_MAX_BLOCK_B, _next_pow2(max(1, bsz)))
    with _DeviceGuard(emb):
        _seq_a2a_send_fwd_kernel[(grid,)](
            emb,
            wire.view(torch.int16),
            blk,
            recat,
            tok_lengths,
            plen,
            rows,
            n_blocks,
            bsz,
            tpp,
            D=D,
            BLOCK_N=block_n,
            BLOCK_R=block_r,
            BLOCK_D=block_d,
            BLOCK_B=block_b,
            num_warps=num_warps or DEFAULT_NUM_WARPS,
        )
    return plen, wire


def send_bwd(
    wire: torch.Tensor,
    blk: torch.Tensor,
    recat: torch.Tensor,
    bwd_recat: torch.Tensor,
    D: int,
    world_size: Optional[int],
    *,
    scale_op: str = "mul",
    block_r: Optional[int] = None,
    num_warps: Optional[int] = None,
    progs_per_sm: Optional[int] = None,
) -> torch.Tensor:
    """K2. Returns the fp32 grad 1-D [rows*D] in source order.

    Bitwise equal to ``decode(wire).view(rows, D)[bidx].view(-1)`` followed by
    ``div_(world_size)`` (``world_size=None``: GRADIENT_DIVISION off, no scaling).
    ``world_size`` must be a power of two.
    """
    n_blocks = recat.numel()
    _check_common(blk, recat, n_blocks)
    if bwd_recat.dtype not in (torch.int32, torch.int64) or bwd_recat.numel() != n_blocks:
        raise ValueError("seq_a2a_fused.send_bwd: bwd_recat must be int32/int64 [n_blocks]")
    if wire.dtype != torch.float16 or not wire.is_contiguous() or wire.numel() % D:
        raise ValueError("seq_a2a_fused.send_bwd: wire must be contiguous fp16, numel % D == 0")
    if world_size is not None and not is_pow2(int(world_size)):
        raise ValueError(f"seq_a2a_fused.send_bwd: world_size={world_size} is not a power of two")
    if scale_op not in ("mul", "div"):
        raise ValueError(f"seq_a2a_fused.send_bwd: scale_op={scale_op!r}")
    rows = wire.numel() // D
    g = torch.empty(rows * D, dtype=torch.float32, device=wire.device)
    block_n, block_d, block_r, tpp, grid = _tiling(
        rows, D, n_blocks, wire.device, block_r, progs_per_sm, DEFAULT_TILE_ELEMS_BWD
    )
    apply_scale = world_size is not None
    if not apply_scale:
        scale = 1.0
    elif scale_op == "mul":
        scale = 1.0 / float(world_size)  # exact: 2**-k
    else:
        scale = float(world_size)
    with _DeviceGuard(wire):
        _seq_a2a_send_bwd_kernel[(grid,)](
            wire,
            g,
            blk,
            recat,
            bwd_recat,
            rows,
            n_blocks,
            tpp,
            scale,
            D=D,
            BLOCK_N=block_n,
            BLOCK_R=block_r,
            BLOCK_D=block_d,
            APPLY_SCALE=apply_scale,
            SCALE_DIV=(scale_op == "div"),
            num_warps=num_warps or DEFAULT_NUM_WARPS,
        )
    return g


def _recv_tiling(U: int, D: int, block_r: Optional[int], tile_elems: int) -> Tuple[int, int, int]:
    """(BLOCK_D, BLOCK_R, grid) for K3'/K4: one tile of BLOCK_R rows per program."""
    block_d = min(_MAX_BLOCK_D, _next_pow2(D))
    if block_r is None:
        block_r = max(1, min(_MAX_BLOCK_R, tile_elems // block_d))
    return block_d, block_r, max(1, _cdiv(U, block_r))


def _check_recv_perm(perm: torch.Tensor, U: int, device: torch.device) -> None:
    if (
        perm.dtype not in (torch.int32, torch.int64)
        or perm.dim() != 1
        or perm.numel() != U
        or not perm.is_contiguous()
        or perm.device != device
    ):
        raise ValueError(
            "seq_a2a_fused: perm must be a contiguous 1-D int32/int64 [U] tensor on the data device"
        )


def recv_fwd(
    wire: torch.Tensor,
    perm: Optional[torch.Tensor],
    D: int,
    *,
    block_r: Optional[int] = None,
    num_warps: Optional[int] = None,
    tile_elems: Optional[int] = None,
) -> torch.Tensor:
    """K3'. Returns fp32 [U, D] with ``out[u] = float(wire[perm[u]])`` (``perm=None``: ``wire[u]``).

    Bitwise equal to ``index_select(wire.float().view(-1, D), 0, perm)`` (resp.
    ``wire.float().view(-1, D)``) for every in-range ``perm``.
    """
    if not _HAVE_TRITON:
        raise RuntimeError("seq_a2a_fused: triton is not available")
    if wire.dtype != torch.float16 or not wire.is_contiguous() or D <= 0 or wire.numel() % D:
        raise ValueError("seq_a2a_fused.recv_fwd: wire must be contiguous fp16, numel % D == 0")
    U = wire.numel() // D
    if perm is not None:
        _check_recv_perm(perm, U, wire.device)
    out = torch.empty((U, D), dtype=torch.float32, device=wire.device)
    if U == 0:
        return out
    block_d, block_r, grid = _recv_tiling(
        U, D, block_r, tile_elems or DEFAULT_TILE_ELEMS_RECV_FWD
    )
    with _DeviceGuard(wire):
        _seq_a2a_recv_fwd_kernel[(grid,)](
            wire,
            perm if perm is not None else wire,  # placeholder, never read
            out,
            U,
            D=D,
            HAS_PERM=perm is not None,
            BLOCK_R=block_r,
            BLOCK_D=block_d,
            num_warps=num_warps or DEFAULT_NUM_WARPS_RECV_FWD,
        )
    return out


def recv_bwd(
    grad: torch.Tensor,
    perm: Optional[torch.Tensor],
    D: int,
    *,
    block_r: Optional[int] = None,
    num_warps: Optional[int] = None,
    tile_elems: Optional[int] = None,
) -> torch.Tensor:
    """K4. Returns the fp16 backward wire [U, D]: ``wire[perm[u]] = enc(grad[u])``.

    ``perm`` must be a bijection on [0, U) (every wire row is written exactly
    once); ``perm=None`` writes ``wire[u] = enc(grad[u])``. Bitwise equal to
    ``encode(dx)`` with ``dx`` the ``_PermuteRows`` backward (gather or scatter
    mode) of ``grad``, for the fp16 clamp codec (stock fbgemm or the lowmem
    cast-then-``clamp_`` patch). ``grad`` must be contiguous fp32.
    """
    if not _HAVE_TRITON:
        raise RuntimeError("seq_a2a_fused: triton is not available")
    if grad.dtype != torch.float32 or not grad.is_contiguous() or D <= 0 or grad.numel() % D:
        raise ValueError("seq_a2a_fused.recv_bwd: grad must be contiguous fp32, numel % D == 0")
    U = grad.numel() // D
    if perm is not None:
        _check_recv_perm(perm, U, grad.device)
    wire = torch.empty((U, D), dtype=torch.float16, device=grad.device)
    if U == 0:
        return wire
    block_d, block_r, grid = _recv_tiling(
        U, D, block_r, tile_elems or DEFAULT_TILE_ELEMS_RECV_BWD
    )
    with _DeviceGuard(grad):
        _seq_a2a_recv_bwd_kernel[(grid,)](
            grad,
            perm if perm is not None else grad,  # placeholder, never read
            wire.view(torch.int16),
            U,
            D=D,
            HAS_PERM=perm is not None,
            BLOCK_R=block_r,
            BLOCK_D=block_d,
            num_warps=num_warps or DEFAULT_NUM_WARPS_RECV_BWD,
        )
    return wire


def fwd_row_map(blk: torch.Tensor, recat: torch.Tensor, rows: int) -> torch.Tensor:
    """Source row of every wire row (== stock fidx), with K1's index helpers."""
    n_blocks = recat.numel()
    _check_common(blk, recat, n_blocks)
    out = torch.empty(rows, dtype=torch.int64, device=blk.device)
    block_n, _, block_r, tpp, grid = _tiling(rows, 512, n_blocks, blk.device, 16, None)
    with _DeviceGuard(blk):
        _seq_a2a_rowmap_fwd_kernel[(grid,)](
            out, blk, recat, rows, n_blocks, tpp, BLOCK_N=block_n, BLOCK_R=block_r
        )
    return out


def bwd_row_map(
    blk: torch.Tensor, recat: torch.Tensor, bwd_recat: torch.Tensor, rows: int
) -> torch.Tensor:
    """Wire row of every grad row (== stock bidx), with K2's index helpers."""
    n_blocks = recat.numel()
    _check_common(blk, recat, n_blocks)
    out = torch.empty(rows, dtype=torch.int64, device=blk.device)
    block_n, _, block_r, tpp, grid = _tiling(rows, 512, n_blocks, blk.device, 16, None)
    with _DeviceGuard(blk):
        _seq_a2a_rowmap_bwd_kernel[(grid,)](
            out, blk, recat, bwd_recat, rows, n_blocks, tpp, BLOCK_N=block_n, BLOCK_R=block_r
        )
    return out
