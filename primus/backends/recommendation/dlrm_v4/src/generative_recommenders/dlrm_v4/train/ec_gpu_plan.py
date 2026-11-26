"""GPU planning for EC segmented reduction.

Hot-piece counts and their prefix sums stay on the caller's current stream.
The reducer preserves the host planner's piece boundaries, fp32 tile arithmetic
and launch geometry; hot atomic accumulation order remains nondeterministic.
"""
import os

import torch
import triton
import triton.language as tl
from generative_recommenders.ops.utils import arm_print_all_ranks
from generative_recommenders.ops.zero_fold import enabled as zero_fold_enabled

from . import ec_segment_reduce as incumbent


def hot_capacity(tokens, limit):
    # For every hot segment n>L, ceil(n/L) <= 2*n/L. Sum over hot
    # segments, then floor: P <= floor(2*T/L). No device scalar is read.
    return 0 if limit <= 0 else (2 * tokens) // limit


@triton.jit(do_not_specialize=["U"])
def _piece_counts(OFF, COUNTS, U, LIMIT: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.load(OFF + row, row < U, other=0)
    hi = tl.load(OFF + row + 1, row < U, other=0)
    length = hi - lo
    pieces = tl.where(length > LIMIT, (length + LIMIT - 1) // LIMIT, 0)
    tl.store(COUNTS + row, pieces, row < U)


@triton.jit
def _piece_location(OFF, PREFIX, piece, U, LIMIT: tl.constexpr):
    left = 0
    right = U
    while left < right:
        middle = (left + right) // 2
        endpoint = tl.load(PREFIX + middle)
        advance = endpoint <= piece
        left = tl.where(advance, middle + 1, left)
        right = tl.where(advance, right, middle)
    row = left
    before = tl.load(PREFIX + row - 1, row > 0, other=0)
    lo = tl.load(OFF + row) + (piece - before) * LIMIT
    hi = tl.minimum(lo + LIMIT, tl.load(OFF + row + 1))
    return row, lo, hi


@triton.jit(do_not_specialize=["U"])
def _reduce(grad_ptr, order_ptr, offsets_ptr, prefix_ptr, out_ptr,
            T, W, U, stride_g, LIMIT: tl.constexpr,
            BLOCK_S: tl.constexpr, BLOCK_W: tl.constexpr, HOT: tl.constexpr,
            ACCUM: tl.constexpr = False):
    p = tl.program_id(0)
    cb = tl.program_id(1)
    if HOT:
        valid = p < tl.load(prefix_ptr + U - 1)
    else:
        start0 = tl.load(offsets_ptr + p)
        end0 = tl.load(offsets_ptr + p + 1)
        valid = (end0 - start0 <= LIMIT) | (LIMIT <= 0)
    if valid:
        if HOT:
            row, start, end = _piece_location(offsets_ptr, prefix_ptr, p, U, LIMIT)
        else:
            row, start, end = p, start0, end0
        # Preserve the incumbent reduction operations, tile sizes and piece
        # boundaries. Cold rows store; hot pieces alone use atomic additions.
        cols = cb * BLOCK_W + tl.arange(0, BLOCK_W)
        cmask = cols < W
        acc = tl.zeros([BLOCK_W], dtype=tl.float32)
        for s0 in range(start, end, BLOCK_S):
            srows = s0 + tl.arange(0, BLOCK_S)
            smask = srows < end
            src = tl.load(order_ptr + srows, mask=smask, other=0)
            tile = tl.load(grad_ptr + src[:, None] * stride_g + cols[None, :],
                           mask=smask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(tile, axis=0)
        if HOT:
            tl.atomic_add(out_ptr + row.to(tl.int64) * W + cols, acc, mask=cmask)
        else:
            if ACCUM:
                # Accumulate into a caller-owned fp32 buffer (grouped EC
                # backward): out[row] = out[row] + this key's segment sum.
                # Hot rows are untouched by this launch and get the hot
                # launch's atomics on top of their existing value.
                acc = tl.load(out_ptr + row.to(tl.int64) * W + cols, mask=cmask, other=0.0) + acc
            tl.store(out_ptr + row.to(tl.int64) * W + cols, acc, mask=cmask)


def make_prefix(offsets, num_rows, limit):
    if num_rows == 0 or limit <= 0:
        return None
    counts = torch.empty(num_rows, dtype=torch.int64, device=offsets.device)
    _piece_counts[(triton.cdiv(num_rows, 256),)](offsets, counts, num_rows, limit, 256)
    return torch.cumsum(counts, dim=0)


# ---------------------------------------------------------------------------
# DLRM_HSTU_FP8_EC_REDUCE_LPT (code default OFF; gin item (28)(d) binds it on;
# absent or =0 = the two launches above).
#
# Arbor R16 iter 20. The cold/hot two-launch reducer above is not bandwidth
# bound at production skew: a 1024-row segment is ~32 dependent tile steps
# (~100 us serial on one program), so cold rows longer than ~128 tokens that
# land late in the row-ordered grid, and the hot launch that only starts after
# the whole cold grid drained, leave the device idling in tails. This path
#   1. issues ONE launch: hot pieces first (program ids < capacity), then all
#      cold rows, longest class first -- rows with 128 < len <= split_above
#      before the rest (a 2-class stable counting sort built on the GPU from
#      the same prefix-sum launch as the hot-piece counts; no host sync), and
#   2. folds each BLOCK_S-row tile on ONE wave as fold(even rows) + fold(odd rows),
#      which is exactly the value the 2-wave tile reduction produces (each wave
#      owns alternating rows and the cross-wave combine is a single add), minus
#      its LDS round-trips and barriers. A one-time self-check compares it with
#      the incumbent kernel bitwise per width, grad dtype and row-stride class
#      and falls back to the incumbent 2-wave fold wherever that identity does
#      not hold.
# Per-row tile boundaries, the fp32 accumulator, the piece boundaries and the
# zero-initialised output are unchanged: cold rows are bitwise identical and hot
# rows differ only by the pre-existing atomic-order nondeterminism.
_LPT_ENABLED = None
_LPT_LONG = 128
_EO_OK = {}


def _lpt_enabled():
    global _LPT_ENABLED
    if _LPT_ENABLED is None:
        _LPT_ENABLED = os.environ.get("DLRM_HSTU_FP8_EC_REDUCE_LPT", "0") != "0"
        if _LPT_ENABLED:
            # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
            arm_print_all_ranks("[r16] ec_reduce_lpt armed")
    return _LPT_ENABLED


@triton.jit(do_not_specialize=["U"])
def _plan_counts(OFF, COUNTS, U, LIMIT: tl.constexpr, LONG: tl.constexpr,
                 BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.load(OFF + row, row < U, other=0)
    hi = tl.load(OFF + row + 1, row < U, other=0)
    length = hi - lo
    pieces = tl.where(length > LIMIT, (length + LIMIT - 1) // LIMIT, 0)
    tl.store(COUNTS + row, pieces, row < U)
    is_long = (length > LONG) & (length <= LIMIT)
    tl.store(COUNTS + U + row, is_long.to(tl.int64), row < U)


# zero-fills (ROCm 10.2 / torch 2.16): _plan_counts plus +0.0 stores into the hot
# rows of a fresh (torch.empty) [U, W] fp32 output, the _ecf_plan_rows pattern. A
# hot row (length > LIMIT) is the only kind whose first write in _reduce_lpt is an
# atomic add; every other row is stored once by the cold section (an empty segment
# stores +0.0). So the output needs no [U, W] zero fill: torch 2.16 issues that
# zero_() as a hipMemsetAsync blit on the compute stream (3-5 per step, 16-21 us).
# Same values as the zero fill: cold rows are stored, hot rows are +0.0 + the same
# atomics (whose order stays the pre-existing nondeterminism).
@triton.jit(do_not_specialize=["U"])
def _plan_counts_z(OFF, COUNTS, OUT, U, W, LIMIT: tl.constexpr, LONG: tl.constexpr,
                   BLOCK: tl.constexpr, ZR: tl.constexpr, ZW: tl.constexpr):
    base = tl.program_id(0) * BLOCK
    row = base + tl.arange(0, BLOCK)
    lo = tl.load(OFF + row, row < U, other=0)
    hi = tl.load(OFF + row + 1, row < U, other=0)
    length = hi - lo
    pieces = tl.where(length > LIMIT, (length + LIMIT - 1) // LIMIT, 0)
    tl.store(COUNTS + row, pieces, row < U)
    is_long = (length > LONG) & (length <= LIMIT)
    tl.store(COUNTS + U + row, is_long.to(tl.int64), row < U)
    zc = tl.arange(0, ZW)
    for z0 in tl.static_range(0, BLOCK, ZR):
        zr = base + z0 + tl.arange(0, ZR)
        zm = zr < U
        zl = tl.load(OFF + zr + 1, zm, other=0) - tl.load(OFF + zr, zm, other=0)
        zz = zm & (zl > LIMIT)
        if tl.max(zz.to(tl.int32), axis=0) > 0:
            for c0 in range(0, W, ZW):
                cc = c0 + zc
                tl.store(OUT + zr[:, None].to(tl.int64) * W + cc[None, :],
                         tl.zeros([ZR, ZW], dtype=tl.float32),
                         mask=zz[:, None] & (cc < W)[None, :])


@triton.jit(do_not_specialize=["U"])
def _plan_perm(OFF, CSUM, PERM, U, LIMIT: tl.constexpr, LONG: tl.constexpr,
               BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = row < U
    lo = tl.load(OFF + row, m, other=0)
    hi = tl.load(OFF + row + 1, m, other=0)
    length = hi - lo
    is_long = (length > LONG) & (length <= LIMIT)
    # CSUM is ONE flat inclusive scan over [pieces | long flags]: subtract the
    # pieces total to get the long-flag scan (a 1-D rocprim scan is ~25x
    # cheaper than torch's innermost-dim scan of the [2, U] view).
    pieces_total = tl.load(CSUM + U - 1)
    c = tl.load(CSUM + U + row, m, other=0) - pieces_total
    n_long = tl.load(CSUM + U + U - 1) - pieces_total
    pos = tl.where(is_long, c - 1, n_long + row - c)
    tl.store(PERM + pos, row.to(tl.int32), m)


@triton.jit
def _fold_two_wave(grad_ptr, order_ptr, start, end, cols, cmask, stride_g,
                   BLOCK_S: tl.constexpr, BLOCK_W: tl.constexpr):
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    for s0 in range(start, end, BLOCK_S):
        srows = s0 + tl.arange(0, BLOCK_S)
        smask = srows < end
        src = tl.load(order_ptr + srows, mask=smask, other=0)
        tile = tl.load(grad_ptr + src[:, None] * stride_g + cols[None, :],
                       mask=smask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(tile, axis=0)
    return acc


@triton.jit
def _fold_even_odd(grad_ptr, order_ptr, start, end, cols, cmask, stride_g,
                   BLOCK_S: tl.constexpr, BLOCK_W: tl.constexpr):
    HALF: tl.constexpr = BLOCK_S // 2
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    even = 2 * tl.arange(0, HALF)
    for s0 in range(start, end, BLOCK_S):
        re = s0 + even
        ro = re + 1
        me = re < end
        mo = ro < end
        se = tl.load(order_ptr + re, mask=me, other=0)
        so = tl.load(order_ptr + ro, mask=mo, other=0)
        te = tl.load(grad_ptr + se[:, None] * stride_g + cols[None, :],
                     mask=me[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        to = tl.load(grad_ptr + so[:, None] * stride_g + cols[None, :],
                     mask=mo[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(te, axis=0) + tl.sum(to, axis=0)
    return acc


@triton.jit(do_not_specialize=["U", "CAP"])
def _reduce_lpt(grad_ptr, order_ptr, offsets_ptr, prefix_ptr, perm_ptr, out_ptr,
                W, U, CAP, stride_g, LIMIT: tl.constexpr,
                BLOCK_S: tl.constexpr, BLOCK_W: tl.constexpr,
                EVEN_ODD: tl.constexpr, ACCUM: tl.constexpr = False):
    q = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BLOCK_W + tl.arange(0, BLOCK_W)
    cmask = cols < W
    if q < CAP:
        if q < tl.load(prefix_ptr + U - 1):
            row, start, end = _piece_location(offsets_ptr, prefix_ptr, q, U, LIMIT)
            if EVEN_ODD:
                acc = _fold_even_odd(grad_ptr, order_ptr, start, end, cols, cmask,
                                     stride_g, BLOCK_S, BLOCK_W)
            else:
                acc = _fold_two_wave(grad_ptr, order_ptr, start, end, cols, cmask,
                                     stride_g, BLOCK_S, BLOCK_W)
            tl.atomic_add(out_ptr + row.to(tl.int64) * W + cols, acc, mask=cmask)
    else:
        p = tl.load(perm_ptr + (q - CAP)).to(tl.int64)
        start = tl.load(offsets_ptr + p)
        end = tl.load(offsets_ptr + p + 1)
        if end - start <= LIMIT:
            if EVEN_ODD:
                acc = _fold_even_odd(grad_ptr, order_ptr, start, end, cols, cmask,
                                     stride_g, BLOCK_S, BLOCK_W)
            else:
                acc = _fold_two_wave(grad_ptr, order_ptr, start, end, cols, cmask,
                                     stride_g, BLOCK_S, BLOCK_W)
            if ACCUM:
                # Same operation as the incumbent _reduce ACCUM branch (grouped
                # EC backward, ship8 item 21): out[p] = out[p] + segment sum.
                # Hot rows (atomics above) and cold rows are disjoint, so the
                # single launch has no cold/hot read-modify-write race.
                acc = tl.load(out_ptr + p * W + cols, mask=cmask, other=0.0) + acc
            tl.store(out_ptr + p * W + cols, acc, mask=cmask)


def _launch_lpt(grad_out, order, offsets, prefix, perm, out, width, num_rows,
                capacity, limit, block_w, columns, even_odd, accum=False):
    _reduce_lpt[(capacity + num_rows, columns)](
        grad_out, order, offsets, prefix, perm, out, width, num_rows, capacity,
        grad_out.stride(0), limit, incumbent._BLOCK_S, block_w, even_odd, accum,
        num_warps=1 if even_odd else incumbent._NUM_WARPS, num_stages=1)


def _even_odd_ok(width, block_w, columns, device, dtype, stride0):
    """One-time (per width, grad dtype and row-stride class) bitwise check of
    the one-wave even/odd fold against the incumbent kernel on cold rows; it
    only holds for the incumbent's [1, 8] x [1, 64] x [2, 1] tile layout, so
    verify instead of assuming. The probe grad has the caller's dtype, width
    and row stride, which fix the tile layout and the fp32 reduction tree
    (Triton specializes the row stride on == 1 and % 16 == 0). It does not
    reproduce every other specialization of the production launch (AMD
    buffer ops for tensors under 2 GiB, pointer alignment, the LIMIT and
    ACCUM constexprs); those change load lowering, not the reduction order,
    and the production-size ACCUM path is covered by offline checks. A row
    stride below the width (broadcast or overlapping grads) falls back."""
    key = (width, block_w, incumbent._BLOCK_S, incumbent._NUM_WARPS, str(device),
           dtype, stride0 == 1, stride0 % 16 == 0)
    if key not in _EO_OK:
        ok = (incumbent._BLOCK_S % 2 == 0 and incumbent._NUM_WARPS == 2
              and stride0 >= width)
        if ok:
            gen = torch.Generator(device="cpu").manual_seed(20)
            rows = 64
            lengths = torch.randint(0, 6 * incumbent._BLOCK_S + 5, (rows,), generator=gen)
            lengths[0], lengths[1] = 0, 1
            ids = torch.repeat_interleave(torch.arange(rows), lengths)
            ids = ids[torch.randperm(ids.numel(), generator=gen)].to(device)
            tokens = ids.numel()
            # Wide random exponents: partial sums of same-scale bf16 values are
            # often EXACT in fp32 and would hide a different association.
            scale = torch.exp2(torch.randint(-12, 13, (tokens, width), generator=gen).float())
            # The caller's dtype and row stride: the first `width` columns of a
            # [tokens, max(stride0, width)] buffer.
            grad = torch.zeros((tokens, max(stride0, width)), dtype=dtype,
                               device=device)[:, :width]
            grad.copy_((torch.randn(tokens, width, generator=gen) * scale).to(
                device=device, dtype=dtype))
            order = torch.argsort(ids)
            offsets = torch.zeros(rows + 1, dtype=torch.int64, device=device)
            torch.cumsum(lengths.to(device), 0, out=offsets[1:])
            limit = 1 << 20  # everything cold
            ref = torch.zeros((rows, width), dtype=torch.float32, device=device)
            _reduce[(rows, columns)](grad, order, offsets, offsets, ref, tokens, width,
                rows, grad.stride(0), limit, incumbent._BLOCK_S, block_w, False,
                num_warps=incumbent._NUM_WARPS, num_stages=1)
            got = torch.zeros_like(ref)
            perm = torch.arange(rows, dtype=torch.int32, device=device)
            _launch_lpt(grad, order, offsets, offsets, perm, got, width, rows, 0,
                        limit, block_w, columns, True)
            ok = bool(torch.equal(got, ref))
        _EO_OK[key] = ok
        # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks): each
        # rank runs its own self-check, so each reports its own verdict.
        what = (f"[r16] ec_reduce_lpt width={width} "
                f"dtype={str(dtype).replace('torch.', '')}: one-wave even/odd fold")
        if ok:
            arm_print_all_ranks(f"{what} bitwise OK")
        else:
            arm_print_all_ranks(f"{what} not bitwise for this key -> incumbent "
                                "2-wave fold (cold rows stay bitwise)")
    return _EO_OK[key]


def _segment_reduce_rows_lpt(grad_out, order, offsets, num_rows, out_dtype,
                             accumulate_into=None):
    tokens, width = grad_out.shape
    device = grad_out.device
    accum = accumulate_into is not None
    limit = incumbent._SPLIT_ABOVE
    block_w = min(triton.next_power_of_2(width), incumbent._BLOCK_W_CAP)
    columns = triton.cdiv(width, block_w)
    # zero-fills: a fresh output is written whole by _plan_counts_z (hot rows) and
    # _reduce_lpt (every other row), so no zero fill. With no launch below (no row
    # or no column) the output has no elements to write either.
    fold = not accum and zero_fold_enabled()
    if accum:
        out = accumulate_into  # validated (fp32, contiguous, [U, W]) by the caller
    elif fold:
        out = torch.empty((num_rows, width), dtype=torch.float32, device=device)
    else:
        out = torch.zeros((num_rows, width), dtype=torch.float32, device=device)
    if num_rows and columns:
        counts = torch.empty((2, num_rows), dtype=torch.int64, device=device)
        if fold:
            _plan_counts_z[(triton.cdiv(num_rows, 256),)](
                offsets, counts, out, num_rows, width, limit, _LPT_LONG, 256, 16,
                min(block_w, 128), num_warps=4)
        else:
            _plan_counts[(triton.cdiv(num_rows, 256),)](offsets, counts, num_rows, limit,
                                                        _LPT_LONG, 256)
        csum = torch.cumsum(counts.view(-1), dim=0)
        perm = torch.empty(num_rows, dtype=torch.int32, device=device)
        _plan_perm[(triton.cdiv(num_rows, 256),)](offsets, csum, perm, num_rows, limit,
                                                  _LPT_LONG, 256)
        even_odd = _even_odd_ok(width, block_w, columns, device, grad_out.dtype,
                                grad_out.stride(0))
        _launch_lpt(grad_out, order, offsets, csum, perm, out, width, num_rows,
                    hot_capacity(tokens, limit), limit, block_w, columns, even_odd,
                    accum)
    return out if out_dtype == torch.float32 else out.to(out_dtype)


def segment_reduce_rows(grad_out, reverse_indices, num_rows,
                        out_dtype=torch.float32, order=None, offsets=None,
                        accumulate_into=None):
    # Rows are addressed as grad_out + row * stride(0) + col; any tensor whose
    # columns are unit-stride (or that has <= 1 column / no elements, where the
    # column stride is never used) is read correctly in place.
    assert grad_out.dim() == 2 and (
        grad_out.stride(1) == 1 or grad_out.size(1) <= 1 or grad_out.numel() == 0
    ), grad_out.stride()
    tokens, width = grad_out.shape
    if order is None or offsets is None:
        order, offsets = incumbent.build_segments(reverse_indices, num_rows)
    accum = accumulate_into is not None
    if accum:
        assert out_dtype == torch.float32, "accumulate_into requires fp32"
        out = accumulate_into
        assert (out.dtype == torch.float32 and out.is_contiguous()
                and tuple(out.shape) == (num_rows, width)), (out.dtype, out.shape)
    # EC_REDUCE_LPT (gin item (28)(d)) takes the whole call; with
    # EC_REDUCE_LPT=0 the trunk cold + hot launches below run.
    if incumbent._SPLIT_ABOVE > 0 and _lpt_enabled():
        return _segment_reduce_rows_lpt(grad_out, order, offsets, num_rows, out_dtype,
                                        out if accum else None)
    if not accum:
        out = torch.zeros((num_rows, width), dtype=torch.float32, device=grad_out.device)
    limit = incumbent._SPLIT_ABOVE
    prefix = make_prefix(offsets, num_rows, limit)
    block_w = min(triton.next_power_of_2(width), incumbent._BLOCK_W_CAP)
    columns = triton.cdiv(width, block_w)
    if num_rows and columns:
        _reduce[(num_rows, columns)](grad_out, order, offsets,
            prefix if prefix is not None else offsets, out, tokens, width,
            num_rows, grad_out.stride(0), limit, incumbent._BLOCK_S, block_w, False, accum,
            num_warps=incumbent._NUM_WARPS, num_stages=1)
        capacity = hot_capacity(tokens, limit)
        if capacity:
            _reduce[(capacity, columns)](grad_out, order, offsets, prefix, out,
                tokens, width, num_rows, grad_out.stride(0), limit, incumbent._BLOCK_S, block_w, True,
                num_warps=incumbent._NUM_WARPS, num_stages=1)
    return out if out_dtype == torch.float32 else out.to(out_dtype)


# ---------------------------------------------------------------------------
# DLRM_HSTU_FP8_EC_REDUCE_FUSED (rccl iter 72, track ec; code default OFF; the
# gin binding apply_env_bootstrap.DLRM_HSTU_FP8_EC_REDUCE_FUSED turns it on,
# "0" is the kill switch; read at first use; acts only where EC_REDUCE_LPT
# takes the call).
#
# The grouped EC backward (ec_bf16_output._ECExpandGroup) runs the LPT reducer
# once per hoisted member: [U, W] zero fill, then per member _plan_counts ->
# cumsum -> _plan_perm -> _reduce_lpt over EVERY row (member 1 in ACCUM mode,
# i.e. a read-modify-write of all U rows). Per row r (L = SPLIT_ABOVE):
#   member 0: la <= L: out[r] = fold0(r) (+0.0 if empty) | la > L: 0 + atomics
#   member 1: lb <= L: out[r] = out[r] + fold1(r)        | lb > L: out[r] + atomics
# segment_reduce_grouped computes exactly those values with one plan chain and
# one main launch:
#   _ecf_plan_rows   per-row piece counts of both members, the LPT long flag of
#                    the fused cold work, and +0.0 stores into exactly the rows
#                    whose first write in _ecf_main is an atomic add (member-0
#                    hot rows; member-1 hot rows with an empty member-0
#                    segment) -- no [U, W] zero fill;
#   cumsum           one flat scan over [pieces0 | pieces1 | long];
#   _ecf_plan_tables the LPT cold-row order (r16's 2-class rule, applied to the
#                    fused work) and explicit (row, lo, hi) piece tables -- no
#                    binary search per hot program;
#   _ecf_main        member-0 pieces (atomic), member-1 pieces of rows with an
#                    empty member-0 segment (atomic), then every row once:
#                    both cold -> store(out0 + fold1) with the incumbent's
#                    member-1 add (operand order pinned, below); member 1 hot
#                    -> store(fold0) where member 0 is non-empty;
#   _ecf_post        (two members) the incumbent's member-1 RMW on member-0 hot
#                    rows (after every member-0 atomic) and member-1 pieces of
#                    rows with a non-empty member-0 segment.
# Folds are r16's _fold_even_odd / _fold_two_wave unchanged, with LPT's own
# _even_odd_ok decision, BLOCK_S, tile and piece boundaries, so every cold row
# (both members cold) is bit-identical, and hot rows get the same piece sums
# and atomic adds in the same launch order (member 0's atomics before member
# 1's), i.e. stay within the pre-existing atomic-order nondeterminism.
# Member-1 add: LPT compiles `load(out) + acc` with the loaded member-0 value
# as src0 at the production specialization (gfx950 returns src0's payload when
# both operands are NaN). The fused combine is an inline-asm v_add_f32 whose
# operand order comes from _FUSED_VALIDATED only for keys validated offline at
# BLOCK_S=32 / SPLIT_ABOVE=1024 (no runtime probe) and otherwise, including
# the shipped BLOCK_S=16 / SPLIT_ABOVE=8192 geometry, from a one-time
# self-test per (BLOCK_S, SPLIT_ABOVE, width, dtypes, row-stride classes,
# fold) that compares raw bits against the per-member LPT calls on overlapping
# rows with NaN payloads, -0.0, inf and denormals (plus member-0 and member-1
# hot rows); no match -> the per-member calls (fail closed). The compiled
# order differs between specializations (lab: src0 = loaded at W=512 bf16,
# src0 = acc at W=128/256/600 bf16). Production member row sets are disjoint
# (different tables), so one operand is always +0.0 there: order cannot matter.
_FUSED_ENABLED = None
_FUSED_OK = {}
_FUSED_ASM = os.environ.get("TRITON_INTERPRET", "0") != "1"


def _ecf_print(msg):
    # EC_REDUCE_FUSED's arming / self-test lines: every rank, tagged " [rank N]" (the r16
    # arm_print_all_ranks convention; each rank runs its own key check). Self-contained on
    # purpose: it uses no other helper of this module, so a merge that changes those helpers
    # cannot break this part.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = str(torch.distributed.get_rank())
    else:
        rank = os.environ.get("RANK", "0") or "0"
    print(f"{msg} [rank {rank}]", flush=True)


def fused_enabled():
    global _FUSED_ENABLED
    if _FUSED_ENABLED is None:
        _FUSED_ENABLED = os.environ.get("DLRM_HSTU_FP8_EC_REDUCE_FUSED", "").strip().lower() in (
            "1", "true", "on", "yes")
    return _FUSED_ENABLED


@triton.jit
def _ecf_fold(g, o, lo, hi, cols, cmask, sg, BLOCK_S: tl.constexpr, BLOCK_W: tl.constexpr,
              EVEN_ODD: tl.constexpr):
    if EVEN_ODD:
        acc = _fold_even_odd(g, o, lo, hi, cols, cmask, sg, BLOCK_S, BLOCK_W)
    else:
        acc = _fold_two_wave(g, o, lo, hi, cols, cmask, sg, BLOCK_S, BLOCK_W)
    return acc


@triton.jit
def _ecf_combine(x0, acc1, SRC0_OLD: tl.constexpr, ASM: tl.constexpr):
    # The incumbent member-1 add `out + acc1` with a pinned operand order.
    if ASM:
        if SRC0_OLD:
            r = tl.inline_asm_elementwise("v_add_f32 $0, $1, $2", "=v,v,v", [x0, acc1],
                                          dtype=tl.float32, is_pure=True, pack=1)
        else:
            r = tl.inline_asm_elementwise("v_add_f32 $0, $1, $2", "=v,v,v", [acc1, x0],
                                          dtype=tl.float32, is_pure=True, pack=1)
    else:
        r = x0 + acc1
    return r


@triton.jit(do_not_specialize=["U"])
def _ecf_plan_rows(fA, fB, CNT, out_ptr, U, W, LIMIT: tl.constexpr, LONG: tl.constexpr,
                   TWO: tl.constexpr, BLOCK: tl.constexpr, ZR: tl.constexpr,
                   ZW: tl.constexpr):
    base = tl.program_id(0) * BLOCK
    rows = base + tl.arange(0, BLOCK)
    m = rows < U
    a0 = tl.load(fA + rows, m, other=0)
    la = tl.load(fA + rows + 1, m, other=0) - a0
    a_hot = la > LIMIT
    tl.store(CNT + rows, tl.where(a_hot, (la + LIMIT - 1) // LIMIT, 0), m)
    if TWO:
        b0 = tl.load(fB + rows, m, other=0)
        lb = tl.load(fB + rows + 1, m, other=0) - b0
        b_hot = lb > LIMIT
        tl.store(CNT + U + rows, tl.where(b_hot, (lb + LIMIT - 1) // LIMIT, 0), m)
        work = tl.where(a_hot, 0, la + tl.where(b_hot, 0, lb))
        tl.store(CNT + 2 * U + rows, (work > LONG).to(tl.int64), m)
    else:
        work = tl.where(a_hot, 0, la)
        tl.store(CNT + U + rows, (work > LONG).to(tl.int64), m)
    # +0.0 into the rows whose first write in _ecf_main is an atomic add.
    zc = tl.arange(0, ZW)
    for z0 in tl.static_range(0, BLOCK, ZR):
        zr = base + z0 + tl.arange(0, ZR)
        zm = zr < U
        zla = tl.load(fA + zr + 1, zm, other=0) - tl.load(fA + zr, zm, other=0)
        if TWO:
            zlb = tl.load(fB + zr + 1, zm, other=0) - tl.load(fB + zr, zm, other=0)
            zz = zm & ((zla > LIMIT) | ((zla == 0) & (zlb > LIMIT)))
        else:
            zz = zm & (zla > LIMIT)
        if tl.max(zz.to(tl.int32), axis=0) > 0:
            for c0 in range(0, W, ZW):
                cc = c0 + zc
                tl.store(out_ptr + zr[:, None].to(tl.int64) * W + cc[None, :],
                         tl.zeros([ZR, ZW], dtype=tl.float32),
                         mask=zz[:, None] & (cc < W)[None, :])


@triton.jit(do_not_specialize=["U"])
def _ecf_plan_tables(fA, fB, CSUM, PERM, PA, PB, U, LIMIT: tl.constexpr, LONG: tl.constexpr,
                     TWO: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = rows < U
    a0 = tl.load(fA + rows, m, other=0)
    a1 = tl.load(fA + rows + 1, m, other=0)
    la = a1 - a0
    a_hot = la > LIMIT
    pa = tl.where(a_hot & m, (la + LIMIT - 1) // LIMIT, 0)
    endA = tl.load(CSUM + rows, m, other=0)
    if TWO:
        b0 = tl.load(fB + rows, m, other=0)
        b1 = tl.load(fB + rows + 1, m, other=0)
        lb = b1 - b0
        b_hot = lb > LIMIT
        pb = tl.where(b_hot & m, (lb + LIMIT - 1) // LIMIT, 0)
        work = tl.where(a_hot, 0, la + tl.where(b_hot, 0, lb))
        endB = tl.load(CSUM + U + rows, m, other=0) - tl.load(CSUM + U - 1)
        lbase = tl.load(CSUM + 2 * U - 1)
        n_long = tl.load(CSUM + 3 * U - 1) - lbase
        c = tl.load(CSUM + 2 * U + rows, m, other=0) - lbase
    else:
        work = tl.where(a_hot, 0, la)
        lbase = tl.load(CSUM + U - 1)
        n_long = tl.load(CSUM + 2 * U - 1) - lbase
        c = tl.load(CSUM + U + rows, m, other=0) - lbase
    # r16's 2-class stable counting sort, on the fused cold work.
    pos = tl.where(work > LONG, c - 1, n_long + rows - c)
    tl.store(PERM + pos, rows.to(tl.int32), m)
    # Piece tables, entry = (row, lo, hi, member-0 segment non-empty).
    startA = endA - pa
    for k in range(0, tl.max(pa, axis=0)):
        km = m & (k < pa)
        lo = a0 + k * LIMIT
        e = (startA + k) * 4
        tl.store(PA + e, rows.to(tl.int64), km)
        tl.store(PA + e + 1, lo, km)
        tl.store(PA + e + 2, tl.minimum(lo + LIMIT, a1), km)
    if TWO:
        startB = endB - pb
        for k in range(0, tl.max(pb, axis=0)):
            km = m & (k < pb)
            lo = b0 + k * LIMIT
            e = (startB + k) * 4
            tl.store(PB + e, rows.to(tl.int64), km)
            tl.store(PB + e + 1, lo, km)
            tl.store(PB + e + 2, tl.minimum(lo + LIMIT, b1), km)
            tl.store(PB + e + 3, (la > 0).to(tl.int64), km)


@triton.jit(do_not_specialize=["U", "CAPA", "CAPB"])
def _ecf_main(gA, oA, fA, gB, oB, fB, CSUM, PERM, PA, PB, out_ptr, W, U, CAPA, CAPB, sA, sB,
              LIMIT: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_W: tl.constexpr,
              EVEN_ODD: tl.constexpr, TWO: tl.constexpr, SRC0_OLD: tl.constexpr,
              ASM: tl.constexpr):
    q = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BLOCK_W + tl.arange(0, BLOCK_W)
    cmask = cols < W
    if q < CAPA:
        if q < tl.load(CSUM + U - 1):
            e = q.to(tl.int64) * 4
            row = tl.load(PA + e)
            acc = _ecf_fold(gA, oA, tl.load(PA + e + 1), tl.load(PA + e + 2), cols, cmask, sA,
                            BLOCK_S, BLOCK_W, EVEN_ODD)
            tl.atomic_add(out_ptr + row * W + cols, acc, mask=cmask)
    elif q < CAPA + CAPB:
        j = q - CAPA
        if j < tl.load(CSUM + 2 * U - 1) - tl.load(CSUM + U - 1):
            e = j.to(tl.int64) * 4
            if tl.load(PB + e + 3) == 0:
                row = tl.load(PB + e)
                acc = _ecf_fold(gB, oB, tl.load(PB + e + 1), tl.load(PB + e + 2), cols, cmask,
                                sB, BLOCK_S, BLOCK_W, EVEN_ODD)
                tl.atomic_add(out_ptr + row * W + cols, acc, mask=cmask)
    else:
        r = tl.load(PERM + (q - CAPA - CAPB)).to(tl.int64)
        a0 = tl.load(fA + r)
        a1 = tl.load(fA + r + 1)
        if TWO:
            b0 = tl.load(fB + r)
            b1 = tl.load(fB + r + 1)
        if a1 - a0 <= LIMIT:
            acc0 = _ecf_fold(gA, oA, a0, a1, cols, cmask, sA, BLOCK_S, BLOCK_W, EVEN_ODD)
            optr = out_ptr + r * W + cols
            if TWO:
                if b1 - b0 <= LIMIT:
                    acc1 = _ecf_fold(gB, oB, b0, b1, cols, cmask, sB, BLOCK_S, BLOCK_W, EVEN_ODD)
                    tl.store(optr, _ecf_combine(acc0, acc1, SRC0_OLD, ASM), mask=cmask)
                elif a1 > a0:
                    tl.store(optr, acc0, mask=cmask)
            else:
                tl.store(optr, acc0, mask=cmask)


@triton.jit(do_not_specialize=["U", "CAPA"])
def _ecf_post(gB, oB, fA, fB, CSUM, PA, PB, out_ptr, W, U, CAPA, sB,
              LIMIT: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_W: tl.constexpr,
              EVEN_ODD: tl.constexpr, SRC0_OLD: tl.constexpr, ASM: tl.constexpr):
    q = tl.program_id(0)
    cb = tl.program_id(1)
    cols = cb * BLOCK_W + tl.arange(0, BLOCK_W)
    cmask = cols < W
    if q < CAPA:
        if q < tl.load(CSUM + U - 1):
            e = q.to(tl.int64) * 4
            row = tl.load(PA + e)
            if tl.load(PA + e + 1) == tl.load(fA + row):  # once per member-0 hot row
                b0 = tl.load(fB + row)
                b1 = tl.load(fB + row + 1)
                if b1 - b0 <= LIMIT:
                    acc1 = _ecf_fold(gB, oB, b0, b1, cols, cmask, sB, BLOCK_S, BLOCK_W, EVEN_ODD)
                    optr = out_ptr + row * W + cols
                    old = tl.load(optr, mask=cmask, other=0.0)
                    tl.store(optr, _ecf_combine(old, acc1, SRC0_OLD, ASM), mask=cmask)
    else:
        j = q - CAPA
        if j < tl.load(CSUM + 2 * U - 1) - tl.load(CSUM + U - 1):
            e = j.to(tl.int64) * 4
            if tl.load(PB + e + 3) != 0:
                row = tl.load(PB + e)
                acc = _ecf_fold(gB, oB, tl.load(PB + e + 1), tl.load(PB + e + 2), cols, cmask,
                                sB, BLOCK_S, BLOCK_W, EVEN_ODD)
                tl.atomic_add(out_ptr + row * W + cols, acc, mask=cmask)


def _ecf_launch(members, num_rows, even_odd, src0_old):
    """[(grad, order, offsets)] x 1..2 -> fp32 [U, W]; geometry exactly LPT's."""
    gA, oA, fA = members[0]
    two = len(members) == 2
    gB, oB, fB = members[1] if two else members[0]
    width = gA.shape[1]
    dev = gA.device
    out = torch.empty((num_rows, width), dtype=torch.float32, device=dev)
    limit = incumbent._SPLIT_ABOVE
    block_w = min(triton.next_power_of_2(width), incumbent._BLOCK_W_CAP)
    columns = triton.cdiv(width, block_w)
    nw = 1 if even_odd else incumbent._NUM_WARPS
    cnt = torch.empty((3 if two else 2) * num_rows, dtype=torch.int64, device=dev)
    _ecf_plan_rows[(triton.cdiv(num_rows, 256),)](
        fA, fB, cnt, out, num_rows, width, limit, _LPT_LONG, two, 256, 16,
        min(block_w, 128), num_warps=4)
    csum = torch.cumsum(cnt, dim=0)
    cap_a = hot_capacity(gA.shape[0], limit)
    cap_b = hot_capacity(gB.shape[0], limit) if two else 0
    perm = torch.empty(num_rows, dtype=torch.int32, device=dev)
    pa = torch.empty((max(cap_a, 1), 4), dtype=torch.int64, device=dev)
    pb = torch.empty((max(cap_b, 1), 4), dtype=torch.int64, device=dev)
    _ecf_plan_tables[(triton.cdiv(num_rows, 256),)](
        fA, fB, csum, perm, pa, pb, num_rows, limit, _LPT_LONG, two, 256, num_warps=4)
    _ecf_main[(cap_a + cap_b + num_rows, columns)](
        gA, oA, fA, gB, oB, fB, csum, perm, pa, pb, out, width, num_rows, cap_a, cap_b,
        gA.stride(0), gB.stride(0), limit, incumbent._BLOCK_S, block_w, even_odd, two,
        src0_old, _FUSED_ASM, num_warps=nw, num_stages=1)
    if two and cap_a + cap_b:
        _ecf_post[(cap_a + cap_b, columns)](
            gB, oB, fA, fB, csum, pa, pb, out, width, num_rows, cap_a, gB.stride(0), limit,
            incumbent._BLOCK_S, block_w, even_odd, src0_old, _FUSED_ASM, num_warps=nw,
            num_stages=1)
    return out


def _fused_selftest_members(width, dtypes, strides, device):
    """Overlapping two-member input (both members hit most rows) with NaN payloads, -0.0,
    +-inf and denormals on cold rows, plus member-0 and member-1 (empty member 0) hot rows
    of two pieces each, at the caller's widths, grad dtypes and row-stride classes."""
    limit = incumbent._SPLIT_ABOVE
    gen = torch.Generator(device="cpu").manual_seed(72)
    rows = 64
    la = torch.randint(0, 70, (rows,), generator=gen)
    lb = torch.randint(0, 70, (rows,), generator=gen)
    la[:4] = torch.tensor([0, 1, 32, 33])
    lb[:4] = torch.tensor([1, 0, 33, 64])
    la[4], lb[4] = limit + 1, 0          # member-0 hot row, two pieces
    la[5], lb[5] = 0, 2 * limit          # member-1 hot row with an empty member 0
    members = []
    for k, ln in enumerate((la, lb)):
        ids = torch.repeat_interleave(torch.arange(rows), ln)
        ids = ids[torch.randperm(ids.numel(), generator=gen)]
        tokens = ids.numel()
        scale = torch.exp2(torch.randint(-12, 13, (tokens, width), generator=gen).float())
        vals = (torch.randn(tokens, width, generator=gen) * scale).to(dtypes[k])
        if dtypes[k] in (torch.bfloat16, torch.float16):
            bits = vals.view(torch.int16)
            sp = torch.tensor([-32768, 0x7F80, -128, 0x0001, -32767, 0x7FC1 + 6 * k,
                               -63 - 7 * k, 0x7F81 + k], dtype=torch.int16)
            cold = ~torch.isin(ids, torch.tensor([4, 5]))
            pos = torch.nonzero(cold).flatten()
            pick = pos[torch.randint(0, pos.numel(), (pos.numel() // 4,), generator=gen)]
            col = torch.randint(0, width, (pick.numel(),), generator=gen)
            bits[pick, col] = sp[torch.randint(0, sp.numel(), (pick.numel(),), generator=gen)]
        grad = torch.zeros((tokens, max(strides[k], width)), dtype=dtypes[k],
                           device=device)[:, :width]
        grad.copy_(vals.to(device))
        order = torch.argsort(ids.to(device), stable=True)
        offsets = torch.zeros(rows + 1, dtype=torch.int64, device=device)
        torch.cumsum(ln.to(device), 0, out=offsets[1:])
        members.append((grad, order, offsets))
    return members, rows


# Keys validated offline (rccl iter 72 lab, probes/iter72/ec: raw bits vs the per-member LPT
# calls on overlapping rows with NaN payloads, -0.0, inf, denormals, and LPT's compiled ACCUM
# add at the production > 2 GiB specialization `v_add_f32 out, out_loaded, acc`): the member-1
# add takes the loaded member-0 value as src0. No runtime probe (and no compile after
# RUN_START) for these; any other key runs the one-time self-test below first.
# BLOCK_S and SPLIT_ABOVE (the LIMIT constexpr) are part of the key: both are constexprs
# of _ecf_main/_ecf_post, the entries were validated at BLOCK_S=32 / SPLIT_ABOVE=1024 only,
# and another geometry must not reuse them without the self-test.
# (BLOCK_S, SPLIT_ABOVE, width, block_w, dtypes, unit row stride, row stride % 16 == 0,
#  even/odd fold) -> src0_old
_FUSED_VALIDATED = {
    (32, 1024, 512, 512, (torch.bfloat16, torch.bfloat16), (False, False), (True, True),
     (True, True)): True,
    (32, 1024, 512, 512, (torch.bfloat16,), (False,), (True,), (True,)): True,
}


def _fused_src0(width, block_w, columns, device, dtypes, strides):
    """Per (BLOCK_S, SPLIT_ABOVE, width, dtypes, row-stride classes): the member-1 add operand order (True = loaded
    member-0 value first) under which the fused launch is raw-bit equal to the per-member LPT
    calls, from _FUSED_VALIDATED or a one-time self-test, or None (-> per-member calls)."""
    eo = [_even_odd_ok(width, block_w, columns, device, dtypes[k], strides[k])
          for k in range(len(dtypes))]
    vkey = (incumbent._BLOCK_S, incumbent._SPLIT_ABOVE, width, block_w, tuple(dtypes),
            tuple(s == 1 for s in strides), tuple(s % 16 == 0 for s in strides), tuple(eo))
    key = (str(device),) + vkey
    what = (f"[rccl72] ec_reduce_fused width={width} "
            f"dtypes={[str(d).replace('torch.', '') for d in dtypes]}")
    if key not in _FUSED_OK and vkey in _FUSED_VALIDATED and all(s >= width for s in strides):
        _FUSED_OK[key] = _FUSED_VALIDATED[vkey]
        _ecf_print(f"{what}: offline-validated key, no runtime probe (member-1 add src0="
                   f"{'out' if _FUSED_OK[key] else 'acc'})")
    if key not in _FUSED_OK:
        choice = None
        if len(set(eo)) == 1 and all(s >= width for s in strides):
            members, rows = _fused_selftest_members(width, list(dtypes) * (2 // len(dtypes)),
                                                    list(strides) * (2 // len(strides)), device)
            members = members[:len(dtypes)]
            ref = None
            for g, o, f in members:
                ref = _segment_reduce_rows_lpt(g, o, f, rows, torch.float32, ref)
            ref_bits = ref.view(torch.int32)
            for src0 in (True, False):
                got = _ecf_launch(members, rows, eo[0], src0)
                if torch.equal(got.view(torch.int32), ref_bits):
                    choice = src0
                    break
        _FUSED_OK[key] = choice
        if choice is None:
            _ecf_print(f"{what}: self-test found no raw-bit match -> per-member LPT calls")
        else:
            _ecf_print(f"{what}: self-test raw-bit OK (member-1 add src0="
                       f"{'out' if choice else 'acc'})")
    return _FUSED_OK[key]


_FUSED_LOGGED = False


def segment_reduce_grouped(members, num_rows):
    """Grouped EC backward reduce of 1-2 hoisted members [(grad, order, offsets)] into ONE fp32
    [num_rows, W] accumulator: the values of the per-member segment_reduce_rows calls (member 0
    allocates, member 1 accumulates) under EC_REDUCE_LPT, with one plan chain and one main
    launch. Returns None where the fused path does not apply (caller runs the per-member calls).
    """
    global _FUSED_LOGGED
    if not (fused_enabled() and incumbent._SPLIT_ABOVE > 0 and _lpt_enabled()):
        return None
    if not 1 <= len(members) <= 2 or num_rows <= 0:
        return None
    prepped = []
    for g, o, f in members:
        if g.dim() != 2 or g.shape[1] != members[0][0].shape[1] or g.shape[1] == 0:
            return None
        if g.stride(1) != 1 or not (incumbent._STRIDED_GRAD or g.is_contiguous()):
            # segment_reduce_rows' rule: in place only with STRIDED_GRAD and unit
            # column stride, otherwise a contiguous copy
            g = g.contiguous()
        prepped.append((g, o, f))
    width = prepped[0][0].shape[1]
    block_w = min(triton.next_power_of_2(width), incumbent._BLOCK_W_CAP)
    columns = triton.cdiv(width, block_w)
    device = prepped[0][0].device
    src0 = _fused_src0(width, block_w, columns, device, tuple(g.dtype for g, _, _ in prepped),
                       tuple(g.stride(0) for g, _, _ in prepped))
    if src0 is None:
        return None
    even_odd = _even_odd_ok(width, block_w, columns, device, prepped[0][0].dtype,
                            prepped[0][0].stride(0))
    out = _ecf_launch(prepped, num_rows, even_odd, src0)
    if not _FUSED_LOGGED:
        _FUSED_LOGGED = True
        _ecf_print("DLRMV4 armed OK: ec reduce fused -- grouped EC backward: one plan chain + one "
                   f"main launch for {len(prepped)} hoisted member(s) (U={num_rows}, W={width}; no "
                   "[U, W] zero fill, no member-1 RMW pass; kill switch "
                   "DLRM_HSTU_FP8_EC_REDUCE_FUSED=0)")
    return out
