"""Arithmetic pos-sorted permutation of the concat-folded position kernel (yambda_5b.gin item (49)(a),
DLRM_HSTU_FP8_POSENC_POS_ARITH; caller: triton_position._posenc_sort_pairs).

Every position kernel (_add_timestamp_position_embeddings_concat_kernel, _posenc_concat_dropout_kernel and its V2
layout) writes, for token j of sequence b (0 <= j < n = Offsets[b+1] - Offsets[b]):
    h      = Lengths[b] - nt_b            (nt_b = NumTargets[b], x2 when interleaved; h = Lengths[b] without targets)
    key(j) = j                                    if j < C        (C = max_contextual_seq_len)
           = min(h - min(j, h) + C, M - 1)        otherwise       (M = max_pos_ind = pos table rows)
The key is a function of (n, h, C, M) only, so the stable sort of these keys needs no radix pass. With M >= C + 2 the
tokens of key k in sequence b form ONE contiguous run (j0, cnt):
    k <  C        : j0 = k,             cnt = [k < n]
    k == C        : j0 = max(C, h),     cnt = max(0, n - j0)
    C < k < M - 1 : j0 = h + C - k,     cnt = [k <= h and j0 < n]
    k == M - 1    : j0 = C,             cnt = max(0, min(n, h + C - M + 2) - C)
and the number of tokens of sequence b with key < k is
    F_b(k) = min(k, C, n) + [k > C] * (max(0, n - max(C, h)) + max(0, min(h, n) - max(C, h + C - k + 1))).
A stable sort puts key k's runs in sequence order, so token (b, j0 + r) of key k lands at
    sum_b' F_b'(k) + sum_{b' < b} cnt(k, b') + r,
which ONE kernel writes (program = KB keys x all sequences). torch.sort runs its stable radix sort for every 1-D CUDA
input above 4096 elements, so above that size these are exactly torch.sort(pos_inds)'s sorted keys and int64
indices. Precondition shared with the position kernels: every token row is covered by their grid (max_seq_len >= the
longest sequence), so pos_inds holds key(j) for every token.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["B", "C", "M"])
def _pos_perm_kernel(Offsets, Lengths, NumTargets, OutKey, OutVal, B, C, M, HAS_TARGETS: tl.constexpr,
                     INTERLEAVE: tl.constexpr, KB: tl.constexpr, BLOCK_B: tl.constexpr):
    offs_b = tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    s0 = tl.load(Offsets + offs_b, mask=mask_b, other=0).to(tl.int64)
    n = (tl.load(Offsets + offs_b + 1, mask=mask_b, other=0).to(tl.int64) - s0).to(tl.int32)
    L = tl.load(Lengths + offs_b, mask=mask_b, other=0).to(tl.int32)
    if HAS_TARGETS:
        nt = tl.load(NumTargets + offs_b, mask=mask_b, other=0).to(tl.int32)
        if INTERLEAVE:
            h = L - nt * 2
        else:
            h = L - nt
    else:
        h = L
    k0 = tl.program_id(0) * KB
    # every key is <= max(C, max_b h_b) (middle keys are <= h) and <= M - 1: tiles above it hold no token
    kmax = tl.minimum(tl.maximum(tl.max(tl.where(mask_b, h, 0), axis=0), C), M - 1)
    if k0 > kmax:
        return
    k = k0 + tl.arange(0, KB)
    kk = k[:, None]
    nn = n[None, :]
    hh = h[None, :]
    # tokens with key < k, per sequence -> start of key k
    c_cnt = tl.maximum(nn - tl.maximum(hh, C), 0)
    mid_lt = tl.maximum(tl.minimum(hh, nn) - tl.maximum(hh + C - kk + 1, C), 0)
    F = tl.minimum(tl.minimum(kk, C), nn) + tl.where(kk > C, c_cnt + mid_lt, 0)
    F = tl.where(mask_b[None, :], F, 0)
    start = tl.sum(F, axis=1)
    # run of key k in sequence b
    is_ctx = kk < C
    is_c = kk == C
    is_top = kk == (M - 1)
    j0_mid = hh + C - kk
    j0_c = tl.maximum(hh, C)
    j0 = tl.where(is_ctx, kk + 0 * hh, tl.where(is_c, j0_c, tl.where(is_top, C + 0 * hh, j0_mid)))
    cnt_ctx = (kk < nn).to(tl.int32)
    cnt_mid = ((kk <= hh) & (j0_mid < nn)).to(tl.int32)
    cnt_top = tl.maximum(tl.minimum(nn, hh + C - M + 2) - C, 0)
    cnt = tl.where(is_ctx, cnt_ctx, tl.where(is_c, c_cnt, tl.where(is_top, cnt_top, cnt_mid)))
    cnt = tl.where(mask_b[None, :] & (kk < M), cnt, 0)
    incl = tl.cumsum(cnt, axis=1)
    dst = (start[:, None] + incl - cnt).to(tl.int64)
    src = s0[None, :] + j0.to(tl.int64)
    kv = kk + 0 * cnt
    rmax = tl.max(tl.max(cnt, axis=1), axis=0)
    for r in range(0, rmax):
        m = cnt > r
        tl.store(OutVal + dst + r, src + r, mask=m)
        tl.store(OutKey + dst + r, kv, mask=m)


_MAX_B = 2048  # all sequences in one program
_KB = 1  # keys per program (MI350X lab sweep: 1 key / 4 warps ~27 us at N 2.4-2.6M, torch.sort ~145 us)
_WARPS = 4  # lab lengths: the real_lens_dense pool (size-distribution class; disclosure: yambda_5b.gin (49)(a))


def pos_perm_supported(N: int, B: int, C: int, M: int) -> bool:
    """Shapes the arithmetic path covers: above torch's small-sort cutoff (its stable radix sort there), every
    contextual key below the clamped key (one run per key and sequence), int16-range keys, B in one program."""
    return 4096 < N < 2**31 - 1 and 0 < B <= _MAX_B and 0 <= C and C + 2 <= M <= 32768


def pos_sorted_pairs(
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    num_targets,
    interleave_targets: bool,
    C: int,
    M: int,
    N: int,
    key_dtype: torch.dtype,
):
    """(sorted_pos_keys [N] key_dtype, sorted_pos_values [N] int64) == torch.sort(pos_inds) for the pos_inds the
    position kernels write (see the module docstring). Caller checks pos_perm_supported. One launch."""
    dev = seq_offsets.device
    B = seq_lengths.shape[0]
    out_key = torch.empty(N, dtype=key_dtype, device=dev)
    out_val = torch.empty(N, dtype=torch.int64, device=dev)
    has_t = num_targets is not None
    _pos_perm_kernel[(triton.cdiv(M, _KB),)](
        seq_offsets,
        seq_lengths,
        num_targets if has_t else seq_lengths,
        out_key,
        out_val,
        B,
        C,
        M,
        HAS_TARGETS=has_t,
        INTERLEAVE=bool(interleave_targets),
        KB=_KB,
        BLOCK_B=max(64, triton.next_power_of_2(B)),
        num_warps=_WARPS,
    )
    return out_key, out_val
