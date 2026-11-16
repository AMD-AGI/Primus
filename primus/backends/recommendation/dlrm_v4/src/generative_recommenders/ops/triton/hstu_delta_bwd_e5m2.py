# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""Targets-only delta attention backward that emits the E5M2 dV|dK pack.

DLRM_HSTU_FP8_BWD_VK_PRODUCER (default off; ported from the Arbor R14
testbed). With the FP8 VK projection backward armed (DLRM_HSTU_FP8_BWD_UVQK=1,
DLRM_HSTU_FP8_BWD_UVQK_WGRAD=fp8t), the shipped path is

    _hstu_attn_delta_bwd     -> dvk  bf16 [M, H*(D_V+D_Q)]   (write 2B/elem)
    _quantize_amax_colsum    -> e5m2 [Mp, ...] + amax + colsum
                                (read 2B/elem, write 1B/elem)

The producer variant below runs the SAME delta backward math and, instead of
storing the BF16 dV/dK, stores the E5M2 cast of exactly the BF16 value the
stock kernel would have stored (round to BF16, back to fp32, x inv_scale,
clamp +-LIMIT, cast). The FP8 payload is therefore byte-identical to
`bwd_fp8.quantize_amax_colsum(dvk)`; amax is exact (max is order-free); the
fp32 bias column sum is the same math in a different fp32 reduction order
(per-sequence partials, then a sum over sequences). The BF16 dvk is never
materialized and the separate cast pass is gone.

This is a copy of `triton_hstu_attention._hstu_attn_delta_bwd` (left
untouched); keep the math in sync with it.

DLRM_HSTU_FP8_BWD_VKP_V2 (default off; Arbor R14 iter-33 M1) launches
`_hstu_attn_delta_bwd_e5m2_v2` instead: the amax comes from per-row maxima of
|silu| and |dqk| (exact, see the kernel comment), dq accumulates element-wise
over the tiles with one cross-row reduction after the loop instead of one
cross-warp (LDS + barrier) reduction per 32-row tile, and programs run
longest-sequence-first. Payload, amax and colsum are bitwise unchanged; dq is
the same math in a different fp32 summation order (DECLARED).
"""

import os
from typing import Tuple

import torch
import triton
import triton.language as tl

from generative_recommenders.common import switch_to_contiguous_if_needed
from generative_recommenders.ops.triton.triton_hstu_attention import (
    fast_dividef,
    fast_expf,
)


# Launch config (fixed, like the stock kernel's BLOCK_N=32 / 4 warps); chosen
# by a config sweep in the Arbor R14 testbed. Re-timed on synthetic length
# regimes (s15p1b pins gen_lengths.py), job 70738: the original rule picks
# 32 / 4 / False again (V1 SAME).
_BLOCK_N: int = 32
_NUM_WARPS: int = 4
_REDUCE_EACH: bool = False
# DLRM_HSTU_FP8_BWD_VKP_V2=1 (default off): read at import, checked by
# `hstu_attention_delta_bwd_e5m2` at call time.
_VKP_V2: bool = os.environ.get("DLRM_HSTU_FP8_BWD_VKP_V2", "0") == "1"
# The v2 configuration (module constants, not knobs): exact row-max amax,
# element-wise dq accumulator (DECLARED fp32-order change of dq only), and
# longest-first launch order (2: one-program Triton sort for Z <= 4096 else
# torch.argsort; 1: torch.argsort; 0: natural order). Order is result-free.
# Re-timed on synthetic length regimes (s15p1b pins gen_lengths.py), job 70738:
# V2 synthetic pick num_stages=2 by 1.07%, within the 1.36% round spread
# (result-free twin of num_stages=1) -> kept; any change of this tuple makes
# the v2 diet kernel decline.
_VKP_V2_AMAX_ROWMAX: bool = True
_VKP_V2_DQ_ELEM: bool = True
_VKP_V2_LPT: int = 2
_VKP_V2_NUM_STAGES: int = 1


@triton.jit
def _hstu_attn_delta_bwd_e5m2(
    Q,
    K,
    V,
    seq_offsets,
    DOut,
    DQ,
    DVK8,
    PARTIAL_COL,
    PARTIAL_AMAX,
    INV_SCALE,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dvk,
    alpha,
    K_SCALE,
    V_SCALE,
    H,
    MAX_SEQ_LEN,
    COL_K0,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LIMIT: tl.constexpr,
    REDUCE_EACH: tl.constexpr,
):
    """Same math as `_hstu_attn_delta_bwd`; dV|dK leave as E5M2 + partials.

    dV of head h occupies DVK8 columns [h*D_V, (h+1)*D_V); dK of head h
    occupies [COL_K0 + h*D_Q, ...). PARTIAL_COL is [Z, n_cols] fp32 (one row per
    sequence, each program writes its own head's columns), PARTIAL_AMAX is
    [Z*H] fp32. Rows of DVK8 not covered by any sequence are not touched.
    """
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = (off_hz % H).to(tl.int64)
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    inv = tl.load(INV_SCALE)

    offs_q = tl.arange(0, BLOCK_D_Q)
    offs_v = tl.arange(0, BLOCK_D_V)
    q = tl.load(
        Q + off_z * stride_qm + off_h * stride_qh + offs_q,
        mask=offs_q < BLOCK_D_Q,
        other=0.0,
    ).to(tl.float32)
    dout = tl.load(
        DOut + off_z * stride_dom + off_h * stride_doh + offs_v,
        mask=offs_v < BLOCK_D_V,
        other=0.0,
    ).to(tl.float32)
    dq = tl.zeros([BLOCK_D_Q], dtype=tl.float32)
    # REDUCE_EACH=False: element-wise register accumulators reduced ONCE after
    # the loop; True: [D] vectors reduced every block (fewer registers, one
    # cross-warp reduction per block). Same values either way up to fp32 order.
    if REDUCE_EACH:
        col_k = tl.zeros([BLOCK_D_Q], dtype=tl.float32)
        col_v = tl.zeros([BLOCK_D_V], dtype=tl.float32)
        amax_k = tl.zeros([BLOCK_D_Q], dtype=tl.float32)
        amax_v = tl.zeros([BLOCK_D_V], dtype=tl.float32)
    else:
        col_k = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
        col_v = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
        amax_k = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
        amax_v = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)

    for start_n in tl.range(0, seq_len, BLOCK_N, num_stages=1):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        k_ptrs = (
            K
            + (seq_start + offs_n[:, None]) * stride_kn
            + off_h * stride_kh
            + offs_q[None, :]
        )
        v_ptrs = (
            V
            + (seq_start + offs_n[:, None]) * stride_vn
            + off_h * stride_vh
            + offs_v[None, :]
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        k_real = (k.to(tl.float32) * K_SCALE).to(tl.bfloat16).to(tl.float32)
        v_real = (v.to(tl.float32) * V_SCALE).to(tl.bfloat16).to(tl.float32)

        qk = tl.sum(k_real * q[None, :], axis=1) * alpha
        sig = fast_dividef(1.0, 1.0 + tl.exp(-qk))
        silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * (1.0 / MAX_SEQ_LEN)
        silu = tl.where(mask_n, silu, 0.0).to(DOut.dtype.element_ty)

        dv = silu[:, None].to(tl.float32) * dout[None, :]
        dqk = tl.sum(v_real * dout[None, :], axis=1)
        dqk *= sig * (1.0 + qk * (1.0 - sig)) * (1.0 / MAX_SEQ_LEN)
        dqk = tl.where(mask_n, dqk, 0.0).to(tl.bfloat16)
        dk = dqk[:, None].to(tl.float32) * q[None, :] * alpha
        dq += tl.sum(dqk[:, None].to(tl.float32) * k_real, axis=0)

        # The value the stock kernel stores (BF16), back in fp32: what the
        # stock cast kernel would have loaded.
        dk_b = dk.to(tl.bfloat16).to(tl.float32)
        dv_b = dv.to(tl.bfloat16).to(tl.float32)
        # Masked rows are exactly 0 (silu/dqk zeroed above), so they add
        # nothing to the sums or the max.
        if REDUCE_EACH:
            col_k += tl.sum(dk_b, axis=0)
            col_v += tl.sum(dv_b, axis=0)
            amax_k = tl.maximum(amax_k, tl.max(tl.abs(dk_b), axis=0))
            amax_v = tl.maximum(amax_v, tl.max(tl.abs(dv_b), axis=0))
        else:
            col_k += dk_b
            col_v += dv_b
            amax_k = tl.maximum(amax_k, tl.abs(dk_b))
            amax_v = tl.maximum(amax_v, tl.abs(dv_b))
        dk8 = tl.minimum(tl.maximum(dk_b * inv, -LIMIT), LIMIT)
        dv8 = tl.minimum(tl.maximum(dv_b * inv, -LIMIT), LIMIT)

        row_ptr = DVK8 + (seq_start + offs_n[:, None]) * stride_dvk
        tl.store(
            row_ptr + off_h * BLOCK_D_V + offs_v[None, :],
            dv8.to(DVK8.dtype.element_ty),
            mask=mask_n[:, None],
        )
        tl.store(
            row_ptr + COL_K0 + off_h * BLOCK_D_Q + offs_q[None, :],
            dk8.to(DVK8.dtype.element_ty),
            mask=mask_n[:, None],
        )

    tl.store(
        DQ + off_z * stride_dqm + off_h * stride_dqh + offs_q,
        (dq * alpha).to(DQ.dtype.element_ty),
        mask=offs_q < BLOCK_D_Q,
    )
    col_row = PARTIAL_COL + off_z.to(tl.int64) * stride_dvk
    if REDUCE_EACH:
        tl.store(col_row + off_h * BLOCK_D_V + offs_v, col_v)
        tl.store(col_row + COL_K0 + off_h * BLOCK_D_Q + offs_q, col_k)
        tl.store(
            PARTIAL_AMAX + off_hz,
            tl.maximum(tl.max(amax_k, axis=0), tl.max(amax_v, axis=0)),
        )
    else:
        tl.store(col_row + off_h * BLOCK_D_V + offs_v, tl.sum(col_v, axis=0))
        tl.store(col_row + COL_K0 + off_h * BLOCK_D_Q + offs_q, tl.sum(col_k, axis=0))
        tl.store(
            PARTIAL_AMAX + off_hz,
            tl.maximum(
                tl.max(tl.max(amax_k, axis=1), axis=0),
                tl.max(tl.max(amax_v, axis=1), axis=0),
            ),
        )


@triton.jit
def _hstu_attn_delta_bwd_e5m2_lpt_perm(
    seq_offsets, PERM, Z, BLOCK_Z: tl.constexpr, ZBITS: tl.constexpr
):
    # One program: PERM = sequence ids by decreasing length (ties: lower id
    # first). Key = len << ZBITS | (BLOCK_Z - 1 - z); padded lanes sort last.
    # The length is clamped below 2^(31 - ZBITS) so a key never reaches the
    # sign bit: PERM is a permutation of [0, Z) for any offsets.
    offs = tl.arange(0, BLOCK_Z)
    m = offs < Z
    s0 = tl.load(seq_offsets + offs, mask=m, other=0)
    s1 = tl.load(seq_offsets + offs + 1, mask=m, other=0)
    ln = tl.minimum((s1 - s0).to(tl.int32), (1 << (31 - ZBITS)) - 1)
    key = tl.where(m, (ln << ZBITS) | ((BLOCK_Z - 1) - offs), -1)
    key = tl.sort(key, descending=True)
    tl.store(PERM + offs, (BLOCK_Z - 1) - (key & (BLOCK_Z - 1)), mask=m)


def _log_vkp_v2_armed(
    device: torch.device, z: int, h: int, rows: int, lpt: int
) -> None:
    # Imported here: only the v2 path needs bwd_fp8 (its ARMED-line helper).
    from generative_recommenders.ops.triton import bwd_fp8

    bwd_fp8.log_armed_v2(
        "vkp_v2",
        device,
        lambda: (
            f"ARMED vkp_v2 on {device}: _hstu_attn_delta_bwd_e5m2_v2 "
            f"(amax_rowmax={int(_VKP_V2_AMAX_ROWMAX)} "
            f"dq_elem={int(_VKP_V2_DQ_ELEM)} lpt={lpt} "
            f"num_stages={_VKP_V2_NUM_STAGES}; payload/amax/colsum bitwise vs "
            f"_hstu_attn_delta_bwd_e5m2, dq "
            f"{'fp32-order (DECLARED)' if _VKP_V2_DQ_ELEM else 'bitwise'}), "
            f"Z={z} H={h} rows={rows}"
        ),
    )


# DLRM_HSTU_FP8_DELTA_BWD_V2_DIET (X4C19 item (48)(b'); read per call, armed
# only by "1"; code default OFF, gin item (48) binds it on): the VALU-diet form
# of the v2 kernel (hstu_delta_bwd_e5m2_v2_diet.py), byte-identical outputs. It
# replaces the v2 launch ONLY in the configuration it is written for (the v2
# constants above as shipped, BLOCK_N=32, 4 warps, REDUCE_EACH off, D_Q = D_V
# = 128, bf16 q/dout/dQ, e4m3 K/V, e5m2 payload with its 57344 limit, gfx950)
# and only after a RUNTIME BITWISE SELF-CHECK on this device, Triton/LLVM build
# and argument specialization (_v2_diet_verdict; replaces the former Triton
# version pin "3.6.0"): the diet's inline asm mirrors the v2 build's codegen
# choices (codegen profile, hstu_delta_bwd_e5m2_v2_diet.RED_*), so a probe batch
# (random rows plus NaN / sNaN / inf / -0 / overflow / subnormal / clamp rows,
# edge lengths) runs through the v2 kernel and the diet kernel with the same
# Triton specialization as the call (per pointer argument: > 2 GiB storage class
# and 16-byte alignment; per int argument: its ==1 / %16 / i64 class) and every
# output byte must match; the first profile that matches is used for that
# specialization, none -> v2. Any other call keeps the v2 kernel and logs one
# "declined (<reason>)" line per reason. Inert without BWD_VKP_V2 (item (25)(b)).
_V2_DIET_LOGGED = [False]
_V2_DIET_DECLINE_LOGGED = set()
_V2_DIET_ARCH = {}
# specialization key -> (profile, diet binary hash, summary) when the self-check passed, (None, None, reason) when
# it did not. Kept across the fp8 warm-up (fp8_warmup._KEEP_GLOBALS): the check runs before RUN_START there.
_V2_DIET_VERDICT = {}
_MAX_INT_32 = 2**31 - 1


def delta_bwd_v2_diet_requested() -> bool:
    return os.environ.get("DLRM_HSTU_FP8_DELTA_BWD_V2_DIET", "0") == "1"


def _v2_diet_decline_reason(
    q, k, v, dout, dq, dvk_fp8, limit, block_n, num_warps, reduce_each
):
    consts = (_VKP_V2_AMAX_ROWMAX, _VKP_V2_DQ_ELEM, _VKP_V2_LPT, _VKP_V2_NUM_STAGES)
    if consts != (True, True, 2, 1):
        return f"v2 constants (amax_rowmax, dq_elem, lpt, num_stages)={consts}"
    if block_n != 32 or num_warps != 4 or reduce_each:
        return f"block_n={block_n} num_warps={num_warps} reduce_each={reduce_each}"
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        return f"D_Q={q.shape[-1]} D_K={k.shape[-1]} D_V={v.shape[-1]}"
    if (
        q.dtype != torch.bfloat16
        or dout.dtype != torch.bfloat16
        or dq.dtype != torch.bfloat16
    ):
        return f"q/dout/dq dtype {q.dtype}/{dout.dtype}/{dq.dtype}"
    if k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
        return f"k/v dtype {k.dtype}/{v.dtype}"
    if dvk_fp8.dtype != torch.float8_e5m2 or float(limit) != 57344.0:
        return f"payload dtype {dvk_fp8.dtype} limit={limit}"
    arch = _V2_DIET_ARCH.get(q.device)
    if arch is None:
        props = torch.cuda.get_device_properties(q.device)
        arch = _V2_DIET_ARCH[q.device] = str(getattr(props, "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        return f"arch {arch!r}"
    return None


def _spec_ptr(t: torch.Tensor):
    # Triton AMD pointer specialization: storage <= 2 GiB (buffer ops) and 16-byte alignment
    return (t.untyped_storage().size() <= _MAX_INT_32, t.data_ptr() % 16)


def _spec_int(x: int):
    # Triton int specialization: == 1, % 16 == 0, i32 / i64
    return (x == 1, x % 16 == 0, not -(2**31) <= x <= _MAX_INT_32)


def _v2_diet_args(q, k, v, dout, dq, dvk_fp8, seq_offsets, inv_scale, perm):
    H, D_V = q.shape[1], v.shape[2]
    ptrs = (q, k, v, seq_offsets, dout, dq, dvk_fp8, inv_scale, perm)
    ints = (q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), dout.stride(0),
            dout.stride(1), dq.stride(0), dq.stride(1), dvk_fp8.stride(0), H, H * D_V)
    return ptrs, ints


def _v2_diet_key(q, k, v, dout, dq, dvk_fp8, seq_offsets, inv_scale, perm, N, use_perm):
    ptrs, ints = _v2_diet_args(q, k, v, dout, dq, dvk_fp8, seq_offsets, inv_scale, perm)
    return (str(q.device), bool(use_perm), tuple(_spec_ptr(t) for t in ptrs),
            tuple(_spec_int(int(x)) for x in ints) + (_spec_int(int(N)),))


# Probe batch of the self-check: per sequence a length and a value kind. Lengths cover the 32-row tile edges, the
# masked tail after full tiles and one-row / empty sequences; kinds: "rand" (normal values: every fp32 rounding of
# the partial sums shows), "nan" (NaN payloads / sNaN / +-inf / -0 in K, V, q, dout, two NaN payloads meeting in
# one column), "big" (products overflow to inf, inf - inf), "tiny" (subnormal q / dout, -0).
# The 8 long "rand" sequences (32768 rows, 131072 row-heads) are the summation-order detector: products of bf16
# values are exact in fp32, so a different in-thread reduction order or fma contraction only shows where it moves
# the bf16 rounding of silu / dqk. Measured on Triton 3.9 with the 3.6 chain profile: the 1398-row probe alone
# passed it; with these rows it differs in 84 payload bytes, 5 dQ and 1329 column-sum elements (the production
# capture, 2.4M rows: 9797 payload bytes, 441 dQ bytes, 131520 column-sum bytes).
_PROBE = (
    (0, "rand"), (1, "rand"), (31, "rand"), (32, "rand"), (33, "rand"), (64, "rand"), (95, "rand"),
    (129, "rand"), (200, "rand"), (333, "rand"), (47, "nan"), (96, "nan"), (65, "big"), (40, "big"),
    (70, "tiny"), (2, "nan"), (160, "rand"),
) + ((4096, "rand"),) * 8


def _probe_values(dev, H, D):
    """Deterministic probe data for _PROBE: K / V e4m3 bytes [rows, H*D], q / dout bf16 [Z, H, D], host offsets."""
    g = torch.Generator(device=dev)
    g.manual_seed(20261103)
    Z = len(_PROBE)
    offs = [0]
    for n, _ in _PROBE:
        offs.append(offs[-1] + n)
    rows = offs[-1]

    def e4m3_bytes(n):
        # finite e4m3fn codes (|code| <= 0x7E), random sign
        c = torch.randint(0, 0x7F, (n,), generator=g, device=dev, dtype=torch.int32)
        sg = torch.randint(0, 2, (n,), generator=g, device=dev, dtype=torch.int32) * 0x80
        return (c | sg).to(torch.uint8).view(n // (H * D), H * D)

    kb = e4m3_bytes(rows * H * D)
    vb = e4m3_bytes(rows * H * D)
    q16 = (torch.randn((Z, H, D), generator=g, device=dev) * 0.5).to(torch.bfloat16).view(torch.int16)
    d16 = (torch.randn((Z, H, D), generator=g, device=dev) * 0.25).to(torch.bfloat16).view(torch.int16)

    def bits(x):  # bf16 bit pattern -> int16 value
        return x - 0x10000 if x >= 0x8000 else x

    for z, (n, kind) in enumerate(_PROBE):
        a = offs[z]
        if kind == "nan" and n > 0:
            # NaN codes (0x7F / 0xFF) and -0 (0x80) in K / V: both signs in one column over rows (column sums), and
            # in one row in two threads' columns (the qk / dqk cross-lane sums), -0 (columns taken mod H*D, heads
            # mod H: any H, D = 128)
            # A row's 8 threads hold 16 columns each and meet in the butterfly xor 4, xor 2, xor 1: threads t and
            # t^4 meet at the first step, t^2 / t^6 at the second, t^1 / t^3 / t^5 / t^7 at the last; each NaN
            # pair below meets at a stated step (the 3.9 v2 build's dqk xor-2 step is own + partner, the others
            # partner + own).
            HD = H * D
            for t, r, col, x in ((kb, a, 3, 0x7F), (kb, a, 100, 0xFF),       # K row a: threads 0 / 6, step 2
                                 (kb, a + n // 2, 130, 0xFF), (kb, a + n - 1, 260, 0x80),
                                 (vb, a, 5, 0x7F), (vb, a, 37, 0xFF),        # V row a: threads 0 / 2, step 2
                                 (vb, a + n - 1, 5, 0xFF), (vb, a + n - 1, 77, 0x7F),  # threads 0 / 4, step 1
                                 (vb, a + n // 2, 400, 0x80)):
                t[r, col % HD] = x
            # q / dout: NaN payloads of both signs meeting across threads of one head, sNaN, +inf meeting -inf, -0
            for h, col, x in ((0, 7, 0x7FC1), (0, 100, 0xFFC3),             # threads 0 / 6: step 2
                              (1, 9, 0x7F81), (1, 120, 0xFFC5),             # threads 0 / 7: step 3
                              (2, 11, 0xFF80), (2, 77, 0x7F80),             # threads 0 / 4: step 1 (-inf + inf)
                              (3, 0, 0x8000)):
                q16[z, h % H, col % D] = bits(x)
            for h, col, x in ((0, 5, 0xFFA0), (0, 100, 0x7FC1),             # threads 0 / 6: step 2
                              (1, 1, 0x7F80), (1, 120, 0xFFC5),             # threads 0 / 7: step 3
                              (2, 2, 0x7F85), (2, 70, 0xFFC3),              # threads 0 / 4: step 1
                              (3, 3, 0x8000), (3, 40, 0xFFC1), (3, 60, 0x7FC3)):  # threads 2 / 3: step 3
                d16[z, h % H, col % D] = bits(x)
        elif kind == "big":
            # bf16 magnitudes near the top (0x7F7F, 0x7F00), random signs: products overflow, partial sums inf - inf
            neg = torch.rand((H, D), generator=g, device=dev) < 0.5
            q16[z] = torch.where(neg, bits(0xFF7F), 0x7F7F).to(torch.int16)
            d16[z] = torch.where(neg.flip(-1), bits(0xFF00), 0x7F00).to(torch.int16)
        elif kind == "tiny":
            # subnormal bf16 (exponent 0) q / dout, random mantissas (incl. 0 = +-0) and signs
            for t in (q16, d16):
                m = torch.randint(0, 0x80, (H, D), generator=g, device=dev, dtype=torch.int32)
                sg = torch.randint(0, 2, (H, D), generator=g, device=dev, dtype=torch.int32)
                t[z] = torch.where(sg > 0, m - 0x8000, m).to(torch.int16)
    return kb, vb, q16.view(torch.bfloat16), d16.view(torch.bfloat16), offs


def _span_bytes(like, nrows):
    shape = (nrows,) + tuple(like.shape[1:])
    return (1 + sum((sz - 1) * st for sz, st in zip(shape, like.stride()))) * like.element_size()


def _carve(pool_u8, pos, like, nrows):
    """A view with like's dtype, inner shape and strides and nrows rows, placed in pool_u8 (uint8, 1-D) at or after
    byte `pos` with data_ptr % 16 == like.data_ptr() % 16 (Triton's alignment class). -> (view, next free byte)."""
    es = like.element_size()
    mis = like.data_ptr() % 16
    start = pos + ((mis - (pool_u8.data_ptr() + pos)) % 16)
    nbytes = _span_bytes(like, nrows)
    if start + nbytes > pool_u8.numel() or (pool_u8.data_ptr() + start) % es:
        raise RuntimeError("probe pool exhausted / misaligned")
    raw = pool_u8[start:start + nbytes].view(like.dtype)
    t = torch.as_strided(raw, (nrows,) + tuple(like.shape[1:]), like.stride())
    return t, (start + nbytes + 255) // 256 * 256


def _v2_diet_probe_run(profile, q, k, v, dout, dq, dvk_fp8, rows, seq_offsets, N, alpha, inv_scale, limit,
                       k_scale, v_scale, perm, use_perm, force_big=()):
    """One self-check of `profile`, every argument with the call's specialization class (storage class, alignment,
    strides; PERM / USE_PERM as the call); probe tensors named in force_big ("k", "v", "d8") get the > 2 GiB class
    whatever the call's. -> (ok, detail, diet binary hash)."""
    from generative_recommenders.ops.triton import hstu_delta_bwd_e5m2_v2_diet as _v2_diet

    dev = q.device
    H, D = q.shape[1], q.shape[2]
    Z = len(_PROBE)
    kb, vb, qv, dv, offs = _probe_values(dev, H, D)
    prow = offs[-1]
    inv_1 = inv_scale.reshape(1)[:1]
    tensors = [("k", k, prow), ("v", v, prow), ("q", q, Z), ("dout", dout, Z), ("dq", dq, Z),
               ("d8", dvk_fp8, prow), ("so", seq_offsets, Z + 1), ("inv", inv_1, 1)]
    if use_perm:
        tensors.append(("perm", perm, Z))
    need = {True: 0, False: 0}
    small_of = {name: _spec_ptr(t)[0] and name not in force_big for name, t, _ in tensors}
    for name, t, n in tensors:
        need[small_of[name]] += _span_bytes(t, n) + 512
    # pools: <= 2 GiB tensors from a fresh small buffer; > 2 GiB ones carved from the call's own payload storage
    # rows [0, rows) when the call's sequences cover exactly those rows (the call's launch then overwrites every
    # probe byte), else from a fresh > 2 GiB buffer
    big_pool = None
    if need[False]:
        so_ends = seq_offsets[[0, -1]].tolist()
        lo = dvk_fp8.storage_offset() * dvk_fp8.element_size()
        span = rows * dvk_fp8.stride(0) * dvk_fp8.element_size()
        if (not _spec_ptr(dvk_fp8)[0] and so_ends == [0, rows] and dvk_fp8.is_contiguous()
                and span >= need[False]):
            whole = torch.empty(0, dtype=torch.uint8, device=dev).set_(
                dvk_fp8.untyped_storage(), 0, (dvk_fp8.untyped_storage().size(),), (1,))
            big_pool = whole[lo:lo + span]
        else:
            big_pool = torch.empty(_MAX_INT_32 + 1 + need[False], dtype=torch.uint8, device=dev)
    small_pool = torch.empty(max(need[True], 1), dtype=torch.uint8, device=dev)
    pos = {True: 0, False: 0}
    P = {}
    for name, t, n in tensors:
        small = small_of[name]
        P[name], pos[small] = _carve(small_pool if small else big_pool, pos[small], t, n)
    P["k"].view(torch.uint8).copy_(kb.view(prow, H, D))
    P["v"].view(torch.uint8).copy_(vb.view(prow, H, D))
    P["q"].copy_(qv)
    P["dout"].copy_(dv)
    P["so"].copy_(torch.tensor(offs, dtype=torch.int64))
    lens = torch.tensor([n for n, _ in _PROBE])
    if use_perm:
        P["perm"].copy_(torch.argsort(lens, descending=True, stable=True).to(torch.int32))
    perm_p = P["perm"] if use_perm else P["so"]
    n_cols = dvk_fp8.shape[1]
    # scale sets: the call's; and a hard one (alpha sign flipped, subnormal k_real range, clamp-heavy inv)
    sets = ((alpha, k_scale, v_scale, float(inv_1.item())), (-3.0 * alpha, k_scale * 2.0**-120, v_scale * 4.0,
                                                              float(inv_1.item()) * 2.0**20))
    diffs = [0, 0, 0, 0]
    nan_only = [0, 0]
    ck_d = None
    for i_set, (al, ks, vs, inv) in enumerate(sets):
        P["inv"].fill_(inv)
        got = {}
        for which in ("v2", "diet"):
            P["d8"].view(torch.uint8).fill_(0xA5)
            P["dq"].view(torch.int16).fill_(-1)
            pc = torch.full((Z, n_cols), float("nan"), dtype=torch.float32, device=dev)
            pa = torch.full((Z * H,), float("nan"), dtype=torch.float32, device=dev)
            if which == "v2":
                _hstu_attn_delta_bwd_e5m2_v2[(Z * H,)](
                    Q=P["q"], K=P["k"], V=P["v"], seq_offsets=P["so"], DOut=P["dout"], DQ=P["dq"], DVK8=P["d8"],
                    PARTIAL_COL=pc, PARTIAL_AMAX=pa, INV_SCALE=P["inv"], stride_qm=q.stride(0),
                    stride_qh=q.stride(1), stride_kn=k.stride(0), stride_kh=k.stride(1), stride_vn=v.stride(0),
                    stride_vh=v.stride(1), stride_dom=dout.stride(0), stride_doh=dout.stride(1),
                    stride_dqm=dq.stride(0), stride_dqh=dq.stride(1), stride_dvk=dvk_fp8.stride(0), alpha=al,
                    K_SCALE=ks, V_SCALE=vs, H=H, MAX_SEQ_LEN=N, COL_K0=H * D, BLOCK_D_Q=D, BLOCK_D_V=D,
                    BLOCK_N=_BLOCK_N, LIMIT=limit, REDUCE_EACH=False, PERM=perm_p, USE_PERM=bool(use_perm),
                    NUM_STAGES=_VKP_V2_NUM_STAGES, AMAX_ROWMAX=_VKP_V2_AMAX_ROWMAX, DQ_ELEM=_VKP_V2_DQ_ELEM,
                    num_warps=_NUM_WARPS,
                )
            else:
                ck = _v2_diet.launch_kernel(
                    P["dout"], P["q"], P["k"], P["v"], P["dq"], P["d8"], P["so"], N, al, P["inv"], limit, ks,
                    vs, pc, pa, perm_p, bool(use_perm), profile=profile,
                )
                if i_set == 0:
                    ck_d = ck  # the call's own scalars: the binary the call launches
            got[which] = (P["d8"].view(torch.uint8).clone(), P["dq"].clone(), pc, pa)
        cmp = _v2_diet_compare(got["v2"], got["diet"])
        for j in range(4):
            diffs[j] += cmp[0][j]
        nan_only[0] += cmp[1][0]
        nan_only[1] += cmp[1][1]
    detail = (f"{Z} probe sequences / {prow} rows x {len(sets)} scale sets (random, NaN/sNaN/inf/-0, overflow, "
              f"subnormal, clamp), differing DVK8 bytes {diffs[0]}, DQ {diffs[1]}, PARTIAL_COL {diffs[2]}, "
              f"PARTIAL_AMAX {diffs[3]}, NaN payload/sign DQ {nan_only[0]}, PARTIAL_COL {nan_only[1]}")
    return sum(diffs) + sum(nan_only) == 0, detail, getattr(ck_d, "hash", None)


def _v2_diet_compare(a, b):
    """v2 outputs a vs diet outputs b, each (DVK8 bytes, DQ bf16, PARTIAL_COL f32, PARTIAL_AMAX f32) ->
    ([differing DVK8 bytes, DQ, PARTIAL_COL, PARTIAL_AMAX elements], [DQ, PARTIAL_COL elements that are NaN in both
    but differ in payload / sign]). The self-check needs both lists all zero (bit-identical incl. NaN payloads);
    the second list is reported apart because it is where an operand order LLVM picks for v2's reductions (not
    its arithmetic) shows: two NaNs of different payload / sign meeting in a cross-lane or cross-row sum."""
    d = [int((a[0] != b[0]).sum().item())]
    nan_only = []
    for x, y, it in ((a[1], b[1], torch.int16), (a[2], b[2], torch.int32)):
        ne = x.view(it) != y.view(it)
        both = torch.isnan(x) & torch.isnan(y)
        d.append(int((ne & ~both).sum().item()))
        nan_only.append(int((ne & both).sum().item()))
    d.append(int((a[3].view(torch.int32) != b[3].view(torch.int32)).sum().item()))
    return d, nan_only


def _v2_diet_verdict(key, *call, force_big=()):
    """Cached runtime self-check per specialization key: (profile, diet hash, summary) or (None, None, reason).
    force_big: a speculative check of another key (see _v2_diet_presibling); a probe error there caches nothing."""
    got = _V2_DIET_VERDICT.get(key)
    if got is not None:
        return got
    from generative_recommenders.ops.triton import hstu_delta_bwd_e5m2_v2_diet as _v2_diet

    tried = []
    verdict = None
    for profile in _v2_diet.profile_candidates(triton.__version__):
        name = _v2_diet.PROFILE_NAMES.get(profile, str(profile))
        try:
            ok, detail, h = _v2_diet_probe_run(profile, *call, force_big=force_big)
        except Exception as error:  # noqa: BLE001  (a failed probe keeps the v2 kernel)
            if force_big:
                return None
            ok, detail, h = False, f"probe error {type(error).__name__}: {str(error)[:160]}", None
        if ok:
            verdict = (profile, h, f"profile {name} (Triton {triton.__version__}): {detail}")
            break
        tried.append(f"profile {name}: {detail}")
    if verdict is None:
        verdict = (None, None, f"self-check vs v2 on {call[0].device} (Triton {triton.__version__}): "
                               + "; ".join(tried))
    _V2_DIET_VERDICT[key] = verdict
    return verdict


def _v2_diet_presibling(key, *call):
    """After a <= 2 GiB key passed: check its > 2 GiB siblings now, everything else as the call: (a) K, V and the
    payload in > 2 GiB storages (the in-model build), (b) only the payload > 2 GiB (rows within one uvqk_tm_pad of
    the 2 GiB boundary: the K/V pack fits, the padded payload does not; recorded once in the v26.3 TSPEC campaign),
    so that the first such call of the run finds its verdict and binaries ready (the fp8 warm-up's T2e item runs
    <= 2 GiB stores only; otherwise that first call would compile the v2 and diet builds and probe inside the timed
    region). Costs one transient > 2 GiB buffer per sibling; a probe error caches nothing (the real call then
    checks itself)."""
    ptrs = list(key[2])
    if not all(ptrs[i][0] for i in (1, 2, 6)):
        return
    for idx, names in (((1, 2, 6), ("k", "v", "d8")), ((6,), ("d8",))):
        sp = list(ptrs)
        for i in idx:
            sp[i] = (False, sp[i][1])
        sib = key[:2] + (tuple(sp),) + key[3:]
        if sib not in _V2_DIET_VERDICT:
            _v2_diet_verdict(sib, *call, force_big=names)


@triton.jit
def _hstu_attn_delta_bwd_e5m2_v2(
    Q,
    K,
    V,
    seq_offsets,
    DOut,
    DQ,
    DVK8,
    PARTIAL_COL,
    PARTIAL_AMAX,
    INV_SCALE,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dvk,
    alpha,
    K_SCALE,
    V_SCALE,
    H,
    MAX_SEQ_LEN,
    COL_K0,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LIMIT: tl.constexpr,
    REDUCE_EACH: tl.constexpr,
    PERM,
    USE_PERM: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    AMAX_ROWMAX: tl.constexpr,
    DQ_ELEM: tl.constexpr,
):
    """`_hstu_attn_delta_bwd_e5m2` v2 (DLRM_HSTU_FP8_BWD_VKP_V2, default off).

    Same outputs and layout as the shipped kernel. AMAX_ROWMAX: the amax from
    running row-maxima of |silu| and |dqk| (exact; payload, amax and colsum
    bitwise). DQ_ELEM: element-wise dq accumulator, one cross-row reduction
    after the loop (DECLARED fp32-order change of dq). USE_PERM: program ->
    sequence through PERM (longest first); outputs are per sequence, so the
    order does not change any value.
    """
    off_hz = tl.program_id(0)
    if USE_PERM:
        # longest-first (LPT) sequence order; outputs are per-sequence, order-free
        off_z = tl.load(PERM + off_hz // H)
    else:
        off_z = off_hz // H
    off_h = (off_hz % H).to(tl.int64)
    off_hz = off_z * H + off_h
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    inv = tl.load(INV_SCALE)

    offs_q = tl.arange(0, BLOCK_D_Q)
    offs_v = tl.arange(0, BLOCK_D_V)
    q = tl.load(
        Q + off_z * stride_qm + off_h * stride_qh + offs_q,
        mask=offs_q < BLOCK_D_Q,
        other=0.0,
    ).to(tl.float32)
    dout = tl.load(
        DOut + off_z * stride_dom + off_h * stride_doh + offs_v,
        mask=offs_v < BLOCK_D_V,
        other=0.0,
    ).to(tl.float32)
    if DQ_ELEM:
        # DECLARED fp32-order change: element-wise [BLOCK_N, D] accumulator over
        # the tiles, one cross-row reduction after the loop (ship10: one
        # cross-warp reduction per tile, summed in tile order).
        dq_acc = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    else:
        dq = tl.zeros([BLOCK_D_Q], dtype=tl.float32)
    if AMAX_ROWMAX:
        # Exact: dv = s_n*dout_d and dk = (dqk_n*q_d)*alpha are products of bf16
        # values (exact in fp32) followed by monotone roundings, hence
        # max|dv_b| = bf16(max|s| * max|dout|), max|dk_b| = bf16((max|dqk| * max|q|) * |alpha|) (RNE is sign-symmetric).
        m_s = tl.zeros([BLOCK_N], dtype=tl.float32)
        m_d = tl.zeros([BLOCK_N], dtype=tl.float32)
    # REDUCE_EACH=False: element-wise register accumulators reduced ONCE after
    # the loop; True: [D] vectors reduced every block (fewer registers, one
    # cross-warp reduction per block). Same values either way up to fp32 order.
    if REDUCE_EACH:
        col_k = tl.zeros([BLOCK_D_Q], dtype=tl.float32)
        col_v = tl.zeros([BLOCK_D_V], dtype=tl.float32)
        amax_k = tl.zeros([BLOCK_D_Q], dtype=tl.float32)
        amax_v = tl.zeros([BLOCK_D_V], dtype=tl.float32)
    else:
        col_k = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
        col_v = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
        if not AMAX_ROWMAX:
            amax_k = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
            amax_v = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)

    for start_n in tl.range(0, seq_len, BLOCK_N, num_stages=NUM_STAGES):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        k_ptrs = (
            K
            + (seq_start + offs_n[:, None]) * stride_kn
            + off_h * stride_kh
            + offs_q[None, :]
        )
        v_ptrs = (
            V
            + (seq_start + offs_n[:, None]) * stride_vn
            + off_h * stride_vh
            + offs_v[None, :]
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        k_real = (k.to(tl.float32) * K_SCALE).to(tl.bfloat16).to(tl.float32)
        v_real = (v.to(tl.float32) * V_SCALE).to(tl.bfloat16).to(tl.float32)

        qk = tl.sum(k_real * q[None, :], axis=1) * alpha
        sig = fast_dividef(1.0, 1.0 + tl.exp(-qk))
        silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * (1.0 / MAX_SEQ_LEN)
        silu = tl.where(mask_n, silu, 0.0).to(DOut.dtype.element_ty)

        dv = silu[:, None].to(tl.float32) * dout[None, :]
        dqk = tl.sum(v_real * dout[None, :], axis=1)
        dqk *= sig * (1.0 + qk * (1.0 - sig)) * (1.0 / MAX_SEQ_LEN)
        dqk = tl.where(mask_n, dqk, 0.0).to(tl.bfloat16)
        dk = dqk[:, None].to(tl.float32) * q[None, :] * alpha
        if DQ_ELEM:
            dq_acc += dqk[:, None].to(tl.float32) * k_real
        else:
            dq += tl.sum(dqk[:, None].to(tl.float32) * k_real, axis=0)

        # The value the stock kernel stores (BF16), back in fp32: what the
        # stock cast kernel would have loaded.
        dk_b = dk.to(tl.bfloat16).to(tl.float32)
        dv_b = dv.to(tl.bfloat16).to(tl.float32)
        # Masked rows are exactly 0 (silu/dqk zeroed above), so they add
        # nothing to the sums or the max.
        if REDUCE_EACH:
            col_k += tl.sum(dk_b, axis=0)
            col_v += tl.sum(dv_b, axis=0)
            amax_k = tl.maximum(amax_k, tl.max(tl.abs(dk_b), axis=0))
            amax_v = tl.maximum(amax_v, tl.max(tl.abs(dv_b), axis=0))
        else:
            col_k += dk_b
            col_v += dv_b
            if AMAX_ROWMAX:
                m_s = tl.maximum(m_s, tl.abs(silu.to(tl.float32)))
                m_d = tl.maximum(m_d, tl.abs(dqk.to(tl.float32)))
            else:
                amax_k = tl.maximum(amax_k, tl.abs(dk_b))
                amax_v = tl.maximum(amax_v, tl.abs(dv_b))
        dk8 = tl.minimum(tl.maximum(dk_b * inv, -LIMIT), LIMIT)
        dv8 = tl.minimum(tl.maximum(dv_b * inv, -LIMIT), LIMIT)

        row_ptr = DVK8 + (seq_start + offs_n[:, None]) * stride_dvk
        tl.store(
            row_ptr + off_h * BLOCK_D_V + offs_v[None, :],
            dv8.to(DVK8.dtype.element_ty),
            mask=mask_n[:, None],
        )
        tl.store(
            row_ptr + COL_K0 + off_h * BLOCK_D_Q + offs_q[None, :],
            dk8.to(DVK8.dtype.element_ty),
            mask=mask_n[:, None],
        )

    if DQ_ELEM:
        dq = tl.sum(dq_acc, axis=0)
    tl.store(
        DQ + off_z * stride_dqm + off_h * stride_dqh + offs_q,
        (dq * alpha).to(DQ.dtype.element_ty),
        mask=offs_q < BLOCK_D_Q,
    )
    col_row = PARTIAL_COL + off_z.to(tl.int64) * stride_dvk
    if REDUCE_EACH:
        tl.store(col_row + off_h * BLOCK_D_V + offs_v, col_v)
        tl.store(col_row + COL_K0 + off_h * BLOCK_D_Q + offs_q, col_k)
        tl.store(
            PARTIAL_AMAX + off_hz,
            tl.maximum(tl.max(amax_k, axis=0), tl.max(amax_v, axis=0)),
        )
    else:
        tl.store(col_row + off_h * BLOCK_D_V + offs_v, tl.sum(col_v, axis=0))
        tl.store(col_row + COL_K0 + off_h * BLOCK_D_Q + offs_q, tl.sum(col_k, axis=0))
        if AMAX_ROWMAX:
            av = (tl.max(m_s, axis=0) * tl.max(tl.abs(dout), axis=0)).to(tl.bfloat16).to(tl.float32)
            ak = ((tl.max(m_d, axis=0) * tl.max(tl.abs(q), axis=0)) * tl.abs(alpha)).to(tl.bfloat16).to(tl.float32)
            tl.store(PARTIAL_AMAX + off_hz, tl.maximum(ak, av))
        else:
            tl.store(
                PARTIAL_AMAX + off_hz,
                tl.maximum(
                    tl.max(tl.max(amax_k, axis=1), axis=0),
                    tl.max(tl.max(amax_v, axis=1), axis=0),
                ),
            )


def hstu_attention_delta_bwd_e5m2(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dvk_fp8: torch.Tensor,
    rows: int,
    seq_offsets: torch.Tensor,
    N: int,
    alpha: float,
    inv_scale: torch.Tensor,
    limit: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    block_n: int = _BLOCK_N,
    num_warps: int = _NUM_WARPS,
    reduce_each: bool = _REDUCE_EACH,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Delta attention backward writing dQ (BF16) and the E5M2 dV|dK pack.

    `dvk_fp8` is a contiguous [Mp, H*D_V + H*D_Q] FP8 tensor, Mp >= rows; rows
    [rows, Mp) are zeroed here (K padding for the token-major wgrad). Returns
    (amax, colsum fp32 [n_cols]) of the BF16 dV|dK the stock kernel would have
    stored -- the same triple `bwd_fp8.quantize_amax_colsum` returns.
    """
    assert dvk_fp8.dim() == 2 and dvk_fp8.is_contiguous()
    # Same input handling as the stock wrapper: the kernel takes row/head
    # strides, only the innermost dimension must be unit-stride.
    dout = switch_to_contiguous_if_needed(dout)
    q = switch_to_contiguous_if_needed(q)
    k = switch_to_contiguous_if_needed(k)
    v = switch_to_contiguous_if_needed(v)
    assert dq.stride(-1) == 1
    Z = seq_offsets.numel() - 1
    _, H, DimQ = q.shape
    _, _, DimV = v.shape
    n_cols = H * (DimQ + DimV)
    assert dvk_fp8.shape[1] == n_cols and dvk_fp8.shape[0] >= rows
    assert k.shape[0] == rows and v.shape[0] == rows
    if q.shape[0] != Z or dout.shape[0] != Z:
        raise ValueError("delta attention training requires one query per sequence")
    if dvk_fp8.shape[0] > rows:
        dvk_fp8[rows:].zero_()
    partial_col = torch.empty((max(Z, 1), n_cols), dtype=torch.float32, device=q.device)
    partial_amax = torch.empty(max(Z * H, 1), dtype=torch.float32, device=q.device)
    if Z == 0:
        partial_col.zero_()
        partial_amax.zero_()
        return partial_amax.amax(), partial_col.sum(dim=0)
    if _VKP_V2:
        lpt = _VKP_V2_LPT
        perm = seq_offsets  # placeholder pointer when USE_PERM is False
        if lpt == 2 and Z <= 4096 and N < (1 << 19):
            # longest-processing-time-first launch order (one tiny sort kernel)
            bz = max(16, triton.next_power_of_2(Z))
            perm = torch.empty(Z, dtype=torch.int32, device=q.device)
            _hstu_attn_delta_bwd_e5m2_lpt_perm[(1,)](
                seq_offsets, perm, Z, BLOCK_Z=bz, ZBITS=bz.bit_length() - 1, num_warps=4
            )
        elif lpt:
            lens = seq_offsets[1:] - seq_offsets[:-1]
            perm = torch.argsort(lens, descending=True, stable=True).to(torch.int32)
        if delta_bwd_v2_diet_requested():
            why = _v2_diet_decline_reason(
                q, k, v, dout, dq, dvk_fp8, limit, block_n, num_warps, reduce_each
            )
            if why is None:
                key = _v2_diet_key(q, k, v, dout, dq, dvk_fp8, seq_offsets, inv_scale, perm, N, lpt)
                fresh = key not in _V2_DIET_VERDICT
                profile, vhash, summary = _v2_diet_verdict(
                    key, q, k, v, dout, dq, dvk_fp8, rows, seq_offsets, N, alpha, inv_scale, limit, k_scale,
                    v_scale, perm, bool(lpt),
                )
                if profile is None:
                    why = summary
                elif fresh:
                    # once per key, right after its own check (in training: the fp8 warm-up's T2e item)
                    _v2_diet_presibling(
                        key, q, k, v, dout, dq, dvk_fp8, rows, seq_offsets, N, alpha, inv_scale, limit, k_scale,
                        v_scale, perm, bool(lpt),
                    )
            if why is None:
                # Item (48)(b'): same grid, PERM and outputs as the v2 launch.
                from generative_recommenders.ops.triton import (
                    hstu_delta_bwd_e5m2_v2_diet as _v2_diet,
                )

                if not _V2_DIET_LOGGED[0]:
                    _V2_DIET_LOGGED[0] = True
                    print(
                        "[hstu_delta_bwd_e5m2] DLRM_HSTU_FP8_DELTA_BWD_V2_DIET=1 "
                        "armed: VALU-diet kernel _hstu_attn_delta_bwd_e5m2_v2_diet "
                        "(bitwise vs _hstu_attn_delta_bwd_e5m2_v2: runtime self-check on "
                        f"{q.device}, {summary}), (stages, peel, sp)="
                        f"{_v2_diet.default_config(k, v, dvk_fp8)}",
                        flush=True,
                    )
                ck = _v2_diet.launch_kernel(
                    dout, q, k, v, dq, dvk_fp8, seq_offsets, N, alpha, inv_scale,
                    limit, k_scale, v_scale, partial_col, partial_amax, perm,
                    bool(lpt), profile=profile,
                )
                if vhash is None or getattr(ck, "hash", vhash) == vhash:
                    _log_vkp_v2_armed(q.device, Z, H, rows, lpt)
                    return partial_amax.amax(), partial_col.sum(dim=0)
                # the launch used a binary the self-check did not run: v2 recomputes every output, and this
                # specialization stays on v2
                why = f"diet binary {getattr(ck, 'hash', '?')[:12]} != self-checked {vhash[:12]}"
                _V2_DIET_VERDICT[key] = (None, None, why)
            if why not in _V2_DIET_DECLINE_LOGGED:
                _V2_DIET_DECLINE_LOGGED.add(why)
                print(
                    "[hstu_delta_bwd_e5m2] DLRM_HSTU_FP8_DELTA_BWD_V2_DIET=1 "
                    f"declined ({why}): v2 kernel",
                    flush=True,
                )
        _hstu_attn_delta_bwd_e5m2_v2[(Z * H,)](
            Q=q,
            K=k,
            V=v,
            seq_offsets=seq_offsets,
            DOut=dout,
            DQ=dq,
            DVK8=dvk_fp8,
            PARTIAL_COL=partial_col,
            PARTIAL_AMAX=partial_amax,
            INV_SCALE=inv_scale,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_vn=v.stride(0),
            stride_vh=v.stride(1),
            stride_dom=dout.stride(0),
            stride_doh=dout.stride(1),
            stride_dqm=dq.stride(0),
            stride_dqh=dq.stride(1),
            stride_dvk=dvk_fp8.stride(0),
            alpha=alpha,
            K_SCALE=k_scale,
            V_SCALE=v_scale,
            H=H,
            MAX_SEQ_LEN=N,
            COL_K0=H * DimV,
            BLOCK_D_Q=DimQ,
            BLOCK_D_V=DimV,
            BLOCK_N=block_n,
            LIMIT=limit,
            REDUCE_EACH=reduce_each,
            PERM=perm,
            USE_PERM=bool(lpt),
            NUM_STAGES=_VKP_V2_NUM_STAGES,
            AMAX_ROWMAX=_VKP_V2_AMAX_ROWMAX,
            DQ_ELEM=_VKP_V2_DQ_ELEM,
            num_warps=num_warps,
        )
        _log_vkp_v2_armed(q.device, Z, H, rows, lpt)
        return partial_amax.amax(), partial_col.sum(dim=0)
    _hstu_attn_delta_bwd_e5m2[(Z * H,)](
        Q=q,
        K=k,
        V=v,
        seq_offsets=seq_offsets,
        DOut=dout,
        DQ=dq,
        DVK8=dvk_fp8,
        PARTIAL_COL=partial_col,
        PARTIAL_AMAX=partial_amax,
        INV_SCALE=inv_scale,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_dom=dout.stride(0),
        stride_doh=dout.stride(1),
        stride_dqm=dq.stride(0),
        stride_dqh=dq.stride(1),
        stride_dvk=dvk_fp8.stride(0),
        alpha=alpha,
        K_SCALE=k_scale,
        V_SCALE=v_scale,
        H=H,
        MAX_SEQ_LEN=N,
        COL_K0=H * DimV,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        BLOCK_N=block_n,
        LIMIT=limit,
        REDUCE_EACH=reduce_each,
        num_warps=num_warps,
    )
    return partial_amax.amax(), partial_col.sum(dim=0)
