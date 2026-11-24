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

"""dq-split e5m2 producer for the ordinary HSTU layer (Arbor R14 M2, L1a-dq).

DLRM_HSTU_FP8_BWD_DQ_E5M2=1 (default off; read by bwd_fp8 at import). With the
FP8 uvqk projection backward armed (DLRM_HSTU_FP8_BWD_UVQK=1, WGRAD=fp8t) and
the C8 attention backward on its dq-split path (DLRM_HSTU_FP8_BWD_DQ_SPLIT=1
with a DQ_SPLIT_SCHED), the shipped ordinary-layer path is

    _dq_from_ds_sched_kernel  -> dq   bf16 duvqk[:, q columns]   (write 2B/elem)
    _quantize_amax_colsum     -> e5m2 dz [Mp, u|v|q|k] + amax + bias colsum
                                 (read 2B/elem of all 2048 columns)

With the knob, the ordinary backward hands the FlyDSL attention backward a
`DqE5m2Sink` (the optional `dq_fp8_sink` hook of hstu_flydsl_mha); the
dq-split call then runs `_dq8_sched_kernel` below instead of r15's
`_dq_from_ds_sched_kernel`:

  * main loop: `hstu_dq_split._dq_unit` verbatim (same tiles, same dot order,
    same `out = acc * (rs * sc)` and the same BF16 rounding);
  * epilogue: the BF16 value back to fp32, x inv_scale, clamp +-LIMIT, cast to
    e5m2 into the q columns of the layer's dz8 [Mp, 2048] (exactly the bytes
    `bwd_fp8.quantize_amax_colsum` writes for those columns); per (b, h,
    128-row block) an MFMA column sum (ones[16, 128] @ x_bf16) and a flat amax
    partial; the BF16 dq is NOT stored (its duvqk columns stay unwritten);
  * compiled with waves_per_eu=4 (otherwise 132-134 VGPRs, occupancy 4 -> 3,
    +1.2-1.5 ms per call; iter-32 dqprod).

The consumer (`DqE5m2Sink.finish`) then casts ONLY the u|v|k columns of the
BF16 duvqk (`_quantize_amax_colsum_skip_kernel`: the shipped cast tile on the
COLSUM_V2 1-D XCD-grouped launch with the q column blocks skipped), reduces the
dq partials (deterministic 2-stage tree) and combines: amax = max(u|v|k amax,
q amax) and colsum = u|v|k column sums with the q block replaced. The result
feeds `bwd_fp8.uvqk_projection_backward(prequant=...)` (per-layer scale).

Numerics (DECLARED): the e5m2 payload (all 2048 columns, pad rows zero) and
the amax are byte-identical to the shipped cast, so every GEMM operand and the
delayed-scale state are unchanged; the u|v|k bias column sums are bitwise; the
512 q-column bias sums are the same fp32 math in a different reduction order
(norm-rel error vs fp64 1.6-1.8e-7 against the shipped 1.4-1.6e-7).

Fork of r15's ops/flydsl/hstu_dq_split.py (`_dq_unit` /
`_dq_from_ds_sched_kernel`, left untouched); keep the main loop in sync.
Gated in Arbor r14 iter 32 (probes/iter32b/tracks/dqprod) and iter 34 (m2).
"""

import os
from typing import Optional, Sequence, Tuple, Union  # B1a (r16): Sequence, Union

import torch
import triton
import triton.language as tl

from generative_recommenders.ops.flydsl import hstu_dq_split as _dqs

# The fork exceeds 128 VGPRs without the cap (occupancy 4 -> 3).
# Selection evidence for this kernel and its launch constants (user ruling 2026-10-13: an in-model A/B of
# the benchmark run suffices for speed-only kernel choices): in-model ABBA trace attribution of the r14
# iteration-34 bench (job 63630, 1n, steps 53-56, 8 ranks): dq / dq8 / colsum kernel families -1.14 ms per
# rank-step (32/32 pairs), M2 net about -0.76 (r14 FINDINGS_iter35 section 1). The iter-32 lab numbers in the
# docstring were timed on recorded length pools and are characterization only.
_WAVES_PER_EU: int = 4
# Stage-1 programs of the dq partial reduction.
_FIN_PROGS: int = 256
_SCHEDS = ("il", "lpt", "ilb", "ilr", "bc")
# Triton launch options dq_from_ds forwards through **launch_kw (e.g. the
# DQ_SPLIT_OPT e<N> token); anything else makes the sink decline.
_LAUNCH_OPTS = frozenset(("waves_per_eu", "num_stages", "num_ctas",
                          "matrix_instr_nonkdim", "kpack"))
# Gin item (91) (r14): dq8 epilogue diet V3. Read once at import like the other DLRM_HSTU_FP8_
# kernel switches; only "1" arms it (absent = off in code; the gin binding turns it on); env "0" is the
# kill switch (a pre-set env value wins over the binding).
_DQ8_DIET_ENV = os.environ.get("DLRM_HSTU_FP8_DQ8_DIET")  # the import-time value (arming line)
_DQ8_DIET: bool = _DQ8_DIET_ENV == "1"
_DQ8_DIET_LOGGED: bool = False


def _no_packed_fp32_launch_opts() -> dict:
    """Launch option that compiles `_dq8_sched_kernel` without packed-FP32
    VALU instructions (v_pk_mul/add/fma_f32): {"llvm_fn_attrs":
    "target-features=-packed-fp32-ops"} for a gfx950 target on Triton builds
    whose HIP backend takes LLVM function attributes (3.9), {} otherwise
    (Triton 3.6 or another target: the launch and the binary are unchanged).
    Resolved once, at the first sink launch.

    Why: in the epilogue each lane holds two rows of the 128-row tile (two
    MFMA repetitions), so LLVM packs the per-row scale products into
    v_pk_mul_f32 and folds the scale pair v[s:s+1] in with op_sel. For
    _dq8_epi_full, Triton 3.9's bundled LLVM selects `v_pk_mul_f32 vD, vA,
    v[s:s+1] op_sel:[0,1]` (low result = src0.lo * src1.HI) for the second
    repetition's rows. That encoding is not valid on gfx950: AMD LLVM 24
    (ROCm 10.2) rejects it ("invalid op_sel operand"), while gfx90a / gfx942
    accept it. On MI355X the low result was a signed zero in lanes 48-63 of
    a few tiles per launch, so the dz8 q bytes and the pcol partials changed
    from one launch to the next. In the relaunch census 100% of the wrong
    elements were those low results. Triton 3.6's LLVM only emitted the
    op_sel_hi:[1,0] form here. Without packed-FP32 instructions the products
    are plain v_mul_f32: the same IEEE fp32 multiply, so bit-identical, and
    the compiler still inserts the MFMA-result wait states (inline asm would
    not get them).

    hstu_dq_e5m2_test's Dq8IsaDeterminismTest checks the compiled ISA of both
    epilogues for the invalid forms. It also requires DIET=1 to equal DIET=0
    bitwise across relaunches.
    """
    global _DQ8_FN_OPTS
    if _DQ8_FN_OPTS is None:
        opts = {}
        try:
            from triton.backends.amd.compiler import HIPOptions

            target = triton.runtime.driver.active.get_current_target()
            if (target.backend == "hip" and str(target.arch).startswith("gfx950")
                    and "llvm_fn_attrs" in getattr(HIPOptions, "__dataclass_fields__", {})):
                opts = {"llvm_fn_attrs": "target-features=-packed-fp32-ops"}
        except Exception:  # noqa: BLE001 (no HIP backend / driver: leave the launch as is)
            opts = {}
        _DQ8_FN_OPTS = opts
    return _DQ8_FN_OPTS


_DQ8_FN_OPTS: Optional[dict] = None


@triton.jit
def _dq8_unit_v0(
    ws_ptr, hdr_ptr, k_ptr, e8_ptr, cu_ptr, dq_ptr, sc, WS_HDR, rs_k, rs_dq,
    b, h, u, dz8_ptr, rs8, pfx_ptr, pcol_ptr, pamax_ptr, inv_ptr,
    H: tl.constexpr, D: tl.constexpr, BMQ: tl.constexpr,
    FP8_DOT: tl.constexpr, STAGES: tl.constexpr, NBLK: tl.constexpr,
    RM: tl.constexpr, CMW: tl.constexpr, EVW: tl.constexpr,
    CMK: tl.constexpr, EVK: tl.constexpr, SPTR: tl.constexpr,
    WS_C64: tl.constexpr, STORE_BF16: tl.constexpr, LIMIT: tl.constexpr,
):
    """`hstu_dq_split._dq_unit` + the e5m2 / partials epilogue (red 12)."""
    st = tl.load(cu_ptr + b).to(tl.int64)
    L = tl.load(cu_ptr + b + 1).to(tl.int64) - st
    m0 = u * BMQ
    if m0 < L:
        nb = (L + 127) // 128
        pre = tl.load(hdr_ptr + b).to(tl.int64)
        base = WS_HDR + (pre + h * nb * (nb + 1)) * 8192
        t0 = m0 // 64
        offs_m = tl.arange(0, BMQ)
        offs_k = tl.arange(0, 128 * NBLK)
        offs_d = tl.arange(0, D)
        kb = offs_k // 128
        kr = offs_k % 128
        if WS_C64:
            kp = ((kr // 16) % 2) * 64 + (kr // 32) * 16 + kr % 16
        else:
            kp = kr
        acc = tl.zeros((BMQ, D), dtype=tl.float32)
        nj = (t0 + BMQ // 64 - 1) // 2 + 1
        for jc in tl.range(0, (nj + NBLK - 1) // NBLK, num_stages=STAGES):
            if SPTR:
                pa = ws_ptr + base + (2 * jc * nb - jc * (jc - 1)
                                      + (t0 - 2 * jc)) * 8192
                aoff = offs_m[:, None] * 128 + kp[None, :]
                if RM:
                    a = tl.load(pa + aoff, mask=((m0 + offs_m) < L)[:, None],
                                other=0, cache_modifier=CMW,
                                eviction_policy=EVW)
                else:
                    a = tl.load(pa + aoff, cache_modifier=CMW,
                                eviction_policy=EVW)
                pk = k_ptr + (st + 128 * jc) * rs_k + h * D
                kt = tl.load(pk + kr[:, None] * rs_k + offs_d[None, :],
                             mask=((128 * jc + kr) < L)[:, None], other=0,
                             cache_modifier=CMK, eviction_policy=EVK)
            else:
                j = jc * NBLK + kb
                slot = 2 * j * nb - j * (j - 1) + (t0 - 2 * j)
                ok = ((t0 + offs_m // 64)[:, None] >= 2 * j[None, :]) & \
                    (j < nj)[None, :]
                if RM:
                    ok = ok & ((m0 + offs_m) < L)[:, None]
                a = tl.load(ws_ptr + base + slot[None, :] * 8192
                            + offs_m[:, None] * 128 + kp[None, :], mask=ok,
                            other=0, cache_modifier=CMW, eviction_policy=EVW)
                kn = 128 * j + kr
                kt = tl.load(k_ptr + (st + kn)[:, None] * rs_k + h * D
                             + offs_d[None, :], mask=(kn < L)[:, None],
                             other=0, cache_modifier=CMK, eviction_policy=EVK)
            a = a.to(tl.float8e4nv, bitcast=True)
            kt = kt.to(tl.float8e4nv, bitcast=True)
            if FP8_DOT:
                acc = tl.dot(a, kt, acc)
            else:
                acc = tl.dot(a.to(tl.bfloat16), kt.to(tl.bfloat16), acc)
        rows = m0 + offs_m
        rmask = rows < L
        e8 = tl.load(e8_ptr + (st + rows) * H + h, mask=rmask, other=127)
        rs = (e8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        out = acc * (rs * sc)[:, None]
        ob = out.to(tl.bfloat16)
        if STORE_BF16:
            # Diagnostic dual store (gates only; costs +0.6-0.9 ms per call).
            tl.store(dq_ptr + (st + rows)[:, None] * rs_dq + h * D
                     + offs_d[None, :], ob, mask=rmask[:, None])
        # The value the shipped cast kernel loads, back in fp32, cast with the
        # same op order as _quantize_amax_colsum_kernel.
        x = tl.where(rmask[:, None], ob.to(tl.float32), 0.0)
        inv = tl.load(inv_ptr)
        y = tl.minimum(tl.maximum(x * inv, -LIMIT), LIMIT)
        tl.store(dz8_ptr + (st + rows)[:, None] * rs8 + h * D
                 + offs_d[None, :], y.to(dz8_ptr.dtype.element_ty),
                 mask=rmask[:, None])
        prow = tl.load(pfx_ptr + b).to(tl.int64) + u
        # Column sum by MFMA: ones[16, BMQ] @ x (x is bf16-exact, products
        # exact, fp32 accumulate in hardware order); row 0 is stored.
        ones = tl.full((16, BMQ), 1.0, tl.bfloat16)
        xb = tl.where(rmask[:, None], ob, 0.0).to(tl.bfloat16)
        cs16 = tl.dot(ones, xb)
        r16 = tl.arange(0, 16)
        tl.store(pcol_ptr + prow * (H * D) + h * D + offs_d[None, :]
                 + 0 * r16[:, None], cs16, mask=(r16 == 0)[:, None])
        am = tl.max(tl.reshape(tl.abs(x), (BMQ * D,), can_reorder=True),
                    axis=0)
        tl.store(pamax_ptr + prow * H + h, am)


@triton.jit
def _dq8_unit_v3(
    ws_ptr, hdr_ptr, k_ptr, e8_ptr, cu_ptr, dq_ptr, sc, WS_HDR, rs_k, rs_dq,
    b, h, u, dz8_ptr, rs8, pfx_ptr, pcol_ptr, pamax_ptr, inv_ptr,
    H: tl.constexpr, D: tl.constexpr, BMQ: tl.constexpr,
    FP8_DOT: tl.constexpr, STAGES: tl.constexpr, NBLK: tl.constexpr,
    RM: tl.constexpr, CMW: tl.constexpr, EVW: tl.constexpr,
    CMK: tl.constexpr, EVK: tl.constexpr, SPTR: tl.constexpr,
    WS_C64: tl.constexpr, STORE_BF16: tl.constexpr, LIMIT: tl.constexpr,
):
    """`hstu_dq_split._dq_unit` + the e5m2 / partials epilogue (red 12)."""
    st = tl.load(cu_ptr + b).to(tl.int64)
    L = tl.load(cu_ptr + b + 1).to(tl.int64) - st
    m0 = u * BMQ
    if m0 < L:
        nb = (L + 127) // 128
        pre = tl.load(hdr_ptr + b).to(tl.int64)
        base = WS_HDR + (pre + h * nb * (nb + 1)) * 8192
        t0 = m0 // 64
        offs_m = tl.arange(0, BMQ)
        offs_k = tl.arange(0, 128 * NBLK)
        offs_d = tl.arange(0, D)
        kb = offs_k // 128
        kr = offs_k % 128
        if WS_C64:
            kp = ((kr // 16) % 2) * 64 + (kr // 32) * 16 + kr % 16
        else:
            kp = kr
        acc = tl.zeros((BMQ, D), dtype=tl.float32)
        nj = (t0 + BMQ // 64 - 1) // 2 + 1
        for jc in tl.range(0, (nj + NBLK - 1) // NBLK, num_stages=STAGES):
            if SPTR:
                pa = ws_ptr + base + (2 * jc * nb - jc * (jc - 1)
                                      + (t0 - 2 * jc)) * 8192
                aoff = offs_m[:, None] * 128 + kp[None, :]
                if RM:
                    a = tl.load(pa + aoff, mask=((m0 + offs_m) < L)[:, None],
                                other=0, cache_modifier=CMW,
                                eviction_policy=EVW)
                else:
                    a = tl.load(pa + aoff, cache_modifier=CMW,
                                eviction_policy=EVW)
                pk = k_ptr + (st + 128 * jc) * rs_k + h * D
                kt = tl.load(pk + kr[:, None] * rs_k + offs_d[None, :],
                             mask=((128 * jc + kr) < L)[:, None], other=0,
                             cache_modifier=CMK, eviction_policy=EVK)
            else:
                j = jc * NBLK + kb
                slot = 2 * j * nb - j * (j - 1) + (t0 - 2 * j)
                ok = ((t0 + offs_m // 64)[:, None] >= 2 * j[None, :]) & \
                    (j < nj)[None, :]
                if RM:
                    ok = ok & ((m0 + offs_m) < L)[:, None]
                a = tl.load(ws_ptr + base + slot[None, :] * 8192
                            + offs_m[:, None] * 128 + kp[None, :], mask=ok,
                            other=0, cache_modifier=CMW, eviction_policy=EVW)
                kn = 128 * j + kr
                kt = tl.load(k_ptr + (st + kn)[:, None] * rs_k + h * D
                             + offs_d[None, :], mask=(kn < L)[:, None],
                             other=0, cache_modifier=CMK, eviction_policy=EVK)
            a = a.to(tl.float8e4nv, bitcast=True)
            kt = kt.to(tl.float8e4nv, bitcast=True)
            if FP8_DOT:
                acc = tl.dot(a, kt, acc)
            else:
                acc = tl.dot(a.to(tl.bfloat16), kt.to(tl.bfloat16), acc)
        # Full-tile fast path (R15 iter53 dq8diet, amax hoisted by r14
        # iter50): a FULL 128-row tile (every row < L, L = this sequence's
        # length; ~94% of in-model tiles) takes the mask-free epilogue
        # `_dq8_epi_full`; a partial tile (and the STORE_BF16 diagnostic) runs
        # the shipped epilogue verbatim.
        if (m0 + BMQ <= L) & (STORE_BF16 == 0):
            _dq8_epi_full(acc, st, m0, b, h, u, e8_ptr, sc, dz8_ptr, rs8,
                          pfx_ptr, pcol_ptr, pamax_ptr, inv_ptr, H, D, BMQ,
                          LIMIT)
        else:
            rows = m0 + offs_m
            rmask = rows < L
            e8 = tl.load(e8_ptr + (st + rows) * H + h, mask=rmask, other=127)
            rs = (e8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            out = acc * (rs * sc)[:, None]
            ob = out.to(tl.bfloat16)
            if STORE_BF16:
                # Diagnostic dual store (gates only; +0.6-0.9 ms per call).
                tl.store(dq_ptr + (st + rows)[:, None] * rs_dq + h * D
                         + offs_d[None, :], ob, mask=rmask[:, None])
            # The value the shipped cast kernel loads, back in fp32, cast with
            # the same op order as _quantize_amax_colsum_kernel.
            x = tl.where(rmask[:, None], ob.to(tl.float32), 0.0)
            inv = tl.load(inv_ptr)
            y = tl.minimum(tl.maximum(x * inv, -LIMIT), LIMIT)
            tl.store(dz8_ptr + (st + rows)[:, None] * rs8 + h * D
                     + offs_d[None, :], y.to(dz8_ptr.dtype.element_ty),
                     mask=rmask[:, None])
            prow = tl.load(pfx_ptr + b).to(tl.int64) + u
            # Column sum by MFMA: ones[16, BMQ] @ x (x is bf16-exact,
            # products exact, fp32 accumulate in hardware order); row 0 is
            # stored.
            ones = tl.full((16, BMQ), 1.0, tl.bfloat16)
            xb = tl.where(rmask[:, None], ob, 0.0).to(tl.bfloat16)
            cs16 = tl.dot(ones, xb)
            r16 = tl.arange(0, 16)
            tl.store(pcol_ptr + prow * (H * D) + h * D + offs_d[None, :]
                     + 0 * r16[:, None], cs16, mask=(r16 == 0)[:, None])
            am = tl.max(tl.reshape(tl.abs(x), (BMQ * D,), can_reorder=True),
                        axis=0)
            tl.store(pamax_ptr + prow * H + h, am)


@triton.jit
def _dq8_epi_full(
    acc, st, m0, b, h, u, e8_ptr, sc, dz8_ptr, rs8, pfx_ptr, pcol_ptr,
    pamax_ptr, inv_ptr,
    H: tl.constexpr, D: tl.constexpr, BMQ: tl.constexpr,
    LIMIT: tl.constexpr,
):
    """`_dq8_unit`'s epilogue for a FULL tile (m0 + BMQ <= L, so the shipped
    rmask is all-true). Byte identity with the shipped (masked) epilogue:
      * where(True, v, 0) == v and an all-true mask is no mask, so x, xb, the
        e5m2 bytes and the colsum MFMA operands are the shipped ones bit for
        bit (same dot shape, dtypes and layouts -> the same MFMA chain);
      * the int64 row bases are folded into scalars (same byte addresses);
      * amax: fp32 -> bf16 (v_cvt_pk_bf16_f32, RNE) is odd-symmetric and
        monotone non-decreasing, so max_i |bf16(out_i)| == bf16(max_i
        |out_i|) under maxnum (NaNs dropped either way). This skips unpacking
        the bf16 tile back to fp32 and the |x| selects. Only an all-NaN
        tile's partial could differ (NaN payload); _dq8_fin_kernel's maxnum
        from a 0 init drops NaN partials, so the finalized amax is unaffected.
    The amax is reduced right after `out`, so `out` dies before the cast and
    the colsum. Reduced last (R15's first form), the kernel compiled to 128
    VGPRs with 11 spilled; in this order it compiles to 126 VGPRs, no spill,
    occupancy 4 (r14 iter50: Triton 3.6.0, gfx950, spec rows 87/88).
    """
    offs_m = tl.arange(0, BMQ)
    offs_d = tl.arange(0, D)
    e8 = tl.load(e8_ptr + ((st + m0) * H + h) + offs_m * H)
    rs = (e8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    # Triton 3.9 packed these products into a gfx950-invalid v_pk_mul_f32
    # op_sel form (nondeterministic signed zeros); the sink compiles the
    # kernel without packed-FP32 ops (_no_packed_fp32_launch_opts).
    out = acc * (rs * sc)[:, None]
    # amax first (see the docstring); rounded to bf16 at the store.
    amf = tl.max(tl.reshape(tl.abs(out), (BMQ * D,), can_reorder=True),
                 axis=0)
    ob = out.to(tl.bfloat16)
    # The value the shipped cast kernel loads, back in fp32, cast with the
    # same op order as _quantize_amax_colsum_kernel (no row mask: all valid).
    x = ob.to(tl.float32)
    inv = tl.load(inv_ptr)
    y = tl.minimum(tl.maximum(x * inv, -LIMIT), LIMIT)
    p8 = dz8_ptr + ((st + m0) * rs8 + h * D)
    tl.store(p8 + offs_m[:, None] * rs8 + offs_d[None, :],
             y.to(dz8_ptr.dtype.element_ty))
    prow = tl.load(pfx_ptr + b).to(tl.int64) + u
    # Column sum by MFMA: ones[16, BMQ] @ ob (the shipped xb, unmasked).
    ones = tl.full((16, BMQ), 1.0, tl.bfloat16)
    cs16 = tl.dot(ones, ob)
    r16 = tl.arange(0, 16)
    tl.store(pcol_ptr + prow * (H * D) + h * D + offs_d[None, :]
             + 0 * r16[:, None], cs16, mask=(r16 == 0)[:, None])
    am = amf.to(tl.bfloat16).to(tl.float32)
    tl.store(pamax_ptr + prow * H + h, am)


# Gin item (91) dq8 epilogue diet V3: `_dq8_sched_kernel` below calls `_dq8_unit`, which is
# `_dq8_unit_v3` (a FULL 128-row tile takes the mask-free epilogue `_dq8_epi_full`; bit-identical dz8 q
# block, pcol, q colsum and finalized amax; only the per-tile amax partial of an all-NaN tile may differ,
# in NaN payload, and `_dq8_fin_kernel`'s maxnum drops it) when DLRM_HSTU_FP8_DQ8_DIET == "1" at import,
# else the shipped `_dq8_unit_v0`.
# `_dq8_sched_kernel` keeps its name and parameter list either way, so its TSPEC rows compile it (with the
# selected unit) before RUN_START in BOTH settings (counted as src_changed).
_dq8_unit = _dq8_unit_v3 if _DQ8_DIET else _dq8_unit_v0


@triton.jit
def _dq8_sched_kernel(
    ws_ptr, hdr_ptr, k_ptr, e8_ptr, cu_ptr, dq_ptr, perm_ptr, sc, WS_HDR,
    rs_k, rs_dq, U_TILES, BH, dz8_ptr, rs8, pfx_ptr, pcol_ptr, pamax_ptr,
    inv_ptr,
    H: tl.constexpr, D: tl.constexpr, BMQ: tl.constexpr,
    FP8_DOT: tl.constexpr, STAGES: tl.constexpr, NBLK: tl.constexpr,
    LPT: tl.constexpr, MODE: tl.constexpr, RM: tl.constexpr = 0,
    CMW: tl.constexpr = "", EVW: tl.constexpr = "", CMK: tl.constexpr = "",
    EVK: tl.constexpr = "", SPTR: tl.constexpr = 0,
    WS_C64: tl.constexpr = False, STORE_BF16: tl.constexpr = 0,
    LIMIT: tl.constexpr = 57344.0,
):
    """`hstu_dq_split._dq_from_ds_sched_kernel` verbatim (MODE 0/1/2/3)."""
    pid = tl.program_id(0)
    x = pid % 8
    i = pid // 8
    u = U_TILES - 1 - i % U_TILES
    J = i // U_TILES
    if MODE == 3:
        NJ = tl.num_programs(0) // 8 // U_TILES
        lo = tl.load(perm_ptr + x) * H
        hi = tl.load(perm_ptr + x + 1) * H
        for s in range(lo + J, hi, NJ):
            _dq8_unit(ws_ptr, hdr_ptr, k_ptr, e8_ptr, cu_ptr, dq_ptr, sc,
                      WS_HDR, rs_k, rs_dq, s // H, s % H, u, dz8_ptr, rs8,
                      pfx_ptr, pcol_ptr, pamax_ptr, inv_ptr, H, D, BMQ,
                      FP8_DOT, STAGES, NBLK, RM, CMW, EVW, CMK, EVK, SPTR,
                      WS_C64, STORE_BF16, LIMIT)
    else:
        if MODE == 2:
            NJ = (BH + 7) // 8
            J = (J + x * (NJ // 8)) % NJ
        if MODE == 1:
            r = (J // H) * 8 + x
            h = J % H
            s = r * H + h
        else:
            s = J * 8 + x
            h = s % H
            r = s // H
        if s < BH:
            if LPT:
                b = tl.load(perm_ptr + r)
            else:
                b = r
            _dq8_unit(ws_ptr, hdr_ptr, k_ptr, e8_ptr, cu_ptr, dq_ptr, sc,
                      WS_HDR, rs_k, rs_dq, b, h, u, dz8_ptr, rs8, pfx_ptr,
                      pcol_ptr, pamax_ptr, inv_ptr, H, D, BMQ, FP8_DOT,
                      STAGES, NBLK, RM, CMW, EVW, CMK, EVK, SPTR, WS_C64,
                      STORE_BF16, LIMIT)


@triton.jit
def _dq8_pfx_kernel(so_ptr, pfx_ptr, B, BB: tl.constexpr):
    """pfx[0..B] = exclusive prefix of ceil(L_b / 128) (one program)."""
    i = tl.arange(0, BB)
    m = i < B
    a = tl.load(so_ptr + i, mask=m, other=0).to(tl.int64)
    b = tl.load(so_ptr + i + 1, mask=m, other=0).to(tl.int64)
    nb = tl.where(m, (b - a + 127) // 128, 0).to(tl.int32)
    c = tl.cumsum(nb, 0)
    tl.store(pfx_ptr + i, c - nb, mask=m)
    tl.store(pfx_ptr + 1 + i, c, mask=i == B - 1)


@triton.jit
def _dq8_fin_kernel(pcol_ptr, pamax_ptr, pfx_ptr, B, ocol_ptr, oam_ptr,
                    RPP: tl.constexpr, NC: tl.constexpr, AK: tl.constexpr,
                    BLK: tl.constexpr):
    """Stage 1 of the partial reduction: program p sums rows
    [p*RPP, (p+1)*RPP) of pcol [R, NC] (rows >= R = pfx[B] masked) and maxes
    pamax [R, AK]; fixed order -> deterministic."""
    pid = tl.program_id(0)
    R = tl.load(pfx_ptr + B).to(tl.int64)
    cols = tl.arange(0, NC)
    ak = tl.arange(0, AK)
    acc = tl.zeros((BLK, NC), tl.float32)
    am = tl.zeros((BLK, AK), tl.float32)
    r0 = pid.to(tl.int64) * RPP
    for r in range(0, RPP, BLK):
        rr = r0 + r + tl.arange(0, BLK)
        m = rr < R
        acc += tl.load(pcol_ptr + rr[:, None] * NC + cols[None, :],
                       mask=m[:, None], other=0.0)
        am = tl.maximum(am, tl.load(pamax_ptr + rr[:, None] * AK
                                    + ak[None, :], mask=m[:, None], other=0.0))
    tl.store(ocol_ptr + pid * NC + cols, tl.sum(acc, axis=0))
    tl.store(oam_ptr + pid, tl.max(tl.max(am, axis=1), axis=0))


@triton.jit
def _cast_skip_tile(
    SRC, DST, PARTIAL_AMAX, PARTIAL_COL, inv, pid_m, pid_n, amax_slot,
    n_rows, n_rows_out,
    N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ROW_ITERS: tl.constexpr, LIMIT: tl.constexpr,
):
    """`bwd_fp8._quantize_amax_colsum_v2_tile` (M1, bitwise to the shipped
    tile) with the amax partial written to `amax_slot` instead of
    pid_m * n_cb + pid_n (the q column blocks have no slot)."""
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    amax = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    row0 = pid_m.to(tl.int64) * (ROW_ITERS * BLOCK_M)
    rows = row0 + tl.arange(0, BLOCK_M).to(tl.int64)
    offs = rows[:, None] * N + cols[None, :]
    for _ in range(ROW_ITERS // 4):
        x0 = tl.load(SRC + offs, mask=rows[:, None] < n_rows, other=0.0)
        x1 = tl.load(SRC + offs + BLOCK_M * N,
                     mask=(rows + BLOCK_M)[:, None] < n_rows, other=0.0)
        x2 = tl.load(SRC + offs + 2 * BLOCK_M * N,
                     mask=(rows + 2 * BLOCK_M)[:, None] < n_rows, other=0.0)
        x3 = tl.load(SRC + offs + 3 * BLOCK_M * N,
                     mask=(rows + 3 * BLOCK_M)[:, None] < n_rows, other=0.0)
        xf = x0.to(tl.float32)
        acc += xf
        amax = tl.maximum(amax, tl.abs(xf))
        tl.store(DST + offs,
                 tl.minimum(tl.maximum(xf * inv, -LIMIT), LIMIT).to(
                     DST.dtype.element_ty),
                 mask=rows[:, None] < n_rows_out, cache_modifier=".cs")
        xf = x1.to(tl.float32)
        acc += xf
        amax = tl.maximum(amax, tl.abs(xf))
        tl.store(DST + offs + BLOCK_M * N,
                 tl.minimum(tl.maximum(xf * inv, -LIMIT), LIMIT).to(
                     DST.dtype.element_ty),
                 mask=(rows + BLOCK_M)[:, None] < n_rows_out,
                 cache_modifier=".cs")
        xf = x2.to(tl.float32)
        acc += xf
        amax = tl.maximum(amax, tl.abs(xf))
        tl.store(DST + offs + 2 * BLOCK_M * N,
                 tl.minimum(tl.maximum(xf * inv, -LIMIT), LIMIT).to(
                     DST.dtype.element_ty),
                 mask=(rows + 2 * BLOCK_M)[:, None] < n_rows_out,
                 cache_modifier=".cs")
        xf = x3.to(tl.float32)
        acc += xf
        amax = tl.maximum(amax, tl.abs(xf))
        tl.store(DST + offs + 3 * BLOCK_M * N,
                 tl.minimum(tl.maximum(xf * inv, -LIMIT), LIMIT).to(
                     DST.dtype.element_ty),
                 mask=(rows + 3 * BLOCK_M)[:, None] < n_rows_out,
                 cache_modifier=".cs")
        rows += 4 * BLOCK_M
        offs += 4 * BLOCK_M * N
    if row0 < n_rows:
        tl.store(PARTIAL_COL + pid_m.to(tl.int64) * N + cols,
                 tl.sum(acc, axis=0))
        tl.store(PARTIAL_AMAX + amax_slot,
                 tl.max(tl.max(amax, axis=1), axis=0))


@triton.jit
def _quantize_amax_colsum_skip_kernel(
    SRC, DST, PARTIAL_AMAX, PARTIAL_COL, INV_SCALE, n_rows, n_rows_out,
    n_tiles_m,
    N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ROW_ITERS: tl.constexpr, LIMIT: tl.constexpr, XCD_GROUP: tl.constexpr,
    SKIP_LO: tl.constexpr, SKIP_N: tl.constexpr,
):
    """The shipped cast over every column block EXCEPT [SKIP_LO,
    SKIP_LO + SKIP_N) (the q block, written by the dq producer).

    1-D XCD-grouped launch as `_quantize_amax_colsum_v2_kernel`; logical
    column block c maps to physical block c (c < SKIP_LO) or c + SKIP_N.
    Payload, partial column sums (PARTIAL_COL [grid_m, N], physical columns)
    and the amax partial of every skipped-in tile are those of the shipped
    kernel; PARTIAL_AMAX is [grid_m, n_cb - SKIP_N].
    """
    tl.static_assert(ROW_ITERS % 4 == 0)
    k = tl.program_id(0)
    xcd = k % 8
    r = k // 8
    t = ((r // XCD_GROUP) * 8 + xcd) * XCD_GROUP + (r % XCD_GROUP)
    n_cb = N // BLOCK_N - SKIP_N
    if t < n_tiles_m * n_cb:
        inv = tl.load(INV_SCALE)
        pid_m = t // n_cb
        c = t % n_cb
        pid_n = tl.where(c >= SKIP_LO, c + SKIP_N, c)
        _cast_skip_tile(SRC, DST, PARTIAL_AMAX, PARTIAL_COL, inv, pid_m,
                        pid_n, pid_m * n_cb + c, n_rows, n_rows_out, N,
                        BLOCK_M, BLOCK_N, ROW_ITERS, LIMIT)


@triton.jit
def _quantize_amax_colsum_skipn_kernel(
    SRC, DST, PARTIAL_AMAX, PARTIAL_COL, INV_SCALE, n_rows, n_rows_out,
    n_tiles_m,
    N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ROW_ITERS: tl.constexpr, LIMIT: tl.constexpr, XCD_GROUP: tl.constexpr,
    SKIP_MASK: tl.constexpr, SKIP_N: tl.constexpr,
):
    """B1a (r16): `_quantize_amax_colsum_skip_kernel` with a SET of skipped
    physical column blocks (bit b of SKIP_MASK; SKIP_N bits set), e.g. u and
    q when both producers wrote their e5m2 blocks. Logical block c maps to
    the c-th unskipped physical block; the tile code, the [grid_m, N]
    PARTIAL_COL layout (physical columns, skipped ones unwritten) and the
    XCD-grouped launch are the single-range kernel's, so payload, colsum
    partials and amax partials of every cast tile are bitwise the shipped
    cast's. PARTIAL_AMAX is [grid_m, n_cb - SKIP_N].

    MERGE NOTE (r14 B4, probes/iter35b/followups/followups.patch): B4 adds
    AMAX_ACC / AMAX_ATOMIC to `_cast_skip_tile`; this positional call must
    take them too, or the first B1a consumer cast fails to compile (lossy
    decline + latch). The B4-merged form is r16
    probes/iter28/b1a/b4merge/b1a_on_b4.patch."""
    tl.static_assert(ROW_ITERS % 4 == 0)
    k = tl.program_id(0)
    xcd = k % 8
    r = k // 8
    t = ((r // XCD_GROUP) * 8 + xcd) * XCD_GROUP + (r % XCD_GROUP)
    n_cb = N // BLOCK_N - SKIP_N
    if t < n_tiles_m * n_cb:
        inv = tl.load(INV_SCALE)
        pid_m = t // n_cb
        c = t % n_cb
        pid_n = c
        for s in tl.static_range(N // BLOCK_N):
            if (SKIP_MASK >> s) & 1:
                pid_n = tl.where(pid_n >= s, pid_n + 1, pid_n)
        _cast_skip_tile(SRC, DST, PARTIAL_AMAX, PARTIAL_COL, inv, pid_m,
                        pid_n, pid_m * n_cb + c, n_rows, n_rows_out, N,
                        BLOCK_M, BLOCK_N, ROW_ITERS, LIMIT)


def _quantize_amax_colsum_skipn(
    value: torch.Tensor,
    inv_scale: torch.Tensor,
    out: torch.Tensor,
    skips,  # pyre-ignore[2]: sorted [(lo, hi), ...], >= 2 ranges
) -> Tuple[torch.Tensor, torch.Tensor]:
    """B1a (r16): `quantize_amax_colsum_skip` over several skipped ranges."""
    from generative_recommenders.ops.triton import bwd_fp8

    assert value.dim() == 2 and value.is_contiguous()
    m, n = value.shape
    bn = bwd_fp8._COLSUM_BLOCK_N
    assert n % bn == 0
    mask = 0
    for lo, hi in skips:
        assert lo % bn == 0 and hi % bn == 0 and 0 <= lo < hi <= n
        for blk in range(lo // bn, hi // bn):
            assert not (mask >> blk) & 1, f"overlapping skip ranges {skips}"
            mask |= 1 << blk
    skip_n = bin(mask).count("1")
    n_cb = n // bn - skip_n
    assert n_cb >= 1
    assert out.dim() == 2 and out.is_contiguous() and out.shape[1] == n
    assert out.shape[0] >= m and out.device == value.device
    m_out = out.shape[0]
    rows_per_prog = bwd_fp8._COLSUM_BLOCK_M * bwd_fp8._COLSUM_ROW_ITERS
    grid_m = max(1, (m + rows_per_prog - 1) // rows_per_prog)
    grid_m_launch = max(grid_m, (m_out + rows_per_prog - 1) // rows_per_prog)
    partial_amax = torch.empty(grid_m * n_cb, dtype=torch.float32,
                               device=value.device)
    partial_col = torch.empty((grid_m, n), dtype=torch.float32,
                              device=value.device)
    if m == 0:
        partial_amax.zero_()
        partial_col.zero_()
    group = 8 * bwd_fp8._COLSUM_V2_XCD_GROUP
    n_programs = (grid_m_launch * n_cb + group - 1) // group * group
    _quantize_amax_colsum_skipn_kernel[(n_programs,)](
        value,
        out,
        partial_amax,
        partial_col,
        inv_scale,
        m,
        m_out,
        grid_m_launch,
        N=n,
        BLOCK_M=bwd_fp8._COLSUM_BLOCK_M,
        BLOCK_N=bn,
        ROW_ITERS=bwd_fp8._COLSUM_ROW_ITERS,
        LIMIT=bwd_fp8._GRAD_MAX,
        XCD_GROUP=bwd_fp8._COLSUM_V2_XCD_GROUP,
        SKIP_MASK=mask,
        SKIP_N=skip_n,
        num_warps=8,
    )
    return partial_amax, partial_col


def _dq8_partials(seq_offsets: torch.Tensor, n_rows: int, H: int, D: int):
    """(pfx, pcol, pamax): row prefix (one tiny kernel, no host sync) and
    uninitialized partials (the finalize masks rows >= R = pfx[B]). Host
    bound: sum_b ceil(L_b / 128) <= (T + 127 B) / 128 (as build_workspace)."""
    dev = seq_offsets.device
    B = seq_offsets.numel() - 1
    pfx = torch.empty(B + 1, dtype=torch.int32, device=dev)
    _dq8_pfx_kernel[(1,)](seq_offsets, pfx, B, BB=triton.next_power_of_2(B),
                          num_warps=4)
    r_bound = (int(n_rows) + 127 * B) // 128 + 1
    pcol = torch.empty((r_bound, H * D), dtype=torch.float32, device=dev)
    pamax = torch.empty((r_bound * H,), dtype=torch.float32, device=dev)
    return pfx, pcol, pamax


def _dq8_finalize(pamax: torch.Tensor, pcol: torch.Tensor, pfx: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(amax, colsum [H*D]) of the dq partials: deterministic 2-stage tree."""
    r_bound, nc = pcol.shape
    ak = pamax.numel() // r_bound
    B = pfx.numel() - 1
    rpp = triton.cdiv(r_bound, _FIN_PROGS)
    blk = 8 if ak <= 128 else (4 if ak <= 512 else 1)
    rpp = triton.cdiv(rpp, blk) * blk
    ocol = torch.empty((_FIN_PROGS, nc), dtype=torch.float32, device=pcol.device)
    oam = torch.empty((_FIN_PROGS,), dtype=torch.float32, device=pcol.device)
    _dq8_fin_kernel[(_FIN_PROGS,)](pcol, pamax, pfx, B, ocol, oam, RPP=rpp,
                                   NC=nc, AK=ak, BLK=blk, num_warps=4)
    return oam.amax(), ocol.sum(dim=0)


def quantize_amax_colsum_skip(
    value: torch.Tensor,
    inv_scale: torch.Tensor,
    out: torch.Tensor,
    # B1a (r16): one (lo, hi) range or a sequence of them.
    skip: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """`bwd_fp8.quantize_amax_colsum(value, inv_scale, out=out)` over every
    column except [skip[0], skip[1]) -> (partial_amax, partial_col).

    partial_col is [grid_m, N] (physical columns; the skipped columns are left
    unwritten), so `partial_col.sum(dim=0)` is bitwise the shipped colsum in
    every cast column; partial_amax holds the cast tiles only.

    B1a (r16): `skip` may also be a sequence of (lo, hi) ranges (u and q);
    several ranges launch `_quantize_amax_colsum_skipn_kernel`, a single
    range this function's original single-range kernel.
    """
    from generative_recommenders.ops.triton import bwd_fp8

    if not isinstance(skip[0], int):
        skips = sorted((int(a), int(b)) for a, b in skip)
        if len(skips) > 1:
            return _quantize_amax_colsum_skipn(value, inv_scale, out, skips)
        skip = skips[0]
    assert value.dim() == 2 and value.is_contiguous()
    m, n = value.shape
    bn = bwd_fp8._COLSUM_BLOCK_N
    lo, hi = skip
    assert n % bn == 0 and lo % bn == 0 and hi % bn == 0 and 0 <= lo < hi <= n
    assert out.dim() == 2 and out.is_contiguous() and out.shape[1] == n
    assert out.shape[0] >= m and out.device == value.device
    m_out = out.shape[0]
    rows_per_prog = bwd_fp8._COLSUM_BLOCK_M * bwd_fp8._COLSUM_ROW_ITERS
    grid_m = max(1, (m + rows_per_prog - 1) // rows_per_prog)
    grid_m_launch = max(grid_m, (m_out + rows_per_prog - 1) // rows_per_prog)
    skip_n = (hi - lo) // bn
    n_cb = n // bn - skip_n
    partial_amax = torch.empty(grid_m * n_cb, dtype=torch.float32,
                               device=value.device)
    partial_col = torch.empty((grid_m, n), dtype=torch.float32,
                              device=value.device)
    if m == 0:
        partial_amax.zero_()
        partial_col.zero_()
    group = 8 * bwd_fp8._COLSUM_V2_XCD_GROUP
    n_programs = (grid_m_launch * n_cb + group - 1) // group * group
    _quantize_amax_colsum_skip_kernel[(n_programs,)](
        value,
        out,
        partial_amax,
        partial_col,
        inv_scale,
        m,
        m_out,
        grid_m_launch,
        N=n,
        BLOCK_M=bwd_fp8._COLSUM_BLOCK_M,
        BLOCK_N=bn,
        ROW_ITERS=bwd_fp8._COLSUM_ROW_ITERS,
        LIMIT=bwd_fp8._GRAD_MAX,
        XCD_GROUP=bwd_fp8._COLSUM_V2_XCD_GROUP,
        SKIP_LO=lo // bn,
        SKIP_N=skip_n,
        num_warps=8,
    )
    return partial_amax, partial_col


class DqE5m2Sink:
    """The `dq_fp8_sink` handed to the FlyDSL attention backward.

    Callable with `hstu_dq_split.dq_from_ds`'s signature; the dq-split call
    site calls it instead of `dq_from_ds`. It runs the e5m2 fork into the q
    columns of `dz8` (allocated on first use) and keeps the partials; the BF16
    dq is NOT written. Configurations the fork does not cover run the stock
    `dq_from_ds` (BF16 dq written) and leave `filled` False, as does a call
    that never reaches the dq-split (Triton fallback, calibration step, shape
    decline): the consumer then takes the shipped BF16 cast.
    """

    def __init__(self, rows: int, width: int, q_cols: Tuple[int, int],
                 inv_scale: torch.Tensor, cast_scale: torch.Tensor, pad: int,
                 grad_dtype: torch.dtype, limit: float,
                 device: torch.device,
                 dz8: Optional[torch.Tensor] = None,
                 pad_zeroed: bool = False) -> None:
        self.rows = int(rows)
        self.width = int(width)
        self.q_cols = (int(q_cols[0]), int(q_cols[1]))
        self.inv_scale = inv_scale
        self.cast_scale = cast_scale
        self.rows_out = (self.rows + pad - 1) // pad * pad
        self.grad_dtype = grad_dtype
        self.limit = float(limit)
        self.device = device
        # Shared dz8 (one contract for both producers that allocate it before
        # the attention backward): an optional buffer adopted here instead of
        # the lazy allocation below. B1a (r16): the LNMD backward's emission
        # allocated it (u block and its pad rows written); pad_zeroed=False,
        # so the q pad rows are zeroed below as for a fresh buffer. R15 M-E1
        # (_dkdv_e5m2_arm): it allocated or adopted the buffer and zeroed the
        # pad rows of every column once; pad_zeroed=True, so this sink leaves
        # them alone. dz8=None: the behaviour is unchanged.
        self.pad_zeroed = bool(pad_zeroed)
        assert dz8 is not None or not self.pad_zeroed
        if dz8 is not None:
            # Same device: an index-less `device` ("cuda", as r14's B1a test
            # passes) matches any index of its type.
            want = torch.device(device)
            assert (dz8.dim() == 2 and dz8.is_contiguous()
                    and tuple(dz8.shape) == (self.rows_out, self.width)
                    and dz8.dtype == grad_dtype
                    and dz8.device.type == want.type
                    and want.index in (None, dz8.device.index)), (
                tuple(dz8.shape), dz8.dtype, self.rows_out, self.width,
                str(dz8.device), str(want))
        self.dz8: Optional[torch.Tensor] = dz8
        self.partials = None
        self.filled = False
        self.declined: Optional[str] = None
        self.config: dict = {}

    def __call__(self, ws, k, e8, seq_offsets, dq, N, sc, rs_k, rs_dq,
                 fp8_dot=True, bmq=128, num_warps=8, stages=2, nblk=1,
                 order="bh", sched="", rm=0, cmw="", evw="", cmk="", evk="",
                 sptr=0, ws_cm="", ws_ev="", ws_c64=False, rsk=0,
                 **launch_kw):
        # rsk (r15's DQ_SPLIT_OPT kb token, ship11 gin item (27)): a constexpr
        # K row stride that only turns the stock kernel's K address arithmetic
        # into buffer loads (same tiles, same dots, bitwise dq). The fork keeps
        # the runtime-stride K addressing it was gated with, so an accepted
        # call ignores it (precedence: this sink, item (25)(d), takes the
        # dq-split calls it accepts); a declined call forwards it to r15's
        # dq_from_ds unchanged.
        B = seq_offsets.numel() - 1
        H, D = k.shape[1], k.shape[2]
        cmw, evw = cmw or ws_cm, evw or ws_ev
        reason = None
        if self.filled:
            reason = "sink already filled"
        elif sched not in _SCHEDS:
            reason = f"dq-split sched {sched!r} (fork covers {_SCHEDS})"
        elif int(bmq) != 128:
            reason = f"bmq={bmq} (partials assume 128-row blocks)"
        elif sptr and int(nblk) != 1:
            reason = f"sptr with nblk={nblk}"
        elif H * D != self.q_cols[1] - self.q_cols[0]:
            reason = f"H*D={H * D} != q block width"
        elif dq.dtype is not torch.bfloat16:
            reason = f"dq dtype {dq.dtype}"
        elif B <= 0 or int(N) <= 0:
            reason = "empty batch"
        elif set(launch_kw) - _LAUNCH_OPTS:
            # A dq_from_ds argument this fork does not know (r15 changed the
            # dq-split contract): run r15's kernel rather than guess.
            reason = f"unknown dq_from_ds args {sorted(set(launch_kw) - _LAUNCH_OPTS)}"
        if reason is not None:
            self.declined = reason
            return _dqs.dq_from_ds(
                ws, k, e8, seq_offsets, dq, N, sc, rs_k, rs_dq,
                fp8_dot=fp8_dot, bmq=bmq, num_warps=num_warps, stages=stages,
                nblk=nblk, order=order, sched=sched, rm=rm, cmw=cmw, evw=evw,
                cmk=cmk, evk=evk, sptr=sptr, ws_c64=ws_c64, rsk=rsk,
                **launch_kw)
        if self.dz8 is None:
            self.dz8 = torch.empty((self.rows_out, self.width),
                                   dtype=self.grad_dtype, device=self.device)
        q8 = self.dz8[:, self.q_cols[0]:self.q_cols[1]]
        if self.rows_out > self.rows and not self.pad_zeroed:
            # Pad rows of the q block: zero, as the shipped cast writes them
            # (an R15 M-E1 owner zeroes the pad rows of every column once).
            q8[self.rows:].zero_()
        pfx, pcol, pamax = _dq8_partials(seq_offsets, self.rows, H, D)
        U = triton.cdiv(int(N), 128)
        hdr = ws[: 4 * (B + 1)].view(torch.int32)
        BH = B * H
        if sched == "bc":
            perm = torch.empty(16, dtype=torch.int32, device=ws.device)
            _dqs._dq_bc_bounds_kernel[(1,)](hdr, perm, B,
                                            int(B).bit_length() + 1,
                                            num_warps=1)
        elif sched == "lpt":
            lens = seq_offsets[1:] - seq_offsets[:-1]
            perm = torch.argsort(lens, descending=True, stable=True).to(
                torch.int32)
        else:
            perm = hdr
        units = (triton.cdiv(B, 8) * H if sched == "ilb"
                 else triton.cdiv(BH, 8))
        launch_kw = dict(launch_kw)
        launch_kw["waves_per_eu"] = _WAVES_PER_EU
        # No packed-FP32 VALU ops on Triton 3.9 (see _no_packed_fp32_launch_opts).
        fn_opts = _no_packed_fp32_launch_opts()
        launch_kw.update(fn_opts)
        global _DQ8_DIET_LOGGED
        if not _DQ8_DIET_LOGGED:  # once per process, at the first dq8 launch
            _DQ8_DIET_LOGGED = True
            print(
                "[dq8-diet] epilogue diet V3 "
                + ("ARMED: _dq8_sched_kernel runs _dq8_unit_v3" if _DQ8_DIET
                   else "OFF: _dq8_sched_kernel runs the shipped _dq8_unit_v0")
                + f" (DLRM_HSTU_FP8_DQ8_DIET={_DQ8_DIET_ENV!r} at import)"
                + f" fn_opts={fn_opts or 'none'}"
                + f" rank={os.environ.get('RANK', '?')}",
                flush=True,
            )
        _dq8_sched_kernel[(units * 8 * U,)](
            ws, hdr, k.view(torch.uint8), e8, seq_offsets, dq, perm,
            float(sc), _dqs.ws_header_bytes(B), int(rs_k), int(rs_dq), U, BH,
            q8, int(q8.stride(0)), pfx, pcol, pamax, self.inv_scale,
            H=H, D=D, BMQ=128, FP8_DOT=bool(fp8_dot), STAGES=int(stages),
            NBLK=int(nblk), LPT=int(sched == "lpt"),
            MODE={"il": 0, "lpt": 0, "ilb": 1, "ilr": 2, "bc": 3}[sched],
            RM=int(rm), CMW=cmw, EVW=evw, CMK=cmk, EVK=evk,
            SPTR=int(bool(sptr)), WS_C64=bool(ws_c64), STORE_BF16=0,
            LIMIT=self.limit, num_warps=int(num_warps), **launch_kw)
        self.partials = (pamax, pcol, pfx)
        self.filled = True
        self.config = dict(sched=sched, rm=int(rm), sptr=int(bool(sptr)),
                           cmw=cmw, stages=int(stages), num_warps=int(num_warps),
                           ws_c64=bool(ws_c64))
        return None

    def finish(self, dz: torch.Tensor, du_emit=None,  # pyre-ignore[2]
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Cast the u|v|k columns of the BF16 `dz` [M, width] into dz8 and
        combine with the dq partials -> `prequant` (dz8, amax, colsum, M).

        B1a (r16): with `du_emit` (hstu_du_e5m2.DuE5m2Emit sharing this dz8)
        the u block is producer-cast too: the cast covers v|k only, amax =
        max(v|k tiles, q, u) and the u colsum is the producer's.

        MERGE NOTE (r14 B4 fold, probes/iter35b/followups/followups.patch):
        any early-return merge branch here must handle `du_emit` as well
        (skip [u, q], fold u_amax into the amax, write u_colsum into the u
        slice); a branch that casts with skip=q_cols re-casts the u block from
        never-written bf16 columns. Tested merged form: r16
        probes/iter28/b1a/b4merge/b1a_on_b4.patch."""
        assert self.filled and self.dz8 is not None and self.partials is not None
        assert dz.shape == (self.rows, self.width) and dz.is_contiguous()
        skip = self.q_cols
        if du_emit is not None:
            assert du_emit.dz8 is self.dz8 and du_emit.rows == self.rows
            skip = [du_emit.u_cols, self.q_cols]
        part_amax, part_col = quantize_amax_colsum_skip(
            dz, self.inv_scale, self.dz8, skip)
        pamax, pcol, pfx = self.partials
        self.partials = None
        q_amax, q_col = _dq8_finalize(pamax, pcol, pfx)
        amax = torch.maximum(part_amax.amax(), q_amax)
        if du_emit is not None:
            amax = torch.maximum(amax, du_emit.u_amax)
        colsum = part_col.sum(dim=0)
        colsum[self.q_cols[0]:self.q_cols[1]] = q_col
        if du_emit is not None:
            colsum[du_emit.u_cols[0]:du_emit.u_cols[1]] = du_emit.u_colsum
        return self.dz8, amax, colsum, self.rows

    def finish_q(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(q amax, q colsum [H*D]) of the dq partials only (R15 M-E1: the
        layer's shared-dz8 owner casts the other columns itself and combines
        every producer's partials; same 2-stage reduction as finish())."""
        assert self.filled and self.dz8 is not None and self.partials is not None
        pamax, pcol, pfx = self.partials
        self.partials = None
        return _dq8_finalize(pamax, pcol, pfx)

    def restore_bf16(self, dz: torch.Tensor) -> None:
        """Decline/failure fallback (first-call failures only): rebuild the
        unwritten BF16 q columns of `dz` from the e5m2 payload (LOSSY), so the
        shipped BF16 projection backward can run."""
        assert self.filled and self.dz8 is not None
        lo, hi = self.q_cols
        dz[:, lo:hi].copy_(
            (self.dz8[: self.rows, lo:hi].float() * self.cast_scale).to(dz.dtype))
