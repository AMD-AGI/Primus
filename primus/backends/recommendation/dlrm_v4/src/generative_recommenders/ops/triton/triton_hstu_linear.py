# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

#!/usr/bin/env python3


import functools
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.autograd.profiler import record_function

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from generative_recommenders.common import (
    switch_to_contiguous_if_needed,
    triton_autotune,
)
from generative_recommenders.ops.eval_fwd_hygiene import (
    backward_possible,
    eval_fwd_hygiene_enabled,
)
from generative_recommenders.ops.hip import l11_outproj_lnmd
from generative_recommenders.ops.hip import l11_do8
from generative_recommenders.ops.triton import bwd_fp8
from generative_recommenders.ops.triton import fp8_tm_alloc
from generative_recommenders.ops.triton import fp8_wgrad_snr
from generative_recommenders.ops.triton import hstu_du_e5m2
from generative_recommenders.ops.triton import wgrad_ck, wgrad_ck_w3
from generative_recommenders.ops.triton.triton_addmm import maybe_triton_addmm_fwd
from generative_recommenders.ops.triton.triton_fp8 import (
    activation_scale_value,
    calibrate_scale_buffer,
    fp8_scale_tensor,
    fp8_supported,
    hstu_fp8_fail_closed,
    hstu_fp8_fuse_outln_enabled,
    record_fp8_capability,
    record_fp8_contract,
    scaled_addmm_fp8_preq,
)
from generative_recommenders.ops.utils import maybe_register_custom_op


def _get_layer_norm_mul_dropout_fwd_multirow_configs() -> List[triton.Config]:
    """Generate autotune configs for multi-row LayerNorm multiplication with dropout kernels."""
    configs = []
    for BLOCK_N in [1, 2, 4, 8, 16]:
        for num_warps in [1, 2, 4]:
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs


from generative_recommenders.ops.utils import (
    arm_print_all_ranks,
    use_separated_rng_ln_mul_dropout,
)
from generative_recommenders.ops.eval_lnmd_v3 import (
    eval_lnmd_v3_enabled,
    log_eval_lnmd_v3_armed,
    log_eval_lnmd_v3_ignored,
)

# FP8 output-projection wgrad. Default OFF (DLRM_HSTU_FP8_BWD_OUT_WGRAD absent or
# not "1" = the shipped bf16 wgrad, same kernels and saved tensors). When on, a
# declared precision change on d_output_weight: e4m3 y x FP8 dout (e4m3 copy at
# an adaptive power-of-two scale by default, see bwd_fp8.py), fp32 accumulate.
# The FP8 output-LN forward keeps the e4m3 y it already produced for the forward
# GEMM (store instead of recompute), the backward skips the bf16 y recompute in
# the LN-mul-dropout backward, and the wgrad reuses the dout cast of the FP8
# output-projection dgrad (one cast serves both). Engages only with the FP8
# backward scale controller (DLRM_HSTU_FP8_BWD=1) published; warmup calls take
# the bf16 path. Armed proof: one "[fp8-out-wgrad] armed impl=... path=..."
# line per process on the first FP8 wgrad call.
_FP8_OUT_WGRAD: bool = os.environ.get("DLRM_HSTU_FP8_BWD_OUT_WGRAD", "0") == "1"
# Sub-mode (DLRM_HSTU_FP8_BWD_OUT_WGRAD_IMPL), read only when the knob is on:
# "nn" (default): the default FP8 forward keeps the row-major e4m3 y, the dout
#   cast also writes a token-major copy, and the wgrad is one hipBLASLt FP8 NN
#   GEMM called through the C API (ops/hipblaslt_fp8_nn.py) that reads y as
#   stored, so neither a y transpose nor a y^T producer store is needed. If the
#   library offers no NN algorithm it falls back to a y transpose + FP8 TN GEMM
#   (path=nn_fallback_tn on the armed line).
# "producer": a dedicated LN-mul-dropout FP8 forward and the dout cast write
#   their FP8 bytes token-major, so the wgrad is one FP8 TN GEMM.
# "tn": Triton byte transposes of both operands + FP8 TN GEMM.
# "triton": row-major split-K Triton kernel (ops/triton/fp8_wgrad.py).
_FP8_OUT_WGRAD_IMPL: str = os.environ.get("DLRM_HSTU_FP8_BWD_OUT_WGRAD_IMPL", "nn")
_fp8_out_wgrad_logged: bool = False
# Arbor r14 iteration 38 (component B), DEFAULT OFF:
#   DLRM_HSTU_FP8_BWD_WGRAD_CK_W3=1  the NN-mode wgrad above (fp8_wgrad_nn) runs the
#                                    custom K=tokens kernel wgrad_tm_ck_w3.cu
#                                    (wgrad_ck_w3.py) instead of hipBLASLt; the
#                                    first call per device is checked against the
#                                    hipBLASLt result and any failure latches back
#                                    to hipBLASLt (armed line: path=nn_ck).
#                                    d_output_weight fp32-order (DECLARED),
#                                    deterministic. Independent of
#                                    DLRM_HSTU_FP8_BWD_WGRAD_CK (uvqk / vk wgrad).
# Iteration 39 (B13), DEFAULT OFF, needs _CK_W3=1:
#   DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED=1  the dual cast writes the token-major
#                                    dout copy step-tiled [ceil(M/128), 512, 128]
#                                    and the custom kernel reads that layout
#                                    (wgrad_ck_w3_t; same output bits). Decided
#                                    before the cast (_out_wgrad_dout_tiled); any
#                                    other reader re-lays it out first
#                                    (wgrad_ck_w3.untile_to_row_major, bitwise).
# wgrad_ck_w3 prints its own "requested: wgrad_ck_w3=1" line at import.
if wgrad_ck_w3.requested() and not (_FP8_OUT_WGRAD and _FP8_OUT_WGRAD_IMPL == "nn"):
    wgrad_ck_w3.log(
        "wgrad_ck_w3 INERT without DLRM_HSTU_FP8_BWD_OUT_WGRAD=1 IMPL=nn"
    )


def prebuild_wgrad_ck_w3() -> bool:
    """DLRM_HSTU_FP8_BWD_WGRAD_CK_W3: build + load the custom out-proj wgrad before
    RUN_START (triton_fp8.prebuild_extensions, gin DLRM_HSTU_FP8_PREBUILD). No-op
    (False) unless the knob is on and the FP8 out-proj wgrad runs NN mode; a
    failure is latched and logged inside (hipBLASLt then runs)."""
    if not (wgrad_ck_w3.requested() and _FP8_OUT_WGRAD and _FP8_OUT_WGRAD_IMPL == "nn"):
        return False
    import time

    t0 = time.perf_counter()
    if not wgrad_ck_w3.prebuild():
        return False
    wgrad_ck_w3.log(f"prebuild: wgrad_ck_w3 built ({time.perf_counter() - t0:.1f}s)")
    return True


def _dout_row_major(
    dout_fp8: torch.Tensor, m: int, required: bool
) -> Optional[torch.Tensor]:
    """B13: the row-major dout_t a hipBLASLt reader needs; a 2-D dout_fp8 as is. The
    re-laid-out copy of a step-tiled one comes from the B11 pool when it is active
    (fp8_tm_alloc.empty_copy: normally its prewarmed spare block, no new device
    memory). Out of memory (logged once): None for the first-call guard's reference
    (required=False; wgrad_ck_w3.run_w3 defers the guard, nothing latched), raised
    for a hipBLASLt fallback (required=True: no allocation-free fallback exists at
    this site for a step-tiled dout_t; reached only after a W3 decline or latch)."""
    if dout_fp8.dim() != 3:
        return dout_fp8
    try:
        return wgrad_ck_w3.untile_to_row_major(dout_fp8, m, empty=fp8_tm_alloc.empty_copy)
    except Exception as error:
        if not fp8_tm_alloc.is_oom(error):
            raise
        wgrad_ck_w3.log_once(
            "untile_oom",
            f"out_projection: re-laid-out copy of the step-tiled dout_t "
            f"{tuple(dout_fp8.shape)} out of memory on {dout_fp8.device} "
            f"({type(error).__name__}) -> "
            + (
                "no allocation-free fallback for this call, raised"
                if required
                else "guard deferred for this call (nothing latched)"
            ),
        )
        if required:
            raise
        return None


def _out_wgrad_nn_ck_w3(
    y_fp8: torch.Tensor,
    dout_fp8: torch.Tensor,
    y_scale: torch.Tensor,
    dout_scale: torch.Tensor,
    fp8_wgrad_nn,  # pyre-ignore[2]: ops/hipblaslt_fp8_nn.fp8_wgrad_nn
) -> Optional[Tuple[Optional[torch.Tensor], str]]:
    """DLRM_HSTU_FP8_BWD_WGRAD_CK_W3: (dw_t, "nn_ck") from the custom kernel; or
    (the hipBLASLt dw_t its first-call guard computed, possibly None, "nn") when
    the guard rejected / the launch failed after it; or None: the caller runs
    fp8_wgrad_nn exactly as without the knob (declined, latched, or failed before
    the reference ran). The hipBLASLt reference runs at most once per call.
    A step-tiled dout_fp8 (B13, 3-D) is re-laid out for the reference
    (`_dout_row_major`; out of memory -> wgrad_ck.GuardDeferred, see run_w3); after a
    None the caller re-lays it out too."""
    hip: List[Optional[torch.Tensor]] = []

    def reference() -> torch.Tensor:
        if not hip:
            operand = _dout_row_major(dout_fp8, int(y_fp8.shape[0]), required=False)
            if operand is None:
                raise wgrad_ck.GuardDeferred("re-laid-out dout_t copy out of memory")
            hip.append(
                fp8_wgrad_nn(
                    y_fp8,
                    operand,
                    y_scale,
                    dout_scale,
                    out_dtype=torch.float32,
                )
            )
        ref = hip[0]
        if ref is None:
            raise RuntimeError("hipBLASLt NN reference unavailable (no algorithm)")
        return ref

    dw_t = wgrad_ck_w3.run_w3(
        y_fp8, dout_fp8, y_scale, dout_scale, int(y_fp8.shape[0]), reference
    )
    if dw_t is not None:
        return dw_t, "nn_ck"
    if hip:
        return hip[0], "nn"
    return None


def _out_wgrad_dout_tiled(ctx, dout: torch.Tensor, y_fp8: torch.Tensor) -> bool:  # pyre-ignore[2]
    """B13 (DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED=1 with _CK_W3=1, default off): the
    NN-mode dual cast writes dout_t step-tiled for the custom W3 kernel. Decided
    BEFORE the cast from the W3 latch state (wgrad_ck_w3.tiled_active), so the
    steady state never re-lays the operand out; producer mode (TN) never tiles."""
    return (
        wgrad_ck_w3.tiled_requested()
        and getattr(ctx, "fp8_dout_token_major", False)
        and not getattr(ctx, "fp8_y_token_major", False)
        and wgrad_ck_w3.tiled_active(dout.device, y_fp8, int(dout.shape[-1]))
    )


# @manual=//triton:triton
from triton.language.extra import libdevice

try:
    # @manual=//triton:triton
    from triton.language.extra.libdevice import fast_dividef
except ImportError:
    try:
        # @manual=//triton:triton
        from triton.language.extra.cuda.libdevice import fast_dividef
    except ImportError:
        # pyre-ignore: Undefined import [21]
        # @manual=//triton:triton
        from triton.language.math import fast_dividef


COMPUTE_OUTPUT_LN_FAST_DROPOUT = False


def set_compute_output_ln_fast_dropout(value: bool) -> None:
    global COMPUTE_OUTPUT_LN_FAST_DROPOUT
    COMPUTE_OUTPUT_LN_FAST_DROPOUT = value


FUSE_OUTPUT_LN_RNG_BLACKWELL = False


# Only impact B200 training when CONCAT_UX is False
def set_fuse_output_ln_rng_blackwell(value: bool) -> None:
    global FUSE_OUTPUT_LN_RNG_BLACKWELL
    FUSE_OUTPUT_LN_RNG_BLACKWELL = value


@triton.jit
def rand3x(seed, offsets, n_rounds: tl.constexpr = 10):  # pyre-ignore [9]
    i1, i2, i3, _ = tl.randint4x(seed, offsets, n_rounds)
    u1 = tl.uint_to_uniform_float(i1)
    u2 = tl.uint_to_uniform_float(i2)
    u3 = tl.uint_to_uniform_float(i3)
    return u1, u2, u3


@triton.jit
def _philox4x32_10_mad64(offset, seed):
    """Bit-identical tl.randint4x(seed, offset) for 32-bit offsets.

    Each Philox round needs hi and lo halves of the same 32x32 products. Taking
    both from one 64-bit product lowers to v_mad_u64_u32 on CDNA instead of a
    separate v_mul_hi_u32 + v_mul_lo_u32 pair, which halves multiply issue in
    the VALU-bound mask generator.
    """
    PHILOX_ROUND_A: tl.constexpr = 0xD2511F53
    PHILOX_ROUND_B: tl.constexpr = 0xCD9E8D57
    seed = seed.to(tl.uint64)
    k0 = (seed & 0xFFFFFFFF).to(tl.uint32)
    k1 = ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
    c0 = offset.to(tl.uint32, bitcast=True)
    zero = c0 * 0
    c1 = zero
    c2 = zero
    c3 = zero
    for _ in tl.static_range(10):
        prod_a = c0.to(tl.uint64) * PHILOX_ROUND_A
        prod_b = c2.to(tl.uint64) * PHILOX_ROUND_B
        c0 = (prod_b >> 32).to(tl.uint32) ^ c1 ^ k0
        c2 = (prod_a >> 32).to(tl.uint32) ^ c3 ^ k1
        c1 = prod_b.to(tl.uint32)
        c3 = prod_a.to(tl.uint32)
        k0 = tl.add(k0, 0x9E3779B9, sanitize_overflow=False)
        k1 = tl.add(k1, 0xBB67AE85, sanitize_overflow=False)
    return c0, c1, c2, c3


@triton.jit
def _mask_keep(r, keep_above):
    """Integer form of tl.uint_to_uniform_float(r) > dropout_ratio.

    uint_to_uniform_float maps int32 x to fp32(x' * scale) with
    x' = x ^ (x >> 31), which is monotone in x'. keep_above is the largest x'
    that is dropped (see _mask_keep_above), so the decision is bit-identical.
    """
    x = r.to(tl.int32, bitcast=True)
    return (x ^ (x >> 31)) > keep_above


@triton.jit
def _generate_random_mask(
    MASK_BUFFER,
    N,
    keep_above,
    seed,
    D: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_MASKS: tl.constexpr,
):
    """Generate bit-packed dropout masks for (N, D) tensors. Outputs int8.

    Processes 4 rows per program using rand4x. Mask j occupies bit j.
    Extraction: y = val & 1, x = val & 2, u = val & 4.
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    start_row = pid.to(tl.int64) * 4

    base_ptr = MASK_BUFFER + start_row * STRIDE + cols
    row0_mask = (start_row < N) & col_mask
    row1_mask = ((start_row + 1) < N) & col_mask
    row2_mask = ((start_row + 2) < N) & col_mask
    row3_mask = ((start_row + 3) < N) & col_mask

    # Each pid uses NUM_MASKS consecutive BLOCK_D chunks for Philox offsets
    rand_offset = pid * (NUM_MASKS * BLOCK_D) + cols

    packed0 = tl.zeros([BLOCK_D], dtype=tl.int8)
    packed1 = tl.zeros([BLOCK_D], dtype=tl.int8)
    packed2 = tl.zeros([BLOCK_D], dtype=tl.int8)
    packed3 = tl.zeros([BLOCK_D], dtype=tl.int8)

    for j in tl.static_range(NUM_MASKS):
        r0, r1, r2, r3 = _philox4x32_10_mad64(rand_offset, seed)
        packed0 |= _mask_keep(r0, keep_above).to(tl.int8) << j
        packed1 |= _mask_keep(r1, keep_above).to(tl.int8) << j
        packed2 |= _mask_keep(r2, keep_above).to(tl.int8) << j
        packed3 |= _mask_keep(r3, keep_above).to(tl.int8) << j
        rand_offset += BLOCK_D

    tl.store(base_ptr, packed0, mask=row0_mask)
    tl.store(base_ptr + STRIDE, packed1, mask=row1_mask)
    tl.store(base_ptr + 2 * STRIDE, packed2, mask=row2_mask)
    tl.store(base_ptr + 3 * STRIDE, packed3, mask=row3_mask)


@triton.jit
def _generate_indexed_random_mask(
    MASK_BUFFER,
    ROW_INDICES,
    N,
    keep_above,
    seed,
    D: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_MASKS: tl.constexpr,
):
    """Generate masks for compact rows using their full-tensor RNG row IDs."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    logical_row = tl.load(ROW_INDICES + row).to(tl.int64)
    philox_group = logical_row // 4
    philox_lane = logical_row % 4
    # int32 offsets, matching _generate_random_mask's pid-based offsets.
    rand_offset = (philox_group * (NUM_MASKS * BLOCK_D)).to(tl.int32) + cols
    packed = tl.zeros([BLOCK_D], dtype=tl.int8)
    for j in tl.static_range(NUM_MASKS):
        r0, r1, r2, r3 = _philox4x32_10_mad64(rand_offset, seed)
        selected = tl.where(
            philox_lane == 0,
            r0,
            tl.where(philox_lane == 1, r1, tl.where(philox_lane == 2, r2, r3)),
        )
        packed |= _mask_keep(selected, keep_above).to(tl.int8) << j
        rand_offset += BLOCK_D
    tl.store(
        MASK_BUFFER + row.to(tl.int64) * STRIDE + cols,
        packed,
        mask=(row < N) & col_mask,
    )


@triton.jit
def _keep16(r, keep_min16, HI: tl.constexpr):
    """Keep decision from one 16-bit half of a Philox word: half >= keep_min16."""
    if HI:
        h = (r >> 16).to(tl.int32)
    else:
        h = (r & 0xFFFF).to(tl.int32)
    return h >= keep_min16


@triton.jit
def _generate_random_mask16(
    MASK_BUFFER,
    N,
    keep_min16,
    seed,
    D: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_MASKS: tl.constexpr,
):
    """Like _generate_random_mask, but one keep decision per 16-bit half-word.

    Processes 8 rows per program with one Philox4x32 call per (mask, column):
    word i's low half decides row i and its high half decides row i + 4. That
    halves Philox work in this VALU-bound kernel. P(drop) = keep_min16 / 65536
    (6554 / 65536 = 0.1000061 for p = 0.1). Every output element still gets
    its own fresh random decision; only the RNG stream layout changes.
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    start_row = pid.to(tl.int64) * 8
    base_ptr = MASK_BUFFER + start_row * STRIDE + cols

    rand_offset = pid * (NUM_MASKS * BLOCK_D) + cols

    p0 = tl.zeros([BLOCK_D], dtype=tl.int8)
    p1 = tl.zeros([BLOCK_D], dtype=tl.int8)
    p2 = tl.zeros([BLOCK_D], dtype=tl.int8)
    p3 = tl.zeros([BLOCK_D], dtype=tl.int8)
    p4 = tl.zeros([BLOCK_D], dtype=tl.int8)
    p5 = tl.zeros([BLOCK_D], dtype=tl.int8)
    p6 = tl.zeros([BLOCK_D], dtype=tl.int8)
    p7 = tl.zeros([BLOCK_D], dtype=tl.int8)

    for j in tl.static_range(NUM_MASKS):
        r0, r1, r2, r3 = _philox4x32_10_mad64(rand_offset, seed)
        p0 |= _keep16(r0, keep_min16, False).to(tl.int8) << j
        p1 |= _keep16(r1, keep_min16, False).to(tl.int8) << j
        p2 |= _keep16(r2, keep_min16, False).to(tl.int8) << j
        p3 |= _keep16(r3, keep_min16, False).to(tl.int8) << j
        p4 |= _keep16(r0, keep_min16, True).to(tl.int8) << j
        p5 |= _keep16(r1, keep_min16, True).to(tl.int8) << j
        p6 |= _keep16(r2, keep_min16, True).to(tl.int8) << j
        p7 |= _keep16(r3, keep_min16, True).to(tl.int8) << j
        rand_offset += BLOCK_D

    tl.store(base_ptr, p0, mask=(start_row < N) & col_mask)
    tl.store(base_ptr + STRIDE, p1, mask=((start_row + 1) < N) & col_mask)
    tl.store(base_ptr + 2 * STRIDE, p2, mask=((start_row + 2) < N) & col_mask)
    tl.store(base_ptr + 3 * STRIDE, p3, mask=((start_row + 3) < N) & col_mask)
    tl.store(base_ptr + 4 * STRIDE, p4, mask=((start_row + 4) < N) & col_mask)
    tl.store(base_ptr + 5 * STRIDE, p5, mask=((start_row + 5) < N) & col_mask)
    tl.store(base_ptr + 6 * STRIDE, p6, mask=((start_row + 6) < N) & col_mask)
    tl.store(base_ptr + 7 * STRIDE, p7, mask=((start_row + 7) < N) & col_mask)


@triton.jit
def _generate_indexed_random_mask16(
    MASK_BUFFER,
    ROW_INDICES,
    N,
    keep_min16,
    seed,
    D: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_MASKS: tl.constexpr,
):
    """Compact-row counterpart of _generate_random_mask16 (same row mapping)."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    logical_row = tl.load(ROW_INDICES + row).to(tl.int64)
    philox_group = logical_row // 8
    lane = logical_row % 8
    word = lane % 4
    rand_offset = (philox_group * (NUM_MASKS * BLOCK_D)).to(tl.int32) + cols
    packed = tl.zeros([BLOCK_D], dtype=tl.int8)
    for j in tl.static_range(NUM_MASKS):
        r0, r1, r2, r3 = _philox4x32_10_mad64(rand_offset, seed)
        selected = tl.where(
            word == 0,
            r0,
            tl.where(word == 1, r1, tl.where(word == 2, r2, r3)),
        )
        half = tl.where(lane >= 4, selected >> 16, selected & 0xFFFF).to(tl.int32)
        packed |= (half >= keep_min16).to(tl.int8) << j
        rand_offset += BLOCK_D
    tl.store(
        MASK_BUFFER + row.to(tl.int64) * STRIDE + cols,
        packed,
        mask=(row < N) & col_mask,
    )


@triton_autotune(
    configs=_get_layer_norm_mul_dropout_fwd_multirow_configs(),
    key=["BLOCK_D"],
)
@triton.jit
def _ln_mul_dropout_fwd_rng(
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    RANDOM_MASK,
    N,
    D,
    eps,
    dropout_ratio,
    stride_x,
    stride_u,
    stride_y,
    stride_mask,
    Y_INV_SCALE,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_U: tl.constexpr,
    CONCAT_X: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
):
    block_id = tl.program_id(0)
    start_row = block_id * BLOCK_N

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    rows = start_row + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    # Pre-compute 2D mask for reuse in dropout and masked operations
    mask_2d = row_mask[:, None] & col_mask[None, :]
    # ROCm 10.2 / Triton 3.9 removed tl.make_block_ptr and the load/store
    # boundary_check / padding_option arguments. X, U and the Y segments use
    # the pointer + mask form Triton 3.6's RewriteTensorPointer lowered block
    # pointers to (the LayerNorm port's _bp_i64 / _bp_ptrs / _bp_mask, imported
    # after this kernel): every load and store has an int64 mask of its own,
    # never mask_2d (see the note after this kernel).
    bp_n = _bp_i64(N)
    bp_d = _bp_i64(D)
    bp_row0 = _bp_i64(start_row)

    # Load data blocks
    x_ptrs, x_offs = _bp_ptrs(X, bp_row0, _bp_i64(stride_x), BLOCK_N, BLOCK_D)
    x_block = tl.load(
        x_ptrs, mask=_bp_mask(x_offs, bp_n, bp_d, BLOCK_N, BLOCK_D), other=0.0
    ).to(tl.float32)
    u_ptrs, u_offs = _bp_ptrs(U, bp_row0, _bp_i64(stride_u), BLOCK_N, BLOCK_D)
    u_block = tl.load(
        u_ptrs, mask=_bp_mask(u_offs, bp_n, bp_d, BLOCK_N, BLOCK_D), other=0.0
    ).to(tl.float32)

    # Pre-compute inv_D to replace divisions with multiplications (optimization)
    inv_D = 1.0 / D

    mean = tl.sum(x_block, axis=1) * inv_D
    tl.store(Mean + rows, mean, mask=row_mask)
    mean = tl.expand_dims(mean, 1)

    x_mean = x_block - mean
    x_mean = tl.where(mask_2d, x_mean, 0.0)
    _var = x_mean * x_mean
    var = tl.sum(_var, axis=1) * inv_D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + rows, rstd, mask=row_mask)
    rstd = tl.expand_dims(rstd, 1)

    y = x_mean * rstd
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask).to(tl.float32)
    y = y * w[None, :] + b[None, :]

    # Pre-compute sigmoid once to avoid redundant computation
    sigmoid_u_block = tl.sigmoid(u_block)
    silu_u_block = u_block * sigmoid_u_block

    if MUL_U_ACTIVATION_TYPE == "silu":
        y = y * silu_u_block
    elif MUL_U_ACTIVATION_TYPE == "sigmoid":
        y = y * sigmoid_u_block
    else:
        y = y * u_block

    if CONCAT_U and SILU_U:
        # pyre-fixme[16]
        u_block = silu_u_block

    if TRAINING:
        # Reuse rows (as int64 for pointer arithmetic) and pre-computed mask_2d
        row_offsets_i64 = rows.to(tl.int64)
        # Pre-compute loop-invariant values
        dropout_scale = 1.0 / (1.0 - dropout_ratio)
        offsets = row_offsets_i64[:, None] * stride_mask + cols[None, :]

        if CONCAT_U or CONCAT_X:
            # All 2+ mask cases use compressed int8 format - load once
            compressed = tl.load(RANDOM_MASK + offsets, mask=mask_2d, other=0).to(
                tl.int32
            )
            # Bit 0 is always y_mask
            y_keep = (compressed & 1) != 0

            if CONCAT_U and CONCAT_X:
                # 3-mask: (u_mask << 2) | (x_mask << 1) | y_mask
                x_keep = (compressed & 2) != 0
                u_keep = (compressed & 4) != 0
                u_block = tl.where(u_keep, u_block * dropout_scale, 0.0)
                x_block = tl.where(x_keep, x_block * dropout_scale, 0.0)
            elif CONCAT_U:
                # 2-mask: (u_mask << 1) | y_mask
                u_keep = (compressed & 2) != 0
                u_block = tl.where(u_keep, u_block * dropout_scale, 0.0)
            else:  # CONCAT_X
                # 2-mask: (x_mask << 1) | y_mask
                x_keep = (compressed & 2) != 0
                x_block = tl.where(x_keep, x_block * dropout_scale, 0.0)

            y = tl.where(y_keep, y * dropout_scale, 0.0)
        else:
            # 1-mask: y_mask at bit 0
            y_keep = tl.load(RANDOM_MASK + offsets, mask=mask_2d, other=True)
            y = tl.where(y_keep, y * dropout_scale, 0.0)

    if OUTPUT_FP8:
        u_block = u_block * Y_INV_SCALE
        x_block = x_block * Y_INV_SCALE
        y = y * Y_INV_SCALE

    # Output segment s (0, 1, 2 in the concat layouts) is Y[:, s * D : s * D + D]:
    # the block pointer of shape (N, (s + 1) * D) at (start_row, s * D), lowered
    # as above, i.e. bounded at the segment's own end (s + 1) * D so that every
    # element is written once. Each store keeps a mask of its own: one mask for
    # all three changes the bits too (see the note after this kernel).
    y_ptrs, y_offs = _bp_ptrs(Y, bp_row0, _bp_i64(stride_y), BLOCK_N, BLOCK_D)
    y_rows = _bp_row_mask(y_offs, bp_n, BLOCK_N, BLOCK_D)
    y_c0 = tl.arange(0, BLOCK_D).to(tl.int64)[None, :]
    y_c1 = y_c0 + _bp_i64(D)
    y_c2 = y_c0 + _bp_i64(2 * D)
    y_m0 = y_rows & ((y_c0 >= 0) & (y_c0 < bp_d))
    y_m1 = y_rows & ((y_c1 >= 0) & (y_c1 < _bp_i64(2 * D)))
    y_m2 = y_rows & ((y_c2 >= 0) & (y_c2 < _bp_i64(3 * D)))
    if CONCAT_U and CONCAT_X:
        tl.store(y_ptrs, u_block.to(Y.dtype.element_ty), mask=y_m0)
        tl.store(y_ptrs + D, x_block.to(Y.dtype.element_ty), mask=y_m1)
        tl.store(y_ptrs + 2 * D, y.to(Y.dtype.element_ty), mask=y_m2)
    elif CONCAT_U:
        tl.store(y_ptrs, u_block.to(Y.dtype.element_ty), mask=y_m0)
        tl.store(y_ptrs + D, y.to(Y.dtype.element_ty), mask=y_m1)
    elif CONCAT_X:
        tl.store(y_ptrs, x_block.to(Y.dtype.element_ty), mask=y_m0)
        tl.store(y_ptrs + D, y.to(Y.dtype.element_ty), mask=y_m1)
    else:
        tl.store(y_ptrs, y.to(Y.dtype.element_ty), mask=y_m0)


# ROCm 10.2 / Triton 3.9: the LayerNorm port's block-pointer helpers used by
# _ln_mul_dropout_fwd_rng above, imported after it so that no line above moves
# (the kernel resolves them when first compiled; no import cycle).
from generative_recommenders.ops.triton.triton_layer_norm import (  # noqa: E402
    _bp_i64,
    _bp_mask,
    _bp_ptrs,
    _bp_row_mask,
)

# Line padding for _ln_mul_dropout_fwd_rng above: its pointer + mask port is
# shorter than the block-pointer code it replaced, and this block takes up the
# difference so that every later Triton decorator of this module stays on its
# original line. Triton's JIT cache key covers a kernel's starting line, and
# so do the disk-cache key and the TSPEC table's fn_cache_key
# (ops/triton/triton_spec_table.jsonl). The lines are pinned by
# ops/tests/l11_do8_mx6_seam_test.py (_BASE_DECORATORS). An edit above this
# block that adds or removes lines must remove or add as many lines here, or
# that test fails and every later kernel gets a new cache key.
#
# Masks. Where X, U or Y lives in a storage over 2 GB (no buffer ops; TSPEC
# rows 252/253 have U and Y over 2 GB, and Y is over 2 GB from N ~ 0.7M up)
# every masked load / store is a branch on its mask. LLVM merges branches on
# one mask value and moves the math they guard into the merged block, which
# changes how the sum of squares and y are contracted. With mask_2d (the mask
# of the tl.where feeding the sum of squares) as the load / store mask, rstd and
# y changed in a few percent of the rows on Triton 3.6 and 3.9 alike, breaking
# "_ln_mul_dropout_fwd_rng_v2 == this kernel at BLOCK_N=1, num_warps=1,
# bitwise"; one mask for all Y stores changes the 3.9 code the same way. The
# masks above have the block pointers' form (cf. the note above
# triton_layer_norm._bp_i64): the loads' (N, D) mask and a mask per Y segment.
# Segment 0's equals the loads' mask, which is harmless: the loads' and the
# stores' branches are not adjacent, so LLVM does not merge them.
#
# The replaced block-pointer code (Triton 3.6 lowered it on AMD with
# RewriteTensorPointer: int64 offsets, boundary_check -> 0 <= offset < shape
# per checked dim, padding_option "zero" -> other=0), for reference:
#
#     X_block_ptr = tl.make_block_ptr(base=X, shape=(N, D), strides=(stride_x, 1),
#         offsets=(start_row, 0), block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
#     U_block_ptr: the same with base=U, strides=(stride_u, 1)
#     x_block = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero")
#     u_block: the same from U_block_ptr
#     ...
#     if CONCAT_U and CONCAT_X:
#         Y_block_ptr_u = tl.make_block_ptr(base=Y, shape=(N, 3 * D),
#             strides=(stride_y, 1), offsets=(start_row, 0),
#             block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
#         Y_block_ptr_x / Y_block_ptr_y: the same with offsets (start_row, D) /
#         (start_row, 2 * D); one tl.store(..., boundary_check=(0, 1)) each
#     elif CONCAT_U or CONCAT_X: two such pointers, shape=(N, 2 * D)
#     else: one, shape=(N, D)
#
# Output bounds. Those pointers bounded every segment by the full row width
# (2D or 3D), not at the segment's own end (s + 1) * D. For BLOCK_D == D (D = 512
# in production) that is the same set of stores. For BLOCK_D > D the old
# stores also wrote a segment's padding lanes over the first BLOCK_D - D
# columns of the next segment, which that segment's own store then overwrote:
# in program order within one warp, unordered across warps when num_warps > 1
# (the autotune configs go up to 4). The port bounds segment s at (s + 1) * D
# and writes every element once; the loads are the block pointers' exactly.


@triton.jit
def _amd_v_exp_f32(x):
    return tl.inline_asm_elementwise(
        "v_exp_f32 $0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _amd_v_rcp_f32(x):
    return tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _sigmoid_ieee_fast(u):
    """Bit-identical to tl.sigmoid(u) on gfx950 whenever d = 1 + exp(-u) is in
    [1, 2^126]. tl.sigmoid lowers to llvm.exp.f32 (v_exp_f32 of -u*log2(e),
    plus a denormal-range rescale that cannot change 1 + exp(-u) when
    exp(-u) < 2^-126) and an IEEE fdiv 1/d (div_scale, v_rcp_f32, five FMAs,
    div_fmas, div_fixup). For d in [1, 2^126] div_scale is the identity,
    div_fmas is a plain FMA and div_fixup returns its input, so the same
    v_rcp_f32 + FMA chain gives the same bits with 7 instead of 11 VALU ops
    and 2 instead of 7 for the exp. Returns (sigmoid, d); callers must fall
    back to tl.sigmoid when any d is outside that range (u < -87.3 or NaN)."""
    e = _amd_v_exp_f32(-u * 1.4426950408889634)
    d = 1.0 + e
    r = _amd_v_rcp_f32(d)
    t = tl.fma(-d, r, 1.0)
    r = tl.fma(t, r, r)
    t = tl.fma(-d, r, 1.0)
    q = tl.fma(t, r, r)
    t = tl.fma(-d, q, 1.0)
    q = tl.fma(t, r, q)
    return q, d


@triton.jit
def _keep_bit_and(v, c32, BIT: tl.constexpr):
    """tl.where(bit BIT of c32 set, v, 0.0) as v & (0 or ~0): same bits (+0.0
    when dropped) with v_bfe_i32 + v_and_b32 instead of shift/and/cmp/cndmask."""
    if BIT == 0:
        m = tl.inline_asm_elementwise(
            "v_bfe_i32 $0, $1, 0, 1", "=v,v", [c32], dtype=tl.int32, is_pure=True, pack=1
        )
    elif BIT == 1:
        m = tl.inline_asm_elementwise(
            "v_bfe_i32 $0, $1, 1, 1", "=v,v", [c32], dtype=tl.int32, is_pure=True, pack=1
        )
    else:
        m = tl.inline_asm_elementwise(
            "v_bfe_i32 $0, $1, 2, 1", "=v,v", [c32], dtype=tl.int32, is_pure=True, pack=1
        )
    return (v.to(tl.int32, bitcast=True) & m).to(tl.float32, bitcast=True)


@triton.jit
def _ln_mul_dropout_fwd_rng_v2(
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    RANDOM_MASK,
    N,
    D,
    eps,
    dropout_ratio,
    stride_x,
    stride_u,
    stride_y,
    stride_mask,
    Y_INV_SCALE,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_U: tl.constexpr,
    CONCAT_X: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """AMD (gfx950) v2 of _ln_mul_dropout_fwd_rng: same per-element math and the
    same per-row reduction layout as its BLOCK_N=1, num_warps=1 config (the one
    the autotuner picks at production N), so y/mean/rstd are bitwise equal to
    that config. Differences: persistent grid with a pipelined tl.range row
    loop and raw pointer loads; 1/D and 1/(1-p) hoisted out of the row loop;
    exact sigmoid via _sigmoid_ieee_fast (wave-uniform fallback to tl.sigmoid
    for rows with any d > 2^126 or NaN); dropout keep via bit masks. The old
    kernel was VALU bound (~491 VALU/row); this one issues ~310.
    Gate: DLRM_HSTU_FP8_LNMD_FWD_V2 (gin default on, yambda_5b.gin item (20);
    absent or =0 = the old kernel)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask).to(tl.float32)
    inv_D = 1.0 / D
    dropout_scale = 1.0 / (1.0 - dropout_ratio)
    NEED_SIG: tl.constexpr = (
        (MUL_U_ACTIVATION_TYPE == "silu")
        or (MUL_U_ACTIVATION_TYPE == "sigmoid")
        or (CONCAT_U and SILU_U)
    )
    for row in tl.range(pid, N, nprog, num_stages=NUM_STAGES):
        rows = row + tl.arange(0, 1)
        row_mask = rows < N
        mask_2d = row_mask[:, None] & col_mask[None, :]
        r64 = rows.to(tl.int64)
        x_block = tl.load(
            X + r64[:, None] * stride_x + cols[None, :], mask=mask_2d, other=0.0
        ).to(tl.float32)
        u_block = tl.load(
            U + r64[:, None] * stride_u + cols[None, :], mask=mask_2d, other=0.0
        ).to(tl.float32)
        if TRAINING:
            offsets = r64[:, None] * stride_mask + cols[None, :]
            if CONCAT_U or CONCAT_X:
                c32 = tl.load(RANDOM_MASK + offsets, mask=mask_2d, other=0).to(
                    tl.int32
                )
            else:
                y_keep1 = tl.load(RANDOM_MASK + offsets, mask=mask_2d, other=True)

        mean = tl.sum(x_block, axis=1) * inv_D
        tl.store(Mean + rows, mean, mask=row_mask)
        mean = tl.expand_dims(mean, 1)
        x_mean = x_block - mean
        x_mean = tl.where(mask_2d, x_mean, 0.0)
        _var = x_mean * x_mean
        var = tl.sum(_var, axis=1) * inv_D
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd + rows, rstd, mask=row_mask)
        rstd = tl.expand_dims(rstd, 1)

        y = x_mean * rstd
        y = y * w[None, :] + b[None, :]
        if NEED_SIG:
            sigmoid_u_block, d_sig = _sigmoid_ieee_fast(u_block)
            # d bits as uint32 > 0x7E800000 <=> d > 2^126, +inf or NaN (any sign)
            if tl.max(d_sig.to(tl.uint32, bitcast=True)) > 0x7E800000:
                sigmoid_u_block = tl.sigmoid(u_block)
            silu_u_block = u_block * sigmoid_u_block
        if MUL_U_ACTIVATION_TYPE == "silu":
            y = y * silu_u_block
        elif MUL_U_ACTIVATION_TYPE == "sigmoid":
            y = y * sigmoid_u_block
        else:
            y = y * u_block
        if CONCAT_U and SILU_U:
            u_block = silu_u_block

        if TRAINING:
            if CONCAT_U or CONCAT_X:
                if CONCAT_U and CONCAT_X:
                    u_block = _keep_bit_and(u_block * dropout_scale, c32, 2)
                    x_block = _keep_bit_and(x_block * dropout_scale, c32, 1)
                elif CONCAT_U:
                    u_block = _keep_bit_and(u_block * dropout_scale, c32, 1)
                else:
                    x_block = _keep_bit_and(x_block * dropout_scale, c32, 1)
                y = _keep_bit_and(y * dropout_scale, c32, 0)
            else:
                y = tl.where(y_keep1, y * dropout_scale, 0.0)

        if OUTPUT_FP8:
            u_block = u_block * Y_INV_SCALE
            x_block = x_block * Y_INV_SCALE
            y = y * Y_INV_SCALE

        yp = Y + r64[:, None] * stride_y + cols[None, :]
        if CONCAT_U and CONCAT_X:
            tl.store(yp, u_block.to(Y.dtype.element_ty), mask=mask_2d)
            tl.store(yp + D, x_block.to(Y.dtype.element_ty), mask=mask_2d)
            tl.store(yp + 2 * D, y.to(Y.dtype.element_ty), mask=mask_2d)
        elif CONCAT_U:
            tl.store(yp, u_block.to(Y.dtype.element_ty), mask=mask_2d)
            tl.store(yp + D, y.to(Y.dtype.element_ty), mask=mask_2d)
        elif CONCAT_X:
            tl.store(yp, x_block.to(Y.dtype.element_ty), mask=mask_2d)
            tl.store(yp + D, y.to(Y.dtype.element_ty), mask=mask_2d)
        else:
            tl.store(yp, y.to(Y.dtype.element_ty), mask=mask_2d)


@triton.jit
def _ln_mul_dropout_fwd_rng_v3(
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    RANDOM_MASK,
    N,
    D,
    eps,
    dropout_ratio,
    stride_x,
    stride_u,
    stride_y,
    stride_mask,
    Y_INV_SCALE,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_U: tl.constexpr,
    CONCAT_X: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
    PF2: tl.constexpr = False,
    PF_MASK: tl.constexpr = True,
    EVEN_D: tl.constexpr = False,
):
    """_ln_mul_dropout_fwd_rng_v2 with the row loop software-pipelined BY HAND.

    The per-row arithmetic is _ln_mul_dropout_fwd_rng_v2's, expression for
    expression (one row per loop step, num_warps=1 layout). The only change is
    that the next row's x/u/mask loads are issued into loop-carried registers
    before the current row is reduced, under tl.range(num_stages=1). v2 needed
    num_stages=1 for bitwise output because Triton's pipeliner peels an
    iteration whose copy of the math contracts differently; here the math
    exists exactly once, so every row runs the same code as v2's single loop
    body. PF2 keeps two rows in flight instead of one.
    Gate: DLRM_HSTU_FP8_LNMD_FWD_V3 (gin default on in yambda_5b.gin;
    absent or =0 = v2)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_D)
    if EVEN_D:
        col_mask = cols < BLOCK_D
    else:
        col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask).to(tl.float32)
    inv_D = 1.0 / D
    dropout_scale = 1.0 / (1.0 - dropout_ratio)
    NEED_SIG: tl.constexpr = (
        (MUL_U_ACTIVATION_TYPE == "silu")
        or (MUL_U_ACTIVATION_TYPE == "sigmoid")
        or (CONCAT_U and SILU_U)
    )
    MASKED: tl.constexpr = CONCAT_U or CONCAT_X

    # prologue: this program's first row (grid <= N, so pid < N)
    rows_n = pid + tl.arange(0, 1)
    m2_n = (rows_n < N)[:, None] & col_mask[None, :]
    r64_n = rows_n.to(tl.int64)
    x_n = tl.load(X + r64_n[:, None] * stride_x + cols[None, :], mask=m2_n, other=0.0)
    u_n = tl.load(U + r64_n[:, None] * stride_u + cols[None, :], mask=m2_n, other=0.0)
    if TRAINING and PF_MASK:
        if MASKED:
            k_n = tl.load(
                RANDOM_MASK + r64_n[:, None] * stride_mask + cols[None, :],
                mask=m2_n,
                other=0,
            )
        else:
            k_n = tl.load(
                RANDOM_MASK + r64_n[:, None] * stride_mask + cols[None, :],
                mask=m2_n,
                other=True,
            )
    else:
        k_n = x_n
    if PF2:
        rows_q = pid + nprog + tl.arange(0, 1)
        m2_q = (rows_q < N)[:, None] & col_mask[None, :]
        r64_q = rows_q.to(tl.int64)
        x_q = tl.load(X + r64_q[:, None] * stride_x + cols[None, :], mask=m2_q, other=0.0)
        u_q = tl.load(U + r64_q[:, None] * stride_u + cols[None, :], mask=m2_q, other=0.0)
        if TRAINING and PF_MASK:
            if MASKED:
                k_q = tl.load(
                    RANDOM_MASK + r64_q[:, None] * stride_mask + cols[None, :],
                    mask=m2_q,
                    other=0,
                )
            else:
                k_q = tl.load(
                    RANDOM_MASK + r64_q[:, None] * stride_mask + cols[None, :],
                    mask=m2_q,
                    other=True,
                )
        else:
            k_q = x_q

    for row in tl.range(pid, N, nprog, num_stages=1):
        rows = row + tl.arange(0, 1)
        row_mask = rows < N
        mask_2d = row_mask[:, None] & col_mask[None, :]
        r64 = rows.to(tl.int64)
        x_block = x_n.to(tl.float32)
        u_block = u_n.to(tl.float32)
        if TRAINING:
            if PF_MASK:
                if MASKED:
                    c32 = k_n.to(tl.int32)
                else:
                    y_keep1 = k_n
            else:
                offsets = r64[:, None] * stride_mask + cols[None, :]
                if MASKED:
                    c32 = tl.load(RANDOM_MASK + offsets, mask=mask_2d, other=0).to(
                        tl.int32
                    )
                else:
                    y_keep1 = tl.load(RANDOM_MASK + offsets, mask=mask_2d, other=True)
        # issue the next row's loads before this row's math
        if PF2:
            x_n = x_q
            u_n = u_q
            k_n = k_q
            rows_q = row + 2 * nprog + tl.arange(0, 1)
            m2_q = (rows_q < N)[:, None] & col_mask[None, :]
            r64_q = rows_q.to(tl.int64)
            x_q = tl.load(
                X + r64_q[:, None] * stride_x + cols[None, :], mask=m2_q, other=0.0
            )
            u_q = tl.load(
                U + r64_q[:, None] * stride_u + cols[None, :], mask=m2_q, other=0.0
            )
            if TRAINING and PF_MASK:
                if MASKED:
                    k_q = tl.load(
                        RANDOM_MASK + r64_q[:, None] * stride_mask + cols[None, :],
                        mask=m2_q,
                        other=0,
                    )
                else:
                    k_q = tl.load(
                        RANDOM_MASK + r64_q[:, None] * stride_mask + cols[None, :],
                        mask=m2_q,
                        other=True,
                    )
            else:
                k_q = x_q
        else:
            rows_n = row + nprog + tl.arange(0, 1)
            m2_n = (rows_n < N)[:, None] & col_mask[None, :]
            r64_n = rows_n.to(tl.int64)
            x_n = tl.load(
                X + r64_n[:, None] * stride_x + cols[None, :], mask=m2_n, other=0.0
            )
            u_n = tl.load(
                U + r64_n[:, None] * stride_u + cols[None, :], mask=m2_n, other=0.0
            )
            if TRAINING and PF_MASK:
                if MASKED:
                    k_n = tl.load(
                        RANDOM_MASK + r64_n[:, None] * stride_mask + cols[None, :],
                        mask=m2_n,
                        other=0,
                    )
                else:
                    k_n = tl.load(
                        RANDOM_MASK + r64_n[:, None] * stride_mask + cols[None, :],
                        mask=m2_n,
                        other=True,
                    )
            else:
                k_n = x_n

        mean = tl.sum(x_block, axis=1) * inv_D
        tl.store(Mean + rows, mean, mask=row_mask)
        mean = tl.expand_dims(mean, 1)
        x_mean = x_block - mean
        if not EVEN_D:
            # row < N inside the loop, so only padding columns can be masked
            x_mean = tl.where(mask_2d, x_mean, 0.0)
        _var = x_mean * x_mean
        var = tl.sum(_var, axis=1) * inv_D
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd + rows, rstd, mask=row_mask)
        rstd = tl.expand_dims(rstd, 1)

        y = x_mean * rstd
        y = y * w[None, :] + b[None, :]
        if NEED_SIG:
            sigmoid_u_block, d_sig = _sigmoid_ieee_fast(u_block)
            # d bits as uint32 > 0x7E800000 <=> d > 2^126, +inf or NaN (any sign)
            if tl.max(d_sig.to(tl.uint32, bitcast=True)) > 0x7E800000:
                sigmoid_u_block = tl.sigmoid(u_block)
            silu_u_block = u_block * sigmoid_u_block
        if MUL_U_ACTIVATION_TYPE == "silu":
            y = y * silu_u_block
        elif MUL_U_ACTIVATION_TYPE == "sigmoid":
            y = y * sigmoid_u_block
        else:
            y = y * u_block
        if CONCAT_U and SILU_U:
            u_block = silu_u_block

        if TRAINING:
            if CONCAT_U or CONCAT_X:
                if CONCAT_U and CONCAT_X:
                    u_block = _keep_bit_and(u_block * dropout_scale, c32, 2)
                    x_block = _keep_bit_and(x_block * dropout_scale, c32, 1)
                elif CONCAT_U:
                    u_block = _keep_bit_and(u_block * dropout_scale, c32, 1)
                else:
                    x_block = _keep_bit_and(x_block * dropout_scale, c32, 1)
                y = _keep_bit_and(y * dropout_scale, c32, 0)
            else:
                y = tl.where(y_keep1, y * dropout_scale, 0.0)

        if OUTPUT_FP8:
            u_block = u_block * Y_INV_SCALE
            x_block = x_block * Y_INV_SCALE
            y = y * Y_INV_SCALE

        yp = Y + r64[:, None] * stride_y + cols[None, :]
        if CONCAT_U and CONCAT_X:
            tl.store(yp, u_block.to(Y.dtype.element_ty), mask=mask_2d)
            tl.store(yp + D, x_block.to(Y.dtype.element_ty), mask=mask_2d)
            tl.store(yp + 2 * D, y.to(Y.dtype.element_ty), mask=mask_2d)
        elif CONCAT_U:
            tl.store(yp, u_block.to(Y.dtype.element_ty), mask=mask_2d)
            tl.store(yp + D, y.to(Y.dtype.element_ty), mask=mask_2d)
        elif CONCAT_X:
            tl.store(yp, x_block.to(Y.dtype.element_ty), mask=mask_2d)
            tl.store(yp + D, y.to(Y.dtype.element_ty), mask=mask_2d)
        else:
            tl.store(yp, y.to(Y.dtype.element_ty), mask=mask_2d)


# ---------------------------------------------------------------------------
# DLRM_HSTU_FP8_LNMD_FWD_MASKGEN (item (28)(e); gin default on in yambda_5b.gin,
# absent or =0 = _generate_random_mask16 + _ln_mul_dropout_fwd_rng_v3 / _v2).
# Mask-generating LN*U*dropout forward: one launch computes the generator's
# Philox keep decisions in registers and the v3 (= v2) row arithmetic. Gate:
# _use_lnmd_fwd_maskgen. Arbor r16 iteration 26 (M2 step 1b) / 27.


@triton.jit
def _pair_keep_bits(r, kk2, one2):
    """bit 0 = (r & 0xFFFF) >= K, bit 16 = (r >> 16) >= K, all other bits 0.
    kk2 = (K - 1) * 0x10001, one2 = 0x10001 (1 <= K <= 65535): v_pk_sub_u16
    clamp saturates each half to max(half - (K - 1), 0), which is >= 1 exactly
    when half >= K, and v_pk_min_u16 with 1 turns that into the 0/1 decision.
    Same decisions as _keep16(r, K, False) / _keep16(r, K, True) in 2 VALU."""
    s = tl.inline_asm_elementwise(
        "v_pk_sub_u16 $0, $1, $2 clamp", "=v,v,v", [r, kk2], dtype=tl.int32, is_pure=True, pack=1
    )
    return tl.inline_asm_elementwise(
        "v_pk_min_u16 $0, $1, $2", "=v,v,v", [s, one2], dtype=tl.int32, is_pure=True, pack=1
    )


@triton.jit
def _xor3s(a, b, k):
    """a ^ b ^ k in one gfx950 v_bitop3_b32 (truth table 0x96); k wave-uniform (SGPR)."""
    return tl.inline_asm_elementwise(
        "v_bitop3_b32 $0, $3, $1, $2 bitop3:0x96", "=v,v,v,s", [a, b, k],
        dtype=tl.uint32, is_pure=True, pack=1,
    )


@triton.jit
def _philox4x32_10_b3(offset, seed):
    """Bit-identical _philox4x32_10_mad64 (same words for every offset/seed): rounds
    1-2 keep plain ops (LLVM folds the zero counters and the uniform products), the
    3-input XORs of rounds 3-10 use v_bitop3_b32 instead of two v_xor_b32 (gfx950)."""
    PHILOX_ROUND_A: tl.constexpr = 0xD2511F53
    PHILOX_ROUND_B: tl.constexpr = 0xCD9E8D57
    seed = seed.to(tl.uint64)
    k0 = (seed & 0xFFFFFFFF).to(tl.uint32)
    k1 = ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
    c0 = offset.to(tl.uint32, bitcast=True)
    zero = c0 * 0
    c1 = zero
    c2 = zero
    c3 = zero
    for rnd in tl.static_range(10):
        prod_a = c0.to(tl.uint64) * PHILOX_ROUND_A
        prod_b = c2.to(tl.uint64) * PHILOX_ROUND_B
        if rnd < 2:
            c0n = (prod_b >> 32).to(tl.uint32) ^ c1 ^ k0
            c2n = (prod_a >> 32).to(tl.uint32) ^ c3 ^ k1
        elif rnd == 2:
            # c3 is still wave-uniform here (lo of a uniform product): LLVM folds c3 ^ k1 on the SALU
            c0n = _xor3s((prod_b >> 32).to(tl.uint32), c1, k0)
            c2n = (prod_a >> 32).to(tl.uint32) ^ c3 ^ k1
        else:
            c0n = _xor3s((prod_b >> 32).to(tl.uint32), c1, k0)
            c2n = _xor3s((prod_a >> 32).to(tl.uint32), c3, k1)
        c1 = prod_b.to(tl.uint32)
        c3 = prod_a.to(tl.uint32)
        c0 = c0n
        c2 = c2n
        k0 = tl.add(k0, 0x9E3779B9, sanitize_overflow=False)
        k1 = tl.add(k1, 0xBB67AE85, sanitize_overflow=False)
    return c0, c1, c2, c3


@triton.jit
def _pk_word(pk, r, kk2, one2, SHIFT: tl.constexpr):
    """pk |= the keep bits of both 16-bit halves of Philox word r, at bits SHIFT
    (low half) and SHIFT + 16 (high half)."""
    pk |= _pair_keep_bits(r.to(tl.int32, bitcast=True), kk2, one2) << SHIFT
    return pk


@triton.jit
def _load_row(X, U, row, N, stride_x, stride_u, cols, col_mask):
    rows_n = row + tl.arange(0, 1)
    m2_n = (rows_n < N)[:, None] & col_mask[None, :]
    r64_n = rows_n.to(tl.int64)
    x_n = tl.load(X + r64_n[:, None] * stride_x + cols[None, :], mask=m2_n, other=0.0)
    u_n = tl.load(U + r64_n[:, None] * stride_u + cols[None, :], mask=m2_n, other=0.0)
    return x_n, u_n


@triton.jit
def _row_c32(
    x_n, u_n, c32, row, N, Y, Mean, Rstd, RANDOM_MASK, w, b, cols, col_mask,
    inv_D, eps, dropout_scale, Y_INV_SCALE, D, stride_y, stride_mask,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
    SILU_U: tl.constexpr,
):
    """One row of _ln_mul_dropout_fwd_maskgen: stores the int8 mask byte c32
    (the value _generate_random_mask16 stores), then _ln_mul_dropout_fwd_rng_v3's
    per-row body (EVEN_D, CONCAT_U, CONCAT_X, TRAINING; SILU_U either way: the u
    plane holds silu(u) as in v3's `CONCAT_U and SILU_U` branch), expression for
    expression."""
    rows = row + tl.arange(0, 1)
    row_mask = rows < N
    mask_2d = row_mask[:, None] & col_mask[None, :]
    r64 = rows.to(tl.int64)
    tl.store(RANDOM_MASK + r64[:, None] * stride_mask + cols[None, :], c32.to(tl.int8), mask=mask_2d)
    x_block = x_n.to(tl.float32)
    u_block = u_n.to(tl.float32)
    mean = tl.sum(x_block, axis=1) * inv_D
    tl.store(Mean + rows, mean, mask=row_mask)
    mean = tl.expand_dims(mean, 1)
    x_mean = x_block - mean
    _var = x_mean * x_mean
    var = tl.sum(_var, axis=1) * inv_D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + rows, rstd, mask=row_mask)
    rstd = tl.expand_dims(rstd, 1)
    y = x_mean * rstd
    y = y * w[None, :] + b[None, :]
    sigmoid_u_block, d_sig = _sigmoid_ieee_fast(u_block)
    # d bits as uint32 > 0x7E800000 <=> d > 2^126, +inf or NaN (any sign)
    if tl.max(d_sig.to(tl.uint32, bitcast=True)) > 0x7E800000:
        sigmoid_u_block = tl.sigmoid(u_block)
    silu_u_block = u_block * sigmoid_u_block
    if MUL_U_ACTIVATION_TYPE == "silu":
        y = y * silu_u_block
    elif MUL_U_ACTIVATION_TYPE == "sigmoid":
        y = y * sigmoid_u_block
    else:
        y = y * u_block
    if SILU_U:
        u_block = silu_u_block
    u_block = _keep_bit_and(u_block * dropout_scale, c32, 2)
    x_block = _keep_bit_and(x_block * dropout_scale, c32, 1)
    y = _keep_bit_and(y * dropout_scale, c32, 0)
    if OUTPUT_FP8:
        u_block = u_block * Y_INV_SCALE
        x_block = x_block * Y_INV_SCALE
        y = y * Y_INV_SCALE
    yp = Y + r64[:, None] * stride_y + cols[None, :]
    tl.store(yp, u_block.to(Y.dtype.element_ty), mask=mask_2d)
    tl.store(yp + D, x_block.to(Y.dtype.element_ty), mask=mask_2d)
    tl.store(yp + 2 * D, y.to(Y.dtype.element_ty), mask=mask_2d)


@triton.jit
def _ln_mul_dropout_fwd_maskgen(
    X, U, Y, W, B, Mean, Rstd, RANDOM_MASK,
    N, D, eps, dropout_ratio,
    stride_x, stride_u, stride_y, stride_mask,
    Y_INV_SCALE,
    keep_min16, kk2_arg, seed,
    BLOCK_D: tl.constexpr,
    NUM_MASKS: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
    SILU_U: tl.constexpr,
):
    """_generate_random_mask16 + _ln_mul_dropout_fwd_rng_v3 (= v2) in ONE launch,
    byte-identical: the int8 mask equals the generator's, the y bytes, mean and
    rstd equal v3's / v2's. Only the configuration _use_lnmd_fwd_maskgen arms:
    training, CONCAT_U, CONCAT_X, NUM_MASKS=3, mask16 keep decisions, BLOCK_D ==
    D (the kernel hard-codes the first three), gfx950 (the asm below); SILU_U
    (silu(u) in the u plane, the production STU layers' setting) or not. Arbor
    r16 iteration 26, M2 step 1b ("f_b3a2_rl"; SILU_U added in iteration 27).

    Layout. One program = one wave (num_warps=1) = the generator's 8-row Philox
    group g (rows 8g..8g+7; grid cdiv(N, 8)). Each lane owns the same 8
    contiguous columns as v3's [1, BLOCK_D] row layout, so it computes the Philox
    words of exactly the columns it later uses: no LDS exchange, no barrier.

    Phase 1, Philox: NUM_MASKS x 8 columns = 24 Philox4x32-10 calls per lane with
    the generator's counter g * NUM_MASKS * BLOCK_D + j * BLOCK_D + col (int32)
    and seed. Word w's low 16-bit half decides row w, its high half row w + 4,
    plane j (bit j of the int8 mask: y = 1, x = 2, u = 4). Bit layout of the
    per-column int32 pk: row w (0..3) plane j at bit 3w + j, row w + 4 plane j at
    bit 16 + 3w + j; c32 = (pk >> base(row)) & 7 is the generator's byte.
      PAIR_BITS: v_pk_sub_u16 clamp with (K - 1) * 0x10001, then v_pk_min_u16
        with 0x10001, gives both 16-bit keep decisions of a word in 2 VALU
        (_pair_keep_bits; needs 1 <= K16 <= 65535).
      BITOP3: the 3-input XORs of Philox rounds 3-10 are one gfx950
        v_bitop3_b32 (0x96) instead of two v_xor_b32 (_philox4x32_10_b3; same
        words). LLVM forms neither v_xor3 nor bitop3 for these chains itself.
      ANCHOR: the counters pass through a 1-instruction v_mov_b32 asm that also
        names the row-0 x load. Without it Triton computes the Philox/pk chain
        (rooted at tl.arange, downstream of no load) in its default
        1-column-per-lane layout and converts 24 keep-bit tensors plus 8 int8
        mask rows per group to the data layout through LDS (160 ds_write + 128
        ds_read per group, occupancy 3); with it the chain stays in the data
        layout (no LDS ops, occupancy 4).

    Phase 2, rows: a runtime 8-step loop; row i + 1's x/u are issued before row
    i's math (row 0 before the Philox phase). Per row: store c32 to the int8 mask
    (the backward and the recompute of y read it), then _row_c32 = v3's row body
    (two-pass mean/var, rstd = 1/sqrt(var + eps), y*w + b, _sigmoid_ieee_fast
    with the wave-uniform tl.sigmoid fallback, mul, _keep_bit_and dropout,
    * Y_INV_SCALE, u | x | y stores in that order).

    Counter limit: g * NUM_MASKS * BLOCK_D must fit int32 (as in the generator's
    pid * NUM_MASKS * BLOCK_D), so N < 11,184,810 rows; the gate refuses larger N.
    keep_min16 (= K16) is passed for parity with the generator's launch
    arguments; the keep test uses kk2_arg = (K16 - 1) * 0x10001 (as int32)."""
    g = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < BLOCK_D
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask).to(tl.float32)
    inv_D = 1.0 / D
    dropout_scale = 1.0 / (1.0 - dropout_ratio)
    row0 = g * 8

    # row 0's x/u, issued before the Philox phase
    xa, ua = _load_row(X, U, row0, N, stride_x, stride_u, cols, col_mask)

    # ---- phase 1: the generator's Philox for this 8-row group, packed per column ----
    kk2 = tl.full([1, BLOCK_D], 0, tl.int32) + kk2_arg
    one2 = tl.full([1, BLOCK_D], 0x10001, tl.int32)
    pk = tl.zeros([1, BLOCK_D], dtype=tl.int32)
    rand_offset = g * (NUM_MASKS * BLOCK_D) + cols[None, :]
    # layout anchor (see docstring): the counters "depend" on the row-0 x load
    rand_offset = tl.inline_asm_elementwise(
        "v_mov_b32 $0, $1", "=v,v,v", [rand_offset, xa], dtype=tl.int32, is_pure=True, pack=1
    )
    for j in tl.static_range(NUM_MASKS):
        r0, r1, r2, r3 = _philox4x32_10_b3(rand_offset, seed)
        pk = _pk_word(pk, r0, kk2, one2, 0 + j)
        pk = _pk_word(pk, r1, kk2, one2, 3 + j)
        pk = _pk_word(pk, r2, kk2, one2, 6 + j)
        pk = _pk_word(pk, r3, kk2, one2, 9 + j)
        rand_offset += BLOCK_D

    # ---- phase 2: rows 0..7 of the group, next row's x/u carried in registers ----
    for i in tl.range(0, 8, num_stages=1):
        xc = xa
        uc = ua
        nrow = row0 + i + 1
        rows_n = nrow + tl.arange(0, 1)
        m2_n = ((rows_n < N) & (i < 7))[:, None] & col_mask[None, :]
        r64_n = rows_n.to(tl.int64)
        xa = tl.load(X + r64_n[:, None] * stride_x + cols[None, :], mask=m2_n, other=0.0)
        ua = tl.load(U + r64_n[:, None] * stride_u + cols[None, :], mask=m2_n, other=0.0)
        # row i (0..3): bits 3i..3i+2; row i (4..7): bits 16+3(i-4)..
        base = 3 * (i % 4) + 16 * (i // 4)
        _row_c32(
            xc, uc, (pk >> base) & 7, row0 + i, N, Y, Mean, Rstd, RANDOM_MASK, w, b, cols, col_mask,
            inv_D, eps, dropout_scale, Y_INV_SCALE, D, stride_y, stride_mask,
            MUL_U_ACTIVATION_TYPE, OUTPUT_FP8, SILU_U,
        )


@triton.jit
def _ln_mul_dropout_fwd(
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    D,
    eps,
    seed,
    dropout_ratio,
    stride_x,
    stride_u,
    stride_y,
    Y_INV_SCALE,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_U: tl.constexpr,
    CONCAT_X: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    FAST_DROPOUT: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)

    # Compute mean
    mean = 0.0
    x = tl.load(X + cols, mask=cols < D, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D

    # Compute variance
    _var = tl.zeros([BLOCK_D], dtype=tl.float32)
    x_mean = tl.where(cols < D, x - mean, 0.0)
    _var += x_mean * x_mean
    var = tl.sum(_var, axis=0) / D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    mask = cols < D
    y = x_mean * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    y = y * w + b
    u = tl.load(U + cols, mask=cols < D, other=0.0).to(tl.float32)
    sigmoid_u = tl.sigmoid(u)
    silu_u = u * sigmoid_u

    if MUL_U_ACTIVATION_TYPE == "silu":
        y = y * silu_u
    elif MUL_U_ACTIVATION_TYPE == "sigmoid":
        y = y * sigmoid_u
    else:
        y = y * u

    if CONCAT_U and SILU_U:
        u = silu_u

    if TRAINING:
        random_offsets = 3 * row * BLOCK_D + cols
        if CONCAT_U and CONCAT_X:
            # apply dropout on u
            if FAST_DROPOUT:
                random_u, random_x, random_y = rand3x(seed, random_offsets)
            else:
                random_u = tl.rand(seed, random_offsets)
            u_keep = random_u > dropout_ratio
            u = tl.where(u_keep, u / (1.0 - dropout_ratio), 0.0)
            # apply dropout on x
            if not FAST_DROPOUT:
                random_x = tl.rand(seed, random_offsets + D)
            x_keep = random_x > dropout_ratio  # pyre-ignore [61]
            x = tl.where(x_keep, x / (1.0 - dropout_ratio), 0.0)
            # apply dropout on y
            if not FAST_DROPOUT:
                random_y = tl.rand(seed, random_offsets + 2 * D)
            y_keep = random_y > dropout_ratio  # pyre-ignore [61]
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        elif CONCAT_U:
            # apply dropout on u
            if FAST_DROPOUT:
                random_u, random_y, _ = rand3x(seed, random_offsets)
            else:
                random_u = tl.rand(seed, random_offsets)
            u_keep = random_u > dropout_ratio
            u = tl.where(u_keep, u / (1.0 - dropout_ratio), 0.0)
            # apply dropout on y
            if not FAST_DROPOUT:
                random_y = tl.rand(seed, random_offsets + D)
            y_keep = random_y > dropout_ratio  # pyre-ignore [61]
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        elif CONCAT_X:
            # apply dropout on x
            if FAST_DROPOUT:
                random_x, random_y, _ = rand3x(seed, random_offsets)
            else:
                random_x = tl.rand(seed, random_offsets)
            x_keep = random_x > dropout_ratio
            x = tl.where(x_keep, x / (1.0 - dropout_ratio), 0.0)
            # apply dropout on y
            if not FAST_DROPOUT:
                random_y = tl.rand(seed, random_offsets + D)
            y_keep = random_y > dropout_ratio  # pyre-ignore [61]
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        else:
            random = tl.rand(seed, random_offsets)
            y_keep = random > dropout_ratio
            # write-back
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)

    if OUTPUT_FP8:
        u = u * Y_INV_SCALE
        x = x * Y_INV_SCALE
        y = y * Y_INV_SCALE

    # Write output
    if CONCAT_U and CONCAT_X:
        tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + D + cols, x.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + 2 * D + cols, y.to(Y.dtype.element_ty), mask=mask)
    elif CONCAT_U:
        tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + D + cols, y.to(Y.dtype.element_ty), mask=mask)
    elif CONCAT_X:
        tl.store(Y + cols, x.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + D + cols, y.to(Y.dtype.element_ty), mask=mask)
    else:
        tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _ln_mul_dropout_bwd_dx_du_rng(
    DX,
    DU,
    DY,
    DW,
    DB,
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    RANDOM_MASK,
    stride_dx,
    stride_du,
    stride_dy,
    stride_x,
    stride_u,
    stride_y,
    stride_mask,
    D,
    eps,
    dropout_ratio,
    N,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_U: tl.constexpr,
    CONCAT_X: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    COMPUTE_Y: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    if pid < N % tile_num:
        rows_per_tile += 1

    if rows_per_tile == 0:
        return

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    row = pid
    # Pre-compute row and pid as int64 once for initial pointer setup
    row_i64 = row.to(tl.int64)
    pid_i64 = pid.to(tl.int64)
    X += row_i64 * stride_x
    U += row_i64 * stride_u
    if COMPUTE_Y:
        Y += row_i64 * stride_y
    DY += row_i64 * stride_dy
    DX += row_i64 * stride_dx
    DU += row_i64 * stride_du
    DW = DW + pid_i64 * D + cols
    DB = DB + pid_i64 * D + cols

    # Pre-compute mask pointer offset (all cases use stride_mask for (N, D) shape)
    RANDOM_MASK += row_i64 * stride_mask

    partial_dw = tl.zeros((BLOCK_D,), dtype=tl.float32)
    partial_db = tl.zeros((BLOCK_D,), dtype=tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)

    dropout_scale = 0.0
    if TRAINING:
        dropout_scale = 1.0 / (1.0 - dropout_ratio)

    # Pre-compute inv_D to replace divisions with multiplications (optimization)
    inv_D = 1.0 / D

    # Pre-compute tile_num as int64 to avoid repeated conversion in the loop
    tile_num_i64 = tile_num.to(tl.int64)
    for _ in range(0, rows_per_tile):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        if CONCAT_U and CONCAT_X:
            du = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dx = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + 2 * D + cols, mask=mask, other=0).to(tl.float32)
        elif CONCAT_U:
            du = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
        elif CONCAT_X:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
        else:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if TRAINING:
            if CONCAT_U or CONCAT_X:
                # All 2+ mask cases use compressed int8 format - load once
                compressed = tl.load(RANDOM_MASK + cols, mask=mask, other=0).to(
                    tl.int32
                )
                dy_keep = (compressed & 1) != 0  # Bit 0 always y_mask

                if CONCAT_U and CONCAT_X:
                    # Format: (u_mask << 2) | (x_mask << 1) | y_mask
                    dx_keep = (compressed & 2) != 0
                    du_keep = (compressed & 4) != 0
                    du = tl.where(du_keep, du * dropout_scale, 0.0)
                    dx = tl.where(dx_keep, dx * dropout_scale, 0.0)
                elif CONCAT_U:
                    # Format: (u_mask << 1) | y_mask
                    du_keep = (compressed & 2) != 0
                    du = tl.where(du_keep, du * dropout_scale, 0.0)
                else:  # CONCAT_X
                    # Format: (x_mask << 1) | y_mask
                    dx_keep = (compressed & 2) != 0
                    dx = tl.where(dx_keep, dx * dropout_scale, 0.0)
                dy = tl.where(dy_keep, dy * dropout_scale, 0.0)
            else:
                # 1-mask: y_mask at bit 0
                dy_keep = tl.load(RANDOM_MASK + cols, mask=mask, other=True)
                dy = tl.where(dy_keep, dy * dropout_scale, 0.0)

        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)

        # Compute dx
        xhat = (x - mean) * rstd
        u = tl.load(U + cols, mask=mask, other=0).to(tl.float32)
        ln = xhat * w + b
        du_y = dy * ln
        mul_u = u
        sig_u = tl.sigmoid(u)

        # Pre-compute commonly used expressions to avoid redundant computation
        silu_u = u * sig_u  # silu(u) - used multiple times
        dsig_u = sig_u * (1.0 - sig_u)  # sigmoid derivative - used multiple times
        dsilu_u = sig_u + silu_u * (
            1.0 - sig_u
        )  # silu derivative - used multiple times

        if MUL_U_ACTIVATION_TYPE == "silu":
            mul_u = silu_u
            du_y = dy * ln * dsilu_u
            dy = dy * silu_u
        elif MUL_U_ACTIVATION_TYPE == "sigmoid":
            mul_u = sig_u
            du_y = dy * ln * dsig_u
            dy = dy * sig_u
        else:
            dy = dy * u

        du_u = du
        if CONCAT_U and SILU_U:
            du_u *= dsilu_u
            u = silu_u

        du = du_y + du_u

        tl.store(DU + cols, du.to(DU.dtype.element_ty), mask=mask)

        wdy = w * dy
        if COMPUTE_Y:
            y = ln * mul_u
            if TRAINING:
                if CONCAT_U:
                    u = tl.where(
                        du_keep,  # pyre-ignore [61]
                        u * dropout_scale,
                        0.0,
                    )
                if CONCAT_X:
                    x = tl.where(
                        dx_keep,  # pyre-ignore [61]
                        x * dropout_scale,
                        0.0,
                    )
                y = tl.where(
                    dy_keep,  # pyre-ignore [61]
                    y * dropout_scale,
                    0.0,
                )
            if CONCAT_U and CONCAT_X:
                tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, x.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + 2 * D + cols, y.to(Y.dtype.element_ty), mask=mask)
            elif CONCAT_U:
                tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, y.to(Y.dtype.element_ty), mask=mask)
            elif CONCAT_X:
                tl.store(Y + cols, x.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, y.to(Y.dtype.element_ty), mask=mask)
            else:
                tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
            Y += tile_num_i64 * stride_y

        # Note: xhat and wdy are already 0 outside valid range due to masked loads,
        # so no additional tl.where masking is needed before reduction
        c1 = tl.sum(xhat * wdy, axis=0) * inv_D
        c2 = tl.sum(wdy, axis=0) * inv_D
        dx += (wdy - (xhat * c1 + c2)) * rstd
        # Write dx
        tl.store(DX + cols, dx, mask=mask)

        # Accumulate partial sums for dw/db
        partial_dw += dy * xhat
        partial_db += dy
        X += tile_num_i64 * stride_x
        U += tile_num_i64 * stride_u
        DY += tile_num_i64 * stride_dy
        DX += tile_num_i64 * stride_dx
        DU += tile_num_i64 * stride_du
        # Increment mask pointer
        RANDOM_MASK += tile_num_i64 * stride_mask
        row += tile_num
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)


@triton.jit
def _ln_mul_dropout_bwd_dx_du_rng_v2(
    DX, DU, DY, DW, DB, X, U, Y, W, B, Mean, Rstd, RANDOM_MASK,
    stride_dx, stride_du, stride_dy, stride_x, stride_u, stride_y, stride_mask,
    D, eps, dropout_ratio, N,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_U: tl.constexpr,
    CONCAT_X: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    COMPUTE_Y: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    CONTIG: tl.constexpr,
    EVEN_D: tl.constexpr,
    SCM: tl.constexpr = "",
    LEV: tl.constexpr = "",
):
    """Same per-element math (same op order) and same strided row-to-program
    assignment as _ln_mul_dropout_bwd_dx_du_rng. Differences: index-addressed
    raw pointer loads, a tl.range row loop, streaming (.cs) stores, and it is
    launched with half the warps (1 at D=512), which changes the row reduction
    tree for c1/c2. Measured vs the old kernel (bf16, production flags): du and
    y bitwise; dx differs in ~1e-5 of elements (reduction-order rounding);
    dw/db bitwise at N=2.37M, 1 bf16 ulp in <=1/256 elements at some N
    (compiler FMA contraction of the fp32 partial sums). Deterministic.
    Gate: DLRM_HSTU_FP8_LNMD_BWD_V2 (gin default on, yambda_5b.gin item (14); absent or =0 = the old kernel)."""
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    extra = N % tile_num
    if CONTIG:
        start = pid * rows_per_tile + tl.minimum(pid, extra)
        step = 1
    else:
        start = pid
        step = tile_num
    if pid < extra:
        rows_per_tile += 1

    cols = tl.arange(0, BLOCK_D)
    if EVEN_D:
        mask = cols < BLOCK_D
    else:
        mask = cols < D
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    partial_dw = tl.zeros((BLOCK_D,), dtype=tl.float32)
    partial_db = tl.zeros((BLOCK_D,), dtype=tl.float32)
    dropout_scale = 0.0
    if TRAINING:
        dropout_scale = 1.0 / (1.0 - dropout_ratio)
    inv_D = 1.0 / D

    for i in tl.range(0, rows_per_tile, num_stages=NUM_STAGES):
        row = start + i * step
        r64 = row.to(tl.int64)
        xp = X + r64 * stride_x + cols
        up = U + r64 * stride_u + cols
        dyp = DY + r64 * stride_dy + cols
        x = tl.load(xp, mask=mask, other=0, eviction_policy=LEV).to(tl.float32)
        if CONCAT_U and CONCAT_X:
            du = tl.load(dyp, mask=mask, other=0, eviction_policy=LEV).to(tl.float32)
            dx = tl.load(dyp + D, mask=mask, other=0, eviction_policy=LEV).to(tl.float32)
            dy = tl.load(dyp + 2 * D, mask=mask, other=0, eviction_policy=LEV).to(tl.float32)
        elif CONCAT_U:
            du = tl.load(dyp, mask=mask, other=0, eviction_policy=LEV).to(tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(dyp + D, mask=mask, other=0, eviction_policy=LEV).to(tl.float32)
        elif CONCAT_X:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.load(dyp, mask=mask, other=0, eviction_policy=LEV).to(tl.float32)
            dy = tl.load(dyp + D, mask=mask, other=0, eviction_policy=LEV).to(tl.float32)
        else:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(dyp, mask=mask, other=0, eviction_policy=LEV).to(tl.float32)
        u = tl.load(up, mask=mask, other=0, eviction_policy=LEV).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        if TRAINING:
            mp = RANDOM_MASK + r64 * stride_mask + cols
            if CONCAT_U or CONCAT_X:
                compressed = tl.load(mp, mask=mask, other=0, eviction_policy=LEV).to(tl.int32)
                dy_keep = (compressed & 1) != 0
                if CONCAT_U and CONCAT_X:
                    dx_keep = (compressed & 2) != 0
                    du_keep = (compressed & 4) != 0
                    du = tl.where(du_keep, du * dropout_scale, 0.0)
                    dx = tl.where(dx_keep, dx * dropout_scale, 0.0)
                elif CONCAT_U:
                    du_keep = (compressed & 2) != 0
                    du = tl.where(du_keep, du * dropout_scale, 0.0)
                else:
                    dx_keep = (compressed & 2) != 0
                    dx = tl.where(dx_keep, dx * dropout_scale, 0.0)
                dy = tl.where(dy_keep, dy * dropout_scale, 0.0)
            else:
                dy_keep = tl.load(mp, mask=mask, other=True)
                dy = tl.where(dy_keep, dy * dropout_scale, 0.0)

        xhat = (x - mean) * rstd
        ln = xhat * w + b
        du_y = dy * ln
        mul_u = u
        sig_u = tl.sigmoid(u)
        silu_u = u * sig_u
        dsig_u = sig_u * (1.0 - sig_u)
        dsilu_u = sig_u + silu_u * (1.0 - sig_u)
        if MUL_U_ACTIVATION_TYPE == "silu":
            mul_u = silu_u
            du_y = dy * ln * dsilu_u
            dy = dy * silu_u
        elif MUL_U_ACTIVATION_TYPE == "sigmoid":
            mul_u = sig_u
            du_y = dy * ln * dsig_u
            dy = dy * sig_u
        else:
            dy = dy * u
        du_u = du
        if CONCAT_U and SILU_U:
            du_u *= dsilu_u
            u = silu_u
        du = du_y + du_u
        tl.store(DU + r64 * stride_du + cols, du.to(DU.dtype.element_ty), mask=mask, cache_modifier=SCM)

        wdy = w * dy
        if COMPUTE_Y:
            yp = Y + r64 * stride_y + cols
            y = ln * mul_u
            if TRAINING:
                if CONCAT_U:
                    u = tl.where(du_keep, u * dropout_scale, 0.0)
                if CONCAT_X:
                    x = tl.where(dx_keep, x * dropout_scale, 0.0)
                y = tl.where(dy_keep, y * dropout_scale, 0.0)
            if CONCAT_U and CONCAT_X:
                tl.store(yp, u.to(Y.dtype.element_ty), mask=mask, cache_modifier=SCM)
                tl.store(yp + D, x.to(Y.dtype.element_ty), mask=mask, cache_modifier=SCM)
                tl.store(yp + 2 * D, y.to(Y.dtype.element_ty), mask=mask, cache_modifier=SCM)
            elif CONCAT_U:
                tl.store(yp, u.to(Y.dtype.element_ty), mask=mask, cache_modifier=SCM)
                tl.store(yp + D, y.to(Y.dtype.element_ty), mask=mask, cache_modifier=SCM)
            elif CONCAT_X:
                tl.store(yp, x.to(Y.dtype.element_ty), mask=mask, cache_modifier=SCM)
                tl.store(yp + D, y.to(Y.dtype.element_ty), mask=mask, cache_modifier=SCM)
            else:
                tl.store(yp, y.to(Y.dtype.element_ty), mask=mask, cache_modifier=SCM)

        c1 = tl.sum(xhat * wdy, axis=0) * inv_D
        c2 = tl.sum(wdy, axis=0) * inv_D
        dx += (wdy - (xhat * c1 + c2)) * rstd
        tl.store(DX + r64 * stride_dx + cols, dx, mask=mask, cache_modifier=SCM)
        partial_dw += dy * xhat
        partial_db += dy
    pid64 = pid.to(tl.int64)
    tl.store(DW + pid64 * D + cols, partial_dw, mask=mask)
    tl.store(DB + pid64 * D + cols, partial_db, mask=mask)


@triton.jit
def _ln_mul_dropout_bwd_dx_du_rng_v3(
    DX, DY, DW, DB, X, U, W, B, Mean, Rstd, RANDOM_MASK,
    DZ8, PARTIAL_AMAX, PARTIAL_COL, INV_SCALE,
    stride_dx, stride_dy, stride_x, stride_u, stride_mask, stride_dz8,
    D, dropout_ratio, N, N_PAD,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LIMIT: tl.constexpr,
    SCM: tl.constexpr = ".cs",
    LEV: tl.constexpr = "",
    SCM8: tl.constexpr = ".cs",
):
    """B1a (Arbor r16, DLRM_HSTU_FP8_BWD_DU_E5M2, default off): the e5m2 du
    producer for the ordinary layer (hstu_du_e5m2.py).

    `_ln_mul_dropout_bwd_dx_du_rng_v2` specialised to the production contract
    (TRAINING, CONCAT_U and CONCAT_X, MUL_U_ACTIVATION_TYPE "silu", COMPUTE_Y
    False, int8 mask16 mask: bit0 y, bit1 x, bit2 u; BLOCK_D == D, EVEN_D,
    NUM_STAGES 1, CONTIG False) with r16's iter-26 microbench switches fixed
    to DIET=1 EMIT_FP8=1 STORE_DU=0 AMAX_MODE=4 ZERO_TAIL=1:
      * diet (value-preserving instruction changes): the exact
        `_sigmoid_ieee_fast` with the wave-uniform `tl.sigmoid` fallback when
        any d = 1 + exp(-u) is > 2^126 / inf / NaN (as the v2 forward),
        `_keep_bit_and` for the keep bits, the y-plane loaded first and the
        x-gradient plane loaded after du is consumed;
      * emission: du rounded to bf16 (the value v2 stores), back to fp32,
        clamp(du * inv_scale, +-LIMIT) -> e5m2 into DZ8[row, 0:BLOCK_D]
        (exactly `bwd_fp8._quantize_amax_colsum_kernel`'s ops); per-program
        fp32 column sums of that bf16 value (PARTIAL_COL[pid, :]) and the max
        of its int16 bit patterns & 0x7FFF with NaN patterns mapped to 0
        (= |bf16 du| max with maxnum semantics; PARTIAL_AMAX[pid]); NO bf16
        du store;
      * rows [N, N_PAD) of the u block are zeroed by a grid-stride tail loop
        (pid, pid + tile_num, ...), so any N_PAD - N is covered.
    Same row -> program assignment as v2 at the same tile_num, so dx and the
    DW/DB partials are bitwise v2's (iter-26: byte-equal at N = 2,364,904 incl.
    special-value rows; payload / amax byte-equal to the shipped cast at three
    inv_scales incl. clipping and deep subnormals).
    """
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    extra = N % tile_num
    start = pid
    step = tile_num
    if pid < extra:
        rows_per_tile += 1

    cols = tl.arange(0, BLOCK_D)
    mask = cols < BLOCK_D
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    partial_dw = tl.zeros((BLOCK_D,), dtype=tl.float32)
    partial_db = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc_col = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc_amax_i = tl.zeros((BLOCK_D,), dtype=tl.int16)
    inv = tl.load(INV_SCALE)
    dropout_scale = 1.0 / (1.0 - dropout_ratio)
    inv_D = 1.0 / D

    for i in tl.range(0, rows_per_tile, num_stages=1):
        row = start + i * step
        r64 = row.to(tl.int64)
        xp = X + r64 * stride_x + cols
        up = U + r64 * stride_u + cols
        dyp = DY + r64 * stride_dy + cols
        mp = RANDOM_MASK + r64 * stride_mask + cols
        # y-plane (third dY plane) first; the x-gradient plane is deferred.
        dy = tl.load(dyp + 2 * D, mask=mask, other=0, eviction_policy=LEV).to(tl.float32)
        u = tl.load(up, mask=mask, other=0, eviction_policy=LEV).to(tl.float32)
        x = tl.load(xp, mask=mask, other=0, eviction_policy=LEV).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        c32 = tl.load(mp, mask=mask, other=0, eviction_policy=LEV).to(tl.int32)
        du = tl.load(dyp, mask=mask, other=0, eviction_policy=LEV).to(tl.float32)
        dy = _keep_bit_and(dy * dropout_scale, c32, 0)
        du = _keep_bit_and(du * dropout_scale, c32, 2)
        xhat = (x - mean) * rstd
        ln = xhat * w + b
        sig_u, d_sig = _sigmoid_ieee_fast(u)
        # d bits as uint32 > 0x7E800000 <=> d > 2^126, +inf or NaN (any sign)
        if tl.max(d_sig.to(tl.uint32, bitcast=True)) > 0x7E800000:
            sig_u = tl.sigmoid(u)
        silu_u = u * sig_u
        dsilu_u = sig_u + silu_u * (1.0 - sig_u)
        du_y = dy * ln * dsilu_u
        dy = dy * silu_u
        du_u = du
        if SILU_U:
            du_u *= dsilu_u
        du = du_y + du_u
        # The bf16 value v2 stores, then the shipped cast's ops on it.
        du_b = du.to(tl.bfloat16)
        duf = du_b.to(tl.float32)
        acc_col += duf
        mb = du_b.to(tl.int16, bitcast=True) & 0x7FFF
        acc_amax_i = tl.maximum(acc_amax_i, tl.where(mb > 0x7F80, 0, mb).to(tl.int16))
        scaled = tl.minimum(tl.maximum(duf * inv, -LIMIT), LIMIT)
        tl.store(
            DZ8 + r64 * stride_dz8 + cols,
            scaled.to(DZ8.dtype.element_ty),
            mask=mask,
            cache_modifier=SCM8,
        )
        dx = tl.load(dyp + D, mask=mask, other=0, eviction_policy=LEV).to(tl.float32)
        dx = _keep_bit_and(dx * dropout_scale, c32, 1)
        wdy = w * dy
        c1 = tl.sum(xhat * wdy, axis=0) * inv_D
        c2 = tl.sum(wdy, axis=0) * inv_D
        dx += (wdy - (xhat * c1 + c2)) * rstd
        tl.store(DX + r64 * stride_dx + cols, dx, mask=mask, cache_modifier=SCM)
        partial_dw += dy * xhat
        partial_db += dy
    pid64 = pid.to(tl.int64)
    tl.store(DW + pid64 * D + cols, partial_dw, mask=mask)
    tl.store(DB + pid64 * D + cols, partial_db, mask=mask)
    tl.store(PARTIAL_COL + pid64 * D + cols, acc_col, mask=mask)
    mi = tl.max(acc_amax_i, axis=0).to(tl.int32) << 16
    tl.store(PARTIAL_AMAX + pid, mi.to(tl.float32, bitcast=True))
    # Pad rows [N, N_PAD) of the u block: zero, as the shipped cast writes them.
    zero8 = tl.zeros((BLOCK_D,), dtype=tl.float32).to(DZ8.dtype.element_ty)
    for zr in range(N + pid, N_PAD, tile_num):
        tl.store(DZ8 + zr.to(tl.int64) * stride_dz8 + cols, zero8, mask=mask)


@triton.jit
def _ln_mul_dropout_bwd_dx_du(
    DX,
    DU,
    DY,
    DW,
    DB,
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride_dx,
    stride_du,
    stride_dy,
    stride_x,
    stride_u,
    stride_y,
    D,
    eps,
    seed,
    dropout_ratio,
    N,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_U: tl.constexpr,
    CONCAT_X: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    COMPUTE_Y: tl.constexpr,
    FAST_DROPOUT: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    if pid < N % tile_num:
        rows_per_tile += 1

    if rows_per_tile == 0:
        return

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    row = pid
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    if COMPUTE_Y:
        Y += row.to(tl.int64) * stride_y
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx
    DU += row.to(tl.int64) * stride_du
    DW = DW + pid * D + cols
    DB = DB + pid * D + cols

    partial_dw = tl.zeros((BLOCK_D,), dtype=tl.float32)
    partial_db = tl.zeros((BLOCK_D,), dtype=tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    for _idx in range(0, rows_per_tile):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        if CONCAT_U and CONCAT_X:
            du = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dx = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + 2 * D + cols, mask=mask, other=0).to(tl.float32)
        elif CONCAT_U:
            du = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
        elif CONCAT_X:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
        else:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if TRAINING:
            random_offsets = 3 * row * BLOCK_D + cols
            if CONCAT_U and CONCAT_X:
                # apply dropout on du
                if FAST_DROPOUT:
                    random_du, random_dx, random_dy = rand3x(seed, random_offsets)
                else:
                    random_du = tl.rand(seed, random_offsets)
                du_keep = random_du > dropout_ratio
                du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
                # apply dropout on dx
                if not FAST_DROPOUT:
                    random_dx = tl.rand(seed, random_offsets + D)
                dx_keep = random_dx > dropout_ratio  # pyre-ignore [61]
                dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
                # apply dropout on dy
                if not FAST_DROPOUT:
                    random_dy = tl.rand(seed, random_offsets + 2 * D)
                dy_keep = random_dy > dropout_ratio  # pyre-ignore [61]
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
            elif CONCAT_U:
                # apply dropout on du
                if FAST_DROPOUT:
                    random_du, _, random_dy = rand3x(seed, random_offsets)
                else:
                    random_du = tl.rand(seed, random_offsets)
                du_keep = random_du > dropout_ratio
                du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
                # apply dropout on dy
                if not FAST_DROPOUT:
                    random_dy = tl.rand(seed, random_offsets + D)
                dy_keep = random_dy > dropout_ratio  # pyre-ignore [61]
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
            elif CONCAT_X:
                # apply dropout on dx
                if FAST_DROPOUT:
                    _, random_dx, random_dy = rand3x(seed, random_offsets)
                else:
                    random_dx = tl.rand(seed, random_offsets)
                dx_keep = random_dx > dropout_ratio  # pyre-ignore [61]
                dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
                # apply dropout on dy
                if not FAST_DROPOUT:
                    random_dy = tl.rand(seed, random_offsets + D)
                dy_keep = random_dy > dropout_ratio  # pyre-ignore [61]
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
            else:
                random = tl.rand(seed, random_offsets)
                dy_keep = random > dropout_ratio
                # write-back
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)

        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)

        # Compute dx
        xhat = (x - mean) * rstd
        u = tl.load(U + cols, mask=mask, other=0).to(tl.float32)
        ln = xhat * w + b
        du_y = dy * ln
        mul_u = u
        sig_u = tl.sigmoid(u)

        if MUL_U_ACTIVATION_TYPE == "silu":
            mul_u = u * sig_u
            du_y = dy * ln * (sig_u + u * sig_u * (1.0 - sig_u))
            dy = dy * u * sig_u
        elif MUL_U_ACTIVATION_TYPE == "sigmoid":
            mul_u = sig_u
            du_y = dy * ln * (sig_u * (1.0 - sig_u))
            dy = dy * sig_u
        else:
            dy = dy * u

        du_u = du
        if CONCAT_U:
            if SILU_U:
                du_u *= sig_u + u * sig_u * (1.0 - sig_u)
                u = u * sig_u

        du = du_y + du_u

        tl.store(DU + cols, du.to(DU.dtype.element_ty), mask=mask)
        wdy = w * dy
        if COMPUTE_Y:
            y = ln * mul_u
            if TRAINING:
                if CONCAT_U:
                    u = tl.where(
                        du_keep,  # pyre-ignore [61]
                        u / (1.0 - dropout_ratio),
                        0.0,
                    )
                if CONCAT_X:
                    x = tl.where(
                        dx_keep,  # pyre-ignore [61]
                        x / (1.0 - dropout_ratio),
                        0.0,
                    )
                y = tl.where(
                    dy_keep,  # pyre-ignore [61]
                    y / (1.0 - dropout_ratio),
                    0.0,
                )
            if CONCAT_U and CONCAT_X:
                tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, x.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + 2 * D + cols, y.to(Y.dtype.element_ty), mask=mask)
            elif CONCAT_U:
                tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, y.to(Y.dtype.element_ty), mask=mask)
            elif CONCAT_X:
                tl.store(Y + cols, x.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, y.to(Y.dtype.element_ty), mask=mask)
            else:
                tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
            Y += tile_num.to(tl.int64) * stride_y

        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        c1 = tl.sum(xhat * wdy, axis=0) / D
        c2 = tl.sum(wdy, axis=0) / D
        dx += (wdy - (xhat * c1 + c2)) * rstd
        # Write dx
        tl.store(DX + cols, dx, mask=mask)

        # Accumulate partial sums for dw/db
        partial_dw += dy * xhat
        partial_db += dy
        X += tile_num.to(tl.int64) * stride_x
        U += tile_num.to(tl.int64) * stride_u
        DY += tile_num.to(tl.int64) * stride_dy
        DX += tile_num.to(tl.int64) * stride_dx
        DU += tile_num.to(tl.int64) * stride_du
        row += tile_num
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)


def _get_bwd_dwdb_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_N in [32, 64, 128, 256]:
        for num_warps in [8, 16] + ([] if torch.ops.hip else [32]):
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs


@triton_autotune(
    configs=_get_bwd_dwdb_configs(),
    key=["D"],
)
@triton.jit
def _ln_mul_dropout_bwd_dwdb(
    DW,
    DB,
    FINAL_DW,
    FINAL_DB,
    N,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    db = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=cols < D)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.dtype.element_ty), mask=cols < D)


@functools.lru_cache(maxsize=64)
def _mask_keep_above(dropout_ratio: float) -> int:
    """Largest x' in [-1, 2**31 - 1] for which the fp32 dropout test fails.

    Emulates Triton's tl.uint_to_uniform_float(x) > dropout_ratio in fp32:
    fp32(fp32(x') * fp32(4.6566127342e-10)) > fp32(dropout_ratio), which is
    monotone in x'. Returns -1 when every value is kept.
    """
    scale = np.float32(4.6566127342e-10)
    p = np.float32(dropout_ratio)

    def keep(v: int) -> bool:
        return bool(np.float32(np.float32(v) * scale) > p)

    lo, hi = 0, 2**31  # first kept x' lies in [lo, hi]; hi means none kept
    while lo < hi:
        mid = (lo + hi) // 2
        if keep(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo - 1


def _use_mask16() -> bool:
    """16-bit-half dropout decisions (half the Philox work). Opt in with
    DLRM_HSTU_FP8_MASK16=1 (gin: apply_env_bootstrap.DLRM_HSTU_FP8_MASK16);
    absent or 0 keeps the 32-bit, bit-identical-to-randint4x path.
    Read per call (cheap) so the switch also works when set after import."""
    global _MASK16_REPORTED
    on = os.environ.get("DLRM_HSTU_FP8_MASK16", "0") == "1"
    if on and not _MASK16_REPORTED:
        _MASK16_REPORTED = True
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0
        ):
            print("armed OK: dropout mask16 ACTIVE", flush=True)
    return on


def _use_lnmd_bwd_v2() -> bool:
    """LN*U*dropout backward v2 kernel (separated-mask path). Gin default on
    (yambda_5b.gin item (14)); absent or DLRM_HSTU_FP8_LNMD_BWD_V2=0 keeps
    _ln_mul_dropout_bwd_dx_du_rng."""
    return os.environ.get("DLRM_HSTU_FP8_LNMD_BWD_V2", "0") != "0"


def _use_lnmd_fwd_v2(N: int, needs_sigmoid: bool = True) -> bool:
    """LN*U*dropout forward v2 kernel (separated-mask path, AMD only). Gin
    default on (yambda_5b.gin item (20)); absent or "0" = the old kernel. Used
    for N >= 2^20 rows when the kernel computes sigmoid(u) (silu/sigmoid
    activation or silu_u with concat_u): below 2^20 rows the old kernel is
    already at the copy-bandwidth floor, and without sigmoid v2 has nothing
    to save and its rstd is not bitwise equal to the old kernel's.
    DLRM_HSTU_FP8_LNMD_FWD_V2=0 restores _ln_mul_dropout_fwd_rng everywhere."""
    return (
        needs_sigmoid
        and N >= (1 << 20)
        and torch.version.hip is not None
        and os.environ.get("DLRM_HSTU_FP8_LNMD_FWD_V2", "0") != "0"
    )


# DLRM_HSTU_FP8_LNMD_FWD_V3, read once at the first call (the gin env
# bootstrap sets it after import).
_LNMD_FWD_V3_ENABLED: Optional[bool] = None
# Persistent grid of _ln_mul_dropout_fwd_rng_v3: programs per CU.
_LNMD_FWD_V3_GRID_MULT = 64


def _use_lnmd_fwd_v3() -> bool:
    """Only consulted where v2 would run. DLRM_HSTU_FP8_LNMD_FWD_V3 (gin default
    on) runs _ln_mul_dropout_fwd_rng_v3 (same per-row math, hand-pipelined row
    loop, bitwise equal outputs) instead of v2; =0 restores the v2 launch.
    Read once, at the first call."""
    global _LNMD_FWD_V3_ENABLED
    if _LNMD_FWD_V3_ENABLED is None:
        _LNMD_FWD_V3_ENABLED = (
            os.environ.get("DLRM_HSTU_FP8_LNMD_FWD_V3", "0") != "0"
        )
        if _LNMD_FWD_V3_ENABLED:
            # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
            arm_print_all_ranks(
                "armed OK: LN*U*dropout fwd v3 ACTIVE "
                "(DLRM_HSTU_FP8_LNMD_FWD_V3)"
            )
    return _LNMD_FWD_V3_ENABLED


# DLRM_HSTU_FP8_LNMD_FWD_MASKGEN (item (28)(e)), read once at the first call (the
# gin env bootstrap sets it after import).
_LNMD_FWD_MASKGEN_ENABLED: Optional[bool] = None
_LNMD_FWD_MASKGEN_ARMED_LOGGED = False
_LNMD_FWD_MASKGEN_ARCH_LOGGED = False
# Philox counters of _ln_mul_dropout_fwd_maskgen (as of _generate_random_mask16):
# 8-row group g * NUM_MASKS * BLOCK_D (= g * 1536) must fit int32. That bounds
# the counter base; the full counters g * 1536 + plane * 512 + col all fit only
# for N <= 11,184,808, and at N = 11,184,809 the last group's upper-plane
# counters wrap exactly as the generator's do (the mask stays byte-identical).
_LNMD_FWD_MASKGEN_MAX_N = 11_184_810


@functools.lru_cache(maxsize=16)
def _gcn_arch_name(device_index: int) -> str:
    props = torch.cuda.get_device_properties(device_index)
    return str(getattr(props, "gcnArchName", "") or "")


def _use_lnmd_fwd_maskgen(
    x: torch.Tensor,
    u: torch.Tensor,
    N: int,
    D: int,
    BLOCK_D: int,
    dropout_ratio: float,
    training: bool,
    silu_u: bool,
    concat_u: bool,
    concat_x: bool,
    mul_u_activation_type: str,
    rng_row_indices: torch.Tensor,
    output_fp8: bool,
) -> bool:
    """DLRM_HSTU_FP8_LNMD_FWD_MASKGEN (gin default on, item (28)(e)): run
    _ln_mul_dropout_fwd_maskgen (one launch, byte-identical int8 mask / e4m3 y /
    mean / rstd) instead of _generate_random_mask16 + _ln_mul_dropout_fwd_rng_v3
    (or _v2). Arms only where today's code would run exactly that pair for the
    configuration the kernel implements: the separated-RNG training branch with
    the full-row generator (no rng_row_indices), mask16, item (20b)'s v2
    condition, concat_u and concat_x, silu_u either way (the STU layers pass
    True; SILU_U constexpr), 'silu' activation, e4m3 output, D == BLOCK_D ==
    512, unit column strides, bf16 x/u, 1 <= K16 <= 65535, N < 11,184,810
    (int32 Philox counter base; see _LNMD_FWD_MASKGEN_MAX_N) and a gfx950
    device (the kernel's asm). Anything else returns False and the caller runs
    today's code unchanged; =0 (or absent) restores generator + v3 / v2
    everywhere (any other value arms). This gate does not consult
    _use_lnmd_fwd_v3: with DLRM_HSTU_FP8_LNMD_FWD_V3=0 it still arms and runs
    v3's row arithmetic. y/mean/rstd equal v3's bytes, and v2's for a
    16-byte-aligned x (every production call). The env var is read once, at
    the first call."""
    global _LNMD_FWD_MASKGEN_ENABLED, _LNMD_FWD_MASKGEN_ARMED_LOGGED
    global _LNMD_FWD_MASKGEN_ARCH_LOGGED
    if _LNMD_FWD_MASKGEN_ENABLED is None:
        _LNMD_FWD_MASKGEN_ENABLED = (
            os.environ.get("DLRM_HSTU_FP8_LNMD_FWD_MASKGEN", "0") != "0"
        )
    if not _LNMD_FWD_MASKGEN_ENABLED:
        return False
    arch = _gcn_arch_name(
        x.device.index if x.device.index is not None else torch.cuda.current_device()
    )
    if not arch.startswith("gfx950"):
        if not _LNMD_FWD_MASKGEN_ARCH_LOGGED:
            _LNMD_FWD_MASKGEN_ARCH_LOGGED = True
            # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
            arm_print_all_ranks(
                "LN*U*dropout fwd maskgen not armed: arch "
                f"{arch or '<unknown>'} is not gfx950 "
                "(DLRM_HSTU_FP8_LNMD_FWD_MASKGEN, item (28)(e)); "
                "_generate_random_mask16 + v3/v2 kept"
            )
        return False
    if not (
        training
        and not FUSE_OUTPUT_LN_RNG_BLACKWELL
        and use_separated_rng_ln_mul_dropout()
        and rng_row_indices.numel() == 0
        and concat_u
        and concat_x
        and mul_u_activation_type == "silu"
        and output_fp8
        and D == BLOCK_D == 512
        and x.stride(1) == 1
        and u.stride(1) == 1
        and x.dtype == torch.bfloat16
        and u.dtype == torch.bfloat16
        and 1 <= _mask_keep_min16(dropout_ratio) <= 65535
        and N < _LNMD_FWD_MASKGEN_MAX_N
        and _use_mask16()
        and _use_lnmd_fwd_v2(
            N,
            mul_u_activation_type in ("silu", "sigmoid") or (concat_u and silu_u),
        )
    ):
        return False
    if not _LNMD_FWD_MASKGEN_ARMED_LOGGED:
        _LNMD_FWD_MASKGEN_ARMED_LOGGED = True
        # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
        arm_print_all_ranks(
            "armed OK: LN*U*dropout fwd maskgen ACTIVE "
            "(DLRM_HSTU_FP8_LNMD_FWD_MASKGEN, item (28)(e)): one "
            "_ln_mul_dropout_fwd_maskgen launch replaces "
            "_generate_random_mask16 + _ln_mul_dropout_fwd_rng_v3/_v2, "
            "bitwise (same int8 mask, e4m3 y, mean, rstd)"
        )
    return True


@functools.lru_cache(maxsize=16)
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


_MASK16_REPORTED = False


def _mask_keep_min16(dropout_ratio: float) -> int:
    """Smallest 16-bit value that is kept: P(drop) = round(p * 65536) / 65536."""
    return min(max(int(round(float(dropout_ratio) * 65536.0)), 0), 65536)


def _create_dropout_mask(
    N: int,
    D: int,
    BLOCK_D: int,
    concat_u: bool,
    concat_x: bool,
    dropout_ratio: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Create dropout mask tensor for layer norm mul dropout.

    Args:
        N: Number of rows
        D: Feature dimension
        BLOCK_D: Block size for D dimension
        concat_u: Whether to concatenate u
        concat_x: Whether to concatenate x
        dropout_ratio: Dropout ratio
        seed: Random seed
        device: Device to create tensor on

    Returns:
        random_mask: (N, D) int8 tensor. Mask j at bit j.

    Bit layout: y = val & 1, x = val & 2, u = val & 4.
    """
    num_masks = 1 + int(concat_u) + int(concat_x)
    # Torch uses 1 byte for bool internally, same as int8, so always use int8.
    random_mask = torch.empty([N, D], dtype=torch.int8, device=device)
    if _use_mask16():
        _generate_random_mask16[(triton.cdiv(N, 8),)](
            random_mask,
            N,
            _mask_keep_min16(dropout_ratio),
            seed,
            D,  # pyre-ignore[6]
            random_mask.stride(0),  # pyre-ignore[6]
            BLOCK_D,  # pyre-fixme[6]: Triton constexpr param
            num_masks,  # pyre-ignore[6]: NUM_MASKS constexpr
        )
        return random_mask
    _generate_random_mask[(triton.cdiv(N, 4),)](
        random_mask,
        N,
        _mask_keep_above(dropout_ratio),
        seed,
        D,  # pyre-ignore[6]
        random_mask.stride(0),  # pyre-ignore[6]
        BLOCK_D,  # pyre-fixme[6]: Triton constexpr param
        num_masks,  # pyre-ignore[6]: NUM_MASKS constexpr
    )
    return random_mask


def _create_indexed_dropout_mask(
    row_indices: torch.Tensor,
    N: int,
    D: int,
    BLOCK_D: int,
    concat_u: bool,
    concat_x: bool,
    dropout_ratio: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Create compact masks identical to selected rows of a full mask."""
    num_masks = 1 + int(concat_u) + int(concat_x)
    random_mask = torch.empty([N, D], dtype=torch.int8, device=device)
    if _use_mask16():
        _generate_indexed_random_mask16[(N,)](
            random_mask,
            row_indices,
            N,
            _mask_keep_min16(dropout_ratio),
            seed,
            D,  # pyre-ignore[6]
            random_mask.stride(0),  # pyre-ignore[6]
            BLOCK_D,  # pyre-fixme[6]: Triton constexpr param
            num_masks,  # pyre-ignore[6]: NUM_MASKS constexpr
        )
        return random_mask
    _generate_indexed_random_mask[(N,)](
        random_mask,
        row_indices,
        N,
        _mask_keep_above(dropout_ratio),
        seed,
        D,  # pyre-ignore[6]
        random_mask.stride(0),  # pyre-ignore[6]
        BLOCK_D,  # pyre-fixme[6]: Triton constexpr param
        num_masks,  # pyre-ignore[6]: NUM_MASKS constexpr
    )
    return random_mask


def _eval_lnmd_fwd_v3(
    N: int,
    D: int,
    BLOCK_D: int,
    silu_u: bool,
    concat_u: bool,
    mul_u_activation_type: str,
) -> bool:
    """DLRM_HSTU_FP8_EVAL_LNMD_V3 (ops/eval_lnmd_v3.py; default OFF, a declared
    eval-only numerics change): whether a training=False call runs
    _ln_mul_dropout_fwd_rng_v3 with TRAINING=False instead of the legacy
    per-row _ln_mul_dropout_fwd. Called only with training=False. The
    eligibility is the one the training branch requires to reach its v3
    launch: not FUSE_OUTPUT_LN_RNG_BLACKWELL, a separated-RNG GPU, item
    (20)(b)'s _use_lnmd_fwd_v2 condition, BLOCK_D == D and item (28)(c)'s
    _use_lnmd_fwd_v3 (item (28)(e) maskgen is training-only: it generates the
    dropout mask). With the switch off nothing after the env read runs."""
    if not eval_lnmd_v3_enabled():
        return False
    if (
        not FUSE_OUTPUT_LN_RNG_BLACKWELL
        and use_separated_rng_ln_mul_dropout()
        and _use_lnmd_fwd_v2(
            N,
            mul_u_activation_type in ("silu", "sigmoid") or (concat_u and silu_u),
        )
        and BLOCK_D == D
        and _use_lnmd_fwd_v3()
    ):
        log_eval_lnmd_v3_armed()
        return True
    # Reported once when v3 cannot run on this configuration at all. A call
    # refused only by its shape (N < 2^20 rows, no sigmoid activation, BLOCK_D
    # != D) takes the legacy kernel silently, as its training forward keeps
    # the non-v3 launches.
    if FUSE_OUTPUT_LN_RNG_BLACKWELL:
        log_eval_lnmd_v3_ignored("FUSE_OUTPUT_LN_RNG_BLACKWELL is set")
    elif not use_separated_rng_ln_mul_dropout():
        log_eval_lnmd_v3_ignored("not a separated-RNG GPU (gfx950 / sm_100)")
    elif not _use_lnmd_fwd_v2(1 << 20, True):
        log_eval_lnmd_v3_ignored("DLRM_HSTU_FP8_LNMD_FWD_V2 off or not ROCm")
    elif not _use_lnmd_fwd_v3():
        log_eval_lnmd_v3_ignored("DLRM_HSTU_FP8_LNMD_FWD_V3 off")
    return False


@maybe_register_custom_op(
    "generative_recommenders::_triton_layer_norm_mul_dropout_fwd_impl", mutates_args=()
)
def _triton_layer_norm_mul_dropout_fwd_impl(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool,
    concat_u: bool,
    concat_x: bool,
    mul_u_activation_type: str,
    seed: int,
    rng_row_indices: torch.Tensor,
    output_fp8: bool,
    inv_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal implementation that returns only tensors for custom_op compatibility.

    Returns (y, mean, rstd, random_mask) where random_mask is empty when not used.
    """
    N, D = x.shape

    output_dtype = torch.float8_e4m3fn if output_fp8 else x.dtype
    if concat_u and concat_x:
        y = torch.empty((N, 3 * D), dtype=output_dtype, device=x.device)
    elif concat_u:
        y = torch.empty((N, 2 * D), dtype=output_dtype, device=x.device)
    elif concat_x:
        y = torch.empty((N, 2 * D), dtype=output_dtype, device=x.device)
    else:
        y = torch.empty((N, D), dtype=output_dtype, device=x.device)
    mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    if N == 0:
        return y, mean, rstd, torch.empty(0, dtype=x.dtype, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    num_warps: int = min(max(BLOCK_D // 256, 1), 8)
    random_mask: torch.Tensor = torch.empty(0, dtype=x.dtype, device=x.device)
    # Separating RNG from the ln_mul_dropout kernel lets us batch multiple rows per
    # program (autotuned _ln_mul_dropout_fwd_rng) and reuse the precomputed mask in the
    # backward, instead of launching one program per row with fused RNG. This is a large
    # win on Blackwell (sm_100) and AMD MI350 (gfx950); other GPUs keep the fused path.
    # Extended to support concat_u + concat_x for mask reuse optimization
    if (
        not FUSE_OUTPUT_LN_RNG_BLACKWELL
        and use_separated_rng_ln_mul_dropout()
        and training
    ):
        if _use_lnmd_fwd_maskgen(
            x,
            u,
            N,
            D,
            BLOCK_D,
            dropout_ratio,
            training,
            silu_u,
            concat_u,
            concat_x,
            mul_u_activation_type,
            rng_row_indices,
            output_fp8,
        ):
            # Item (28)(e): one launch writes the int8 mask (the bytes
            # _create_dropout_mask's _generate_random_mask16 would write, from
            # the same seed) and y/mean/rstd (the bytes v3/v2 would write).
            random_mask = _nib if (_nib := _lnmd_fwd_maskgen_nib_try(x, u, y, weight, bias, mean, rstd, N, D, eps, dropout_ratio, inv_scale, seed, BLOCK_D, concat_u, concat_x, mul_u_activation_type, output_fp8, silu_u)) is not None else torch.empty([N, D], dtype=torch.int8, device=x.device)  # noqa: E501  NIB (module end), default off: its uint8 [N, 256] mask if it armed and launched, else today's int8 mask
            keep_min16 = _mask_keep_min16(dropout_ratio)
            kk2 = (keep_min16 - 1) * 0x10001  # both u16 halves, as int32
            if kk2 >= 2**31:
                kk2 -= 2**32
            if _nib is not None or _lnmd_fwd_maskgen_pk_try(x, u, y, weight, bias, mean, rstd, random_mask, N, D, eps, dropout_ratio, inv_scale, keep_min16, kk2, seed, BLOCK_D, concat_u, concat_x, mul_u_activation_type, output_fp8, silu_u): return y, mean, rstd, random_mask  # noqa: E501,E701  NIB / MGPK, default off (module end); was: pyre-ignore[28]
            _ln_mul_dropout_fwd_maskgen[(triton.cdiv(N, 8),)](
                x,
                u,
                y,
                weight,
                bias,
                mean,
                rstd,
                random_mask,
                N,
                D,
                eps,
                dropout_ratio,
                x.stride(0),
                u.stride(0),
                y.stride(0),
                random_mask.stride(0),
                float(inv_scale),
                keep_min16,
                kk2,
                seed,
                BLOCK_D=BLOCK_D,
                NUM_MASKS=1 + int(concat_u) + int(concat_x),
                MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
                OUTPUT_FP8=output_fp8,
                SILU_U=silu_u,
                num_warps=1,
            )
            return y, mean, rstd, random_mask
        if rng_row_indices.numel() > 0:
            random_mask = _create_indexed_dropout_mask(
                row_indices=rng_row_indices,
                N=N,
                D=D,
                BLOCK_D=BLOCK_D,
                concat_u=concat_u,
                concat_x=concat_x,
                dropout_ratio=dropout_ratio,
                seed=seed,
                device=x.device,
            )
        else:
            random_mask = _create_dropout_mask(
                N=N,
                D=D,
                BLOCK_D=BLOCK_D,
                concat_u=concat_u,
                concat_x=concat_x,
                dropout_ratio=dropout_ratio,
                seed=seed,
                device=x.device,
            )

        if _use_lnmd_fwd_v2(
            N,
            mul_u_activation_type in ("silu", "sigmoid") or (concat_u and silu_u),
        ):
            if BLOCK_D == D and _use_lnmd_fwd_v3():
                # Persistent: 64 one-wave programs per CU, one row per loop
                # step, next row's x/u prefetched by hand.
                # pyre-ignore[28]
                _ln_mul_dropout_fwd_rng_v3[(min(N, _LNMD_FWD_V3_GRID_MULT * _num_cus(x.device.index if x.device.index is not None else torch.cuda.current_device())),)](
                    x,
                    u,
                    y,
                    weight,
                    bias,
                    mean,
                    rstd,
                    random_mask,
                    N,
                    D,
                    eps,
                    dropout_ratio,
                    x.stride(0),
                    u.stride(0),
                    y.stride(0),
                    random_mask.stride(0),
                    float(inv_scale),
                    SILU_U=silu_u,
                    BLOCK_D=BLOCK_D,
                    TRAINING=training,
                    CONCAT_U=concat_u,
                    CONCAT_X=concat_x,
                    MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
                    OUTPUT_FP8=output_fp8,
                    PF2=False,
                    PF_MASK=False,
                    EVEN_D=BLOCK_D == D,
                    num_warps=1,
                )
                return y, mean, rstd, random_mask
            # Persistent: 32 one-wave programs per CU, one row per loop step.
            # pyre-ignore[28]
            _ln_mul_dropout_fwd_rng_v2[(min(N, 32 * _num_cus(x.device.index if x.device.index is not None else torch.cuda.current_device())),)](
                x,
                u,
                y,
                weight,
                bias,
                mean,
                rstd,
                random_mask,
                N,
                D,
                eps,
                dropout_ratio,
                x.stride(0),
                u.stride(0),
                y.stride(0),
                random_mask.stride(0),
                float(inv_scale),
                SILU_U=silu_u,
                BLOCK_D=BLOCK_D,
                TRAINING=training,
                CONCAT_U=concat_u,
                CONCAT_X=concat_x,
                MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
                OUTPUT_FP8=output_fp8,
                NUM_STAGES=1,
                num_warps=1,
            )
            return y, mean, rstd, random_mask

        def grid(META):
            return (triton.cdiv(N, META["BLOCK_N"]),)

        # pyre-ignore[28]
        _ln_mul_dropout_fwd_rng[grid](
            x,
            u,
            y,
            weight,
            bias,
            mean,
            rstd,
            random_mask,
            N,
            D,
            eps,
            dropout_ratio,
            x.stride(0),
            u.stride(0),
            y.stride(0),
            random_mask.stride(0),
            float(inv_scale),
            SILU_U=silu_u,
            BLOCK_D=BLOCK_D,
            TRAINING=training,
            CONCAT_U=concat_u,
            CONCAT_X=concat_x,
            MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
            OUTPUT_FP8=output_fp8,
        )

    else:
        if rng_row_indices.numel() > 0 and training:
            raise RuntimeError("indexed output dropout requires separated RNG masks")
        if not training and _eval_lnmd_fwd_v3(
            N, D, BLOCK_D, silu_u, concat_u, mul_u_activation_type
        ):
            # DLRM_HSTU_FP8_EVAL_LNMD_V3: the training branch's v3 launch with
            # TRAINING=False (no dropout step). RANDOM_MASK is never read with
            # TRAINING=False; the empty placeholder and its stride are passed
            # where the training branch passes its mask.
            # pyre-ignore[28]
            _ln_mul_dropout_fwd_rng_v3[(min(N, _LNMD_FWD_V3_GRID_MULT * _num_cus(x.device.index if x.device.index is not None else torch.cuda.current_device())),)](
                x,
                u,
                y,
                weight,
                bias,
                mean,
                rstd,
                random_mask,
                N,
                D,
                eps,
                dropout_ratio,
                x.stride(0),
                u.stride(0),
                y.stride(0),
                random_mask.stride(0),
                float(inv_scale),
                SILU_U=silu_u,
                BLOCK_D=BLOCK_D,
                TRAINING=False,
                CONCAT_U=concat_u,
                CONCAT_X=concat_x,
                MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
                OUTPUT_FP8=output_fp8,
                PF2=False,
                PF_MASK=False,
                EVEN_D=BLOCK_D == D,
                num_warps=1,
            )
            return y, mean, rstd, random_mask
        # Default path: fused RNG generation
        # Mask cannot be saved with fused RNG - it's generated inline in the kernel
        # pyre-ignore[28]
        _ln_mul_dropout_fwd[(N,)](
            x,
            u,
            y,
            weight,
            bias,
            mean,
            rstd,
            D,
            eps,
            seed,
            dropout_ratio,
            x.stride(0),
            u.stride(0),
            y.stride(0),
            float(inv_scale),
            SILU_U=silu_u,
            BLOCK_D=BLOCK_D,
            TRAINING=training,
            CONCAT_U=concat_u,
            CONCAT_X=concat_x,
            MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
            FAST_DROPOUT=COMPUTE_OUTPUT_LN_FAST_DROPOUT,
            num_warps=num_warps,
            OUTPUT_FP8=output_fp8,
        )
    return y, mean, rstd, random_mask


@_triton_layer_norm_mul_dropout_fwd_impl.register_fake
def _triton_layer_norm_mul_dropout_fwd_impl_fake(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool,
    concat_u: bool,
    concat_x: bool,
    mul_u_activation_type: str,
    seed: int,
    rng_row_indices: torch.Tensor,
    output_fp8: bool,
    inv_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake implementation for FakeTensor tracing."""
    N, D = x.shape
    output_dtype = torch.float8_e4m3fn if output_fp8 else x.dtype
    if concat_u and concat_x:
        y = torch.empty((N, 3 * D), dtype=output_dtype, device=x.device)
    elif concat_u:
        y = torch.empty((N, 2 * D), dtype=output_dtype, device=x.device)
    elif concat_x:
        y = torch.empty((N, 2 * D), dtype=output_dtype, device=x.device)
    else:
        y = torch.empty((N, D), dtype=output_dtype, device=x.device)
    mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    random_mask = torch.empty(0, dtype=x.dtype, device=x.device)
    return y, mean, rstd, random_mask


def triton_layer_norm_mul_dropout_fwd(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_u: bool = False,
    concat_x: bool = False,
    mul_u_activation_type: str = "none",
    seed: Optional[int] = None,
    rng_row_indices: Optional[torch.Tensor] = None,
    output_fp8: bool = False,
    inv_scale: float = 1.0,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, Optional[torch.Tensor]
]:  # y, mean, rstd, BLOCK_D, num_warps, seed, random_mask
    """Forward pass for layer norm + mul + dropout.

    Args:
        x: Input tensor of shape (N, D)
        u: Second input tensor of shape (N, D)
        weight: Layer norm weight of shape (D,)
        bias: Layer norm bias of shape (D,)
        eps: Layer norm epsilon
        dropout_ratio: Dropout probability
        training: Whether in training mode
        silu_u: Whether to apply SiLU to u before concatenation
        concat_u: Whether to concatenate u to output
        concat_x: Whether to concatenate x to output
        mul_u_activation_type: Activation type for u multiplication
        seed: Random seed for dropout

    Returns:
        Tuple of (y, mean, rstd, BLOCK_D, num_warps, seed, random_mask)
        - random_mask is None when using fused RNG path (non-SM100+)
        - random_mask is always returned when using separate RNG path (SM100+)
          for reuse in backward pass (avoids redundant mask generation)
    """
    assert x.dim() == 2
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    assert weight.dim() == 1
    assert bias.dim() == 1
    assert weight.numel() == D
    assert bias.numel() == D
    if rng_row_indices is not None:
        assert rng_row_indices.dim() == 1
        assert rng_row_indices.numel() == N

    if N == 0:
        D = x.shape[1]
        output_dtype = torch.float8_e4m3fn if output_fp8 else x.dtype
        if concat_u and concat_x:
            y = torch.empty((0, 3 * D), dtype=output_dtype, device=x.device)
        elif concat_u or concat_x:
            y = torch.empty((0, 2 * D), dtype=output_dtype, device=x.device)
        else:
            y = torch.empty((0, D), dtype=output_dtype, device=x.device)
        return (
            y,
            torch.empty((N,), dtype=torch.float32, device=x.device),
            torch.empty((N,), dtype=torch.float32, device=x.device),
            0,
            0,
            0,
            None,
        )

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if seed is None and training:
        # pyre-ignore[9]: torch.randint with dtype=int64 always returns int
        seed = torch.randint(low=0, high=2**62, size=(1,), dtype=torch.int64).item()
    num_warps: int = min(max(BLOCK_D // 256, 1), 8)

    # Call internal implementation
    rng_row_indices_tensor = (
        rng_row_indices
        if rng_row_indices is not None
        else torch.empty(0, dtype=torch.int64, device=x.device)
    )
    y, mean, rstd, random_mask_tensor = _triton_layer_norm_mul_dropout_fwd_impl(
        x,
        u,
        weight,
        bias,
        eps,
        dropout_ratio,
        training,
        silu_u,
        concat_u,
        concat_x,
        mul_u_activation_type,
        seed if seed is not None else 0,
        rng_row_indices_tensor,
        output_fp8,
        inv_scale,
    )

    # Convert empty tensor back to None
    random_mask: Optional[torch.Tensor] = (
        random_mask_tensor if random_mask_tensor.numel() > 0 else None
    )

    return y, mean, rstd, BLOCK_D, num_warps, seed, random_mask  # pyre-ignore[7]


@triton.jit
def _ln_mul_dropout_fwd_fp8_yt_kernel(
    X,
    U,
    Y,
    YT,
    W,
    B,
    Mean,
    Rstd,
    RANDOM_MASK,
    N,
    M_PAD,
    eps,
    dropout_ratio,
    stride_x,
    stride_u,
    stride_y,
    stride_mask,
    stride_yt,
    Y_INV_SCALE,
    D: tl.constexpr,
    SILU_U: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STATS_CHUNKED: tl.constexpr,
):
    """`_ln_mul_dropout_fwd_rng` for TRAINING, CONCAT_U, CONCAT_X, OUTPUT_FP8 with
    the compressed mask, processed in column chunks so a program can own enough
    rows (BLOCK_N) for the transposed store to write long contiguous runs. Per
    element it is the same arithmetic in the same order; it additionally stores
    the e4m3 output token-major into YT [3D, M_PAD] (zero in padded tokens) for
    the FP8 output-projection wgrad (DLRM_HSTU_FP8_BWD_OUT_WGRAD only).
    """
    block_id = tl.program_id(0)
    rows = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    rows64 = rows.to(tl.int64)
    inv_D = 1.0 / D
    # Row statistics (D is the LN width), two-pass like the default kernel.
    if STATS_CHUNKED:
        # Chunked sums keep register use flat for large BLOCK_N; the re-reads
        # of the x tile hit cache.
        acc = tl.zeros((BLOCK_N,), tl.float32)
        for c in tl.static_range(0, D, BLOCK_C):
            xc = tl.load(
                X + rows64[:, None] * stride_x + (c + tl.arange(0, BLOCK_C))[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(xc, axis=1)
        mean = acc * inv_D
        acc = tl.zeros((BLOCK_N,), tl.float32)
        for c in tl.static_range(0, D, BLOCK_C):
            xc = tl.load(
                X + rows64[:, None] * stride_x + (c + tl.arange(0, BLOCK_C))[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            xm = tl.where(row_mask[:, None], xc - mean[:, None], 0.0)
            acc += tl.sum(xm * xm, axis=1)
        var = acc * inv_D
    else:
        cols_full = tl.arange(0, D)
        x_full = tl.load(
            X + rows64[:, None] * stride_x + cols_full[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        mean = tl.sum(x_full, axis=1) * inv_D
        x_mean_full = tl.where(row_mask[:, None], x_full - mean[:, None], 0.0)
        var = tl.sum(x_mean_full * x_mean_full, axis=1) * inv_D
    tl.store(Mean + rows, mean, mask=row_mask)
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + rows, rstd, mask=row_mask)
    mean = tl.expand_dims(mean, 1)
    rstd = tl.expand_dims(rstd, 1)
    dropout_scale = 1.0 / (1.0 - dropout_ratio)
    keep_row = row_mask[:, None]
    yt_rows = rows64[None, :]
    yt_row_mask = (rows64 < M_PAD)[None, :]
    for c in tl.static_range(0, D, BLOCK_C):
        cols = c + tl.arange(0, BLOCK_C)
        x_block = tl.load(
            X + rows64[:, None] * stride_x + cols[None, :], mask=keep_row, other=0.0
        ).to(tl.float32)
        u_block = tl.load(
            U + rows64[:, None] * stride_u + cols[None, :], mask=keep_row, other=0.0
        ).to(tl.float32)
        x_mean = tl.where(keep_row, x_block - mean, 0.0)
        y = x_mean * rstd
        w = tl.load(W + cols).to(tl.float32)
        b = tl.load(B + cols).to(tl.float32)
        y = y * w[None, :] + b[None, :]
        sigmoid_u_block = tl.sigmoid(u_block)
        silu_u_block = u_block * sigmoid_u_block
        if MUL_U_ACTIVATION_TYPE == "silu":
            y = y * silu_u_block
        elif MUL_U_ACTIVATION_TYPE == "sigmoid":
            y = y * sigmoid_u_block
        else:
            y = y * u_block
        if SILU_U:
            u_block = silu_u_block
        compressed = tl.load(
            RANDOM_MASK + rows64[:, None] * stride_mask + cols[None, :],
            mask=keep_row,
            other=0,
        ).to(tl.int32)
        y_keep = (compressed & 1) != 0
        x_keep = (compressed & 2) != 0
        u_keep = (compressed & 4) != 0
        u_block = tl.where(u_keep, u_block * dropout_scale, 0.0)
        x_block = tl.where(x_keep, x_block * dropout_scale, 0.0)
        y = tl.where(y_keep, y * dropout_scale, 0.0)
        u_q = (u_block * Y_INV_SCALE).to(Y.dtype.element_ty)
        x_q = (x_block * Y_INV_SCALE).to(Y.dtype.element_ty)
        y_q = (y * Y_INV_SCALE).to(Y.dtype.element_ty)
        y_ptr = Y + rows64[:, None] * stride_y + cols[None, :]
        tl.store(y_ptr, u_q, mask=keep_row)
        tl.store(y_ptr + D, x_q, mask=keep_row)
        tl.store(y_ptr + 2 * D, y_q, mask=keep_row)
        # Token-major copy. Offsets are int64 (3D * M_PAD passes 2^31 above
        # ~1.4M tokens). Masked-out rows load 0 and every mask bit is 0 there,
        # so u/x/y are exactly +0 in the padded tokens.
        yt_c = cols.to(tl.int64)[:, None]
        tl.store(YT + yt_c * stride_yt + yt_rows, tl.trans(u_q), mask=yt_row_mask)
        tl.store(
            YT + (yt_c + D) * stride_yt + yt_rows, tl.trans(x_q), mask=yt_row_mask
        )
        tl.store(
            YT + (yt_c + 2 * D) * stride_yt + yt_rows,
            tl.trans(y_q),
            mask=yt_row_mask,
        )


# (BLOCK_N rows, BLOCK_C columns, num_warps) of the token-major-y forward launch.
_FP8_OUT_WGRAD_YT_CFG: Tuple[int, int, int] = tuple(  # pyre-ignore[9]
    int(v)
    for v in os.environ.get("DLRM_HSTU_FP8_BWD_OUT_WGRAD_YT_CFG", "64,128,8").split(",")
)


def triton_layer_norm_mul_dropout_fwd_fp8_yt(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool,
    concat_u: bool,
    concat_x: bool,
    mul_u_activation_type: str,
    seed: Optional[int],
    inv_scale: float,
    pad_to: int = 16,
    config: Optional[Tuple[int, int, int]] = None,
) -> Optional[
    Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, torch.Tensor, torch.Tensor
    ]
]:
    """FP8 `triton_layer_norm_mul_dropout_fwd` (separated-RNG path) that also
    writes the e4m3 output token-major, [3D, M_pad] zero-padded, for the FP8
    output-projection wgrad (DLRM_HSTU_FP8_BWD_OUT_WGRAD only). The row-major
    output, mean, rstd and mask use the same per-element arithmetic as the default
    launch (`_ln_mul_dropout_fwd_fp8_yt_kernel`); its row blocking differs from
    the autotuned one, which can move the last bit of mean/rstd (reduction tree),
    so the arm is not bitwise equal to the default forward. Returns the usual 7-tuple plus
    y_t, or None for any contract it does not cover (the caller falls back).
    """
    if not (
        training
        and concat_u
        and concat_x
        and not FUSE_OUTPUT_LN_RNG_BLACKWELL
        and use_separated_rng_ln_mul_dropout()
    ):
        return None
    assert x.dim() == 2
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    if N == 0:
        return None
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D or BLOCK_D != D or u.stride(1) != 1:
        return None
    if seed is None:
        # Same host RNG draw as triton_layer_norm_mul_dropout_fwd.
        # pyre-ignore[9]
        seed = torch.randint(low=0, high=2**62, size=(1,), dtype=torch.int64).item()
    num_warps: int = min(max(BLOCK_D // 256, 1), 8)
    block_n, block_c, kernel_warps = config or _FP8_OUT_WGRAD_YT_CFG
    m_pad = (N + pad_to - 1) // pad_to * pad_to
    y = torch.empty((N, 3 * D), dtype=torch.float8_e4m3fn, device=x.device)
    y_t = torch.empty((3 * D, m_pad), dtype=torch.float8_e4m3fn, device=x.device)
    mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    random_mask = _create_dropout_mask(
        N=N,
        D=D,
        BLOCK_D=BLOCK_D,
        concat_u=concat_u,
        concat_x=concat_x,
        dropout_ratio=dropout_ratio,
        seed=seed,  # pyre-ignore[6]
        device=x.device,
    )
    _ln_mul_dropout_fwd_fp8_yt_kernel[(triton.cdiv(m_pad, block_n),)](
        x,
        u,
        y,
        y_t,
        weight,
        bias,
        mean,
        rstd,
        random_mask,
        N,
        m_pad,
        eps,
        dropout_ratio,
        x.stride(0),
        u.stride(0),
        y.stride(0),
        random_mask.stride(0),
        m_pad,
        float(inv_scale),
        D=D,
        SILU_U=silu_u,
        MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
        BLOCK_N=block_n,
        BLOCK_C=block_c,
        STATS_CHUNKED=block_n > 64,
        num_warps=kernel_warps,
    )
    return y, mean, rstd, BLOCK_D, num_warps, seed, random_mask, y_t  # pyre-ignore[7]


HSTU_DUVQK_WIDTH_ATTR = "_hstu_duvqk_width"
HSTU_DUVQK_BUFFER_ATTR = "_hstu_duvqk_buffer"
# B1a (Arbor r16, DLRM_HSTU_FP8_BWD_DU_E5M2=1 only; hstu_du_e5m2.py): the
# deferred-SiLU preprocess forward also tags its u with the layer's uvqk site
# (a DuE5m2Site keyed on uvqk_weight.data_ptr()), and the compute-output
# backward tags an emitting du view with its DuE5m2Emit record.
HSTU_DUVQK_SITE_ATTR = "_hstu_duvqk_site"
HSTU_DU_E5M2_ATTR = "_hstu_du_e5m2"


def _lnmd_bwd_finalize(
    _dweight: torch.Tensor,
    _dbias: torch.Tensor,
    dweight: torch.Tensor,
    dbias: torch.Tensor,
    n_rows: int,
    D: int,
    sms: int,
) -> None:
    """dweight/dbias = column sums of the [n_rows, D] fp32 partials (_ln_mul_dropout_bwd_dwdb, autotuned on D).

    Shared by the LN*U*dropout backward (n_rows = its tile_num programs) and
    the fused L11 path (n_rows = P per-wave partial rows, gin item (39)), so
    both launch the same finalize code with the same BLOCK_D rule."""

    def grid(META):
        return (triton.cdiv(D, META["BLOCK_D"]),)

    blocks = triton.next_power_of_2(sms * 4)
    BLOCK_D_bwd = triton.next_power_of_2(triton.cdiv(D, blocks))
    BLOCK_D_bwd = min(max(BLOCK_D_bwd, 4), 128)
    _ln_mul_dropout_bwd_dwdb[grid](
        _dweight,
        _dbias,
        dweight,
        dbias,
        n_rows,
        D,
        BLOCK_D=BLOCK_D_bwd,
    )


def _tag_duvqk_buffer(du: torch.Tensor, du_row_width: int) -> None:
    """Hand the wider d(uvqk) buffer to the preprocess backward (see _alloc_du)."""
    if du_row_width > du.shape[1] and du.stride(0) == du_row_width:
        setattr(
            du,
            HSTU_DUVQK_BUFFER_ATTR,
            du.as_strided(
                (du.shape[0], du_row_width),
                (du_row_width, 1),
                du.storage_offset(),
            ),
        )


def _alloc_du(u: torch.Tensor, du_row_width: int) -> torch.Tensor:
    """Allocate the u gradient, optionally as the leading columns of a wider row.

    With du_row_width > u.shape[1] the gradient is returned as the first
    u.shape[1] columns of a fresh (N, du_row_width) buffer. The deferred-SiLU
    preprocess backward (triton_hstu_preprocess_and_attention) then adopts that
    buffer as its d(uvqk) scratch instead of allocating one and copying du into
    it. Only the leading columns are written here; the rest is scratch that the
    adopter fills. The kernels take du.stride(0), so values are unchanged.
    """
    if du_row_width > u.shape[1] and u.dim() == 2 and u.shape[0] > 0:
        return torch.empty(
            (u.shape[0], du_row_width), dtype=u.dtype, device=u.device
        )[:, : u.shape[1]]
    return torch.empty_like(u)


@maybe_register_custom_op(
    "generative_recommenders::_triton_layer_norm_mul_dropout_bwd_impl", mutates_args=()
)
def _triton_layer_norm_mul_dropout_bwd_impl(
    dy: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    num_warps: int,
    eps: float,
    training: bool,
    dropout_ratio: float,
    seed: int,
    silu_u: bool,
    concat_u: bool,
    concat_x: bool,
    mul_u_activation_type: str,
    compute_y: bool,
    random_mask: torch.Tensor,
    du_row_width: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal implementation that returns only tensors for custom_op compatibility.

    When compute_y is False, y is returned as an empty tensor.
    random_mask with numel() == 0 means no mask (fused RNG path).
    """
    N, D = x.shape; random_mask = lnmd_mask_nib.as_int8(random_mask, "_triton_layer_norm_mul_dropout_bwd_impl (Triton LN*U*dropout bwd v2 / v1)")  # noqa: E501,E702  NIB (module end): exact int8 unpack; int8 unchanged
    if compute_y:
        if concat_u and concat_x:
            y = torch.empty((N, 3 * D), dtype=x.dtype, device=x.device)
        elif concat_u:
            y = torch.empty((N, 2 * D), dtype=x.dtype, device=x.device)
        elif concat_x:
            y = torch.empty((N, 2 * D), dtype=x.dtype, device=x.device)
        else:
            y = torch.empty_like(x)
    else:
        y = torch.empty(0, dtype=x.dtype, device=x.device)

    if N == 0:
        return (
            torch.zeros_like(x),
            torch.zeros_like(u),
            torch.zeros((D,), dtype=weight.dtype, device=x.device),
            torch.zeros((D,), dtype=weight.dtype, device=x.device),
            y,
        )
    dx = torch.empty_like(x)
    du = _alloc_du(u, du_row_width)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    tile_num = max(1, min(sms * 64, N // 4))
    _dweight = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
    dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)

    # Use separated RNG when random_mask is provided (from forward pass on SM100+ path)
    has_random_mask = random_mask.numel() > 0
    if has_random_mask and _use_lnmd_bwd_v2():
        # pyre-ignore[28]
        _ln_mul_dropout_bwd_dx_du_rng_v2[(tile_num,)](
            dx,
            du,
            dy,
            _dweight,
            _dbias,
            x,
            u,
            y if compute_y else None,
            weight,
            bias,
            mean,
            rstd,
            random_mask,
            dx.stride(0),
            du.stride(0),
            dy.stride(0),
            x.stride(0),
            u.stride(0),
            y.stride(0) if compute_y else 0,
            random_mask.stride(0),
            D,
            eps,
            dropout_ratio,
            N,
            SILU_U=silu_u,
            BLOCK_D=BLOCK_D,
            TRAINING=training,
            CONCAT_U=concat_u,
            CONCAT_X=concat_x,
            MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
            COMPUTE_Y=compute_y,
            NUM_STAGES=1,
            CONTIG=False,
            EVEN_D=BLOCK_D == D,
            SCM=".cs",
            LEV="",
            num_warps=max(1, num_warps // 2),
        )
    elif has_random_mask:
        # pyre-ignore[28]
        _ln_mul_dropout_bwd_dx_du_rng[(tile_num,)](
            dx,
            du,
            dy,
            _dweight,
            _dbias,
            x,
            u,
            y if compute_y else None,
            weight,
            bias,
            mean,
            rstd,
            random_mask,
            dx.stride(0),
            du.stride(0),
            dy.stride(0),
            x.stride(0),
            u.stride(0),
            y.stride(0) if compute_y else 0,
            random_mask.stride(0),
            D,
            eps,
            dropout_ratio,
            N=N,
            SILU_U=silu_u,
            BLOCK_D=BLOCK_D,
            TRAINING=training,
            CONCAT_U=concat_u,
            CONCAT_X=concat_x,
            MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
            COMPUTE_Y=compute_y,
            num_warps=num_warps,
        )

    else:
        # pyre-ignore[28]
        _ln_mul_dropout_bwd_dx_du[(tile_num,)](
            dx,
            du,
            dy,
            _dweight,
            _dbias,
            x,
            u,
            y if compute_y else None,
            weight,
            bias,
            mean,
            rstd,
            dx.stride(0),
            du.stride(0),
            dy.stride(0),
            x.stride(0),
            u.stride(0),
            y.stride(0) if compute_y else 0,
            D,
            eps,
            seed,
            dropout_ratio,
            N=N,
            SILU_U=silu_u,
            BLOCK_D=BLOCK_D,
            TRAINING=training,
            CONCAT_U=concat_u,
            CONCAT_X=concat_x,
            MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
            COMPUTE_Y=compute_y,
            FAST_DROPOUT=COMPUTE_OUTPUT_LN_FAST_DROPOUT,
            num_warps=num_warps,
        )

    _lnmd_bwd_finalize(_dweight, _dbias, dweight, dbias, tile_num, D, sms)
    return dx, du, dweight, dbias, y


@_triton_layer_norm_mul_dropout_bwd_impl.register_fake
def _triton_layer_norm_mul_dropout_bwd_impl_fake(
    dy: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    num_warps: int,
    eps: float,
    training: bool,
    dropout_ratio: float,
    seed: int,
    silu_u: bool,
    concat_u: bool,
    concat_x: bool,
    mul_u_activation_type: str,
    compute_y: bool,
    random_mask: torch.Tensor,
    du_row_width: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake implementation for FakeTensor tracing."""
    N, D = x.shape
    dx = torch.empty_like(x)
    du = _alloc_du(u, du_row_width)
    dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
    dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
    if compute_y:
        if concat_u and concat_x:
            y = torch.empty((N, 3 * D), dtype=x.dtype, device=x.device)
        elif concat_u:
            y = torch.empty((N, 2 * D), dtype=x.dtype, device=x.device)
        elif concat_x:
            y = torch.empty((N, 2 * D), dtype=x.dtype, device=x.device)
        else:
            y = torch.empty_like(x)
    else:
        y = torch.empty(0, dtype=x.dtype, device=x.device)
    return dx, du, dweight, dbias, y


def triton_layer_norm_mul_dropout_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    num_warps: int,
    eps: float,
    training: bool,
    dropout_ratio: float,
    seed: Optional[int] = None,
    silu_u: bool = False,
    concat_u: bool = False,
    concat_x: bool = False,
    mul_u_activation_type: str = "none",
    compute_y: bool = False,
    random_mask: Optional[torch.Tensor] = None,
    du_row_width: int = 0,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]:
    N, D = x.shape

    # Use empty tensor as sentinel for no random_mask
    random_mask_tensor = (
        random_mask
        if random_mask is not None
        else torch.empty(0, dtype=x.dtype, device=x.device)
    )

    dx, du, dweight, dbias, y_tensor = _triton_layer_norm_mul_dropout_bwd_impl(
        dy,
        x,
        u,
        weight,
        bias,
        mean,
        rstd,
        BLOCK_D,
        num_warps,
        eps,
        training,
        dropout_ratio,
        seed if seed is not None else 0,
        silu_u,
        concat_u,
        concat_x,
        mul_u_activation_type,
        compute_y,
        random_mask_tensor,
        du_row_width,
    )

    # Convert empty tensor back to None
    y: Optional[torch.Tensor] = y_tensor if compute_y else None
    return dx, du, dweight, dbias, y


def _du_e5m2_plan(
    ctx,  # pyre-ignore[2]
    site: "hstu_du_e5m2.DuE5m2Site",
    dy: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    random_mask: Optional[torch.Tensor],
    du_row_width: int,
    compute_y: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, Tuple[object, int]]]:
    """B1a eligibility, decided BEFORE the bf16 du store is given up.

    (inv_scale, cast_scale snapshot, scale generation) when the LN*U*dropout
    backward of this ordinary-layer call may emit e5m2 du, else None (the
    shipped v2 path; one 'du_e5m2 NOT used (<reason>)' line per reason).
    `site` exists only with DLRM_HSTU_FP8_BWD_DU_E5M2=1 and when the
    preprocess forward planned the fp8t uvqk wgrad for this call.
    """
    bn = bwd_fp8._COLSUM_BLOCK_N
    reason: Optional[Tuple[str, str]] = None
    latched = hstu_du_e5m2.broken()
    if latched is not None:
        reason = ("latched", f"latched off after: {latched}")
    elif getattr(site, "consumed", False):
        # Review fix (iter 28): a retained-graph re-backward of this call. Its
        # preprocess backward recomputes the e4m3 Q/K/V (the first backward
        # released them), so the fp8t wgrad cannot consume an emission.
        reason = ("retained", "site already consumed (retained-graph re-backward)")
    elif not bwd_fp8.du_producer_enabled():
        reason = ("uvqk", "FP8 uvqk backward (e5m2, fp8t wgrad) not enabled")
    elif ctx.group_norm:
        reason = ("group_norm", "group_norm")
    elif not ctx.training:
        reason = ("training", "not training / dropout 0")
    elif not (ctx.concat_u and ctx.concat_x):
        reason = ("concat", "needs concat_u and concat_x")
    elif ctx.mul_u_activation_type != "silu":
        reason = ("act", f"mul_u_activation_type={ctx.mul_u_activation_type!r}")
    elif compute_y:
        reason = ("compute_y", "bf16 y recompute requested (compute_y)")
    elif random_mask is None or not _use_lnmd_bwd_v2() or not _use_mask16():
        reason = ("mask", "needs the mask16 random mask and LNMD bwd v2")
    elif torch.is_anomaly_enabled():
        reason = ("anomaly", "autograd anomaly mode reads the returned du")
    elif hstu_du_e5m2.diagnostics_on():
        reason = ("diag", "uvqk backward diagnostics read the BF16 dz")
    elif not (
        x.dim() == 2
        and u.dim() == 2
        and dy.dim() == 2
        and x.shape[1] == 512
        and ctx.BLOCK_D == x.shape[1]
        and du_row_width == 4 * x.shape[1]
        and x.shape[1] % bn == 0
        and du_row_width % bn == 0
        and x.shape[0] > 0
        and tuple(u.shape) == tuple(x.shape)
        and tuple(dy.shape) == (x.shape[0], 3 * x.shape[1])
        and x.dtype == torch.bfloat16
        and u.dtype == torch.bfloat16
        and dy.dtype == torch.bfloat16
        and x.stride(1) == 1
        and u.stride(1) == 1
        and dy.stride(1) == 1
        and (random_mask.dtype == torch.int8 or lnmd_mask_nib.is_nib(random_mask))  # NIB: its layout below
        and (tuple(random_mask.shape) == tuple(x.shape) or lnmd_mask_nib.layout_ok(random_mask, x.shape[0]))  # noqa: E501  NIB (module end): B1a v3 / L11 run_e5 unpack it, DO8 reads it
        and random_mask.stride(1) == 1
    ):
        reason = (
            "layout",
            f"layout: D={x.shape[-1]} BLOCK_D={ctx.BLOCK_D} "
            f"du_row_width={du_row_width} rows={x.shape[0]} "
            f"(needs D=512=BLOCK_D, width 2048, bf16, unit column stride)",
        )
    else:
        scale = bwd_fp8.producer_inv_scale(hstu_du_e5m2.SITE, site.key)
        generation = bwd_fp8.producer_generation(hstu_du_e5m2.SITE, site.key)
        if scale is not None and generation is not None:
            return scale[0], scale[1], generation
        reason = ("unpublished", "uvqk_projection site not published yet (warm-up)")
    hstu_du_e5m2.log_not_used(
        reason[0], reason[1], "shipped LNMD bwd v2 + bf16 du + cast"
    )
    return None


def _triton_layer_norm_mul_dropout_bwd_du_e5m2(
    dy: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    num_warps: int,
    dropout_ratio: float,
    silu_u: bool,
    random_mask: torch.Tensor,
    du_row_width: int,
    site_key: int,
    inv_scale: torch.Tensor,
    cast_scale: torch.Tensor,
    generation: Optional[Tuple[object, int]],
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "hstu_du_e5m2.DuE5m2Emit"
]:
    """B1a host side of `_ln_mul_dropout_bwd_dx_du_rng_v3` (contract checked
    by `_du_e5m2_plan`) -> (dx, du, dweight, dbias, emission record).

    `du` is `hstu_du_e5m2.placeholder_du` (N, D), a stride-0 view of one
    cached NaN element: NO bf16 du is written and NO (N, du_row_width) d(uvqk)
    buffer is allocated (review fix, iter 28: that buffer's never-written u
    block plus dz8 raised the step peak by 4.5 GiB; the preprocess backward
    allocates its d(uvqk) later, outside this window). Only the returned
    record (dz8 [Mp, du_row_width] e5m2 with the u block and its pad rows
    written, u amax, u colsum; buffer_ptr 0) may be consumed. Same tile_num,
    launch width and stock `_ln_mul_dropout_bwd_dwdb` finalize as the v2
    path, so dx, dweight and dbias are bitwise; the u colsum / amax partials
    are reduced with deterministic torch ops.
    """
    N, D = x.shape; random_mask = lnmd_mask_nib.as_int8(random_mask, "_triton_layer_norm_mul_dropout_bwd_du_e5m2 (B1a v3)")  # noqa: E501,E702  NIB (module end): exact int8 unpack; int8 unchanged
    grad_dtype, limit = bwd_fp8.grad_dtype_and_limit()
    assert grad_dtype is not None
    pad = bwd_fp8.uvqk_tm_pad()
    rows_out = (N + pad - 1) // pad * pad
    dx = torch.empty_like(x)
    du = hstu_du_e5m2.placeholder_du(u, N, D)
    dz8 = torch.empty((rows_out, du_row_width), dtype=grad_dtype, device=x.device)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    tile_num = max(1, min(sms * 64, N // 4))
    _dweight = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    partial_col = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    partial_amax = torch.empty((tile_num,), dtype=torch.float32, device=x.device)
    dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
    dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
    # pyre-ignore[28]
    _ln_mul_dropout_bwd_dx_du_rng_v3[(tile_num,)](
        dx,
        dy,
        _dweight,
        _dbias,
        x,
        u,
        weight,
        bias,
        mean,
        rstd,
        random_mask,
        dz8,
        partial_amax,
        partial_col,
        inv_scale,
        dx.stride(0),
        dy.stride(0),
        x.stride(0),
        u.stride(0),
        random_mask.stride(0),
        dz8.stride(0),
        D,
        dropout_ratio,
        N,
        rows_out,
        SILU_U=silu_u,
        BLOCK_D=BLOCK_D,
        LIMIT=limit,
        SCM=".cs",
        LEV="",
        SCM8=".cs",
        num_warps=max(1, num_warps // 2),
    )

    def grid(META):
        return (triton.cdiv(D, META["BLOCK_D"]),)

    blocks = triton.next_power_of_2(sms * 4)
    BLOCK_D_bwd = triton.next_power_of_2(triton.cdiv(D, blocks))
    BLOCK_D_bwd = min(max(BLOCK_D_bwd, 4), 128)
    _ln_mul_dropout_bwd_dwdb[grid](
        _dweight,
        _dbias,
        dweight,
        dbias,
        tile_num,
        D,
        BLOCK_D=BLOCK_D_bwd,
    )
    emit = hstu_du_e5m2.DuE5m2Emit(
        dz8=dz8,
        u_cols=(0, D),
        u_amax=partial_amax.amax(),
        u_colsum=partial_col.sum(dim=0),
        inv_scale=inv_scale,
        cast_scale=cast_scale,
        rows=N,
        width=du_row_width,
        site_key=site_key,
        generation=generation,
        buffer_ptr=0,  # no d(uvqk) buffer handed over (placeholder du)
    )
    return dx, du, dweight, dbias, emit


class LayerNormMulDropoutFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        u: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        dropout_ratio: float,
        training: bool,
        silu_u: bool = False,
        concat_ux: bool = False,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if dropout_ratio == 0.0:
            # skip dropout computation if dropout ratio is 0
            training = False
        # skipping supporting concat_u and concat_x separately here because seems like this code path is only used in v1 of hstu_linear
        concat_u, concat_x = concat_ux, concat_ux

        # Call forward function which generates and returns random_mask
        # On SM100+ path, random_mask is always returned for backward reuse
        # On fused RNG path, random_mask is None (mask generated inline in kernel)
        y, mean, rstd, BLOCK_D, num_warps, returned_seed, random_mask = (
            triton_layer_norm_mul_dropout_fwd(
                x=x,
                u=u,
                weight=weight,
                bias=bias,
                eps=eps,
                dropout_ratio=dropout_ratio,
                training=training,
                silu_u=silu_u,
                concat_u=concat_u,
                concat_x=concat_x,
                seed=seed,
            )
        )

        # Save tensors for backward pass
        # When random_mask is generated (SM100+ path), always save it for reuse
        # in backward pass. This avoids redundant _generate_random_mask execution.
        if random_mask is not None:
            ctx.save_for_backward(x, u, weight, bias, mean, rstd, random_mask)
            ctx.has_random_mask = True
        else:
            ctx.save_for_backward(x, u, weight, bias, mean, rstd)
            ctx.has_random_mask = False

        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.seed = returned_seed
        ctx.training = training
        ctx.concat_ux = concat_ux
        ctx.silu_u = silu_u
        ctx.dropout_ratio = dropout_ratio
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        # Extract saved tensors including optional random mask
        if ctx.has_random_mask:
            x, u, weight, bias, mean, rstd, random_mask = ctx.saved_tensors
        else:
            x, u, weight, bias, mean, rstd = ctx.saved_tensors
            random_mask = None

        dx, du, dweight, dbias, _ = triton_layer_norm_mul_dropout_bwd(
            dy=dy,
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            BLOCK_D=ctx.BLOCK_D,
            num_warps=ctx.num_warps,
            eps=ctx.eps,
            training=ctx.training,
            dropout_ratio=ctx.dropout_ratio,
            seed=ctx.seed,
            silu_u=ctx.silu_u,
            concat_u=ctx.concat_ux,
            concat_x=ctx.concat_ux,
            compute_y=False,
            random_mask=random_mask,  # Pass saved mask to backward
        )
        return dx, du, dweight, dbias, None, None, None, None, None, None


@triton.jit
def _group_norm_mul_dropout_fwd(
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    D,
    Heads,
    eps,
    seed,
    dropout_ratio,
    stride_x,
    stride_u,
    stride_y,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_UX: tl.constexpr,
):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, BLOCK_H)
    offsets = heads[:, None] * D + cols[None, :]
    mask_h = heads < Heads
    mask_c = cols < D
    mask = mask_c[None, :] & mask_h[:, None]

    # Compute mean
    mean = 0.0
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / D
    mean = tl.ravel(mean)

    # Compute variance
    _var = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    x_mean = tl.where(mask, x - mean[:, None], 0.0)
    _var += x_mean * x_mean
    var = tl.sum(_var, axis=1) / D
    var = tl.ravel(var)
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row * Heads + heads, mean, mask=mask_h)
    tl.store(Rstd + row * Heads + heads, rstd, mask=mask_h)

    # Normalize and apply linear transformation
    y = x_mean * rstd[:, None]  # pyre-ignore [16]
    w = tl.load(W + heads, mask=mask_h).to(tl.float32)
    b = tl.load(B + heads, mask=mask_h).to(tl.float32)
    y = y * w[:, None] + b[:, None]
    u = tl.load(U + offsets, mask=mask, other=0.0).to(tl.float32)
    if SILU_U:
        u = u * tl.sigmoid(u)
    y = y * u

    if TRAINING:
        if CONCAT_UX:
            random_offsets = row * 3 * D * Heads + offsets
            # apply dropout on u
            random_u = tl.rand(seed, random_offsets)
            u_keep = random_u > dropout_ratio
            u = tl.where(u_keep, u / (1.0 - dropout_ratio), 0.0)
            # apply dropout on x
            random_x = tl.rand(seed, random_offsets + Heads * D)
            x_keep = random_x > dropout_ratio
            x = tl.where(x_keep, x / (1.0 - dropout_ratio), 0.0)
            # apply dropout on y
            random_y = tl.rand(seed, random_offsets + 2 * Heads * D)
            y_keep = random_y > dropout_ratio
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        else:
            random_offsets = row * D * Heads + offsets
            random = tl.rand(seed, random_offsets)
            y_keep = random > dropout_ratio
            # write-back
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)

    # Write output
    if CONCAT_UX:
        tl.store(Y + offsets, u.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + Heads * D + offsets, x.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + 2 * Heads * D + offsets, y.to(Y.dtype.element_ty), mask=mask)
    else:
        tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _group_norm_mul_dropout_bwd_dx_du(
    DX,
    DU,
    DY,
    DW,
    DB,
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride_dx,
    stride_du,
    stride_dy,
    stride_x,
    stride_u,
    stride_y,
    D,
    Heads,
    eps,
    seed,
    dropout_ratio,
    SILU_U: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_UX: tl.constexpr,
    COMPUTE_Y: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    off_heads = tl.arange(0, BLOCK_H)
    mask_c = cols < D
    mask_h = off_heads < Heads
    mask = mask_c[None, :] & mask_h[:, None]
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx
    DU += row.to(tl.int64) * stride_du
    offsets = off_heads[:, None] * D + cols[None, :]

    # Load data to SRAM
    x = tl.load(X + offsets, mask=mask, other=0).to(tl.float32)
    if CONCAT_UX:
        du = tl.load(DY + offsets, mask=mask, other=0).to(tl.float32)
        dx = tl.load(DY + Heads * D + offsets, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + 2 * Heads * D + offsets, mask=mask, other=0).to(tl.float32)
    else:
        du = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        dx = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        dy = tl.load(DY + offsets, mask=mask, other=0).to(tl.float32)
    if TRAINING:
        if CONCAT_UX:
            random_offsets = row * 3 * D * Heads + offsets
            # apply dropout on du
            random_du = tl.rand(seed, random_offsets)
            du_keep = random_du > dropout_ratio
            du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
            # apply dropout on dx
            random_dx = tl.rand(seed, random_offsets + Heads * D)
            dx_keep = random_dx > dropout_ratio
            dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
            # apply dropout on dy
            random_dy = tl.rand(seed, random_offsets + 2 * Heads * D)
            dy_keep = random_dy > dropout_ratio
            dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
        else:
            random_offsets = row * D * Heads + offsets
            random = tl.rand(seed, random_offsets)
            dy_keep = random > dropout_ratio
            # write-back
            dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)

    mean = tl.load(Mean + row * Heads + off_heads)
    rstd = tl.load(Rstd + row * Heads + off_heads)

    # Compute dx
    xhat = (x - mean[:, None]) * rstd[:, None]
    w = tl.load(W + off_heads, mask=mask_h).to(tl.float32)
    b = tl.load(B + off_heads, mask=mask_h).to(tl.float32)
    u = tl.load(U + offsets, mask=mask, other=0).to(tl.float32)
    ln = xhat * w[:, None] + b[:, None]
    du += dy * ln
    if SILU_U:
        sig_u = tl.sigmoid(u)
        silu_u = u * sig_u
        du = du * sig_u * (1 + u - silu_u)
        u = silu_u
    tl.store(DU + offsets, du.to(DU.dtype.element_ty), mask=mask)
    dy = dy * u
    wdy = w[:, None] * dy
    if COMPUTE_Y:
        Y += row.to(tl.int64) * stride_y
        y = ln * u
        if TRAINING:
            if CONCAT_UX:
                u = tl.where(
                    du_keep,  # pyre-ignore [61]
                    u / (1.0 - dropout_ratio),
                    0.0,
                )
                x = tl.where(
                    dx_keep,  # pyre-ignore [61]
                    x / (1.0 - dropout_ratio),
                    0.0,
                )
                y = tl.where(
                    dy_keep,  # pyre-ignore [61]
                    y / (1.0 - dropout_ratio),
                    0.0,
                )
            else:
                y = tl.where(
                    dy_keep,  # pyre-ignore [61]
                    y / (1.0 - dropout_ratio),
                    0.0,
                )
        if CONCAT_UX:
            tl.store(Y + offsets, u.to(Y.dtype.element_ty), mask=mask)
            tl.store(Y + Heads * D + offsets, x.to(Y.dtype.element_ty), mask=mask)
            tl.store(Y + 2 * Heads * D + offsets, y.to(Y.dtype.element_ty), mask=mask)
        else:
            tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)

    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=1) / D
    c2 = tl.sum(wdy, axis=1) / D
    dx += (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
    # Write dx
    tl.store(DX + offsets, dx, mask=mask)

    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_N
    DW = DW + lock_id * Heads + off_heads
    DB = DB + lock_id * Heads + off_heads
    # Accumulate partial sums for dw/db
    partial_dw = tl.sum(dy * xhat, axis=1)
    partial_dw = tl.ravel(partial_dw)
    partial_db = tl.sum(dy, axis=1)
    partial_db = tl.ravel(partial_db)
    tl.atomic_add(
        DW,
        partial_dw,
        mask=mask_h,
        sem="relaxed",
    )
    tl.atomic_add(
        DB,
        partial_db,
        mask=mask_h,
        sem="relaxed",
    )


def triton_group_norm_mul_dropout_fwd(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_ux: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
    seed: Optional[int] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int
]:  # y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed
    assert x.dim() == 2
    assert x.shape == u.shape
    assert x.shape[1] == num_heads * linear_dim
    x = switch_to_contiguous_if_needed(x)
    u = switch_to_contiguous_if_needed(u)
    N, _ = x.shape
    assert weight.dim() == 1
    assert bias.dim() == 1
    assert weight.numel() == num_heads
    assert bias.numel() == num_heads

    if concat_ux:
        y = torch.empty((N, 3 * num_heads * linear_dim), dtype=x.dtype, device=x.device)
    else:
        y = torch.empty((N, num_heads * linear_dim), dtype=x.dtype, device=x.device)
    mean = torch.empty((N * num_heads,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N * num_heads,), dtype=torch.float32, device=x.device)
    if N == 0:
        return y, mean, rstd, 0, 0, 0, 0
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = triton.next_power_of_2(linear_dim)
    BLOCK_H: int = triton.next_power_of_2(num_heads)
    if BLOCK_D * BLOCK_H > MAX_FUSED_SIZE:
        raise RuntimeError(
            "This group norm doesn't support num_heads * linear_dim >= 64KB."
        )

    if seed is None:
        seed = torch.randint(low=0, high=2**62, size=(1,), dtype=torch.int64).item()
    num_warps: int = min(max(BLOCK_D * BLOCK_H // 256, 1), 8)
    # pyre-ignore[28]
    _group_norm_mul_dropout_fwd[(N,)](
        x,
        u,
        y,
        weight,
        bias,
        mean,
        rstd,
        linear_dim,
        num_heads,
        eps,
        seed,
        dropout_ratio,
        x.stride(0),
        u.stride(0),
        y.stride(0),
        SILU_U=silu_u,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        TRAINING=training,
        CONCAT_UX=concat_ux,
        num_warps=num_warps,
    )
    return y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed  # pyre-ignore [7]


def triton_group_norm_mul_dropout_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    BLOCK_H: int,
    num_warps: int,
    eps: float,
    training: bool,
    dropout_ratio: float,
    seed: Optional[int] = None,
    silu_u: bool = False,
    concat_ux: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
    compute_y: bool = False,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]:
    y = None
    N, dim = x.shape
    if compute_y:
        if concat_ux:
            y = torch.empty(
                (N, 3 * num_heads * linear_dim), dtype=x.dtype, device=x.device
            )
        else:
            y = torch.empty((N, num_heads * linear_dim), dtype=x.dtype, device=x.device)
    if N == 0:
        return (
            torch.zeros_like(x),
            torch.zeros_like(u),
            torch.zeros_like(weight),
            torch.zeros_like(bias),
            y,
        )
    dx = torch.empty_like(x)
    du = torch.empty_like(u)
    if dim <= 1024:
        GROUP_N = 256 * 8
    elif dim <= 4096:
        GROUP_N = 128 * 8
    elif dim <= 8192:
        GROUP_N = 96 * 8
    else:
        GROUP_N = 64 * 8
    GROUP_N = N if GROUP_N > N else GROUP_N
    _dweight = torch.zeros((GROUP_N, num_heads), dtype=torch.float32, device=x.device)
    _dbias = torch.zeros((GROUP_N, num_heads), dtype=torch.float32, device=x.device)
    dweight = torch.empty((num_heads,), dtype=weight.dtype, device=x.device)
    dbias = torch.empty((num_heads,), dtype=weight.dtype, device=x.device)
    # pyre-ignore[28]
    _group_norm_mul_dropout_bwd_dx_du[(N,)](
        dx,
        du,
        dy,
        _dweight,
        _dbias,
        x,
        u,
        y,
        weight,
        bias,
        mean,
        rstd,
        dx.stride(0),
        du.stride(0),
        dy.stride(0),
        x.stride(0),
        u.stride(0),
        y.stride(0) if compute_y else 0,  # pyre-ignore [16]
        linear_dim,
        num_heads,
        eps,
        seed,
        dropout_ratio,
        SILU_U=silu_u,
        GROUP_N=GROUP_N,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        TRAINING=training,
        CONCAT_UX=concat_ux,
        COMPUTE_Y=compute_y,
        num_warps=num_warps,
    )
    _group_norm_bwd_dwdb[(num_heads,)](
        _dweight,
        _dbias,
        dweight,
        dbias,
        GROUP_N,
    )
    return dx, du, dweight, dbias, y


def _get_bwd_dwdb_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_N in [32, 64, 128, 256]:
        for num_warps in [8, 16] + ([] if torch.ops.hip else [32]):
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs


@triton_autotune(
    configs=_get_bwd_dwdb_configs(),
    key=[],
)
@triton.jit
def _group_norm_bwd_dwdb(
    DW,
    DB,
    FINAL_DW,
    FINAL_DB,
    N,
    BLOCK_N: tl.constexpr,
):
    col = tl.program_id(0)
    num_heads = tl.num_programs(0)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    db = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        mask = rows < N
        offs = rows * num_heads + col
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + col, sum_dw.to(FINAL_DW.dtype.element_ty))
    tl.store(FINAL_DB + col, sum_db.to(FINAL_DB.dtype.element_ty))


class GroupNormMulDropoutFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        u: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        dropout_ratio: float,
        training: bool,
        silu_u: bool = False,
        concat_ux: bool = False,
        num_heads: int = 1,
        linear_dim: int = -1,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed = (
            triton_group_norm_mul_dropout_fwd(
                x=x,
                u=u,
                weight=weight,
                bias=bias,
                eps=eps,
                dropout_ratio=dropout_ratio,
                training=training,
                silu_u=silu_u,
                concat_ux=concat_ux,
                num_heads=num_heads,
                linear_dim=linear_dim,
                seed=seed,
            )
        )
        ctx.save_for_backward(x, u, weight, bias, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.BLOCK_H = BLOCK_H
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.seed = seed
        ctx.training = training
        ctx.silu_u = silu_u
        ctx.concat_ux = concat_ux
        ctx.dropout_ratio = dropout_ratio
        ctx.num_heads = num_heads
        ctx.linear_dim = linear_dim
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        x, u, weight, bias, mean, rstd = ctx.saved_tensors
        dx, du, dweight, dbias, _ = triton_group_norm_mul_dropout_bwd(
            dy=dy,
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            BLOCK_D=ctx.BLOCK_D,
            BLOCK_H=ctx.BLOCK_H,
            num_warps=ctx.num_warps,
            eps=ctx.eps,
            training=ctx.training,
            dropout_ratio=ctx.dropout_ratio,
            seed=ctx.seed,
            silu_u=ctx.silu_u,
            concat_ux=ctx.concat_ux,
            num_heads=ctx.num_heads,
            linear_dim=ctx.linear_dim,
            compute_y=False,
        )
        return (
            dx,
            du,
            dweight,
            dbias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# L11 (gin item (39)) pre-RUN_START self-check: shape and seed. M = 32,805 rows =
# 513 64-row blocks (more than one block per WG on 256 CUs, and a ragged 37-row
# last block), the production layer contract otherwise.
_L11_SELF_CHECK_M = 32_805
_L11_SELF_CHECK_SEED = 20261009


def l11_sk_tail_start(M: int) -> int:
    """First row of hipBLASLt's stream-K (SK3) tail region of the out-proj dgrad
    [M, 512] x [512, 1536]: the last (T mod 256) + 256 of the T = 6 * ceil(M/256)
    256x256 tiles, minus a 3-M-tile margin (L11-1b builder formula; rows below
    it are hipBLASLt's data-parallel tiles, where L11's dy is bitwise)."""
    mt = (M + 255) // 256
    tail_tiles = (mt * 6) % 256 + 256
    return max(0, 256 * (mt - (tail_tiles + 5) // 6 - 3))


def _l11_self_check(e5: Optional[bool] = None) -> Tuple[bool, str]:
    """L11 vs the production chain on seeded SYNTHETIC tensors (a private
    generator: no training data, no global RNG draw, no FP8 scale state), run by
    prebuild_l11 before RUN_START. Gates: du and dx bitwise vs v2 run on L11's
    own dy (same mainloop, debug store); vs the production chain (hipBLASLt dy +
    v2) du bitwise on hipBLASLt's data-parallel rows and dx mismatching in at
    most 1e-4 of its elements; dW/dB relL2 vs the chain <= 1e-3; all outputs
    finite; du columns 512..2047 of the d(uvqk) buffer untouched.

    e5 (None = when item (38)(a) B1a is enabled, i.e. when the backward can
    route a call to the e5m2-du mode): the same for the E5 instantiation of the
    same .so, against B1a's own producer _triton_layer_norm_mul_dropout_bwd_du_e5m2
    with a synthetic non-power-of-two inv_scale (no FP8 scale state is read or
    written; the records are never consumed): the e5m2 u block of dz8 incl. its
    pad rows, the u amax and dx bitwise vs B1a on L11's own dy; the u block
    bitwise vs B1a on hipBLASLt's dy on the data-parallel rows; dz8 columns
    512..2047 untouched (a raw launch into a prefilled dz8, bitwise equal to the
    entry's record); u colsum relL2 vs B1a <= 1e-5 (declared fp32 order); dW/dB
    relL2 vs the chain <= 1e-3; the record's fields as B1a's; finite."""
    dev = torch.device("cuda", torch.cuda.current_device())
    M, Dm, p = _L11_SELF_CHECK_M, 512, 0.1
    g = torch.Generator(device=dev)
    g.manual_seed(_L11_SELF_CHECK_SEED)
    g8 = (
        (torch.randn((M, Dm), generator=g, device=dev) * 1024.0)
        .clamp_(-57344.0, 57344.0)
        .to(torch.float8_e5m2)
    )
    w8 = (
        (torch.randn((1536, Dm), generator=g, device=dev) * 90.0)
        .clamp_(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )
    sa = torch.full((1,), 2.0**-19, dtype=torch.float32, device=dev)
    sb = torch.full((1,), 0.0613 / 448.0, dtype=torch.float32, device=dev)
    x = (torch.randn((M, Dm), generator=g, device=dev) * 0.5).to(torch.bfloat16)
    u = torch.randn((M, Dm), generator=g, device=dev).to(torch.bfloat16)
    wn = (1.0 + 0.05 * torch.randn((Dm,), generator=g, device=dev)).to(torch.bfloat16)
    bn = (0.05 * torch.randn((Dm,), generator=g, device=dev)).to(torch.bfloat16)
    xf = x.float()
    mean = xf.mean(dim=1)
    rstd = torch.rsqrt(xf.var(dim=1, unbiased=False) + 1e-6)
    del xf
    mask = _create_dropout_mask(
        N=M,
        D=Dm,
        BLOCK_D=Dm,
        concat_u=True,
        concat_x=True,
        dropout_ratio=p,
        seed=_L11_SELF_CHECK_SEED,
        device=dev,
    )

    def chain(dy_: torch.Tensor):  # pyre-ignore[3]
        return triton_layer_norm_mul_dropout_bwd(
            dy=dy_,
            x=x,
            u=u,
            weight=wn,
            bias=bn,
            mean=mean,
            rstd=rstd,
            BLOCK_D=Dm,
            num_warps=2,
            eps=1e-6,
            training=True,
            dropout_ratio=p,
            seed=0,
            silu_u=True,
            concat_u=True,
            concat_x=True,
            mul_u_activation_type="silu",
            compute_y=False,
            random_mask=mask,
            du_row_width=2048,
        )

    dy_h = bwd_fp8.output_projection_dgrad_gemm(g8, w8, sa, sb, torch.bfloat16)
    dx_c, du_c, dw_c, db_c, _ = chain(dy_h)
    dy_l = l11_outproj_lnmd.dy_debug(g8, w8, sa, sb)
    dx_s, du_s, dw_s, db_s, _ = chain(dy_l)
    sentinel = 0x5A5A

    def alloc_du_sentinel(u_: torch.Tensor, width: int) -> torch.Tensor:
        buf = torch.full((u_.shape[0], width), sentinel, dtype=torch.int16, device=dev)
        return buf.view(torch.bfloat16)[:, : u_.shape[1]]

    out = l11_outproj_lnmd.run(
        g8, w8, sa, sb, x, u, mask, mean, rstd, wn, bn, p, alloc_du_sentinel,
        log_active=False,
    )
    if out is None:
        return False, f"run() declined/failed ({l11_outproj_lnmd.broken()})"
    dx_l, du_l, dwp, dbp, P = out
    dw_l = torch.empty((Dm,), dtype=wn.dtype, device=dev)
    db_l = torch.empty((Dm,), dtype=wn.dtype, device=dev)
    _lnmd_bwd_finalize(
        dwp, dbp, dw_l, db_l, P, Dm,
        torch.cuda.get_device_properties(dev).multi_processor_count,
    )
    torch.cuda.synchronize()
    tail = l11_sk_tail_start(M)

    def nbits(a: torch.Tensor, b: torch.Tensor) -> int:
        return int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum())

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        a64, b64 = a.double(), b.double()
        return float((a64 - b64).norm() / b64.norm().clamp_min(1e-300))

    du_same = nbits(du_l, du_s)
    dx_same = nbits(dx_l, dx_s)
    dy_dp = nbits(dy_l[:tail], dy_h[:tail])
    du_dp = nbits(du_l[:tail], du_c[:tail])
    dx_all = nbits(dx_l, dx_c)
    rw, rb = rel(dw_l, dw_c), rel(db_l, db_c)
    finite = all(
        bool(torch.isfinite(t).all()) for t in (dx_l, du_l, dw_l, db_l, dwp, dbp)
    )
    full = du_l.as_strided((M, 2048), (2048, 1), du_l.storage_offset())
    guard = int((full[:, Dm:].view(torch.int16) != sentinel).sum())
    detail = (
        f"M={M} P={P} sk_tail_row={tail}: same-dy du/dx mism {du_same}/{dx_same}; "
        f"vs chain dy DP mism {dy_dp}, du DP mism {du_dp}, dx mism {dx_all}/{M * Dm}; "
        f"dW/dB relL2 {rw:.2e}/{rb:.2e}; finite {int(finite)}; du guard words changed {guard}"
    )
    ok = (
        du_same == 0
        and dx_same == 0
        and du_dp == 0
        and dx_all <= 1e-4 * M * Dm
        and rw <= 1e-3
        and rb <= 1e-3
        and finite
        and guard == 0
    )
    if e5 is None:
        e5 = bwd_fp8.du_producer_enabled() and hstu_du_e5m2.broken() is None
    if not e5:
        return ok, detail + "; e5 not checked (B1a off)"
    # E5 mode (B1a contract), same inputs. inv_scale: 0.7371 * 2^floor(log2(
    # 57344 / max|bf16 du|)) (not a power of two: the multiply rounds).
    grad_dtype, limit = bwd_fp8.grad_dtype_and_limit()
    pad = bwd_fp8.uvqk_tm_pad()
    rows_out = (M + pad - 1) // pad * pad
    du_amax = float(du_s.float().abs().amax())
    inv_v = 0.7371 * 2.0 ** float(np.floor(np.log2(limit / max(du_amax, 1e-30))))
    inv = torch.full((1,), inv_v, dtype=torch.float32, device=dev)
    cs = torch.full((1,), 1.0 / inv_v, dtype=torch.float32, device=dev)
    e5_kw = dict(
        x=x,
        u=u,
        weight=wn,
        bias=bn,
        mean=mean,
        rstd=rstd,
        BLOCK_D=Dm,
        num_warps=2,
        dropout_ratio=p,
        silu_u=True,
        random_mask=mask,
        du_row_width=2048,
        site_key=0,
        inv_scale=inv,
        cast_scale=cs,
        generation=None,
    )
    out_e = l11_outproj_lnmd.run_e5(
        g8, w8, sa, sb, x, u, wn, bn, mean, rstd, Dm, 2, p, True, mask, 2048, 0,
        inv, cs, None,
        log_active=False,
    )
    if out_e is None:
        return False, detail + (
            f"; e5 run_e5() declined/failed ({l11_outproj_lnmd.broken()})"
        )
    dx_e, du_e, dw_e, db_e, rec = out_e
    raw = torch.full((rows_out, 2048), 0x5A, dtype=torch.uint8, device=dev)
    l11_outproj_lnmd.launch_e5(
        g8, w8, sa, sb, x, u, mask, mean, rstd, wn, bn, p, inv, raw.view(grad_dtype)
    )
    ref_s = _triton_layer_norm_mul_dropout_bwd_du_e5m2(dy=dy_l, **e5_kw)
    ref_c = _triton_layer_norm_mul_dropout_bwd_du_e5m2(dy=dy_h, **e5_kw)
    torch.cuda.synchronize()
    u8 = torch.uint8
    z = rec.dz8.view(u8)[:, :Dm]
    z_s = ref_s[4].dz8.view(u8)[:, :Dm]
    z_c = ref_c[4].dz8.view(u8)[:, :Dm]
    e_dz_same = int((z != z_s).sum())
    e_dz_dp = int((z[:tail] != z_c[:tail]).sum())
    e_pad = int((z[M:] != 0).sum())
    e_raw = int((raw[:, :Dm] != z).sum())
    e_guard = int((raw[:, Dm:] != 0x5A).sum())
    e_amax = bool(
        torch.equal(
            rec.u_amax.reshape(1).view(torch.int32),
            ref_s[4].u_amax.reshape(1).view(torch.int32),
        )
    )
    e_dx_same = nbits(dx_e, ref_s[0])
    e_dx_all = nbits(dx_e, ref_c[0])
    e_cs = rel(rec.u_colsum, ref_s[4].u_colsum)
    e_rw, e_rb = rel(dw_e, ref_c[2]), rel(db_e, ref_c[3])
    e_fields = (
        rec.rows,
        rec.width,
        rec.rows_out,
        rec.u_cols,
        rec.buffer_ptr,
        rec.dz8.dtype,
        tuple(du_e.shape),
        du_e.stride(),
    ) == (
        ref_s[4].rows,
        ref_s[4].width,
        ref_s[4].rows_out,
        ref_s[4].u_cols,
        ref_s[4].buffer_ptr,
        ref_s[4].dz8.dtype,
        tuple(ref_s[1].shape),
        ref_s[1].stride(),
    )
    e_finite = all(
        bool(torch.isfinite(t).all())
        for t in (dx_e, dw_e, db_e, rec.u_amax, rec.u_colsum)
    )
    detail += (
        f"; e5 inv={inv_v:.6g} Mp={rows_out}: same-dy dz8 u+pad mism {e_dz_same}, "
        f"amax eq {int(e_amax)}, dx mism {e_dx_same}; vs chain dz8 DP mism "
        f"{e_dz_dp}, dx mism {e_dx_all}/{M * Dm}; pad nonzero {e_pad}; raw==rec "
        f"mism {e_raw}, guard bytes changed {e_guard}; u colsum relL2 {e_cs:.2e}; "
        f"dW/dB relL2 {e_rw:.2e}/{e_rb:.2e}; fields {int(e_fields)}; "
        f"finite {int(e_finite)}"
    )
    ok = (
        ok
        and e_dz_same == 0
        and e_amax
        and e_dx_same == 0
        and e_dz_dp == 0
        and e_dx_all <= 1e-4 * M * Dm
        and e_pad == 0
        and e_raw == 0
        and e_guard == 0
        and e_cs <= 1e-5
        and e_rw <= 1e-3
        and e_rb <= 1e-3
        and e_fields
        and e_finite
    )
    return ok, detail


def prebuild_l11() -> bool:
    """DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED (gin item (39)): build + load +
    self-check the fused kernel before RUN_START (triton_fp8.prebuild_extensions,
    gin DLRM_HSTU_FP8_PREBUILD). No-op without the knob; INERT line without the
    FP8 out-proj wgrad backward it rides on; a failure latches L11 off (logged)
    and the chain runs. Without a successful prebuild the backward never arms
    L11 (decline D11): no hipcc build or unchecked kernel inside the timed run."""
    if not l11_outproj_lnmd.requested():
        return False
    static_ok = _FP8_OUT_WGRAD and bwd_fp8.scale_controller_enabled()
    if static_ok and not _use_lnmd_bwd_v2():
        l11_outproj_lnmd.note_decline(
            "D7 DLRM_HSTU_FP8_LNMD_BWD_V2 off (reference consumer is v2)"
        )
        return False
    return l11_outproj_lnmd.prebuild(_l11_self_check, static_ok)


def prebuild_l11_do8() -> bool:
    """DLRM_HSTU_FP8_L11_DO8 (gin item (DO8)): build + load + self-check L11's
    dO8/E8 emission variant before RUN_START (triton_fp8.prebuild_extensions,
    right after prebuild_l11). Its own .so, flags, resource gate, self-check and
    latch (ops/hip/l11_do8.py): a failure latches DO8 off, never L11. No-op
    without the knob (with item (104) alone on: its one INERT line); INERT line
    without a prebuilt L11 or without item (38)(a)'s e5m2-du mode (DO8's only mode)."""
    if not l11_do8.requested():
        return l11_do8.prebuild_mx6()  # MX6 (module end): False; knob off: nothing read beyond it, no line
    return l11_do8.prebuild(
        l11_ready=l11_outproj_lnmd.enabled() and l11_outproj_lnmd.prebuilt(),
        e5_expected=bwd_fp8.du_producer_enabled() and hstu_du_e5m2.broken() is None,
    )


def _l11_du_e5m2_or_none(
    g8: torch.Tensor,
    w8: torch.Tensor,
    g8_scale: torch.Tensor,
    w8_scale: torch.Tensor,
    *,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    num_warps: int,
    dropout_ratio: float,
    silu_u: bool,
    random_mask: torch.Tensor,
    du_row_width: int,
    site_key: int,
    inv_scale: torch.Tensor,
    cast_scale: torch.Tensor,
    generation: Optional[Tuple[object, int]],
) -> Optional[
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        "hstu_du_e5m2.DuE5m2Emit",
    ]
]:
    """L11 e5m2-du mode (gin item (39) together with item (38)(a) B1a).

    Called by HSTUComputeOutputFunction.backward on an L11-eligible call
    (decline_reason None, FP8 operands cast) for which B1a's `_du_e5m2_plan`
    returned its plan, i.e. B1a would emit this call's e5m2 u block. The
    keyword arguments are EXACTLY those the backward passes to B1a's
    `_triton_layer_norm_mul_dropout_bwd_du_e5m2` for this call (x = attn,
    weight/bias = the LN weight/bias, ctx.BLOCK_D / num_warps /
    dropout_ratio / silu_u, site_key = the B1a site key, inv_scale /
    cast_scale / generation = the plan), with the dgrad operands (g8, w8,
    g8_scale, w8_scale: e5m2 dout [M, 512], e4m3 W [1536, 512] and their
    decode scales, the cast's outputs) in place of dy.

    When L11 is requested, prebuilt (the pre-RUN_START build + self-check of
    both instantiations passed) and not latched, this runs
    l11_outproj_lnmd.run_e5: ONE l11_outproj_lnmd_bwd launch in its E5 mode
    (dy never in HBM) plus the stock dW/dB finalize and the amax / colsum
    reductions, returning EXACTLY what
    `_triton_layer_norm_mul_dropout_bwd_du_e5m2(dy=bwd_fp8.output_projection_dgrad_gemm(
    g8, w8, g8_scale, w8_scale, torch.bfloat16), **the same keywords)`
    returns: (dx [M, 512] bf16, du = hstu_du_e5m2.placeholder_du(u, M, 512),
    dweight [512], dbias [512] in weight's dtype, hstu_du_e5m2.DuE5m2Emit
    with dz8 [roundup(M, bwd_fp8.uvqk_tm_pad()), du_row_width] in
    bwd_fp8.grad_dtype_and_limit()'s dtype (u block cols [0, 512) and its pad
    rows written with inv_scale), u_cols (0, 512), u_amax, u_colsum [512]
    fp32, inv_scale, cast_scale, rows M, width du_row_width, site_key,
    generation, buffer_ptr 0). The caller then skips the dgrad GEMM and
    attaches the record exactly as on B1a's path. Numerics vs that chain:
    payload (incl. pad rows), amax and dx bitwise on hipBLASLt's data-parallel
    rows (bitwise vs B1a on the same dy everywhere), SK3-tail rows in dy's
    fp32 order; u colsum and dW/dB a declared fp32 reduction order.

    None = nothing emitted and no state touched (all outputs are the entry's
    own fresh buffers; no FP8 scale state is read or written beyond the
    plan): the caller runs the dgrad GEMM on the same operands and then B1a's
    unchanged path. That happens with the knob off, without the prebuild,
    after a latch, or when run_e5 declines (its own E5 checks, one NOT armed
    line per reason) or fails (it latches L11 off for the process with one
    DISABLED line and returns None; it never raises unless
    DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED_FAIL_CLOSED=1 -- the caller has no
    try/except around this hook).
    """
    if not (l11_outproj_lnmd.enabled() and l11_outproj_lnmd.prebuilt()):
        return None
    with record_function("## hstu_l11_outproj_lnmd_bwd ##"):
        return l11_outproj_lnmd.run_e5(
            g8,
            w8,
            g8_scale,
            w8_scale,
            x,
            u,
            weight,
            bias,
            mean,
            rstd,
            BLOCK_D,
            num_warps,
            dropout_ratio,
            silu_u,
            lnmd_mask_nib.as_int8(random_mask, "L11 run_e5 (_l11_du_e5m2_or_none: DO8 declined, latched or off)"),  # noqa: E501  NIB (module end)
            du_row_width,
            site_key,
            inv_scale,
            cast_scale,
            generation,
        )


def _l11_do8_e5_or_none(
    g8: torch.Tensor,
    w8: torch.Tensor,
    g8_scale: torch.Tensor,
    w8_scale: torch.Tensor,
    *,
    site: object,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    num_warps: int,
    dropout_ratio: float,
    silu_u: bool,
    random_mask: torch.Tensor,
    du_row_width: int,
    site_key: int,
    inv_scale: torch.Tensor,
    cast_scale: torch.Tensor,
    generation: Optional[Tuple[object, int]],
) -> Optional[
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        "hstu_du_e5m2.DuE5m2Emit",
        torch.Tensor,
        torch.Tensor,
    ]
]:
    """DO8 (gin item (DO8), DLRM_HSTU_FP8_L11_DO8) in place of
    _l11_du_e5m2_or_none on an e5m2-du call whose attention backward plans the
    C8 do_blk dO quantization (`site`: l11_do8.emission_site, decided before
    the launch). Same keywords and the same first five return values as
    _l11_du_e5m2_or_none (one l11_outproj_lnmd_bwd_do8 launch instead of L11's
    E5 launch), plus (dO8 [M, 512] e4m3fn, E8 [M, 4] uint8) for the caller to
    publish after its last write to d(attn). None = nothing emitted, nothing
    visible (fresh buffers only): DO8 declined or failed (latched off: DO8
    only), and the caller runs _l11_du_e5m2_or_none on the same operands."""
    if site is None or not (
        l11_outproj_lnmd.enabled() and l11_outproj_lnmd.prebuilt()
    ):
        return None
    with record_function("## hstu_l11_outproj_lnmd_bwd ##"):
        return l11_do8.run_e5_plus(
            g8,
            w8,
            g8_scale,
            w8_scale,
            x,
            u,
            weight,
            bias,
            mean,
            rstd,
            BLOCK_D,
            num_warps,
            dropout_ratio,
            silu_u,
            random_mask,
            du_row_width,
            site_key,
            inv_scale,
            cast_scale,
            generation,
            site,
        )


class HSTUComputeOutputFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        attn: torch.Tensor,
        u: torch.Tensor,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        output_weight: torch.Tensor,
        eps: float,
        dropout_ratio: float,
        training: bool,
        silu_u: bool = False,
        concat_u: bool = False,
        concat_x: bool = False,
        mul_u_activation_type: str = "none",
        group_norm: bool = False,
        num_heads: int = 1,
        linear_dim: int = -1,
        seed: Optional[int] = None,
        recompute_y_in_backward: bool = False,
        rng_row_indices: Optional[torch.Tensor] = None,
        fp8_output_scale: Optional[torch.Tensor] = None,
        need_bwd: bool = True,
    ) -> torch.Tensor:
        if dropout_ratio == 0.0:
            training = False

        fp8_scale_value = activation_scale_value(fp8_output_scale)
        fp8_scale = (
            fp8_scale_tensor(fp8_output_scale)
            if fp8_output_scale is not None
            else None
        )
        fp8_requested = (
            hstu_fp8_fuse_outln_enabled() and fp8_output_scale is not None
        )
        use_fp8 = (
            fp8_requested
            and not group_norm
            and recompute_y_in_backward
            and rng_row_indices is None
            and fp8_supported(
                attn,
                output_weight,
                input_columns=output_weight.shape[0],
            )
        )
        if fp8_requested and not use_fp8 and hstu_fp8_fail_closed():
            raise RuntimeError(
                "Plan 08 output FP8 path requested for an unsupported contract"
            )
        fp8_yt_result = None
        if group_norm:
            y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed = (
                triton_group_norm_mul_dropout_fwd(
                    x=attn,
                    u=u,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=eps,
                    dropout_ratio=dropout_ratio,
                    training=training,
                    silu_u=silu_u,
                    concat_ux=concat_u and concat_x,
                    num_heads=num_heads,
                    linear_dim=linear_dim,
                    seed=seed,
                )
            )
            ctx.BLOCK_H = BLOCK_H
            random_mask = None
        else:
            if (
                _FP8_OUT_WGRAD
                and _FP8_OUT_WGRAD_IMPL == "producer"
                and use_fp8
                and fp8_scale_value is not None
                and bwd_fp8.scale_controller_enabled()
                and output_weight.shape[1] % 256 == 0
            ):
                fp8_yt_result = triton_layer_norm_mul_dropout_fwd_fp8_yt(
                    x=attn,
                    u=u,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=eps,
                    dropout_ratio=dropout_ratio,
                    training=training,
                    silu_u=silu_u,
                    concat_u=concat_u,
                    concat_x=concat_x,
                    mul_u_activation_type=mul_u_activation_type,
                    seed=seed,
                    inv_scale=1.0 / fp8_scale_value,
                )
            if fp8_yt_result is not None:
                assert fp8_scale is not None
                y, mean, rstd, BLOCK_D, num_warps, seed, random_mask, y_t = (
                    fp8_yt_result
                )
                out = scaled_addmm_fp8_preq(
                    activation_fp8=y,
                    scale_a=fp8_scale,
                    weight=output_weight,
                    bias_or_residual=x,
                    out_dtype=x.dtype,
                )
                record_fp8_capability("phi3_output_ln_projection")
                record_fp8_contract(
                    "phi3_output_ln_projection",
                    final_output=out,
                    normalized_dropout_input=y,
                )
            elif use_fp8 and fp8_scale_value is not None:
                assert fp8_scale is not None; _lnmd_mask_nib_note_call(attn, output_weight, need_bwd)  # noqa: E702  NIB (module end): this call's out-proj context for the maskgen hook; knob off: returns at once
                y, mean, rstd, BLOCK_D, num_warps, seed, random_mask = (
                    triton_layer_norm_mul_dropout_fwd(
                        x=attn,
                        u=u,
                        weight=norm_weight,
                        bias=norm_bias,
                        eps=eps,
                        dropout_ratio=dropout_ratio,
                        training=training,
                        silu_u=silu_u,
                        concat_u=concat_u,
                        concat_x=concat_x,
                        mul_u_activation_type=mul_u_activation_type,
                        seed=seed,
                        rng_row_indices=rng_row_indices,
                        output_fp8=True,
                        inv_scale=1.0 / fp8_scale_value,
                    )
                )
                out = scaled_addmm_fp8_preq(
                    activation_fp8=y,
                    scale_a=fp8_scale,
                    weight=output_weight,
                    bias_or_residual=x,
                    out_dtype=x.dtype,
                )
                record_fp8_capability("phi3_output_ln_projection")
                record_fp8_contract(
                    "phi3_output_ln_projection",
                    final_output=out,
                    normalized_dropout_input=y,
                )
            else:
                y, mean, rstd, BLOCK_D, num_warps, seed, random_mask = (
                    triton_layer_norm_mul_dropout_fwd(
                        x=attn,
                        u=u,
                        weight=norm_weight,
                        bias=norm_bias,
                        eps=eps,
                        dropout_ratio=dropout_ratio,
                        training=training,
                        silu_u=silu_u,
                        concat_u=concat_u,
                        concat_x=concat_x,
                        mul_u_activation_type=mul_u_activation_type,
                        seed=seed,
                        rng_row_indices=rng_row_indices,
                    )
                )
                if use_fp8:
                    calibrate_scale_buffer(fp8_output_scale, y)
                out = maybe_triton_addmm_fwd(
                    x=y,
                    w=output_weight.to(y.dtype),
                    y=x,
                )
        if group_norm:
            out = maybe_triton_addmm_fwd(
                x=y,
                w=output_weight.to(y.dtype),
                y=x,
            )

        saved_tensors = [attn, u, norm_weight, norm_bias, mean, rstd, output_weight]
        if not recompute_y_in_backward:
            saved_tensors.append(y)
        # FP8 output wgrad: keep the e4m3 y the forward GEMM consumed, plus its
        # dequant scale as a device value (fp8_scale_value is the host float the
        # cast used; a fill launches no sync). Not when no backward can run
        # (need_bwd=False: DLRM_HSTU_FP8_EVAL_FWD_HYGIENE, decided by the wrapper).
        fp8_y_saved = (
            need_bwd
            and _FP8_OUT_WGRAD
            and use_fp8
            and fp8_scale_value is not None
            and bwd_fp8.scale_controller_enabled()
            and y.shape[0] > 0
            and recompute_y_in_backward
            and y.dim() == 2
            and y.stride(1) == 1
            and y.shape[1] % 256 == 0
            and output_weight.shape[1] % 256 == 0
        )
        fp8_y_token_major = fp8_y_saved and fp8_yt_result is not None
        if fp8_y_saved:
            # Producer mode keeps only the token-major copy (same bytes); the
            # row-major y dies after the forward GEMM, as in the default path.
            saved_tensors.append(y_t if fp8_y_token_major else y)
            saved_tensors.append(
                torch.full((1,), fp8_scale_value, dtype=torch.float32, device=y.device)
            )
        ctx.fp8_y_saved = fp8_y_saved
        ctx.fp8_y_token_major = fp8_y_token_major
        # The NN wgrad reads row-major y and a token-major dout.
        ctx.fp8_dout_token_major = fp8_y_token_major or (
            fp8_y_saved and _FP8_OUT_WGRAD_IMPL == "nn"
        )
        # Save random_mask for reuse in backward pass (avoids regenerating mask)
        # When random_mask is available (SM100+ path), always save it.
        if random_mask is not None:
            saved_tensors.append(random_mask)
            ctx.has_random_mask = True
        else:
            ctx.has_random_mask = False
        ctx.save_for_backward(*saved_tensors)
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.seed = seed
        ctx.training = training
        ctx.concat_u = concat_u
        ctx.concat_x = concat_x
        ctx.dropout_ratio = dropout_ratio
        ctx.num_heads = num_heads
        ctx.linear_dim = linear_dim
        ctx.group_norm = group_norm
        ctx.recompute_y_in_backward = recompute_y_in_backward
        ctx.silu_u = silu_u
        ctx.mul_u_activation_type = mul_u_activation_type
        ctx.fp8_forward = use_fp8 and fp8_scale_value is not None
        # Set by the deferred-SiLU preprocess forward on the u it returns: the
        # width of the d(uvqk) buffer its backward would otherwise copy du into.
        ctx.du_row_width = int(getattr(u, HSTU_DUVQK_WIDTH_ATTR, 0))
        if bwd_fp8.du_producer_requested():
            # B1a (r16): the layer's uvqk site handle (hstu_du_e5m2.DuE5m2Site)
            # the preprocess forward tagged when it planned the fp8t wgrad.
            ctx.du_e5m2_site = _l11_mx6_forward_tag(ctx, getattr(u, HSTU_DUVQK_SITE_ATTR, None))  # MX6 (module end)
        # DO8 (gin item (DO8), DLRM_HSTU_FP8_L11_DO8): the dO8 site R15's
        # attention forward tagged on this call's attention output (carried to
        # the `attn` view; hstu_do8_registry.site_of). None with the knob off,
        # without the registry, or when the attention backward does not plan
        # the C8 do_blk dO quantization (e.g. the targets-only layer).
        ctx.do8_site = l11_do8.site_of(attn)
        return _erm_forward_tag(ctx, out, output_weight, need_bwd)

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dout: torch.Tensor
    ) -> Tuple[
        torch.Tensor,  # dattn
        torch.Tensor,  # du
        torch.Tensor,  # dx
        torch.Tensor,  # d_norm_weight
        torch.Tensor,  # d_norm_bias
        torch.Tensor,  # d_output_weight
        None,  # eps
        None,  # dropout_ratio
        None,  # training
        None,  # silu_u
        None,  # concat_u
        None,  # concat_x
        None,  # mul_u_activation_type
        None,  # group_norm
        None,  # num_heads
        None,  # linear_dim
        None,  # seed
        None,  # recompute_y_in_backward
        None,  # rng_row_indices
        None,  # fp8_output_scale
        None,  # need_bwd
    ]:
        attn, u, norm_weight, norm_bias, mean, rstd, output_weight = ctx.saved_tensors[
            :7
        ]
        # Extract optional saved tensors based on flags
        next_idx = 7
        if not ctx.recompute_y_in_backward:
            saved_y = ctx.saved_tensors[next_idx]
            next_idx += 1
        else:
            saved_y = None
        fp8_y_saved = getattr(ctx, "fp8_y_saved", False)
        if fp8_y_saved:
            y_fp8 = ctx.saved_tensors[next_idx]
            y_fp8_scale = ctx.saved_tensors[next_idx + 1]
            next_idx += 2
        if ctx.has_random_mask:
            random_mask = ctx.saved_tensors[next_idx]
        else:
            random_mask = None
        dout_fp8 = None
        # L11 (gin item (39), DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED): the FP8 dgrad and the LN*U*dropout backward
        # below run as ONE fused kernel (dy never in HBM). Every dispatch condition is host metadata checked BEFORE
        # the cast; a decline runs today's code unchanged, a latched failure today's dgrad on the already-cast
        # operands, then today's LN*U*dropout backward. With item (38)(a) B1a, B1a's own plan for this call is
        # evaluated once, between the cast and the GEMM: no plan (B1a off or declined) -> L11's bf16-du mode; a
        # plan -> its e5m2-du mode (_l11_du_e5m2_or_none -> run_e5: B1a's emission contract, no GEMM; None if it
        # declines or latched: the GEMM, then B1a's path unchanged). MX6 (item (104)): see the branch below.
        l11_out = None
        l11_e5 = None
        l11_go = False
        l11_du_planned = False
        du_plan = None
        # DO8 (gin item (DO8)): (site, dO8, E8) of an emitting e5m2-du call,
        # published below after the last write to dattn.
        l11_do8_pair = None
        l11_do8_out = None
        if fp8_y_saved and l11_outproj_lnmd.requested():
            l11_why = l11_outproj_lnmd.decline_reason(
                ctx,
                dout,
                output_weight,
                attn,
                u,
                mean,
                rstd,
                norm_weight,
                norm_bias,
                random_mask,
                snr_enabled=fp8_wgrad_snr.SNR_ENABLED,
                lnmd_bwd_v2=_use_lnmd_bwd_v2(),
            )
            if l11_why is None:
                l11_go = True
            elif l11_why.split(" ", 1)[0] not in ("D0", "D1"):
                l11_outproj_lnmd.note_decline(l11_why)
        with record_function("## hstu_output_projection_backward ##"):
            if l11_go and getattr(ctx, "l11_mx6_layer", None) is not None and (l11_mx6 := _l11_mx6_backward(ctx, dout,
                    y_fp8, attn, u, mean, rstd, norm_weight, norm_bias, output_weight, random_mask)) is not None:
                dy, dout_fp8, dout_fp8_scale, l11_out, l11_e5, l11_do8_pair, du_plan, l11_du_planned = l11_mx6
            elif l11_go:
                l11_ops = bwd_fp8.output_projection_dgrad_operands(
                    dout,
                    output_weight,
                    token_major=getattr(ctx, "fp8_dout_token_major", False),
                    # r14 (38)(f) B13: the same step-tiled dout_t decision as
                    # the unfused branch below; L11 reads only the row-major
                    # e5m2 gradient, W3 reads this operand.
                    t_tiled=_out_wgrad_dout_tiled(ctx, dout, y_fp8), erm=_erm_take(ctx),
                )
                if l11_ops is None:
                    dy = None  # scale-controller warmup: bf16 torch.mm below, as today
                else:
                    g8, g8_scale, dout_fp8, dout_fp8_scale, w8, w8_scale = l11_ops
                    l11_site = getattr(ctx, "du_e5m2_site", None)
                    if l11_site is not None:
                        # B1a's plan for this call, decided here instead of
                        # after the GEMM (it reads host state and dy's
                        # metadata only: [M, 1536] in dout's dtype, as the
                        # GEMM returns). compute_y is False: dout_fp8 is set
                        # and the SNR probe is off (decline D5).
                        du_plan = _du_e5m2_plan(
                            ctx,
                            l11_site,
                            torch.empty(
                                (dout.shape[0], output_weight.shape[0]),
                                dtype=dout.dtype,
                                device="meta",
                            ),
                            attn,
                            u,
                            random_mask,
                            getattr(ctx, "du_row_width", 0),
                            compute_y=False,
                        )
                        l11_du_planned = True
                    if du_plan is None:
                        l11_do8.note_bf16_mode(ctx)  # DO8: no bf16-du variant
                        with record_function("## hstu_l11_outproj_lnmd_bwd ##"):
                            l11_out = l11_outproj_lnmd.run(
                                g8,
                                w8,
                                g8_scale,
                                w8_scale,
                                attn,
                                u,
                                lnmd_mask_nib.as_int8(random_mask, "L11 run (bf16-du mode)"),  # NIB (module end): exact int8 unpack
                                mean,
                                rstd,
                                norm_weight,
                                norm_bias,
                                ctx.dropout_ratio,
                                _alloc_du,
                            )
                    elif l11_do8.requested() and (
                        l11_do8_out := _l11_do8_e5_or_none(
                            g8,
                            w8,
                            g8_scale,
                            w8_scale,
                            # DO8: decided here, before the launch (host
                            # metadata only); None -> L11's own e5 launch below
                            site=l11_do8.emission_site(ctx, attn),
                            x=attn,
                            u=u,
                            weight=norm_weight,
                            bias=norm_bias,
                            mean=mean,
                            rstd=rstd,
                            BLOCK_D=ctx.BLOCK_D,
                            num_warps=ctx.num_warps,
                            dropout_ratio=ctx.dropout_ratio,
                            silu_u=ctx.silu_u,
                            random_mask=random_mask,
                            du_row_width=getattr(ctx, "du_row_width", 0),
                            site_key=l11_site.key,
                            inv_scale=du_plan[0],
                            cast_scale=du_plan[1],
                            generation=du_plan[2],
                        )
                    ) is not None:
                        # DO8: the e5m2-du mode's record + d(attn) as from
                        # _l11_du_e5m2_or_none, and dO8/E8 to publish below.
                        l11_e5 = l11_do8_out[:5]
                        l11_do8_pair = (
                            ctx.do8_site,
                            l11_do8_out[5],
                            l11_do8_out[6],
                        )
                        l11_do8_out = None
                    else:
                        # The keywords of B1a's call below, the cast's
                        # operands instead of dy.
                        l11_e5 = _l11_du_e5m2_or_none(
                            g8,
                            w8,
                            g8_scale,
                            w8_scale,
                            x=attn,
                            u=u,
                            weight=norm_weight,
                            bias=norm_bias,
                            mean=mean,
                            rstd=rstd,
                            BLOCK_D=ctx.BLOCK_D,
                            num_warps=ctx.num_warps,
                            dropout_ratio=ctx.dropout_ratio,
                            silu_u=ctx.silu_u,
                            random_mask=random_mask,
                            du_row_width=getattr(ctx, "du_row_width", 0),
                            site_key=l11_site.key,
                            inv_scale=du_plan[0],
                            cast_scale=du_plan[1],
                            generation=du_plan[2],
                        )
                    # Latched off or declined (either mode): today's dgrad
                    # GEMM on the same operands (no second cast, no second
                    # scale-state update).
                    dy = (
                        None
                        if l11_out is not None or l11_e5 is not None
                        else bwd_fp8.output_projection_dgrad_gemm(
                            g8, w8, g8_scale, w8_scale, dout.dtype
                        )
                    )
            elif fp8_y_saved:
                dgrad_and_operand = bwd_fp8.output_projection_dgrad_with_operand(
                    dout,
                    output_weight,
                    token_major=getattr(ctx, "fp8_dout_token_major", False),
                    t_tiled=_out_wgrad_dout_tiled(ctx, dout, y_fp8), erm=_erm_take(ctx),
                )
                if dgrad_and_operand is None:
                    dy = None
                else:
                    dy, dout_fp8, dout_fp8_scale = dgrad_and_operand
            else:
                dy = bwd_fp8.output_projection_dgrad(dout, output_weight)
            if dy is None and l11_out is None and l11_e5 is None:
                dy = torch.mm(dout, output_weight.to(dout.dtype).t())
        # Numerics probe (DLRM_HSTU_FP8_BWD_OUT_WGRAD_SNR, default off): on sampled
        # calls also recompute the bf16 y so the probe can compare against it.
        wgrad_snr = False
        if dout_fp8 is not None and fp8_wgrad_snr.SNR_ENABLED:
            wgrad_snr = fp8_wgrad_snr.snr_requested(dout.device)

        if ctx.group_norm:
            dattn, du, d_norm_weight, d_norm_bias, y = (
                triton_group_norm_mul_dropout_bwd(
                    dy=dy,
                    x=attn,
                    u=u,
                    weight=norm_weight,
                    bias=norm_bias,
                    mean=mean,
                    rstd=rstd,
                    BLOCK_D=ctx.BLOCK_D,
                    BLOCK_H=ctx.BLOCK_H,
                    num_warps=ctx.num_warps,
                    eps=ctx.eps,
                    training=ctx.training,
                    dropout_ratio=ctx.dropout_ratio,
                    seed=ctx.seed,
                    silu_u=ctx.silu_u,
                    concat_ux=ctx.concat_u and ctx.concat_x,
                    num_heads=ctx.num_heads,
                    linear_dim=ctx.linear_dim,
                    compute_y=ctx.recompute_y_in_backward,
                )
            )
        elif l11_out is not None:
            # L11: dx and du were written by the fused kernel; reduce its
            # [P, 512] dW/dB partials with the production finalize.
            du_row_width = getattr(ctx, "du_row_width", 0)
            dattn, du, l11_dwp, l11_dbp, l11_p = l11_out
            d_norm_weight = torch.empty(
                (attn.shape[1],), dtype=norm_weight.dtype, device=attn.device
            )
            d_norm_bias = torch.empty(
                (attn.shape[1],), dtype=norm_weight.dtype, device=attn.device
            )
            _lnmd_bwd_finalize(
                l11_dwp,
                l11_dbp,
                d_norm_weight,
                d_norm_bias,
                l11_p,
                attn.shape[1],
                torch.cuda.get_device_properties(attn.device).multi_processor_count,
            )
            y = None
            _tag_duvqk_buffer(du, du_row_width)
        else:
            du_row_width = getattr(ctx, "du_row_width", 0)
            # B1a (r16, DLRM_HSTU_FP8_BWD_DU_E5M2=1 only; default off): emit
            # the e5m2 u columns of this layer's dz8 instead of the bf16 du
            # (hstu_du_e5m2.py). du_e5m2_site is set only with the knob on.
            du_site = getattr(ctx, "du_e5m2_site", None)
            du_emit = None
            if l11_e5 is not None:
                # L11 e5m2-du mode (gin item (39) with (38)(a)): B1a's
                # emission contract, produced without the dgrad GEMM.
                dattn, du, d_norm_weight, d_norm_bias, du_emit = l11_e5
                y = None
                du_plan = None
            elif not l11_du_planned:
                du_plan = (
                    _du_e5m2_plan(
                        ctx,
                        du_site,
                        dy,
                        attn,
                        u,
                        random_mask,
                        du_row_width,
                        compute_y=ctx.recompute_y_in_backward
                        and (dout_fp8 is None or wgrad_snr),
                    )
                    if du_site is not None
                    else None
                )
            if du_plan is not None:
                try:
                    dattn, du, d_norm_weight, d_norm_bias, du_emit = (
                        _triton_layer_norm_mul_dropout_bwd_du_e5m2(
                            dy=dy,
                            x=attn,
                            u=u,
                            weight=norm_weight,
                            bias=norm_bias,
                            mean=mean,
                            rstd=rstd,
                            BLOCK_D=ctx.BLOCK_D,
                            num_warps=ctx.num_warps,
                            dropout_ratio=ctx.dropout_ratio,
                            silu_u=ctx.silu_u,
                            random_mask=lnmd_mask_nib.as_int8(random_mask, "B1a v3 (HSTUComputeOutputFunction.backward)"),  # noqa: E501  NIB (module end)
                            du_row_width=du_row_width,
                            site_key=du_site.key,
                            inv_scale=du_plan[0],
                            cast_scale=du_plan[1],
                            generation=du_plan[2],
                        )
                    )
                    y = None
                except Exception as error:  # nothing emitted: shipped path
                    du_emit = None
                    reason = f"producer failed: {type(error).__name__}: {str(error)[:160]}"
                    hstu_du_e5m2.set_broken(reason)
                    hstu_du_e5m2.log_not_used(
                        "producer", reason, "shipped LNMD bwd v2 (latched)"
                    )
                du_plan = None
            if du_emit is None:
                dattn, du, d_norm_weight, d_norm_bias, y = (
                    triton_layer_norm_mul_dropout_bwd(
                        dy=dy,
                        x=attn,
                        u=u,
                        weight=norm_weight,
                        bias=norm_bias,
                        mean=mean,
                        rstd=rstd,
                        BLOCK_D=ctx.BLOCK_D,
                        num_warps=ctx.num_warps,
                        eps=ctx.eps,
                        training=ctx.training,
                        dropout_ratio=ctx.dropout_ratio,
                        seed=ctx.seed,
                        silu_u=ctx.silu_u,
                        concat_u=ctx.concat_u,
                        concat_x=ctx.concat_x,
                        mul_u_activation_type=ctx.mul_u_activation_type,
                        compute_y=ctx.recompute_y_in_backward
                        and (dout_fp8 is None or wgrad_snr),
                        random_mask=lnmd_mask_nib.as_int8(random_mask, "Triton LN*U*dropout bwd v2 (HSTUComputeOutputFunction.backward)"),  # noqa: E501  NIB (module end)
                        du_row_width=du_row_width,
                    )
                )
            _tag_duvqk_buffer(du, du_row_width)
            if du_emit is not None:
                # B1a: du is the stride-0 placeholder (no bf16 du, no d(uvqk)
                # buffer, hence no BUFFER tag above: stride(0) == 0); the
                # record is the only valid u gradient. The preprocess backward
                # allocates its own d(uvqk) and never reads du.
                assert getattr(du, HSTU_DUVQK_BUFFER_ATTR, None) is None
                setattr(du, HSTU_DU_E5M2_ATTR, du_emit)
                du_site.emitted = du_emit.token
                du_emit = None
            if l11_do8_pair is not None:
                # DO8 (gin item (DO8)): dattn is final here (nothing below
                # writes it; it is returned as d(attn)): hand dO8/E8 to the C8
                # attention backward (R15 hstu_do8_registry.publish, 'plus';
                # a refusal is counted and the attention backward quantizes).
                l11_do8.publish(
                    l11_do8_pair[0], dattn, l11_do8_pair[1], l11_do8_pair[2]
                )
                l11_do8_pair = None
        if not ctx.recompute_y_in_backward:
            y = saved_y
        with record_function("## hstu_output_projection_backward ##"):
            if fp8_y_saved and dout_fp8 is not None:
                global _fp8_out_wgrad_logged
                from generative_recommenders.ops.triton.fp8_wgrad import (
                    fp8_wgrad,
                    fp8_wgrad_tn,
                )

                if getattr(ctx, "fp8_y_token_major", False):
                    # Producer mode: both operands already token-major
                    # ([C, M_pad], zero-padded), straight into the FP8 TN GEMM.
                    d_output_weight = torch._scaled_mm(
                        y_fp8,
                        dout_fp8.t(),
                        scale_a=y_fp8_scale,
                        scale_b=dout_fp8_scale,
                        out_dtype=torch.float32,
                    ).to(output_weight.dtype)
                    wgrad_path = "producer_tn"
                elif getattr(ctx, "fp8_dout_token_major", False):
                    # NN mode: row-major y as stored, token-major dout [N, M_pad].
                    from generative_recommenders.ops.hipblaslt_fp8_nn import (
                        fp8_wgrad_nn,
                    )
                    from generative_recommenders.ops.triton.fp8_wgrad import (
                        transpose_fp8_token_major,
                    )

                    # DLRM_HSTU_FP8_BWD_WGRAD_CK_W3 (default off): the custom
                    # kernel first; None -> the hipBLASLt call below, unchanged.
                    ck_w3 = (
                        _erm_out_wgrad_nn_ck_w3(ctx,
                            y_fp8,
                            dout_fp8,
                            y_fp8_scale,
                            dout_fp8_scale,
                            fp8_wgrad_nn,
                        )
                        if wgrad_ck_w3.enabled()
                        else None
                    )
                    if ck_w3 is None:
                        # B13: a step-tiled dout_t (3-D) is re-laid out first
                        # (bitwise the 2-D cast's bytes; B11 pool copy); 2-D: unchanged.
                        dout_fp8 = _erm_dout_row_major(ctx,
                            dout_fp8, int(y_fp8.shape[0]), required=True
                        )
                        dw_t = fp8_wgrad_nn(
                            y_fp8,
                            dout_fp8,
                            y_fp8_scale,
                            dout_fp8_scale,
                            out_dtype=torch.float32,
                        )
                        nn_path = "nn"
                    else:
                        dw_t, nn_path = ck_w3
                    if dw_t is not None:
                        wgrad_path = nn_path
                        d_output_weight = (
                            dw_t.to(output_weight.dtype).t().contiguous()
                        )
                    else:
                        wgrad_path = "nn_fallback_tn"
                        dout_fp8 = _erm_dout_row_major(ctx,
                            dout_fp8, int(y_fp8.shape[0]), required=True
                        )
                        # No NN algorithm in this library: transpose y, TN GEMM.
                        d_output_weight = torch._scaled_mm(
                            transpose_fp8_token_major(y_fp8),
                            dout_fp8.t(),
                            scale_a=y_fp8_scale,
                            scale_b=dout_fp8_scale,
                            out_dtype=torch.float32,
                        ).to(output_weight.dtype)
                else:
                    wgrad_fn = (
                        fp8_wgrad if _FP8_OUT_WGRAD_IMPL == "triton" else fp8_wgrad_tn
                    )
                    wgrad_path = (
                        "triton" if _FP8_OUT_WGRAD_IMPL == "triton" else "transpose_tn"
                    )
                    d_output_weight = wgrad_fn(
                        y_fp8,
                        dout_fp8,
                        y_fp8_scale,
                        dout_fp8_scale,
                        out_dtype=torch.float32,
                    ).to(output_weight.dtype)
                if not _fp8_out_wgrad_logged:
                    _fp8_out_wgrad_logged = True
                    # Global rank from the process group (RANK is not always in
                    # the worker env), so a log checker can prove the arm reached
                    # every rank at multi-node, where device ids repeat.
                    _rank = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_available()
                        and torch.distributed.is_initialized()
                        else os.environ.get("RANK", "?")
                    )
                    print(
                        f"[fp8-out-wgrad] armed impl={_FP8_OUT_WGRAD_IMPL} "
                        f"path={wgrad_path} "
                        f"dfmt={bwd_fp8._OUT_WGRAD_DFMT} "
                        f"shift={bwd_fp8._OUT_WGRAD_DFMT_SHIFT_RAW} "
                        f"rank={_rank} "
                        f"device={dout.device} y={tuple(y_fp8.shape)}:{y_fp8.dtype} "
                        f"dout={tuple(dout_fp8.shape)}:{dout_fp8.dtype}",
                        flush=True,
                    )
                if wgrad_snr:
                    fp8_wgrad_snr.log_out_wgrad_snr(
                        y_bf16=y if ctx.recompute_y_in_backward else None,
                        y_fp8=y_fp8,
                        y_scale=y_fp8_scale,
                        dout=dout,
                        dout_fp8=_erm_untile_to_row_major(ctx,
                            dout_fp8, int(y_fp8.shape[0])
                        ),
                        dout_scale=dout_fp8_scale,
                        dout_token_major=getattr(ctx, "fp8_dout_token_major", False),
                        d_output_weight=d_output_weight,
                    )
            else:
                # Default path, and the FP8-wgrad fallback (scale-controller
                # warmup, or no e5m2 dout): y was recomputed in bf16 above, so
                # this matches the default path exactly.
                d_output_weight = torch.mm(y.t(), dout).to(output_weight.dtype)
        return (
            dattn,
            du,
            dout,
            d_norm_weight,
            d_norm_bias,
            d_output_weight,
            None,  # eps
            None,  # dropout_ratio
            None,  # training
            None,  # silu_u
            None,  # concat_u
            None,  # concat_x
            None,  # mul_u_activation_type
            None,  # group_norm
            None,  # num_heads
            None,  # linear_dim
            None,  # seed
            None,  # recompute_y_in_backward
            None,  # rng_row_indices
            None,  # fp8_output_scale
            None,  # need_bwd
        )


@triton.jit
def _helion_ln_mul_dropout_fwd(
    x,
    weight,
    bias,
    u,
    y,
    mean,
    rstd,
    eps,
    seed,
    dropout_ratio,
    D: tl.constexpr,
    stride_x: tl.constexpr,
    stride_u: tl.constexpr,
    stride_y: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CONCAT_UX: tl.constexpr,
    SILU_U: tl.constexpr,
    TRAINING: tl.constexpr,
):
    row = tl.program_id(0)
    x += row.to(tl.int64) * stride_x
    u += row.to(tl.int64) * stride_u
    y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    # Load input
    x_val = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)

    # Precompute inverse of D for faster computation
    inv_D = 1.0 / D

    # Compute mean
    mean_val = tl.sum(x_val, axis=0) * inv_D

    # Center the data
    x_mean = tl.where(mask, x_val - mean_val, 0.0)

    # Compute variance
    var = tl.sum(x_mean * x_mean, axis=0) * inv_D

    # Compute reciprocal standard deviation
    # pyre-fixme[16]
    rstd_val = libdevice.rsqrt(var + eps)

    # Normalize
    y_norm = x_mean * rstd_val

    # Apply weight and bias
    w = tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias + cols, mask=mask, other=0.0).to(tl.float32)
    y_ln = y_norm * w + b

    # Load u and optionally apply SiLU activation
    u_val = tl.load(u + cols, mask=mask, other=0.0).to(tl.float32)
    if SILU_U:
        u_processed = u_val * tl.sigmoid(u_val)
    else:
        u_processed = u_val

    y_out = y_ln * u_processed

    if TRAINING:
        # Compute dropout scale
        # pyre-fixme[16]
        dropout_scale = fast_dividef(1.0, 1.0 - dropout_ratio)

        if CONCAT_UX:
            # Generate dropout masks
            random_offsets = 3 * row * BLOCK_D + cols
            random_u, random_x, random_y = rand3x(seed, random_offsets)

            u_keep = random_u > dropout_ratio
            x_keep = random_x > dropout_ratio
            y_keep = random_y > dropout_ratio

            # Apply dropout to u, x, y
            u_output = tl.where(u_keep, u_processed * dropout_scale, 0.0)
            x_output = tl.where(x_keep, x_val * dropout_scale, 0.0)
            y_output = tl.where(y_keep, y_out * dropout_scale, 0.0)
        else:
            # Generate dropout mask for y
            random_offsets = row * BLOCK_D + cols
            random_y = tl.rand(seed, random_offsets)
            y_keep = random_y > dropout_ratio

            # Apply dropout to y
            y_output = tl.where(y_keep, y_out * dropout_scale, 0.0)
    else:
        if CONCAT_UX:
            u_output = u_processed
            x_output = x_val
        y_output = y_out

    # Store outputs
    if CONCAT_UX:
        tl.store(y + cols, u_output.to(y.dtype.element_ty), mask=mask)
        tl.store(y + D + cols, x_output.to(y.dtype.element_ty), mask=mask)
        tl.store(y + 2 * D + cols, y_output.to(y.dtype.element_ty), mask=mask)
    else:
        tl.store(y + cols, y_output.to(y.dtype.element_ty), mask=mask)

    # Store mean and rstd
    tl.store(mean + row, mean_val)
    tl.store(rstd + row, rstd_val)


def helion_layer_norm_mul_dropout_fwd(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_ux: bool = False,
    seed: Optional[int] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, int, int
]:  # y, mean, rstd, BLOCK_D, num_warps, seed
    N, D = x.shape

    if seed is None:
        seed = torch.randint(low=0, high=2**62, size=(1,), dtype=torch.int64).item()

    if concat_ux:
        y = torch.empty([N, 3 * D], dtype=x.dtype, device=x.device)
    else:
        y = torch.empty([N, D], dtype=x.dtype, device=x.device)
    mean = torch.empty([N], dtype=torch.float32, device=x.device)
    rstd = torch.empty([N], dtype=torch.float32, device=x.device)

    BLOCK_D = triton.next_power_of_2(D)
    # pyre-ignore[28]
    _helion_ln_mul_dropout_fwd[(N,)](
        x,
        weight,
        bias,
        u,
        y,
        mean,
        rstd,
        eps,
        seed,
        dropout_ratio,
        D,
        x.stride(0),
        u.stride(0),
        y.stride(0),
        BLOCK_D,
        CONCAT_UX=concat_ux,
        SILU_U=silu_u,
        TRAINING=training,
        num_warps=1,
    )

    return y, mean, rstd, BLOCK_D, 1, seed  # pyre-ignore [7]


@triton.jit
def _helion_ln_mul_dropout_bwd_dx_du(
    DX,
    DU,
    DY,
    DW,
    DB,
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride_dx,
    stride_du,
    stride_dy,
    stride_x,
    stride_u,
    stride_y,
    D: tl.constexpr,
    eps,
    seed,
    dropout_ratio,
    N,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_UX: tl.constexpr,
    COMPUTE_Y: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    if pid < N % tile_num:
        rows_per_tile += 1

    if rows_per_tile == 0:
        return

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    # precompute inverse of D
    inv_D: tl.constexpr = 1.0 / D

    row = pid
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    if COMPUTE_Y:
        Y += row.to(tl.int64) * stride_y
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx
    DU += row.to(tl.int64) * stride_du
    DW = DW + pid * D + cols
    DB = DB + pid * D + cols

    partial_dw = tl.zeros((BLOCK_D,), dtype=tl.float32)
    partial_db = tl.zeros((BLOCK_D,), dtype=tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)

    for _idx in range(0, rows_per_tile):
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        if CONCAT_UX:
            du = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dx = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + 2 * D + cols, mask=mask, other=0).to(tl.float32)
        else:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)

        if TRAINING:
            # pyre-fixme[16]
            dropout_scale = fast_dividef(1.0, 1.0 - dropout_ratio)
            if CONCAT_UX:
                random_offsets = 3 * row * BLOCK_D + cols
                # apply dropout on du
                random_du, random_dx, random_dy = rand3x(seed, random_offsets)
                du_keep = random_du > dropout_ratio
                du = tl.where(du_keep, du * dropout_scale, 0.0)
                # apply dropout on dx
                dx_keep = random_dx > dropout_ratio
                dx = tl.where(dx_keep, dx * dropout_scale, 0.0)
                # apply dropout on dy
                dy_keep = random_dy > dropout_ratio
                dy = tl.where(dy_keep, dy * dropout_scale, 0.0)
            else:
                random_offsets = row * BLOCK_D + cols
                random = tl.rand(seed, random_offsets)
                dy_keep = random > dropout_ratio
                dy = tl.where(dy_keep, dy * dropout_scale, 0.0)

        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)

        # Compute dx
        xhat = (x - mean) * rstd
        u = tl.load(U + cols, mask=mask, other=0).to(tl.float32)
        ln = xhat * w + b
        du += dy * ln

        if SILU_U:
            sig_u = tl.sigmoid(u)
            silu_u = u * sig_u
            du = du * sig_u * (1 + u - silu_u)
            u = silu_u

        tl.store(DU + cols, du.to(DU.dtype.element_ty), mask=mask)
        dy = dy * u
        wdy = w * dy

        if COMPUTE_Y:
            y = ln * u
            if TRAINING:
                # pyre-fixme[16]
                dropout_scale_y = fast_dividef(1.0, 1.0 - dropout_ratio)
                if CONCAT_UX:
                    u = tl.where(du_keep, u * dropout_scale_y, 0.0)  # pyre-ignore [61]
                    x = tl.where(dx_keep, x * dropout_scale_y, 0.0)  # pyre-ignore [61]
                    y = tl.where(dy_keep, y * dropout_scale_y, 0.0)  # pyre-ignore [61]
                else:
                    y = tl.where(dy_keep, y * dropout_scale_y, 0.0)  # pyre-ignore [61]
            if CONCAT_UX:
                tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, x.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + 2 * D + cols, y.to(Y.dtype.element_ty), mask=mask)
            else:
                tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
            Y += tile_num.to(tl.int64) * stride_y

        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        # multiply by inv_D
        c1 = tl.sum(xhat * wdy, axis=0) * inv_D
        c2 = tl.sum(wdy, axis=0) * inv_D
        dx += (wdy - (xhat * c1 + c2)) * rstd

        # Write dx
        tl.store(DX + cols, dx, mask=mask)

        # Accumulate partial sums for dw/db
        partial_dw += dy * xhat
        partial_db += dy

        X += tile_num.to(tl.int64) * stride_x
        U += tile_num.to(tl.int64) * stride_u
        DY += tile_num.to(tl.int64) * stride_dy
        DX += tile_num.to(tl.int64) * stride_dx
        DU += tile_num.to(tl.int64) * stride_du
        row += tile_num

    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)


@triton_autotune(
    configs=_get_bwd_dwdb_configs(),
    key=["D"],
)
@triton.jit
def _helion_ln_mul_dropout_bwd_dwdb(
    DW,
    DB,
    FINAL_DW,
    FINAL_DB,
    N,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    db = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        off_mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=off_mask, other=0.0)
        db += tl.load(DB + offs, mask=off_mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=cols < D)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.dtype.element_ty), mask=cols < D)


def helion_layer_norm_mul_dropout_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    num_warps: int,
    eps: float,
    training: bool,
    dropout_ratio: float,
    seed: Optional[int] = None,
    silu_u: bool = False,
    concat_ux: bool = False,
    compute_y: bool = False,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]:
    y = None
    N, D = x.shape
    if compute_y:
        if concat_ux:
            y = torch.empty((N, 3 * D), dtype=x.dtype, device=x.device)
        else:
            y = torch.empty_like(x)
    if N == 0:
        return (
            torch.zeros_like(x),
            torch.zeros_like(u),
            torch.zeros((D,), dtype=weight.dtype, device=x.device),
            torch.zeros((D,), dtype=weight.dtype, device=x.device),
            y,
        )
    dx = torch.empty_like(x)
    du = torch.empty_like(u)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    tile_num = max(1, min(sms * 64, N // 4))
    _dweight = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
    dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)

    # pyre-ignore[28]
    _helion_ln_mul_dropout_bwd_dx_du[(tile_num,)](
        dx,
        du,
        dy,
        _dweight,
        _dbias,
        x,
        u,
        y,
        weight,
        bias,
        mean,
        rstd,
        dx.stride(0),
        du.stride(0),
        dy.stride(0),
        x.stride(0),
        u.stride(0),
        y.stride(0) if compute_y else 0,  # pyre-ignore [16]
        D,
        eps,
        seed,
        dropout_ratio,
        N=N,
        SILU_U=silu_u,
        BLOCK_D=BLOCK_D,
        TRAINING=training,
        CONCAT_UX=concat_ux,
        COMPUTE_Y=compute_y,
        num_warps=num_warps,
    )

    blocks = triton.next_power_of_2(sms * 4)
    BLOCK_D_DWDB = triton.next_power_of_2(triton.cdiv(D, blocks))
    BLOCK_D_DWDB = min(max(BLOCK_D_DWDB, 4), 128)
    _helion_ln_mul_dropout_bwd_dwdb[(triton.cdiv(D, BLOCK_D_DWDB),)](
        _dweight,
        _dbias,
        dweight,
        dbias,
        tile_num,
        D,
        BLOCK_D=BLOCK_D_DWDB,
    )
    return dx, du, dweight, dbias, y


class HelionLayerNormMulDropoutFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        u: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        dropout_ratio: float,
        training: bool,
        silu_u: bool = False,
        concat_ux: bool = False,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if dropout_ratio == 0.0:
            # skip dropout computation if dropout ratio is 0
            training = False
        y, mean, rstd, BLOCK_D, num_warps, seed = helion_layer_norm_mul_dropout_fwd(
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            eps=eps,
            dropout_ratio=dropout_ratio,
            training=training,
            silu_u=silu_u,
            concat_ux=concat_ux,
            seed=seed,
        )
        ctx.save_for_backward(x, u, weight, bias, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.seed = seed
        ctx.training = training
        ctx.silu_u = silu_u
        ctx.concat_ux = concat_ux
        ctx.dropout_ratio = dropout_ratio
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        x, u, weight, bias, mean, rstd = ctx.saved_tensors
        dx, du, dweight, dbias, _ = helion_layer_norm_mul_dropout_bwd(
            dy=dy,
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            BLOCK_D=ctx.BLOCK_D,
            num_warps=ctx.num_warps,
            eps=ctx.eps,
            training=ctx.training,
            dropout_ratio=ctx.dropout_ratio,
            seed=ctx.seed,
            silu_u=ctx.silu_u,
            concat_ux=ctx.concat_ux,
            compute_y=False,
        )
        return dx, du, dweight, dbias, None, None, None, None, None, None


@torch.fx.wrap
def helion_norm_mul_dropout(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_ux: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    return HelionLayerNormMulDropoutFunction.apply(
        x, u, weight, bias, eps, dropout_ratio, training, silu_u, concat_ux, seed
    )


@torch.fx.wrap
def triton_norm_mul_dropout(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_u: bool = False,
    concat_x: bool = False,
    group_norm: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
    seed: Optional[int] = None,
) -> torch.Tensor:
    if group_norm:
        return GroupNormMulDropoutFunction.apply(
            x,
            u,
            weight,
            bias,
            eps,
            dropout_ratio,
            training,
            silu_u,
            concat_u and concat_x,
            num_heads,
            linear_dim,
            seed,
        )
    else:
        return LayerNormMulDropoutFunction.apply(
            x,
            u,
            weight,
            bias,
            eps,
            dropout_ratio,
            training,
            silu_u,
            concat_u and concat_x,
            seed,
        )


@torch.jit.unused
@torch.fx.wrap
def triton_hstu_compute_output(
    attn: torch.Tensor,
    u: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    output_weight: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_u: bool = False,
    concat_x: bool = False,
    mul_u_activation_type: str = "none",
    group_norm: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
    seed: Optional[int] = None,
    recompute_y_in_backward: bool = False,
    rng_row_indices: Optional[torch.Tensor] = None,
    fp8_output_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # DLRM_HSTU_FP8_EVAL_FWD_HYGIENE: decided here, not in the Function's
    # forward (grad mode is off in there). Training forwards never read it.
    need_bwd = backward_possible(
        attn,
        u,
        x,
        norm_weight,
        norm_bias,
        output_weight,
        rng_row_indices,
        fp8_output_scale,
    ) or not eval_fwd_hygiene_enabled()
    return HSTUComputeOutputFunction.apply(
        attn,
        u,
        x,
        norm_weight,
        norm_bias,
        output_weight,
        eps,
        dropout_ratio,
        training,
        silu_u,
        concat_u,
        concat_x,
        mul_u_activation_type,
        group_norm,
        num_heads,
        linear_dim,
        seed,
        recompute_y_in_backward,
        rng_row_indices,
        fp8_output_scale,
        need_bwd,
    )


# =============================================================================
# DLRM_HSTU_FP8_MASKGEN_PK (MGPK, yambda_5b.gin item (74); code default off). Appended
# at the END of the module on purpose: no @triton.jit function defined above
# moves, so every shipped kernel keeps its starting line and therefore its
# Triton cache_key (JITCallable hash = source + starting line), its disk-cache
# key and its TSPEC fn_cache_key. The only change above is the one-line hook
# in _triton_layer_norm_mul_dropout_fwd_impl (it replaced a "# pyre-ignore[28]"
# comment line, so no line moved), which calls _lnmd_fwd_maskgen_pk_try below.
# With DLRM_HSTU_FP8_MASKGEN_PK absent or "0" that call returns False after
# one lazy environment read per process and the shipped
# _ln_mul_dropout_fwd_maskgen launch runs unchanged; the kernels below are
# defined (triton.jit wrappers) but never compiled or launched.
# =============================================================================
import math  # noqa: E402
import struct  # noqa: E402

# ---------------------------------------------------------------------------
# DLRM_HSTU_FP8_MASKGEN_PK (MGPK; Arbor r16 iteration 35, ship-12 port in
# iteration 36; absent or =0 = the shipped _ln_mul_dropout_fwd_maskgen above,
# whose source this block leaves untouched). _ln_mul_dropout_fwd_maskgen_pk is
# a bitwise VALU diet of that kernel: same launch arguments, grid, Philox
# counters and per-row arithmetic, same int8 mask / e4m3 y / mean / rstd bytes.
# Gate: _use_lnmd_fwd_maskgen_pk, consulted only on calls _use_lnmd_fwd_maskgen
# armed. gfx950 issue cost per wave64 instruction (r16 iteration-35
# micro-benchmark, relative): v_and / v_mul / v_fma class ~1; v_lshlrev_b32,
# v_bfe, v_perm, v_lshl_or, v_cndmask_e64, DPP, 64-bit and packed-FP32
# (v_pk_*_f32) ~2; v_exp / v_rcp / v_sqrt and the fp8 converts ~4. Packing two
# fp32 ops into one v_pk_* therefore saves little; most of the gain is removing
# the 24 v_bfe_i32 + 24 v_and per row of the fp32-domain dropout (applied on
# the e4m3 bytes instead).


@triton.jit
def _mg_sp(t, H: tl.constexpr):
    """[1, 2H] -> ([1, H] even columns, [1, H] odd columns): a register split inside each lane."""
    return tl.split(tl.reshape(t, [1, H, 2]))


@triton.jit
def _mg_p2(lo, hi):
    """two fp32 tensors -> uint64 (lo in bits 0-31): one 64-bit VGPR pair."""
    return lo.to(tl.uint32, bitcast=True).to(tl.uint64) | (hi.to(tl.uint32, bitcast=True).to(tl.uint64) << 32)


@triton.jit
def _mg_bc(s):
    """fp32 scalar -> uint64 with the value in the low dword (consumers broadcast it with op_sel_hi)."""
    return s.to(tl.uint32, bitcast=True).to(tl.uint64)


@triton.jit
def _mg_lo(p):
    return p.to(tl.uint32).to(tl.float32, bitcast=True)


@triton.jit
def _mg_hi(p):
    return (p >> 32).to(tl.uint32).to(tl.float32, bitcast=True)


@triton.jit
def _mg_pk_mul(a, b):
    """(a.lo * b.lo, a.hi * b.hi): IEEE RNE per half, the bits of two v_mul_f32."""
    return tl.inline_asm_elementwise("v_pk_mul_f32 $0, $1, $2", "=v,v,v", [a, b],
                                     dtype=tl.uint64, is_pure=True, pack=1)


@triton.jit
def _mg_pk_mul_b(a, s):
    """(a.lo * s.lo, a.hi * s.lo)"""
    return tl.inline_asm_elementwise("v_pk_mul_f32 $0, $1, $2 op_sel_hi:[1,0]", "=v,v,v", [a, s],
                                     dtype=tl.uint64, is_pure=True, pack=1)


@triton.jit
def _mg_pk_nmul_b(a, s):
    """((-a.lo) * s.lo, (-a.hi) * s.lo): the shipped -u * log2(e) (v_mul_f32_e64 -u, c) per half."""
    return tl.inline_asm_elementwise("v_pk_mul_f32 $0, $1, $2 op_sel_hi:[1,0] neg_lo:[1,0] neg_hi:[1,0]",
                                     "=v,v,v", [a, s], dtype=tl.uint64, is_pure=True, pack=1)


@triton.jit
def _mg_pk_fma(a, b, c):
    return tl.inline_asm_elementwise("v_pk_fma_f32 $0, $1, $2, $3", "=v,v,v,v", [a, b, c],
                                     dtype=tl.uint64, is_pure=True, pack=1)


@triton.jit
def _mg_pk_add1(a):
    """a + 1.0 per half (the shipped kernel: the same v_pk_add_f32 ..., 1.0 op_sel_hi:[1,0])."""
    return tl.inline_asm_elementwise("v_pk_add_f32 $0, $1, 1.0 op_sel_hi:[1,0]", "=v,v", [a],
                                     dtype=tl.uint64, is_pure=True, pack=1)


@triton.jit
def _mg_pk_nfma1(d, r):
    """fma(-d, r, 1.0) per half with -d as a neg modifier. The shipped kernel materializes 0 - d first (one
    more v_pk_add_f32); identical for every d the fast path keeps (d in [1, 2^126]); any other d takes the
    wave-uniform tl.sigmoid fallback, which discards this chain."""
    return tl.inline_asm_elementwise(
        "v_pk_fma_f32 $0, $1, $2, 1.0 op_sel_hi:[1,1,0] neg_lo:[1,0,0] neg_hi:[1,0,0]", "=v,v,v", [d, r],
        dtype=tl.uint64, is_pure=True, pack=1)


@triton.jit
def _mg_perm(s0, s1, sel):
    return tl.inline_asm_elementwise("v_perm_b32 $0, $1, $2, $3", "=v,v,v,s", [s0, s1, sel],
                                     dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _mg_or_and_s(a, m, b):
    """(a | b) & m in one v_bitop3_b32 (table 0xC8 = (S0 | S2) & S1, symmetric in S0 / S2); m in an SGPR."""
    return tl.inline_asm_elementwise("v_bitop3_b32 $0, $1, $2, $3 bitop3:0xc8", "=v,v,s,v", [a, m, b],
                                     dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _mg_bytemask(m, sel, J: tl.constexpr):
    """byte k = 0xFF if bit J of byte k of the mask dword m is set, else 0x00. V = (m, m << 8) as a 64-bit
    pair; V << (7 - J) puts bit J of bytes 1/3 at bit 7 of the low dword's bytes 1/3 and bit J of bytes 0/2 at
    bit 7 of the high dword's bytes 1/3 (plus <= 2 spilled bits of m in high-dword bits 0-1, never read);
    v_perm_b32 codes 8-11 (sel = 0x090B080A) broadcast those sign bits to whole bytes."""
    v = m.to(tl.uint32, bitcast=True).to(tl.uint64) | ((m << 8).to(tl.uint32, bitcast=True).to(tl.uint64) << 32)
    if J == 0:
        v = tl.inline_asm_elementwise("v_lshlrev_b64 $0, 7, $1", "=v,v", [v], dtype=tl.uint64, is_pure=True, pack=1)
    elif J == 1:
        v = tl.inline_asm_elementwise("v_lshlrev_b64 $0, 6, $1", "=v,v", [v], dtype=tl.uint64, is_pure=True, pack=1)
    else:
        v = tl.inline_asm_elementwise("v_lshlrev_b64 $0, 5, $1", "=v,v", [v], dtype=tl.uint64, is_pure=True, pack=1)
    s1 = v.to(tl.uint32).to(tl.int32, bitcast=True)
    s0 = (v >> 32).to(tl.uint32).to(tl.int32, bitcast=True)
    return tl.inline_asm_elementwise("v_perm_b32 $0, $1, $2, $3", "=v,v,v,v", [s0, s1, sel],
                                     dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _mg_cvt4(a, b, c, d):
    """4 fp32 -> 4 e4m3 bytes (a in byte 0): the instruction pair Triton's own e4m3 conversion emits."""
    return tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp8_f32 $0, $1, $2, 1.0\n\tv_cvt_scalef32_pk_fp8_f32 $0, $3, $4, 1.0 op_sel:[0,0,0,1]",
        "=&v,v,v,v,v", [a, b, c, d], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _mg_fp8_sat_mode(g):
    """MODE bit 23 = 1 (gfx950 fp8 conversions saturate overflow to +-448), which Triton's own e4m3
    conversion sets at kernel entry; needed because this kernel converts through _mg_cvt4 only."""
    return tl.inline_asm_elementwise("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1", "=s,s", [g],
                                     dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _mg_pk_word(pk, r, kk2, one2, sh):
    """_pk_word with the merge as one v_lshl_or_b32 (the shipped kernel: v_lshlrev_b32 + v_or3_b32; gfx950
    issues v_lshlrev_b32 at half rate). _pair_keep_bits unchanged; sh is the SHIFT constant (SGPR). Same
    pk bits."""
    pb = _pair_keep_bits(r.to(tl.int32, bitcast=True), kk2, one2)
    return tl.inline_asm_elementwise("v_lshl_or_b32 $0, $1, $2, $3", "=v,v,s,v", [pb, sh, pk],
                                     dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _row_c32_pk(
    x_n, u_n, pk, base, row, N, Y, Mean, Rstd, RANDOM_MASK, Wp, Bp, cols4,
    inv_D, eps, DSp, INVp, L2Ep, BMSEL, D, stride_y, stride_mask,
    BLOCK_D: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
    SILU_U: tl.constexpr,
):
    """One row of _ln_mul_dropout_fwd_maskgen_pk: the bytes _row_c32 writes (the generator's int8 mask byte;
    _ln_mul_dropout_fwd_rng_v3's y / mean / rstd), MGPK form:
      - the int8 mask row is built as 2 dwords per lane: (pk >> base), byte 0 of 4 columns gathered by
        two v_perm_b32, merged and masked to 3 bits by one v_bitop3, stored as int32;
      - mean / var / rstd: the shipped expressions unchanged (same reduction order and contraction);
      - per-element fp32 arithmetic on adjacent-column pairs in v_pk_mul_f32 / v_pk_fma_f32 (same IEEE op
        per element: x_mean * rstd, fma(y, w, b), the fast sigmoid, u * sig, y * silu, * dropout_scale,
        * Y_INV_SCALE), the fast sigmoid's -d folded into neg modifiers;
      - dropout applied on the e4m3 bytes: fp8(v * ds * inv) & bytemask, which equals the shipped
        fp8(((v * ds) & keep) * inv) because a dropped element is +0.0 there and +0.0 * inv = +0.0 -> 0x00
        for every fp32 inv in (0, inf) (the only values _use_lnmd_fwd_maskgen_pk lets through)."""
    H: tl.constexpr = BLOCK_D // 2
    Q4: tl.constexpr = BLOCK_D // 4
    tl.static_assert(MUL_U_ACTIVATION_TYPE == "silu")
    tl.static_assert(OUTPUT_FP8)
    rows = row + tl.arange(0, 1)
    row_mask = rows < N
    r64 = rows.to(tl.int64)
    p32 = pk >> base
    pa, pb = _mg_sp(p32, H)
    p0, p2 = _mg_sp(pa, Q4)
    p1, p3 = _mg_sp(pb, Q4)
    mdw = _mg_or_and_s(_mg_perm(p1, p0, 0x0C0C0400), 0x07070707, _mg_perm(p3, p2, 0x04000C0C))
    tl.store(RANDOM_MASK.to(tl.pointer_type(tl.int32)) + r64[:, None] * (stride_mask // 4) + cols4,
             mdw, mask=row_mask[:, None])
    x_block = x_n.to(tl.float32)
    u_block = u_n.to(tl.float32)
    mean = tl.sum(x_block, axis=1) * inv_D
    tl.store(Mean + rows, mean, mask=row_mask)
    mean = tl.expand_dims(mean, 1)
    x_mean = x_block - mean
    _var = x_mean * x_mean
    var = tl.sum(_var, axis=1) * inv_D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + rows, rstd, mask=row_mask)
    rstd = tl.expand_dims(rstd, 1)
    # _sigmoid_ieee_fast on column pairs (same ops per element), wave-uniform tl.sigmoid fallback
    ue, uo = _mg_sp(u_block, H)
    U = _mg_p2(ue, uo)
    NE = _mg_pk_nmul_b(U, L2Ep)
    E = _mg_p2(_amd_v_exp_f32(_mg_lo(NE)), _amd_v_exp_f32(_mg_hi(NE)))
    Dd = _mg_pk_add1(E)
    R = _mg_p2(_amd_v_rcp_f32(_mg_lo(Dd)), _amd_v_rcp_f32(_mg_hi(Dd)))
    T = _mg_pk_nfma1(Dd, R)
    R = _mg_pk_fma(T, R, R)
    T = _mg_pk_nfma1(Dd, R)
    Q = _mg_pk_fma(T, R, R)
    T = _mg_pk_nfma1(Dd, Q)
    SIG = _mg_pk_fma(T, R, Q)
    # d bits as uint32 > 0x7E800000 <=> d > 2^126, +inf or NaN (any sign)
    dmax = tl.max(tl.maximum(_mg_lo(Dd).to(tl.uint32, bitcast=True), _mg_hi(Dd).to(tl.uint32, bitcast=True)))
    if dmax > 0x7E800000:
        sge, sgo = _mg_sp(tl.sigmoid(u_block), H)
        SIG = _mg_p2(sge, sgo)
    xme, xmo = _mg_sp(x_mean, H)
    Yp = _mg_pk_mul_b(_mg_p2(xme, xmo), _mg_bc(rstd))  # y = x_mean * rstd
    Yp = _mg_pk_fma(Yp, Wp, Bp)  # y = y * w + b
    SIL = _mg_pk_mul(U, SIG)  # silu = u * sigmoid(u)
    Yp = _mg_pk_mul(Yp, SIL)  # y = y * silu
    if SILU_U:
        UP = SIL
    else:
        UP = U
    xe, xo = _mg_sp(x_block, H)
    UF = _mg_pk_mul_b(_mg_pk_mul_b(UP, DSp), INVp)  # (u * dropout_scale) * Y_INV_SCALE
    XF = _mg_pk_mul_b(_mg_pk_mul_b(_mg_p2(xe, xo), DSp), INVp)
    YF = _mg_pk_mul_b(_mg_pk_mul_b(Yp, DSp), INVp)
    ua, ub = _mg_sp(UF, Q4)
    xa, xb = _mg_sp(XF, Q4)
    ya, yb = _mg_sp(YF, Q4)
    u8 = _mg_cvt4(_mg_lo(ua), _mg_hi(ua), _mg_lo(ub), _mg_hi(ub)) & _mg_bytemask(mdw, BMSEL, 2)
    x8 = _mg_cvt4(_mg_lo(xa), _mg_hi(xa), _mg_lo(xb), _mg_hi(xb)) & _mg_bytemask(mdw, BMSEL, 1)
    y8 = _mg_cvt4(_mg_lo(ya), _mg_hi(ya), _mg_lo(yb), _mg_hi(yb)) & _mg_bytemask(mdw, BMSEL, 0)
    y32 = Y.to(tl.pointer_type(tl.int32)) + r64[:, None] * (stride_y // 4) + cols4
    tl.store(y32, u8, mask=row_mask[:, None])
    tl.store(y32 + D // 4, x8, mask=row_mask[:, None])
    tl.store(y32 + D // 2, y8, mask=row_mask[:, None])


@triton.jit
def _ln_mul_dropout_fwd_maskgen_pk(
    X, U, Y, W, B, Mean, Rstd, RANDOM_MASK,
    N, D, eps, dropout_ratio,
    stride_x, stride_u, stride_y, stride_mask,
    Y_INV_SCALE,
    keep_min16, kk2_arg, seed,
    BLOCK_D: tl.constexpr,
    NUM_MASKS: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
    SILU_U: tl.constexpr,
):
    """_ln_mul_dropout_fwd_maskgen (above) with a bitwise VALU diet: the same arguments, grid cdiv(N, 8),
    num_warps=1 and contract (MUL_U_ACTIVATION_TYPE 'silu' and OUTPUT_FP8 are asserted here; NUM_MASKS=3,
    BLOCK_D == D == 512 as the shipped gate guarantees), the same Philox counters, keep decisions and row
    arithmetic, and the same int8 mask, e4m3 y, mean and rstd bytes. Value-preserving differences:
      F8M: the keep masks are applied on the e4m3 bytes, fp8(v * ds * inv) & bytemask, instead of
        fp8(((v * ds) & keep) * inv): a dropped element is +0.0 in the shipped kernel and +0.0 * inv =
        +0.0 -> 0x00 for every fp32 inv in (0, inf), the only Y_INV_SCALE values the gate lets through; a
        kept element runs the same IEEE multiplies and the same conversion instruction.
      CVT: the e4m3 bytes come from v_cvt_scalef32_pk_fp8_f32 (scale 1.0), the instructions Triton's own
        fp32 -> e4m3 conversion emits, with the MODE bit 23 (saturate) that Triton sets at kernel entry for
        them set explicitly (_mg_fp8_sat_mode; without it e4m3 overflow gives 0x7F instead of 0x7E).
      MDP: the int8 mask row is assembled per lane as 2 dwords and stored as int32, the y planes likewise
        (the gate checks dword alignment of the base pointers and row strides).
      PKF: the per-element fp32 multiplies / FMAs run on adjacent-column pairs in v_pk_mul_f32 /
        v_pk_fma_f32 (the same IEEE operation per element); mean / var / rstd keep the shipped expressions.
      LSO: the Philox keep-bit merge is one v_lshl_or_b32 per word.
      UNR2: two rows per loop trip: row i + 1's x/u are issued before row i's math, row i + 2's after it
        (into the registers row i freed: no loop-carried register rotation).
    Same counter limit as the shipped kernel (N < 11,184,810; the shipped gate refuses larger N).
    Iteration-35 build (Triton 3.6, gfx950): 112 VGPRs / 71 SGPRs, occupancy 4, no LDS, no scratch (the
    shipped kernel: 104 / 58, occupancy 4)."""
    g = tl.program_id(0)
    H: tl.constexpr = BLOCK_D // 2
    Q4: tl.constexpr = BLOCK_D // 4
    _mg_fp8_sat_mode(g)  # e4m3 bytes come from _mg_cvt4 only (see _mg_fp8_sat_mode)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < BLOCK_D
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask).to(tl.float32)
    inv_D = 1.0 / D
    dropout_scale = 1.0 / (1.0 - dropout_ratio)
    row0 = g * 8
    # loop invariants: column-pair w / b, broadcast scalars, constants
    we, wo = _mg_sp(w[None, :], H)
    be, bo = _mg_sp(b[None, :], H)
    Wp = _mg_p2(we, wo)
    Bp = _mg_p2(be, bo)
    DSp = _mg_bc(dropout_scale)
    INVp = _mg_bc(Y_INV_SCALE)
    L2Ep = tl.full([1, H], 0x3FB8AA3B, tl.uint64)  # fp32 1.4426950408889634 (_sigmoid_ieee_fast's constant)
    BMSEL = tl.full([1, Q4], 0x090B080A, tl.int32)
    cols4 = tl.arange(0, Q4)[None, :]  # int32 dword index of columns 4a..4a+3 (2 dwords per lane)

    # row 0's x/u, issued before the Philox phase
    xa, ua = _load_row(X, U, row0, N, stride_x, stride_u, cols, col_mask)

    # ---- phase 1: the generator's Philox for this 8-row group, packed per column ----
    kk2 = tl.full([1, BLOCK_D], 0, tl.int32) + kk2_arg
    one2 = tl.full([1, BLOCK_D], 0x10001, tl.int32)
    pk = tl.zeros([1, BLOCK_D], dtype=tl.int32)
    rand_offset = g * (NUM_MASKS * BLOCK_D) + cols[None, :]
    # layout anchor (see _ln_mul_dropout_fwd_maskgen): the counters "depend" on the row-0 x load
    rand_offset = tl.inline_asm_elementwise(
        "v_mov_b32 $0, $1", "=v,v,v", [rand_offset, xa], dtype=tl.int32, is_pure=True, pack=1
    )
    for j in tl.static_range(NUM_MASKS):
        r0, r1, r2, r3 = _philox4x32_10_b3(rand_offset, seed)
        pk = _mg_pk_word(pk, r0, kk2, one2, 0 + j)
        pk = _mg_pk_word(pk, r1, kk2, one2, 3 + j)
        pk = _mg_pk_word(pk, r2, kk2, one2, 6 + j)
        pk = _mg_pk_word(pk, r3, kk2, one2, 9 + j)
        rand_offset += BLOCK_D

    # ---- phase 2: rows 0..7 of the group, two rows per trip ----
    for i in tl.range(0, 8, 2, num_stages=1):
        rows_b = row0 + i + 1 + tl.arange(0, 1)
        m2_b = (rows_b < N)[:, None] & col_mask[None, :]
        r64_b = rows_b.to(tl.int64)
        xb = tl.load(X + r64_b[:, None] * stride_x + cols[None, :], mask=m2_b, other=0.0)
        ub = tl.load(U + r64_b[:, None] * stride_u + cols[None, :], mask=m2_b, other=0.0)
        # row i (0..3): bits 3i..3i+2; row i (4..7): bits 16+3(i-4)..
        _row_c32_pk(
            xa, ua, pk, 3 * (i % 4) + 16 * (i // 4), row0 + i, N, Y, Mean, Rstd, RANDOM_MASK, Wp, Bp, cols4,
            inv_D, eps, DSp, INVp, L2Ep, BMSEL, D, stride_y, stride_mask,
            BLOCK_D, MUL_U_ACTIVATION_TYPE, OUTPUT_FP8, SILU_U,
        )
        rows_n = row0 + i + 2 + tl.arange(0, 1)
        m2_n = ((rows_n < N) & (i < 6))[:, None] & col_mask[None, :]
        r64_n = rows_n.to(tl.int64)
        xa = tl.load(X + r64_n[:, None] * stride_x + cols[None, :], mask=m2_n, other=0.0)
        ua = tl.load(U + r64_n[:, None] * stride_u + cols[None, :], mask=m2_n, other=0.0)
        _row_c32_pk(
            xb, ub, pk, 3 * ((i + 1) % 4) + 16 * ((i + 1) // 4), row0 + i + 1, N, Y, Mean, Rstd, RANDOM_MASK,
            Wp, Bp, cols4, inv_D, eps, DSp, INVp, L2Ep, BMSEL, D, stride_y, stride_mask,
            BLOCK_D, MUL_U_ACTIVATION_TYPE, OUTPUT_FP8, SILU_U,
        )


# DLRM_HSTU_FP8_MASKGEN_PK (MGPK), read once at the first call of its gate (the
# gin env bootstrap sets it after import). Plain scalars only: the FP8 first-use
# warmup (dlrm_v4/train/fp8_warmup.py, _SCALAR_MODULES) snapshots and restores
# them, so the arming line prints at the real first use.
_LNMD_FWD_MASKGEN_PK_ENABLED: Optional[bool] = None
_LNMD_FWD_MASKGEN_PK_ARMED_LOGGED = False
_LNMD_FWD_MASKGEN_PK_DECLINE_LOGGED = False
# Set (with one line on every rank) when a _ln_mul_dropout_fwd_maskgen_pk launch
# raised: the shipped kernel runs for the rest of the process.
_LNMD_FWD_MASKGEN_PK_BROKEN: Optional[str] = None
# DLRM_HSTU_FP8_MASKGEN_PK_LAYERS (MGPK layer list), read together with the
# knob: "" = every STU layer; otherwise the canonical ",i,j," form of the
# zero-based STUStack indices on which MGPK may run. On any other layer the
# shipped _ln_mul_dropout_fwd_maskgen runs (the same bytes either way, so the
# list only chooses which kernel spends the time).
_LNMD_FWD_MASKGEN_PK_LAYERS: Optional[str] = None
_LNMD_FWD_MASKGEN_PK_GATED_LOGGED = False
# Zero-based STUStack index of the STU layer whose compute output is running:
# modules/stu.py STULayer._tag_lnmd_layer sets it right before each of that
# layer's hstu_compute_output calls (forward and targets-only forward). -1 =
# untagged (a direct call, the FP8 first-use warmup, a layer outside a stack),
# which every layer list accepts.
_LNMD_LAYER_TAG: int = -1


def set_lnmd_layer_tag(index: int) -> None:
    """STULayer: the LN*U*dropout forward that follows serves STUStack layer
    `index` (-1 = not owned by a stack). Read only by the MGPK layer list."""
    global _LNMD_LAYER_TAG
    _LNMD_LAYER_TAG = int(index)


def _lnmd_fwd_maskgen_pk_layers() -> str:
    """DLRM_HSTU_FP8_MASKGEN_PK_LAYERS -> "" (unset or empty: every layer) or
    ",i,j," (sorted, de-duplicated). A malformed list (a token that is not a
    non-negative integer; the env bootstrap already rejects one at startup)
    latches MGPK off for the process with one line on every rank: the shipped
    kernel then runs everywhere (bitwise either way)."""
    global _LNMD_FWD_MASKGEN_PK_BROKEN
    raw = os.environ.get("DLRM_HSTU_FP8_MASKGEN_PK_LAYERS", "")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return ""
    if not all(t.isascii() and t.isdigit() for t in tokens):
        _LNMD_FWD_MASKGEN_PK_BROKEN = (
            f"malformed DLRM_HSTU_FP8_MASKGEN_PK_LAYERS={raw!r} (need "
            "comma-separated zero-based STU layer indices)"
        )
        # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
        arm_print_all_ranks(
            "LN*U*dropout fwd maskgen pk DISABLED (DLRM_HSTU_FP8_MASKGEN_PK): "
            f"{_LNMD_FWD_MASKGEN_PK_BROKEN}; the shipped "
            "_ln_mul_dropout_fwd_maskgen runs for the rest of the process"
        )
        return ""
    return "," + ",".join(str(i) for i in sorted({int(t) for t in tokens})) + ","


def _fp32_kernel_arg(value: float) -> float:
    """value as the fp32 kernel argument a Python float becomes in a Triton
    launch: the C (float) conversion, round-to-nearest-even, +-inf past the fp32
    range. struct's 'f' packing is that conversion and raises OverflowError
    exactly where it overflows (a finite double such as 1e39 becomes +inf)."""
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _use_lnmd_fwd_maskgen_pk(
    inv_scale: float, y: torch.Tensor, random_mask: torch.Tensor
) -> bool:
    """DLRM_HSTU_FP8_MASKGEN_PK (MGPK; absent or =0: the shipped
    _ln_mul_dropout_fwd_maskgen, exactly as without this item; any other value
    arms). Consulted only on calls _use_lnmd_fwd_maskgen armed (gfx950, D ==
    BLOCK_D == 512, concat u and x, 'silu', e4m3 output, mask16, ...), whose
    contract _ln_mul_dropout_fwd_maskgen_pk shares. True runs that kernel (same
    int8 mask, e4m3 y, mean and rstd bytes) when, for this call,
      - the fp32 Y_INV_SCALE the kernel receives (inv_scale as Triton converts
        a Python float, _fp32_kernel_arg) is in (0, inf): the kernel applies the
        dropout on the e4m3 bytes, exact because a dropped element is
        +0.0 * inv = +0.0 there. Testing the double instead would let e.g.
        1e39 through, which reaches the kernel as +inf;
      - y and the mask have dword-aligned base pointers and row strides (the
        kernel stores them as int32; always true for the launch site's fresh
        [N, 1536] / [N, 512] allocations);
      - no earlier launch of it raised (_LNMD_FWD_MASKGEN_PK_BROKEN);
      - the STU layer this call serves (_LNMD_LAYER_TAG) is on the
        DLRM_HSTU_FP8_MASKGEN_PK_LAYERS list (unset or empty: every layer; an
        untagged call, -1, is always on it).
    Otherwise False: the shipped kernel runs (bitwise either way); the first
    declined call prints one line on every rank, and so does the first call
    the layer list sends to the shipped kernel. The ACTIVE line is printed
    after the first successful launch (_lnmd_fwd_maskgen_pk_armed), not
    here."""
    global _LNMD_FWD_MASKGEN_PK_ENABLED, _LNMD_FWD_MASKGEN_PK_DECLINE_LOGGED
    global _LNMD_FWD_MASKGEN_PK_LAYERS, _LNMD_FWD_MASKGEN_PK_GATED_LOGGED
    if _LNMD_FWD_MASKGEN_PK_ENABLED is None:
        _LNMD_FWD_MASKGEN_PK_ENABLED = (
            os.environ.get("DLRM_HSTU_FP8_MASKGEN_PK", "0") != "0"
        )
        _LNMD_FWD_MASKGEN_PK_LAYERS = (
            _lnmd_fwd_maskgen_pk_layers() if _LNMD_FWD_MASKGEN_PK_ENABLED else ""
        )
    if not _LNMD_FWD_MASKGEN_PK_ENABLED or _LNMD_FWD_MASKGEN_PK_BROKEN is not None:
        return False
    if (
        _LNMD_FWD_MASKGEN_PK_LAYERS
        and _LNMD_LAYER_TAG >= 0
        and f",{_LNMD_LAYER_TAG}," not in _LNMD_FWD_MASKGEN_PK_LAYERS
    ):
        if not _LNMD_FWD_MASKGEN_PK_GATED_LOGGED:
            _LNMD_FWD_MASKGEN_PK_GATED_LOGGED = True
            # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
            arm_print_all_ranks(
                "LN*U*dropout fwd maskgen pk layer-gated "
                "(DLRM_HSTU_FP8_MASKGEN_PK_LAYERS="
                f"{_LNMD_FWD_MASKGEN_PK_LAYERS.strip(',')}): STU layer "
                f"{_LNMD_LAYER_TAG} runs the shipped _ln_mul_dropout_fwd_maskgen "
                "(bitwise either way; printed once)"
            )
        return False
    inv32 = _fp32_kernel_arg(float(inv_scale))
    why: Optional[str] = None
    if not 0.0 < inv32 < math.inf:
        why = (
            f"fp32 Y_INV_SCALE {inv32!r} (inv_scale {float(inv_scale)!r}) "
            "is not in (0, inf)"
        )
    elif (
        y.data_ptr() | random_mask.data_ptr() | y.stride(0) | random_mask.stride(0)
    ) & 3:
        why = (
            f"y / mask rows not dword-aligned (strides {y.stride(0)}, "
            f"{random_mask.stride(0)})"
        )
    if why is None:
        return True
    if not _LNMD_FWD_MASKGEN_PK_DECLINE_LOGGED:
        _LNMD_FWD_MASKGEN_PK_DECLINE_LOGGED = True
        # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
        arm_print_all_ranks(
            "LN*U*dropout fwd maskgen pk declined (DLRM_HSTU_FP8_MASKGEN_PK): "
            f"{why}; the shipped _ln_mul_dropout_fwd_maskgen runs such calls "
            "(bitwise either way; printed once)"
        )
    return False


def _lnmd_fwd_maskgen_pk_armed() -> None:
    """After the first successful _ln_mul_dropout_fwd_maskgen_pk launch: one
    line on every rank, so the line means the kernel really runs."""
    global _LNMD_FWD_MASKGEN_PK_ARMED_LOGGED
    if not _LNMD_FWD_MASKGEN_PK_ARMED_LOGGED:
        _LNMD_FWD_MASKGEN_PK_ARMED_LOGGED = True
        # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
        layers = (_LNMD_FWD_MASKGEN_PK_LAYERS or "").strip(",") or "all"
        arm_print_all_ranks(
            "armed OK: LN*U*dropout fwd maskgen pk ACTIVE "
            "(DLRM_HSTU_FP8_MASKGEN_PK): _ln_mul_dropout_fwd_maskgen_pk runs "
            "in place of _ln_mul_dropout_fwd_maskgen, bitwise (same int8 "
            f"mask, e4m3 y, mean, rstd); STU layers {layers}, first launch on "
            f"layer tag {_LNMD_LAYER_TAG}"
        )


def _lnmd_fwd_maskgen_pk_latch(error: BaseException) -> None:
    """A _ln_mul_dropout_fwd_maskgen_pk launch raised (e.g. a compile error):
    latch the item off for the process (one line on every rank); the caller
    then runs the shipped kernel, which writes every output byte."""
    global _LNMD_FWD_MASKGEN_PK_BROKEN
    _LNMD_FWD_MASKGEN_PK_BROKEN = f"{type(error).__name__}: {str(error)[:200]}"
    # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
    arm_print_all_ranks(
        "LN*U*dropout fwd maskgen pk DISABLED (DLRM_HSTU_FP8_MASKGEN_PK): "
        f"launch raised {_LNMD_FWD_MASKGEN_PK_BROKEN}; the shipped "
        "_ln_mul_dropout_fwd_maskgen runs for the rest of the process"
    )


def _lnmd_fwd_maskgen_pk_try(
    x: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    random_mask: torch.Tensor,
    N: int,
    D: int,
    eps: float,
    dropout_ratio: float,
    inv_scale: float,
    keep_min16: int,
    kk2: int,
    seed: int,
    BLOCK_D: int,
    concat_u: bool,
    concat_x: bool,
    mul_u_activation_type: str,
    output_fp8: bool,
    silu_u: bool,
) -> bool:
    """MGPK at item (28)(e)'s launch site: True when _ln_mul_dropout_fwd_maskgen_pk
    ran (the caller then returns y, mean, rstd, random_mask), False when the
    caller must run the shipped _ln_mul_dropout_fwd_maskgen launch (knob off,
    a declined or layer-gated call, a launch that raised and latched the item
    off, or an untagged call with a restricted layer list, which runs both)."""
    if not _use_lnmd_fwd_maskgen_pk(inv_scale, y, random_mask):
        return False
    # DLRM_HSTU_FP8_MASKGEN_PK: the same launch on the VALU-diet kernel (same
    # bytes). If it raises, the item latches off and the shipped launch writes
    # every output.
    try:
        # pyre-ignore[28]
        _ln_mul_dropout_fwd_maskgen_pk[(triton.cdiv(N, 8),)](
            x,
            u,
            y,
            weight,
            bias,
            mean,
            rstd,
            random_mask,
            N,
            D,
            eps,
            dropout_ratio,
            x.stride(0),
            u.stride(0),
            y.stride(0),
            random_mask.stride(0),
            float(inv_scale),
            keep_min16,
            kk2,
            seed,
            BLOCK_D=BLOCK_D,
            NUM_MASKS=1 + int(concat_u) + int(concat_x),
            MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
            OUTPUT_FP8=output_fp8,
            SILU_U=silu_u,
            num_warps=1,
        )
    except Exception as error:
        _lnmd_fwd_maskgen_pk_latch(error)
        return False
    _lnmd_fwd_maskgen_pk_armed()
    if _LNMD_LAYER_TAG < 0 and _LNMD_FWD_MASKGEN_PK_LAYERS:
        # An untagged call (the FP8 first-use warmup, a direct call) with a
        # restricted layer list: the training step runs BOTH kernels (the _pk
        # one on the listed layers, the shipped one on the others), so this
        # call also runs the shipped launch into the same buffers (it rewrites
        # the same bytes). The warmup thereby compiles both kernels' warm-up
        # specializations before RUN_START; tagged calls never take this branch.
        return False
    return True


# =============================================================================
# DLRM_HSTU_FP8_LNMD_MASK_NIB ("NIB": the LN*U*dropout keep mask in the nibble
# layout; r16 iteration 54, gin item (94); code default OFF,
# exactly "1" arms it). Appended at the END of the module, after the MGPK block
# (the same precedent): no @triton.jit function above moves, so every existing
# kernel keeps its starting line and therefore its Triton cache_key, its
# disk-cache key and its TSPEC fn_cache_key. The lines above that reach this
# block are IN-PLACE edits of existing lines (no physical line was inserted or
# deleted above this banner):
#   * _triton_layer_norm_mul_dropout_fwd_impl, item (28)(e)'s maskgen branch:
#     the int8 mask allocation line first asks _lnmd_fwd_maskgen_nib_try (NIB
#     armed and launched -> the uint8 [N, 256] mask, and no int8 [N, 512] mask
#     is allocated), and the MGPK hook line then returns that mask before MGPK /
#     the shipped maskgen launch;
#   * HSTUComputeOutputFunction.forward, FP8-output branch: one call,
#     _lnmd_mask_nib_note_call(attn, output_weight, need_bwd), records the
#     out-proj facts the predicate cannot see from inside the forward impl;
#   * fail-closed backward: _du_e5m2_plan's layout check accepts the NIB form;
#     every int8 consumer receives lnmd_mask_nib.as_int8(mask, where) (the
#     exact unpack, counted, cached on the mask; a no-op for an int8 mask): the
#     backward's L11 run (bf16-du) / B1a v3 / Triton v2 call sites and, as
#     defense in depth, _triton_layer_norm_mul_dropout_bwd_impl,
#     _triton_layer_norm_mul_dropout_bwd_du_e5m2 and _l11_du_e5m2_or_none (L11
#     run_e5). A NIB mask reaches only L11's D8 gate (which accepts it) and DO8
#     (l11_do8.run_e5_plus, or MX6's preflight and launch: the NIB 'only' / MX6
#     build reads it; a format mismatch is a decline there, never a latch).
# Knob absent or not "1": _lnmd_fwd_maskgen_nib_try and _lnmd_mask_nib_note_call
# return after one environment read, no NIB state is touched, and every call
# above runs exactly as before (same kernels, same lines, same int8 mask).
# =============================================================================
import atexit  # noqa: E402
from typing import Any, Dict  # noqa: E402

from generative_recommenders.ops import lnmd_mask_nib  # noqa: E402

# ---------------------------------------------------------------------------
# NIB layout (ops/lnmd_mask_nib.py, "layout A"): uint8 [N, 256], 256 B per row
# instead of the int8 [N, 512] mask. The writer lane that owns columns
# 8l..8l+7 (the L11 / DO8 reader reads the same 8 columns in lane l) stores ONE
# dword at byte 4l of the row: byte k = keep bits of column 8l+k (bit 0 y, 1 x,
# 2 u) in bits 0-2 | keep bits of column 8l+4+k in bits 4-6; bits 3 and 7 are
# 0. The keep decisions (Philox counters, seed, K16, _pair_keep_bits) and the
# y / mean / rstd arithmetic are untouched, so y / mean / rstd are the same
# bytes and lnmd_mask_nib.unpack(NIB) == the int8 mask byte for byte (the
# generator's bytes are c32 = (pk >> base) & 7 <= 7). The six kernels below are
# r16 iteration 41's maskpack writers, byte for byte (GPU-verified bitwise
# there, 314/314: S/probes/iter41/maskpack/README.md sec. 2); every function
# they call above is byte-identical to that tree's (r16 iteration 54,
# function-level compare: 30/30).
# ---------------------------------------------------------------------------


@triton.jit
def _mg_nib(mdw, Q8: tl.constexpr):
    """[1, 2 * Q8] MGPK mask dwords (lane l: dword 2l = columns 8l..8l+3, dword 2l+1 = columns 8l+4..8l+7, one
    3-bit field per byte) -> [1, Q8] NIB dwords = dword 2l | dword 2l+1 << 4: one v_lshl_or_b32 per row-lane."""
    ev, od = _mg_sp(mdw, Q8)
    return tl.inline_asm_elementwise("v_lshl_or_b32 $0, $1, 4, $2", "=v,v,v", [od, ev],
                                     dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _row_c32_pk_nib(
    x_n, u_n, pk, base, row, N, Y, Mean, Rstd, RANDOM_MASK, Wp, Bp, cols4, cols8,
    inv_D, eps, DSp, INVp, L2Ep, BMSEL, D, stride_y, stride_mask,
    BLOCK_D: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
    SILU_U: tl.constexpr,
):
    """_row_c32_pk with the keep mask stored as NIB (one int32 per row-lane at RANDOM_MASK + row * stride_mask +
    4 * lane, stride_mask = BLOCK_D // 2 bytes); every other expression is _row_c32_pk's."""
    H: tl.constexpr = BLOCK_D // 2
    Q4: tl.constexpr = BLOCK_D // 4
    Q8: tl.constexpr = BLOCK_D // 8
    tl.static_assert(MUL_U_ACTIVATION_TYPE == "silu")
    tl.static_assert(OUTPUT_FP8)
    rows = row + tl.arange(0, 1)
    row_mask = rows < N
    r64 = rows.to(tl.int64)
    p32 = pk >> base
    pa, pb = _mg_sp(p32, H)
    p0, p2 = _mg_sp(pa, Q4)
    p1, p3 = _mg_sp(pb, Q4)
    mdw = _mg_or_and_s(_mg_perm(p1, p0, 0x0C0C0400), 0x07070707, _mg_perm(p3, p2, 0x04000C0C))
    tl.store(RANDOM_MASK.to(tl.pointer_type(tl.int32)) + r64[:, None] * (stride_mask // 4) + cols8,
             _mg_nib(mdw, Q8), mask=row_mask[:, None])
    x_block = x_n.to(tl.float32)
    u_block = u_n.to(tl.float32)
    mean = tl.sum(x_block, axis=1) * inv_D
    tl.store(Mean + rows, mean, mask=row_mask)
    mean = tl.expand_dims(mean, 1)
    x_mean = x_block - mean
    _var = x_mean * x_mean
    var = tl.sum(_var, axis=1) * inv_D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + rows, rstd, mask=row_mask)
    rstd = tl.expand_dims(rstd, 1)
    # _sigmoid_ieee_fast on column pairs (same ops per element), wave-uniform tl.sigmoid fallback
    ue, uo = _mg_sp(u_block, H)
    U = _mg_p2(ue, uo)
    NE = _mg_pk_nmul_b(U, L2Ep)
    E = _mg_p2(_amd_v_exp_f32(_mg_lo(NE)), _amd_v_exp_f32(_mg_hi(NE)))
    Dd = _mg_pk_add1(E)
    R = _mg_p2(_amd_v_rcp_f32(_mg_lo(Dd)), _amd_v_rcp_f32(_mg_hi(Dd)))
    T = _mg_pk_nfma1(Dd, R)
    R = _mg_pk_fma(T, R, R)
    T = _mg_pk_nfma1(Dd, R)
    Q = _mg_pk_fma(T, R, R)
    T = _mg_pk_nfma1(Dd, Q)
    SIG = _mg_pk_fma(T, R, Q)
    # d bits as uint32 > 0x7E800000 <=> d > 2^126, +inf or NaN (any sign)
    dmax = tl.max(tl.maximum(_mg_lo(Dd).to(tl.uint32, bitcast=True), _mg_hi(Dd).to(tl.uint32, bitcast=True)))
    if dmax > 0x7E800000:
        sge, sgo = _mg_sp(tl.sigmoid(u_block), H)
        SIG = _mg_p2(sge, sgo)
    xme, xmo = _mg_sp(x_mean, H)
    Yp = _mg_pk_mul_b(_mg_p2(xme, xmo), _mg_bc(rstd))  # y = x_mean * rstd
    Yp = _mg_pk_fma(Yp, Wp, Bp)  # y = y * w + b
    SIL = _mg_pk_mul(U, SIG)  # silu = u * sigmoid(u)
    Yp = _mg_pk_mul(Yp, SIL)  # y = y * silu
    if SILU_U:
        UP = SIL
    else:
        UP = U
    xe, xo = _mg_sp(x_block, H)
    UF = _mg_pk_mul_b(_mg_pk_mul_b(UP, DSp), INVp)  # (u * dropout_scale) * Y_INV_SCALE
    XF = _mg_pk_mul_b(_mg_pk_mul_b(_mg_p2(xe, xo), DSp), INVp)
    YF = _mg_pk_mul_b(_mg_pk_mul_b(Yp, DSp), INVp)
    ua, ub = _mg_sp(UF, Q4)
    xa, xb = _mg_sp(XF, Q4)
    ya, yb = _mg_sp(YF, Q4)
    u8 = _mg_cvt4(_mg_lo(ua), _mg_hi(ua), _mg_lo(ub), _mg_hi(ub)) & _mg_bytemask(mdw, BMSEL, 2)
    x8 = _mg_cvt4(_mg_lo(xa), _mg_hi(xa), _mg_lo(xb), _mg_hi(xb)) & _mg_bytemask(mdw, BMSEL, 1)
    y8 = _mg_cvt4(_mg_lo(ya), _mg_hi(ya), _mg_lo(yb), _mg_hi(yb)) & _mg_bytemask(mdw, BMSEL, 0)
    y32 = Y.to(tl.pointer_type(tl.int32)) + r64[:, None] * (stride_y // 4) + cols4
    tl.store(y32, u8, mask=row_mask[:, None])
    tl.store(y32 + D // 4, x8, mask=row_mask[:, None])
    tl.store(y32 + D // 2, y8, mask=row_mask[:, None])


@triton.jit
def _ln_mul_dropout_fwd_maskgen_pk_nib(
    X, U, Y, W, B, Mean, Rstd, RANDOM_MASK,
    N, D, eps, dropout_ratio,
    stride_x, stride_u, stride_y, stride_mask,
    Y_INV_SCALE,
    keep_min16, kk2_arg, seed,
    BLOCK_D: tl.constexpr,
    NUM_MASKS: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
    SILU_U: tl.constexpr,
):
    """The NIB pk keep-mask writer: _ln_mul_dropout_fwd_maskgen_pk writing the NIB keep mask (RANDOM_MASK = uint8
    [N, BLOCK_D // 2], dword-aligned rows, stride_mask in bytes) with the same launch arguments, grid cdiv(N, 8),
    num_warps=1, gate contract (fp32 Y_INV_SCALE in (0, inf); y / mask rows dword-aligned; D == BLOCK_D == 512) and
    the same y / mean / rstd / NIB bytes as the previous body, kept verbatim as _ln_mul_dropout_fwd_maskgen_pk_nib_r1
    (the knob-off kernel). Philox straight into NIB bytes (_tp_philox_v3 / _tp_nib_v3), NIB byte masks
    (_mg_bytemask_n_v3), clamped rows instead of per-row masks, buffer stores from scalar row pointers: see the block
    right after the NIB kernels. This body fills the previous body's lines exactly, so no later kernel moves. Same
    counter limit as the previous body (N < 11,184,810)."""
    g = tl.program_id(0)
    H: tl.constexpr = BLOCK_D // 2
    Q4: tl.constexpr = BLOCK_D // 4
    Q8: tl.constexpr = BLOCK_D // 8
    _mg_fp8_sat_mode(g)  # e4m3 bytes come from _mg_cvt4 only (see _mg_fp8_sat_mode)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < BLOCK_D
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask).to(tl.float32)
    inv_D = 1.0 / D
    dropout_scale = 1.0 / (1.0 - dropout_ratio)
    row0 = g * 8
    last = N - 1 - row0  # >= 0: row0 < N for every program of grid cdiv(N, 8)
    emax = last - (last & 1)  # the group's last even row index
    omax = tl.maximum(last - 1 + (last & 1), 0)  # its last odd row index (0 for a one-row group, see below)
    # loop invariants: column-pair w / b, broadcast scalars, constants
    we, wo = _mg_sp(w[None, :], H)
    be, bo = _mg_sp(b[None, :], H)
    Wp = _mg_p2(we, wo)
    Bp = _mg_p2(be, bo)
    DSp = _mg_bc(dropout_scale)
    INVp = _mg_bc(Y_INV_SCALE)
    L2Ep = tl.full([1, H], 0x3FB8AA3B, tl.uint64)  # fp32 1.4426950408889634 (_sigmoid_ieee_fast's constant)
    BMSEL8 = tl.full([1, Q8], 0x090B080A, tl.int32)
    cols4 = tl.arange(0, Q4)[None, :]  # int32 dword index of columns 4a..4a+3 (2 dwords per lane)
    cols8 = tl.arange(0, Q8)[None, :]  # int32 dword index of the NIB dword of lane l (1 dword per lane)

    # row 0's x/u (row0 < N), issued before the Philox phase
    r64_0 = (row0 + tl.arange(0, 1)).to(tl.int64)
    xa = tl.load(X + r64_0[:, None] * stride_x + cols[None, :])
    ua = tl.load(U + r64_0[:, None] * stride_u + cols[None, :])

    # ---- phase 1: the generator's Philox for this 8-row group, straight into NIB bytes ----
    qe0, qe1, qe2, qe3, qo0, qo1, qo2, qo3 = _tp_philox_v3(xa, ua, g, seed, kk2_arg, BLOCK_D, NUM_MASKS)
    if last == 0:
        # a one-row group (the last group when N % 8 == 1): the odd slot recomputes row 0 from row 0's bytes
        qo0 = qe0
        qo1 = qe1
        qo2 = qe2
        qo3 = qe3

    # ---- phase 2: rows 0..7 of the group, two rows per trip, no per-row masks ----
    for i in tl.range(0, 8, 2, num_stages=1):
        ia = tl.minimum(i, emax)
        ib = tl.minimum(i + 1, omax)
        r64_b = (row0 + ib + tl.arange(0, 1)).to(tl.int64)
        xb = tl.load(X + r64_b[:, None] * stride_x + cols[None, :])
        ub = tl.load(U + r64_b[:, None] * stride_u + cols[None, :])
        _row_c32_pk_nib_v3(
            xa, ua, _tp_nib_v3(qe0, qe1, qe2, qe3, ia // 2), row0 + ia, Y, Mean, Rstd, RANDOM_MASK, Wp, Bp, cols4,
            cols8, inv_D, eps, DSp, INVp, L2Ep, BMSEL8, stride_y, stride_mask,
            BLOCK_D, MUL_U_ACTIVATION_TYPE, OUTPUT_FP8, SILU_U,
        )
        rows_n = row0 + tl.minimum(i + 2, emax) + tl.arange(0, 1)
        m2_n = ((rows_n >= 0) & (i < 6))[:, None] & col_mask[None, :]
        r64_n = rows_n.to(tl.int64)
        xa = tl.load(X + r64_n[:, None] * stride_x + cols[None, :], mask=m2_n, other=0.0)
        ua = tl.load(U + r64_n[:, None] * stride_u + cols[None, :], mask=m2_n, other=0.0)
        _row_c32_pk_nib_v3(
            xb, ub, _tp_nib_v3(qo0, qo1, qo2, qo3, ib // 2), row0 + ib, Y, Mean, Rstd, RANDOM_MASK, Wp, Bp, cols4,
            cols8, inv_D, eps, DSp, INVp, L2Ep, BMSEL8, stride_y, stride_mask,
            BLOCK_D, MUL_U_ACTIVATION_TYPE, OUTPUT_FP8, SILU_U,
        )


@triton.jit
def _c32_nib(c32, BLOCK_D: tl.constexpr):
    """[1, BLOCK_D] per-column mask bytes c32 (0..7; lane l holds columns 8l..8l+7) -> [1, BLOCK_D // 8] NIB
    dwords: col 8l+k at bits 8k..8k+2, col 8l+4+k at bits 8k+4..8k+6. In-lane register splits plus 7
    v_lshl_or_b32 (inline shifts 4..28) per row-lane."""
    H: tl.constexpr = BLOCK_D // 2
    Q4: tl.constexpr = BLOCK_D // 4
    Q8: tl.constexpr = BLOCK_D // 8
    a, b = _mg_sp(c32, H)  # columns 2k, 2k+1
    a0, a1 = _mg_sp(a, Q4)  # columns 4k, 4k+2
    b0, b1 = _mg_sp(b, Q4)  # columns 4k+1, 4k+3
    c0, c4 = _mg_sp(a0, Q8)  # columns 8k, 8k+4
    c2, c6 = _mg_sp(a1, Q8)  # columns 8k+2, 8k+6
    c1, c5 = _mg_sp(b0, Q8)  # columns 8k+1, 8k+5
    c3, c7 = _mg_sp(b1, Q8)  # columns 8k+3, 8k+7
    v = tl.inline_asm_elementwise("v_lshl_or_b32 $0, $1, 4, $2", "=v,v,v", [c4, c0],
                                  dtype=tl.int32, is_pure=True, pack=1)
    v = tl.inline_asm_elementwise("v_lshl_or_b32 $0, $1, 8, $2", "=v,v,v", [c1, v],
                                  dtype=tl.int32, is_pure=True, pack=1)
    v = tl.inline_asm_elementwise("v_lshl_or_b32 $0, $1, 12, $2", "=v,v,v", [c5, v],
                                  dtype=tl.int32, is_pure=True, pack=1)
    v = tl.inline_asm_elementwise("v_lshl_or_b32 $0, $1, 16, $2", "=v,v,v", [c2, v],
                                  dtype=tl.int32, is_pure=True, pack=1)
    v = tl.inline_asm_elementwise("v_lshl_or_b32 $0, $1, 20, $2", "=v,v,v", [c6, v],
                                  dtype=tl.int32, is_pure=True, pack=1)
    v = tl.inline_asm_elementwise("v_lshl_or_b32 $0, $1, 24, $2", "=v,v,v", [c3, v],
                                  dtype=tl.int32, is_pure=True, pack=1)
    return tl.inline_asm_elementwise("v_lshl_or_b32 $0, $1, 28, $2", "=v,v,v", [c7, v],
                                     dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _row_c32_nib(
    x_n, u_n, c32, row, N, Y, Mean, Rstd, RANDOM_MASK, w, b, cols, col_mask, cols8,
    inv_D, eps, dropout_scale, Y_INV_SCALE, D, stride_y, stride_mask,
    BLOCK_D: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
    SILU_U: tl.constexpr,
):
    """_row_c32 with the keep mask stored as NIB (one int32 per row-lane at RANDOM_MASK + row * stride_mask +
    4 * lane, stride_mask = BLOCK_D // 2 bytes) instead of the int8 byte per column; every other expression is
    _row_c32's (the dropout still reads c32 in registers)."""
    rows = row + tl.arange(0, 1)
    row_mask = rows < N
    mask_2d = row_mask[:, None] & col_mask[None, :]
    r64 = rows.to(tl.int64)
    tl.store(RANDOM_MASK.to(tl.pointer_type(tl.int32)) + r64[:, None] * (stride_mask // 4) + cols8,
             _c32_nib(c32, BLOCK_D), mask=row_mask[:, None])
    x_block = x_n.to(tl.float32)
    u_block = u_n.to(tl.float32)
    mean = tl.sum(x_block, axis=1) * inv_D
    tl.store(Mean + rows, mean, mask=row_mask)
    mean = tl.expand_dims(mean, 1)
    x_mean = x_block - mean
    _var = x_mean * x_mean
    var = tl.sum(_var, axis=1) * inv_D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + rows, rstd, mask=row_mask)
    rstd = tl.expand_dims(rstd, 1)
    y = x_mean * rstd
    y = y * w[None, :] + b[None, :]
    sigmoid_u_block, d_sig = _sigmoid_ieee_fast(u_block)
    # d bits as uint32 > 0x7E800000 <=> d > 2^126, +inf or NaN (any sign)
    if tl.max(d_sig.to(tl.uint32, bitcast=True)) > 0x7E800000:
        sigmoid_u_block = tl.sigmoid(u_block)
    silu_u_block = u_block * sigmoid_u_block
    if MUL_U_ACTIVATION_TYPE == "silu":
        y = y * silu_u_block
    elif MUL_U_ACTIVATION_TYPE == "sigmoid":
        y = y * sigmoid_u_block
    else:
        y = y * u_block
    if SILU_U:
        u_block = silu_u_block
    u_block = _keep_bit_and(u_block * dropout_scale, c32, 2)
    x_block = _keep_bit_and(x_block * dropout_scale, c32, 1)
    y = _keep_bit_and(y * dropout_scale, c32, 0)
    if OUTPUT_FP8:
        u_block = u_block * Y_INV_SCALE
        x_block = x_block * Y_INV_SCALE
        y = y * Y_INV_SCALE
    yp = Y + r64[:, None] * stride_y + cols[None, :]
    tl.store(yp, u_block.to(Y.dtype.element_ty), mask=mask_2d)
    tl.store(yp + D, x_block.to(Y.dtype.element_ty), mask=mask_2d)
    tl.store(yp + 2 * D, y.to(Y.dtype.element_ty), mask=mask_2d)


@triton.jit
def _ln_mul_dropout_fwd_maskgen_nib(
    X, U, Y, W, B, Mean, Rstd, RANDOM_MASK,
    N, D, eps, dropout_ratio,
    stride_x, stride_u, stride_y, stride_mask,
    Y_INV_SCALE,
    keep_min16, kk2_arg, seed,
    BLOCK_D: tl.constexpr,
    NUM_MASKS: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
    SILU_U: tl.constexpr,
):
    """_ln_mul_dropout_fwd_maskgen (item (28)(e), the layer-0 writer in the iteration-40 tree) writing the NIB keep
    mask (RANDOM_MASK = uint8 [N, BLOCK_D // 2], dword-aligned rows): the same launch arguments, grid cdiv(N, 8),
    num_warps=1, Philox phase, keep decisions and row arithmetic (y / mean / rstd bytes unchanged); only the
    per-row mask store changes (_row_c32_nib)."""
    g = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < BLOCK_D
    cols8 = tl.arange(0, BLOCK_D // 8)[None, :]  # int32 dword index of the NIB dword of lane l
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask).to(tl.float32)
    inv_D = 1.0 / D
    dropout_scale = 1.0 / (1.0 - dropout_ratio)
    row0 = g * 8

    # row 0's x/u, issued before the Philox phase
    xa, ua = _load_row(X, U, row0, N, stride_x, stride_u, cols, col_mask)

    # ---- phase 1: the generator's Philox for this 8-row group, packed per column ----
    kk2 = tl.full([1, BLOCK_D], 0, tl.int32) + kk2_arg
    one2 = tl.full([1, BLOCK_D], 0x10001, tl.int32)
    pk = tl.zeros([1, BLOCK_D], dtype=tl.int32)
    rand_offset = g * (NUM_MASKS * BLOCK_D) + cols[None, :]
    # layout anchor (see _ln_mul_dropout_fwd_maskgen): the counters "depend" on the row-0 x load
    rand_offset = tl.inline_asm_elementwise(
        "v_mov_b32 $0, $1", "=v,v,v", [rand_offset, xa], dtype=tl.int32, is_pure=True, pack=1
    )
    for j in tl.static_range(NUM_MASKS):
        r0, r1, r2, r3 = _philox4x32_10_b3(rand_offset, seed)
        pk = _pk_word(pk, r0, kk2, one2, 0 + j)
        pk = _pk_word(pk, r1, kk2, one2, 3 + j)
        pk = _pk_word(pk, r2, kk2, one2, 6 + j)
        pk = _pk_word(pk, r3, kk2, one2, 9 + j)
        rand_offset += BLOCK_D

    # ---- phase 2: rows 0..7 of the group, next row's x/u carried in registers ----
    for i in tl.range(0, 8, num_stages=1):
        xc = xa
        uc = ua
        nrow = row0 + i + 1
        rows_n = nrow + tl.arange(0, 1)
        m2_n = ((rows_n < N) & (i < 7))[:, None] & col_mask[None, :]
        r64_n = rows_n.to(tl.int64)
        xa = tl.load(X + r64_n[:, None] * stride_x + cols[None, :], mask=m2_n, other=0.0)
        ua = tl.load(U + r64_n[:, None] * stride_u + cols[None, :], mask=m2_n, other=0.0)
        # row i (0..3): bits 3i..3i+2; row i (4..7): bits 16+3(i-4)..
        base = 3 * (i % 4) + 16 * (i // 4)
        _row_c32_nib(
            xc, uc, (pk >> base) & 7, row0 + i, N, Y, Mean, Rstd, RANDOM_MASK, w, b, cols, col_mask, cols8,
            inv_D, eps, dropout_scale, Y_INV_SCALE, D, stride_y, stride_mask,
            BLOCK_D, MUL_U_ACTIVATION_TYPE, OUTPUT_FP8, SILU_U,
        )


# =============================================================================
# (111) _pk_nib V3 (r16 iteration 59 pknib-design "V3"; lab kitV, seal
# b1d40d93, two runs on two nodes: G1 bitwise 360/360 in both). Ship17 gin
# item (111), DLRM_HSTU_FP8_LNMD_MASK_NIB_PK_V3: code default OFF, exactly "1"
# arms it, yambda_5b.gin binds it ON, a pre-set env value wins ("0" = kill
# switch). BITWISE: the same Philox counters, keep decisions and y / mean /
# rstd / NIB bytes as the iteration-54 _pk_nib body. This block sits right
# after the last NIB kernel and before the NIB host side: no existing kernel
# moves, and the helpers' starting lines (Triton hashes a callee's starting
# line into its caller's cache key) do not depend on the host code below
# (r14's erm_integ_cpu_test keeps the E-RM helpers below every @triton.jit
# function; they still are). Above this banner exactly two places change,
# both IN PLACE (no physical line inserted or deleted there):
#   * _ln_mul_dropout_fwd_maskgen_pk_nib: the V3 body fills the old body's
#     lines (same decorator, def and parameter lines, same last line), so the
#     launched kernel keeps its name, signature, constexprs and launch
#     arguments (the checked-in TSPEC rows of that qualname replay the V3
#     source before RUN_START), and every other @triton.jit function keeps
#     its starting line and source, hence its Triton cache_key, disk-cache key
#     and TSPEC fn_cache_key;
#   * _lnmd_fwd_maskgen_nib_launch (below the block): the line that picks the
#     kernel asks _lnmd_mask_nib_pk_kernel (in this block) for the pk writer.
# Below: the eight V3 helpers (kitV's lab alias functions byte for byte, with
# the alias's "_v3" names), the knob-off kernel
# _ln_mul_dropout_fwd_maskgen_pk_nib_r1 (the iteration-54 / 544e691 _pk_nib
# verbatim under a new name: the same .text machine code; no TSPEC rows) and
# the selector.
#   TP: the Philox phase accumulates the keep bits straight into NIB bytes
#     (_tp_philox_v3: qe_k byte t = row 2t's NIB byte k, qo_k byte t = row
#     2t+1's), so a row's NIB dword is 2 v_perm_b32 + v_or_b32 (_tp_nib_v3);
#   NBM: the u / x / y byte masks come from the NIB dword (_mg_bytemask_n_v3);
#   CL: no per-row masks: row A of a trip is clamped to the group's last even
#     row < N, row B to its last odd row; in the last, partial group an
#     out-of-range trip recomputes an in-range row of its own parity and
#     stores the same bytes to it again (a one-row group copies qe into qo);
#     no row >= N is read or written;
#   ADDR: scalar row pointers and BLOCK_D plane offsets (D == BLOCK_D), so
#     every store is a buffer store; ANC2: the layout anchor names both row-0
#     loads, so no load is outstanding when the row loop starts.
# =============================================================================


@triton.jit
def _mg_bytemask_n_v3(n, sel, S: tl.constexpr):
    """[1, Q8] NIB dwords n -> byte k = 0xFF if bit (7 - S) of byte k of n is set, else 0x00 (S = 7 - J: plane J
    of columns 8l..8l+3, bits 0-2; S = 3 - J: plane J of columns 8l+4..8l+7, bits 4-6). V = (n, n << 8) as a
    64-bit pair; V << S puts bit 7-S of bytes 1/3 at bit 7 of the low dword's bytes 1/3 and bit 7-S of bytes 0/2
    at bit 7 of the high dword's bytes 1/3 (the <= 3 spilled bits of n land in high-dword bits 0-2, never read);
    v_perm_b32 codes 8-11 (sel = 0x090B080A) broadcast them. _mg_bytemask's instructions on ONE pair per row."""
    v = n.to(tl.uint32, bitcast=True).to(tl.uint64) | ((n << 8).to(tl.uint32, bitcast=True).to(tl.uint64) << 32)
    if S == 7:
        v = tl.inline_asm_elementwise("v_lshlrev_b64 $0, 7, $1", "=v,v", [v], dtype=tl.uint64, is_pure=True, pack=1)
    elif S == 6:
        v = tl.inline_asm_elementwise("v_lshlrev_b64 $0, 6, $1", "=v,v", [v], dtype=tl.uint64, is_pure=True, pack=1)
    elif S == 5:
        v = tl.inline_asm_elementwise("v_lshlrev_b64 $0, 5, $1", "=v,v", [v], dtype=tl.uint64, is_pure=True, pack=1)
    elif S == 3:
        v = tl.inline_asm_elementwise("v_lshlrev_b64 $0, 3, $1", "=v,v", [v], dtype=tl.uint64, is_pure=True, pack=1)
    elif S == 2:
        v = tl.inline_asm_elementwise("v_lshlrev_b64 $0, 2, $1", "=v,v", [v], dtype=tl.uint64, is_pure=True, pack=1)
    else:
        tl.static_assert(S == 1)
        v = tl.inline_asm_elementwise("v_lshlrev_b64 $0, 1, $1", "=v,v", [v], dtype=tl.uint64, is_pure=True, pack=1)
    s1 = v.to(tl.uint32).to(tl.int32, bitcast=True)
    s0 = (v >> 32).to(tl.uint32).to(tl.int32, bitcast=True)
    return tl.inline_asm_elementwise("v_perm_b32 $0, $1, $2, $3", "=v,v,v,v", [s0, s1, sel],
                                     dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _mg_j2_v3(a, b, Q4: tl.constexpr):
    """two [1, Q8] tensors (lane l: a = dword of columns 8l..8l+3, b = 8l+4..8l+7) -> [1, Q4] (element 2l = a,
    2l+1 = b): the inverse of _mg_sp(., Q8), a register pairing inside each lane."""
    return tl.reshape(tl.join(a, b), [1, Q4])


@triton.jit
def _mg_split8_v3(t, BLOCK_D: tl.constexpr):
    """[1, BLOCK_D] (lane l: columns 8l..8l+7) -> 8 x [1, BLOCK_D // 8], c_k = column 8l + k (in-lane splits)."""
    H: tl.constexpr = BLOCK_D // 2
    Q4: tl.constexpr = BLOCK_D // 4
    Q8: tl.constexpr = BLOCK_D // 8
    a, b = _mg_sp(t, H)
    a0, a1 = _mg_sp(a, Q4)
    b0, b1 = _mg_sp(b, Q4)
    c0, c4 = _mg_sp(a0, Q8)
    c2, c6 = _mg_sp(a1, Q8)
    c1, c5 = _mg_sp(b0, Q8)
    c3, c7 = _mg_sp(b1, Q8)
    return c0, c1, c2, c3, c4, c5, c6, c7


@triton.jit
def _mg_lso_v3(acc, v, sh):
    """acc | (v << sh) in one v_lshl_or_b32 (sh a wave-uniform constant, SGPR)."""
    return tl.inline_asm_elementwise("v_lshl_or_b32 $0, $1, $2, $3", "=v,v,s,v", [v, sh, acc],
                                     dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _tp_word_v3(a0, a1, a2, a3, r, kk2, one2, SH: tl.constexpr, BLOCK_D: tl.constexpr):
    """Philox word r of plane j (low half: row w, high half: row w + 4) into the NIB-byte accumulators of one row
    parity: _pair_keep_bits gives bit 0 (row w) and bit 16 (row w + 4) per column; column 8l+k (k < 4) is OR-ed in
    at SH and column 8l+4+k at SH + 4 of a_k (SH = 8 * (w // 2) + j), so a_k's byte b holds row (w % 2) + 2b's
    NIB byte k. Same keep decisions as _mg_pk_word, the same 8 v_lshl_or_b32 per word."""
    pb = _pair_keep_bits(r.to(tl.int32, bitcast=True), kk2, one2)
    c0, c1, c2, c3, c4, c5, c6, c7 = _mg_split8_v3(pb, BLOCK_D)
    a0 = _mg_lso_v3(_mg_lso_v3(a0, c0, SH), c4, SH + 4)
    a1 = _mg_lso_v3(_mg_lso_v3(a1, c1, SH), c5, SH + 4)
    a2 = _mg_lso_v3(_mg_lso_v3(a2, c2, SH), c6, SH + 4)
    a3 = _mg_lso_v3(_mg_lso_v3(a3, c3, SH), c7, SH + 4)
    return a0, a1, a2, a3


@triton.jit
def _tp_nib_v3(q0, q1, q2, q3, t):
    """the NIB dword of the row held in byte t of q0..q3: two v_perm_b32 (SGPR selectors) + v_or_b32."""
    return _mg_perm(q1, q0, 0x0C0C0400 + 0x0101 * t) | _mg_perm(q3, q2, 0x04000C0C + 0x01010000 * t)


@triton.jit
def _tp_philox_v3(xa, ua, g, seed, kk2_arg, BLOCK_D: tl.constexpr, NUM_MASKS: tl.constexpr):
    """The generator's Philox for the 8-row group (same counters, seed and _pair_keep_bits as
    _ln_mul_dropout_fwd_maskgen_pk), accumulated straight into NIB bytes: qe_k byte t = row 2t's NIB byte k,
    qo_k byte t = row 2t+1's. The layout anchor names both row-0 loads, so no load is outstanding when the row
    loop starts and its s_waitcnt counts come from the loop's own back edge."""
    Q8: tl.constexpr = BLOCK_D // 8
    cols = tl.arange(0, BLOCK_D)
    kk2 = tl.full([1, BLOCK_D], 0, tl.int32) + kk2_arg
    one2 = tl.full([1, BLOCK_D], 0x10001, tl.int32)
    qe0 = tl.zeros([1, Q8], dtype=tl.int32)
    qe1 = tl.zeros([1, Q8], dtype=tl.int32)
    qe2 = tl.zeros([1, Q8], dtype=tl.int32)
    qe3 = tl.zeros([1, Q8], dtype=tl.int32)
    qo0 = tl.zeros([1, Q8], dtype=tl.int32)
    qo1 = tl.zeros([1, Q8], dtype=tl.int32)
    qo2 = tl.zeros([1, Q8], dtype=tl.int32)
    qo3 = tl.zeros([1, Q8], dtype=tl.int32)
    rand_offset = g * (NUM_MASKS * BLOCK_D) + cols[None, :]
    rand_offset = tl.inline_asm_elementwise(
        "v_mov_b32 $0, $1", "=v,v,v,v", [rand_offset, xa, ua], dtype=tl.int32, is_pure=True, pack=1
    )
    for j in tl.static_range(NUM_MASKS):
        r0, r1, r2, r3 = _philox4x32_10_b3(rand_offset, seed)
        qe0, qe1, qe2, qe3 = _tp_word_v3(qe0, qe1, qe2, qe3, r0, kk2, one2, 0 + j, BLOCK_D)  # rows 0 / 4
        qo0, qo1, qo2, qo3 = _tp_word_v3(qo0, qo1, qo2, qo3, r1, kk2, one2, 0 + j, BLOCK_D)  # rows 1 / 5
        qe0, qe1, qe2, qe3 = _tp_word_v3(qe0, qe1, qe2, qe3, r2, kk2, one2, 8 + j, BLOCK_D)  # rows 2 / 6
        qo0, qo1, qo2, qo3 = _tp_word_v3(qo0, qo1, qo2, qo3, r3, kk2, one2, 8 + j, BLOCK_D)  # rows 3 / 7
        rand_offset += BLOCK_D
    return qe0, qe1, qe2, qe3, qo0, qo1, qo2, qo3


@triton.jit
def _row_c32_pk_nib_v3(
    x_n, u_n, nibw, row, Y, Mean, Rstd, RANDOM_MASK, Wp, Bp, cols4, cols8,
    inv_D, eps, DSp, INVp, L2Ep, BMSEL8, stride_y, stride_mask,
    BLOCK_D: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
    SILU_U: tl.constexpr,
):
    """One row (row < N, always: the caller clamps) of _ln_mul_dropout_fwd_maskgen_pk_nib: the NIB dword nibw at
    RANDOM_MASK + row * stride_mask + 4 * lane, then _row_c32_pk's mean / rstd / y arithmetic unchanged, with the u /
    x / y byte masks built from nibw. Stores have no mask; scalar row pointers and BLOCK_D plane offsets (D ==
    BLOCK_D) make every store a buffer store with a loop-invariant voffset."""
    H: tl.constexpr = BLOCK_D // 2
    Q4: tl.constexpr = BLOCK_D // 4
    tl.static_assert(MUL_U_ACTIVATION_TYPE == "silu")
    tl.static_assert(OUTPUT_FP8)
    mrow = RANDOM_MASK.to(tl.pointer_type(tl.int32)) + row.to(tl.int64) * (stride_mask // 4)
    tl.store(mrow + cols8, nibw)
    x_block = x_n.to(tl.float32)
    u_block = u_n.to(tl.float32)
    mean = tl.sum(x_block, axis=1) * inv_D
    tl.store((Mean + row) + tl.arange(0, 1), mean)
    mean = tl.expand_dims(mean, 1)
    x_mean = x_block - mean
    _var = x_mean * x_mean
    var = tl.sum(_var, axis=1) * inv_D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store((Rstd + row) + tl.arange(0, 1), rstd)
    rstd = tl.expand_dims(rstd, 1)
    # _sigmoid_ieee_fast on column pairs (same ops per element), wave-uniform tl.sigmoid fallback
    ue, uo = _mg_sp(u_block, H)
    U = _mg_p2(ue, uo)
    NE = _mg_pk_nmul_b(U, L2Ep)
    E = _mg_p2(_amd_v_exp_f32(_mg_lo(NE)), _amd_v_exp_f32(_mg_hi(NE)))
    Dd = _mg_pk_add1(E)
    R = _mg_p2(_amd_v_rcp_f32(_mg_lo(Dd)), _amd_v_rcp_f32(_mg_hi(Dd)))
    T = _mg_pk_nfma1(Dd, R)
    R = _mg_pk_fma(T, R, R)
    T = _mg_pk_nfma1(Dd, R)
    Q = _mg_pk_fma(T, R, R)
    T = _mg_pk_nfma1(Dd, Q)
    SIG = _mg_pk_fma(T, R, Q)
    # d bits as uint32 > 0x7E800000 <=> d > 2^126, +inf or NaN (any sign)
    dmax = tl.max(tl.maximum(_mg_lo(Dd).to(tl.uint32, bitcast=True), _mg_hi(Dd).to(tl.uint32, bitcast=True)))
    if dmax > 0x7E800000:
        sge, sgo = _mg_sp(tl.sigmoid(u_block), H)
        SIG = _mg_p2(sge, sgo)
    xme, xmo = _mg_sp(x_mean, H)
    Yp = _mg_pk_mul_b(_mg_p2(xme, xmo), _mg_bc(rstd))  # y = x_mean * rstd
    Yp = _mg_pk_fma(Yp, Wp, Bp)  # y = y * w + b
    SIL = _mg_pk_mul(U, SIG)  # silu = u * sigmoid(u)
    Yp = _mg_pk_mul(Yp, SIL)  # y = y * silu
    if SILU_U:
        UP = SIL
    else:
        UP = U
    xe, xo = _mg_sp(x_block, H)
    UF = _mg_pk_mul_b(_mg_pk_mul_b(UP, DSp), INVp)  # (u * dropout_scale) * Y_INV_SCALE
    XF = _mg_pk_mul_b(_mg_pk_mul_b(_mg_p2(xe, xo), DSp), INVp)
    YF = _mg_pk_mul_b(_mg_pk_mul_b(Yp, DSp), INVp)
    ua, ub = _mg_sp(UF, Q4)
    xa, xb = _mg_sp(XF, Q4)
    ya, yb = _mg_sp(YF, Q4)
    bu = _mg_j2_v3(_mg_bytemask_n_v3(nibw, BMSEL8, 5), _mg_bytemask_n_v3(nibw, BMSEL8, 1), Q4)
    bx = _mg_j2_v3(_mg_bytemask_n_v3(nibw, BMSEL8, 6), _mg_bytemask_n_v3(nibw, BMSEL8, 2), Q4)
    by = _mg_j2_v3(_mg_bytemask_n_v3(nibw, BMSEL8, 7), _mg_bytemask_n_v3(nibw, BMSEL8, 3), Q4)
    u8 = _mg_cvt4(_mg_lo(ua), _mg_hi(ua), _mg_lo(ub), _mg_hi(ub)) & bu
    x8 = _mg_cvt4(_mg_lo(xa), _mg_hi(xa), _mg_lo(xb), _mg_hi(xb)) & bx
    y8 = _mg_cvt4(_mg_lo(ya), _mg_hi(ya), _mg_lo(yb), _mg_hi(yb)) & by
    yrow = Y.to(tl.pointer_type(tl.int32)) + row.to(tl.int64) * (stride_y // 4)
    tl.store(yrow + cols4, u8)
    tl.store(yrow + (cols4 + Q4), x8)  # D == BLOCK_D: the x plane starts BLOCK_D bytes into the row
    tl.store(yrow + (cols4 + 2 * Q4), y8)


@triton.jit
def _ln_mul_dropout_fwd_maskgen_pk_nib_r1(
    X, U, Y, W, B, Mean, Rstd, RANDOM_MASK,
    N, D, eps, dropout_ratio,
    stride_x, stride_u, stride_y, stride_mask,
    Y_INV_SCALE,
    keep_min16, kk2_arg, seed,
    BLOCK_D: tl.constexpr,
    NUM_MASKS: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    OUTPUT_FP8: tl.constexpr,
    SILU_U: tl.constexpr,
):
    """_ln_mul_dropout_fwd_maskgen_pk writing the NIB keep mask (RANDOM_MASK = uint8 [N, BLOCK_D // 2], dword-aligned
    rows, stride_mask in bytes): the same launch arguments, grid cdiv(N, 8), num_warps=1, gate contract (fp32
    Y_INV_SCALE in (0, inf); y / mask rows dword-aligned) and the same y / mean / rstd bytes; the mask store is one
    v_lshl_or_b32 + one dword store per row-lane instead of a dwordx2 store."""
    g = tl.program_id(0)
    H: tl.constexpr = BLOCK_D // 2
    Q4: tl.constexpr = BLOCK_D // 4
    Q8: tl.constexpr = BLOCK_D // 8
    _mg_fp8_sat_mode(g)  # e4m3 bytes come from _mg_cvt4 only (see _mg_fp8_sat_mode)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < BLOCK_D
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask).to(tl.float32)
    inv_D = 1.0 / D
    dropout_scale = 1.0 / (1.0 - dropout_ratio)
    row0 = g * 8
    # loop invariants: column-pair w / b, broadcast scalars, constants
    we, wo = _mg_sp(w[None, :], H)
    be, bo = _mg_sp(b[None, :], H)
    Wp = _mg_p2(we, wo)
    Bp = _mg_p2(be, bo)
    DSp = _mg_bc(dropout_scale)
    INVp = _mg_bc(Y_INV_SCALE)
    L2Ep = tl.full([1, H], 0x3FB8AA3B, tl.uint64)  # fp32 1.4426950408889634 (_sigmoid_ieee_fast's constant)
    BMSEL = tl.full([1, Q4], 0x090B080A, tl.int32)
    cols4 = tl.arange(0, Q4)[None, :]  # int32 dword index of columns 4a..4a+3 (2 dwords per lane)
    cols8 = tl.arange(0, Q8)[None, :]  # int32 dword index of the NIB dword of lane l (1 dword per lane)

    # row 0's x/u, issued before the Philox phase
    xa, ua = _load_row(X, U, row0, N, stride_x, stride_u, cols, col_mask)

    # ---- phase 1: the generator's Philox for this 8-row group, packed per column ----
    kk2 = tl.full([1, BLOCK_D], 0, tl.int32) + kk2_arg
    one2 = tl.full([1, BLOCK_D], 0x10001, tl.int32)
    pk = tl.zeros([1, BLOCK_D], dtype=tl.int32)
    rand_offset = g * (NUM_MASKS * BLOCK_D) + cols[None, :]
    # layout anchor (see _ln_mul_dropout_fwd_maskgen): the counters "depend" on the row-0 x load
    rand_offset = tl.inline_asm_elementwise(
        "v_mov_b32 $0, $1", "=v,v,v", [rand_offset, xa], dtype=tl.int32, is_pure=True, pack=1
    )
    for j in tl.static_range(NUM_MASKS):
        r0, r1, r2, r3 = _philox4x32_10_b3(rand_offset, seed)
        pk = _mg_pk_word(pk, r0, kk2, one2, 0 + j)
        pk = _mg_pk_word(pk, r1, kk2, one2, 3 + j)
        pk = _mg_pk_word(pk, r2, kk2, one2, 6 + j)
        pk = _mg_pk_word(pk, r3, kk2, one2, 9 + j)
        rand_offset += BLOCK_D

    # ---- phase 2: rows 0..7 of the group, two rows per trip ----
    for i in tl.range(0, 8, 2, num_stages=1):
        rows_b = row0 + i + 1 + tl.arange(0, 1)
        m2_b = (rows_b < N)[:, None] & col_mask[None, :]
        r64_b = rows_b.to(tl.int64)
        xb = tl.load(X + r64_b[:, None] * stride_x + cols[None, :], mask=m2_b, other=0.0)
        ub = tl.load(U + r64_b[:, None] * stride_u + cols[None, :], mask=m2_b, other=0.0)
        # row i (0..3): bits 3i..3i+2; row i (4..7): bits 16+3(i-4)..
        _row_c32_pk_nib(
            xa, ua, pk, 3 * (i % 4) + 16 * (i // 4), row0 + i, N, Y, Mean, Rstd, RANDOM_MASK, Wp, Bp, cols4, cols8,
            inv_D, eps, DSp, INVp, L2Ep, BMSEL, D, stride_y, stride_mask,
            BLOCK_D, MUL_U_ACTIVATION_TYPE, OUTPUT_FP8, SILU_U,
        )
        rows_n = row0 + i + 2 + tl.arange(0, 1)
        m2_n = ((rows_n < N) & (i < 6))[:, None] & col_mask[None, :]
        r64_n = rows_n.to(tl.int64)
        xa = tl.load(X + r64_n[:, None] * stride_x + cols[None, :], mask=m2_n, other=0.0)
        ua = tl.load(U + r64_n[:, None] * stride_u + cols[None, :], mask=m2_n, other=0.0)
        _row_c32_pk_nib(
            xb, ub, pk, 3 * ((i + 1) % 4) + 16 * ((i + 1) // 4), row0 + i + 1, N, Y, Mean, Rstd, RANDOM_MASK,
            Wp, Bp, cols4, cols8, inv_D, eps, DSp, INVp, L2Ep, BMSEL, D, stride_y, stride_mask,
            BLOCK_D, MUL_U_ACTIVATION_TYPE, OUTPUT_FP8, SILU_U,
        )


# (111) selector state: two once-per-process line marks, each holding the line it printed ("" = not yet). The FP8
# first-use warmup (dlrm_v4/train/fp8_warmup.py) keeps both out of its scalar snapshot / restore (_KEEP_GLOBALS) and
# prints a line its item L4n latched (inside the item's stdout capture) again right after that item, outside the
# capture (_pk_v3_reprint): the line prints once per process, before RUN_START, and no training launch prints it.
_LNMD_MASK_NIB_PK_V3_KNOB = "DLRM_HSTU_FP8_LNMD_MASK_NIB_PK_V3"
_LNMD_MASK_NIB_PK_V3_ITEM = "(111)"
_LNMD_MASK_NIB_PK_V3_ARMED_LOGGED = ""
_LNMD_MASK_NIB_PK_V3_OFF_LOGGED = ""


def _lnmd_mask_nib_pk_kernel(N: int, D: int, BLOCK_D: int) -> Any:
    """(111): the kernel _lnmd_fwd_maskgen_nib_launch launches for pk=True. _ln_mul_dropout_fwd_maskgen_pk_nib
    (the V3 body) when DLRM_HSTU_FP8_LNMD_MASK_NIB_PK_V3 is exactly "1" (yambda_5b.gin binds it; surrounding
    whitespace ignored) and D == BLOCK_D (V3 addresses the x / y planes with BLOCK_D; every production caller has
    D == BLOCK_D == 512); otherwise _ln_mul_dropout_fwd_maskgen_pk_nib_r1: the iteration-54 body verbatim under its
    own name, i.e. the same source body, the same .text machine code and the same launch arguments as the kernel
    before this item, but a different JITFunction (its own trace symbol and disk-cache key, and no TSPEC rows).
    Read per call: one environment read; no state but two once-per-process line marks. Lines (every rank, tagged
    [rank N]): "armed OK: (111) LNMD NIB pk writer V3 ACTIVE ..." at the first V3 pick; "(111) LNMD NIB pk writer
    V3 OFF ..." at the first r1 pick with the knob set to any other value ("0": a gin False binding or the env kill
    switch); "(111) LNMD NIB pk writer V3 NOT armed ..." when the knob is "1" but D != BLOCK_D. Knob absent (the
    code default): r1 and no line. In production the first pick is the FP8 warmup's (item L4n, before RUN_START,
    stdout captured): the warmup prints the latched line after that item and keeps the mark, so each line prints
    once per process, before RUN_START, and never in the timed region. Only with the warmup off (FP8_WARMUP=0) or
    L4n not reaching the pk writer does the line print at the first training launch instead."""
    global _LNMD_MASK_NIB_PK_V3_ARMED_LOGGED, _LNMD_MASK_NIB_PK_V3_OFF_LOGGED
    value = os.environ.get(_LNMD_MASK_NIB_PK_V3_KNOB)
    on = value is not None and value.strip() == "1"
    if on and int(D) == int(BLOCK_D):
        if not _LNMD_MASK_NIB_PK_V3_ARMED_LOGGED:
            _LNMD_MASK_NIB_PK_V3_ARMED_LOGGED = (
                f"armed OK: {_LNMD_MASK_NIB_PK_V3_ITEM} LNMD NIB pk writer V3 ACTIVE "
                f"({_LNMD_MASK_NIB_PK_V3_KNOB}=1): _ln_mul_dropout_fwd_maskgen_pk_nib runs the V3 body (r16 "
                "iteration 59: Philox straight into NIB bytes, NIB byte masks, clamped rows, buffer stores), "
                "bitwise (the same y / mean / rstd / NIB bytes as the r1 body); first V3 launch on layer tag "
                f"{_LNMD_LAYER_TAG}, N={N}"
            )
            # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
            arm_print_all_ranks(_LNMD_MASK_NIB_PK_V3_ARMED_LOGGED)
        return _ln_mul_dropout_fwd_maskgen_pk_nib
    if value is not None and not _LNMD_MASK_NIB_PK_V3_OFF_LOGGED:
        if on:
            why = (
                f"NOT armed ({_LNMD_MASK_NIB_PK_V3_KNOB}=1): D={D} != BLOCK_D={BLOCK_D} (V3 addresses the x / y "
                "planes with BLOCK_D)"
            )
        else:
            why = (
                f"OFF ({_LNMD_MASK_NIB_PK_V3_KNOB}={value.strip() or '<empty>'}: not 1; 0 is a gin False binding "
                "or the env kill switch; only the value 1 arms it)"
            )
        _LNMD_MASK_NIB_PK_V3_OFF_LOGGED = (
            f"{_LNMD_MASK_NIB_PK_V3_ITEM} LNMD NIB pk writer V3 {why}: _ln_mul_dropout_fwd_maskgen_pk_nib_r1 runs "
            "(the 544e691 _pk_nib body verbatim under its own name: the same output bytes and .text machine code); "
            "it has no TSPEC rows, so a specialization the FP8 warmup (item L4n) did not compile compiles at its "
            "first use; one line per process"
        )
        # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
        arm_print_all_ranks(_LNMD_MASK_NIB_PK_V3_OFF_LOGGED)
    return _ln_mul_dropout_fwd_maskgen_pk_nib_r1


# ---------------------------------------------------------------------------
# NIB host side: the torch references, the launch helper, the per-call gate
# (_lnmd_mask_nib_why_not), the hook item (28)(e)'s maskgen branch calls
# (_lnmd_fwd_maskgen_nib_try), its lines, counters and latch, the out-proj
# context HSTUComputeOutputFunction.forward records (_lnmd_mask_nib_note_call)
# and the FP8 first-use warmup's launch (lnmd_mask_nib_warmup_launch).
# ---------------------------------------------------------------------------

# One torch reference (iteration 41's names): the shared contract module's exact pack / unpack.
lnmd_mask_nibpack_ref = lnmd_mask_nib.pack_ref
lnmd_mask_nibunpack_ref = lnmd_mask_nib.unpack


def _lnmd_fwd_maskgen_nib_launch(
    x: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    packed_mask: torch.Tensor,
    N: int,
    D: int,
    eps: float,
    dropout_ratio: float,
    inv_scale: float,
    keep_min16: int,
    kk2: int,
    seed: int,
    BLOCK_D: int,
    concat_u: bool,
    concat_x: bool,
    mul_u_activation_type: str,
    output_fp8: bool,
    silu_u: bool,
    pk: bool,
) -> None:
    """Item (28)(e)'s maskgen launch writing the NIB keep mask (r16 iteration 41's helper, unchanged): the
    production launch arguments of the shipped kernel (pk=False -> _ln_mul_dropout_fwd_maskgen_nib) or of MGPK
    (pk=True -> _ln_mul_dropout_fwd_maskgen_pk_nib; the caller must have passed MGPK's gate: fp32 Y_INV_SCALE in
    (0, inf), dword-aligned y rows). packed_mask: uint8 [N, BLOCK_D // 2], dword-aligned base and row stride.
    Called by _lnmd_fwd_maskgen_nib_try (production) and lnmd_mask_nib_warmup_launch (FP8 first-use warmup)."""
    assert packed_mask.dtype == torch.uint8 and tuple(packed_mask.shape) == (N, BLOCK_D // 2)
    assert packed_mask.stride(1) == 1 and (packed_mask.data_ptr() | packed_mask.stride(0)) & 3 == 0
    kernel = _lnmd_mask_nib_pk_kernel(N, D, BLOCK_D) if pk else _ln_mul_dropout_fwd_maskgen_nib  # (111) V3 or r1
    # pyre-ignore[28]
    kernel[(triton.cdiv(N, 8),)](
        x,
        u,
        y,
        weight,
        bias,
        mean,
        rstd,
        packed_mask,
        N,
        D,
        eps,
        dropout_ratio,
        x.stride(0),
        u.stride(0),
        y.stride(0),
        packed_mask.stride(0),
        float(inv_scale),
        keep_min16,
        kk2,
        seed,
        BLOCK_D=BLOCK_D,
        NUM_MASKS=1 + int(concat_u) + int(concat_x),
        MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
        OUTPUT_FP8=output_fp8,
        SILU_U=silu_u,
        num_warps=1,
        # Triton 3.9 (ROCm 10.2 port, triton-perf replay of the production launch, N=2.4M, D=512):
        # waves_per_eu=4 lets the r1 kernel use 128 VGPRs (default 105, same occupancy 4), no spill:
        # 1985 -> 1891 us (-5%, bitwise). Not for the PK kernel (pk=True): 20 B scratch, no gain.
        # Not on Triton 3.6 (v26.3): there the waves_per_eu=4 build spills (128 VGPRs, 8 B scratch) and the
        # shipped launch keeps its options (and its TSPEC table keys).
        **({} if pk or triton.__version__ == "3.6.0" else {"waves_per_eu": 4}),
    )


def _lnmd_maskgen_kk2(dropout_ratio: float) -> Tuple[int, int]:
    """(keep_min16, kk2) exactly as item (28)(e)'s launch site computes them (kk2 = both u16 halves of
    keep_min16 - 1, as int32)."""
    keep_min16 = _mask_keep_min16(dropout_ratio)
    kk2 = (keep_min16 - 1) * 0x10001
    if kk2 >= 2**31:
        kk2 -= 2**32
    return keep_min16, kk2


_LNMD_MASK_NIB_KNOB = lnmd_mask_nib.KNOB
_LNMD_MASK_NIB_ITEM = "(94)"
_LNMD_MASK_NIB_FAMILY = "LN*U*dropout keep mask NIB"
_LNMD_MASK_NIB_SH_KERNEL = "_ln_mul_dropout_fwd_maskgen_nib"
_LNMD_MASK_NIB_PK_KERNEL = "_ln_mul_dropout_fwd_maskgen_pk_nib"
# The predicate's answer with the knob off (never counted, never printed).
_LNMD_MASK_NIB_OFF = "N0 knob off"
# Decline keys of expected states: counted, never printed (_lnmd_mask_nib_note_decline).
# "N8 b1a-site" (review R1-F7): B1a tags its uvqk site on u only once that
# site's scale is published, so B1a's warm-up calls decline there ("N8
# b1a-warmup" stays as a guard for a site tagged before it publishes).
_LNMD_MASK_NIB_QUIET = "|N1 latched|N2 no-context|N3 out-warmup|N7 no-site|N8 b1a-site|N8 b1a-warmup|"
# Process state. Plain scalars only: the FP8 first-use warmup (dlrm_v4/train/
# fp8_warmup.py, _SCALAR_MODULES) snapshots and restores them, so its direct
# LN*U*dropout calls leave no count, latch or printed-once mark behind.
# Latch: set (one DISABLED line on every rank) when a NIB writer launch raised;
# the int8 keep mask (MGPK / the shipped maskgen) is written from then on.
_LNMD_MASK_NIB_BROKEN: Optional[str] = None
_LNMD_MASK_NIB_ARMED_LOGGED = False
_LNMD_MASK_NIB_SH_LOGGED = False
_LNMD_MASK_NIB_PK_LOGGED = False
# "|key|key|": the decline-reason keys already printed (one line per key and process).
_LNMD_MASK_NIB_DECLINES_LOGGED = ""
_LNMD_MASK_NIB_N_ARMED = 0
_LNMD_MASK_NIB_N_SH = 0
_LNMD_MASK_NIB_N_PK = 0
_LNMD_MASK_NIB_N_DECLINED = 0
_LNMD_MASK_NIB_N_FAILED = 0
_LNMD_MASK_NIB_ATEXIT = False
# The exit counters line ran (once: clean_exit.EXIT_HOOKS runs it before
# os._exit, atexit after a normal exit).
_LNMD_MASK_NIB_FINAL_DONE = False
# The out-proj context of the coming maskgen call, recorded by
# HSTUComputeOutputFunction.forward (FP8-output branch) right before its
# LN*U*dropout forward: id() of its attn (the x the forward impl receives) and
# why that call's backward could not hand the mask to DO8 ("" = nothing
# known). Consumed (reset) by the next _lnmd_mask_nib_why_not. A call whose x
# is not that attn has no context and declines (a direct call, e.g. the FP8
# first-use warmup's L4). Plain ints / str: no tensor is kept alive.
_LNMD_MASK_NIB_CALL_ID = 0
_LNMD_MASK_NIB_CALL_WHY = ""


def _lnmd_mask_nib_note_call(
    attn: torch.Tensor, output_weight: torch.Tensor, need_bwd: bool
) -> None:
    """HSTUComputeOutputFunction.forward, FP8-output branch, right before its LN*U*dropout forward (an in-place
    edit of one line there): record what this call's backward needs for L11's e5m2-du mode and DO8 that the NIB
    predicate cannot see from inside _triton_layer_norm_mul_dropout_fwd_impl (it has no output_weight):
      - a backward exists (need_bwd);
      - the FP8 out-proj wgrad branch (fp8_y_saved, L11 D4): DLRM_HSTU_FP8_BWD_OUT_WGRAD and the scale controller;
      - output_weight (1536, 512) (L11 D8) with a stable identity (the dgrad cast's per-layer site key);
      - that out-proj dgrad scale site already published: until it is (its first DLRM_HSTU_FP8_BWD_WARMUP_CALLS
        calls) output_projection_dgrad_operands returns None and the backward runs bf16 without L11 / DO8.
    Knob absent or not "1" (or under torch.compile): returns at once, nothing read or written."""
    global _LNMD_MASK_NIB_CALL_ID, _LNMD_MASK_NIB_CALL_WHY
    if torch.compiler.is_compiling() or not lnmd_mask_nib.requested():
        return
    why = ""
    try:
        peek = getattr(bwd_fp8, "_peek_site_state", None)
        stable = getattr(bwd_fp8, "_has_stable_identity", None)
        if not need_bwd:
            why = "N3 no-backward: need_bwd False (nothing reads this mask)"
        elif not (_FP8_OUT_WGRAD and bwd_fp8.scale_controller_enabled()):
            why = "N3 out-wgrad: the FP8 out-proj wgrad branch is off (L11 D4: fp8_y_saved False)"
        elif tuple(output_weight.shape) != (l11_outproj_lnmd.NOUT, l11_outproj_lnmd.D):
            why = f"N3 out-weight: output_weight {tuple(output_weight.shape)} (L11 D8: (1536, 512))"
        elif peek is None or stable is None:
            why = "N3 out-site: bwd_fp8 has no _peek_site_state / _has_stable_identity in this tree"
        elif not stable(output_weight):
            why = "N3 out-site: output_weight has no stable identity (the out-proj dgrad cast declines)"
        else:
            state = peek("output_projection", output_weight.data_ptr())
            if state is None or not state.published:
                why = (
                    "N3 out-warmup: the out-proj dgrad scale site is not published yet (its warm-up calls run "
                    "bf16, without L11 / DO8)"
                )
    except Exception as error:  # noqa: BLE001 -- a context check never breaks a forward; it declines NIB
        why = f"N3 out-error: {type(error).__name__}: {str(error)[:160]}"
    _LNMD_MASK_NIB_CALL_ID = id(attn)
    _LNMD_MASK_NIB_CALL_WHY = why


def _lnmd_mask_nib_rows512(t: torch.Tensor, n: int) -> bool:
    """L11's D8 row layout for x / u: bf16 [n, 512], row stride 512, unit column stride, 16-byte aligned base (L11's
    own check, l11_outproj_lnmd._rows512, so the mirror cannot drift from the gate it mirrors)."""
    return l11_outproj_lnmd._rows512(t, torch.bfloat16, n)


def _lnmd_mask_nib_why_not(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    N: int,
    D: int,
    BLOCK_D: int,
    silu_u: bool,
) -> Optional[str]:
    """DLRM_HSTU_FP8_LNMD_MASK_NIB per-call gate, evaluated inside item (28)(e)'s maskgen branch of
    _triton_layer_norm_mul_dropout_fwd_impl (so maskgen already armed for this call: gfx950, training, the
    full-row generator, mask16, e4m3 output, 'silu', D == BLOCK_D == 512, N < 11,184,810, ...). None = write the
    NIB mask for this call; otherwise the reason ("<code> <key>: <text>"). NIB arms only where this call's
    backward will hand the mask to DO8's NIB 'only' build (or, at an MX6 site, its NIB MX6 build) and to nothing
    else, so that steady-state training runs no exact unpack (the in-model census treats one as VOID):
      N0 the knob (exactly "1", read per call); off -> _LNMD_MASK_NIB_OFF before any state is read or written;
      N1 not latched (_LNMD_MASK_NIB_BROKEN: a NIB writer launch raised);
      N2/N3 the out-proj context HSTUComputeOutputFunction.forward recorded for THIS x (_lnmd_mask_nib_note_call:
         a backward, the FP8 out-proj wgrad branch, output_weight (1536, 512), its dgrad scale site published);
      N4 the FP8 wgrad SNR probe off (L11 D5) and not compiling (L11 D3);
      N5 L11 (item (39)) requested, L11-VD on (DO8 then builds l11_do8.cu, the NIB reader's source), enabled
         and prebuilt;
      N6 DO8's NIB reader ready (l11_do8.nib_ready(): DO8 requested, enabled, prebuilt, not latched; NIB v2:
         DO8-ONLY requested and its loaded 'only' .so (its format is MX6's too) prebuilt, not latched, reading NIB);
      N7 a dO8 site on x (l11_do8.site_of; x is the attention output HSTUComputeOutputFunction.forward
         received: the custom op hands the impl that same tensor object, so the site tag is visible) with
         T == N, 4 heads of 128 and well-formed seq_offsets (DO8's emission_site checks);
      N8 B1a (item (38)(a)) requested and enabled, not latched, its diagnostics and anomaly mode off, the uvqk
         site on u published (B1a's plan: L11's e5m2-du mode, the only mode DO8 has a variant for);
      N9 L11's contract: e5m2 gradients (E5), silu_u (D6), LNMD bwd v2 (D7), N <= M_MAX and the x / u / weight
         / bias layout (D8), the d(uvqk) row width 2048 on u (D9).
    Host metadata only: no sync, no allocation, no launch, no line (the caller counts and prints a decline)."""
    global _LNMD_MASK_NIB_CALL_ID, _LNMD_MASK_NIB_CALL_WHY
    if not lnmd_mask_nib.requested():
        return _LNMD_MASK_NIB_OFF
    call_id, call_why = _LNMD_MASK_NIB_CALL_ID, _LNMD_MASK_NIB_CALL_WHY
    _LNMD_MASK_NIB_CALL_ID, _LNMD_MASK_NIB_CALL_WHY = 0, ""
    if _LNMD_MASK_NIB_BROKEN is not None:
        return f"N1 latched: {_LNMD_MASK_NIB_BROKEN}"
    if call_id == 0 or call_id != id(x):
        return (
            "N2 no-context: not an HSTUComputeOutputFunction FP8-output call (a direct call, e.g. the FP8 "
            "first-use warmup)"
        )
    if call_why:
        return call_why
    if fp8_wgrad_snr.SNR_ENABLED:
        return "N4 snr: the FP8 out-proj wgrad SNR probe is on (L11 D5 declines every call)"
    if torch.compiler.is_compiling() or os.environ.get("AOTI_LOWER"):
        return "N4 compile: compiling / AOTI (L11 is eager only, D3)"
    if not l11_outproj_lnmd.requested():
        return "N5 l11-off: L11 item (39) off (DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED != 1)"
    if not l11_outproj_lnmd.vd_requested():
        return "N5 l11-vd: L11-VD off (DLRM_HSTU_FP8_L11_VD != 1): DO8 builds l11_do8_novd.cu, without a NIB reader"
    if not (l11_outproj_lnmd.enabled() and l11_outproj_lnmd.prebuilt()):
        return "N5 l11-ready: L11 item (39) latched off or not prebuilt before RUN_START"
    nib_ready = getattr(l11_do8, "nib_ready", None)
    if nib_ready is None:
        return "N6 do8-api: no DO8 NIB reader in this tree (l11_do8.nib_ready missing)"
    if not nib_ready():
        return (
            "N6 do8-ready: DO8 NIB reader not ready (DO8 / DO8-ONLY off, latched or not prebuilt, or its loaded "
            "'only' .so reads the int8 mask)"
        )
    site = l11_do8.site_of(x)
    if site is None:
        return "N7 no-site: no dO8 site on x (the C8 attention backward does not plan do_blk for this call)"
    if int(site.T) != int(N) or (int(site.H), int(site.D)) != (int(l11_do8.H), int(l11_do8.HD)):
        return f"N7 site-shape: dO8 site T={site.T} H={site.H} D={site.D} vs N={N} (DO8: 4 heads of 128)"
    if site.device != x.device:
        return f"N7 site-device: dO8 site on {site.device}, x on {x.device}"
    offsets_ok = getattr(l11_do8, "_offsets_ok", None)
    offsets_why = offsets_ok(site.seq_offsets, x.device) if offsets_ok is not None else None
    if offsets_why is not None:
        return f"N7 site-offsets: dO8 site {offsets_why}"
    if not bwd_fp8.du_producer_requested():
        return "N8 b1a-off: B1a item (38)(a) off (DLRM_HSTU_FP8_BWD_DU_E5M2): DO8 has no bf16-du variant"
    if not bwd_fp8.du_producer_enabled() or hstu_du_e5m2.broken() is not None:
        return "N8 b1a-ready: B1a item (38)(a) not enabled (e5m2 grads + fp8t uvqk wgrad) or latched off"
    if hstu_du_e5m2.diagnostics_on() or torch.is_anomaly_enabled():
        return "N8 b1a-diag: uvqk backward diagnostics or autograd anomaly mode on (B1a's plan declines)"
    b1a_site = getattr(u, HSTU_DUVQK_SITE_ATTR, None)
    if b1a_site is None:
        return "N8 b1a-site: no B1a uvqk site on u (the preprocess forward did not plan the fp8t uvqk wgrad)"
    if bwd_fp8.producer_generation(hstu_du_e5m2.SITE, b1a_site.key) is None:
        return "N8 b1a-warmup: the B1a uvqk scale site is not published yet (L11 runs its bf16-du mode)"
    grad_dtype, limit = bwd_fp8.grad_dtype_and_limit()
    if grad_dtype is not torch.float8_e5m2 or float(limit) != l11_outproj_lnmd.E5_LIMIT:
        return f"N9 e5: gradient dtype {grad_dtype} limit {limit} (L11 E5: e5m2, {l11_outproj_lnmd.E5_LIMIT})"
    if not silu_u or not _use_lnmd_bwd_v2():
        return f"N9 flags: silu_u={silu_u} lnmd_bwd_v2={_use_lnmd_bwd_v2()} (L11 D6 / D7)"
    if not (1 <= int(N) <= l11_outproj_lnmd.M_MAX) or int(D) != l11_outproj_lnmd.D or int(BLOCK_D) != int(D):
        return f"N9 shape: N={N} D={D} BLOCK_D={BLOCK_D} (L11 D8: 1 <= N <= {l11_outproj_lnmd.M_MAX}, D = 512)"
    if not (_lnmd_mask_nib_rows512(x, int(N)) and _lnmd_mask_nib_rows512(u, int(N))):
        return f"N9 layout: x {x.dtype}{tuple(x.stride())} / u {u.dtype}{tuple(u.stride())} (L11 D8)"
    for name, t in (("weight", weight), ("bias", bias)):
        if (
            t.dtype != torch.bfloat16
            or tuple(t.shape) != (l11_outproj_lnmd.D,)
            or not t.is_contiguous()
            or t.data_ptr() % 16 != 0
        ):
            return f"N9 layout: LN {name} {t.dtype} {tuple(t.shape)} (L11 D8)"
    width = int(getattr(u, HSTU_DUVQK_WIDTH_ATTR, 0))
    if width != l11_outproj_lnmd.DU_ROW:
        return f"N9 du-width: d(uvqk) row width {width} on u (L11 D9: {l11_outproj_lnmd.DU_ROW})"
    return None


def lnmd_mask_nib_counters() -> Dict[str, Any]:
    """armed = calls that wrote a NIB mask (written_sh + written_pk); declined = knob-on maskgen calls the
    predicate refused (int8 mask written); failed = NIB writer launches that raised (latched); unpack = exact
    int8 unpacks of a NIB mask by a non-DO8 consumer (ops/lnmd_mask_nib.as_int8; 0 in steady state)."""
    return {
        "armed": _LNMD_MASK_NIB_N_ARMED,
        "written_sh": _LNMD_MASK_NIB_N_SH,
        "written_pk": _LNMD_MASK_NIB_N_PK,
        "declined": _LNMD_MASK_NIB_N_DECLINED,
        "failed": _LNMD_MASK_NIB_N_FAILED,
        "latched": int(_LNMD_MASK_NIB_BROKEN is not None),
        "unpack": int(lnmd_mask_nib.counters().get("unpack", 0)),
    }


def _lnmd_mask_nib_counters_line(final: bool = False) -> str:
    c = lnmd_mask_nib_counters()
    return (
        f"[lnmd-mask-nib] counters: armed={c['armed']} written_sh={c['written_sh']} written_pk={c['written_pk']} "
        f"declined={c['declined']} failed={c['failed']} unpack={c['unpack']} latched={c['latched']}"
        + (" final=1" if final else "")
    )


def _lnmd_mask_nib_final() -> None:
    """The exit counters line (" final=1"), knob "1" only, once per process. Registered with atexit at the first NIB
    write and listed in dlrm_v4/train/clean_exit.EXIT_HOOKS (review F1: the default teardown ends in os._exit, which
    skips atexit), so a knob-on run prints it even when NIB never armed."""
    global _LNMD_MASK_NIB_FINAL_DONE
    if _LNMD_MASK_NIB_FINAL_DONE or not lnmd_mask_nib.requested():
        return
    _LNMD_MASK_NIB_FINAL_DONE = True
    arm_print_all_ranks(_lnmd_mask_nib_counters_line(final=True))


def _lnmd_mask_nib_note_decline(why: str) -> None:
    """A knob-on maskgen call the predicate refused: counted; one "NOT armed for this call" line per reason key
    ("<code> <key>") per process, except for the expected states (_LNMD_MASK_NIB_QUIET: a latched process, which
    printed its DISABLED line; a direct call, e.g. the FP8 first-use warmup; the first FP8 steps before the
    out-proj / B1a scale sites publish, B1a's showing up as no uvqk site on u; a call without a dO8 site), which
    are counted only (the counters line, incl. the exit line, carries them)."""
    global _LNMD_MASK_NIB_N_DECLINED, _LNMD_MASK_NIB_DECLINES_LOGGED
    _LNMD_MASK_NIB_N_DECLINED += 1
    key = "|" + why.split(":", 1)[0] + "|"
    if key in _LNMD_MASK_NIB_QUIET or key in _LNMD_MASK_NIB_DECLINES_LOGGED:
        return
    _LNMD_MASK_NIB_DECLINES_LOGGED += key
    # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
    arm_print_all_ranks(
        f"{_LNMD_MASK_NIB_FAMILY} NOT armed for this call ({_LNMD_MASK_NIB_KNOB}): {why}; the int8 keep mask is "
        "written (MGPK / the shipped maskgen; bitwise either way; one line per reason)"
    )


def lnmd_mask_nib_latch(reason: str) -> None:
    """Latch NIB off for the process with one DISABLED line on every rank (no-op when already latched): every later
    maskgen call declines (N1) and writes the int8 keep mask (MGPK / the shipped maskgen). Callers: a NIB writer
    launch that raised (_lnmd_mask_nib_latch) and the FP8 first-use warmup's startup byte check (item L4n,
    dlrm_v4/train/fp8_warmup.py _nib_items, outside the item's stdout capture; that module keeps
    _LNMD_MASK_NIB_BROKEN out of its scalar snapshot / restore, _KEEP_GLOBALS, so the latch outlives the warmup).
    With DO8's NIB 'only' .so loaded, DO8 then runs every later call as 'plus' on the int8 mask (no 'only' plan;
    counted mask_int8_plus, ops/hip/l11_do8.py NIB v2)."""
    global _LNMD_MASK_NIB_BROKEN
    if _LNMD_MASK_NIB_BROKEN is not None:
        return
    _LNMD_MASK_NIB_BROKEN = reason
    # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
    arm_print_all_ranks(
        f"{_LNMD_MASK_NIB_FAMILY} DISABLED ({_LNMD_MASK_NIB_KNOB}): {reason}; the int8 "
        "keep mask (MGPK / the shipped _ln_mul_dropout_fwd_maskgen) is written for the rest of the process"
    )


def _lnmd_mask_nib_latch(error: BaseException, pk: bool) -> None:
    """A NIB writer launch raised (e.g. a compile error): latch NIB off for the process (one line on every rank);
    the caller then runs today's path for this call (MGPK / the shipped maskgen into a fresh int8 mask), which
    writes every output byte."""
    global _LNMD_MASK_NIB_N_FAILED
    kernel = _LNMD_MASK_NIB_PK_KERNEL if pk else _LNMD_MASK_NIB_SH_KERNEL
    _LNMD_MASK_NIB_N_FAILED += 1
    lnmd_mask_nib_latch(f"launch of {kernel} raised {type(error).__name__}: {str(error)[:200]}")


def _lnmd_mask_nib_armed(pk: bool, N: int) -> None:
    """After a successful NIB writer launch: count it; the ACTIVE line once per process (first launch), one
    first-launch line per kernel, log-spaced counter lines (armed = 1, 2, 4, ...) and one at exit."""
    global _LNMD_MASK_NIB_N_ARMED, _LNMD_MASK_NIB_N_SH, _LNMD_MASK_NIB_N_PK, _LNMD_MASK_NIB_ARMED_LOGGED
    global _LNMD_MASK_NIB_SH_LOGGED, _LNMD_MASK_NIB_PK_LOGGED, _LNMD_MASK_NIB_ATEXIT
    kernel = _LNMD_MASK_NIB_PK_KERNEL if pk else _LNMD_MASK_NIB_SH_KERNEL
    _LNMD_MASK_NIB_N_ARMED += 1
    if pk:
        _LNMD_MASK_NIB_N_PK += 1
    else:
        _LNMD_MASK_NIB_N_SH += 1
    if not _LNMD_MASK_NIB_ARMED_LOGGED:
        _LNMD_MASK_NIB_ARMED_LOGGED = True
        # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
        arm_print_all_ranks(
            f"armed OK: {_LNMD_MASK_NIB_FAMILY} ACTIVE ({_LNMD_MASK_NIB_KNOB}, item {_LNMD_MASK_NIB_ITEM}): "
            "item (28)(e)'s maskgen writes the keep mask as uint8 [N, 256] (layout A, 256 B "
            "per row instead of the int8 [N, 512]), bitwise (same keep bits, e4m3 y, mean, rstd); first launch: "
            f"{kernel} on layer tag {_LNMD_LAYER_TAG}, N={N}; the backward hands it only to DO8's NIB 'only' build "
            f"(impl={l11_do8.only_impl()}); any other consumer gets the exact int8 unpack (counted)"
        )
    first = (pk and not _LNMD_MASK_NIB_PK_LOGGED) or (not pk and not _LNMD_MASK_NIB_SH_LOGGED)
    if first:
        if pk:
            _LNMD_MASK_NIB_PK_LOGGED = True
        else:
            _LNMD_MASK_NIB_SH_LOGGED = True
        # every rank, tagged [rank N] (see ops.utils.arm_print_all_ranks)
        arm_print_all_ranks(
            f"[lnmd-mask-nib] first launch of {kernel} ({'MGPK' if pk else 'shipped maskgen'} + NIB store) on "
            f"layer tag {_LNMD_LAYER_TAG}, N={N} ({_LNMD_MASK_NIB_KNOB})"
        )
    if not _LNMD_MASK_NIB_ATEXIT:
        _LNMD_MASK_NIB_ATEXIT = True
        atexit.register(_lnmd_mask_nib_final)
    n = _LNMD_MASK_NIB_N_ARMED
    if n & (n - 1) == 0:
        arm_print_all_ranks(_lnmd_mask_nib_counters_line())


def _lnmd_fwd_maskgen_nib_try(
    x: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    N: int,
    D: int,
    eps: float,
    dropout_ratio: float,
    inv_scale: float,
    seed: int,
    BLOCK_D: int,
    concat_u: bool,
    concat_x: bool,
    mul_u_activation_type: str,
    output_fp8: bool,
    silu_u: bool,
) -> Optional[torch.Tensor]:
    """NIB at item (28)(e)'s maskgen launch site (the int8 mask allocation line calls this first): the NIB mask
    uint8 [N, 256] when _lnmd_mask_nib_why_not armed this call and the writer launch succeeded (the caller then
    returns y, mean, rstd and that mask, and allocates no int8 [N, 512] mask); None when the caller must run
    today's path (knob off; a declined call; a launch that raised, which latches NIB off for the process).
    The writer follows MGPK's own per-call gate: _ln_mul_dropout_fwd_maskgen_pk_nib where MGPK would run (its
    knob, layer list, fp32 inv_scale and alignment checks, latch; its own lines unchanged), else
    _ln_mul_dropout_fwd_maskgen_nib. Same launch arguments (keep_min16 / kk2 as the launch site computes them)."""
    try:
        why = _lnmd_mask_nib_why_not(x, u, weight, bias, N, D, BLOCK_D, silu_u)
    except Exception as error:  # noqa: BLE001 -- a gate error never breaks a forward: the int8 mask for this call
        why = f"N10 gate-error: {type(error).__name__}: {str(error)[:160]}"
    if why is not None:
        if why != _LNMD_MASK_NIB_OFF:
            _lnmd_mask_nib_note_decline(why)
        return None
    keep_min16, kk2 = _lnmd_maskgen_kk2(dropout_ratio)
    packed = torch.empty([N, D // 2], dtype=torch.uint8, device=x.device)
    pk = _use_lnmd_fwd_maskgen_pk(inv_scale, y, packed)
    try:
        _lnmd_fwd_maskgen_nib_launch(
            x, u, y, weight, bias, mean, rstd, packed, N, D, eps, dropout_ratio, inv_scale, keep_min16, kk2,
            seed, BLOCK_D, concat_u, concat_x, mul_u_activation_type, output_fp8, silu_u, pk,
        )
    except Exception as error:
        _lnmd_mask_nib_latch(error, pk)
        return None
    _lnmd_mask_nib_armed(pk, N)
    if pk:
        # MGPK's own ACTIVE line (unchanged text, once per process): its VALU-diet kernel body runs here, with
        # the NIB mask store (the NIB lines above name the kernel and the mask format).
        _lnmd_fwd_maskgen_pk_armed()
    return packed


def _lnmd_fwd_maskgen_int8_launch(
    x: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    random_mask: torch.Tensor,
    N: int,
    D: int,
    eps: float,
    dropout_ratio: float,
    inv_scale: float,
    keep_min16: int,
    kk2: int,
    seed: int,
    BLOCK_D: int,
    concat_u: bool,
    concat_x: bool,
    mul_u_activation_type: str,
    output_fp8: bool,
    silu_u: bool,
    pk: bool,
) -> None:
    """The int8 kernel a NIB writer replaces, launched exactly as item (28)(e)'s launch site (pk=False: the shipped
    _ln_mul_dropout_fwd_maskgen) or _lnmd_fwd_maskgen_pk_try (pk=True: MGPK's _ln_mul_dropout_fwd_maskgen_pk) launch
    it, with no gate and no MGPK / maskgen state (counters, lines, latches) touched: the reference of the FP8
    warmup's startup byte check (lnmd_mask_nib_warmup_launch, compare=True). random_mask: int8 [N, BLOCK_D]."""
    kernel = _ln_mul_dropout_fwd_maskgen_pk if pk else _ln_mul_dropout_fwd_maskgen
    # pyre-ignore[28]
    kernel[(triton.cdiv(N, 8),)](
        x,
        u,
        y,
        weight,
        bias,
        mean,
        rstd,
        random_mask,
        N,
        D,
        eps,
        dropout_ratio,
        x.stride(0),
        u.stride(0),
        y.stride(0),
        random_mask.stride(0),
        float(inv_scale),
        keep_min16,
        kk2,
        seed,
        BLOCK_D=BLOCK_D,
        NUM_MASKS=1 + int(concat_u) + int(concat_x),
        MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
        OUTPUT_FP8=output_fp8,
        SILU_U=silu_u,
        num_warps=1,
    )


# Rows per block of the startup check's comparisons (bounds their temporaries).
_LNMD_MASK_NIB_CHECK_ROWS = 1 << 18


def _lnmd_mask_nib_same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """a and b hold the same bytes (floats compared as integers, so NaN payloads compare too), compared in blocks
    of _LNMD_MASK_NIB_CHECK_ROWS rows so the comparison's temporaries stay small."""
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return False
    if a.dtype.is_floating_point:
        as_int = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        a, b = a.view(as_int), b.view(as_int)
    rows = int(a.shape[0]) if a.dim() > 0 else 1
    for i in range(0, max(rows, 1), _LNMD_MASK_NIB_CHECK_ROWS):
        if not torch.equal(a[i : i + _LNMD_MASK_NIB_CHECK_ROWS], b[i : i + _LNMD_MASK_NIB_CHECK_ROWS]):
            return False
    return True


def _lnmd_mask_nib_check(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    y: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    packed: torch.Tensor,
    N: int,
    D: int,
    eps: float,
    dropout_ratio: float,
    inv_scale: float,
    keep_min16: int,
    kk2: int,
    seed: int,
    BLOCK_D: int,
    silu_u: bool,
    pk: bool,
) -> Optional[str]:
    """The FP8 warmup's startup byte check of one NIB writer launch (review R1-F5): the int8 kernel it replaces runs
    on the same inputs into its own outputs, prefilled with junk bytes other than the NIB side's (an unwritten byte
    cannot compare equal); then y / mean / rstd must be the same bytes, the NIB mask must equal
    lnmd_mask_nib.pack_ref(int8 mask), and the int8 mask's bytes must be <= 7 (its writer's invariant). None = all
    equal; otherwise what differs. Transient memory: the int8 kernel's outputs (y8 is freed after its compare)."""
    y8 = torch.empty_like(y)
    mean8 = torch.empty_like(mean)
    rstd8 = torch.empty_like(rstd)
    mask8 = torch.empty((N, D), dtype=torch.int8, device=x.device)
    y8.view(torch.uint8).fill_(0xA5)
    mean8.view(torch.int32).fill_(0x7FC00012)
    rstd8.view(torch.int32).fill_(0x7FC00022)
    mask8.fill_(0x7F)
    _lnmd_fwd_maskgen_int8_launch(
        x, u, y8, weight, bias, mean8, rstd8, mask8, N, D, eps, dropout_ratio, inv_scale, keep_min16, kk2, seed,
        BLOCK_D, True, True, "silu", True, silu_u, pk,
    )
    bad = []
    if not _lnmd_mask_nib_same_bytes(y, y8):
        bad.append("y")
    del y8
    if not _lnmd_mask_nib_same_bytes(mean, mean8):
        bad.append("mean")
    if not _lnmd_mask_nib_same_bytes(rstd, rstd8):
        bad.append("rstd")
    mask_ok = int8_ok = True
    for i in range(0, N, _LNMD_MASK_NIB_CHECK_ROWS):
        m8 = mask8[i : i + _LNMD_MASK_NIB_CHECK_ROWS]
        if int8_ok and bool((m8.view(torch.uint8) > 7).any()):
            int8_ok = False
        if mask_ok and not torch.equal(packed[i : i + _LNMD_MASK_NIB_CHECK_ROWS], lnmd_mask_nib.pack_ref(m8)):
            mask_ok = False
    if not mask_ok:
        bad.append("keep mask (NIB != pack_ref(int8))")
    if not int8_ok:
        bad.append("int8 reference mask bytes > 7")
    return None if not bad else "bytes differ: " + ", ".join(bad)


def lnmd_mask_nib_warmup_launch(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    seed: int,
    inv_scale: float,
    silu_u: bool,
    compare: bool = False,
    verdict: Optional[Dict[str, str]] = None,
) -> str:
    """FP8 first-use warmup item L4n (dlrm_v4/train/fp8_warmup.py; knob on only): launch BOTH NIB writers once on
    the caller's synthetic x / u [N, 512] bf16 with the maskgen branch's production launch arguments (concat_u,
    concat_x, 'silu', e4m3 output, the caller's dropout ratio, seed and inv_scale: the specializations a training
    step hits for this N % 16 / seed % 16 / storage class) into fresh outputs, so neither compiles inside the
    timed run. The NIB predicate is not consulted (the warmup's x carries no dO8 site), and no NIB, MGPK or
    maskgen state (counters, lines, latches) is touched. pk-NIB only where MGPK's kernel contract holds (fp32
    inv_scale in (0, inf)). A raise propagates (the warmup logs a FAIL line for the item). Returns a short tag
    for the item's ok line ("nib=sh+pk", plus " cmp=ok" / " cmp=MISMATCH(<writers>)" with compare).
    compare=True (review R1-F5; L4n's rows below the > 2 GiB x / u class): the NIB outputs are junk-filled first,
    and after each writer launch _lnmd_mask_nib_check runs the int8 kernel it replaces (the shipped maskgen for
    the sh writer, MGPK's kernel for the pk writer) on the same inputs and compares every output byte. A
    mismatch is NOT latched here (this runs inside the warmup item's stdout capture, which would swallow the
    DISABLED line): its reason goes into verdict["mismatch"] (the first one) and the caller latches NIB
    (lnmd_mask_nib_latch) after the item."""
    N, D = int(x.shape[0]), int(x.shape[1])
    BLOCK_D = min(65536 // x.element_size(), triton.next_power_of_2(D))
    keep_min16, kk2 = _lnmd_maskgen_kk2(dropout_ratio)
    y = torch.empty((N, 3 * D), dtype=torch.float8_e4m3fn, device=x.device)
    mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    packed = torch.empty([N, D // 2], dtype=torch.uint8, device=x.device)
    done = []
    bad = []
    for pk in (False, True):
        if pk and not 0.0 < _fp32_kernel_arg(float(inv_scale)) < math.inf:
            continue
        if compare:
            # junk-fill every NIB output: a byte the writer leaves unwritten cannot match the reference
            y.view(torch.uint8).fill_(0x5A)
            mean.view(torch.int32).fill_(0x7FC00011)
            rstd.view(torch.int32).fill_(0x7FC00021)
            packed.fill_(0xFF)  # bits 3 and 7 set: never a pack_ref byte
        _lnmd_fwd_maskgen_nib_launch(
            x, u, y, weight, bias, mean, rstd, packed, N, D, eps, dropout_ratio, inv_scale, keep_min16, kk2,
            seed, BLOCK_D, True, True, "silu", True, silu_u, pk,
        )
        tag = "pk" if pk else "sh"
        done.append(tag)
        if compare:
            why = _lnmd_mask_nib_check(
                x, u, weight, bias, y, mean, rstd, packed, N, D, eps, dropout_ratio, inv_scale, keep_min16, kk2,
                seed, BLOCK_D, silu_u, pk,
            )
            if why is not None:
                bad.append(tag)
                if verdict is not None and "mismatch" not in verdict:
                    kernel = _LNMD_MASK_NIB_PK_KERNEL if pk else _LNMD_MASK_NIB_SH_KERNEL
                    ref = "_ln_mul_dropout_fwd_maskgen_pk" if pk else "_ln_mul_dropout_fwd_maskgen"
                    verdict["mismatch"] = (
                        f"FP8 warmup startup check (item L4n): {kernel} vs {ref} on the same inputs (N={N}, "
                        f"seed={seed}): {why}"
                    )
    out = "nib=" + "+".join(done)
    if compare:
        out += " cmp=" + (f"MISMATCH({'+'.join(bad)})" if bad else "ok")
    return out


# =============================================================================
# r14 E-RM (DLRM_HSTU_FP8_BWD_ERM, yambda_5b.gin item (97); code default OFF;
# ops/triton/dy_erm.py). Appended at the END of the module on purpose, as MGPK
# above: no @triton.jit function defined above moves, so every shipped kernel
# keeps its starting line and therefore its Triton cache_key, its disk-cache key
# and its TSPEC fn_cache_key. The seams above are seven single-line
# substitutions in HSTUComputeOutputFunction (no line added or removed): the
# forward's `return out` returns _erm_forward_tag(...), the two cast calls of
# the backward pass erm=_erm_take(ctx), and the four reads of the wgrad dout
# operand (the W3 call, the two hipBLASLt re-lay-outs, the SNR probe's copy) go
# through the _erm_* wrappers below. With the knob off every wrapper returns at
# its first test and calls the shipped function with the shipped arguments: no
# kernel, no allocation, no log line.
# =============================================================================
from generative_recommenders.ops.triton import dy_erm as _dy_erm  # noqa: E402


def _erm_forward_tag(ctx, out, output_weight, need_bwd):  # pyre-ignore[2,3]
    """HSTUComputeOutputFunction.forward's return value: `out`, tagged with this
    out-projection's E-RM site when it is the stack's bottom layer (layer tag 0)
    and its FP8 NN wgrad will run (dy_erm.make_site)."""
    if not _dy_erm.requested():
        return out
    site = _dy_erm.make_site(
        output_weight,
        fp8_y_saved=bool(getattr(ctx, "fp8_y_saved", False)),
        y_token_major=bool(getattr(ctx, "fp8_y_token_major", False)),
        dout_token_major=bool(getattr(ctx, "fp8_dout_token_major", False)),
        out_wgrad_nn=_FP8_OUT_WGRAD and _FP8_OUT_WGRAD_IMPL == "nn",
        need_bwd=bool(need_bwd),
        layer_tag=_LNMD_LAYER_TAG,
        helpers_fused_ok=bwd_fp8._helpers_fused_ok(output_weight),
    )
    ctx.erm_site = site
    if site is not None:
        setattr(out, _dy_erm.ERM_SITE_ATTR, site)
    return out


def _erm_take(ctx):  # pyre-ignore[2,3]
    """The backward's `erm=` argument of its one dual-cast call: this site's
    emission record (dy_erm.take), kept on ctx for the wgrad readers; None
    without a site (always with the knob off)."""
    site = getattr(ctx, "erm_site", None)
    if site is None:
        return None
    rec = _dy_erm.take(site)
    ctx.erm_rec = rec
    return rec


def _erm_dout_rm(ctx) -> bool:  # pyre-ignore[2]
    """The wgrad operand the backward holds is the producer's row-major x_rm
    [M, 512] (explicit layout tag; never inferred from dim())."""
    rec = getattr(ctx, "erm_rec", None)
    return rec is not None and rec.used and rec.layout == "rm"


def _erm_out_wgrad_nn_ck_w3(
    ctx,  # pyre-ignore[2]
    y_fp8: torch.Tensor,
    dout_fp8: torch.Tensor,
    y_scale: torch.Tensor,
    dout_scale: torch.Tensor,
    fp8_wgrad_nn,  # pyre-ignore[2]
) -> Optional[Tuple[Optional[torch.Tensor], str]]:
    """`_out_wgrad_nn_ck_w3` (unchanged) unless the operand is E-RM's x_rm; then
    W3's row-major entry (wgrad_ck_w3.run_w3_rm): (dw_t, "nn_ck"), bitwise the
    knob-off x21 result; any row-major failure returns x21's own result on the
    step-tiled re-lay-out for that call; None when that x21 call fails or a
    guard deferred by an out-of-memory copy meets a non-finite xt2 result
    (the caller then re-lays x_rm out for hipBLASLt, _erm_dout_row_major)."""
    if not _erm_dout_rm(ctx):
        return _out_wgrad_nn_ck_w3(y_fp8, dout_fp8, y_scale, dout_scale, fp8_wgrad_nn)
    m = int(y_fp8.shape[0])
    hip: List[Optional[torch.Tensor]] = []

    def reference() -> torch.Tensor:
        # x21's own first-call guard reference (never reached: run_w3_rm runs
        # only where x21 is armed on this device); hipBLASLt on rm_to_tok(x_rm).
        if not hip:
            try:
                operand = wgrad_ck_w3.rm_to_tok(dout_fp8, m, empty=fp8_tm_alloc.empty_copy)
            except Exception as error:
                if not fp8_tm_alloc.is_oom(error):
                    raise
                raise wgrad_ck.GuardDeferred("re-laid-out x_rm copy out of memory")
            hip.append(
                fp8_wgrad_nn(y_fp8, operand, y_scale, dout_scale, out_dtype=torch.float32)
            )
        ref = hip[0]
        if ref is None:
            raise RuntimeError("hipBLASLt NN reference unavailable (no algorithm)")
        return ref

    def retile(x_rm: torch.Tensor, mm: int) -> torch.Tensor:
        # the dual cast's step-tiled dout_t bytes: a transient copy from the B11
        # pool's copy allocator, as _dout_row_major's (fp8_tm_alloc.empty_copy:
        # the mapped spare block, else torch.empty; out of memory propagates
        # to run_w3_rm WITHOUT latching the pool, unlike empty_tm)
        return wgrad_ck_w3.rm_to_tiled(x_rm, mm, empty=fp8_tm_alloc.empty_copy)

    def run_x21(dout_t: torch.Tensor) -> Optional[torch.Tensor]:
        return wgrad_ck_w3.run_w3(y_fp8, dout_t, y_scale, dout_scale, m, reference)

    dw_t = wgrad_ck_w3.run_w3_rm(
        y_fp8, dout_fp8, y_scale, dout_scale, m, retile, run_x21, fp8_tm_alloc.is_oom
    )
    if dw_t is not None:
        return dw_t, "nn_ck"
    if hip:
        return hip[0], "nn"
    return None


def _erm_dout_row_major(
    ctx, dout_fp8: torch.Tensor, m: int, required: bool  # pyre-ignore[2]
) -> Optional[torch.Tensor]:
    """`_dout_row_major` (unchanged) unless the operand is E-RM's x_rm; then its
    [512, roundup(M, 16)] re-lay-out (wgrad_ck_w3.rm_to_tok: bitwise
    untile_to_row_major of the dual cast's dout_t) from the B11 pool's copy
    allocator, with _dout_row_major's out-of-memory policy; the layout tag then
    reads "tok2d", so a second reader takes the copy as it is."""
    if not _erm_dout_rm(ctx):
        return _dout_row_major(dout_fp8, m, required)
    try:
        operand = wgrad_ck_w3.rm_to_tok(dout_fp8, m, empty=fp8_tm_alloc.empty_copy)
    except Exception as error:
        if not fp8_tm_alloc.is_oom(error):
            raise
        wgrad_ck_w3.log_once(
            "untile_oom_rm",
            f"out_projection: re-laid-out copy of the row-major x_rm "
            f"{tuple(dout_fp8.shape)} out of memory on {dout_fp8.device} "
            f"({type(error).__name__}) -> "
            + (
                "no allocation-free fallback for this call, raised"
                if required
                else "guard deferred for this call (nothing latched)"
            ),
        )
        if required:
            raise
        return None
    ctx.erm_rec.layout = "tok2d"
    return operand


def _erm_untile_to_row_major(ctx, dout_fp8: torch.Tensor, m: int) -> torch.Tensor:  # pyre-ignore[2]
    """The SNR probe's row-major copy: `untile_to_row_major` (unchanged) unless
    the operand is E-RM's x_rm (unreachable: the producer declines whenever the
    probe is on, dy_erm D10), then rm_to_tok."""
    if not _erm_dout_rm(ctx):
        return wgrad_ck_w3.untile_to_row_major(dout_fp8, m)
    return wgrad_ck_w3.rm_to_tok(dout_fp8, m)


# =============================================================================
# MX6 (yambda_5b.gin item (104), DLRM_HSTU_FP8_BWD_OUTPROJ_MXFP6; code default OFF, bound OFF; r16 code
# row). On an L11 DO8 call the dO8 registry plans 'only', at a STUStack layer of MX6's static site set, r14's
# producer (bwd_fp8.output_projection_dgrad_operands_mx6: the G6 row records, the W6 / W6S weight shadow, the W3
# dout_t operand and the e5m2 gradient site's scale bookkeeping, exactly once) replaces the FP8 dual cast, and one
# l11_outproj_lnmd_bwd_do8_mx6 launch (E2M3 x E2M3, E8M0 per 32) replaces the FP8 DO8 launch. Every decision,
# registry plan, launch, release / rescue count, latch and line lives in ops/hip/l11_do8.py (module docstring
# "MX6"); this block is the seam's wiring only. Appended at the END of the module on purpose, like MGPK, NIB and
# r14's E-RM rows: no @triton.jit function above moves, so every Triton kernel keeps its starting line and with it
# its Triton cache key, its disk-cache key and its TSPEC fn_cache_key. The seams above are one-for-one line
# substitutions (the line count above the last kernel is unchanged): prebuild_l11_do8's DO8-off return
# (l11_do8.prebuild_mx6: a lone MX6 knob prints its INERT line), the forward's B1a site line
# (_l11_mx6_forward_tag: the static site tag) and the backward's L11 dispatch (_l11_mx6_backward; the comment block
# above it was reflowed to make room). With the knob off the forward tag returns after one l11_do8.mx6_enabled()
# call and the backward dispatch after one getattr on ctx: no import, allocation, registry call, kernel or line.
# E-RM (r14's block above; its site is layer 0's out-proj): MX6 never takes an E-RM site. l11_do8's INERT rule keeps
# MX6 off layer 0 while E-RM is requested (a site set holding layer 0: nothing built); _l11_mx6_backward declines a
# call E-RM's forward tagged (ctx.erm_site) before anything else (one NOT armed line while MX6 is armed, quiet
# otherwise), so the shipped block takes that record (_erm_take, once); and _l11_mx6_fp8_block's cast passes
# erm=_erm_take(ctx) as the shipped cast does (None at every call MX6 reaches: MX6 itself never takes an E-RM record).
# =============================================================================


def _l11_mx6_forward_tag(ctx, du_site):  # pyre-ignore[2,3]
    """HSTUComputeOutputFunction.forward's B1a site line: returns `du_site` (the B1a site the preprocess forward
    tagged on u) unchanged. With MX6 armed (l11_do8.mx6_enabled) and this compute output's STUStack layer in MX6's
    static site set (l11_do8.mx6_site_selected; critique C1), the layer goes to ctx.l11_mx6_layer, the only thing the
    backward's MX6 dispatch reads. Captured in the FORWARD: _LNMD_LAYER_TAG (set by modules/stu.py
    STULayer._tag_lnmd_layer right before this compute output; -1 untagged) holds the LAST forward's layer during
    the backward. MX6 needs B1a's e5m2-du mode, so a call without a B1a site is never tagged. Knob off, INERT or
    latched: one l11_do8.mx6_enabled() call, nothing set."""
    if l11_do8.mx6_enabled() and du_site is not None and l11_do8.mx6_site_selected(_LNMD_LAYER_TAG):
        ctx.l11_mx6_layer = _LNMD_LAYER_TAG
    return du_site


def _l11_mx6_backward(
    ctx,  # pyre-ignore[2]
    dout: torch.Tensor,
    y_fp8: torch.Tensor,
    attn: torch.Tensor,
    u: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    output_weight: torch.Tensor,
    random_mask: torch.Tensor,
) -> Optional[Tuple[Any, ...]]:
    """HSTUComputeOutputFunction.backward's L11 dispatch for a call whose forward tagged an MX6 site (l11_go: L11's
    own decline_reason passed). None = MX6 declined before anything was evaluated (a call E-RM's forward tagged,
    ctx.erm_site, whose record the shipped block takes; l11_do8.mx6_preflight; or a dout_t that is not step-tiled):
    the shipped block then runs exactly as without MX6. Otherwise THIS call's whole L11 block, as the 8-tuple (dy,
    dout_fp8, dout_fp8_scale, l11_out, l11_e5, l11_do8_pair, du_plan, l11_du_planned) the shipped block leaves
    behind, so everything after it runs unchanged (B1a's record, the dO8 publish after the last write to dattn, the
    W3 wgrad).
    Order (the landing plan's transaction boundary): (1) preflight, host metadata only; (2) BEFORE the cast, B1a's
    plan ONCE (the shipped block evaluates it after the cast with the same meta dy and compute_y=False; it reads only
    B1a's own uvqk scale site, which the out-proj cast never touches), then l11_do8.mx6_decide: DO8's emission site
    and the dO8 registry plan, ONCE; (3) on a taken decision r14's producer (l11_do8.mx6_produce); (4) the MX6 launch
    (l11_do8.run_e5_only_mx6) -> an 'only' record published below like any other, with W3's dout_fp8 /
    dout_fp8_scale from the producer. Not taken, or released (the producer declined or raised: scale state untouched)
    -> _l11_mx6_fp8_block, the shipped block with this call's evaluations. A failure after the producer committed
    the scale state (run_e5_only_mx6 returns None: its plan closed, MX6 latched, the rescue counted) -> R2 rescue: the
    FP8 dual cast again at the CAPTURED inv_scale (l11_do8.mx6_rescue -> bwd_fp8.output_projection_g8_rescue: g8 /
    w8, no second scale-state update: checked), then today's chain from "DO8 returned None": L11's own E5 launch
    with the same B1a plan, else the dgrad GEMM (never a second DO8 attempt; W3 keeps the producer's dout_fp8).
    mx6_decide closes its own plan if it raises; from there on any exception leaves through mx6_abandon below."""
    if getattr(ctx, "erm_site", None) is not None:
        # E-RM's call (r14: its forward tags layer 0's out-proj only): never an MX6 call. Unreachable while
        # l11_do8's INERT rule holds (a site set holding layer 0 with E-RM requested); counted, one NOT armed line
        # while MX6 is armed (quiet otherwise, as the preflight's "latched or not prebuilt": a stale tag on a ctx
        # with MX6 off or latched prints nothing).
        l11_do8.mx6_note_decline("E-RM owns this call (the forward's E-RM site tag): the shipped cast takes its record",
                                 quiet=not l11_do8.mx6_enabled())
        return None
    token_major = bool(getattr(ctx, "fp8_dout_token_major", False))
    l11_site = getattr(ctx, "du_e5m2_site", None)
    if l11_site is None or l11_do8.mx6_preflight(
        ctx, attn, dout, output_weight, random_mask, token_major=token_major, snr=fp8_wgrad_snr.SNR_ENABLED
    ) is not None:
        return None
    t_tiled = _out_wgrad_dout_tiled(ctx, dout, y_fp8)
    if not t_tiled:
        l11_do8.mx6_note_decline("C3 the W3 dout_t is not step-tiled (the W3 tiled entry is off or latched)")
        return None
    du_row_width = getattr(ctx, "du_row_width", 0)
    kw: Dict[str, Any] = dict(
        x=attn,
        u=u,
        weight=norm_weight,
        bias=norm_bias,
        mean=mean,
        rstd=rstd,
        BLOCK_D=ctx.BLOCK_D,
        num_warps=ctx.num_warps,
        dropout_ratio=ctx.dropout_ratio,
        silu_u=ctx.silu_u,
        random_mask=random_mask,
        du_row_width=du_row_width,
        site_key=l11_site.key,
    )
    du_plan = _du_e5m2_plan(
        ctx,
        l11_site,
        torch.empty((dout.shape[0], output_weight.shape[0]), dtype=dout.dtype, device="meta"),
        attn,
        u,
        random_mask,
        du_row_width,
        compute_y=False,
    )
    dec = l11_do8.mx6_decide(
        ctx, attn, du_plan, u=u, BLOCK_D=ctx.BLOCK_D, num_warps=ctx.num_warps, silu_u=ctx.silu_u,
        du_row_width=du_row_width,
    )
    try:
        ops6 = None
        if dec.taken:
            ops6 = l11_do8.mx6_produce(dec, dout, output_weight, token_major=token_major, t_tiled=t_tiled)
        if ops6 is None:
            return _l11_mx6_fp8_block(ctx, dout, output_weight, token_major, t_tiled, du_plan, dec, kw)
        b1a = dict(inv_scale=du_plan[0], cast_scale=du_plan[1], generation=du_plan[2])
        with record_function("## hstu_l11_outproj_lnmd_bwd ##"):
            out = l11_do8.run_e5_only_mx6(ops6, **kw, **b1a, site=dec.site, plan=dec)
        if out is not None:
            return (None, ops6.dout_fp8, ops6.dout_fp8_scale, None, out[:5], (ctx.do8_site, out[5], out[6]),
                    du_plan, True)
        # R2 rescue (run_e5_only_mx6 closed the plan, latched MX6 and printed / counted the rescue; mx6_rescue
        # checks R2's contract on the host: no scale-state update, the FP8 dgrad operands)
        g8, g8_scale, w8, w8_scale = l11_do8.mx6_rescue(dout, output_weight, ops6)
        l11_e5 = _l11_du_e5m2_or_none(g8, w8, g8_scale, w8_scale, **kw, **b1a)
        dy = (
            None
            if l11_e5 is not None
            else bwd_fp8.output_projection_dgrad_gemm(g8, w8, g8_scale, w8_scale, dout.dtype)
        )
        return dy, ops6.dout_fp8, ops6.dout_fp8_scale, None, l11_e5, None, du_plan, True
    except BaseException:
        l11_do8.mx6_abandon(dec)  # an exception never leaves an 'only' plan of this decision open
        raise


def _l11_mx6_fp8_block(ctx, dout, output_weight, token_major, t_tiled, du_plan, dec, kw):  # pyre-ignore[2,3]
    """The shipped L11 block of HSTUComputeOutputFunction.backward (from the cast to `dy = ...`) for an MX6-site call
    MX6 did not take, with THIS call's B1a plan (du_plan), DO8 emission site and dO8 registry answer (dec) as the seam
    evaluated them ONCE (critique C10: no counter, plan or line twice). The same calls with the same arguments in the
    same order: the FP8 dual cast (its scale bookkeeping is the call's only one: the producer did not run or left the
    state untouched; erm=_erm_take(ctx) as in the shipped cast, None here: _l11_mx6_backward declined every call E-RM's
    forward tagged), then L11's bf16-du mode (no B1a plan), else DO8 (dec.only_plan False: the registry already
    answered 'plus' and is not asked again; None: as the shipped block, a release before it included), else L11's
    own E5 launch, else the dgrad GEMM. Returns the seam's 8-tuple."""
    random_mask = kw["random_mask"]
    l11_ops = bwd_fp8.output_projection_dgrad_operands(
        dout, output_weight, token_major=token_major, t_tiled=t_tiled, erm=_erm_take(ctx)
    )
    if l11_ops is None:
        # Not reached after the preflight, which checks every reason the cast declines for (warm-up included); the
        # shipped warm-up path if it were: dy None -> bf16 torch.mm, B1a's plan evaluated again on that dy.
        return None, None, None, None, None, None, None, False
    g8, g8_scale, dout_fp8, dout_fp8_scale, w8, w8_scale = l11_ops
    l11_out = l11_e5 = l11_do8_pair = None
    if du_plan is None:
        l11_do8.note_bf16_mode(ctx)  # DO8: no bf16-du variant
        with record_function("## hstu_l11_outproj_lnmd_bwd ##"):
            l11_out = l11_outproj_lnmd.run(
                g8,
                w8,
                g8_scale,
                w8_scale,
                kw["x"],
                kw["u"],
                lnmd_mask_nib.as_int8(random_mask, "L11 run (bf16-du mode)"),
                kw["mean"],
                kw["rstd"],
                kw["weight"],
                kw["bias"],
                ctx.dropout_ratio,
                _alloc_du,
            )
    else:
        b1a = dict(inv_scale=du_plan[0], cast_scale=du_plan[1], generation=du_plan[2])
        l11_do8_out = (
            _l11_mx6_do8_or_none(g8, w8, g8_scale, w8_scale, dec, kw, b1a) if l11_do8.requested() else None
        )
        if l11_do8_out is not None:
            l11_e5 = l11_do8_out[:5]
            l11_do8_pair = (ctx.do8_site, l11_do8_out[5], l11_do8_out[6])
        else:
            l11_e5 = _l11_du_e5m2_or_none(g8, w8, g8_scale, w8_scale, **kw, **b1a)
    dy = (
        None
        if l11_out is not None or l11_e5 is not None
        else bwd_fp8.output_projection_dgrad_gemm(g8, w8, g8_scale, w8_scale, dout.dtype)
    )
    return dy, dout_fp8, dout_fp8_scale, l11_out, l11_e5, l11_do8_pair, du_plan, True


def _l11_mx6_do8_or_none(g8, w8, g8_scale, w8_scale, dec, kw, b1a):  # pyre-ignore[2,3]
    """_l11_do8_e5_or_none for _l11_mx6_fp8_block: dec.only_plan None (the registry was not asked, or MX6 released
    its plan) -> exactly the shipped hook with this call's emission site; False (the registry answered 'plus' to
    mx6_decide) -> the same launch, run_e5_plus told the answer instead of asking again."""
    if dec.only_plan is not False:
        return _l11_do8_e5_or_none(g8, w8, g8_scale, w8_scale, site=dec.site, **kw, **b1a)
    if dec.site is None or not (l11_outproj_lnmd.enabled() and l11_outproj_lnmd.prebuilt()):
        return None
    with record_function("## hstu_l11_outproj_lnmd_bwd ##"):
        return l11_do8.run_e5_plus(
            g8,
            w8,
            g8_scale,
            w8_scale,
            kw["x"],
            kw["u"],
            kw["weight"],
            kw["bias"],
            kw["mean"],
            kw["rstd"],
            kw["BLOCK_D"],
            kw["num_warps"],
            kw["dropout_ratio"],
            kw["silu_u"],
            kw["random_mask"],
            kw["du_row_width"],
            kw["site_key"],
            b1a["inv_scale"],
            b1a["cast_scale"],
            b1a["generation"],
            dec.site,
            only_plan=False,
        )
