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

#!/usr/bin/env python3

# pyre-strict

import functools
import os
import logging
from typing import Callable, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl
from generative_recommenders.ops.blas_solution_pins import install_for_tokens
from generative_recommenders.ops.triton import bwd_fp8
from generative_recommenders.ops.triton.triton_addmm import (
    maybe_triton_addmm_fwd,
    triton_addmm_bwd,
    triton_addmm_fwd,
)
from generative_recommenders.ops.triton.triton_hstu_attention import (
    _should_enable_tma,
    triton_cached_hstu_mha,
    triton_hstu_attention_bwd,
    triton_hstu_attention_delta_bwd,
    triton_hstu_attention_fwd,
)
from generative_recommenders.ops.triton.triton_fp8 import (
    activation_scale_value,
    b15t_row_major_y,
    calibrate_scale_buffer,
    fp8_scale_tensor,
    fp8_supported,
    hstu_fp8_attention_enabled,
    hstu_fp8_fail_closed,
    hstu_fp8_fuse_ln_enabled,
    hstu_fp8_fuse_qkv_enabled,
    k512_tiled_a_requested,
    k512_tiled_a_vk_decline_reason,
    latch_b15t_off,
    log_b15t_decline_once,
    packed_weight_for_layout,
    refuse_fp8_with_flydsl_attention_forward,
    record_fp8_capability,
    record_fp8_contract,
    scaled_addmm_fp8_preq,
    scaled_mm_fp8_out_preq,
    scaled_mm_fp8_out_preq_tiled_a,
)
from generative_recommenders.ops.triton import hstu_du_e5m2
from generative_recommenders.ops.triton import wgrad_ck
from generative_recommenders.ops.flydsl import hstu_do8_registry
from generative_recommenders.ops.triton.triton_hstu_linear import (
    HSTU_DUVQK_BUFFER_ATTR,
    HSTU_DUVQK_SITE_ATTR,
    HSTU_DUVQK_WIDTH_ATTR,
)
from generative_recommenders.ops import glue_fold3
from generative_recommenders.ops.input_dropout_fold import (
    DropoutHandoff,
    input_dropout_fold_enabled,
    take_dropout_handoff,
)
from generative_recommenders.ops.triton.triton_layer_norm import (
    compute_BLOCK_D,
    triton_weighted_layer_norm_bwd,
    wln_bwd_mscale_supported,
    triton_weighted_layer_norm_dual_fwd,
    triton_weighted_layer_norm_fp8_save_bf16_fwd,
    triton_weighted_layer_norm_fwd,
    triton_weighted_layer_norm_fwd_fp8_store_t,
    triton_weighted_layer_norm_recompute_fp8t,
    store_t_skip_y_decline_reason,
)
from torch.nn import functional as F
from torch.autograd.profiler import record_function


_SAVE_FP8_UVQK = os.environ.get("DLRM_HSTU_FP8_SAVE_UVQK", "0") == "1"
_MSCALE_FOLD_LOGGED: bool = False


def _log_mscale_fold_once() -> None:
    global _MSCALE_FOLD_LOGGED
    _MSCALE_FOLD_LOGGED = True
    logging.getLogger(__name__).warning(
        "[hstu] input-dropout mask/scale folded into layer-0 LN backward "
        "ACTIVE (DLRM_HSTU_FP8_DROPOUT_MSCALE_FOLD=0 to disable)"
    )
_EARLY_RELEASE_FP8_UVQK = (
    os.environ.get("DLRM_HSTU_FP8_EARLY_RELEASE_UVQK", "0") == "1"
)
# Store-instead-of-recompute (instance F, user-approved class 2026-10-05): the
# FP8 forward LayerNorm also writes the BF16 normalized input, which is saved
# for the projection wgrad so backward no longer re-runs the LayerNorm. The
# stored tensor is bit-identical to the recompute from the stored fp32
# mean/rstd (L0, iter 2). The mean/rstd themselves come from a separately
# compiled/autotuned kernel variant, so their last fp32 bit can differ from
# the flag-off kernel, exactly as it already differs between autotune configs
# of the flag-off kernel: a declared reduction-order change, same formula.
# Costs 2 B per element of x held from forward to backward (~1 KiB/token/layer).
_SAVE_BF16_NORMED_X = (
    os.environ.get("DLRM_HSTU_FP8_SAVE_NORMED_X", "0") == "1"
)
@triton.jit
def _dequantize_fp8_kernel(
    value,
    output,
    scale,
    num_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = (
        tl.program_id(0).to(tl.int64) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    )
    mask = offsets < num_elements
    value_block = tl.load(value + offsets, mask=mask).to(tl.float32)
    dequant_scale = tl.load(scale).to(tl.float32)
    tl.store(output + offsets, value_block * dequant_scale, mask=mask)


@triton.jit
def _dequantize_fp8_strided_output_kernel(
    value,
    output,
    scale,
    rows: tl.constexpr,
    columns: tl.constexpr,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns_offset = (
        tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ).to(tl.int64)
    mask = columns_offset < columns
    input_offsets = row * input_row_stride + columns_offset
    output_offsets = row * output_row_stride + columns_offset
    value_block = tl.load(value + input_offsets, mask=mask).to(tl.float32)
    dequant_scale = tl.load(scale).to(tl.float32)
    tl.store(output + output_offsets, value_block * dequant_scale, mask=mask)


def _dequantize_fp8(
    value: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    *,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not value.is_contiguous():
        raise ValueError("FP8 dequantization requires contiguous input")
    if output is None:
        output = torch.empty_like(value, dtype=dtype)
    elif output.shape != value.shape or output.dtype != dtype:
        raise ValueError("FP8 dequantization output contract mismatch")
    block_size = 1024
    if output.is_contiguous():
        _dequantize_fp8_kernel[(triton.cdiv(value.numel(), block_size),)](
            value,
            output,
            scale,
            num_elements=value.numel(),
            BLOCK_SIZE=block_size,
        )
    else:
        if value.dim() != 2 or output.dim() != 2 or output.stride(1) != 1:
            raise ValueError("strided FP8 dequantization requires row-major matrices")
        _dequantize_fp8_strided_output_kernel[
            (value.shape[0], triton.cdiv(value.shape[1], block_size))
        ](
            value,
            output,
            scale,
            rows=value.shape[0],
            columns=value.shape[1],
            input_row_stride=value.stride(0),
            output_row_stride=output.stride(0),
            BLOCK_SIZE=block_size,
        )
    return output


_TGT_SPARSE_LOGGED = [False]
_GLUE_LAZY_LOGGED = [False]


def _attn_glue_lazy() -> bool:
    """DLRM_HSTU_FP8_ATTN_GLUE_LAZY (X4C19 item (48)(c); read per call, armed
    only by "1", gin default ON).

    The saved-FP8 backward builds a (1, 1) fp32 Q/K/V scale tensor that only
    its Triton attention FALLBACK reads (_dequantize_fp8 when FlyDSL
    declines). While FlyDSL serves the backward -- the default -- that fill
    is dead GPU work. Armed, it is built only where the fallback runs; it
    holds the same frozen value. "0"/absent = build it eagerly.
    The sort_by_length permutation that x4c19's bundle also made lazy is
    owned by rccl's GLUE_FOLD3 LAZY_SORT (ops/glue_fold3.py; kill switch
    DLRM_HSTU_FP8_GLUE_FOLD3_SORT=0), the same code on the same path (ship12
    merge resolution).
    """
    on = os.environ.get("DLRM_HSTU_FP8_ATTN_GLUE_LAZY", "0") == "1"
    if on and not _GLUE_LAZY_LOGGED[0]:
        _GLUE_LAZY_LOGGED[0] = True
        print(
            "[hstu-attn] DLRM_HSTU_FP8_ATTN_GLUE_LAZY=1 armed: the fallback "
            "Q/K/V scale tensor is built only on the Triton fallback path "
            "(the lazy sort_by_length is GLUE_FOLD3_SORT's)",
            flush=True,
        )
    return on


def _adopt_duvqk(
    dsilu_u: Optional[torch.Tensor],
    rows: int,
    width: int,
    u_width: int,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Return the d(uvqk) buffer dsilu_u already lives in, or None.

    With deferred SiLU, the incoming dsilu_u is the finished du. When the
    output-projection backward wrote it as the leading u_width columns of a
    (rows, width) buffer (it tags the view with HSTU_DUVQK_BUFFER_ATTR), that
    buffer is used as d(uvqk) directly, removing the du copy. Any mismatch
    (autograd replaced the gradient, other consumer, other layout) returns None
    and the caller falls back to allocate-and-copy.
    """
    if dsilu_u is None:
        return None
    buf = getattr(dsilu_u, HSTU_DUVQK_BUFFER_ATTR, None)
    if buf is None:
        return None
    if (
        buf.shape != (rows, width)
        or buf.stride() != (width, 1)
        or buf.dtype != dtype
        or dsilu_u.dtype != dtype
        or dsilu_u.shape != (rows, u_width)
        or dsilu_u.stride() != (width, 1)
        or dsilu_u.data_ptr() != buf.data_ptr()
    ):
        return None
    return buf


def _dq_sink_diagnostics_on() -> bool:
    """FlyDSL backward diagnostics that read the BF16 dq after the dq-split
    (HSTU_ATTN_BWD_CHECK, DLRM_HSTU_FP8_BWD_FULL_STATS, DLRM_HSTU_FP8_BWD_DUMP):
    with any of them set the M2 dq producer stays off (it leaves dq unwritten)."""
    return (
        int(os.environ.get("HSTU_ATTN_BWD_CHECK", "0") or 0) > 0
        or int(os.environ.get("DLRM_HSTU_FP8_BWD_FULL_STATS", "0") or 0) > 0
        or bool(os.environ.get("DLRM_HSTU_FP8_BWD_DUMP", ""))
    )


def _dq_sink_layout_ok(ctx, duvqk: torch.Tensor) -> bool:  # pyre-ignore[2]
    """M2 dq producer layout: duvqk is a contiguous [M, u|v|q|k] buffer whose
    q block is whole 256-column blocks of the cast (the u|v|k-only cast skips
    whole blocks)."""
    bn = bwd_fp8._COLSUM_BLOCK_N
    q_lo = 2 * ctx.hidden_dim * ctx.num_heads
    q_w = ctx.attn_dim * ctx.num_heads
    return (
        duvqk.is_contiguous()
        and duvqk.shape[1] % bn == 0
        and q_lo % bn == 0
        and q_w % bn == 0
        and q_lo + q_w <= duvqk.shape[1]
    )


def _log_dq_sink_armed(
    sink, cast: str = "u|v|k only", du_emit: bool = False  # pyre-ignore[2]
) -> None:
    """ARMED line of the M2 dq producer, once per device (and, B1a r16, once
    more when its q block shares dz8 with the B1a u block: the cast that
    actually runs then is the multi-range skip kernel over v|k).

    `cast` (R15 M-E1): the columns the single-range skip cast covers ("u
    only" when the bwd_dkdv sink wrote v|k into the same dz8). With
    `du_emit` and a non-default `cast` (merged B1a + R15 M-E1 call: u, v|k
    and q all producer-cast) no cast kernel runs and `cast` says why."""
    if not du_emit:
        key = "dq_e5m2"
        cast = f"_quantize_amax_colsum_skip_kernel ({cast})"
    elif cast == "u|v|k only":
        key = "dq_e5m2_b1a"
        cast = (
            "_quantize_amax_colsum_skipn_kernel (v|k only; u from the B1a "
            "LNMD bwd)"
        )
    else:
        key = "dq_e5m2_b1a_e1"
        cast = f"no cast kernel ({cast})"
    bwd_fp8.log_armed_v2(
        key,
        sink.device,
        lambda: (
            f"ARMED dq_e5m2 on {sink.device}: _dq8_sched_kernel (dq-split fork, "
            f"e5m2 q columns {sink.q_cols[0]}:{sink.q_cols[1]} + MFMA colsum / "
            f"amax partials, waves_per_eu=4, cfg={getattr(sink, 'config', {})}) + "
            f"{cast}; payload/amax "
            f"bitwise, q colsum fp32-order, M={sink.rows} N={sink.width} "
            f"rows_out={sink.rows_out}"
        ),
    )


# R15 M-E1: DLRM_HSTU_FP8_BWD_DKDV_E5M2 (gin item (47), default off). The
# ordinary layer's bwd_dkdv epilogue writes the e5m2 dV|dK columns (+amax /
# colsum partials) of the uvqk projection's dz8 itself (no BF16 dV/dK stores).
# Arms only when the FP8 uvqk backward will take its token-major fp8t wgrad
# with a published per-layer scale (bwd_fp8.producer_fp8t_inv_scale); every
# other call runs the knob-off path unchanged. Shared-dz8 contract with the
# item (25)(d) dq sink (component A14 / B dz8 contract v2):
#   * ONE dz8 [Mp, 2048] e5m2 per layer call, allocated here BEFORE the
#     attention backward, pad rows [M, Mp) zeroed once; the dq sink adopts
#     it (DqE5m2Sink(dz8=...)) for the q columns.
#   * Scale epoch: ONE immutable (inv, decode-scale) snapshot pair of the
#     layer's scale state per call. The bwd_dkdv emission, the dq sink and
#     the cast of the remaining columns use the inv snapshot; every projection
#     GEMM decodes with the scale snapshot. calls/amax/refresh still advance
#     once per projection call, after the producers, in stream order.
#   * The cast covers u only when the dq sink is engaged (item (25)'s skip
#     cast), else u|q; amax = max of every producer, colsum = [u | v | q | k],
#     ONE prequant projection backward per call.
#   * Exact fallback, never an e5m2 -> BF16 decode: when the FP8 projection
#     backward cannot consume the emitted dz8 (it returns None) or needs BF16
#     dz for a BF16 wgrad, the attention backward is re-run with identical
#     arguments and NO sink (BF16 dq/dk/dv written, bitwise the knob-off
#     values) and the call continues exactly like the knob-off path. Q/K/V
#     and dO of an engaged call therefore stay alive until its projection
#     backward returns.
#   * With B1a (DLRM_HSTU_FP8_BWD_DU_E5M2, gin item (38)(a); merged
#     ship-12 contract M1-M6): when the LN*U*dropout backward emitted this
#     layer's e5m2 u block and the preprocess backward took a consumable
#     record, that record's dz8 IS the layer's dz8 (adopted here, no second
#     allocation; the pad rows of every column are still zeroed once here,
#     re-zeroing the u block's). The scale epoch is still this call's own
#     snapshot pair: check_consumable proved the per-layer scale generation
#     did not move since the emission, so it holds the values B1a cast with.
#     The cast then covers nothing when the dq sink is engaged (no cast
#     kernel), else q only; amax / colsum also take B1a's u maxima and sums.
#     A consumer cast / finalize failure after emission happens before the
#     projection backward runs (scale state untouched) and leaves B1a's u
#     block intact (no other producer writes u columns), so it stays exact:
#     the attention backward is re-run without sinks (exact BF16 dq/dV/dK)
#     and B1a's own consumer finishes the call (hstu_du_e5m2.consume without
#     a dq sink: v|q|k cast into the same dz8 incl. their pad rows, B1a's u
#     amax / colsum, ONE prequant projection call; bitwise the call with
#     this knob and the dq sink off). Only if THAT fails does consume()
#     restore u LOSSY and latch B1a off. The projection backward returning
#     None after emission cannot be exact for u (no BF16 du exists): B1a's
#     documented LOSSY restore of the u columns (latched off), then the
#     attention backward is re-run without sinks (exact BF16 dq/dV/dK) and
#     the BF16-input projection call follows.
#   * Latch (iter-41 review MINOR-3): every fallback after emission (above)
#     latches this knob off for the process (_DKDV_E5M2_BROKEN = its reason,
#     one "DISABLED (latched)" line); later calls take the knob-off path at
#     the arm decision (_dkdv_e5m2_arm -> None, counter "latched") instead
#     of paying an extra attention backward per call for a repeating
#     failure. Never set on a call without a fallback.
#   * Per-process counters (dkdv_e5m2_counters()); one ARMED line per
#     variant, one DECLINED / FALLBACK line per reason, and an engagement
#     line at engaged calls 1, 10, 100, 1000, 10000 (host-side counters and
#     the allocator's peak, no device sync).
_DKDV_E5M2_LOGGED: set = set()
_DKDV_E5M2_COUNTS: Dict[str, int] = {}
_DKDV_E5M2_LOG_AT = (1, 10, 100, 1000, 10000)
_DKDV_E5M2_BROKEN: Optional[str] = None


def _dkdv_e5m2_requested() -> bool:
    return os.environ.get("DLRM_HSTU_FP8_BWD_DKDV_E5M2", "0") == "1"


def dkdv_e5m2_counters() -> Dict[str, int]:
    """This process's DLRM_HSTU_FP8_BWD_DKDV_E5M2 counters: not_planned (knob
    on, no fp8t wgrad this call, e.g. the scale warm-up), armed (sink built),
    engaged (bwd_dkdv emitted dz8), joint_dq (engaged with the item (25)(d)
    dq sink on the same dz8), consumed (the FP8 projection backward ran on
    it), declined:<reason> (BF16 dV/dK, knob-off path), fallback:<reason>
    (exact fallback after emission; "<reason> (B1a record kept)": a consumer
    cast failure finished by B1a's consumer), latched (calls on the
    knob-off path because a fallback latched the knob off); with B1a also
    adopted_b1a (armed on the B1a dz8) and b1a_u (engaged with the B1a u
    block)."""
    return dict(_DKDV_E5M2_COUNTS)


def _dkdv_e5m2_count(key: str) -> int:
    n = _DKDV_E5M2_COUNTS.get(key, 0) + 1
    _DKDV_E5M2_COUNTS[key] = n
    return n


def _dkdv_e5m2_log_once(key: str, message: str) -> None:
    if key in _DKDV_E5M2_LOGGED:
        return
    _DKDV_E5M2_LOGGED.add(key)
    logging.getLogger(__name__).warning(message)


def _dkdv_e5m2_decline(why: str) -> None:
    # Counter key: the reason without its per-call detail.
    key = "declined:" + why.split(" (")[0].split(": ")[0]
    _dkdv_e5m2_count(key)
    _dkdv_e5m2_log_once(
        key,
        f"[hstu-attn] bwd dkdv e5m2 sink DECLINED: {why} -> BF16 dV/dK, "
        "knob-off path",
    )


def _dkdv_e5m2_latch(why: str) -> None:
    """Latch the knob off for the process (first reason wins, one line)."""
    global _DKDV_E5M2_BROKEN
    if _DKDV_E5M2_BROKEN is None:
        _DKDV_E5M2_BROKEN = why
        logging.getLogger(__name__).warning(
            "[hstu-attn] bwd dkdv e5m2 DISABLED (latched): %s -> this call's "
            "fallback, then the knob-off path for the rest of the process",
            why,
        )


def _dkdv_e5m2_fallback(why: str, action: str) -> None:
    key = "fallback:" + why
    _dkdv_e5m2_count(key)
    _dkdv_e5m2_log_once(
        key, f"[hstu-attn] bwd dkdv e5m2 FALLBACK: {why} -> {action}"
    )
    # Every caller is a fallback after emission (never a clean call).
    _dkdv_e5m2_latch(why)


def _dkdv_e5m2_arm(
    duvqk: torch.Tensor,
    uvqk_weight: torch.Tensor,
    normed_x_fp8_t: Optional[torch.Tensor],
    input_scale_value: Optional[float],
    num_heads: int,
    hidden_dim: int,
    attn_dim: int,
    du_emit=None,  # pyre-ignore[2]
):  # pyre-ignore[3]
    """A DkdvE5m2Sink over a fresh dz8 [Mp, 4*H*D] (pad rows zeroed) bound to
    this call's scale epoch, or None (declined). Mp = roundup(M,
    uvqk_tm_pad()), the rows the full cast would allocate for the
    token-major wgrad. With `du_emit` (a taken, consumable B1a record) the
    dz8 is du_emit.dz8 instead (M1: the first producer in backward order
    allocated it; adopted, its pad rows zeroed here once all the same).
    None without touching anything once a fallback latched the knob off."""
    if _DKDV_E5M2_BROKEN is not None:
        _dkdv_e5m2_count("latched")
        return None
    rows, width = duvqk.shape
    hd = num_heads * hidden_dim
    if (
        attn_dim != hidden_dim
        or width != 4 * hd
        or hd % bwd_fp8._COLSUM_BLOCK_N != 0
        or uvqk_weight.dim() != 2
        or uvqk_weight.shape[1] != width
        or not duvqk.is_contiguous()
        or duvqk.dtype != torch.bfloat16
    ):
        _dkdv_e5m2_decline(
            f"layout (duvqk {tuple(duvqk.shape)} {duvqk.dtype}, H={num_heads} "
            f"Dv={hidden_dim} Dqk={attn_dim})"
        )
        return None
    if _dq_sink_diagnostics_on():
        # HSTU_ATTN_BWD_CHECK / FULL_STATS / DUMP read the BF16 dV/dK.
        _dkdv_e5m2_decline("diagnostics (FlyDSL backward check/stats/dump on)")
        return None
    epoch = bwd_fp8.producer_fp8t_inv_scale(
        "uvqk_projection", uvqk_weight, rows, normed_x_fp8_t, input_scale_value
    )
    if epoch is None:
        _dkdv_e5m2_decline(
            "scale (the FP8 projection backward would not take the token-major "
            "fp8t wgrad with a published per-layer scale; "
            "bwd_fp8.producer_fp8t_inv_scale)"
        )
        return None
    from generative_recommenders.ops.flydsl.hstu_dkdv_e5m2 import DkdvE5m2Sink

    grad_dtype, _ = bwd_fp8.grad_dtype_and_limit()
    pad = bwd_fp8.uvqk_tm_pad()
    rows_out = (rows + pad - 1) // pad * pad
    if du_emit is None:
        dz8 = torch.empty((rows_out, width), dtype=grad_dtype, device=duvqk.device)
    else:
        # Merged ship-12 contract M1: B1a's LN*U*dropout backward (the first
        # producer in backward order) allocated this layer's dz8 and wrote its
        # u block; v|k and q go into the same buffer. check_consumable already
        # matched the record to this call (rows, width, u block, site,
        # generation); the checks below only guard the adoption itself.
        dz8 = du_emit.dz8
        if (
            dz8 is None
            or dz8.dim() != 2
            or tuple(dz8.shape) != (rows_out, width)
            or dz8.dtype != grad_dtype
            or not dz8.is_contiguous()
            or dz8.device != duvqk.device
            or du_emit.u_cols != (0, hd)
            or du_emit.rows != rows
        ):
            _dkdv_e5m2_decline(
                f"b1a dz8 ({None if dz8 is None else tuple(dz8.shape)} "
                f"u_cols={du_emit.u_cols} rows={du_emit.rows}; needs "
                f"[{rows_out}, {width}] u_cols=(0, {hd}) rows={rows})"
            )
            return None
        _dkdv_e5m2_count("adopted_b1a")
    if rows_out > rows:
        # The only pad-row fill of this dz8: no producer writes rows >= M and
        # the casts rewrite their own columns' pad rows with the same zeros
        # (an adopted B1a dz8: its u pad rows are re-zeroed, same bytes).
        dz8[rows:].zero_()
    _dkdv_e5m2_count("armed")
    return DkdvE5m2Sink(dz8, epoch[0], decode_scale=epoch[1])


def _dkdv_e5m2_engaged(sink, served: bool):  # pyre-ignore[2,3]
    """`sink` when the bwd_dkdv epilogue emitted into it, else None: the stock
    kernel (or Triton) wrote BF16 dV/dK and the knob-off path runs (a dq sink
    that adopted the dz8 keeps it; its pad rows are zero)."""
    if served and sink.declined is None and sink.filled:
        _dkdv_e5m2_count("engaged")
        return sink
    _dkdv_e5m2_decline(sink.declined or "attention backward not served")
    return None


def _dkdv_e5m2_prequant(
    duvqk: torch.Tensor, sink, hd: int, dq_sink=None,  # pyre-ignore[2]
    du_emit=None,  # pyre-ignore[2]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Finish an engaged sink (and an engaged dq sink on the same dz8) with
    the epoch's inv snapshot -> the `prequant` tuple of
    bwd_fp8.uvqk_projection_backward.

    With the dq sink: the u columns only, by item (25)'s skip cast (the
    COLSUM_V2 tile, partial layout [grid_m, N]) and the q partials by its
    finish_q(); without it the u|q keep-list cast. Either way the cast
    columns' bytes, amax partials and column sums are those of the full
    cast. amax = max(cast, q, dV|dK); colsum = [u | v | q | k].

    Merged ship-12 contract M4: with `du_emit` (B1a's record on the same
    dz8) the u block is producer-cast too, so the cast covers only the
    complement {q if no engaged dq sink}: with the dq sink nothing is cast
    (no cast kernel), else q by the multi-range skip cast (the same tile,
    partial layout and column reduction as the full cast). amax also takes
    B1a's u amax and the u colsum is B1a's."""
    dz8 = sink.dz8
    rows = duvqk.shape[0]
    if du_emit is not None:
        assert du_emit.dz8 is dz8 and du_emit.rows == rows
        assert du_emit.u_cols == (0, hd)
        if dq_sink is not None:
            # u (B1a), v|k (this sink) and q (dq sink) all producer-cast.
            q_amax, q_col = dq_sink.finish_q()
            amax = torch.maximum(du_emit.u_amax, q_amax)
            colsum = torch.empty(
                (4 * hd,), dtype=torch.float32, device=duvqk.device
            )
            colsum[2 * hd : 3 * hd] = q_col
        else:
            from generative_recommenders.ops.triton.hstu_dq_e5m2 import (
                quantize_amax_colsum_skip,
            )

            part_amax, part_col = quantize_amax_colsum_skip(
                duvqk, sink.inv_scale, dz8, [(0, 2 * hd), (3 * hd, 4 * hd)]
            )
            amax = torch.maximum(part_amax.amax(), du_emit.u_amax)
            colsum = part_col.sum(dim=0)
        colsum[0:hd] = du_emit.u_colsum
    elif dq_sink is not None:
        from generative_recommenders.ops.triton.hstu_dq_e5m2 import (
            quantize_amax_colsum_skip,
        )

        part_amax, part_col = quantize_amax_colsum_skip(
            duvqk, sink.inv_scale, dz8, (hd, 4 * hd)
        )
        q_amax, q_col = dq_sink.finish_q()
        amax = torch.maximum(part_amax.amax(), q_amax)
        colsum = part_col.sum(dim=0)
        colsum[2 * hd : 3 * hd] = q_col
    else:
        amax, colsum = bwd_fp8.quantize_amax_colsum_select(
            duvqk, sink.inv_scale, dz8, group_cols=hd
        )
    amax_vk, colsum_vk = sink.finish()
    amax = torch.maximum(amax, amax_vk)
    colsum[hd : 2 * hd] = colsum_vk[:hd]
    colsum[3 * hd : 4 * hd] = colsum_vk[hd:]
    return dz8, amax, colsum, rows


def _dkdv_e5m2_projection_backward(
    sink,  # pyre-ignore[2]
    duvqk: torch.Tensor,
    hd: int,
    rerun_attention_backward: Callable[..., bool],
    dq_sink,  # pyre-ignore[2]
    make_dq_sink: Optional[Callable[[torch.Tensor], object]],
    normed_x: Optional[torch.Tensor],
    input_scale_value: Optional[float],
    weight: torch.Tensor,
    normed_x_fp8_t: Optional[torch.Tensor],
    recompute_normed_x: Callable[[], torch.Tensor],
    du_emit=None,  # pyre-ignore[2]
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """`bwd_fp8.uvqk_projection_backward` of an engaged call: the FP8
    projection backward on the producers' dz8 (bwd_dkdv v|k, dq sink q when
    engaged, cast u or u|q), decoded with the epoch's scale snapshot.

    Every fallback continues EXACTLY like this call with the knob off (the
    twin: E1 off, the dq sink as engaged here), and never decodes e5m2 where
    the twin reads BF16:
      * BF16 wgrad inside the FP8 projection backward (token-major wgrad
        failed or off): joint call -> the twin decodes the byte-identical
        dz8, so this call does too (no provider); dq sink not engaged -> the
        twin reads its exact BF16 dz, so the attention backward is re-run
        without the sink and the wgrad reads that exact dz;
      * failure after emission (the u/u|q cast or a finalize raised, or the
        projection backward returned None): the attention backward is re-run
        with ONLY this sink removed -- a fresh dq sink on the same dz8 and
        scale epoch when one was engaged -- and the twin's own projection
        call follows (item (25) M2 call, else the BF16-input call). Where the
        twin would then decode its e5m2 q columns (M2 restore_bf16), the
        attention backward is re-run without any sink instead (exact BF16 dq,
        same scale state) and the caller's BF16 path follows.

    With `du_emit` (merged ship-12 contract M4-M6: B1a's u block in the same
    dz8, record taken and consumable) the cast shrinks to the complement
    (nothing with the dq sink, else q) and a BF16 wgrad inside the FP8
    projection backward decodes the payload like the B1a twin (consume()).
    A consumer cast / finalize failure (raised before the projection call:
    scale state untouched, B1a's u block intact) re-runs the attention
    backward without any sink (exact BF16 dq/dV/dK) and finishes with B1a's
    own consumer, hstu_du_e5m2.consume(du_emit, None): v|q|k cast into the
    same dz8 + B1a's u amax / colsum, ONE prequant projection call (the B1a
    twin without the dq sink, bitwise); only if that fails does consume()
    restore u LOSSY and latch B1a off. The projection backward returning
    None after emission restores u LOSSY (hstu_du_e5m2.decline: B1a's
    documented class, latched off; no BF16 du exists), re-runs the attention
    backward without any sink (exact BF16 dq/dV/dK) and runs the BF16-input
    projection call.

    Every fallback after emission latches this knob off for the process
    (_dkdv_e5m2_fallback -> _dkdv_e5m2_latch).
    """
    rows = duvqk.shape[0]
    joint = dq_sink is not None
    if joint:
        _dkdv_e5m2_count("joint_dq")
    if du_emit is not None:
        _dkdv_e5m2_count("b1a_u")
        mode, cast = (
            ("b1a_joint", "cast none: u from the LN*U*dropout backward (B1a), "
             "q from the dq-split sink")
            if joint
            else ("b1a_solo", "cast q only: u from the LN*U*dropout backward "
                  "(B1a)")
        )
        _dkdv_e5m2_log_once(
            "armed:" + mode,
            "[hstu-attn] bwd dkdv e5m2 sink ARMED: mode=only (bwd_dkdv epilogue "
            "writes e5m2 dV|dK + amax/colsum partials into the shared dz8 "
            f"[{sink.dz8.shape[0]}, {sink.dz8.shape[1]}] adopted from the "
            "LN*U*dropout backward (B1a e5m2 u block), no BF16 dV/dK; "
            + ("dq-split sink adopts it (e5m2 q, no BF16 dq); " if joint else "")
            + cast
            + "; per-layer uvqk_projection scale epoch; a failure after "
            "emission restores u LOSSY (B1a latched off) and re-runs the "
            f"attention backward without sinks) M={rows} "
            "(DLRM_HSTU_FP8_BWD_DKDV_E5M2 + DLRM_HSTU_FP8_BWD_DU_E5M2)",
        )
    else:
        _dkdv_e5m2_log_once(
            "armed:" + ("joint" if joint else "solo"),
            "[hstu-attn] bwd dkdv e5m2 sink ARMED: mode=only (bwd_dkdv epilogue "
            "writes e5m2 dV|dK + amax/colsum partials into the shared dz8 "
            f"[{sink.dz8.shape[0]}, {sink.dz8.shape[1]}], no BF16 dV/dK; "
            + (
                "dq-split sink adopts it (e5m2 q, no BF16 dq); cast u only"
                if joint
                else "cast u|q only"
            )
            + "; per-layer uvqk_projection scale epoch; fallbacks follow the "
            f"knob-off call exactly) M={rows} (DLRM_HSTU_FP8_BWD_DKDV_E5M2)",
        )
    exact: list = []

    def exact_dz(why: str) -> torch.Tensor:
        # The attention backward without any sink: exact BF16 dq/dV/dK.
        _dkdv_e5m2_fallback(
            why, "attention backward re-run without the sinks (exact BF16 dq/dV/dK)"
        )
        if not exact:
            if not rerun_attention_backward():
                raise RuntimeError(
                    "[hstu-attn] bwd dkdv e5m2: the attention backward re-run "
                    "without the sinks was not served by FlyDSL; the BF16 "
                    "dV/dK of this call do not exist"
                )
            exact.append(True)
        return duvqk

    dz8 = sink.dz8
    why = None
    cast_failed = False
    try:
        prequant = _dkdv_e5m2_prequant(duvqk, sink, hd, dq_sink, du_emit)
    except Exception as error:  # first-call failure of a consumer kernel
        prequant = None
        cast_failed = True
        why = f"consumer {type(error).__name__}: {str(error)[:120]}"
    fp8_grads = None
    if prequant is not None:
        fp8_grads = bwd_fp8.uvqk_projection_backward(
            dz=None,
            normed_x=normed_x,
            input_scale_value=input_scale_value,
            weight=weight,
            normed_x_fp8_t=normed_x_fp8_t,
            recompute_normed_x=recompute_normed_x,
            prequant=prequant,
            producer_desc=(
                (
                    "bwd_dkdv e5m2 dV|dK + "
                    + ("dq-split e5m2 q + u-only cast" if joint else "u|q cast")
                    + ", shared dz8, no BF16 dV/dK" + (" or dq" if joint else "")
                )
                if du_emit is None
                else (
                    "LNMD bwd e5m2 u + bwd_dkdv e5m2 dV|dK + dq-split e5m2 q, "
                    "shared dz8, no cast, no BF16 du/dV/dK/dq"
                    if joint
                    else "LNMD bwd e5m2 u + bwd_dkdv e5m2 dV|dK + q-only cast, "
                    "shared dz8, no BF16 du/dV/dK"
                )
            ),
            prequant_scale=sink.decode_scale,
            **(
                # B1a merged (M5): no BF16 du exists, so a BF16 wgrad inside
                # decodes the payload like the B1a twin (consume()).
                {}
                if joint or du_emit is not None
                else {
                    "prequant_bf16_dz": lambda: exact_dz(
                        "BF16 wgrad inside the FP8 projection backward"
                    )
                }
            ),
        )
        prequant = None
        if joint and du_emit is not None:
            _log_dq_sink_armed(
                dq_sink,
                "u from the B1a LNMD bwd, v|k from the R15 M-E1 dV|dK sink",
                du_emit=True,
            )
        elif joint:
            _log_dq_sink_armed(
                dq_sink, "u only: shared dz8 with the R15 M-E1 dV|dK sink"
            )
        if fp8_grads is None:
            why = "FP8 projection backward returned None after emission"
    sink.release()
    if fp8_grads is not None:
        _dkdv_e5m2_count("consumed")
        if du_emit is not None:
            # B1a's ARMED line for the merged call (its consume() is not
            # called here), then drop the record's payload reference.
            lo, hi = du_emit.u_cols
            rows_out, device = du_emit.rows_out, du_emit.device
            bwd_fp8.log_armed_v2(
                "du_e5m2_e1" if joint else "du_e5m2_e1_u",
                device,
                lambda: (
                    f"ARMED du_e5m2 on {device}: _ln_mul_dropout_bwd_dx_du_rng_v3 "
                    f"(diet, e5m2 u cols {lo}:{hi} + amax/colsum partials, no "
                    "bf16 du) -> dz8 shared with the R15 M-E1 dV|dK sink"
                    + (" and the dq-split sink, no cast" if joint else ", q-only cast")
                    + f"; payload/amax bitwise, u colsum fp32-order, M={rows} "
                    f"Mp={rows_out}"
                ),
            )
            du_emit.release()
    elif du_emit is not None and cast_failed:
        # Merged ship-12 contract M5, cast-failure sub-case (iter-41 review
        # MINOR-1): the consumer cast / finalize raised BEFORE the projection
        # backward ran (delayed-scale state untouched) and B1a's e5m2 u block
        # in dz8 is intact (E1 and the dq sink never write u columns), so no
        # LOSSY restore: drop the dq sink's dz8 / partials, re-run the
        # attention backward without sinks (exact BF16 dq/dV/dK in duvqk) and
        # let B1a's own consumer finish the call (finish_u_only: v|q|k cast
        # from the exact BF16 dz into the same dz8 incl. their pad rows, B1a's
        # u amax / colsum merged, ONE prequant projection call -- the call
        # with this knob and the dq sink off, bitwise). Only if THAT fails
        # does consume() restore u LOSSY and latch B1a (its documented class);
        # the caller then runs BF16. consume() without a dq sink runs neither
        # the E1 nor the dq-sink finalizer, so a failure of either one does
        # not repeat inside it.
        if joint:
            dq_sink.dz8 = None
            dq_sink.partials = None
        dz8 = None
        why = f"{why} (B1a record kept)"
        _dkdv_e5m2_fallback(
            why,
            "attention backward re-run without the sinks (exact BF16 "
            "dq/dV/dK), then the B1a consumer (hstu_du_e5m2.consume: v|q|k "
            "cast into its dz8 + its u amax/colsum, one FP8 projection call; "
            "LOSSY u and B1a latched off only if that fails)",
        )
        if not rerun_attention_backward():
            raise RuntimeError(
                "[hstu-attn] bwd dkdv e5m2: the attention backward re-run "
                "without the sinks was not served by FlyDSL; the BF16 "
                "dV/dK of this call do not exist"
            )
        exact.append(True)
        fp8_grads = hstu_du_e5m2.consume(
            du_emit,
            None,
            duvqk,
            normed_x=normed_x,
            input_scale_value=input_scale_value,
            weight=weight,
            normed_x_fp8_t=normed_x_fp8_t,
            recompute_normed_x=recompute_normed_x,
        )
    elif du_emit is not None:
        # Merged ship-12 contract M5, projection-None sub-case: the FP8
        # projection backward returned None after the emission, and u has no
        # BF16 source (B1a wrote only e5m2), so its documented LOSSY restore
        # (latched off) first; then the attention backward without any sink
        # (exact BF16 dq/dV/dK) and the BF16-input projection call, as the E1
        # fallback without a dq sink. The dq sink's dz8 / partials are
        # dropped first.
        if joint:
            dq_sink.dz8 = None
            dq_sink.partials = None
        dz8 = None
        lo, hi = du_emit.u_cols
        hstu_du_e5m2.decline(
            du_emit,
            duvqk[:, lo:hi],
            "declined",
            f"R15 M-E1 merged call: {why}",
        )
        exact_dz(why)
        fp8_grads = bwd_fp8.uvqk_projection_backward(
            dz=duvqk,
            normed_x=normed_x,
            input_scale_value=input_scale_value,
            weight=weight,
            normed_x_fp8_t=normed_x_fp8_t,
            recompute_normed_x=recompute_normed_x,
        )
    elif joint:
        # The twin's call: re-run with ONLY the dV/dK sink removed (BF16
        # dV/dK written; a fresh dq sink writes the same q bytes and
        # partials into the same dz8), then its M2 projection call.
        dq_sink.dz8 = None
        dq_sink.partials = None
        _dkdv_e5m2_fallback(
            why,
            "attention backward re-run with only the dV/dK sink removed (fresh "
            "dq sink, same dz8 and scale epoch), then the item (25) M2 call",
        )
        # dq2 holds the only dz8 reference from here (dropped after its call,
        # before any no-sink re-run).
        dq2 = make_dq_sink(dz8=dz8) if make_dq_sink is not None else None
        dz8 = None
        if dq2 is None or not rerun_attention_backward(dq_fp8_sink=dq2):
            raise RuntimeError(
                "[hstu-attn] bwd dkdv e5m2: the attention backward re-run "
                "with the dq sink was not served by FlyDSL"
            )
        if dq2.filled:
            fp8_grads = bwd_fp8.uvqk_projection_backward(
                dz=None,
                normed_x=normed_x,
                input_scale_value=input_scale_value,
                weight=weight,
                normed_x_fp8_t=normed_x_fp8_t,
                recompute_normed_x=recompute_normed_x,
                prequant=dq2.finish(duvqk),
                producer_desc=(
                    "dq-split e5m2 q columns + u|v|k-only cast, no BF16 dq"
                ),
            )
            _log_dq_sink_armed(dq2)
            dq2.dz8 = None
            if fp8_grads is None:
                # The twin decodes its e5m2 q here (restore_bf16): exact BF16
                # dq instead, same scale state; the caller's BF16 path.
                exact_dz("item (25) projection backward returned None")
        else:
            # The dq sink declined the re-run: BF16 dq written, the twin's
            # BF16-input call.
            fp8_grads = bwd_fp8.uvqk_projection_backward(
                dz=duvqk,
                normed_x=normed_x,
                input_scale_value=input_scale_value,
                weight=weight,
                normed_x_fp8_t=normed_x_fp8_t,
                recompute_normed_x=recompute_normed_x,
            )
        dz8 = None
        dq2 = None
    else:
        # No dq sink engaged: the twin is the BF16-input call on exact dz.
        dz8 = None
        exact_dz(why)
        fp8_grads = bwd_fp8.uvqk_projection_backward(
            dz=duvqk,
            normed_x=normed_x,
            input_scale_value=input_scale_value,
            weight=weight,
            normed_x_fp8_t=normed_x_fp8_t,
            recompute_normed_x=recompute_normed_x,
        )
    engaged = _DKDV_E5M2_COUNTS.get("engaged", 0)
    if engaged in _DKDV_E5M2_LOG_AT:
        logging.getLogger(__name__).warning(
            "[hstu-attn] bwd dkdv e5m2 sink engaged=%d counters=%s "
            "max_allocated=%.2f GiB",
            engaged,
            ",".join(f"{k}={v}" for k, v in sorted(_DKDV_E5M2_COUNTS.items())),
            torch.cuda.max_memory_allocated(duvqk.device) / 2**30,
        )
    return fp8_grads


@functools.lru_cache(maxsize=1)
def _flydsl_bwd() -> Callable[..., bool]:
    """The FlyDSL in-place attention backward.

    This is the backward the production TRITON path actually runs (the non-fused
    ops/hstu_attention.py dispatch is a different call site), so $HSTU_ATTN_BWD
    has to be honoured here too. Only the *import* is cached: the callee re-reads
    the selection per call, which is what lets a test flip backends in-process.

    The import is NOT guarded. hstu_flydsl_mha pulls in only torch and triton --
    the flydsl library itself loads lazily, behind a hardware check that raises
    with an actionable message -- so a failure here means a broken tree, not an
    unsupported host, and swallowing it would silently cost the default backward.
    """
    from generative_recommenders.ops.flydsl.hstu_flydsl_mha import (
        try_flydsl_attention_bwd,
    )

    return try_flydsl_attention_bwd


@functools.lru_cache(maxsize=1)
def _flydsl_do8_planned() -> Callable[..., bool]:
    """R15 A11: hstu_flydsl_mha.c8_do_blk_planned, the forward's dO8 site gate
    (only the import is cached, as in _flydsl_bwd)."""
    from generative_recommenders.ops.flydsl.hstu_flydsl_mha import (
        c8_do_blk_planned,
    )

    return c8_do_blk_planned


@functools.lru_cache(maxsize=1)
def _flydsl_fwd() -> Callable[..., Optional[torch.Tensor]]:
    """The FlyDSL attention forward, selected by $HSTU_ATTN_FWD.

    Separate from _flydsl_bwd because the two halves are independently
    selectable: HSTU has no softmax, so the forward hands the backward nothing,
    and running one on FlyDSL with the other on Triton is both valid and the only
    way to A/B a single half. Returns None per call when not selected or when the
    shape is not specialized, and the caller falls back to Triton.
    """
    from generative_recommenders.ops.flydsl.hstu_flydsl_mha import (
        try_flydsl_attention_fwd,
    )

    return try_flydsl_attention_fwd


@functools.lru_cache(maxsize=1)
def _flydsl_fwd_fp8() -> Callable[..., Optional[torch.Tensor]]:
    """The FlyDSL e4m3 attention forward.

    Selected by DLRM_HSTU_FP8_FWD=1 plus HSTU_ATTN_FWD=flydsl (deprecated
    alias: DLRM_HSTU_FP8_FWD_FLYDSL=1). Separate from _flydsl_fwd because it
    is a different kernel specialization: e4m3 q/k/v plus the fused vqk GEMM's
    per-tensor scale, vs bf16 and no scale. Returns None only when the knob
    is off -- once on it is fail-closed and raises rather than falling back.
    """
    from generative_recommenders.ops.flydsl.hstu_flydsl_mha import (
        try_flydsl_attention_fwd_fp8,
    )

    return try_flydsl_attention_fwd_fp8


# DLRM_HSTU_FP8_FWD_MXFP6 (yambda_5b.gin item (92), DECLARED PRECISION CHANGE,
# default off; read once at import): the training forward's LN store_t + U +
# V/Q/K (ordinary layers) and LN store_t + V/K (targets-only layer) on the r14
# MX6 LN producer and pr7's MXFP6 K=512 GEMMs (ops/hip/mx6_fwd.py). Every arm
# below is an `if _MX6_FWD and ...:` whose `else` is the shipped statement,
# verbatim, so with the knob off the adapter is never imported and the shipped
# code runs unchanged. Defined below the module's @triton.jit kernels on
# purpose: a kernel's starting line enters Triton's cache key.
_MX6_FWD: bool = os.environ.get("DLRM_HSTU_FP8_FWD_MXFP6", "0") == "1"
if _MX6_FWD:
    from generative_recommenders.ops.hip import mx6_fwd as _mx6_fwd


def _dy_erm():  # pyre-ignore[3]
    """r14 E-RM (DLRM_HSTU_FP8_BWD_ERM, yambda_5b.gin item (97), default off;
    ops/triton/dy_erm.py). Defined below the module's @triton.jit kernels on
    purpose (their starting lines enter Triton's cache key). The ordinary
    Function's wrapper claims the bottom layer's site off its input x (claim
    returns None at its first test with the knob off) and its backward asks
    run_producer only when its ctx holds a claimed site, so with the knob off
    the shipped LN-backward call below runs exactly as before."""
    from generative_recommenders.ops.triton import dy_erm  # noqa: PLC0415

    return dy_erm


class _HSTUPreprocessAndAttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore [14]
    def forward(
        ctx,  # pyre-ignore [2]
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        norm_eps: float,
        num_heads: int,
        attn_dim: int,
        hidden_dim: int,
        uvqk_weight: torch.Tensor,
        uvqk_bias: torch.Tensor,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        attn_alpha: float,
        num_targets: Optional[torch.Tensor],
        max_attn_len: int,
        contextual_seq_len: int,
        recompute_uvqk_in_backward: bool,
        recompute_normed_x_in_backward: bool,
        sort_by_length: bool,
        enable_tma: bool,
        fp8_input_scale: Optional[torch.Tensor] = None,
        fp8_qkv_scale: Optional[torch.Tensor] = None,
        defer_silu: bool = False,
        allow_save_fp8_uvqk: bool = True,
        num_softmax_heads: int = 0,
        dropout_handoff: Optional[DropoutHandoff] = None,
        erm_site: Optional[object] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert num_softmax_heads == 0, "Softmax attention is not supported"
        refuse_fp8_with_flydsl_attention_forward()
        fp8_scale_value = activation_scale_value(fp8_input_scale)
        fp8_scale = (
            fp8_scale_tensor(fp8_input_scale)
            if fp8_input_scale is not None
            else None
        )
        qkv_scale_value = activation_scale_value(fp8_qkv_scale)
        fp8_requested = hstu_fp8_fuse_ln_enabled() and fp8_input_scale is not None
        use_fp8 = (
            fp8_requested
            and recompute_normed_x_in_backward
            and fp8_supported(x, uvqk_weight)
        )
        if fp8_requested and not use_fp8 and hstu_fp8_fail_closed():
            raise RuntimeError(
                "Plan 08 input FP8 path requested for an unsupported contract"
            )
        fp8_qkv_feature_requested = (
            use_fp8
            and hstu_fp8_fuse_qkv_enabled()
            and hstu_fp8_attention_enabled()
        )
        fp8_qkv_requested = (
            fp8_qkv_feature_requested and recompute_uvqk_in_backward
        )
        if (
            fp8_qkv_feature_requested
            and not fp8_qkv_requested
            and hstu_fp8_fail_closed()
        ):
            raise RuntimeError(
                "Plan 08 QKV FP8 requires BF16 UVQK recompute in backward"
            )
        use_fp8_qkv = fp8_qkv_requested and fp8_qkv_scale is not None
        if fp8_qkv_requested and not use_fp8_qkv and hstu_fp8_fail_closed():
            raise RuntimeError("Plan 08 QKV FP8 scale state is missing")
        fp8_qkv_forward = False
        # The projection GEMMs' tuned BLAS solutions are keyed on the token
        # count, which changes every step, so they have to be registered per
        # batch. x.shape[0] is that count -- the LN below is row-wise, so the
        # projection sees the same M -- and registering here covers the
        # out-projection and the backward GEMMs too, since all the pins go in
        # together. It has to precede the branch rather than sit after the
        # projection, or the forward pins would miss on every new count. Under
        # fx tracing the shape is a Proxy, hence the guard.
        num_tokens = x.shape[0]
        if isinstance(num_tokens, int):
            install_for_tokens(num_tokens)
        saved_normed_x_bf16 = None
        stored_x_fp8_t = None
        fp8t_planned = False  # B1a (r16): also read at the d(uvqk) tag below
        if use_fp8 and fp8_scale_value is not None:
            assert fp8_scale is not None
            # Only worth storing when backward would otherwise take the plain
            # BF16 LayerNorm recompute, i.e. when the e4m3 V/Q/K pack is
            # saved (the no-save path recomputes BF16+E4M3 in one dual pass).
            # needs_input_grad[0] (x) is False for eval/no_grad forwards,
            # which must not pay the extra BF16 write.
            save_bf16_normed_x = (
                _SAVE_BF16_NORMED_X
                and bool(ctx.needs_input_grad[0])
                and recompute_normed_x_in_backward
                and recompute_uvqk_in_backward
                and _SAVE_FP8_UVQK
                and allow_save_fp8_uvqk
                and use_fp8_qkv
                and qkv_scale_value is not None
            )
            # FP8 uvqk wgrad (bwd_fp8 DLRM_HSTU_FP8_BWD_UVQK_WGRAD=fp8t,
            # default off): when backward will take it for this call (mirrors
            # backward's fp8t_wgrad predicate), the projection wgrad reads a
            # token-major E4M3 normed_x instead of the BF16 one, so the BF16
            # store above would be dead weight and is skipped; with STORE_T the
            # token-major copy is written by this same LN pass, otherwise
            # backward recomputes it. A misprediction is safe: backward then
            # recomputes the BF16 normed_x (or drops the stored copy).
            fp8t_planned = (
                bwd_fp8.uvqk_fp8t_requested()
                and bool(ctx.needs_input_grad[0])
                and uvqk_bias.dim() == 1
                and (
                    not recompute_uvqk_in_backward
                    or (
                        _SAVE_FP8_UVQK
                        and allow_save_fp8_uvqk
                        and use_fp8_qkv
                        and qkv_scale_value is not None
                    )
                )
                and bwd_fp8.uvqk_fp8t_will_run(
                    "uvqk_projection",
                    uvqk_weight,
                    rows=x.shape[0],
                    dtype=x.dtype,
                    input_scale_value=fp8_scale_value,
                )
            )
            if fp8t_planned and save_bf16_normed_x:
                save_bf16_normed_x = False
                bwd_fp8.log_uvqk_once(
                    "skip_bf16_save:uvqk_projection",
                    "uvqk_projection: fp8t wgrad planned, forward skips the "
                    "BF16 normed_x save (DLRM_HSTU_FP8_SAVE_NORMED_X); the "
                    "token-major E4M3 normed_x replaces it",
                )
            if _MX6_FWD and (
                mx6 := _mx6_fwd.uvqk_ln_store(
                    x=x,
                    norm_weight=norm_weight,
                    norm_bias=norm_bias,
                    eps=norm_eps,
                    fp8_scale_value=fp8_scale_value,
                    uvqk_weight=uvqk_weight,
                    uvqk_bias=uvqk_bias,
                    u_columns=hidden_dim * num_heads,
                    qkv_frozen=use_fp8_qkv and qkv_scale_value is not None,
                    grad=bool(ctx.needs_input_grad[0]),
                    store_t=fp8t_planned and bwd_fp8.uvqk_store_fp8t(),
                    pack_saved=(
                        recompute_uvqk_in_backward
                        and _SAVE_FP8_UVQK
                        and allow_save_fp8_uvqk
                    ),
                    pad_to=bwd_fp8.uvqk_tm_pad(),
                    tiled_xt=wgrad_ck.tiled_active(x.device),
                )
            ) is not None:
                # MX6 LN producer in place of the store_t: the same mean /
                # rstd / token-major xT8, plus x6 / xs for the MX6 GEMMs and
                # the e4m3 x8 (None once U runs MX6) for the FP8 U.
                normed_x = mx6.x8
                x_mean = mx6.mean
                x_rstd = mx6.rstd
                stored_x_fp8_t = mx6.xt8
            elif fp8t_planned and bwd_fp8.uvqk_store_fp8t():
                (
                    normed_x,
                    x_mean,
                    x_rstd,
                    stored_x_fp8_t,
                ) = triton_weighted_layer_norm_fwd_fp8_store_t(
                    x=x,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=norm_eps,
                    inv_scale=1.0 / fp8_scale_value,
                    pad_to=bwd_fp8.uvqk_tm_pad(),
                    tiled_xt=wgrad_ck.tiled_active(x.device),
                )
            elif save_bf16_normed_x:
                (
                    saved_normed_x_bf16,
                    normed_x,
                    x_mean,
                    x_rstd,
                ) = triton_weighted_layer_norm_fp8_save_bf16_fwd(
                    x=x,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=norm_eps,
                    inv_scale=1.0 / fp8_scale_value,
                )
            else:
                normed_x, x_mean, x_rstd = triton_weighted_layer_norm_fwd(
                    x=x,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=norm_eps,
                    output_fp8=True,
                    inv_scale=1.0 / fp8_scale_value,
                )
            if use_fp8_qkv and qkv_scale_value is not None:
                assert fp8_qkv_scale is not None
                u_columns = hidden_dim * num_heads
                if _MX6_FWD and mx6 is not None and mx6.u_mx6:
                    # MXFP6 U (pr7 N=512 kernel, bf16 + bias epilogue).
                    u = _mx6_fwd.gemm_u(
                        mx6,
                        uvqk_weight=uvqk_weight,
                        uvqk_bias=uvqk_bias,
                        u_columns=u_columns,
                        fp8_scale=fp8_scale,
                        out_dtype=x.dtype,
                    )
                else:
                    u = scaled_addmm_fp8_preq(
                        activation_fp8=normed_x,
                        scale_a=fp8_scale,
                        weight=uvqk_weight,
                        bias_or_residual=uvqk_bias[:u_columns],
                        out_dtype=x.dtype,
                        weight_layout=((0, u_columns),),
                        data_parallel=True,
                    ).contiguous()
                if _MX6_FWD and mx6 is not None:
                    # MXFP6 V/Q/K (pr7 N=1536 kernel), e4m3 at the qkv scale.
                    vqk = _mx6_fwd.gemm_vqk(
                        mx6,
                        ctx=ctx,
                        uvqk_weight=uvqk_weight,
                        uvqk_bias=uvqk_bias,
                        u_columns=u_columns,
                        fp8_scale=fp8_scale,
                        fp8_scale_value=fp8_scale_value,
                        fp8_qkv_scale=fp8_qkv_scale,
                        qkv_scale_value=qkv_scale_value,
                    )
                else:
                    vqk = scaled_mm_fp8_out_preq(
                        activation_fp8=normed_x,
                        scale_a=fp8_scale,
                        weight=uvqk_weight,
                        bias=uvqk_bias[u_columns:],
                        weight_layout=((u_columns, uvqk_weight.shape[1]),),
                        output_scale=fp8_qkv_scale,
                        scale_a_value=fp8_scale_value,
                        output_scale_value=qkv_scale_value,
                    ).contiguous()
                v, q, k = vqk.split(
                    [
                        hidden_dim * num_heads,
                        attn_dim * num_heads,
                        attn_dim * num_heads,
                    ],
                    dim=1,
                )
                uvqk = torch.empty(0, dtype=x.dtype, device=x.device)
                fp8_qkv_forward = True
            else:
                uvqk = scaled_addmm_fp8_preq(
                    activation_fp8=normed_x,
                    scale_a=fp8_scale,
                    weight=uvqk_weight,
                    bias_or_residual=uvqk_bias,
                    out_dtype=x.dtype,
                ).contiguous()
                u, v, q, k = uvqk.split(
                    [
                        hidden_dim * num_heads,
                        hidden_dim * num_heads,
                        attn_dim * num_heads,
                        attn_dim * num_heads,
                    ],
                    dim=1,
                )
                if use_fp8_qkv:
                    assert fp8_qkv_scale is not None
                    calibrate_scale_buffer(
                        fp8_qkv_scale,
                        uvqk[:, hidden_dim * num_heads :],
                        spf_alpha=attn_alpha,  # item (105) SPF snap
                    )
        else:
            normed_x, x_mean, x_rstd = triton_weighted_layer_norm_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
            )
            if use_fp8:
                calibrate_scale_buffer(fp8_input_scale, normed_x)
            uvqk = maybe_triton_addmm_fwd(
                x=normed_x,
                w=uvqk_weight.to(normed_x.dtype),
                y=uvqk_bias.to(normed_x.dtype),
            ).contiguous()
            u, v, q, k = uvqk.split(
                [
                    hidden_dim * num_heads,
                    hidden_dim * num_heads,
                    attn_dim * num_heads,
                    attn_dim * num_heads,
                ],
                dim=1,
            )
            if use_fp8_qkv:
                assert fp8_qkv_scale is not None
                calibrate_scale_buffer(
                    fp8_qkv_scale,
                    uvqk[:, hidden_dim * num_heads :],
                    spf_alpha=attn_alpha,  # item (105) SPF snap
                )
        if _MX6_FWD and normed_x is None:
            # MX6-served call without an e4m3 normed_x (U on MX6).
            _mx6_fwd.record_contract("phi1_input_ln_projection", mx6, u=u)
        elif use_fp8 and fp8_scale_value is not None:
            record_fp8_capability("phi1_input_ln_projection")
            record_fp8_contract(
                "phi1_input_ln_projection",
                normed_x=normed_x,
                u=u,
            )
        BLOCK_D = compute_BLOCK_D(x)
        q = q.view(-1, num_heads, attn_dim)
        k = k.view(-1, num_heads, attn_dim)
        v = v.view(-1, num_heads, hidden_dim)
        attention_qkv_scale = (
            qkv_scale_value if fp8_qkv_forward and qkv_scale_value else 1.0
        )
        fuse_silu = defer_silu
        silu_u = u if fuse_silu else F.silu(u)
        sort_by_length_indices = None
        # GLUE_FOLD3 LAZY_SORT (ops/glue_fold3.py): the permutation feeds only
        # the Triton attention kernels; compute it where one is about to run.
        lazy_sort = sort_by_length and glue_fold3.LAZY_SORT
        if sort_by_length and not lazy_sort:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        out = None
        if fp8_qkv_forward:
            # FlyDSL e4m3 forward when HSTU_ATTN_FWD=flydsl (or the deprecated
            # DLRM_HSTU_FP8_FWD_FLYDSL alias). q/k/v are e4m3 column slices of
            # the vqk buffer; attention_qkv_scale is the scale they were
            # quantized with -- the same tensors the Triton FP8 forward gets.
            out = _flydsl_fwd_fp8()(
                q=q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                N=max_seq_len,
                alpha=attn_alpha,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
                num_softmax_heads=num_softmax_heads,
                qkv_scale=attention_qkv_scale,
            )
        else:
            # bf16 FlyDSL forward. The e4m3 path is the branch above; this
            # returns None while DLRM_HSTU_FP8_FWD=1 so we never feed e4m3
            # into the bf16 kernel.
            out = _flydsl_fwd()(
                q=q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                N=max_seq_len,
                alpha=attn_alpha,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
                num_softmax_heads=num_softmax_heads,
            )
        if out is None:
            if lazy_sort:
                sort_by_length_indices = glue_fold3.sort_by_length_indices(
                    seq_offsets
                )
            out = triton_hstu_attention_fwd(
                N=max_seq_len,
                alpha=attn_alpha,
                q=q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
                sort_by_length_indices=sort_by_length_indices,
                enable_tma=enable_tma,
                num_softmax_heads=num_softmax_heads,
                q_scale=attention_qkv_scale,
                k_scale=attention_qkv_scale,
                v_scale=attention_qkv_scale,
            )
        if fp8_qkv_forward:
            record_fp8_capability("phi2_qkv_attention")
            record_fp8_contract(
                "phi2_qkv_attention",
                attention_output=out,
                k=k,
                q=q,
                u=u,
                v=v,
            )
        # update ctx
        saved_tensors = [
            x,
            norm_weight,
            norm_bias,
            x_mean,
            x_rstd,
            uvqk_weight,
            seq_offsets,
        ]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if not recompute_normed_x_in_backward:
            saved_tensors.append(normed_x)
        save_fp8_uvqk = (
            recompute_uvqk_in_backward
            and fp8_qkv_forward
            and _SAVE_FP8_UVQK
            and allow_save_fp8_uvqk
        )
        early_release_fp8_uvqk = save_fp8_uvqk and _EARLY_RELEASE_FP8_UVQK
        if save_fp8_uvqk:
            if early_release_fp8_uvqk:
                if not fuse_silu:
                    saved_tensors.append(u)
                # Retained-graph repeated backward recomputes V/Q/K after the
                # one-shot ephemeral pack has been consumed.
                saved_tensors.append(uvqk_bias)
            elif fuse_silu:
                saved_tensors.append(vqk)
            else:
                saved_tensors.extend((u, vqk))
        elif recompute_uvqk_in_backward:
            saved_tensors.append(uvqk_bias)
        else:
            saved_tensors.append(uvqk)
        # LAZY_SORT: nothing is saved when no Triton kernel used the permutation
        # (the backward then recomputes it only for its own Triton fallback).
        ctx.g3_sort_saved = sort_by_length and sort_by_length_indices is not None
        if lazy_sort and not ctx.g3_sort_saved:
            glue_fold3.note_armed(
                "sort",
                "sort_by_length permutation skipped (FlyDSL attention served; "
                "computed lazily for a Triton fallback); kill switch "
                "DLRM_HSTU_FP8_GLUE_FOLD3_SORT=0",
            )
        if ctx.g3_sort_saved:
            saved_tensors.append(sort_by_length_indices)
        # Must stay the LAST saved tensor: backward reads it as [-1].
        ctx.saved_bf16_normed_x = (
            save_fp8_uvqk and saved_normed_x_bf16 is not None
        )
        if ctx.saved_bf16_normed_x:
            saved_tensors.append(saved_normed_x_bf16)
        saved_normed_x_bf16 = None
        ctx.save_for_backward(*saved_tensors)
        ctx.attn_alpha = attn_alpha
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_seq_len = max_seq_len
        ctx.max_attn_len = max_attn_len
        ctx.recompute_normed_x_in_backward = recompute_normed_x_in_backward
        ctx.recompute_uvqk_in_backward = recompute_uvqk_in_backward
        ctx.hidden_dim = hidden_dim
        ctx.attn_dim = attn_dim
        ctx.num_heads = num_heads
        ctx.uvqk_bias_1d = uvqk_bias.dim() == 1
        ctx.norm_eps = norm_eps
        ctx.norm_BLOCK_D = BLOCK_D
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        ctx.enable_tma = enable_tma
        ctx.num_softmax_heads = num_softmax_heads
        ctx.fp8_forward = fp8_qkv_forward
        ctx.fuse_silu = fuse_silu
        ctx.save_fp8_uvqk = save_fp8_uvqk
        ctx.early_release_fp8_uvqk = early_release_fp8_uvqk
        ctx._ephemeral_fp8_vqk = vqk if early_release_fp8_uvqk else None
        ctx.fp8_input_scale_value = fp8_scale_value
        ctx.fp8_qkv_scale_value = qkv_scale_value
        # STORE_T token-major E4M3 normed_x (None unless armed). One-shot ctx
        # attribute, dropped on first backward use; a retained-graph second
        # backward recomputes it.
        ctx._stored_x_fp8_t = stored_x_fp8_t
        stored_x_fp8_t = None
        # Input-dropout fold (ops/input_dropout_fold.py): x is the dropout
        # output and this Function is its only consumer (the caller returns
        # the residual proxy), so its d_x is the whole d(dropout output).
        ctx.dropout_handoff = dropout_handoff
        # r14 E-RM (DLRM_HSTU_FP8_BWD_ERM, default off): the bottom layer's
        # out-proj site this Function claimed off x (dy_erm.claim; None with the
        # knob off). Its backward's LN backward then emits that consumer's FP8
        # operands (dy_erm.run_producer), its d_x being the consumer's dout.
        ctx.erm_site = erm_site
        if fuse_silu:
            # Ask the consumer's backward (HSTUComputeOutputFunction) to write
            # d(silu_u) into the leading columns of a d(uvqk)-wide buffer, which
            # backward below adopts instead of copying du into a fresh one.
            setattr(silu_u, HSTU_DUVQK_WIDTH_ATTR, uvqk_weight.shape[1])
            if (
                bwd_fp8.du_producer_requested()
                and fp8t_planned
                and bwd_fp8.du_producer_enabled()
            ):
                # B1a (r16, DLRM_HSTU_FP8_BWD_DU_E5M2=1): this call's backward
                # plans the fp8t uvqk wgrad, so the LN*U*dropout backward may
                # emit the e5m2 u columns into dz8 (hstu_du_e5m2.py). One site
                # handle per call, shared with the compute-output context.
                ctx.du_e5m2_site = hstu_du_e5m2.DuE5m2Site(uvqk_weight.data_ptr())
                setattr(silu_u, HSTU_DUVQK_SITE_ATTR, ctx.du_e5m2_site)
        if (
            hstu_do8_registry.sites_enabled()
            and bool(ctx.needs_input_grad[0])
            and _flydsl_do8_planned()(
                q=q,
                k=k,
                v=v,
                out=out,
                num_targets=num_targets,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
                num_softmax_heads=num_softmax_heads,
                qkv_scale=attention_qkv_scale,
            )
        ):
            # R15 A11 (hstu_do8_registry; DLRM_HSTU_FP8_BWD_DO8=1, default
            # off, gin apply_env_bootstrap.DLRM_HSTU_FP8_BWD_DO8): this
            # training call's FlyDSL backward plans the C8 do_blk dO
            # quantization (the backward's own predicates,
            # hstu_flydsl_mha.c8_do_blk_planned), so the d(attn) producer
            # (r16 L11 'plus') may emit those dO8/E8 bytes. Eval/no_grad
            # forwards get no site (needs_input_grad[0], x, is False there,
            # as for B1a above). One site handle per call, tagged on `out`
            # (ops/hstu_compute.py carries it to the [T, H*D] view the
            # compute-output context reads) and bound to dout by the backward
            # below. Separate from B1a's DuE5m2Site above (the u columns'
            # site, on silu_u).
            ctx.do8_site = hstu_do8_registry.new_site(out, seq_offsets, max_seq_len)
        return silu_u, out, x

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dsilu_u, dout, dresidual):  # pyre-ignore[2,3]
        # R15 A11 contract v2: an 'only' dO8 record taken by this layer
        # call's C8 attention backward stays retained for the M-E1 exact
        # re-runs in _backward_impl and is released when this backward ends,
        # on every exit (a no-op without such a record).
        try:
            return _HSTUPreprocessAndAttentionFunction._backward_impl(
                ctx, dsilu_u, dout, dresidual
            )
        finally:
            hstu_do8_registry.release(getattr(ctx, "do8_site", None))

    @staticmethod
    # pyre-ignore[14]
    def _backward_impl(
        ctx,  # pyre-ignore[2]
        dsilu_u: torch.Tensor,
        dout: torch.Tensor,
        dresidual: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,  # d_x
        torch.Tensor,  # d_norm_weight
        torch.Tensor,  # d_norm_bias
        None,
        None,
        None,
        None,
        torch.Tensor,  # d_uvqk_weight
        torch.Tensor,  # d_uvqk_bias
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        x, norm_weight, norm_bias, x_mean, x_rstd, uvqk_weight, seq_offsets = (
            ctx.saved_tensors[:7]
        )
        idx = 7
        if ctx.has_multiple_targets:
            num_targets = ctx.saved_tensors[idx]
            idx += 1
        else:
            num_targets = None
        ephemeral_vqk_fp8 = (
            getattr(ctx, "_ephemeral_fp8_vqk", None)
            if ctx.early_release_fp8_uvqk
            else None
        )
        recompute_saved_fp8 = (
            ctx.save_fp8_uvqk
            and ctx.early_release_fp8_uvqk
            and ephemeral_vqk_fp8 is None
        )
        recompute_fp8_qkv = (
            ctx.recompute_uvqk_in_backward
            and ctx.fp8_forward
            and (not ctx.save_fp8_uvqk or recompute_saved_fp8)
        )
        normed_x_fp8 = None
        release_fp8_after_attention = (
            ctx.recompute_normed_x_in_backward
            and ctx.fp8_forward
            and (recompute_fp8_qkv or ctx.early_release_fp8_uvqk)
        )
        # FP8 uvqk projection backward (bwd_fp8 DLRM_HSTU_FP8_BWD_UVQK, default
        # off). With WGRAD=fp8t the wgrad reads a token-major E4M3 normed_x
        # (stored by the forward LN under STORE_T, else recomputed below)
        # INSTEAD of the BF16 one. fp8t declines to the BF16 wgrad (FP8 dgrad
        # kept) when the BF16 normed_x is needed anyway: the dual-LN Q/K/V
        # recompute, a BF16 uvqk recompute, or a forward-saved BF16 normed_x.
        uvqk_fp8_bwd = ctx.uvqk_bias_1d and bwd_fp8.uvqk_fp8_backward_enabled()
        stored_x_fp8_t = getattr(ctx, "_stored_x_fp8_t", None)
        ctx._stored_x_fp8_t = None
        fp8t_wgrad = (
            uvqk_fp8_bwd
            and bwd_fp8.uvqk_fp8t_requested()
            and ctx.recompute_normed_x_in_backward
            and not recompute_fp8_qkv
            and not getattr(ctx, "saved_bf16_normed_x", False)
            and (ctx.save_fp8_uvqk or not ctx.recompute_uvqk_in_backward)
            and bwd_fp8.uvqk_fp8t_will_run(
                "uvqk_projection",
                uvqk_weight,
                rows=x.shape[0],
                dtype=x.dtype,
                input_scale_value=ctx.fp8_input_scale_value,
            )
        )
        wgrad_x_fp8_t = stored_x_fp8_t if fp8t_wgrad else None
        # Drop the local alias either way so `wgrad_x_fp8_t = None` below
        # really frees the [D, Mp] bytes (and an unused stored copy now).
        stored_x_fp8_t = None
        if ctx.recompute_normed_x_in_backward:
            if recompute_fp8_qkv:
                if ctx.fp8_input_scale_value is None:
                    raise RuntimeError(
                        "missing frozen FP8 input scale for LayerNorm recompute"
                    )
                normed_x, normed_x_fp8 = triton_weighted_layer_norm_dual_fwd(
                    x=x,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=ctx.norm_eps,
                    mean=x_mean,
                    rstd=x_rstd,
                    inv_scale=1.0 / ctx.fp8_input_scale_value,
                )
            elif getattr(ctx, "saved_bf16_normed_x", False):
                # Forward stored exactly what this LayerNorm would rebuild.
                normed_x = ctx.saved_tensors[-1]
            elif fp8t_wgrad:
                # No BF16 normed_x: the fp8t wgrad consumes the token-major
                # E4M3 copy; a later decline rebuilds BF16 via the hook below.
                normed_x = None
                if wgrad_x_fp8_t is not None:
                    bwd_fp8.log_uvqk_store_t_once("uvqk_projection", True)
                else:
                    if bwd_fp8.uvqk_store_fp8t():
                        bwd_fp8.log_uvqk_store_t_once("uvqk_projection", False)
                    _, wgrad_x_fp8_t = triton_weighted_layer_norm_recompute_fp8t(
                        x=x,
                        weight=norm_weight,
                        bias=norm_bias,
                        eps=ctx.norm_eps,
                        mean=x_mean,
                        rstd=x_rstd,
                        inv_scale=1.0 / ctx.fp8_input_scale_value,
                        pad_to=bwd_fp8.uvqk_tm_pad(),
                        tiled_xt=wgrad_ck.tiled_active(x.device),
                    )
            else:
                normed_x, _, _ = triton_weighted_layer_norm_fwd(
                    x=x,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=ctx.norm_eps,
                    mean=x_mean,
                    rstd=x_rstd,
                )
        else:
            normed_x = ctx.saved_tensors[idx]
            idx += 1
        vqk_fp8 = None
        attention_qkv_scale = 1.0
        if ctx.recompute_uvqk_in_backward:
            if ctx.save_fp8_uvqk:
                if ctx.early_release_fp8_uvqk:
                    if not ctx.fuse_silu:
                        u = ctx.saved_tensors[idx]
                        idx += 1
                    uvqk_bias = ctx.saved_tensors[idx]
                    idx += 1
                    vqk_fp8 = ephemeral_vqk_fp8
                elif ctx.fuse_silu:
                    vqk_fp8 = ctx.saved_tensors[idx]
                    idx += 1
                else:
                    u = ctx.saved_tensors[idx]
                    vqk_fp8 = ctx.saved_tensors[idx + 1]
                    idx += 2
                if ctx.fp8_qkv_scale_value is None:
                    raise RuntimeError(
                        "missing frozen FP8 Q/K/V scale for saved activations"
                    )
                # Read only by the _dequantize_fp8 Triton fallback below;
                # with item (48)(c) armed it is built there, on demand.
                qkv_scale = (
                    None
                    if _attn_glue_lazy()
                    else torch.full(
                        (1, 1),
                        ctx.fp8_qkv_scale_value,
                        dtype=torch.float32,
                        device=x.device,
                    )
                )
                if vqk_fp8 is not None:
                    v, q, k = vqk_fp8.split(
                        [
                            ctx.hidden_dim * ctx.num_heads,
                            ctx.attn_dim * ctx.num_heads,
                            ctx.attn_dim * ctx.num_heads,
                        ],
                        dim=1,
                    )
                uvqk = None
                attention_qkv_scale = ctx.fp8_qkv_scale_value
            else:
                uvqk_bias = ctx.saved_tensors[idx]
                idx += 1
            if recompute_fp8_qkv:
                if (
                    ctx.fp8_input_scale_value is None
                    or ctx.fp8_qkv_scale_value is None
                ):
                    raise RuntimeError(
                        "missing frozen FP8 scales for Q/K/V backward recompute"
                    )
                input_scale = torch.full(
                    (1, 1),
                    ctx.fp8_input_scale_value,
                    dtype=torch.float32,
                    device=x.device,
                )
                qkv_scale = torch.full(
                    (1, 1),
                    ctx.fp8_qkv_scale_value,
                    dtype=torch.float32,
                    device=x.device,
                )
                if normed_x_fp8 is None:
                    normed_x_fp8, _, _ = triton_weighted_layer_norm_fwd(
                        x=x,
                        weight=norm_weight,
                        bias=norm_bias,
                        eps=ctx.norm_eps,
                        mean=x_mean,
                        rstd=x_rstd,
                        output_fp8=True,
                        inv_scale=1.0 / ctx.fp8_input_scale_value,
                    )
                u_columns = ctx.hidden_dim * ctx.num_heads
                if not ctx.fuse_silu and not ctx.save_fp8_uvqk:
                    u = scaled_addmm_fp8_preq(
                        activation_fp8=normed_x_fp8,
                        scale_a=input_scale,
                        weight=uvqk_weight,
                        bias_or_residual=uvqk_bias[:u_columns],
                        out_dtype=x.dtype,
                        weight_layout=((0, u_columns),),
                        data_parallel=True,
                    ).contiguous()
                if _MX6_FWD and (
                    vqk_mx6 := _mx6_fwd.recompute_vqk(
                        ctx,
                        x=x,
                        norm_weight=norm_weight,
                        norm_bias=norm_bias,
                        uvqk_weight=uvqk_weight,
                        uvqk_bias=uvqk_bias,
                        input_scale=input_scale,
                        qkv_scale=qkv_scale,
                    )
                ) is not None:
                    # Retained-graph backward of an MX6-served forward:
                    # rebuild V/Q/K through MX6 (as the forward did).
                    vqk_fp8 = vqk_mx6
                else:
                    vqk_fp8 = scaled_mm_fp8_out_preq(
                        activation_fp8=normed_x_fp8,
                        scale_a=input_scale,
                        weight=uvqk_weight,
                        bias=uvqk_bias[u_columns:],
                        weight_layout=((u_columns, uvqk_weight.shape[1]),),
                        output_scale=qkv_scale,
                        scale_a_value=ctx.fp8_input_scale_value,
                        output_scale_value=ctx.fp8_qkv_scale_value,
                    ).contiguous()
                v, q, k = vqk_fp8.split(
                    [
                        ctx.hidden_dim * ctx.num_heads,
                        ctx.attn_dim * ctx.num_heads,
                        ctx.attn_dim * ctx.num_heads,
                    ],
                    dim=1,
                )
                uvqk = None
                attention_qkv_scale = ctx.fp8_qkv_scale_value
                # scaled_mm has consumed this input. Drop the full-row FP8
                # LayerNorm output before allocating the much larger gradient
                # projection scratch below.
                normed_x_fp8 = None
            elif not ctx.save_fp8_uvqk:
                uvqk = maybe_triton_addmm_fwd(
                    x=normed_x,
                    w=uvqk_weight.to(normed_x.dtype),
                    y=uvqk_bias.to(normed_x.dtype),
                )
        else:
            uvqk = ctx.saved_tensors[idx]
            idx += 1
        if ctx.sort_by_length and getattr(ctx, "g3_sort_saved", True):
            sort_by_length_indices = ctx.saved_tensors[idx]
        else:
            sort_by_length_indices = None

        duvqk = (
            _adopt_duvqk(
                dsilu_u,
                rows=x.shape[0],
                width=uvqk_weight.shape[1],
                u_width=ctx.hidden_dim * ctx.num_heads,
                dtype=x.dtype,
            )
            if ctx.fuse_silu
            else None
        )
        du_in_place = duvqk is not None
        if duvqk is None:
            duvqk = torch.empty(
                (x.shape[0], uvqk_weight.shape[1]),
                dtype=x.dtype,
                device=x.device,
            )
        # B1a (r16, DLRM_HSTU_FP8_BWD_DU_E5M2=1 only; hstu_du_e5m2.py): the
        # LN*U*dropout backward may have emitted this layer's e5m2 u columns
        # (+ amax / colsum) into a dz8 and returned a placeholder du (no bf16
        # du, no d(uvqk) buffer: duvqk above is this call's fresh one). Take
        # the record, or (cannot consume it this call) restore the bf16 u
        # columns of duvqk now -- lossy, logged, latched -- before anything
        # reads them. Either way dsilu_u is never read (du_taken).
        du_emit = None
        du_taken = False
        du_site = getattr(ctx, "du_e5m2_site", None)
        if du_site is not None:
            du_emit = hstu_du_e5m2.take(du_site, dsilu_u)
            if du_emit is not None:
                du_taken = True
                why = hstu_du_e5m2.check_consumable(
                    du_emit,
                    duvqk,
                    du_in_place,
                    uvqk_weight,
                    u_width=ctx.hidden_dim * ctx.num_heads,
                    fp8t_wgrad=fp8t_wgrad,
                    tm_operand_ok=bwd_fp8._tm_operand_ok(
                        wgrad_x_fp8_t, uvqk_weight, duvqk.shape[0]
                    ),
                )
                if why is not None:
                    hstu_du_e5m2.decline(
                        du_emit,
                        duvqk[:, du_emit.u_cols[0] : du_emit.u_cols[1]],
                        why[0],
                        why[1],
                    )
                    du_emit = None
        du, dv, dq, dk = duvqk.split(
            [
                ctx.hidden_dim * ctx.num_heads,
                ctx.hidden_dim * ctx.num_heads,
                ctx.attn_dim * ctx.num_heads,
                ctx.attn_dim * ctx.num_heads,
            ],
            dim=1,
        )
        if uvqk is not None:
            u, v, q, k = uvqk.split(
                [
                    ctx.hidden_dim * ctx.num_heads,
                    ctx.hidden_dim * ctx.num_heads,
                    ctx.attn_dim * ctx.num_heads,
                    ctx.attn_dim * ctx.num_heads,
                ],
                dim=1,
            )
        q = q.view(-1, ctx.num_heads, ctx.attn_dim)
        k = k.view(-1, ctx.num_heads, ctx.attn_dim)
        v = v.view(-1, ctx.num_heads, ctx.hidden_dim)
        dq = dq.view(-1, ctx.num_heads, ctx.attn_dim)
        dk = dk.view(-1, ctx.num_heads, ctx.attn_dim)
        dv = dv.view(-1, ctx.num_heads, ctx.hidden_dim)
        # DLRM_HSTU_FP8_BWD_DQ_E5M2 (Arbor R14 M2, default off): the C8
        # dq-split emits the e5m2 q columns of dz (+ amax/colsum partials)
        # through the FlyDSL backward's dq_fp8_sink hook and leaves the BF16
        # dq columns of duvqk unwritten; the cast below then covers u|v|k
        # only. Only on calls whose fp8t projection backward will run with a
        # published per-layer scale (warm-up / decline calls keep the shipped
        # path). The sink reports whether the dq-split actually used it.
        # R15 M-E1 (DLRM_HSTU_FP8_BWD_DKDV_E5M2, gin item (47), default off):
        # armed FIRST so the dq sink below adopts its dz8 and scale epoch (see
        # _dkdv_e5m2_arm). fp8t_wgrad already implies the FP8 uvqk backward
        # with a published per-layer scale; the helper re-checks all of it.
        # With the knob off nothing below changes. Merged ship-12 contract
        # M1: with a taken, consumable B1a record (du_emit) it adopts that
        # record's dz8 instead of allocating one; when it declines, the B1a
        # path below runs exactly as without the knob.
        dkdv_sink = None
        dq_sink_fresh = None
        if _dkdv_e5m2_requested():
            if fp8t_wgrad:
                dkdv_sink = _dkdv_e5m2_arm(
                    duvqk,
                    uvqk_weight,
                    wgrad_x_fp8_t,
                    ctx.fp8_input_scale_value,
                    ctx.num_heads,
                    ctx.hidden_dim,
                    ctx.attn_dim,
                    **({} if du_emit is None else {"du_emit": du_emit}),
                )
            else:
                _dkdv_e5m2_count("not_planned")
        dq_sink = None
        if (
            fp8t_wgrad
            and bwd_fp8.dq_producer_enabled()
            and _dq_sink_layout_ok(ctx, duvqk)
            # the token-major wgrad must consume the payload (a BF16 wgrad
            # would otherwise read a decoded e5m2 dz instead of the BF16 dz)
            and bwd_fp8._tm_operand_ok(wgrad_x_fp8_t, uvqk_weight, duvqk.shape[0])
            and not _dq_sink_diagnostics_on()
        ):
            dq_scale = (
                bwd_fp8.producer_inv_scale("uvqk_projection", uvqk_weight.data_ptr())
                if dkdv_sink is None
                # M-E1: the layer call's ONE scale epoch (same inv tensor)
                else (dkdv_sink.inv_scale, dkdv_sink.decode_scale)
            )
            if dq_scale is not None:
                from generative_recommenders.ops.triton.hstu_dq_e5m2 import (
                    DqE5m2Sink,
                )

                grad_dtype, grad_limit = bwd_fp8.grad_dtype_and_limit()
                q_lo = 2 * ctx.hidden_dim * ctx.num_heads
                dq_sink = DqE5m2Sink(
                    rows=duvqk.shape[0],
                    width=duvqk.shape[1],
                    q_cols=(q_lo, q_lo + ctx.attn_dim * ctx.num_heads),
                    inv_scale=dq_scale[0],
                    cast_scale=dq_scale[1],
                    pad=bwd_fp8.uvqk_tm_pad(),
                    grad_dtype=grad_dtype,
                    limit=grad_limit,
                    device=duvqk.device,
                    # Shared dz8 (one DqE5m2Sink contract for both owners):
                    # M-E1 armed -> its dz8 (allocated, or B1a's adopted),
                    # pad rows already zeroed; else B1a (r16): q lands in the
                    # LNMD producer's dz8 and this sink zeroes its q pad rows.
                    **(
                        {"dz8": dkdv_sink.dz8, "pad_zeroed": True}
                        if dkdv_sink is not None
                        else {"dz8": None if du_emit is None else du_emit.dz8}
                    ),
                )
                if dkdv_sink is not None:
                    # M-E1 fallback: a fresh dq sink, same arguments and
                    # scale epoch (the knob-off twin's q bytes); dz8 is passed
                    # at call time (no reference held here).
                    dq_sink_fresh = functools.partial(
                        DqE5m2Sink,
                        rows=duvqk.shape[0],
                        width=duvqk.shape[1],
                        q_cols=(q_lo, q_lo + ctx.attn_dim * ctx.num_heads),
                        inv_scale=dq_scale[0],
                        cast_scale=dq_scale[1],
                        pad=bwd_fp8.uvqk_tm_pad(),
                        grad_dtype=grad_dtype,
                        limit=grad_limit,
                        device=duvqk.device,
                        pad_zeroed=True,
                    )
            dq_scale = None
        # Note: the operations below update duvqk in place. The FlyDSL backward
        # is the default ($HSTU_ATTN_BWD) but declines any shape it cannot serve,
        # so the Triton call below is still both the opt-out and the fallback.
        # It is *not* the fallback for an unsupported host: that raises inside.
        vqk_bf16 = None
        # R15 A11 hook: a dO8 record that the d(attn) producer published on
        # this call's site is bound to the dO autograd delivered (a new view
        # of the producer's dx); the C8 branch takes it or quantizes. A site
        # without a record is counted (unbound), so partial arming shows.
        do8_site = getattr(ctx, "do8_site", None)
        hstu_do8_registry.bind(do8_site, dout)

        def _flydsl_attention_bwd(dq_fp8_sink=None, dkdv_fp8_sink=None) -> bool:  # pyre-ignore[2]
            # One call site for the FlyDSL backward, so an engaged R15 M-E1
            # call can re-run it with identical arguments and no sinks.
            return _flydsl_bwd()(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                N=ctx.max_seq_len,
                alpha=ctx.attn_alpha,
                max_attn_len=ctx.max_attn_len,
                contextual_seq_len=ctx.contextual_seq_len,
                num_softmax_heads=ctx.num_softmax_heads,
                qkv_scale=attention_qkv_scale,
                **({} if dq_fp8_sink is None else {"dq_fp8_sink": dq_fp8_sink}),
                **(
                    {}
                    if dkdv_fp8_sink is None
                    else {"dkdv_fp8_sink": dkdv_fp8_sink}
                ),
            )

        try:
            served = _flydsl_attention_bwd(dq_sink, dkdv_sink)
            # R15 A11: no record stays pending past the first call.
            hstu_do8_registry.discard(do8_site, "not_taken")
            if dkdv_sink is not None:
                dkdv_sink = _dkdv_e5m2_engaged(dkdv_sink, served)
            if dq_sink is not None and (not served or not dq_sink.filled):
                # The dq-split never used the sink (Triton fallback, no C8
                # dq-split this call, uncovered config): BF16 dq was written.
                bwd_fp8.log_uvqk_once(
                    f"dq_e5m2_unused:{served}:{dq_sink.declined}",
                    f"dq_e5m2 sink NOT used (served={served}, "
                    f"declined={dq_sink.declined}) -> shipped BF16 dq + cast",
                )
                dq_sink = None
            if not served:
                if vqk_fp8 is not None:
                    if qkv_scale is None:  # item (48)(c): built on demand
                        qkv_scale = torch.full(
                            (1, 1),
                            ctx.fp8_qkv_scale_value,
                            dtype=torch.float32,
                            device=x.device,
                        )
                    vqk_bf16 = _dequantize_fp8(
                        vqk_fp8,
                        qkv_scale,
                        x.dtype,
                    )
                    v, q, k = vqk_bf16.split(
                        [
                            ctx.hidden_dim * ctx.num_heads,
                            ctx.attn_dim * ctx.num_heads,
                            ctx.attn_dim * ctx.num_heads,
                        ],
                        dim=1,
                    )
                    q = q.view(-1, ctx.num_heads, ctx.attn_dim)
                    k = k.view(-1, ctx.num_heads, ctx.attn_dim)
                    v = v.view(-1, ctx.num_heads, ctx.hidden_dim)
                if sort_by_length_indices is None and ctx.sort_by_length:
                    # LAZY_SORT: the forward did not need it (FlyDSL served).
                    sort_by_length_indices = glue_fold3.sort_by_length_indices(
                        seq_offsets
                    )
                triton_hstu_attention_bwd(
                    dout=dout,
                    q=q,
                    k=k,
                    v=v,
                    dq=dq,
                    dk=dk,
                    dv=dv,
                    seq_offsets=seq_offsets,
                    num_targets=num_targets,
                    N=ctx.max_seq_len,
                    max_attn_len=ctx.max_attn_len,
                    alpha=ctx.attn_alpha,
                    contextual_seq_len=ctx.contextual_seq_len,
                    sort_by_length_indices=sort_by_length_indices,
                    enable_tma=ctx.enable_tma,
                    num_softmax_heads=ctx.num_softmax_heads,
                )
        finally:
            if ctx.early_release_fp8_uvqk:
                ctx._ephemeral_fp8_vqk = None
        if ctx.fuse_silu:
            # B1a (r16): with a taken emission record dsilu_u is the producer's
            # placeholder; the u columns are producer-cast (or restored above).
            if not du_in_place and not du_taken:
                du.copy_(dsilu_u)
        else:
            torch.ops.aten.silu_backward(dsilu_u, u, grad_input=du)
        # R15 M-E1: an engaged call (dkdv_sink) releases after its projection
        # backward instead: its exact fallback re-runs the attention backward.
        if release_fp8_after_attention and dkdv_sink is None:
            # Q/K/V are dead after attention backward. Releasing their packed
            # full-row storage and its context owner before projection backward
            # prevents the saved or recomputed FP8 input from inflating that
            # operation's HBM high-water mark. The BF16 LayerNorm result
            # remains live for the projection gradient (fp8t: the token-major
            # E4M3 copy instead).
            ephemeral_vqk_fp8 = None
            q = None
            k = None
            v = None
            vqk_fp8 = None
            vqk_bf16 = None
        assert normed_x is not None or wgrad_x_fp8_t is not None
        # An engaged R15 M-E1 sink implies fp8t_wgrad, hence uvqk_fp8_bwd, so
        # the BF16 triton_addmm_bwd below never reads its unwritten dV/dK.
        assert dkdv_sink is None or uvqk_fp8_bwd
        with record_function("## hstu_ordinary_uvqk_backward ##"):
            fp8_grads = None
            if uvqk_fp8_bwd:

                def _recompute_normed_x() -> torch.Tensor:
                    return triton_weighted_layer_norm_fwd(
                        x=x,
                        weight=norm_weight,
                        bias=norm_bias,
                        eps=ctx.norm_eps,
                        mean=x_mean,
                        rstd=x_rstd,
                    )[0]

                if dkdv_sink is not None:
                    # R15 M-E1 engaged call: v|k from the bwd_dkdv epilogue
                    # (+ q from the dq sink, + u from B1a when its record was
                    # taken: merged ship-12 contract M4) in the shared dz8;
                    # exact fallbacks re-run the attention backward without
                    # sinks (B1a's u: its LOSSY restore, M5).
                    fp8_grads = _dkdv_e5m2_projection_backward(
                        dkdv_sink,
                        duvqk,
                        ctx.hidden_dim * ctx.num_heads,
                        _flydsl_attention_bwd,
                        dq_sink,
                        dq_sink_fresh,
                        normed_x=normed_x,
                        input_scale_value=ctx.fp8_input_scale_value,
                        weight=uvqk_weight,
                        normed_x_fp8_t=wgrad_x_fp8_t,
                        recompute_normed_x=_recompute_normed_x,
                        **({} if du_emit is None else {"du_emit": du_emit}),
                    )
                    dkdv_sink = None
                    dq_sink = None
                    dq_sink_fresh = None
                    # Consumed (released) or declined (LOSSY u restored,
                    # released, latched) inside.
                    du_emit = None
                    if release_fp8_after_attention:
                        # The release deferred above.
                        ephemeral_vqk_fp8 = None
                        q = None
                        k = None
                        v = None
                        vqk_fp8 = None
                        vqk_bf16 = None
                elif dq_sink is None and du_emit is None:
                    fp8_grads = bwd_fp8.uvqk_projection_backward(
                        dz=duvqk,
                        normed_x=normed_x,
                        input_scale_value=ctx.fp8_input_scale_value,
                        weight=uvqk_weight,
                        normed_x_fp8_t=wgrad_x_fp8_t,
                        recompute_normed_x=_recompute_normed_x,
                    )
                elif du_emit is not None:
                    # B1a (r16): the u block (and the q block when M2's sink
                    # filled the same dz8) is producer-cast; the cast covers
                    # the rest. On None the bf16 u (and q) columns are
                    # restored (lossy) and B1a is latched off.
                    fp8_grads = hstu_du_e5m2.consume(
                        du_emit,
                        dq_sink,
                        duvqk,
                        normed_x=normed_x,
                        input_scale_value=ctx.fp8_input_scale_value,
                        weight=uvqk_weight,
                        normed_x_fp8_t=wgrad_x_fp8_t,
                        recompute_normed_x=_recompute_normed_x,
                    )
                    if dq_sink is not None and fp8_grads is not None:
                        _log_dq_sink_armed(dq_sink, du_emit=True)
                    dq_sink = None
                    du_emit = None
                else:
                    # M2: u|v|k-only cast + dq partials -> producer-cast dz.
                    fp8_grads = bwd_fp8.uvqk_projection_backward(
                        dz=None,
                        normed_x=normed_x,
                        input_scale_value=ctx.fp8_input_scale_value,
                        weight=uvqk_weight,
                        normed_x_fp8_t=wgrad_x_fp8_t,
                        recompute_normed_x=_recompute_normed_x,
                        prequant=dq_sink.finish(duvqk),
                        producer_desc=(
                            "dq-split e5m2 q columns + u|v|k-only cast, "
                            "no BF16 dq"
                        ),
                    )
                    _log_dq_sink_armed(dq_sink)
                    if fp8_grads is None:
                        # Declined/failed after the producer already cast dz
                        # (first-call failures only): lossy BF16 decode of the
                        # unwritten q columns for the shipped path.
                        dq_sink.restore_bf16(duvqk)
                    dq_sink = None
                wgrad_x_fp8_t = None
                if fp8_grads is None and normed_x is None:
                    # The FP8 path declined after the BF16 normed_x was
                    # skipped for it: rebuild it for the shipped path.
                    normed_x = _recompute_normed_x()
            if du_emit is not None:
                # B1a (r16): not reached (check_consumable requires the fp8t
                # wgrad, i.e. the branch above); no path reads unwritten du.
                hstu_du_e5m2.decline(
                    du_emit,
                    duvqk[:, du_emit.u_cols[0] : du_emit.u_cols[1]],
                    "bf16",
                    "BF16 uvqk projection backward",
                )
                du_emit = None
            if fp8_grads is not None:
                d_normed_x, d_uvqk_weight, d_uvqk_bias = fp8_grads
            else:
                d_normed_x, d_uvqk_weight, d_uvqk_bias = triton_addmm_bwd(
                    x=normed_x,
                    w=uvqk_weight.to(normed_x.dtype),
                    dz=duvqk,
                    is_y_1d=ctx.uvqk_bias_1d,
                )
            fp8_grads = None
        d_uvqk_weight = d_uvqk_weight.to(uvqk_weight.dtype)
        d_uvqk_bias = d_uvqk_bias.to(uvqk_weight.dtype)
        handoff = ctx.dropout_handoff
        fold_dropout = (
            handoff is not None
            and dresidual is not None
            and wln_bwd_mscale_supported(x, handoff.mask)
        )
        # r14 E-RM (DLRM_HSTU_FP8_BWD_ERM, default off; ops/triton/dy_erm.py):
        # only with a claimed site (never with the knob off) and only where the
        # call below would run item (29)'s FlyDSL residual kernel and the
        # consumer's W3 would run the armed x21 entry, the E-RM variant of that
        # kernel computes the same d_x / d_norm_weight / d_norm_bias and records
        # the bottom layer's FP8 out-proj operands for the consumer. None:
        # today's call below, unchanged.
        erm_out = None
        if getattr(ctx, "erm_site", None) is not None:
            # With a site only: drop this layer call's retained DO8-only
            # dO8/E8 now instead of in backward()'s finally. Their last reader
            # (the C8 attention backward and its M-E1 re-runs) is done; the
            # release is idempotent (the finally's becomes a no-op), so the
            # producer's g8 (also M*512 B) can reuse dO8's memory instead of
            # coexisting with it in this LN-backward tail.
            hstu_do8_registry.release(getattr(ctx, "do8_site", None))
            erm_out = _dy_erm().run_producer(
                ctx.erm_site,
                d_normed_x,
                x,
                norm_weight,
                norm_bias,
                x_mean,
                x_rstd,
                dresidual,
                fold_dropout,
            )
        if erm_out is not None:
            d_x, d_norm_weight, d_norm_bias = erm_out
        else:
            d_x, d_norm_weight, d_norm_bias = triton_weighted_layer_norm_bwd(
                dy=d_normed_x,
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                mean=x_mean,
                rstd=x_rstd,
                learnable=True,
                eps=ctx.norm_eps,
                BLOCK_D=ctx.norm_BLOCK_D,
                residual=dresidual,
                dropout_mask=handoff.mask if fold_dropout else None,
                dropout_scale=handoff.scale if fold_dropout else 1.0,
            )
        erm_out = None
        if fold_dropout:
            # The dropout backward passes exactly this tensor through.
            handoff.folded_dx = d_x
            if not _MSCALE_FOLD_LOGGED:
                _log_mscale_fold_once()
        # pyre-ignore[7]
        return (
            d_x,
            d_norm_weight,
            d_norm_bias,
            None,
            None,
            None,
            None,
            d_uvqk_weight,
            d_uvqk_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # num_softmax_heads
            None,  # dropout_handoff
            None,  # erm_site (r14 E-RM)
        )


def _b15t_skip_y(
    x: torch.Tensor, tiled_xt: bool, pad_to: int, vk_columns: int
) -> bool:
    """B15T(L2) (yambda_5b.gin item B15T(L2), DLRM_HSTU_FP8_FWD_K512_TILED_A):
    whether the targets-only store_t may run with store_y=False because the
    V/K projection will read the step-tiled yT on K512's tiled-A entry. The
    unmodified Triton v2 store_t kernel then still WRITES y's bytes, but into
    a dedicated 1/32-size scratch that is discarded (no [M, 512] y is
    allocated; fewer DRAM write-backs, not fewer stores), and the yT must be
    step-tiled (`tiled_xt` is the one wgrad_ck.tiled_active value the store_t
    call gets too). Host only (no tensor read); with the knob on, a "no" is
    logged once per reason (kind and detail)."""
    if not k512_tiled_a_requested():
        return False
    reason = store_t_skip_y_decline_reason(x, pad_to, tiled_xt)
    if reason is None:
        reason = k512_tiled_a_vk_decline_reason(
            x.device, int(x.shape[1]), vk_columns, int(x.shape[0])
        )
    if reason is not None:
        log_b15t_decline_once(*reason)
        return False
    return True


class _HSTUPreprocessAndAttentionTargetsOnlyFunction(torch.autograd.Function):
    """Final-layer training path with full K/V and one compact U/Q per sequence."""

    @staticmethod
    # pyre-ignore [14]
    def forward(
        ctx,  # pyre-ignore [2]
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        norm_eps: float,
        num_heads: int,
        attn_dim: int,
        hidden_dim: int,
        uvqk_weight: torch.Tensor,
        uvqk_bias: torch.Tensor,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        attn_alpha: float,
        num_targets: torch.Tensor,
        enable_tma: bool,
        fp8_input_scale: Optional[torch.Tensor],
        fp8_qkv_scale: Optional[torch.Tensor],
        residual_rows_only: bool = False,
        allow_save_fp8_vk: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u_dim = hidden_dim * num_heads
        v_dim = hidden_dim * num_heads
        q_dim = attn_dim * num_heads
        k_dim = attn_dim * num_heads
        refuse_fp8_with_flydsl_attention_forward()
        fp8_scale_value = activation_scale_value(fp8_input_scale)
        fp8_scale = (
            fp8_scale_tensor(fp8_input_scale)
            if fp8_input_scale is not None
            else None
        )
        qkv_scale_value = activation_scale_value(fp8_qkv_scale)
        fp8_requested = (
            hstu_fp8_fuse_qkv_enabled()
            and hstu_fp8_attention_enabled()
        )
        use_fp8_vk = (
            fp8_requested
            and fp8_input_scale is not None
            and fp8_qkv_scale is not None
            and fp8_supported(x, uvqk_weight)
        )
        if fp8_requested and not use_fp8_vk and hstu_fp8_fail_closed():
            raise RuntimeError(
                "Plan 08 targets-only FP8 path requested for an unsupported contract"
            )
        input_fp8_forward = use_fp8_vk and fp8_scale_value is not None
        qkv_fp8_forward = input_fp8_forward and qkv_scale_value is not None
        # See _SAVE_BF16_NORMED_X: keep the BF16 normalized input from the
        # forward LayerNorm for the backward VK wgrad / compact UQ rows
        # instead of re-running the LayerNorm there.
        saved_normed_x_bf16 = None
        # FP8 VK wgrad (bwd_fp8 DLRM_HSTU_FP8_BWD_UVQK_WGRAD=fp8t, default
        # off): as in the ordinary layer, when backward will take it the BF16
        # normed_x store is skipped (the compact U/Q rows then take the
        # per-row LayerNorm below, as without SAVE_NORMED_X) and STORE_T writes
        # the token-major E4M3 copy in this LN pass. Backward keys this site
        # on its name (per_layer=False) and checks the [D, v+k] VK weight's
        # shape, for which a column view of uvqk_weight stands in here.
        stored_vk_x_fp8_t = None
        vk_fp8t_planned = (
            input_fp8_forward
            and bwd_fp8.uvqk_fp8t_requested()
            and bool(ctx.needs_input_grad[0])
            and bwd_fp8.uvqk_fp8t_will_run(
                "vk_projection",
                uvqk_weight.narrow(1, 0, v_dim + k_dim),
                rows=x.shape[0],
                dtype=x.dtype,
                input_scale_value=fp8_scale_value,
                per_layer=False,
            )
        )
        if vk_fp8t_planned and _SAVE_BF16_NORMED_X:
            bwd_fp8.log_uvqk_once(
                "skip_bf16_save:vk_projection",
                "vk_projection: fp8t wgrad planned, forward skips the BF16 "
                "normed_x save (DLRM_HSTU_FP8_SAVE_NORMED_X); the token-major "
                "E4M3 normed_x replaces it",
            )
        if _MX6_FWD and (
            mx6 := _mx6_fwd.vk_ln_store(
                x=x,
                norm_weight=norm_weight,
                norm_bias=norm_bias,
                eps=norm_eps,
                fp8_scale_value=fp8_scale_value,
                uvqk_weight=uvqk_weight,
                uvqk_bias=uvqk_bias,
                vk_layout=(
                    (u_dim, u_dim + v_dim),
                    (u_dim + v_dim + q_dim, uvqk_weight.shape[1]),
                ),
                qkv_frozen=qkv_fp8_forward,
                grad=bool(ctx.needs_input_grad[0]),
                store_t=vk_fp8t_planned and bwd_fp8.uvqk_store_fp8t(),
                pack_handed=(
                    qkv_fp8_forward
                    and _SAVE_FP8_UVQK
                    and _EARLY_RELEASE_FP8_UVQK
                    and allow_save_fp8_vk
                ),
                pad_to=bwd_fp8.uvqk_tm_pad(),
                tiled_xt=wgrad_ck.tiled_active(x.device),
            )
        ) is not None:
            # MX6 LN producer in place of the store_t (same mean / rstd /
            # xT8); V/K is the only reader of the e4m3 normed_x here, so x8
            # is written on the first-call guard only.
            normed_x = mx6.x8
            x_mean = mx6.mean
            x_rstd = mx6.rstd
            stored_vk_x_fp8_t = mx6.xt8
        elif vk_fp8t_planned and bwd_fp8.uvqk_store_fp8t():
            assert fp8_scale is not None
            tiled_xt = wgrad_ck.tiled_active(x.device)
            tm_pad = bwd_fp8.uvqk_tm_pad()
            # B15T(L2) (gin item B15T(L2)): this LayerNorm's row-major output
            # y has one reader, the e4m3 V/K projection below (it is not saved
            # for backward, and the compact U/Q rows are normalized from x and
            # mean/rstd), so when that projection will read the step-tiled yT
            # on K512's tiled-A entry, the store_t runs with store_y=False: the
            # unmodified v2 kernel still writes y's bytes, but into a
            # dedicated 1/32-size scratch that is discarded (y is None; fewer
            # DRAM write-backs, not fewer stores). Decided on the host.
            skip_y = qkv_fp8_forward and _b15t_skip_y(
                x, tiled_xt, tm_pad, v_dim + k_dim
            )
            (
                normed_x,
                x_mean,
                x_rstd,
                stored_vk_x_fp8_t,
            ) = triton_weighted_layer_norm_fwd_fp8_store_t(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
                inv_scale=1.0 / fp8_scale_value,
                pad_to=tm_pad,
                tiled_xt=tiled_xt,
                store_y=not skip_y,
            )
            if skip_y and normed_x is not None:
                # Not expected (the pre-check mirrors the store_t's choice);
                # y was written, so this call is the knob-off path as is.
                latch_b15t_off(
                    "store_t",
                    "the store_t wrote y although store_y=False (not the v2 "
                    "kernel for this call)",
                )
        elif (
            input_fp8_forward
            and _SAVE_BF16_NORMED_X
            and bool(ctx.needs_input_grad[0])
            and not vk_fp8t_planned
        ):
            assert fp8_scale is not None
            (
                saved_normed_x_bf16,
                normed_x,
                x_mean,
                x_rstd,
            ) = triton_weighted_layer_norm_fp8_save_bf16_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
                inv_scale=1.0 / fp8_scale_value,
            )
        elif input_fp8_forward:
            assert fp8_scale is not None
            normed_x, x_mean, x_rstd = triton_weighted_layer_norm_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
                output_fp8=True,
                inv_scale=1.0 / fp8_scale_value,
            )
        else:
            normed_x, x_mean, x_rstd = triton_weighted_layer_norm_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
            )
            if use_fp8_vk:
                calibrate_scale_buffer(fp8_input_scale, normed_x)
        BLOCK_D = compute_BLOCK_D(x)
        w_u, w_v, w_q, w_k = uvqk_weight.split(
            [u_dim, v_dim, q_dim, k_dim], dim=1
        )
        b_u, b_v, b_q, b_k = uvqk_bias.split(
            [u_dim, v_dim, q_dim, k_dim], dim=0
        )
        candidate_rows = seq_offsets[1:] - 1
        candidate_normed_x_bf16 = None
        candidate_x = None
        if saved_normed_x_bf16 is not None:
            # Same values as the per-row LayerNorm below (same fp32
            # mean/rstd, same kernel arithmetic), without the gathers + launch.
            candidate_normed_x_bf16 = torch.index_select(
                saved_normed_x_bf16, 0, candidate_rows
            )
        elif input_fp8_forward:
            candidate_x = torch.index_select(x, 0, candidate_rows)
            candidate_normed_x_bf16, _, _ = triton_weighted_layer_norm_fwd(
                x=candidate_x,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
                mean=torch.index_select(x_mean, 0, candidate_rows),
                rstd=torch.index_select(x_rstd, 0, candidate_rows),
            )

        b_vk = torch.cat((b_v, b_k), dim=0)
        if qkv_fp8_forward:
            assert fp8_scale is not None
            assert fp8_qkv_scale is not None
            if _MX6_FWD and mx6 is not None:
                # MXFP6 V/K (pr7 N=1024 kernel), e4m3 at the qkv scale.
                vk = _mx6_fwd.gemm_vk(
                    mx6,
                    ctx=ctx,
                    uvqk_weight=uvqk_weight,
                    b_vk=b_vk,
                    vk_layout=(
                        (u_dim, u_dim + v_dim),
                        (u_dim + v_dim + q_dim, uvqk_weight.shape[1]),
                    ),
                    fp8_scale=fp8_scale,
                    fp8_scale_value=fp8_scale_value,
                    fp8_qkv_scale=fp8_qkv_scale,
                    qkv_scale_value=qkv_scale_value,
                )
            else:
                vk_layout = (
                    (u_dim, u_dim + v_dim),
                    (u_dim + v_dim + q_dim, uvqk_weight.shape[1]),
                )
                vk = None
                if normed_x is None:
                    # B15T(L2): the store_t skipped y; the projection reads y's
                    # bytes from the step-tiled yT (K512 tiled-A entry). A decline
                    # there (logged once; B15T then latches off) rebuilds y from
                    # yT, bit for bit, for the row-major projection.
                    assert stored_vk_x_fp8_t is not None
                    vk = scaled_mm_fp8_out_preq_tiled_a(
                        stored_vk_x_fp8_t,
                        int(x.shape[0]),
                        fp8_scale,
                        uvqk_weight,
                        b_vk,
                        weight_layout=vk_layout,
                        output_scale=fp8_qkv_scale,
                        scale_a_value=fp8_scale_value,
                        output_scale_value=qkv_scale_value,
                    )
                    if vk is None:
                        normed_x = b15t_row_major_y(
                            stored_vk_x_fp8_t, int(x.shape[0])
                        )
                if vk is None:
                    vk = scaled_mm_fp8_out_preq(
                        activation_fp8=normed_x,
                        scale_a=fp8_scale,
                        weight=uvqk_weight,
                        bias=b_vk,
                        weight_layout=vk_layout,
                        output_scale=fp8_qkv_scale,
                        scale_a_value=fp8_scale_value,
                        output_scale_value=qkv_scale_value,
                    )
                vk = vk.contiguous()
            assert candidate_normed_x_bf16 is not None
            candidate_normed_x = candidate_normed_x_bf16
        elif input_fp8_forward:
            assert fp8_scale is not None
            vk = scaled_addmm_fp8_preq(
                activation_fp8=normed_x,
                scale_a=fp8_scale,
                weight=uvqk_weight,
                bias_or_residual=b_vk,
                out_dtype=x.dtype,
                weight_layout=(
                    (u_dim, u_dim + v_dim),
                    (u_dim + v_dim + q_dim, uvqk_weight.shape[1]),
                ),
            ).contiguous()
            assert candidate_normed_x_bf16 is not None
            candidate_normed_x = candidate_normed_x_bf16
        else:
            # Packing only the small weights keeps the large activation
            # projection at [L, 2D] without changing checkpoint layout.
            if use_fp8_vk:
                w_vk = packed_weight_for_layout(
                    uvqk_weight,
                    (
                        (u_dim, u_dim + v_dim),
                        (
                            u_dim + v_dim + q_dim,
                            uvqk_weight.shape[1],
                        ),
                    ),
                    dtype=normed_x.dtype,
                )
            else:
                w_vk = torch.cat((w_v, w_k), dim=1).to(normed_x.dtype)
            vk = maybe_triton_addmm_fwd(
                x=normed_x,
                w=w_vk,
                y=b_vk.to(normed_x.dtype),
            ).contiguous()
            candidate_normed_x = torch.index_select(
                normed_x, 0, candidate_rows
            )
        if use_fp8_vk and not qkv_fp8_forward:
            assert fp8_qkv_scale is not None
            # Item (105) SPF snap: every qkv scale is on the grid under the
            # knob, so an eval full forward of this layer can arm.
            calibrate_scale_buffer(fp8_qkv_scale, vk, spf_alpha=attn_alpha)
        v, k = vk.split([v_dim, k_dim], dim=1)
        if use_fp8_vk:
            w_uq = packed_weight_for_layout(
                uvqk_weight,
                (
                    (0, u_dim),
                    (u_dim + v_dim, u_dim + v_dim + q_dim),
                ),
                dtype=candidate_normed_x.dtype,
            )
        else:
            w_uq = torch.cat((w_u, w_q), dim=1).to(
                candidate_normed_x.dtype
            )
        b_uq = torch.cat((b_u, b_q), dim=0)
        uq = maybe_triton_addmm_fwd(
            x=candidate_normed_x,
            w=w_uq,
            y=b_uq.to(candidate_normed_x.dtype),
        ).contiguous()
        if input_fp8_forward:
            record_fp8_capability(
                "phi1_targets_compact_uq_banked_bf16"
            )
            record_fp8_contract(
                "phi1_targets_compact_uq_banked_bf16",
                compact_normed_x=candidate_normed_x,
                compact_uq=uq,
            )
        u, q = uq.split([u_dim, q_dim], dim=1)
        q = q.view(-1, num_heads, attn_dim)
        k = k.view(-1, num_heads, attn_dim)
        v = v.view(-1, num_heads, hidden_dim)
        attention_qkv_scale = (
            qkv_scale_value if qkv_fp8_forward and qkv_scale_value else 1.0
        )
        out = triton_cached_hstu_mha(
            N=max_seq_len,
            alpha=attn_alpha,
            delta_q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            max_attn_len=0,
            contextual_seq_len=0,
            enable_tma=enable_tma,
            q_scale=1.0,
            k_scale=attention_qkv_scale,
            v_scale=attention_qkv_scale,
        )
        if _MX6_FWD and mx6 is not None and normed_x is None:
            # MX6-served call without an e4m3 normed_x (after the guard call).
            # mx6 is checked too: with B15T(L2) a call MX6 does not serve can
            # also leave normed_x None; it takes the shipped record below.
            _mx6_fwd.record_contract(
                "phi1_targets_input_ln", mx6, compact_q=q, compact_u=u
            )
        elif input_fp8_forward:
            record_fp8_capability("phi1_targets_input_ln")
            record_fp8_contract(
                "phi1_targets_input_ln",
                compact_q=q,
                compact_u=u,
                normed_x=(
                    normed_x
                    if normed_x is not None
                    # B15T(L2): no [M, 512] y (its bytes went to the discarded
                    # scratch). The contract records only dtype and shape:
                    # those of the e4m3 y the yT holds.
                    else torch.empty(
                        x.shape, dtype=torch.float8_e4m3fn, device="meta"
                    )
                ),
            )
        if qkv_fp8_forward:
            record_fp8_capability("phi2_targets_vk_attention")
            record_fp8_contract(
                "phi2_targets_vk_attention",
                attention_output=out,
                compact_q=q,
                compact_u=u,
                full_k=k,
                full_v=v,
            )

        ctx.saved_bf16_normed_x = saved_normed_x_bf16 is not None
        ctx.save_for_backward(
            x,
            norm_weight,
            norm_bias,
            x_mean,
            x_rstd,
            uvqk_weight,
            uvqk_bias,
            seq_offsets,
            *((saved_normed_x_bf16,) if ctx.saved_bf16_normed_x else ()),
        )
        saved_normed_x_bf16 = None
        ctx.attn_alpha = attn_alpha
        ctx.max_seq_len = max_seq_len
        ctx.hidden_dim = hidden_dim
        ctx.attn_dim = attn_dim
        ctx.num_heads = num_heads
        ctx.norm_eps = norm_eps
        ctx.norm_BLOCK_D = BLOCK_D
        ctx.fp8_contract = use_fp8_vk
        ctx.fp8_input_forward = input_fp8_forward
        ctx.fp8_qkv_forward = qkv_fp8_forward
        ctx.fp8_input_scale_value = fp8_scale_value
        ctx.fp8_qkv_scale_value = qkv_scale_value
        # STORE_T token-major E4M3 normed_x (None unless armed); one-shot.
        ctx._stored_vk_x_fp8_t = stored_vk_x_fp8_t
        stored_vk_x_fp8_t = None
        # Hand the forward's e4m3 V/K pack to backward instead of rebuilding
        # it there (full-row FP8 LayerNorm + [L,512]x[512,1024] scaled GEMM),
        # under the same SAVE_UVQK / layer-mask / early-release policy as the
        # ordinary layers. It is a one-shot ctx attribute, not a saved tensor:
        # backward drops it after attention backward, so it never coexists
        # with the BF16 normalized input rematerialized for projection
        # backward, and a retained-graph second backward falls back to the
        # recompute. Backward then consumes exactly the values the forward
        # attention used.
        ctx._ephemeral_fp8_vk = (
            vk
            if (
                qkv_fp8_forward
                and _SAVE_FP8_UVQK
                and _EARLY_RELEASE_FP8_UVQK
                and allow_save_fp8_vk
            )
            else None
        )
        ctx.residual_rows_only = residual_rows_only
        if residual_rows_only:
            # Row-sparse residual proxy: return only the candidate rows of x, so
            # the residual gradient arrives as [num_candidates, D] and is added
            # to those rows of d_x after the LayerNorm backward (see backward).
            # This replaces the caller's index_select(x) whose backward
            # zero-fills a dense [L, D] gradient, index_adds into it, and makes
            # the LN backward read it back densely.
            if candidate_x is None:
                candidate_x = torch.index_select(x, 0, candidate_rows)
            return F.silu(u), out, candidate_x
        return F.silu(u), out, x

    @staticmethod
    # pyre-ignore [14]
    def backward(
        ctx,  # pyre-ignore [2]
        dsilu_u: torch.Tensor,
        dout: torch.Tensor,
        dresidual: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,  # norm_eps
        None,  # num_heads
        None,  # attn_dim
        None,  # hidden_dim
        torch.Tensor,
        torch.Tensor,
        None,  # max_seq_len
        None,  # seq_offsets
        None,  # attn_alpha
        None,  # num_targets
        None,  # enable_tma
        None,  # fp8_input_scale
        None,  # fp8_qkv_scale
        None,  # residual_rows_only
        None,  # allow_save_fp8_vk
    ]:
        (
            x,
            norm_weight,
            norm_bias,
            x_mean,
            x_rstd,
            uvqk_weight,
            uvqk_bias,
            seq_offsets,
        ) = ctx.saved_tensors[:8]
        saved_normed_x_bf16 = (
            ctx.saved_tensors[8]
            if getattr(ctx, "saved_bf16_normed_x", False)
            else None
        )
        normed_x = None
        if not ctx.fp8_input_forward:
            normed_x, _, _ = triton_weighted_layer_norm_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=ctx.norm_eps,
                mean=x_mean,
                rstd=x_rstd,
            )
        u_dim = ctx.hidden_dim * ctx.num_heads
        v_dim = ctx.hidden_dim * ctx.num_heads
        q_dim = ctx.attn_dim * ctx.num_heads
        k_dim = ctx.attn_dim * ctx.num_heads
        w_u, w_v, w_q, w_k = uvqk_weight.split(
            [u_dim, v_dim, q_dim, k_dim], dim=1
        )
        b_u, b_v, b_q, b_k = uvqk_bias.split(
            [u_dim, v_dim, q_dim, k_dim], dim=0
        )
        candidate_rows = seq_offsets[1:] - 1
        attention_vk_scale = 1.0
        if saved_normed_x_bf16 is not None:
            candidate_normed_x = torch.index_select(
                saved_normed_x_bf16, 0, candidate_rows
            )
        elif ctx.fp8_input_forward:
            candidate_normed_x, _, _ = triton_weighted_layer_norm_fwd(
                x=torch.index_select(x, 0, candidate_rows),
                weight=norm_weight,
                bias=norm_bias,
                eps=ctx.norm_eps,
                mean=torch.index_select(x_mean, 0, candidate_rows),
                rstd=torch.index_select(x_rstd, 0, candidate_rows),
            )
        else:
            candidate_normed_x = torch.index_select(
                normed_x, 0, candidate_rows
            )
        if ctx.fp8_contract:
            w_vk = packed_weight_for_layout(
                uvqk_weight,
                (
                    (u_dim, u_dim + v_dim),
                    (u_dim + v_dim + q_dim, uvqk_weight.shape[1]),
                ),
                dtype=x.dtype,
            )
        else:
            assert normed_x is not None
            w_vk = torch.cat((w_v, w_k), dim=1).to(normed_x.dtype)
        b_vk = torch.cat((b_v, b_k), dim=0).to(x.dtype)
        if ctx.fp8_contract:
            w_uq = packed_weight_for_layout(
                uvqk_weight,
                (
                    (0, u_dim),
                    (u_dim + v_dim, u_dim + v_dim + q_dim),
                ),
                dtype=x.dtype,
            )
        else:
            assert normed_x is not None
            w_uq = torch.cat((w_u, w_q), dim=1).to(normed_x.dtype)
        b_uq = torch.cat((b_u, b_q), dim=0).to(x.dtype)
        vk_fp8 = None
        saved_vk_fp8 = getattr(ctx, "_ephemeral_fp8_vk", None)
        ctx._ephemeral_fp8_vk = None
        if saved_vk_fp8 is not None:
            if ctx.fp8_qkv_scale_value is None:
                raise RuntimeError(
                    "missing frozen V/K scale for saved targets-only V/K"
                )
            vk_fp8 = saved_vk_fp8
            saved_vk_fp8 = None
            vk = vk_fp8
            attention_vk_scale = ctx.fp8_qkv_scale_value
        elif ctx.fp8_input_forward:
            if ctx.fp8_input_scale_value is None:
                raise RuntimeError(
                    "missing frozen input scale for targets-only backward"
                )
            input_scale = torch.full(
                (1, 1),
                ctx.fp8_input_scale_value,
                dtype=torch.float32,
                device=x.device,
            )
            normed_x_fp8, _, _ = triton_weighted_layer_norm_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=ctx.norm_eps,
                mean=x_mean,
                rstd=x_rstd,
                output_fp8=True,
                inv_scale=1.0 / ctx.fp8_input_scale_value,
            )
            vk_layout = (
                (u_dim, u_dim + v_dim),
                (u_dim + v_dim + q_dim, uvqk_weight.shape[1]),
            )
            if ctx.fp8_qkv_forward:
                if ctx.fp8_qkv_scale_value is None:
                    raise RuntimeError(
                        "missing frozen V/K scale for targets-only backward"
                    )
                qkv_scale = torch.full(
                    (1, 1),
                    ctx.fp8_qkv_scale_value,
                    dtype=torch.float32,
                    device=x.device,
                )
                if _MX6_FWD and (
                    vk_mx6 := _mx6_fwd.recompute_vk(
                        ctx,
                        x=x,
                        norm_weight=norm_weight,
                        norm_bias=norm_bias,
                        uvqk_weight=uvqk_weight,
                        b_vk=torch.cat((b_v, b_k), dim=0),
                        vk_layout=vk_layout,
                        input_scale=input_scale,
                        qkv_scale=qkv_scale,
                    )
                ) is not None:
                    # Retained-graph backward of an MX6-served forward.
                    vk_fp8 = vk_mx6
                else:
                    vk_fp8 = scaled_mm_fp8_out_preq(
                        activation_fp8=normed_x_fp8,
                        scale_a=input_scale,
                        weight=uvqk_weight,
                        bias=torch.cat((b_v, b_k), dim=0),
                        weight_layout=vk_layout,
                        output_scale=qkv_scale,
                        scale_a_value=ctx.fp8_input_scale_value,
                        output_scale_value=ctx.fp8_qkv_scale_value,
                    ).contiguous()
                vk = vk_fp8
                attention_vk_scale = ctx.fp8_qkv_scale_value
            else:
                vk = scaled_addmm_fp8_preq(
                    activation_fp8=normed_x_fp8,
                    scale_a=input_scale,
                    weight=uvqk_weight,
                    bias_or_residual=torch.cat((b_v, b_k), dim=0),
                    out_dtype=x.dtype,
                    weight_layout=vk_layout,
                ).contiguous()
        else:
            assert normed_x is not None
            vk = maybe_triton_addmm_fwd(
                x=normed_x, w=w_vk, y=b_vk
            ).contiguous()
        v, k = vk.split([v_dim, k_dim], dim=1)
        uq = maybe_triton_addmm_fwd(
            x=candidate_normed_x, w=w_uq, y=b_uq
        ).contiguous()
        u, q = uq.split([u_dim, q_dim], dim=1)

        q = q.view(-1, ctx.num_heads, ctx.attn_dim)
        k = k.view(-1, ctx.num_heads, ctx.attn_dim)
        v = v.view(-1, ctx.num_heads, ctx.hidden_dim)
        dq = torch.empty_like(q)
        # FP8 VK projection backward (bwd_fp8 DLRM_HSTU_FP8_BWD_UVQK, default
        # off). The fp8t decision is taken once here, before attention
        # backward, so the VK producer and the normed_x handling agree. fp8t
        # declines (BF16 wgrad, FP8 dgrad kept) when a forward-saved BF16
        # normed_x is present.
        vk_fp8_bwd = bwd_fp8.uvqk_fp8_backward_enabled()
        stored_vk_x_fp8_t = getattr(ctx, "_stored_vk_x_fp8_t", None)
        ctx._stored_vk_x_fp8_t = None
        vk_fp8t_will_run = (
            vk_fp8_bwd
            and ctx.fp8_input_forward
            and bwd_fp8.uvqk_fp8t_requested()
            and saved_normed_x_bf16 is None
            and bwd_fp8.uvqk_fp8t_will_run(
                "vk_projection",
                w_vk,
                rows=x.shape[0],
                dtype=x.dtype,
                input_scale_value=ctx.fp8_input_scale_value,
                per_layer=False,
            )
        )
        if not vk_fp8t_will_run:
            stored_vk_x_fp8_t = None
        vk_x_fp8_t = None
        # DLRM_HSTU_FP8_BWD_VK_PRODUCER (default off): the delta backward
        # emits the E5M2 dV|dK pack (+amax, +bias colsum) itself, byte-identical
        # to casting the BF16 dvk; no BF16 dvk, no cast pass.
        vk_prequant = None
        vk_cast_scale = None
        vk_producer_scale = (
            bwd_fp8.producer_inv_scale("vk_projection")
            if vk_fp8t_will_run and bwd_fp8.vk_producer_enabled()
            else None
        )
        dvk = None
        if (
            vk_producer_scale is not None
            and dq.stride(-1) == 1
            and (v_dim + k_dim) % 256 == 0
        ):
            from generative_recommenders.ops.triton.hstu_delta_bwd_e5m2 import (
                hstu_attention_delta_bwd_e5m2,
            )

            grad_dtype, grad_limit = bwd_fp8.grad_dtype_and_limit()
            rows = x.shape[0]
            pad = bwd_fp8.uvqk_tm_pad()
            dvk_fp8 = torch.empty(
                ((rows + pad - 1) // pad * pad, v_dim + k_dim),
                dtype=grad_dtype,
                device=x.device,
            )
            vk_amax, vk_colsum = hstu_attention_delta_bwd_e5m2(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dvk_fp8=dvk_fp8,
                rows=rows,
                seq_offsets=seq_offsets,
                N=ctx.max_seq_len,
                alpha=ctx.attn_alpha,
                inv_scale=vk_producer_scale[0],
                limit=grad_limit,
                k_scale=attention_vk_scale,
                v_scale=attention_vk_scale,
            )
            vk_prequant = (dvk_fp8, vk_amax, vk_colsum, rows)
            vk_cast_scale = vk_producer_scale[1]
            dvk_fp8 = None
            vk_amax = None
            vk_colsum = None
        else:
            # Write dV/dK directly in the packed column order consumed by the
            # full-row projection backward. The delta kernel honors row strides,
            # so this avoids an O(LD) torch.cat materialization.
            dvk = torch.empty(
                (x.shape[0], v_dim + k_dim),
                dtype=x.dtype,
                device=x.device,
            )
            dv = dvk[:, :v_dim].view(-1, ctx.num_heads, ctx.hidden_dim)
            dk = dvk[:, v_dim:].view(-1, ctx.num_heads, ctx.attn_dim)
            triton_hstu_attention_delta_bwd(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=seq_offsets,
                N=ctx.max_seq_len,
                alpha=ctx.attn_alpha,
                k_scale=attention_vk_scale,
                v_scale=attention_vk_scale,
            )
        vk_producer_scale = None

        du = torch.empty_like(u)
        torch.ops.aten.silu_backward(dsilu_u, u, grad_input=du)
        duq = torch.cat((du, dq.flatten(1, 2)), dim=1)
        with record_function("## hstu_targets_compact_uq_backward ##"):
            d_candidate_x, d_w_uq, d_b_uq = triton_addmm_bwd(
                x=candidate_normed_x,
                w=w_uq,
                dz=duq,
                is_y_1d=True,
            )
        if ctx.fp8_input_forward:
            # The full-row FP8 normalized input and V/K pack are dead once
            # attention backward has produced dV/dK. Release them before
            # materializing the BF16 normalized input needed by projection
            # backward so the two representations do not overlap at peak HBM.
            q = None
            k = None
            v = None
            vk = None
            vk_fp8 = None
            normed_x_fp8 = None
            if saved_normed_x_bf16 is not None:
                normed_x = saved_normed_x_bf16
                saved_normed_x_bf16 = None
            elif vk_fp8t_will_run:
                # fp8t: token-major E4M3 normed_x only (forward-stored under
                # STORE_T, else recomputed); a later decline rebuilds BF16.
                if stored_vk_x_fp8_t is not None:
                    vk_x_fp8_t = stored_vk_x_fp8_t
                    bwd_fp8.log_uvqk_store_t_once("vk_projection", True)
                else:
                    if bwd_fp8.uvqk_store_fp8t():
                        bwd_fp8.log_uvqk_store_t_once("vk_projection", False)
                    _, vk_x_fp8_t = triton_weighted_layer_norm_recompute_fp8t(
                        x=x,
                        weight=norm_weight,
                        bias=norm_bias,
                        eps=ctx.norm_eps,
                        mean=x_mean,
                        rstd=x_rstd,
                        inv_scale=1.0 / ctx.fp8_input_scale_value,
                        pad_to=bwd_fp8.uvqk_tm_pad(),
                        tiled_xt=wgrad_ck.tiled_active(x.device),
                    )
            else:
                normed_x, _, _ = triton_weighted_layer_norm_fwd(
                    x=x,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=ctx.norm_eps,
                    mean=x_mean,
                    rstd=x_rstd,
                )
        stored_vk_x_fp8_t = None
        assert normed_x is not None or vk_x_fp8_t is not None
        with record_function("## hstu_targets_full_vk_backward ##"):
            fp8_grads = None
            if vk_fp8_bwd:

                def _recompute_normed_x() -> torch.Tensor:
                    return triton_weighted_layer_norm_fwd(
                        x=x,
                        weight=norm_weight,
                        bias=norm_bias,
                        eps=ctx.norm_eps,
                        mean=x_mean,
                        rstd=x_rstd,
                    )[0]

                # One such layer per step, and its weight is a per-call packed
                # copy, so the scale keys on the site (per_layer=False).
                fp8_grads = bwd_fp8.uvqk_projection_backward(
                    dz=dvk,
                    normed_x=normed_x,
                    input_scale_value=ctx.fp8_input_scale_value,
                    weight=w_vk,
                    site="vk_projection",
                    per_layer=False,
                    normed_x_fp8_t=vk_x_fp8_t,
                    recompute_normed_x=_recompute_normed_x,
                    prequant=vk_prequant,
                )
                vk_x_fp8_t = None
                if fp8_grads is None and dvk is None:
                    # The producer already cast dz and the FP8 projection
                    # backward declined/failed (first-call failures only):
                    # lossy BF16 decode for the shipped path.
                    assert vk_prequant is not None
                    dvk = bwd_fp8.dequantize_prequant(
                        vk_prequant[0], vk_prequant[3], vk_cast_scale
                    )
                vk_prequant = None
                if fp8_grads is None and normed_x is None:
                    # Declined after the BF16 normed_x was skipped for it.
                    normed_x = _recompute_normed_x()
            if fp8_grads is not None:
                d_normed_x, d_w_vk, d_b_vk = fp8_grads
                d_w_vk = d_w_vk.to(d_w_uq.dtype)
                d_b_vk = d_b_vk.to(d_b_uq.dtype)
            else:
                d_normed_x, d_w_vk, d_b_vk = triton_addmm_bwd(
                    x=normed_x,
                    w=w_vk,
                    dz=dvk,
                    is_y_1d=True,
                )
            fp8_grads = None
        d_normed_x.index_add_(0, candidate_rows, d_candidate_x)

        d_w_u, d_w_q = d_w_uq.split([u_dim, q_dim], dim=1)
        d_w_v, d_w_k = d_w_vk.split([v_dim, k_dim], dim=1)
        d_uvqk_weight = torch.cat((d_w_u, d_w_v, d_w_q, d_w_k), dim=1)
        d_b_u, d_b_q = d_b_uq.split([u_dim, q_dim], dim=0)
        d_b_v, d_b_k = d_b_vk.split([v_dim, k_dim], dim=0)
        d_uvqk_bias = torch.cat((d_b_u, d_b_v, d_b_q, d_b_k), dim=0)
        d_uvqk_weight = d_uvqk_weight.to(uvqk_weight.dtype)
        d_uvqk_bias = d_uvqk_bias.to(uvqk_bias.dtype)

        d_x, d_norm_weight, d_norm_bias = triton_weighted_layer_norm_bwd(
            dy=d_normed_x,
            x=x,
            weight=norm_weight,
            bias=norm_bias,
            mean=x_mean,
            rstd=x_rstd,
            learnable=True,
            eps=ctx.norm_eps,
            BLOCK_D=ctx.norm_BLOCK_D,
            residual=None if ctx.residual_rows_only else dresidual,
        )
        if ctx.residual_rows_only and dresidual is not None:
            # Same value as the dense ADD_RESIDUAL epilogue: that computes
            # bf16(f32(bf16(dx)) + f32(res)), and a bf16 index_add on the
            # stored bf16 dx does the same single fp32 add + RNE round. Rows
            # outside candidate_rows got res == +0 there, which is the identity
            # except for the sign of an exact -0.0 dx (numerically equal).
            d_x.index_add_(0, seq_offsets[1:] - 1, dresidual.to(d_x.dtype))
            if not _TGT_SPARSE_LOGGED[0]:
                _TGT_SPARSE_LOGGED[0] = True
                logging.getLogger(__name__).warning(
                    "[hstu] targets-only row-sparse residual gradient ACTIVE "
                    "(DLRM_HSTU_FP8_TGT_SPARSE_RESID): rows=%d of %d",
                    int(dresidual.shape[0]), int(d_x.shape[0]))
        return (
            d_x,
            d_norm_weight,
            d_norm_bias,
            None,
            None,
            None,
            None,
            d_uvqk_weight,
            d_uvqk_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # residual_rows_only
            None,  # allow_save_fp8_vk
        )


def triton_hstu_preprocess_and_attention(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    attn_alpha: float,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    recompute_uvqk_in_backward: bool = False,
    recompute_normed_x_in_backward: bool = False,
    sort_by_length: bool = False,
    enable_tma: Optional[bool] = None,
    fp8_input_scale: Optional[torch.Tensor] = None,
    fp8_qkv_scale: Optional[torch.Tensor] = None,
    defer_silu: bool = False,
    allow_save_fp8_uvqk: bool = True,
    return_residual_proxy: bool = False,
):
    # When the caller does not specify enable_tma, auto-detect whether the
    # TMA / TLX fast path is safe on this device. Resolving here (vs inside
    # the autograd Function.forward) keeps a concrete bool flowing through
    # ctx.save_for_backward / ctx attributes.
    if enable_tma is None:
        enable_tma = _should_enable_tma()
    outputs = _HSTUPreprocessAndAttentionFunction.apply(
        x,
        norm_weight,
        norm_bias,
        norm_eps,
        num_heads,
        attn_dim,
        hidden_dim,
        uvqk_weight,
        uvqk_bias,
        max_seq_len,
        seq_offsets,
        attn_alpha,
        num_targets,
        max_attn_len,
        contextual_seq_len,
        recompute_uvqk_in_backward,
        recompute_normed_x_in_backward,
        sort_by_length,
        enable_tma,
        fp8_input_scale,
        fp8_qkv_scale,
        defer_silu,
        allow_save_fp8_uvqk,
        0,  # num_softmax_heads
        # The fold needs this Function to be x's only consumer, which holds
        # when the residual proxy (not x itself) feeds the residual add.
        (
            take_dropout_handoff(x)
            if return_residual_proxy and input_dropout_fold_enabled()
            else None
        ),
        # r14 E-RM (DLRM_HSTU_FP8_BWD_ERM, default off): the bottom layer's
        # out-proj site tagged on x, claimed under the same only-consumer
        # condition (dy_erm.claim: None at its first test with the knob off;
        # also None without a residual proxy or a possible backward).
        _dy_erm().claim(x, residual_proxy=return_residual_proxy, targets_only=False),
    )
    return outputs if return_residual_proxy else outputs[:2]


def triton_hstu_preprocess_and_attention_targets_only(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    attn_alpha: float,
    num_targets: torch.Tensor,
    enable_tma: Optional[bool] = None,
    fp8_input_scale: Optional[torch.Tensor] = None,
    fp8_qkv_scale: Optional[torch.Tensor] = None,
    return_residual_proxy: bool = False,
    residual_rows_only: bool = False,
    allow_save_fp8_vk: bool = True,
):
    if enable_tma is None:
        enable_tma = _should_enable_tma()
    outputs = _HSTUPreprocessAndAttentionTargetsOnlyFunction.apply(
        x,
        norm_weight,
        norm_bias,
        norm_eps,
        num_heads,
        attn_dim,
        hidden_dim,
        uvqk_weight,
        uvqk_bias,
        max_seq_len,
        seq_offsets,
        attn_alpha,
        num_targets,
        enable_tma,
        fp8_input_scale,
        fp8_qkv_scale,
        residual_rows_only and return_residual_proxy,
        allow_save_fp8_vk,
    )
    return outputs if return_residual_proxy else outputs[:2]
