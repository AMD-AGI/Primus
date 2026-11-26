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

"""gfx950 FP8 helpers for the fused HSTU training path.

The public helpers in this module consume activations that were already
quantized by a producer epilogue. They intentionally do not provide a
boundary-cast convenience path: Plan 01 proved that such a path is slower than
BF16 at the production shapes.
"""

from __future__ import annotations

import collections
import functools
import json
import logging
import math
import os
import sys
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from generative_recommenders.ops import blas_solution_pins
from torch.autograd.profiler import record_function


logger: logging.Logger = logging.getLogger(__name__)

_FP8_E4M3: Optional[torch.dtype] = getattr(torch, "float8_e4m3fn", None)
_FP8_MAX: float = 448.0

# The master switch. Nothing here gates on it -- _env_bootstrap has already
# expanded it into the individual knobs below by the time this imports -- but it
# is logged, so a run's own log says whether the arm was selected as a unit or
# assembled knob by knob.
_HSTU_FP8_FWD: bool = os.environ.get("DLRM_HSTU_FP8_FWD", "0") == "1"
_HSTU_FP8_GEMM: bool = os.environ.get("DLRM_HSTU_FP8_GEMM", "0") == "1"
_HSTU_FP8_FAST_ACCUM: bool = (
    os.environ.get("DLRM_HSTU_FP8_GEMM_FASTACC", "0") == "1"
)
_HSTU_FP8_FAIL_CLOSED: bool = (
    os.environ.get("DLRM_HSTU_FP8_FAIL_CLOSED", "0") == "1"
)
# FP8 output-scale mode (see _fp8_output_scale_mode). Unset or "input_scale": fold every
# output scale into scale_a, the code default on every stack. "auto": probe how this backend
# applies scale_result (opt-in, for an A/B of hipBLASLt epilogue scaling). Validated here so
# a bad value fails loudly at import instead of inside fp8_supported's catch-all.
_FP8_OUTPUT_SCALE_MODE: str = (
    os.environ.get("DLRM_HSTU_FP8_OUTPUT_SCALE_MODE", "").strip().lower()
)
if _FP8_OUTPUT_SCALE_MODE not in ("", "input_scale", "auto"):
    raise ValueError(
        f"DLRM_HSTU_FP8_OUTPUT_SCALE_MODE={_FP8_OUTPUT_SCALE_MODE!r}: "
        "only 'input_scale' (default) or 'auto'"
    )
_HSTU_FP8_HOST_SCALE_VALUES: bool = (
    _HSTU_FP8_GEMM
    and os.environ.get("DLRM_HSTU_FP8_HOST_SCALE_VALUES", "0") == "1"
)
_HSTU_FP8_FUSE_LN: bool = (
    _HSTU_FP8_GEMM
    and os.environ.get("DLRM_HSTU_FP8_FUSE_LN", "0") == "1"
)
_HSTU_FP8_FUSE_OUTLN: bool = (
    _HSTU_FP8_GEMM
    and os.environ.get("DLRM_HSTU_FP8_FUSE_OUTLN", "0") == "1"
)
_HSTU_FP8_FUSE_QKV: bool = (
    _HSTU_FP8_FUSE_LN
    and os.environ.get("DLRM_HSTU_FP8_FUSE_QKV", "0") == "1"
)
_HSTU_FP8_ATTENTION: bool = (
    _HSTU_FP8_FUSE_QKV
    and os.environ.get("DLRM_HSTU_FP8_ATTN", "0") == "1"
)
_HSTU_FP8_BACKWARD: bool = (
    _HSTU_FP8_GEMM
    and os.environ.get("DLRM_HSTU_FP8_BWD", "0") == "1"
)
_HSTU_FP8_FUSE_SILU: bool = (
    _HSTU_FP8_GEMM
    and os.environ.get("DLRM_HSTU_FP8_FUSE_SILU", "0") == "1"
)
_HSTU_FP8_FUSE_RESIDUAL: bool = (
    _HSTU_FP8_GEMM
    and os.environ.get("DLRM_HSTU_FP8_FUSE_RESIDUAL", "0") == "1"
)
_HSTU_FP8_FUSE_RESIDUAL_ALGORITHM: int = int(
    os.environ.get("DLRM_HSTU_FP8_FUSE_RESIDUAL_ALGORITHM", "1")
)
_HSTU_FP8_FUSE_RESIDUAL_WORKSPACE_BYTES: int = 64 * 1024 * 1024
_HSTU_FP8_ASCALE_MARGIN: float = float(
    os.environ.get("DLRM_HSTU_FP8_GEMM_ASCALE_MARGIN", "1.5")
)
_HSTU_FP8_CALIBRATION_STEPS: int = max(
    1, int(os.environ.get("DLRM_HSTU_FP8_CALIBRATION_STEPS", "4"))
)
_HSTU_FP8_SCALE_MODE: str = os.environ.get(
    "DLRM_HSTU_FP8_SCALE_MODE", "frozen"
)
_HSTU_FP8_TELEMETRY_INTERVAL: int = max(
    0, int(os.environ.get("DLRM_HSTU_FP8_TELEMETRY_INTERVAL", "0"))
)
# K=512 FP8 forward projection GEMM (ops/hip/k512_fp8_gemm.py) for the e4m3-out
# V/Q/K (N=1536) and targets V/K (N=1024) projections in scaled_mm_fp8_out_preq,
# instead of hipBLASLt. Default off (yambda_5b.gin item (43) turns it on). fp32
# summation order only: bit-identical to hipBLASLt's data-parallel tiles, while
# hipBLASLt's stream-K tail tiles add split-K partials in another order (unit
# test ops/tests/k512_fp8_gemm_test.py); anything it does not cover stays on
# hipBLASLt.
_HSTU_FP8_FWD_K512: bool = os.environ.get("DLRM_HSTU_FP8_FWD_K512", "0") == "1"
# B15T(L2), yambda_5b.gin item B15T(L2) (Arbor P7; code default off, the gin
# binds it on, "0" in the environment is the kill switch): the targets-only
# layer's V/K projection (N=1024, e4m3 out) reads its A from the step-tiled
# token-major yT that the Triton v2 store_t writes for the fp8t V/K wgrad, on
# the K512 library's tiled-A entry (ops/hip/k512_fp8_gemm.py), so that
# LayerNorm needs no [M, 512] y (store_y=False: the unmodified v2 kernel still
# WRITES y's bytes, but into a dedicated 1/32-size scratch that is discarded:
# fewer DRAM write-backs, not fewer stores). Bit-identical: yT holds y's
# bytes and the tiled-A entry equals the row-major K512 entry on y. Needs
# DLRM_HSTU_FP8_FWD_K512 (the row-major entry, not hipBLASLt, is what the tiled
# entry matches). Decided on the host before the LayerNorm
# (k512_tiled_a_vk_decline_reason); a decline keeps the row-major y and
# scaled_mm_fp8_out_preq.
_HSTU_FP8_FWD_K512_TILED_A: bool = (
    os.environ.get("DLRM_HSTU_FP8_FWD_K512_TILED_A", "0") == "1"
)
# K=1536 FP8 output-projection forward GEMM with the fused bf16 residual
# (ops/hip/k1536_fp8_addmm.py) in scaled_addmm_fp8_preq's residual-fused branch,
# instead of the hipBLASLt extension call. Default off (yambda_5b.gin item (44)
# turns it on). fp32 summation order only, as K512 (unit test
# ops/tests/k1536_fp8_addmm_test.py); anything it does not cover stays on
# hipBLASLt.
_HSTU_FP8_FWD_K1536: bool = os.environ.get("DLRM_HSTU_FP8_FWD_K1536", "0") == "1"
# MXFP6 forward K=512 trio (ops/hip/mx6_fwd.py; yambda_5b.gin item (92), DECLARED
# PRECISION CHANGE, default off): only its prebuild hook lives here.
_HSTU_FP8_FWD_MXFP6: bool = os.environ.get("DLRM_HSTU_FP8_FWD_MXFP6", "0") == "1"
# U-slice forward GEMM (QKV-fused path: e4m3 normed_x [M,512] x e4m3 W_u
# [512,512] + bf16 bias -> bf16) always on hipBLASLt's data-parallel grid.
# torch._scaled_mm hands hipBLASLt torch's process-wide 76 MiB workspace, and
# with that much room the stream-K selection picks a persistent 255/256-
# workgroup grid at about 60% of the in-model token counts. That grid's time
# is bimodal and GPU-dependent (about 3% faster than data-parallel when it runs
# right behind itself, 20-35% slower after a different kernel on some GPUs;
# in-model 17-31% slower on average), while the data-parallel grid it picks at
# the other token counts is stable. The choice is a deterministic function of M
# and the workspace size (the persistent grid's partial tiles need >= 63.75
# MiB), so the call sites that pass data_parallel=True go through
# hipblaslt_scaled_mm_bias.cpp -- the same hipBLASLt call torch makes, with a
# private DLRM_HSTU_FP8_FWD_U_DP_WS_MIB workspace. Same kernel, data-parallel
# grid; output bytes measured identical. Other GEMMs keep torch's workspace.
# Default off.
_HSTU_FP8_FWD_U_DP: bool = os.environ.get("DLRM_HSTU_FP8_FWD_U_DP", "0") == "1"
_HSTU_FP8_FWD_U_DP_WS_MIB: int = int(
    os.environ.get("DLRM_HSTU_FP8_FWD_U_DP_WS_MIB", "32")
)
if _HSTU_FP8_FWD_U_DP and _HSTU_FP8_FWD_U_DP_WS_MIB <= 0:
    raise ValueError(
        "DLRM_HSTU_FP8_FWD_U_DP_WS_MIB must be a positive number of MiB, got "
        f"{_HSTU_FP8_FWD_U_DP_WS_MIB}"
    )
_HSTU_FP8_FWD_U_DP_WS_BYTES: int = max(0, _HSTU_FP8_FWD_U_DP_WS_MIB) << 20
# Partial-tile buffer of the 255-workgroup persistent grid at the 256x256 macro
# tile (255 x 256 x 256 x 4 B). A workspace at least this large leaves the
# persistent grid selectable again.
_U_DP_PERSISTENT_PARTIALS_BYTES: int = 255 * 256 * 256 * 4


def _env_on(name: str) -> bool:
    """Default-on kill switch: off only for an explicit 0/false/off/no."""
    value = os.environ.get(name, "").strip().lower()
    return value not in ("0", "false", "off", "no")


def _env_true(name: str) -> bool:
    """Default-off gate: on only for an explicit 1/true/on/yes (gin binding)."""
    value = os.environ.get(name, "").strip().lower()
    return value in ("1", "true", "on", "yes")


# Fused weight-shadow refresh (iter63 small-kill (a)). Default OFF in code;
# the gin binding apply_env_bootstrap.DLRM_HSTU_FP8_WSHADOW_FUSED turns it on
# (BATCH is a refinement, default on once FUSED is on). The aten
# chain below (_pack_weight_layout + calibrated_scale + quantize_with_scale)
# costs ~10 launches per (weight, layout) -- 72 launches, 0.43-0.56 ms (mean
# 0.51) per rank-step on the main stream for the 7 shadows of the 3-layer model
# (ship10 1n traces, iter-63 bench TB1/TB2, ranks 0-7, steps 53-56).
# triton_fp8_wshadow.refresh_shadows produces the same bytes (same
# amax, same scale formula -- including aten's multiply-by-(float)(1/448.0)
# for the CPU-scalar division -- same IEEE division, clamp and software
# RNE e4m3 cast) in two launches. With BATCH on, the first stale shadow of a
# step also refreshes every other stale shadow seen before on the same device,
# dtype and stream; each still re-validates its weight version when used.
# Fail-closed: if the fused path raises (Triton compile/launch), it is disabled
# for the rest of the process with one warning and the aten chain below takes
# over (bitwise by construction); no other lowering is ever substituted.
# Kill switches (forwarded by launch_slurm.sh's DLRM_HSTU_FP8_ prefix rule):
#   DLRM_HSTU_FP8_WSHADOW_FUSED=0  restores the aten chain verbatim
#   DLRM_HSTU_FP8_WSHADOW_BATCH=0  keeps the fused kernels, one shadow per call
# Arming line (rank log, first fused refresh): "DLRMV4 armed OK: fp8 wshadow
# fused (batch=1|0)"; then, with BATCH on, one "DLRMV4 fp8 wshadow batch: N
# weight shadows refreshed in one call" at the first multi-shadow batch.
_HSTU_FP8_WSHADOW_FUSED: bool = _env_true("DLRM_HSTU_FP8_WSHADOW_FUSED")
_HSTU_FP8_WSHADOW_BATCH: bool = _env_on("DLRM_HSTU_FP8_WSHADOW_BATCH")


@dataclass
class _WeightCacheEntry:
    ref: weakref.ReferenceType[torch.Tensor]
    version: int
    data_ptr: int
    shape: torch.Size
    quantized_column_major: torch.Tensor
    scale: torch.Tensor
    # Batch-refresh bookkeeping (fused path); None on entries it must not touch.
    layout: Optional[Tuple[Tuple[int, int], ...]] = None
    stream_id: Optional[int] = None


@dataclass
class _PackedWeightCacheEntry:
    ref: weakref.ReferenceType[torch.Tensor]
    version: int
    data_ptr: int
    packed: torch.Tensor


_WeightLayout = Tuple[Tuple[int, int], ...]
_weight_cache: Dict[Tuple[object, ...], _WeightCacheEntry] = {}
_packed_weight_cache: Dict[Tuple[object, ...], _PackedWeightCacheEntry] = {}
_cache_hits: int = 0
_cache_misses: int = 0
# Shadows refreshed by the fused path as members of another call's batch.
_wshadow_batched: int = 0
_wshadow_armed_logged: bool = False
_wshadow_batch_logged: bool = False
_wshadow_module_ref: Optional[object] = None
_fp8_one_by_device: Dict[int, torch.Tensor] = {}
_scale_value_cache: Dict[Tuple[object, ...], float] = {}
_inverse_scale_cache: Dict[
    Tuple[object, ...], Tuple[float, torch.Tensor]
] = {}
_folded_input_scale_cache: Dict[Tuple[object, ...], torch.Tensor] = {}
_scaled_output_bias_cache: Dict[Tuple[object, ...], torch.Tensor] = {}
# Weak reference to the bias tensor each _scaled_output_bias_cache entry was computed from (same keys). A hit is
# valid only while that very tensor is alive: the key is (id, data_ptr, offset, _version), and a freed temporary
# (e.g. the targets-only site's torch.cat of the V/K bias slices, whose _version is always 0) can hand its id and
# allocator block to a later tensor with different values. Entries whose tensor is gone are purged.
_scaled_output_bias_refs: Dict[Tuple[object, ...], "weakref.ReferenceType[torch.Tensor]"] = {}
_SCALED_BIAS_PURGE_AT = 64
# Kill switch DLRM_HSTU_FP8_BIAS_CACHE_STRICT (gin binding apply_env_bootstrap.DLRM_HSTU_FP8_BIAS_CACHE_STRICT; forwarded
# by launch_slurm.sh's DLRM_HSTU_FP8_ prefix rule): unset or "1" = the weakref-validated hits and the purge above;
# "0" = the legacy cache verbatim (a key hit is returned unchecked, nothing is purged). Read on the first
# _scale_bias_for_fp8_output call, never at import (this module can be imported before the env bootstrap runs),
# cached for the process; one line per process: "[fp8-bias-cache] strict: ..." or "[fp8-bias-cache] legacy ...".
_scaled_bias_strict: Optional[bool] = None
_fp8_output_scale_mode_by_device: Dict[int, str] = {}
_telemetry_calls: Dict[str, int] = {}
_telemetry_last: Dict[str, dict[str, float | int | bool]] = {}
_capability_sites: set[str] = set()
_resolved_logged: bool = False
_capability_contracts: Dict[str, dict[str, object]] = {}
_residual_workspaces: Dict[int, torch.Tensor] = {}
_u_dp_workspaces: Dict[int, torch.Tensor] = {}
_u_dp_logged: set[Tuple[str, str]] = set()


def _u_dp_print(message: str) -> None:
    print(f"[hstu-fp8-fwd] {message}", flush=True)


if _HSTU_FP8_FWD_U_DP:
    # Startup record of what this process requested (the ARMED line follows on
    # the first engaged call per device).
    _u_dp_print(
        f"requested: u_dp=1 workspace={_HSTU_FP8_FWD_U_DP_WS_MIB} MiB"
        + (
            " -> INERT without DLRM_HSTU_FP8_FUSE_QKV=1 (the U slice exists "
            "only on the QKV-fused forward)"
            if not _HSTU_FP8_FUSE_QKV
            else ""
        )
        + (
            " -> WARNING workspace >= 63.75 MiB, the persistent grid stays "
            "selectable"
            if _HSTU_FP8_FWD_U_DP_WS_BYTES >= _U_DP_PERSISTENT_PARTIALS_BYTES
            else ""
        )
    )


def hstu_fp8_gemm_enabled() -> bool:
    return _HSTU_FP8_GEMM


def hstu_fp8_fail_closed() -> bool:
    return _HSTU_FP8_FAIL_CLOSED


def hstu_fp8_fuse_ln_enabled() -> bool:
    return _HSTU_FP8_FUSE_LN


def hstu_fp8_fuse_outln_enabled() -> bool:
    return _HSTU_FP8_FUSE_OUTLN


def hstu_fp8_fuse_qkv_enabled() -> bool:
    return _HSTU_FP8_FUSE_QKV


def hstu_fp8_attention_enabled() -> bool:
    return _HSTU_FP8_ATTENTION


def refuse_fp8_with_flydsl_attention_forward() -> None:
    """No-op. Kept so old call sites and tests still import a named gate.

    ``DLRM_HSTU_FP8_FWD=1`` plus ``HSTU_ATTN_FWD=flydsl`` (the gin default)
    selects the FlyDSL e4m3 attention forward (fused-silu epilogue included).
    ``HSTU_ATTN_FWD=triton`` is the opt-out to the Triton e4m3 forward.
    Backward selection (``HSTU_ATTN_BWD``) is independent and unchanged.
    """
    return


def hstu_fp8_backward_enabled() -> bool:
    return _HSTU_FP8_BACKWARD


def hstu_fp8_fuse_silu_enabled() -> bool:
    return _HSTU_FP8_FUSE_SILU


def hstu_fp8_fuse_residual_enabled() -> bool:
    return _HSTU_FP8_FUSE_RESIDUAL


def hstu_fp8_scale_mode() -> str:
    return _HSTU_FP8_SCALE_MODE


def fp8_e4m3_dtype() -> torch.dtype:
    if _FP8_E4M3 is None:
        raise RuntimeError("torch.float8_e4m3fn is unavailable")
    return _FP8_E4M3


def fp8_scale_state_size() -> int:
    """Scale, completed-update count, and bounded amax history."""

    return 2 + _HSTU_FP8_CALIBRATION_STEPS


def fp8_scale_tensor(scale_state: torch.Tensor) -> torch.Tensor:
    if scale_state.numel() < 1:
        raise ValueError("FP8 scale state cannot be empty")
    return scale_state.reshape(-1)[:1].reshape(1, 1)


def fp8_supported(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    *,
    input_columns: Optional[int] = None,
) -> bool:
    if (
        not _HSTU_FP8_GEMM
        or _FP8_E4M3 is None
        or torch.version.hip is None
        or not hasattr(torch, "_scaled_mm")
        or not x.is_cuda
        or x.dim() != 2
    ):
        return False
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if arch != "gfx950":
        return False
    try:
        _fp8_output_scale_mode(x.device)
    except Exception:
        return False
    if weight is None:
        return x.shape[1] % 16 == 0
    k = x.shape[1] if input_columns is None else input_columns
    return (
        weight.is_cuda
        and weight.dim() == 2
        and k == weight.shape[0]
        and k % 16 == 0
        and weight.shape[1] % 16 == 0
    )


def calibrated_scale(
    value: torch.Tensor,
    *,
    margin: Optional[float] = None,
    fp8_max: float = _FP8_MAX,
) -> torch.Tensor:
    """Return a scalar FP32 dequantization scale.

    This helper is for bounded calibration/diagnostics. It must not be called
    on every steady-state HSTU invocation.
    """

    scale_margin = _HSTU_FP8_ASCALE_MARGIN if margin is None else margin
    # Reduce in the producer dtype and cast only the scalar. Casting a
    # production [M, D] activation to FP32 here would add a multi-GiB warmup
    # allocation without improving an amax statistic.
    amax = value.detach().abs().amax().float().clamp_min(1.0e-12)
    return (amax * scale_margin / fp8_max).reshape(1, 1)


def _scale_key(scale: torch.Tensor) -> Tuple[object, ...]:
    device_index = scale.device.index
    if device_index is None:
        device_index = -1
    return (
        id(scale),
        device_index,
        scale.untyped_storage().data_ptr(),
        scale.storage_offset(),
        int(getattr(scale, "_version", 0)),
    )


def activation_scale_value(scale: Optional[torch.Tensor]) -> Optional[float]:
    """Return a cached positive scale without a steady-state D2H sync."""

    if scale is None or scale.numel() < 1:
        return None
    key = _scale_key(scale)
    cached = _scale_value_cache.get(key)
    if cached is not None:
        return cached
    value = float(fp8_scale_tensor(scale).detach().item())
    if not math.isfinite(value) or value <= 0.0:
        return None
    _scale_value_cache[key] = value
    return value


def _host_scale_cache_key(
    kind: str,
    device: torch.device,
    *values: float,
) -> Tuple[object, ...]:
    device_index = device.index
    if device_index is None:
        device_index = -1
    stream_id = -1
    if device.type == "cuda":
        stream_id = int(torch.cuda.current_stream(device).cuda_stream)
    return (kind, device.type, device_index, stream_id, *values)


def _trusted_host_scale_values(
    scale_a: torch.Tensor,
    output_scale: torch.Tensor,
    scale_a_value: Optional[float],
    output_scale_value: Optional[float],
) -> Optional[Tuple[float, float]]:
    if not _HSTU_FP8_HOST_SCALE_VALUES:
        return None
    valid = (
        scale_a.device == output_scale.device
        and isinstance(scale_a_value, (int, float))
        and not isinstance(scale_a_value, bool)
        and isinstance(output_scale_value, (int, float))
        and not isinstance(output_scale_value, bool)
    )
    if valid:
        scale_a_host = float(scale_a_value)
        output_scale_host = float(output_scale_value)
        valid = (
            math.isfinite(scale_a_host)
            and scale_a_host > 0.0
            and math.isfinite(output_scale_host)
            and output_scale_host > 0.0
        )
    if valid:
        return (scale_a_host, output_scale_host)
    if _HSTU_FP8_FAIL_CLOSED:
        raise RuntimeError(
            "DLRM_HSTU_FP8_HOST_SCALE_VALUES requires finite positive "
            "host values for scale_a and output_scale on the same device"
        )
    return None


def _check_scale_a_honoured(a, b, one, two, dtype, observed_scale_result=None) -> None:
    """The input_scale fold relies on scale_a; refuse a backend that ignores it."""

    scale_a_observed = float(
        _scaled_mm(
            a,
            b,
            two,
            one,
            out_dtype=dtype,
        )[0, 0]
        .float()
        .item()
    )
    if not math.isclose(scale_a_observed, 2.0, rel_tol=0.0, abs_tol=0.01):
        seen = "" if observed_scale_result is None else f"scale_result {observed_scale_result} and "
        raise RuntimeError(
            "torch._scaled_mm ignored FP8 scale_a, so the input_scale fold is invalid: "
            f"observed {seen}scale_a {scale_a_observed}"
        )


def _fp8_output_scale_mode(device: torch.device) -> str:
    """How FP8 output scales are applied: "input_scale" (fold into scale_a), or with
    DLRM_HSTU_FP8_OUTPUT_SCALE_MODE=auto whatever scale_result does on this backend."""

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    cached = _fp8_output_scale_mode_by_device.get(device_index)
    if cached is not None:
        return cached
    # Code default on every stack: fold every FP8 output scale into scale_a. It never relies
    # on scale_result, so it is valid whatever the backend does with it; it is what ROCm 7.2
    # gives anyway (its hipBLASLt ignores scale_result), and the K512 / MX6 forward kernels
    # implement only this fold. ROCm 10.2's hipBLASLt honours scale_result: probing there
    # answers "multiply" and those kernels would decline, so the probe is opt-in ("auto")
    # and the choice no longer depends on an image environment variable.
    dtype = fp8_e4m3_dtype()
    a = torch.ones((16, 16), dtype=torch.bfloat16, device=device).to(dtype)
    b = torch.eye(16, dtype=torch.bfloat16, device=device).to(dtype).t()
    one = torch.ones((1, 1), dtype=torch.float32, device=device)
    two = torch.full((1, 1), 2.0, dtype=torch.float32, device=device)
    if _FP8_OUTPUT_SCALE_MODE != "auto":
        _check_scale_a_honoured(a, b, one, two, dtype)
        _fp8_output_scale_mode_by_device[device_index] = "input_scale"
        return "input_scale"
    observed = float(
        _scaled_mm(
            a,
            b,
            one,
            one,
            out_dtype=dtype,
            scale_result=two,
        )[0, 0]
        .float()
        .item()
    )
    if math.isclose(observed, 2.0, rel_tol=0.0, abs_tol=0.01):
        mode = "multiply"
    elif math.isclose(observed, 0.5, rel_tol=0.0, abs_tol=0.01):
        mode = "divide"
    elif math.isclose(observed, 1.0, rel_tol=0.0, abs_tol=0.01):
        _check_scale_a_honoured(a, b, one, two, dtype, observed_scale_result=observed)
        mode = "input_scale"
    else:
        raise RuntimeError(
            "torch._scaled_mm ignored or misapplied FP8 scale_result: "
            f"expected 2.0 or 0.5, observed {observed}"
        )
    _fp8_output_scale_mode_by_device[device_index] = mode
    return mode


def inverse_scale_tensor(
    scale: torch.Tensor,
    *,
    value: Optional[float] = None,
) -> torch.Tensor:
    """Return the backend-specific result scalar for dequant scale ``scale``."""

    if value is None:
        value = activation_scale_value(scale)
        key = _scale_key(scale)
    else:
        key = _host_scale_cache_key("inverse", scale.device, value)
    if value is None or not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("inverse requested before FP8 scale calibration")
    cached = _inverse_scale_cache.get(key)
    if cached is not None and cached[0] == value:
        return cached[1]
    mode = _fp8_output_scale_mode(scale.device)
    if mode == "multiply":
        result_value = 1.0 / value
    elif mode == "divide":
        result_value = value
    else:
        raise RuntimeError(
            "scale_result is unavailable; fold the output scale into scale_a"
        )
    inverse = torch.full(
        (1, 1),
        result_value,
        dtype=torch.float32,
        device=scale.device,
    )
    _inverse_scale_cache[key] = (value, inverse)
    return inverse


def _fold_output_scale_into_input(
    scale_a: torch.Tensor,
    output_scale: torch.Tensor,
    *,
    scale_a_value: Optional[float] = None,
    output_scale_value: Optional[float] = None,
) -> torch.Tensor:
    if scale_a_value is None or output_scale_value is None:
        key = (*_scale_key(scale_a), *_scale_key(output_scale))
        scale_a_value = activation_scale_value(scale_a)
        output_scale_value = activation_scale_value(output_scale)
    else:
        key = _host_scale_cache_key(
            "folded",
            scale_a.device,
            scale_a_value,
            output_scale_value,
        )
    cached = _folded_input_scale_cache.get(key)
    if cached is not None:
        return cached
    if (
        scale_a_value is None
        or output_scale_value is None
        or not math.isfinite(scale_a_value)
        or scale_a_value <= 0.0
        or not math.isfinite(output_scale_value)
        or output_scale_value <= 0.0
    ):
        raise RuntimeError("FP8 output requested before scale calibration")
    folded = torch.full(
        (1, 1),
        scale_a_value / output_scale_value,
        dtype=torch.float32,
        device=scale_a.device,
    )
    _folded_input_scale_cache[key] = folded
    return folded


def _scaled_bias_cache_strict() -> bool:
    global _scaled_bias_strict
    if _scaled_bias_strict is None:
        _scaled_bias_strict = _env_on("DLRM_HSTU_FP8_BIAS_CACHE_STRICT")
        if _scaled_bias_strict:
            print(
                "[fp8-bias-cache] strict: weakref-validated hits, purge at "
                f"{_SCALED_BIAS_PURGE_AT}",
                flush=True,
            )
        else:
            print("[fp8-bias-cache] legacy (DLRM_HSTU_FP8_BIAS_CACHE_STRICT=0)", flush=True)
    return _scaled_bias_strict


def _scale_bias_for_fp8_output(
    bias: torch.Tensor,
    output_scale: torch.Tensor,
    *,
    output_scale_value: Optional[float] = None,
) -> torch.Tensor:
    if output_scale_value is None:
        key = (*_scale_key(bias), *_scale_key(output_scale))
        output_scale_value = activation_scale_value(output_scale)
    else:
        key = (
            *_scale_key(bias),
            *_host_scale_cache_key(
                "scaled_bias",
                bias.device,
                output_scale_value,
            ),
        )
    strict = _scaled_bias_cache_strict()
    cached = _scaled_output_bias_cache.get(key)
    if cached is not None:
        if not strict:  # DLRM_HSTU_FP8_BIAS_CACHE_STRICT=0: the legacy unchecked hit
            return cached
        ref = _scaled_output_bias_refs.get(key)
        if ref is not None and ref() is bias:
            return cached
        # The tensor this entry was computed from is gone (or unknown): the key was
        # reused by a different tensor, so the cached values may be stale.
        _scaled_output_bias_cache.pop(key, None)
        _scaled_output_bias_refs.pop(key, None)
    if (
        output_scale_value is None
        or not math.isfinite(output_scale_value)
        or output_scale_value <= 0.0
    ):
        raise RuntimeError("FP8 bias requested before output scale calibration")
    scaled = (bias.detach().float() / output_scale_value).to(torch.bfloat16)
    if not strict:  # DLRM_HSTU_FP8_BIAS_CACHE_STRICT=0: the legacy store (no ref, no purge)
        _scaled_output_bias_cache[key] = scaled
        return scaled
    if len(_scaled_output_bias_cache) >= _SCALED_BIAS_PURGE_AT:
        # purge entries whose tensor is gone or has been updated since (key[4] is its _version)
        for dead in [k for k, r in _scaled_output_bias_refs.items()
                     if r() is None or int(getattr(r(), "_version", 0)) != k[4]]:
            _scaled_output_bias_cache.pop(dead, None)
            _scaled_output_bias_refs.pop(dead, None)
    _scaled_output_bias_cache[key] = scaled
    _scaled_output_bias_refs[key] = weakref.ref(bias)
    return scaled


def _spf_snap_published(
    scale: torch.Tensor, spf_alpha: Optional[float]
) -> torch.Tensor:
    """DLRM_HSTU_FP8_LAB_FWD_SPF (ship17 item (105), gin-bound ON; code
    default off; ops/fp8_qkv_spf.py): with the knob at "snap" or "1", the
    attention qkv scale being published is snapped UP onto the SPF grid (the
    smallest f32 s >= it whose fwd_kernel_0 score coefficient
    fl(fl(fl(s*s)*alpha)*log2e) is exactly 2**j). Returns `scale` itself when
    the knob is off or the buffer is not a qkv scale (spf_alpha None). An
    invalid knob value raises ValueError here. One INFO line per publication
    (every rank, once per layer): the arming proof of the real snapped scale;
    calibration already syncs the host here (.item())."""
    if spf_alpha is None:
        return scale
    from generative_recommenders.ops import fp8_qkv_spf

    if fp8_qkv_spf.spf_mode() == "off":
        return scale
    raw = float(scale.item())
    snapped, j = fp8_qkv_spf.snap_scale(raw, float(spf_alpha))
    logger.info(
        "[hstu-fp8] %s: qkv scale snapped onto the SPF grid at publication: %s",
        fp8_qkv_spf.KNOB, fp8_qkv_spf.describe(raw, snapped, j))
    return torch.full_like(scale, snapped)


def calibrate_scale_buffer(
    scale_buffer: torch.Tensor,
    value: torch.Tensor,
    *,
    margin: Optional[float] = None,
    synchronize: bool = True,
    spf_alpha: Optional[float] = None,
) -> float:
    """Update bounded warmup state and publish a frozen scale when complete.

    `spf_alpha` marks the attention qkv scale (its attention alpha): with
    DLRM_HSTU_FP8_LAB_FWD_SPF at "snap" or "1" (item (105)) the published
    value is snapped onto the SPF grid (_spf_snap_published); otherwise it is
    unused."""

    if _HSTU_FP8_SCALE_MODE != "frozen":
        raise RuntimeError(
            f"unsupported Plan 08 FP8 scale mode: {_HSTU_FP8_SCALE_MODE}"
        )
    scale_margin = _HSTU_FP8_ASCALE_MARGIN if margin is None else margin
    amax = (
        value.detach().abs().amax().float()
        if value.numel()
        else torch.zeros((), dtype=torch.float32, device=scale_buffer.device)
    )
    if (
        synchronize
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX)
    if float(amax.item()) <= 0.0:
        existing = activation_scale_value(scale_buffer)
        return 0.0 if existing is None else existing
    amax.clamp_min_(1.0e-12)

    flat_state = scale_buffer.reshape(-1)
    if flat_state.numel() == 1:
        scale = _spf_snap_published(amax * scale_margin / _FP8_MAX, spf_alpha)
        flat_state[:1].copy_(scale.reshape(1))
        result = float(scale.item())
    else:
        if flat_state.numel() < 3:
            raise ValueError(
                "FP8 scale state must contain scale, count, and amax history"
            )
        history = flat_state[2:]
        completed = min(int(flat_state[1].item()), history.numel())
        if completed < history.numel():
            history[completed].copy_(amax)
            completed += 1
            flat_state[1].fill_(completed)
        if completed < history.numel():
            flat_state[0].zero_()
            _scale_value_cache.pop(_scale_key(scale_buffer), None)
            return 0.0
        scale = _spf_snap_published(
            history[:completed].amax() * scale_margin / _FP8_MAX, spf_alpha
        )
        flat_state[0].copy_(scale)
        result = float(scale.item())
    _scale_value_cache[_scale_key(scale_buffer)] = result
    return result


def clear_fp8_scale_runtime_cache() -> None:
    _scale_value_cache.clear()
    _inverse_scale_cache.clear()
    _folded_input_scale_cache.clear()
    _scaled_output_bias_cache.clear()
    _scaled_output_bias_refs.clear()


def quantize_with_scale(
    value: torch.Tensor,
    scale: torch.Tensor,
    *,
    column_major: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    target_dtype = fp8_e4m3_dtype() if dtype is None else dtype
    max_value = torch.finfo(target_dtype).max
    if column_major:
        transposed = value.detach().t().contiguous()
        return (
            (transposed / scale)
            .clamp(-max_value, max_value)
            .to(target_dtype)
            .t()
        )
    return (
        (value.detach().contiguous() / scale)
        .clamp(-max_value, max_value)
        .to(target_dtype)
    )


def _normalize_layout(
    weight: torch.Tensor,
    layout: Optional[_WeightLayout],
) -> _WeightLayout:
    if layout is None:
        return ((0, weight.shape[1]),)
    if not layout:
        raise ValueError("FP8 weight layout cannot be empty")
    for start, end in layout:
        if start < 0 or end <= start or end > weight.shape[1]:
            raise ValueError(
                f"invalid FP8 weight column range {(start, end)} for "
                f"shape {tuple(weight.shape)}"
            )
    return layout


def _weight_key(
    weight: torch.Tensor,
    layout: _WeightLayout,
) -> Tuple[object, ...]:
    device_index = weight.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return (
        id(weight),
        device_index,
        tuple(weight.shape),
        tuple(weight.stride()),
        layout,
    )


def _weight_version(weight: torch.Tensor) -> int:
    return int(getattr(weight, "_version", 0))


def _pack_weight_layout(weight: torch.Tensor, layout: _WeightLayout) -> torch.Tensor:
    if layout == ((0, weight.shape[1]),):
        return weight
    if len(layout) == 1:
        start, end = layout[0]
        return weight[:, start:end]
    packed = torch.empty(
        (weight.shape[0], sum(end - start for start, end in layout)),
        dtype=weight.dtype,
        device=weight.device,
    )
    destination = 0
    for start, end in layout:
        width = end - start
        packed[:, destination : destination + width].copy_(weight[:, start:end])
        destination += width
    return packed


# Arbor r14 iter-34 M1c, DEFAULT OFF: the weight-shadow refresh below as two
# Triton kernels (fp8_helpers_fused.shadow_refresh) instead of the ~9-kernel
# torch chain; byte-identical shadow and scale (see that module's contract).
_SHADOW_FUSED: bool = os.environ.get("DLRM_HSTU_FP8_SHADOW_FUSED", "0") == "1"
_shadow_fused_logged: set = set()


def _shadow_fused_ok(weight: torch.Tensor, layout: _WeightLayout) -> bool:
    return (
        weight.is_cuda
        and weight.dim() == 2
        and weight.dtype in (torch.float32, torch.bfloat16)
        and 1 <= len(layout) <= 4
        and _FP8_E4M3 is torch.float8_e4m3fn
    )


def _shadow_refresh_fused(
    weight: torch.Tensor, layout: _WeightLayout
) -> Tuple[torch.Tensor, torch.Tensor]:
    from generative_recommenders.ops.triton import fp8_helpers_fused

    key = (str(weight.device), tuple(weight.shape), weight.dtype, layout)
    if key not in _shadow_fused_logged:
        _shadow_fused_logged.add(key)
        print(
            f"[hstu-fp8] ARMED shadow_fused on {weight.device}: "
            f"w={tuple(weight.shape)}:{weight.dtype} layout={layout} "
            "(2 kernels: tile |w| max + scale/transpose/e4m3; bitwise)",
            flush=True,
        )
    return fp8_helpers_fused.shadow_refresh(weight.detach(), layout)


def _get_weight_shadow(
    weight: torch.Tensor,
    layout: Optional[_WeightLayout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a version-correct column-major E4M3 shadow and its scale."""

    global _cache_hits, _cache_misses

    if _FP8_E4M3 is None:
        raise RuntimeError("E4M3 weight shadow requested without E4M3 support")
    normalized_layout = _normalize_layout(weight, layout)
    key = _weight_key(weight, normalized_layout)
    version = _weight_version(weight)
    entry = _weight_cache.get(key)
    if (
        entry is not None
        and entry.ref() is weight
        and entry.version == version
        and entry.data_ptr == weight.data_ptr()
        and entry.shape == weight.shape
    ):
        _cache_hits += 1
        return entry.quantized_column_major, entry.scale

    _cache_misses += 1
    if _HSTU_FP8_WSHADOW_FUSED and _wshadow_fused_ok(weight):
        fused = _refresh_weight_shadows_fused(
            weight, normalized_layout, key, version
        )
        if fused is not None:
            return fused
    with record_function("## hstu_fp8_weight_shadow_refresh ##"):
        # Precedence (ship11): the batched WSHADOW_FUSED path above (gin item
        # (24)(e)) takes every call it accepts; SHADOW_FUSED (gin item (25))
        # only refreshes the shadows it declines or after it disabled itself.
        if _SHADOW_FUSED and _shadow_fused_ok(weight, normalized_layout):
            quantized, scale = _shadow_refresh_fused(weight, normalized_layout)
        else:
            packed_weight = _pack_weight_layout(weight, normalized_layout)
            scale = calibrated_scale(packed_weight, margin=1.0)
            quantized = quantize_with_scale(
                packed_weight,
                scale,
                column_major=True,
            )
    _weight_cache[key] = _WeightCacheEntry(
        ref=weakref.ref(weight),
        version=version,
        data_ptr=weight.data_ptr(),
        shape=weight.shape,
        quantized_column_major=quantized,
        scale=scale,
    )
    return quantized, scale


def _wshadow_module():
    global _wshadow_module_ref
    if _wshadow_module_ref is None:
        from generative_recommenders.ops.triton import triton_fp8_wshadow

        _wshadow_module_ref = triton_fp8_wshadow
    return _wshadow_module_ref


def _wshadow_fused_ok(weight: torch.Tensor) -> bool:
    if not weight.is_cuda or torch.compiler.is_compiling():
        return False
    return _wshadow_module().fused_shadow_supported(weight)


def _refresh_weight_shadows_fused(
    weight: torch.Tensor,
    layout: _WeightLayout,
    key: Tuple[object, ...],
    version: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fused refresh of `weight`'s shadow, plus (BATCH) every other stale one.

    A batch member is an existing cache entry whose weight is alive, on the
    same device/dtype, was last refreshed on the current stream, still has the
    geometry of its key, and whose version or data_ptr moved. Its new shadow
    is computed from the weight's bits *now*, exactly as its own call would;
    if the weight changes again before that call, the version check there
    misses and refreshes it once more.

    Returns None (and disables the fused path for the process) if the fused
    kernels raise; the caller then runs the aten chain. Nothing is cached
    before refresh_shadows returns, so a failed batch leaves no partial state.
    """
    global _wshadow_batched, _wshadow_armed_logged, _wshadow_batch_logged
    global _HSTU_FP8_WSHADOW_FUSED
    module = _wshadow_module()
    stream_id = torch.cuda.current_stream(weight.device).stream_id
    jobs = [(weight, layout, key, version)]
    if _HSTU_FP8_WSHADOW_BATCH:
        for other_key, entry in _weight_cache.items():
            if other_key == key or entry.layout is None:
                continue
            if entry.stream_id != stream_id:
                continue
            other = entry.ref()
            if (
                other is None
                or other.device != weight.device
                or other.dtype != weight.dtype
                or tuple(other.shape) != other_key[2]
                or tuple(other.stride()) != other_key[3]
                or not module.fused_shadow_supported(other)
            ):
                continue
            other_version = _weight_version(other)
            if (
                entry.version == other_version
                and entry.data_ptr == other.data_ptr()
                and entry.shape == other.shape
            ):
                continue
            jobs.append((other, entry.layout, other_key, other_version))
    try:
        with record_function("## hstu_fp8_weight_shadow_refresh ##"):
            results = module.refresh_shadows(
                [job[0] for job in jobs], [job[1] for job in jobs], margin=1.0
            )
    except Exception as exc:  # fail closed to the aten chain, never another lowering
        _HSTU_FP8_WSHADOW_FUSED = False
        logger.warning(
            "DLRMV4 fp8 wshadow fused DISABLED for this process (%s: %s); "
            "the aten chain (bitwise reference) refreshes every shadow from "
            "here on",
            type(exc).__name__,
            str(exc)[:300],
        )
        return None
    for (w, w_layout, w_key, w_version), (quantized, scale) in zip(jobs, results):
        _weight_cache[w_key] = _WeightCacheEntry(
            ref=weakref.ref(w),
            version=w_version,
            data_ptr=w.data_ptr(),
            shape=w.shape,
            quantized_column_major=quantized,
            scale=scale,
            layout=w_layout,
            stream_id=stream_id,
        )
    _wshadow_batched += len(jobs) - 1
    if not _wshadow_armed_logged:
        _wshadow_armed_logged = True
        logger.info(
            "DLRMV4 armed OK: fp8 wshadow fused (batch=%d) -- weight shadows "
            "refreshed by 2 Triton launches per refresh call (amax partials + "
            "scale/transpose/e4m3), bitwise the aten chain (kill switches "
            "DLRM_HSTU_FP8_WSHADOW_FUSED=0, DLRM_HSTU_FP8_WSHADOW_BATCH=0)",
            int(_HSTU_FP8_WSHADOW_BATCH),
        )
    if not _wshadow_batch_logged and len(jobs) > 1:
        _wshadow_batch_logged = True
        logger.info(
            "DLRMV4 fp8 wshadow batch: %d weight shadows refreshed in one "
            "call (2 launches per 8 source weights)",
            len(jobs),
        )
    return results[0]


def packed_weight_for_layout(
    weight: torch.Tensor,
    layout: _WeightLayout,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return one version-correct packed training weight without repeated cat."""

    normalized_layout = _normalize_layout(weight, layout)
    key = (*_weight_key(weight, normalized_layout), dtype)
    version = _weight_version(weight)
    entry = _packed_weight_cache.get(key)
    if (
        entry is not None
        and entry.ref() is weight
        and entry.version == version
        and entry.data_ptr == weight.data_ptr()
    ):
        return entry.packed
    packed = _pack_weight_layout(weight, normalized_layout).to(dtype)
    _packed_weight_cache[key] = _PackedWeightCacheEntry(
        ref=weakref.ref(weight),
        version=version,
        data_ptr=weight.data_ptr(),
        packed=packed,
    )
    return packed


def clear_fp8_weight_cache() -> None:
    global _cache_hits, _cache_misses, _wshadow_batched
    _weight_cache.clear()
    _packed_weight_cache.clear()
    _fp8_one_by_device.clear()
    _scale_value_cache.clear()
    _inverse_scale_cache.clear()
    _folded_input_scale_cache.clear()
    _scaled_output_bias_cache.clear()
    _scaled_output_bias_refs.clear()
    _fp8_output_scale_mode_by_device.clear()
    _telemetry_calls.clear()
    _telemetry_last.clear()
    _capability_sites.clear()
    _capability_contracts.clear()
    _cache_hits = 0
    _cache_misses = 0
    _wshadow_batched = 0


def fp8_weight_cache_stats() -> dict[str, int]:
    stale_keys = [key for key, entry in _weight_cache.items() if entry.ref() is None]
    for key in stale_keys:
        _weight_cache.pop(key, None)
    stale_packed_keys = [
        key for key, entry in _packed_weight_cache.items() if entry.ref() is None
    ]
    for key in stale_packed_keys:
        _packed_weight_cache.pop(key, None)
    return {
        "entries": len(_weight_cache),
        "packed_entries": len(_packed_weight_cache),
        "hits": _cache_hits,
        "misses": _cache_misses,
        "batched": _wshadow_batched,
    }


def fp8_tensor_telemetry(
    value: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, float | int | bool]:
    """Synchronizing diagnostic for gated saturation/underflow checks."""

    if value.dtype not in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    ):
        raise TypeError(f"expected an FP8 tensor, got {value.dtype}")
    total = value.numel()
    if total == 0:
        return {
            "numel": 0,
            "sampled_numel": 0,
            "finite": True,
            "saturation_fraction": 0.0,
            "zero_fraction": 0.0,
            "scale": float(scale.item()),
        }
    # Telemetry is deliberately bounded: materializing a production QKV tensor
    # as FP32 can require tens of GiB. An evenly spaced million-value sample is
    # enough to fail closed on material saturation/underflow without restoring
    # the cast tax that the fused path removes.
    max_samples = 1 << 20
    sample_stride = max(1, math.ceil(total / max_samples))
    while math.gcd(sample_stride, 6144) != 1:
        sample_stride += 1
    sampled = value.reshape(-1)[::sample_stride][:max_samples]
    as_float = sampled.float()
    max_value = torch.finfo(value.dtype).max
    return {
        "numel": total,
        "sampled_numel": sampled.numel(),
        "finite": bool(torch.isfinite(as_float).all().item()),
        "saturation_fraction": float(
            (as_float.abs() >= max_value).float().mean().item()
        ),
        "zero_fraction": float((as_float == 0).float().mean().item()),
        "scale": float(scale.item()),
    }


def _capability_rank_info() -> Tuple[int, int, int]:
    """Return global rank, local rank, and world size for capability evidence."""

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = int(torch.distributed.get_rank())
        world_size = int(torch.distributed.get_world_size())
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            str(torch.cuda.current_device() if torch.cuda.is_available() else rank),
        )
    )
    return rank, local_rank, world_size


def _write_fp8_runtime_report() -> None:
    capability_dir = os.environ.get("DLRM_HSTU_FP8_CAPABILITY_DIR")
    if not capability_dir:
        return
    rank, local_rank, world_size = _capability_rank_info()
    record = {
        "schema": "hstu-fp8-capability-v4",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "pid": os.getpid(),
        "torch_hip": torch.version.hip,
        "device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "fail_closed": _HSTU_FP8_FAIL_CLOSED,
        "fast_accum": _HSTU_FP8_FAST_ACCUM,
        "runtime_flags": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("DLRM_HSTU_FP8_")
            and key != "DLRM_HSTU_FP8_CAPABILITY_DIR"
        },
        "sites": sorted(_capability_sites),
        "contracts": _capability_contracts,
        "telemetry": fp8_runtime_telemetry(),
    }
    output_dir = Path(capability_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"rank{rank}.json"
    temporary = output_dir / f".rank{rank}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)


def maybe_record_fp8_telemetry(
    site: str,
    value: torch.Tensor,
    scale: torch.Tensor,
    *,
    cadence_divisor: int = 1,
) -> None:
    """Sample producer output only at an explicitly requested cadence."""

    if _HSTU_FP8_TELEMETRY_INTERVAL == 0:
        return
    calls = _telemetry_calls.get(site, 0) + 1
    _telemetry_calls[site] = calls
    interval = max(1, _HSTU_FP8_TELEMETRY_INTERVAL // cadence_divisor)
    if calls % interval != 0:
        return
    record = fp8_tensor_telemetry(value, scale)
    record["calls"] = calls
    record["samples"] = 1
    previous = _telemetry_last.get(site)
    if previous is not None:
        record["finite"] = bool(previous.get("finite")) and bool(record["finite"])
        record["sampled_numel"] = min(
            int(previous.get("sampled_numel", 0)),
            int(record["sampled_numel"]),
        )
        record["saturation_fraction"] = max(
            float(previous.get("saturation_fraction", 1.0)),
            float(record["saturation_fraction"]),
        )
        record["zero_fraction"] = max(
            float(previous.get("zero_fraction", 1.0)),
            float(record["zero_fraction"]),
        )
        record["samples"] = int(previous.get("samples", 1)) + 1
    _telemetry_last[site] = record
    _write_fp8_runtime_report()


def fp8_runtime_telemetry() -> dict[str, dict[str, float | int | bool]]:
    return {site: dict(record) for site, record in _telemetry_last.items()}


def _log_fp8_resolved_once() -> None:
    """Say once, per rank, what the FP8 knobs RESOLVED to.

    They are chained -- ATTN requires FUSE_QKV, which requires FUSE_LN, which
    requires GEMM -- so a submit line naming a stage does not mean the stage is
    on, and the arm that runs can be a quieter one than the arm that was asked
    for. Only the resolved values tell those apart, so those are what is logged.
    """

    global _resolved_logged
    if _resolved_logged:
        return
    _resolved_logged = True
    logger.info(
        "[hstu-fp8] resolved: fwd=%d gemm=%d fuse_ln=%d fuse_outln=%d "
        "fuse_qkv=%d attn=%d bwd=%d fuse_silu=%d fuse_residual=%d "
        "scale_mode=%s calibration_steps=%d host_scale_values=%d "
        "fail_closed=%d fast_accum=%d",
        _HSTU_FP8_FWD,
        _HSTU_FP8_GEMM,
        _HSTU_FP8_FUSE_LN,
        _HSTU_FP8_FUSE_OUTLN,
        _HSTU_FP8_FUSE_QKV,
        _HSTU_FP8_ATTENTION,
        _HSTU_FP8_BACKWARD,
        _HSTU_FP8_FUSE_SILU,
        _HSTU_FP8_FUSE_RESIDUAL,
        _HSTU_FP8_SCALE_MODE,
        _HSTU_FP8_CALIBRATION_STEPS,
        _HSTU_FP8_HOST_SCALE_VALUES,
        _HSTU_FP8_FAIL_CLOSED,
        _HSTU_FP8_FAST_ACCUM,
    )


def record_fp8_capability(site: str) -> None:
    """Note that a fused production site ran, once per site per rank.

    The capability FILE is what a certification run collects, and it is still
    written only when asked for. The LOG LINE is for every other run: until it
    existed, a production FP8 run left no record that it was FP8, so a BF16 rate
    reported under an FP8 label was indistinguishable from the real thing.
    """

    if site in _capability_sites:
        return
    _capability_sites.add(site)
    _log_fp8_resolved_once()
    logger.info("[hstu-fp8] engaged: %s", site)
    if not os.environ.get("DLRM_HSTU_FP8_CAPABILITY_DIR"):
        return
    _write_fp8_runtime_report()


def record_fp8_contract(site: str, **tensors: torch.Tensor) -> None:
    """Persist actual production dtypes/shapes for a fused-site proof."""

    capability_dir = os.environ.get("DLRM_HSTU_FP8_CAPABILITY_DIR")
    if not capability_dir:
        return
    contract = {
        name: {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        for name, tensor in sorted(tensors.items())
    }
    existing = _capability_contracts.get(site)
    if existing is not None:
        if {
            name: value["dtype"] for name, value in existing.items()
        } != {name: value["dtype"] for name, value in contract.items()}:
            raise RuntimeError(f"FP8 dtype contract changed at {site}")
        return
    _capability_contracts[site] = contract
    _write_fp8_runtime_report()


_HSTU_FP8_QUIET_HEURISTIC: bool = (
    os.environ.get("DLRM_HSTU_FP8_QUIET_HEURISTIC", "1") == "1"
)
_QUIET_SEEN_MAX: int = 256
_quiet_seen: "collections.OrderedDict[Tuple, None]" = collections.OrderedDict()
_quiet_lock = threading.Lock()
_devnull_fd: Optional[int] = None


def _unwrap_scaled_mm(value):
    return value[0] if isinstance(value, tuple) else value


def _scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
    scale_result: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if bias is None and scale_result is None and not _HSTU_FP8_FAST_ACCUM:
        # The stack's pinned hipBLASLt solution for this problem, if its table
        # has a ScaledGemm row (ops/blas_solution_pins.py; ROCm 10.2: the
        # preproc L2 dgrad), else None and the call below.
        pinned = blas_solution_pins.scaled_mm_pinned(a, b, scale_a, scale_b, out_dtype=out_dtype)
        if pinned is not None:
            return pinned
    kwargs = {
        "scale_a": scale_a,
        "scale_b": scale_b,
        "out_dtype": out_dtype,
        "use_fast_accum": _HSTU_FP8_FAST_ACCUM,
    }
    if bias is not None:
        kwargs["bias"] = bias
    if scale_result is not None:
        kwargs["scale_result"] = scale_result
    try:
        return _unwrap_scaled_mm(_scaled_mm_call(a, b, kwargs))
    except TypeError:
        # Some ROCm PyTorch builds do not expose use_fast_accum. This is an API
        # compatibility retry, not a BF16 fallback.
        kwargs.pop("use_fast_accum", None)
        return _unwrap_scaled_mm(_scaled_mm_call(a, b, kwargs))


def _heuristic_key(a: torch.Tensor, b: torch.Tensor, kwargs: Dict) -> Tuple:
    """Everything hipBLASLt's heuristic query can see, minus the data."""
    bias = kwargs.get("bias")
    return (
        a.device.index,
        tuple(a.shape), a.stride(), a.dtype,
        tuple(b.shape), b.stride(), b.dtype,
        kwargs["out_dtype"],
        None if bias is None else bias.dtype,
        tuple(kwargs["scale_a"].shape),
        tuple(kwargs["scale_b"].shape),
        "scale_result" in kwargs,
        kwargs.get("use_fast_accum"),
    )


def _scaled_mm_call(a: torch.Tensor, b: torch.Tensor, kwargs: Dict):
    """torch._scaled_mm, with hipBLASLt's heuristic chatter kept off fd 2.

    The first FP8 GEMM at a new shape makes hipBLASLt's analytical heuristic
    print "Warning: Latency not found for MI_M=16, MI_N=16, MI_K=128,
    mi_input_type=BFloat8Float8_fnuz. Returning latency value of 32 (really
    slow)." about 170 times, through an unbuffered std::cerr that issues one
    write(2) per `<<` token (the torn "16ITER35PROBE" fragments in the log show
    it). No environment variable gates it in hipBLASLt 1.2 / ROCm 7.2.
    Standalone, those writes cost ~0.2 ms. In the trainer, eight ranks share a
    single stderr pipe into launch_slurm.sh's sed|tee, and the same query costs
    8 ms of on-CPU time (median of 1703 calls, zero context switches). M is the
    jagged token count and changes every step, so every step pays it several
    times on the main thread while the GPU queue is shallow.

    Only the chatter changes. The GEMM, the heuristic and the kernel it picks
    are untouched, so the output is bit-identical. Redirection applies only to
    a shape key not seen recently (repeat shapes hit hipBLASLt's cache and
    print nothing). A failed quiet call is re-issued with stderr attached so
    any real diagnostic still reaches the log. DLRM_HSTU_FP8_QUIET_HEURISTIC=0
    restores the old behaviour.
    """
    if not _HSTU_FP8_QUIET_HEURISTIC or torch.compiler.is_compiling():
        return torch._scaled_mm(a, b, **kwargs)
    key = _heuristic_key(a, b, kwargs)
    if key in _quiet_seen:
        _quiet_seen.move_to_end(key)
        return torch._scaled_mm(a, b, **kwargs)
    try:
        output = _scaled_mm_stderr_silenced(a, b, kwargs)
    except TypeError:
        raise
    except Exception:
        return torch._scaled_mm(a, b, **kwargs)
    _quiet_seen[key] = None
    if len(_quiet_seen) > _QUIET_SEEN_MAX:
        _quiet_seen.popitem(last=False)
    return output


def _scaled_mm_stderr_silenced(a: torch.Tensor, b: torch.Tensor, kwargs: Dict):
    global _devnull_fd
    # fd 2 is process-wide. The lock stops two threads from interleaving their
    # save/restore, which could otherwise leave fd 2 pointing at /dev/null.
    # Anything another thread writes to stderr inside this window (~1 ms) is
    # lost, which is the accepted cost.
    with _quiet_lock:
        if _devnull_fd is None:
            _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            sys.stderr.flush()
        except Exception:
            pass
        saved = os.dup(2)
        try:
            os.dup2(_devnull_fd, 2)
            try:
                return torch._scaled_mm(a, b, **kwargs)
            finally:
                os.dup2(saved, 2)
        finally:
            os.close(saved)


@functools.lru_cache(maxsize=1)
def _residual_fused_extension():
    source = Path(__file__).with_name("hipblaslt_scaled_addmm.cpp")
    build_directory = Path("/tmp/dlrm_hstu_fp8_scaled_addmm")
    build_directory.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="dlrm_hstu_fp8_scaled_addmm",
            sources=[str(source)],
            extra_include_paths=["/opt/rocm/include"],
            extra_cflags=["-O3"],
            extra_ldflags=[
                "-L/opt/rocm/lib",
                "-lhipblaslt",
                "-lamdhip64",
            ],
            build_directory=str(build_directory),
            verbose=False,
        )
    except Exception as error:
        raise RuntimeError(
            "requested FP8 residual-fused hipBLASLt path is unavailable"
        ) from error


def prebuild_extensions() -> bool:
    """Build the runtime C++ extensions this configuration will load.

    Called by train_ranker during initialization (before MLPerf INIT_STOP /
    RUN_START, gated by DLRM_HSTU_FP8_PREBUILD) so the one-time
    torch.utils.cpp_extension build of the residual-fused hipBLASLt path is
    not paid inside the timed region on the first FP8 step. Same module, same
    build flags and the same lru_cache entry the lazy path returns, so the
    compiled code is identical. Returns True if something was built/loaded.

    Also builds the FP8 uvqk/vk token-major wgrad extension when (and only
    when) DLRM_HSTU_FP8_BWD_UVQK=1 with DLRM_HSTU_FP8_BWD_UVQK_WGRAD=fp8t is
    requested (bwd_fp8.prebuild_uvqk_extension; a no-op otherwise), the
    U-forward data-parallel extension when DLRM_HSTU_FP8_FWD_U_DP=1, the
    K=512 forward GEMM library when DLRM_HSTU_FP8_FWD_K512=1
    (ops/hip/k512_fp8_gemm.prebuild; with DLRM_HSTU_FP8_FWD_K512_TILED_A=1
    also the startup self-check of its tiled-A entry, prebuild_tiled_a), the
    K=1536 out-projection forward GEMM library when DLRM_HSTU_FP8_FWD_K1536=1
    (ops/hip/k1536_fp8_addmm.prebuild), and the L11 fused out-proj dgrad +
    LN*U*dropout backward library with its
    synthetic-data self-check when DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED=1
    (triton_hstu_linear.prebuild_l11, gin item (39); a no-op otherwise), then
    L11's dO8/E8 emission variant (its own .so and self-check) when
    DLRM_HSTU_FP8_L11_DO8=1 (triton_hstu_linear.prebuild_l11_do8, gin item
    (DO8); a no-op otherwise).
    """
    built = False
    if _HSTU_FP8_FUSE_RESIDUAL:
        _residual_fused_extension()
        built = True
    if _HSTU_FP8_FWD_U_DP and _HSTU_FP8_FUSE_QKV:
        # Raises if it cannot build/load: an armed run must not silently keep
        # the persistent-grid path. Only the QKV-fused FP8 forward calls it,
        # so a run with that forward off never builds it.
        _u_dp_extension()
        _u_dp_print(
            "prebuild: u_dp extension (dlrm_hstu_fp8_scaled_mm_bias) built/loaded"
        )
        built = True
    from generative_recommenders.ops.triton import bwd_fp8

    if bwd_fp8.prebuild_uvqk_extension():
        built = True
    # DLRM_HSTU_FP8_BWD_WGRAD_CK_W3=1 (default off) with the NN-mode FP8 out-proj
    # wgrad: the custom W3 kernel library (wgrad_ck_w3.py); a no-op otherwise.
    from generative_recommenders.ops.triton import triton_hstu_linear

    if triton_hstu_linear.prebuild_wgrad_ck_w3():
        built = True
    if _HSTU_FP8_FWD_K512:
        # DLRM_HSTU_FP8_FWD_K512: hipcc build of both N, plus one launch on
        # synthetic zeros (k512_fp8_gemm.prebuild). A build/self-check failure
        # is latched there, so the forward then declines to hipBLASLt;
        # fail-closed raises here. A device fault always raises.
        from generative_recommenders.ops.hip import k512_fp8_gemm

        try:
            if k512_fp8_gemm.prebuild():
                built = True
        except k512_fp8_gemm.PrebuildError:
            if _HSTU_FP8_FAIL_CLOSED:
                raise
        if _HSTU_FP8_FWD_K512_TILED_A:
            # B15T(L2) (gin item B15T(L2)): the startup self-check of the K512
            # library's tiled-A entry (N=1024, the targets V/K) against its
            # row-major entry on synthetic data (k512_fp8_gemm.prebuild_tiled_a;
            # the library loaded above is reused). A failure is latched there:
            # the targets layer then keeps the row-major y; fail-closed raises.
            try:
                if k512_fp8_gemm.prebuild_tiled_a((1024,)):
                    built = True
            except k512_fp8_gemm.PrebuildError:
                if _HSTU_FP8_FAIL_CLOSED:
                    raise
    if _HSTU_FP8_FWD_K1536 and _HSTU_FP8_FUSE_RESIDUAL:
        # DLRM_HSTU_FP8_FWD_K1536: hipcc build plus one launch on synthetic
        # zeros (k1536_fp8_addmm.prebuild); latched on failure, as for K512.
        from generative_recommenders.ops.hip import k1536_fp8_addmm

        try:
            if k1536_fp8_addmm.prebuild():
                built = True
        except k1536_fp8_addmm.PrebuildError:
            if _HSTU_FP8_FAIL_CLOSED:
                raise
    if _HSTU_FP8_FWD_MXFP6:
        # DLRM_HSTU_FP8_FWD_MXFP6 (item (92)): the r14 MX6 LN producer and pr7's
        # MXFP6 GEMMs, plus one self-check on synthetic zeros. Not ready: one
        # "[mx6-fwd] prebuild DECLINED" line and the forward stays on the
        # shipped FP8 path (PrebuildError only under ..._MXFP6_STRICT=1).
        from generative_recommenders.ops.hip import mx6_fwd

        if mx6_fwd.prebuild():
            built = True
    # L11 fused out-proj dgrad + LN*U*dropout backward (gin item (39)): hipcc
    # build + load + synthetic self-check; a no-op unless
    # DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED=1. Last, after every other build:
    # its self-check launches the FP8 dgrad GEMM and the LN*U*dropout backward
    # kernels on private synthetic tensors (before run_blas_warmup and the
    # first training step; no model, data, RNG or FP8 scale state).
    if triton_hstu_linear.prebuild_l11():
        built = True
    # DO8 (gin item (DO8)): L11's dO8/E8 emission variant, built after L11 (it
    # rides on a prebuilt L11); its own .so, flags, resource gate, self-check
    # and latch: a failure latches DO8 off only. A no-op unless
    # DLRM_HSTU_FP8_L11_DO8=1.
    if triton_hstu_linear.prebuild_l11_do8():
        built = True
    return built


def _residual_workspace(device: torch.device) -> torch.Tensor:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    workspace = _residual_workspaces.get(device_index)
    if workspace is None:
        workspace = torch.empty(
            _HSTU_FP8_FUSE_RESIDUAL_WORKSPACE_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        _residual_workspaces[device_index] = workspace
    return workspace


@functools.lru_cache(maxsize=1)
def _u_dp_extension():
    source = Path(__file__).with_name("hipblaslt_scaled_mm_bias.cpp")
    build_directory = Path("/tmp/dlrm_hstu_fp8_scaled_mm_bias")
    build_directory.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="dlrm_hstu_fp8_scaled_mm_bias",
            sources=[str(source)],
            extra_include_paths=["/opt/rocm/include"],
            extra_cflags=["-O3"],
            extra_ldflags=[
                "-L/opt/rocm/lib",
                "-lhipblaslt",
                "-lamdhip64",
            ],
            build_directory=str(build_directory),
            verbose=False,
        )
    except Exception as error:
        raise RuntimeError(
            "requested FP8 U-forward data-parallel hipBLASLt path "
            "(DLRM_HSTU_FP8_FWD_U_DP=1) is unavailable"
        ) from error


def _u_dp_workspace(device: torch.device) -> torch.Tensor:
    # One per device. Calls are ordered on the stream that issues them; at the
    # data-parallel grid hipBLASLt needs no partials, so the buffer is only the
    # size the heuristic and the call are told about.
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    workspace = _u_dp_workspaces.get(device_index)
    if workspace is None:
        workspace = torch.empty(
            _HSTU_FP8_FWD_U_DP_WS_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        _u_dp_workspaces[device_index] = workspace
    return workspace


def _u_dp_ineligible_reason(
    activation_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> Optional[str]:
    if residual is not None:
        return "residual epilogue"
    if bias is None or bias.dim() != 1:
        return "no 1-D bias"
    if out_dtype != torch.bfloat16 or bias.dtype != torch.bfloat16:
        return "output/bias not bf16"
    if not bias.is_contiguous():
        return "bias not contiguous"
    if not activation_fp8.is_contiguous():
        return "activation not contiguous"
    if (
        weight_fp8.dtype != fp8_e4m3_dtype()
        or weight_fp8.stride() != (1, weight_fp8.shape[0])
    ):
        return "weight shadow not column-major e4m3"
    for scale in (scale_a, scale_b):
        if (
            scale.dtype != torch.float32
            or scale.numel() != 1
            or scale.device != activation_fp8.device
        ):
            return "scales not single-element fp32 on the activation device"
    if bias.device != activation_fp8.device:
        return "bias on another device"
    if torch.compiler.is_compiling():
        return "inside torch.compile"
    return None


def _call_stderr_silenced(fn):
    """_scaled_mm_stderr_silenced for an arbitrary hipBLASLt call."""
    global _devnull_fd
    with _quiet_lock:
        if _devnull_fd is None:
            _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            sys.stderr.flush()
        except Exception:
            pass
        saved = os.dup(2)
        try:
            os.dup2(_devnull_fd, 2)
            try:
                return fn()
            finally:
                os.dup2(saved, 2)
        finally:
            os.close(saved)


def _u_dp_scaled_mm_bias(
    activation_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    extension = _u_dp_extension()
    device = activation_fp8.device
    workspace = _u_dp_workspace(device)

    def call() -> torch.Tensor:
        return extension.scaled_mm_bias(
            activation_fp8, weight_fp8, bias, scale_a, scale_b, workspace
        )

    if not _HSTU_FP8_QUIET_HEURISTIC or torch.compiler.is_compiling():
        output = call()
    else:
        # Same new-shape-only stderr silencing as _scaled_mm_call (hipBLASLt's
        # heuristic chatter on a new problem); its own key space.
        key = (
            "u_dp",
            device.index,
            tuple(activation_fp8.shape),
            tuple(weight_fp8.shape),
            workspace.numel(),
        )
        if key in _quiet_seen:
            _quiet_seen.move_to_end(key)
            output = call()
        else:
            try:
                output = _call_stderr_silenced(call)
            except Exception:
                # Re-issue with stderr attached so the real diagnostic reaches
                # the log; a second failure propagates (no fallback to
                # torch._scaled_mm).
                output = call()
            else:
                _quiet_seen[key] = None
                if len(_quiet_seen) > _QUIET_SEEN_MAX:
                    _quiet_seen.popitem(last=False)
    if ("armed", str(device)) not in _u_dp_logged:
        # After the first call returned: the line means the extension ran.
        _u_dp_logged.add(("armed", str(device)))
        _u_dp_print(
            f"ARMED u_dp on {device}: hipblaslt_scaled_mm_bias (workspace "
            f"{workspace.numel() >> 20} MiB, heuristic top-1 -> data-parallel "
            f"grid), M={activation_fp8.shape[0]} N={weight_fp8.shape[1]} "
            f"K={activation_fp8.shape[1]}"
        )
    return output


def scaled_addmm_fp8_preq(
    activation_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    weight: torch.Tensor,
    bias_or_residual: Optional[torch.Tensor],
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    weight_layout: Optional[_WeightLayout] = None,
    data_parallel: bool = False,
) -> torch.Tensor:
    """FP8 GEMM consuming an activation emitted by a producer epilogue.

    ``data_parallel=True`` marks a call site that may use the data-parallel
    hipBLASLt path (DLRM_HSTU_FP8_FWD_U_DP=1; 1-D bf16 bias, bf16 output). It is
    a no-op while that knob is off or the call does not qualify.
    """

    normalized_layout = _normalize_layout(weight, weight_layout)
    output_columns = sum(end - start for start, end in normalized_layout)
    if not fp8_supported(activation_fp8, weight) or output_columns % 16 != 0:
        raise RuntimeError(
            "unsupported fused FP8 GEMM shape/backend: "
            f"x={tuple(activation_fp8.shape)} w={tuple(weight.shape)} "
            f"layout={normalized_layout} hip={torch.version.hip}"
        )
    if activation_fp8.dtype != fp8_e4m3_dtype():
        raise TypeError(f"expected E4M3 activation, got {activation_fp8.dtype}")
    if bias_or_residual is not None:
        if bias_or_residual.dim() == 1:
            if bias_or_residual.numel() != output_columns:
                raise ValueError(
                    f"expected bias [{output_columns}], got "
                    f"{tuple(bias_or_residual.shape)}"
                )
        elif tuple(bias_or_residual.shape) != (
            activation_fp8.shape[0],
            output_columns,
        ):
            raise ValueError(
                "expected residual "
                f"[{activation_fp8.shape[0]}, {output_columns}], got "
                f"{tuple(bias_or_residual.shape)}"
            )
    if activation_fp8.shape[0] == 0:
        return torch.empty(
            (0, output_columns),
            dtype=out_dtype,
            device=activation_fp8.device,
        )

    maybe_record_fp8_telemetry(
        f"addmm:{id(weight)}:{tuple(weight.shape)}:{normalized_layout}",
        activation_fp8,
        scale_a,
        cadence_divisor=64 if len(normalized_layout) > 1 else 1,
    )
    weight_fp8, scale_b = _get_weight_shadow(weight, normalized_layout)
    bias = None
    residual = None
    if bias_or_residual is not None:
        if bias_or_residual.dim() == 1:
            bias = bias_or_residual.to(out_dtype)
        else:
            residual = bias_or_residual
    if residual is not None and _HSTU_FP8_FUSE_RESIDUAL:
        if (
            out_dtype != torch.bfloat16
            or residual.dtype != torch.bfloat16
            or not activation_fp8.is_contiguous()
            or not residual.is_contiguous()
            or weight_fp8.stride() != (1, weight_fp8.shape[0])
        ):
            raise RuntimeError(
                "requested FP8 residual fusion requires contiguous E4M3 "
                "activation, column-major E4M3 weight, and contiguous BF16 residual"
            )
        if _HSTU_FP8_FWD_K1536:
            output = _k1536_res_or_none(
                activation_fp8, weight_fp8, residual, scale_a, scale_b
            )
            if output is not None:
                return output
        return _residual_fused_extension().scaled_addmm(
            activation_fp8,
            weight_fp8,
            residual,
            scale_a,
            scale_b,
            _residual_workspace(activation_fp8.device),
            _HSTU_FP8_FUSE_RESIDUAL_ALGORITHM,
        )
    if data_parallel and _HSTU_FP8_FWD_U_DP:
        reason = _u_dp_ineligible_reason(
            activation_fp8, weight_fp8, bias, residual, scale_a, scale_b, out_dtype
        )
        if reason is None:
            assert bias is not None
            return _u_dp_scaled_mm_bias(
                activation_fp8, weight_fp8, bias, scale_a, scale_b
            )
        if (reason, str(activation_fp8.device)) not in _u_dp_logged:
            _u_dp_logged.add((reason, str(activation_fp8.device)))
            _u_dp_print(
                f"u_dp NOT engaged on {activation_fp8.device}: {reason} -> "
                "torch._scaled_mm"
            )
    output = _scaled_mm(
        activation_fp8,
        weight_fp8,
        scale_a,
        scale_b,
        out_dtype=out_dtype,
        bias=bias,
    )
    return output if residual is None else output + residual


def _k1536_res_or_none(
    activation_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    residual: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> Optional[torch.Tensor]:
    """DLRM_HSTU_FP8_FWD_K1536=1: the K=1536 out-projection GEMM + residual in
    place of the residual-fused hipBLASLt extension call, on exactly the
    operands that call receives, or None (the caller then runs hipBLASLt).

    weight_fp8 is the _get_weight_shadow view [K, N] with stride (1, K), i.e. an
    [N, K] row-major e4m3 buffer seen transposed, so weight_fp8.t() is the
    kernel's W. The extension computes D = (scale_b * scale_a) * (A x W) + C in
    fp32 with alpha = beta = 1 (scale_a = activation scale, scale_b = weight
    scale); the kernel's product is commutative-exact. Declines are logged once
    per reason kind; an unavailable library raises under
    DLRM_HSTU_FP8_FAIL_CLOSED=1 and is declined otherwise.
    """

    from generative_recommenders.ops.hip import k1536_fp8_addmm as k1536

    reason: Optional[Tuple[str, str]] = None
    if torch.compiler.is_compiling():
        reason = ("compile", "torch.compile tracing (ctypes launch)")
    elif (
        weight_fp8.dim() != 2
        or tuple(weight_fp8.shape) != (k1536.K, k1536.N)
        or weight_fp8.stride() != (1, k1536.K)
    ):
        reason = (
            "weight",
            f"weight_fp8 {tuple(weight_fp8.shape)} stride {weight_fp8.stride()} "
            f"(needs the [{k1536.N}, {k1536.K}] row-major shadow seen transposed)",
        )
    else:
        reason = k1536.ineligible_reason(
            activation_fp8, weight_fp8.t(), residual, scale_a, scale_b
        )
    if reason is not None:
        k1536.log_decline_once(reason[0], reason[1])
        return None
    try:
        kernel = k1536.kernel()
    except Exception as error:
        if _HSTU_FP8_FAIL_CLOSED:
            raise RuntimeError(
                "DLRM_HSTU_FP8_FWD_K1536=1 but the K=1536 FP8 GEMM is unavailable"
            ) from error
        k1536.log_decline_once("library", str(error))
        return None
    k1536.log_armed_once(kernel, activation_fp8.shape[0])
    return k1536.launch(
        kernel, activation_fp8, weight_fp8.t(), residual, scale_a, scale_b
    )


def _k512_fwd_or_none(
    activation_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale_result: Optional[torch.Tensor],
    output_scale_mode: str,
) -> Optional[torch.Tensor]:
    """DLRM_HSTU_FP8_FWD_K512=1: the K=512 forward GEMM in place of the
    hipBLASLt call below, on exactly the operands that call receives, or None
    (the caller then runs hipBLASLt).

    weight_fp8 is the _get_weight_shadow view: [K, N] with stride (1, K), i.e.
    an [N, K] row-major e4m3 buffer seen transposed, so weight_fp8.t() is the
    kernel's B. Covered: input_scale backend mode (no scale_result), no fast
    accumulation, K=512, N in (1024, 1536), contiguous e4m3 activation, bf16
    bias. Anything else is declined with one log line per reason kind; an
    unavailable library raises under DLRM_HSTU_FP8_FAIL_CLOSED=1 and is
    declined otherwise.
    """

    from generative_recommenders.ops.hip import k512_fp8_gemm as k512

    k_dim = activation_fp8.shape[-1]
    n_dim = weight_fp8.shape[-1]
    if k_dim != k512.K or n_dim not in k512.SUPPORTED_N:
        k512.log_decline_once(
            f"shape:{k_dim}x{n_dim}",
            f"K={k_dim} N={n_dim} (the kernel covers K={k512.K}, N in "
            f"{k512.SUPPORTED_N})",
        )
        return None
    reason: Optional[Tuple[str, str]] = None
    if output_scale_mode != "input_scale" or scale_result is not None:
        reason = (
            "mode",
            f"output_scale_mode={output_scale_mode} scale_result="
            f"{'set' if scale_result is not None else 'None'} (needs "
            "input_scale and no scale_result)",
        )
    elif _HSTU_FP8_FAST_ACCUM:
        reason = (
            "fastacc",
            "DLRM_HSTU_FP8_GEMM_FASTACC=1 (parity is defined against "
            "use_fast_accum=False)",
        )
    elif torch.compiler.is_compiling():
        reason = ("compile", "torch.compile tracing (ctypes launch)")
    elif weight_fp8.dim() != 2 or weight_fp8.stride() != (1, k512.K):
        reason = (
            "weight",
            f"weight_fp8 {tuple(weight_fp8.shape)} stride {weight_fp8.stride()} "
            f"(needs the [N, {k512.K}] row-major shadow, stride (1, {k512.K}))",
        )
    else:
        reason = k512.ineligible_reason(
            activation_fp8, weight_fp8.t(), bias, scale_a, scale_b
        )
    if reason is not None:
        k512.log_decline_once(f"N{n_dim}:{reason[0]}", f"N={n_dim} {reason[1]}")
        return None
    assert bias is not None
    try:
        kernel = k512.kernel(n_dim)
    except Exception as error:
        if _HSTU_FP8_FAIL_CLOSED:
            raise RuntimeError(
                "DLRM_HSTU_FP8_FWD_K512=1 but the K=512 FP8 GEMM is unavailable"
            ) from error
        k512.log_decline_once(f"N{n_dim}:library", f"N={n_dim} {error}")
        return None
    k512.log_armed_once(kernel, activation_fp8.shape[0])
    return k512.launch(
        kernel, activation_fp8, weight_fp8.t(), bias, scale_a, scale_b
    )


def _out_preq_operands(
    device: torch.device,
    scale_a: torch.Tensor,
    weight: torch.Tensor,
    normalized_layout: _WeightLayout,
    bias: Optional[torch.Tensor],
    output_scale: Optional[torch.Tensor],
    scale_a_value: Optional[float],
    output_scale_value: Optional[float],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    str,
]:
    """The GEMM operands of scaled_mm_fp8_out_preq for an activation on `device`:
    (weight_fp8, scale_b, effective_scale_a, effective_bias, scale_result,
    output_scale_mode), i.e. the weight shadow, then the output-scale folding
    of the backend mode. One reference for the row-major call and the B15T(L2)
    tiled-A call (scaled_mm_fp8_out_preq_tiled_a), so both hand K512 the same
    operands."""

    weight_fp8, scale_b = _get_weight_shadow(weight, normalized_layout)
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    output_scale_mode = _fp8_output_scale_mode(device)
    effective_scale_a = scale_a
    effective_bias = None if bias is None else bias.to(torch.bfloat16)
    host_scale_values = (
        None
        if output_scale is None
        else _trusted_host_scale_values(
            scale_a,
            output_scale,
            scale_a_value,
            output_scale_value,
        )
    )
    if output_scale is None:
        if output_scale_mode == "input_scale":
            scale_result = None
        else:
            scale_result = _fp8_one_by_device.get(device_index)
            if scale_result is None:
                scale_result = torch.ones(
                    (1, 1),
                    dtype=torch.float32,
                    device=device,
                )
                _fp8_one_by_device[device_index] = scale_result
    elif output_scale_mode == "input_scale":
        effective_scale_a = _fold_output_scale_into_input(
            scale_a,
            output_scale,
            scale_a_value=(
                None if host_scale_values is None else host_scale_values[0]
            ),
            output_scale_value=(
                None if host_scale_values is None else host_scale_values[1]
            ),
        )
        effective_bias = (
            None
            if bias is None
            else _scale_bias_for_fp8_output(
                bias,
                output_scale,
                output_scale_value=(
                    None if host_scale_values is None else host_scale_values[1]
                ),
            )
        )
        scale_result = None
    else:
        scale_result = inverse_scale_tensor(
            output_scale,
            value=(
                None if host_scale_values is None else host_scale_values[1]
            ),
        )
    return (
        weight_fp8,
        scale_b,
        effective_scale_a,
        effective_bias,
        scale_result,
        output_scale_mode,
    )


# -----------------------------------------------------------------------------
# B15T(L2), yambda_5b.gin item B15T(L2), DLRM_HSTU_FP8_FWD_K512_TILED_A. Before
# its LayerNorm the targets-only layer (triton_hstu_preprocess_and_attention)
# asks the store_t (triton_layer_norm.store_t_skip_y_decline_reason) and
# k512_tiled_a_vk_decline_reason; on None from both it runs the Triton v2
# store_t with store_y=False (y's bytes still written, into a 1/32-size
# discarded scratch; y = None) and the V/K projection runs as
# scaled_mm_fp8_out_preq_tiled_a on the step-tiled yT. Every other outcome keeps
# the row-major y for scaled_mm_fp8_out_preq (after the LayerNorm: y rebuilt
# from yT bit for bit, b15t_row_major_y), with one "[b15t] declined" line per
# reason (kind and detail). A decline AFTER the y skip (the projection could not
# read yT: operands, launch error, ...) latches B15T off for the process
# (latch_b15t_off): later calls write y again instead of rebuilding it every
# step.

_B15T_ARMED_LOGGED: bool = False
_b15t_decline_logged: set = set()
# Not None once a post-skip decline latched B15T off: "<kind>: <detail>".
_B15T_LATCHED: Optional[str] = None


def _b15t_print(message: str) -> None:
    print(f"[b15t] {message}", flush=True)


def k512_tiled_a_requested() -> bool:
    """DLRM_HSTU_FP8_FWD_K512_TILED_A (read at import): gin item B15T(L2) on."""
    return _HSTU_FP8_FWD_K512_TILED_A


def log_b15t_decline_once(kind: str, detail: str) -> None:
    """One line per (reason kind, detail) per process: this call keeps the
    row-major y and the V/K projection reads it (the knob-off path, bit for
    bit)."""
    key = (kind, detail)
    if key in _b15t_decline_logged:
        return
    _b15t_decline_logged.add(key)
    _b15t_print(f"declined -> row-major y + V/K projection ({kind}): {detail}")


def latch_b15t_off(kind: str, detail: str) -> None:
    """A decline after the y skip (y is rebuilt from yT for this call): logged
    once (log_b15t_decline_once), then B15T stays off for the process, so later
    targets-only calls write y in the LayerNorm instead of rebuilding it from yT
    every step (r14 B-4)."""
    global _B15T_LATCHED
    log_b15t_decline_once(kind, detail)
    if _B15T_LATCHED is None:
        _B15T_LATCHED = f"{kind}: {detail}"
        _b15t_print(
            "LATCHED OFF for this process after a post-skip decline "
            f"({kind}); the targets-only store_t writes y again"
        )


def b15t_latched() -> Optional[str]:
    """The post-skip decline that latched B15T off ("<kind>: <detail>"), or None."""
    return _B15T_LATCHED


def _log_b15t_armed_once(n: int, rows: int, a_t: torch.Tensor) -> None:
    global _B15T_ARMED_LOGGED
    if _B15T_ARMED_LOGGED:
        return
    _B15T_ARMED_LOGGED = True
    _b15t_print(
        "ARMED targets VK reads tiled yT; store_t skips y: "
        f"k512_2wg_kernel_tiled_a<{n}> on {a_t.device} reads the store_t yT "
        f"{list(a_t.shape)} e4m3, first M={rows}; the Triton v2 store_t writes "
        "y's bytes into a 1/32-size discarded scratch (store_y=False) "
        "(DLRM_HSTU_FP8_FWD_K512_TILED_A=1; kill switch =0)"
    )


def reset_b15t_for_testing() -> None:
    """Test-only: forget the B15T once-lines and the latch."""
    global _B15T_ARMED_LOGGED, _B15T_LATCHED
    _B15T_ARMED_LOGGED = False
    _B15T_LATCHED = None
    _b15t_decline_logged.clear()


def b15t_row_major_y(activation_fp8_t: torch.Tensor, rows: int) -> torch.Tensor:
    """The row-major e4m3 y [rows, D] whose bytes a store_t yT holds (step-tiled
    [KT, 512, 128] or 2-D [D, Mp], zero tail): the fallback of a call whose y
    write was skipped. A contiguous copy (the bytes the store_t writes as y)."""
    if activation_fp8_t.dim() == 3:
        from generative_recommenders.ops.hip import k512_fp8_gemm as k512

        return k512.tiled_a_to_row_major(activation_fp8_t, rows)
    return activation_fp8_t[:, :rows].t().contiguous()


def k512_tiled_a_vk_decline_reason(
    device: torch.device,
    k_dim: int,
    n_dim: int,
    rows: int,
) -> Optional[Tuple[str, str]]:
    """B15T(L2), before the LayerNorm, on the host: why an e4m3-output projection
    (scaled_mm_fp8_out_preq with an output scale) of a [rows, k_dim] e4m3 y into
    n_dim columns would NOT run on the K512 tiled-A entry, or None when it would
    (the LayerNorm may then write y's bytes into the discarded scratch instead
    of an [M, 512] y). These are the checks of _k512_fwd_or_none
    that need no operand (scaled_mm_fp8_out_preq_tiled_a makes them again with
    the operand checks), the activation telemetry (it samples y's values) and
    the tiled-A entry's startup self-check (k512_fp8_gemm.prebuild_tiled_a,
    before RUN_START; without the prebuild it runs here once, with one device
    sync). No tensor access and no device sync otherwise."""
    from generative_recommenders.ops.hip import k512_fp8_gemm as k512

    if _B15T_LATCHED is not None:
        return ("latched", f"B15T latched off earlier: {_B15T_LATCHED}")
    if not _HSTU_FP8_FWD_K512:
        return (
            "k512_off",
            "DLRM_HSTU_FP8_FWD_K512 off (the tiled-A entry is bit-identical to "
            "the K512 row-major entry, not to hipBLASLt)",
        )
    if k_dim != k512.K or n_dim not in k512.SUPPORTED_N:
        return (
            "shape",
            f"K={k_dim} N={n_dim} (the kernel covers K={k512.K}, N in "
            f"{k512.SUPPORTED_N})",
        )
    if rows < 1:
        return ("rows", f"M={rows}")
    if _HSTU_FP8_TELEMETRY_INTERVAL:
        return (
            "telemetry",
            "DLRM_HSTU_FP8_TELEMETRY_INTERVAL > 0 samples the row-major y",
        )
    if _HSTU_FP8_FAST_ACCUM:
        return ("fastacc", "DLRM_HSTU_FP8_GEMM_FASTACC=1 (K512 declines it)")
    if torch.compiler.is_compiling():
        return ("compile", "torch.compile tracing (ctypes launch)")
    if device.type != "cuda" or (
        device.index is not None and device.index != torch.cuda.current_device()
    ):
        return ("device", f"{device} is not the current CUDA device")
    mode = _fp8_output_scale_mode(device)
    if mode != "input_scale":
        return ("mode", f"output_scale_mode={mode} (K512 needs input_scale)")
    if not k512.tiled_a_ready(n_dim, build=False):
        # Not checked yet on this device (no DLRM_HSTU_FP8_PREBUILD) or failed.
        # The lazy self-check launches and syncs: never inside a CUDA graph
        # capture, and any error it raises is a decline, not a crash.
        if torch.cuda.is_current_stream_capturing():
            return (
                "not_ready",
                f"N={n_dim} tiled-A self-check not run yet and the stream is "
                "capturing (DLRM_HSTU_FP8_PREBUILD=1 runs it before RUN_START)",
            )
        try:
            ready = k512.tiled_a_ready(n_dim, build=True)
        except Exception as error:  # noqa: BLE001 - any error declines
            return (
                "not_ready",
                f"N={n_dim} tiled-A self-check raised {type(error).__name__}: "
                f"{error}",
            )
        if not ready:
            why = (
                k512.tiled_a_failure(n_dim)
                or k512.failure(n_dim)
                or "self-check did not pass"
            )
            return ("not_ready", f"N={n_dim} tiled-A entry unavailable: {why}")
    return None


def scaled_mm_fp8_out_preq_tiled_a(
    activation_fp8_t: torch.Tensor,
    rows: int,
    scale_a: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    weight_layout: Optional[_WeightLayout] = None,
    output_scale: Optional[torch.Tensor] = None,
    scale_a_value: Optional[float] = None,
    output_scale_value: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """B15T(L2): scaled_mm_fp8_out_preq(y, scale_a, weight, bias, ...) for the
    e4m3 y [rows, 512] whose bytes the step-tiled yT `activation_fp8_t` holds
    (e4m3 [ceil(rows/128), 512, 128]; token t / feature c at
    ((t // 128) * 512 + c) * 128 + t % 128: the store_t's yT), on the K512
    library's tiled-A entry. The weight shadow, the scale and bias folding
    (_out_preq_operands) and the K512 coverage are those of that call, and the
    tiled-A entry is bit-identical to the row-major entry on y, so the output
    bytes equal scaled_mm_fp8_out_preq(y, ...) whenever K512 takes that call.
    Returns None, with the reason logged once and B15T latched off
    (latch_b15t_off), when that call would not run on K512, the tiled-A entry
    cannot take these operands or its launch fails: the caller then rebuilds y
    (b15t_row_major_y) and calls scaled_mm_fp8_out_preq, which raises where that
    call raises. Output-scaled projections only."""
    from generative_recommenders.ops.hip import k512_fp8_gemm as k512

    normalized_layout = _normalize_layout(weight, weight_layout)
    output_columns = sum(end - start for start, end in normalized_layout)
    reason: Optional[Tuple[str, str]] = None
    if output_scale is None:
        reason = ("output_scale", "no output scale (B15T covers the e4m3 V/K call)")
    elif (
        not _HSTU_FP8_GEMM
        or not activation_fp8_t.is_cuda
        or not weight.is_cuda
        or weight.dim() != 2
        or weight.shape[0] % 16 != 0
        or weight.shape[1] % 16 != 0
        or output_columns % 16 != 0
        or (
            bias is not None
            and (bias.dim() != 1 or bias.numel() != output_columns)
        )
    ):
        # scaled_mm_fp8_out_preq raises on these: the caller's call does.
        reason = (
            "contract",
            f"w={tuple(weight.shape)} layout={normalized_layout} bias="
            f"{None if bias is None else tuple(bias.shape)}",
        )
    else:
        reason = k512_tiled_a_vk_decline_reason(
            activation_fp8_t.device, int(weight.shape[0]), output_columns, rows
        )
    if reason is not None:
        latch_b15t_off(*reason)
        return None
    (
        weight_fp8,
        scale_b,
        effective_scale_a,
        effective_bias,
        scale_result,
        output_scale_mode,
    ) = _out_preq_operands(
        activation_fp8_t.device,
        scale_a,
        weight,
        normalized_layout,
        bias,
        output_scale,
        scale_a_value,
        output_scale_value,
    )
    n_dim = weight_fp8.shape[-1]
    # _k512_fwd_or_none's operand checks, with the tiled A.
    if output_scale_mode != "input_scale" or scale_result is not None:
        reason = (
            "mode",
            f"output_scale_mode={output_scale_mode} scale_result="
            f"{'set' if scale_result is not None else 'None'}",
        )
    elif weight_fp8.dim() != 2 or weight_fp8.stride() != (1, k512.K):
        reason = (
            "weight",
            f"weight_fp8 {tuple(weight_fp8.shape)} stride {weight_fp8.stride()}",
        )
    else:
        reason = k512.ineligible_reason_tiled_a(
            activation_fp8_t,
            rows,
            weight_fp8.t(),
            effective_bias,
            effective_scale_a,
            scale_b,
        )
    if reason is not None:
        latch_b15t_off(*reason)
        return None
    assert effective_bias is not None
    try:
        kernel = k512.kernel(n_dim)
    except Exception as error:  # noqa: BLE001 - any error declines
        latch_b15t_off("library", str(error))
        return None
    try:
        output = k512.launch_tiled_a(
            kernel,
            activation_fp8_t,
            rows,
            weight_fp8.t(),
            effective_bias,
            effective_scale_a,
            scale_b,
        )
    except RuntimeError as error:
        # A launch error (hipError from the entry) is a decline, not a crash:
        # the caller rebuilds y from yT and takes the row-major projection;
        # B15T is latched off (Arbor P7 iter20 review F2).
        latch_b15t_off("launch", str(error))
        return None
    # (43)'s arming line for this N (its library takes the call), then the
    # tiled-A entry's and this item's: printed after the first launch that
    # succeeded.
    k512.log_armed_once(kernel, rows)
    k512.log_tiled_a_armed_once(kernel, rows)
    _log_b15t_armed_once(kernel.n, rows, activation_fp8_t)
    maybe_record_fp8_telemetry(
        f"qkv_output:{id(weight)}:{tuple(weight.shape)}:{normalized_layout}",
        output,
        fp8_scale_tensor(output_scale),
    )
    return output


def scaled_mm_fp8_out_preq(
    activation_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    weight_layout: Optional[_WeightLayout] = None,
    output_scale: Optional[torch.Tensor] = None,
    scale_a_value: Optional[float] = None,
    output_scale_value: Optional[float] = None,
) -> torch.Tensor:
    """Emit a scaled E4M3 projection for a downstream FP8 consumer."""

    normalized_layout = _normalize_layout(weight, weight_layout)
    output_columns = sum(end - start for start, end in normalized_layout)
    if not fp8_supported(activation_fp8, weight) or output_columns % 16 != 0:
        raise RuntimeError(
            "unsupported FP8-output GEMM shape/backend: "
            f"x={tuple(activation_fp8.shape)} w={tuple(weight.shape)} "
            f"layout={normalized_layout}"
        )
    if activation_fp8.dtype != fp8_e4m3_dtype():
        raise TypeError(f"expected E4M3 activation, got {activation_fp8.dtype}")
    if bias is not None and (bias.dim() != 1 or bias.numel() != output_columns):
        raise ValueError(
            f"expected bias [{output_columns}], got {tuple(bias.shape)}"
        )
    if activation_fp8.shape[0] == 0:
        return torch.empty(
            (0, output_columns),
            dtype=fp8_e4m3_dtype(),
            device=activation_fp8.device,
        )
    maybe_record_fp8_telemetry(
        f"mm_out:{id(weight)}:{tuple(weight.shape)}:{normalized_layout}",
        activation_fp8,
        scale_a,
        cadence_divisor=64 if len(normalized_layout) > 1 else 1,
    )
    (
        weight_fp8,
        scale_b,
        effective_scale_a,
        effective_bias,
        scale_result,
        output_scale_mode,
    ) = _out_preq_operands(
        activation_fp8.device,
        scale_a,
        weight,
        normalized_layout,
        bias,
        output_scale,
        scale_a_value,
        output_scale_value,
    )
    output = None
    if _HSTU_FP8_FWD_K512:
        output = _k512_fwd_or_none(
            activation_fp8,
            weight_fp8,
            effective_scale_a,
            scale_b,
            effective_bias,
            scale_result,
            output_scale_mode,
        )
    if output is None:
        output = _scaled_mm(
            activation_fp8,
            weight_fp8,
            effective_scale_a,
            scale_b,
            out_dtype=fp8_e4m3_dtype(),
            bias=effective_bias,
            scale_result=scale_result,
        )
    if output_scale is not None:
        maybe_record_fp8_telemetry(
            f"qkv_output:{id(weight)}:{tuple(weight.shape)}:{normalized_layout}",
            output,
            fp8_scale_tensor(output_scale),
        )
    return output


def scaled_mm_fp8_preq(
    a_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    b_fp8_column_major: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Low-level prequantized GEMM used by selective manual backward."""

    if a_fp8.shape[0] == 0:
        return torch.empty(
            (0, b_fp8_column_major.shape[1]),
            dtype=out_dtype,
            device=a_fp8.device,
        )
    return _scaled_mm(
        a_fp8,
        b_fp8_column_major,
        scale_a,
        scale_b,
        out_dtype=out_dtype,
    )
