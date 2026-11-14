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

"""Custom FP8 K=tokens out-projection weight gradient ("wgrad_ck" W3).

DLRM_HSTU_FP8_BWD_WGRAD_CK_W3=1 (default off; Arbor r14 iteration 38; read once at
import). Independent of DLRM_HSTU_FP8_BWD_WGRAD_CK (W1 / W2, wgrad_ck.py): either
can be on without the other.

What: with the FP8 out-proj wgrad in NN mode (DLRM_HSTU_FP8_BWD_OUT_WGRAD=1,
IMPL=nn), HSTUComputeOutputFunction.backward (triton_hstu_linear.py) computes
dw_t [512, 1536] fp32 = scale_y*scale_d * dout_t[:, :M] @ y8 with one hipBLASLt
NN GEMM (ops/hipblaslt_fp8_nn.fp8_wgrad_nn). With this knob it runs
wgrad_tm_ck_w3.cu instead: the W1 split-K-over-XCD-cohorts design with a
256 x 192 tile (256 WGs, NSPLIT=16), each token chunk of both operands fetched
from HBM once per cohort (lab EA census at M=2650962: 1.00x the minimum bytes vs
hipBLASLt's 3.33x), fp32 partials [16, 512, 1536] in a persistent workspace per
(device, stream) (48 MiB) and a fixed-order reduction kernel that applies
scale_y*scale_d. Kernels in a trace: wgrad_ck_w3_kernel<1536, 192, 16>,
wgrad_ck_w3_reduce<1536, 16>.

How: the library is built by wgrad_ck.build (same hipcc, flags, /tmp cache root
and atomic publish as W1/W2; own cache entry keyed by this source) and bound with
wgrad_ck.bind_entry (same launcher ABI). `prebuild()` builds and loads it before
RUN_START (triton_fp8.prebuild_extensions -> triton_hstu_linear.prebuild_wgrad_ck_w3,
gin DLRM_HSTU_FP8_PREBUILD). Launched on the caller's current stream.

Contract (`run_w3` declines, i.e. returns None and the caller runs hipBLASLt, otherwise):
y8 float8_e4m3fn 2-D, strides (1536, 1), >= M rows (rows >= M are never read);
dout_t float8_e4m3fn [512, ldm], strides (ldm, 1), ldm % 16 == 0,
ldm >= roundup(M, 16), 256 * ldm <= 0x7fffffff; 16-byte aligned data; scale_y /
scale_d single-element fp32 tensors on the same (cuda) device; M >= 1; a gfx950
device. dout_t columns [M, ldm) are padding (as for fp8_wgrad_nn: never part of
the result). The kernel reads tokens [M, min(ldm, roundup(M, 128))) in whole
16-token chunks and multiplies them by zero y8 rows, so `run_w3` zero-fills that
range of dout_t before every launch (production: the dual cast already wrote
zeros there, <= 15 columns at ldm = roundup(M, 16)); a NaN/Inf left in the
padding therefore cannot reach dW.

Guard: the first call per device also computes the production result
(`reference()`, the call site's fp8_wgrad_nn) and requires the custom result to be
all-finite with rel_l2 <= 1e-4 (wgrad_ck.guard_metrics, wgrad_ck._GUARD_TOL), then
prints "wgrad_ck_w3 guard out_projection on cuda:<i>: ... -> armed" and
"ARMED wgrad_ck w3 on cuda:<i>". A mismatch prints "-> rejected"; a mismatch or
ANY exception latches the custom kernel off for the process ("wgrad_ck_w3
DISABLED: <reason>") and the caller runs hipBLASLt for that call and every later
one. W1/W2 and W3 latch independently.

Launch status: a stale HIP error that an earlier call left on this thread (the
last error is sticky) is drained before the padding fill and by the C entry
(returned negated); it is logged once ("stale HIP error ... drained") and is not
a failure (as wgrad_ck, r14 iter38 review S1).

Numerics (declared): fp32-order. e4m3 x e4m3 products are exact in fp32; only the
fp32 accumulation order differs from hipBLASLt. Deterministic (fixed reduction
order, no atomics): bitwise-reproducible run to run.

Step-tiled dout_t (B13, Arbor r14 iteration 39): DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED=1
(default off; read once at import; effective only with DLRM_HSTU_FP8_BWD_WGRAD_CK_W3=1).
The out-proj backward asks `tiled_active()` BEFORE its dual cast; when True (knob on,
not latched, gfx950, library loaded, y8 inside the contract) the cast writes the
token-major e4m3 copy STEP-TILED, a 3-D tensor [ceil(M/128), 512, 128] (step k =
tokens [128k, 128k+128) of all 512 rows, zero tail tokens [M, ceil(M/128)*128);
bwd_fp8.quantize_row_major_dual(t_tiled=True)), and `run_w3` reads it with the entry
wgrad_ck_w3_t (kernel wgrad_ck_w3_t_kernel<1536, 192, 16>: no row stride, so none of
the row-major kernel's stride sensitivity; bitwise-identical output to the row-major
entry). hipBLASLt cannot read that layout: every other consumer (the first-call
guard's reference, hipBLASLt after a decline or the latch, the TN fallback, the SNR
probe) re-lays it out with `untile_to_row_major` (bitwise the 2-D dual-cast bytes).
The tiled entry has its own first-call guard per device (line "... dout_t=step-tiled
... -> armed", then "ARMED wgrad_ck w3 on cuda:<i> (dout_t step-tiled)"); a failure
latches the whole W3 kernel off, after which the cast writes the 2-D layout again.

Robustness (Arbor r14 iteration 40). A decline of a step-tiled dout_t turns tiling off for the
rest of the process (`tiled_active` False from then on, logged once): a steady decline would
otherwise re-lay out every call while the cast keeps tiling. The re-laid-out copy for hipBLASLt
comes from the caller's allocator (`untile_to_row_major` `empty`; triton_hstu_linear passes
fp8_tm_alloc.empty_copy, the B11 pool when active). When that copy is out of memory on the
first-call guard, `reference()` raises wgrad_ck.GuardDeferred: `run_w3` neither arms nor latches
(no verdict, logged once; the next call runs the guard again) and returns the custom result of
this call if it is all-finite (no allocation-free fallback exists at the out-proj site: the bf16
y is not recomputed in FP8 mode and the TN fallback needs the copy too), else latches as a
rejected guard.

x21 retile of the step-tiled entry (Arbor r14 iterations 41-42, ship13 DESIGN row B18):
DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_T256S=1 (default off; read once at import; effective only with
DLRM_HSTU_FP8_BWD_WGRAD_CK_W3=1 AND DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED=1, else it prints
"requested: wgrad_ck_w3_t256s=1 (INERT ...)" and changes nothing). A step-tiled dout_t is then read
by the entry wgrad_ck_w3_t256s instead of wgrad_ck_w3_t: kernel wgrad_ck_w3_t256s_kernel<1536, 21, 2>
(W1's 256 x 256 output tile, NSPLIT=21 token splits x 12 tiles = 252 WGs on 252 of the 256 CUs: 16
XCD-local cohorts + 5 cohorts spilled over the leftover CUs, column-major tile order inside a spilled
cohort) + wgrad_ck_w3_reduce<1536, 21>. Its fp32 partials [21, 512, 1536] (63 MiB) live in the same
persistent per-(device, stream) workspace, which is allocated with the split count of the first entry
that runs on that stream and grows to 21 splits on the first t256s call (a 16-split entry uses the
first 16 splits of a larger workspace). Numerics: fp32-order, deterministic (bitwise run to run); NOT
bitwise to wgrad_ck_w3_t: the same 128-token MFMA chunks, summed sequentially within a split, then a
different fixed-order 21-way split sum. Own first-call guard per device (same hipBLASLt reference and
tolerance; lines "... dout_t=step-tiled entry=wgrad_ck_w3_t256s rel=... -> armed" and "ARMED wgrad_ck
w3 on cuda:<i> (dout_t step-tiled) entry=wgrad_ck_w3_t256s"); declines, the latch (which turns the
whole W3 kernel off), GuardDeferred and S2 / S3 behave exactly as for wgrad_ck_w3_t. A 2-D dout_t
still runs the row-major entry wgrad_ck_w3. With the knob off nothing changes: the entry is bound at
load but never launched.

E-RM row-major dout_x entry (Arbor r14 iterations 53 / 55 / 62; DLRM_HSTU_FP8_BWD_ERM=1, yambda_5b.gin
item (97), default off; read once at import; ops/triton/dy_erm.py). The bottom layer's dY producer
writes the e4m3 wgrad operand ROW-MAJOR, x_rm [M, 512] (no dual cast, no step tiling, no zero tail),
and W3 reads it with wgrad_ck_w3_t256s_xt2 (kernel wgrad_ck_w3_t256s_xt2_kernel<1536, 21, 2, 1> +
wgrad_ck_w3_reduce<1536, 21>: x21 with the X tile DMA'd row-major and read by ds_read_b64_tr_b8; the
same values in the same K order and split map, so the same output bits as x21 on the step-tiled copy;
rows >= M range-masked in-kernel). The entry lives in its OWN source and library,
wgrad_tm_ck_w3_erm.cu -> libwgrad_ck_w3_erm.so (= wgrad_tm_ck_w3.cu verbatim + the r14 iteration-53
xt2 block, the byte-identical source both E-RM labs compile), built and bound only with the knob on,
by prebuild() before RUN_START: with the knob off the shipped library, its source, its cache key and
its load path are exactly today's. `rm_inactive_reason(device)` is None ONLY where the knob-off W3
call would run the ARMED x21 entry on a step-tiled dout_t (W3 on and not latched, tiling requested
and not declined, T256S on, the device checked by tiled_active and x21 armed on it, the xt2 entry
bound and not latched). `run_w3_rm` takes the layout explicitly (a 2-D [M, 512] x_rm and a 2-D
[512, ldm] dout_t are ambiguous at M = 512). First call per device: xt2 must be BITWISE equal to the
armed x21 on the step-tiled re-lay-out of x_rm (`rm_to_tiled`, the dual cast's dout_t bytes; lines
"wgrad_ck_w3 guard out_projection on cuda:<i>: N=1536 tokens=<M> dout_x=row-major
entry=wgrad_ck_w3_t256s_xt2 bitwise=x21 -> armed" and "ARMED wgrad_ck w3 on cuda:<i> (dout_x
row-major) entry=wgrad_ck_w3_t256s_xt2"). Every row-major failure (bind, contract decline, launch
error, guard mismatch) latches ONLY the row-major entry ("wgrad_ck_w3 dout_x row-major OFF: ...",
never `_disable`, so x21 and every shipped entry are untouched) and the failing call returns x21's
own result on `rm_to_tiled(x_rm)`, i.e. the knob-off bits; None only when that x21 call itself
fails (its shipped semantics), and the caller then runs hipBLASLt on `rm_to_tok(x_rm)` (bitwise
`untile_to_row_major` of the dual cast's dout_t). The re-lay-out copies come from
fp8_tm_alloc.empty_copy, as the shipped transient copies (the B11 pool's mapped spare block, else
torch.empty; out of memory propagates WITHOUT latching the pool). A guard re-lay-out that is out of
memory defers the guard exactly as GuardDeferred does for x21: nothing armed or latched and the
xt2 result when it is all-finite; a non-finite one latches the row-major entry and returns None
without retrying that copy (the caller's hipBLASLt fallback on rm_to_tok, as x21's non-finite
deferral returns None to _dout_row_major(required=True)). A fallback re-lay-out (after a decline
or a launch error) has no allocation-free alternative: out of memory there raises, as the shipped
hipBLASLt fallback's required copy does.
"""

import ctypes
import os
import threading
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
from generative_recommenders.ops.triton import wgrad_ck

_REQUESTED: bool = os.environ.get("DLRM_HSTU_FP8_BWD_WGRAD_CK_W3", "0") == "1"
# B13 (iteration 39): the dual cast writes dout_t step-tiled for the custom kernel.
_TILED_REQUESTED: bool = (
    os.environ.get("DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED", "0") == "1"
)
# x21 (iteration 42): a step-tiled dout_t runs wgrad_ck_w3_t256s (inert without _CK_W3 + _TILED).
_T256S_REQUESTED: bool = (
    os.environ.get("DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_T256S", "0") == "1"
)

_SOURCE: Path = Path(__file__).with_name("wgrad_tm_ck_w3.cu")
_LIB_NAME: str = "libwgrad_ck_w3.so"
_ENTRY: str = "wgrad_ck_w3"
_ENTRY_T: str = "wgrad_ck_w3_t"  # step-tiled dout_t [ceil(M/128), 512, 128]
_ENTRY_S: str = "wgrad_ck_w3_t256s"  # x21: step-tiled dout_t, 256 x 256 tiles, 21 splits
_NSPLIT: int = 16
_NSPLIT_S: int = 21  # wgrad_ck_w3_t256s_nsplit()
_KX: int = 512  # output rows (dout features)
_N: int = 1536  # output columns (y8 features)
_TOKEN_ALIGN: int = 16  # dout_t ldm granularity (16-token DMA chunks)
_KSTEP: int = 128  # tokens per K step: dout_t is read up to roundup(M, 128)
_LDM_MAX: int = 0x7FFFFFFF // 256  # 31-bit buffer range (wgrad_tm_ck_w3.cu WGCK3_LDM_MAX)
_SITE: str = "out_projection"
# E-RM (DLRM_HSTU_FP8_BWD_ERM, default off; see the module docstring): the row-major dout_x entry,
# its own source / library, built and bound only with the knob on.
_RM_REQUESTED: bool = os.environ.get("DLRM_HSTU_FP8_BWD_ERM", "0") == "1"
_SOURCE_RM: Path = Path(__file__).with_name("wgrad_tm_ck_w3_erm.cu")
_LIB_NAME_RM: str = "libwgrad_ck_w3_erm.so"
_ENTRY_RM: str = "wgrad_ck_w3_t256s_xt2"  # x21 + the X tile read row-major (ds_read_b64_tr_b8)
_NSPLIT_RM: int = 21  # wgrad_ck_w3_t256s_xt2_nsplit()
_M_RM_MAX: int = 0x7FFFFFFF // _KX  # x_rm rows: M * 512 <= 2^31 - 1

_lock = threading.Lock()
_fn: Optional[Callable[..., int]] = None
_fn_t: Optional[Callable[..., int]] = None  # wgrad_ck_w3_t (step-tiled dout_t)
_fn_s: Optional[Callable[..., int]] = None  # wgrad_ck_w3_t256s (x21; step-tiled dout_t)
_drain: Optional[Callable[[], int]] = None  # wgrad_ck_w3_drain_last_error
_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[Path] = None
_broken: Optional[str] = None
_override: Optional[bool] = None
_tiled_override: Optional[bool] = None
_t256s_override: Optional[bool] = None
_armed: set = set()  # device indices that passed the guard
_armed_t: set = set()  # device indices whose step-tiled entry passed its guard
_armed_s: set = set()  # device indices whose wgrad_ck_w3_t256s entry passed its guard
_tiled_ok: set = set()  # device indices tiled_active() checked (gfx950, library loaded)
# (device index, stream) -> [nsplit, 512, 1536] fp32; nsplit = 16, or the largest split count of the
# entries that ran there (21: wgrad_ck_w3_t256s); never shrinks
_workspaces: Dict[Tuple[int, int], torch.Tensor] = {}
_once: set = set()
# Set by `run_w3` on the first decline of a step-tiled dout_t: the cast stops tiling (S3).
_tiled_declined: Optional[str] = None
# E-RM row-major entry state (knob on only): its library and binding, the per-device bitwise
# guard, and its own latch (never the shipped `_broken` / `_tiled_declined`).
_rm_override: Optional[bool] = None
_fn_rm: Optional[Callable[..., int]] = None  # wgrad_ck_w3_t256s_xt2
_drain_rm: Optional[Callable[[], int]] = None
_lib_rm: Optional[ctypes.CDLL] = None
_lib_rm_path: Optional[Path] = None
_rm_declined: Optional[str] = None
_armed_rm: set = set()  # device indices whose xt2 entry passed its bitwise guard vs x21


def log(message: str) -> None:
    # Rank-local, flushed; the out-proj wgrad's own prefix (triton_hstu_linear).
    print(f"[fp8-out-wgrad] {message}", flush=True)


def _log_once(key: str, message: str) -> None:
    if key in _once:
        return
    _once.add(key)
    log(message)


def log_once(key: str, message: str) -> None:
    """One line per key per process (call sites: triton_hstu_linear)."""
    _log_once(key, message)


if _REQUESTED:
    log("requested: wgrad_ck_w3=1")
if _TILED_REQUESTED:
    log(
        "requested: wgrad_ck_w3_tiled=1"
        + ("" if _REQUESTED else " (INERT without DLRM_HSTU_FP8_BWD_WGRAD_CK_W3=1)")
    )
if _T256S_REQUESTED:
    log(
        "requested: wgrad_ck_w3_t256s=1"
        + (
            ""
            if (_REQUESTED and _TILED_REQUESTED)
            else " (INERT without DLRM_HSTU_FP8_BWD_WGRAD_CK_W3=1 and "
            "DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED=1)"
        )
    )


def requested() -> bool:
    """The knob (or the test-only override) asks for the custom W3 kernel."""
    return _REQUESTED if _override is None else _override


def tiled_requested() -> bool:
    """B13: DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED (or its test-only override) AND the
    W3 knob: the out-proj dual cast should write dout_t step-tiled."""
    tiled = _TILED_REQUESTED if _tiled_override is None else _tiled_override
    return tiled and requested()


def set_tiled_requested_for_testing(value: Optional[bool]) -> None:
    """Test-only: override the import-time tiled knob (None restores it)."""
    global _tiled_override
    _tiled_override = value


def t256s_requested() -> bool:
    """x21: DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_T256S (or its test-only override) AND
    tiled_requested(): a step-tiled dout_t runs the entry wgrad_ck_w3_t256s."""
    t256s = _T256S_REQUESTED if _t256s_override is None else _t256s_override
    return t256s and tiled_requested()


def set_t256s_requested_for_testing(value: Optional[bool]) -> None:
    """Test-only: override the import-time t256s knob (None restores it)."""
    global _t256s_override
    _t256s_override = value


def enabled() -> bool:
    """Requested and not latched off: the NN-mode out-proj wgrad tries it first."""
    return requested() and _broken is None


def broken() -> Optional[str]:
    return _broken


def set_requested_for_testing(value: Optional[bool]) -> None:
    """Test-only: override the import-time knob (None restores it)."""
    global _override
    _override = value


def reset_for_testing(unload: bool = False) -> None:
    """Test-only: clear the latch, the per-device arming and the once-lines.

    unload=True also forgets the loaded library (the next call builds/loads again,
    e.g. after changing DLRM_HSTU_FP8_BWD_WGRAD_CK_HIPCC / _CACHE).
    """
    global _broken, _fn, _fn_t, _fn_s, _drain, _lib, _lib_path, _tiled_declined
    global _rm_declined, _fn_rm, _drain_rm, _lib_rm, _lib_rm_path
    _broken = None
    _tiled_declined = None
    _rm_declined = None
    _armed.clear()
    _armed_t.clear()
    _armed_s.clear()
    _armed_rm.clear()
    _tiled_ok.clear()
    _once.clear()
    if unload:
        with _lock:
            _fn = None
            _fn_t = None
            _fn_s = None
            _drain = None
            _lib = None
            _lib_path = None
            _fn_rm = None
            _drain_rm = None
            _lib_rm = None
            _lib_rm_path = None


def _disable(reason: str) -> None:
    global _broken
    if _broken is not None:
        return
    _broken = reason
    log(f"wgrad_ck_w3 DISABLED: {reason}")


def _load() -> None:
    global _fn, _fn_t, _fn_s, _drain, _lib, _lib_path
    if _fn is not None:
        return
    with _lock:
        if _fn is not None:
            return
        so = wgrad_ck.build(_SOURCE, _LIB_NAME)
        lib = ctypes.CDLL(str(so))
        fn = wgrad_ck.bind_entry(lib, so, _ENTRY, _NSPLIT)
        fn_t = wgrad_ck.bind_entry(lib, so, _ENTRY_T, _NSPLIT)
        # x21: bound (and its nsplit checked) at load, also before RUN_START by
        # prebuild(); launched only when t256s_requested().
        fn_s = wgrad_ck.bind_entry(lib, so, _ENTRY_S, _NSPLIT_S)
        drain = getattr(lib, f"{_ENTRY}_drain_last_error")
        drain.restype = ctypes.c_int
        drain.argtypes = []
        _lib_path = so
        _lib = lib
        _drain = drain
        _fn_s = fn_s
        _fn_t = fn_t
        _fn = fn


def _log_stale(code: int) -> None:
    _log_once(
        f"stale:{code}",
        f"wgrad_ck_w3: stale HIP error {code} drained before a {_ENTRY} launch "
        "(left on this thread by an earlier call, not ours)",
    )


def library_path() -> Optional[Path]:
    return _lib_path


def prebuild() -> bool:
    """Build and load before RUN_START. False (latched off, logged) on failure or
    when not requested; touches no tensor data."""
    if not enabled():
        return False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU")
        wgrad_ck._check_arch(torch.device("cuda", torch.cuda.current_device()))
        _load()
    except Exception as error:
        _disable(f"{type(error).__name__}: {error}")
        return False
    if t256s_requested():
        _log_once(
            "prebuild_t256s",
            f"prebuild: {_ENTRY_S} bound (step-tiled dout_t, nsplit={_NSPLIT_S})",
        )
    # E-RM (knob on only): the row-major entry's own library, built and bound here,
    # before RUN_START (never after it: without this, rm_inactive_reason declines).
    if rm_requested() and t256s_requested() and _load_rm():
        _log_once(
            "prebuild_rm",
            f"prebuild: {_ENTRY_RM} bound (row-major dout_x, nsplit={_NSPLIT_RM}, "
            f"{_LIB_NAME_RM})",
        )
    return True


def _decline_reason(
    y8: torch.Tensor,
    dout_t: torch.Tensor,
    scale_y: torch.Tensor,
    scale_d: torch.Tensor,
    m: int,
) -> Optional[Tuple[str, str]]:
    """(category, detail) when this call is outside the kernel's contract."""
    if m < 1:
        return "M", f"M={m}"
    if y8.dtype != torch.float8_e4m3fn or dout_t.dtype != torch.float8_e4m3fn:
        return "dtype", f"y8 {y8.dtype} dout_t {dout_t.dtype}"
    if y8.dim() != 2 or dout_t.dim() not in (2, 3):
        return "rank", f"y8 {tuple(y8.shape)} dout_t {tuple(dout_t.shape)}"
    if int(y8.shape[1]) != _N or y8.stride() != (_N, 1) or int(y8.shape[0]) < m:
        return "y8", f"y8 shape {tuple(y8.shape)} strides {y8.stride()} M={m}"
    if dout_t.dim() == 3:
        # B13 step-tiled [ceil(M/128), 512, 128], contiguous (wgrad_ck_w3_t).
        kt = (m + _KSTEP - 1) // _KSTEP
        # is_contiguous: dense [KT][512][128] (ignores the stride of a KT == 1 dim)
        if tuple(dout_t.shape) != (kt, _KX, _KSTEP) or not dout_t.is_contiguous():
            return (
                "dout_t",
                f"step-tiled dout_t shape {tuple(dout_t.shape)} strides {dout_t.stride()} "
                f"M={m} expected ({kt}, {_KX}, {_KSTEP}) contiguous",
            )
    else:
        ldm = int(dout_t.stride(0))
        mp = (m + _TOKEN_ALIGN - 1) // _TOKEN_ALIGN * _TOKEN_ALIGN
        if (
            int(dout_t.shape[0]) != _KX
            or dout_t.stride(1) != 1
            or int(dout_t.shape[1]) != ldm
            or ldm % _TOKEN_ALIGN != 0
            or ldm < mp
            or ldm > _LDM_MAX
        ):
            return (
                "dout_t",
                f"dout_t shape {tuple(dout_t.shape)} strides {dout_t.stride()} M={m} Mp16={mp}",
            )
    if not (wgrad_ck._aligned(y8) and wgrad_ck._aligned(dout_t)):
        return "align", f"y8 ptr {y8.data_ptr():#x} dout_t ptr {dout_t.data_ptr():#x}"
    device = y8.device
    if device.type != "cuda":
        return "device", f"y8 on {device}"
    for name, t in (("dout_t", dout_t), ("scale_y", scale_y), ("scale_d", scale_d)):
        if t.device != device:
            return "device", f"{name} on {t.device}, y8 on {device}"
    for name, t in (("scale_y", scale_y), ("scale_d", scale_d)):
        if t.dtype != torch.float32 or t.numel() != 1:
            return "scale", f"{name} {t.dtype} numel={t.numel()}"
    return None


def _launch(
    y8: torch.Tensor,
    dout_t: torch.Tensor,
    scale_y: torch.Tensor,
    scale_d: torch.Tensor,
    m: int,
    index: int,
    t256s: bool = False,
) -> torch.Tensor:
    tiled = dout_t.dim() == 3
    assert tiled or not t256s
    fn = _fn_s if t256s else _fn_t if tiled else _fn
    drain = _drain
    assert fn is not None and drain is not None
    nsplit = _NSPLIT_S if t256s else _NSPLIT
    # step-tiled: the entry's `ldm` argument is the step stride 512*128 (checked by it)
    ldm = _KX * _KSTEP if tiled else int(dout_t.stride(0))
    with torch.cuda.device(index):
        stream = torch.cuda.current_stream(y8.device).cuda_stream
        key = (index, int(stream))
        workspace = _workspaces.get(key)
        # sized by the entry's split count (16; 21 for t256s); a larger one
        # is reused (each entry writes, then reduces, only its own first `nsplit` splits)
        if workspace is None or int(workspace.shape[0]) < nsplit:
            workspace = torch.empty(
                (nsplit, _KX, _N), dtype=torch.float32, device=y8.device
            )
            _workspaces[key] = workspace
        # A stale HIP error an earlier call left on this thread (sticky last error)
        # would fail the torch fill below: drain it first (logged once, not ours).
        stale = int(drain())
        if stale != 0:
            _log_stale(stale)
        # Padding the kernel reads (see the module docstring): zero it.
        if tiled:
            # step-tiled: tokens [M, KT*128) = the last step's columns from M % 128 on
            # (the dual cast already wrote zeros there).
            rem = m - (int(dout_t.shape[0]) - 1) * _KSTEP
            if rem < _KSTEP:
                dout_t[-1, :, rem:].zero_()
        else:
            hi = min(ldm, (m + _KSTEP - 1) // _KSTEP * _KSTEP)
            if hi > m:
                dout_t[:, m:hi].zero_()
        out = torch.empty((_KX, _N), dtype=torch.float32, device=y8.device)
        rc = fn(
            y8.data_ptr(),
            dout_t.data_ptr(),
            ldm,
            workspace.data_ptr(),
            out.data_ptr(),
            scale_y.data_ptr(),
            scale_d.data_ptr(),
            int(m),
            stream,
        )
    if rc < 0:
        # Both kernels launched; the entry drained a stale error that an earlier
        # call had left on this thread: not a launch failure.
        _log_stale(-rc)
    elif rc != 0:
        entry = _ENTRY_S if t256s else _ENTRY
        raise RuntimeError(f"{entry} launch returned hip error {rc}")
    return out


def run_w3(
    y8: torch.Tensor,
    dout_t: torch.Tensor,
    scale_y: torch.Tensor,
    scale_d: torch.Tensor,
    m: int,
    reference: Callable[[], torch.Tensor],
) -> Optional[torch.Tensor]:
    """dW fp32 [512, 1536] (= fp8_wgrad_nn's dw_t), or None: the caller runs hipBLASLt.

    None when not enabled (knob off or latched), when this call is outside the
    contract (logged once per reason), when the first-call guard rejects
    (latched) or on any exception (latched). `reference()` returns the
    production (hipBLASLt NN) dw_t of this call (or raises), computed on the
    row-major operand (`untile_to_row_major(dout_t, m)` for a step-tiled one); it
    is called only by the first-call guard of each device and layout.

    dout_t is either the row-major [512, ldm] operand (entry wgrad_ck_w3) or the
    B13 step-tiled [ceil(M/128), 512, 128] one (entry wgrad_ck_w3_t; same output
    bits). A None for a step-tiled dout_t means the caller must re-lay it out
    (`untile_to_row_major`) before running hipBLASLt. A declined step-tiled dout_t
    also turns tiling off (`tiled_active`). A wgrad_ck.GuardDeferred from
    `reference()`: nothing armed or latched, and the (unverified) custom result is
    returned when it is all-finite (see the module docstring), else latched.
    """
    global _tiled_declined
    if not enabled():
        return None
    m = int(m)
    why = _decline_reason(y8, dout_t, scale_y, scale_d, m)
    if why is not None:
        _log_once(
            f"decline:{why[0]}",
            f"wgrad_ck_w3 declined {_SITE} ({why[0]}: {why[1]}) -> hipBLASLt for such calls",
        )
        if dout_t.dim() == 3 and _tiled_declined is None:
            _tiled_declined = f"{why[0]}: {why[1]}"
            log(
                f"wgrad_ck_w3 dout_t tiling OFF: a step-tiled dout_t was declined "
                f"({_tiled_declined}) -> the dual cast writes the 2-D layout from now on"
            )
        return None
    device = y8.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    tiled = dout_t.dim() == 3
    # a step-tiled dout_t runs wgrad_ck_w3_t256s when its knob is on (with its own
    # guard), else wgrad_ck_w3_t
    t256s = tiled and t256s_requested()
    armed = _armed_s if t256s else (_armed_t if tiled else _armed)
    layout = " dout_t=step-tiled" if tiled else ""
    if t256s:
        layout += f" entry={_ENTRY_S}"
    out: Optional[torch.Tensor] = None
    try:
        first = index not in armed
        if first:
            wgrad_ck._check_arch(device)
        _load()
        if t256s:
            out = _launch(y8, dout_t, scale_y, scale_d, m, index, t256s=True)
        else:
            out = _launch(y8, dout_t, scale_y, scale_d, m, index)
        if first:
            rel, finite = wgrad_ck.guard_metrics(out, reference())
            line = (
                f"wgrad_ck_w3 guard {_SITE} on cuda:{index}: N={_N} tokens={m}{layout} "
                f"rel={rel:.3e}"
            )
            if not (finite and rel <= wgrad_ck._GUARD_TOL):
                log(line + (" -> rejected" if finite else " finite=0 -> rejected"))
                _disable(
                    f"guard rejected {_SITE} on cuda:{index}: N={_N} tokens={m}{layout} "
                    f"rel={rel:.3e} finite={int(finite)}"
                )
                return None
            log(line + " -> armed")
            armed.add(index)
            log(
                f"ARMED wgrad_ck w3 on cuda:{index}"
                + (" (dout_t step-tiled)" if tiled else "")
                + (f" entry={_ENTRY_S}" if t256s else "")
            )
        return out
    except wgrad_ck.GuardDeferred as error:
        # The guard's reference could not be computed for this call only (its
        # re-laid-out copy is out of memory): no verdict, nothing armed.
        finite = out is not None and bool(torch.isfinite(out).all())
        _log_once(
            f"guard_deferred:{index}{layout}",
            f"wgrad_ck_w3 guard {_SITE} on cuda:{index}:{layout} deferred ({error}) -> "
            + (
                "no verdict, nothing latched; this call uses the custom result "
                "(all-finite, unverified), the next call runs the guard again"
                if finite
                else "the custom result is not all-finite -> latched"
            ),
        )
        if not finite:
            _disable(f"guard deferred on cuda:{index}{layout} and the custom result is not all-finite")
            return None
        return out
    except Exception as error:
        _disable(f"{type(error).__name__}: {error}")
        return None


def tiled_active(
    device: torch.device, y8: Optional[torch.Tensor] = None, kx: Optional[int] = None
) -> bool:
    """B13: should the out-proj dual cast write dout_t step-tiled for this call?

    Asked BEFORE the cast, from the latch state, so the steady state never re-lays
    the operand out: True iff tiled_requested(), enabled() (W3 knob on, not
    latched), a gfx950 device and the library loaded (loads it on the first ask
    per device -- prebuild() normally did; a failure latches W3 off like run_w3)
    and, when given, y8 inside run_w3's contract and the dout width `kx` == 512
    (otherwise run_w3 would decline every call and each one would re-lay out).
    """
    if not (tiled_requested() and enabled()) or _tiled_declined is not None:
        return False
    if kx is not None and int(kx) != _KX:
        return False
    if y8 is not None and (
        y8.dtype != torch.float8_e4m3fn
        or y8.dim() != 2
        or int(y8.shape[0]) < 1
        or int(y8.shape[1]) != _N
        or y8.stride() != (_N, 1)
        or not wgrad_ck._aligned(y8)
        or y8.device != device
    ):
        return False
    if device.type != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _tiled_ok and _fn_t is not None:
        return True
    try:
        wgrad_ck._check_arch(device)
        _load()
    except Exception as error:
        _disable(f"{type(error).__name__}: {error}")
        return False
    _tiled_ok.add(index)
    return True


def untile_to_row_major(
    dout_t: torch.Tensor, m: int, empty: Callable[..., torch.Tensor] = torch.empty
) -> torch.Tensor:
    """B13: the row-major token-major operand of a step-tiled dout_t.

    [ceil(M/128), 512, 128] -> [512, roundup(M, 16)] contiguous (the dual cast's
    pad_to=16 layout). A byte copy: tokens < M are the same bytes and tokens
    [M, roundup(M, 16)) are the tiled copy's zero tail, so the result is bitwise
    the 2-D dual cast's output and hipBLASLt reads exactly what it reads without
    tiling. For every consumer other than wgrad_ck_w3_t (the first-call guard's
    reference, hipBLASLt after a decline or the latch, the TN fallback, the SNR
    probe). A 2-D dout_t is returned unchanged. The copy is allocated with
    `empty(shape, dtype=, device=)` (triton_hstu_linear: fp8_tm_alloc.empty_copy).
    """
    if dout_t.dim() != 3:
        return dout_t
    m = int(m)
    kt, kx, ks = (int(s) for s in dout_t.shape)
    if ks != _KSTEP or kt != (m + _KSTEP - 1) // _KSTEP or not dout_t.is_contiguous():
        raise ValueError(
            f"untile_to_row_major: dout_t {tuple(dout_t.shape)} strides {dout_t.stride()} "
            f"is not step-tiled for M={m}"
        )
    mp = (m + _TOKEN_ALIGN - 1) // _TOKEN_ALIGN * _TOKEN_ALIGN
    out = empty((kx, mp), dtype=dout_t.dtype, device=dout_t.device)
    src = dout_t.view(torch.uint8).permute(1, 0, 2)  # [512, KT, 128] view
    dst = out.view(torch.uint8)
    full = mp // _KSTEP
    if full:
        dst[:, : full * _KSTEP].view(kx, full, _KSTEP).copy_(src[:, :full])
    if mp > full * _KSTEP:
        dst[:, full * _KSTEP :].copy_(src[:, full, : mp - full * _KSTEP])
    return out


# ---------------------------------------------------------------------------------------------------
# E-RM row-major dout_x entry (DLRM_HSTU_FP8_BWD_ERM, default off; module docstring). Nothing below
# is reached with the knob off: prebuild() binds nothing new, run_w3 / tiled_active / untile_to_row_major
# are unchanged, and only ops/triton/dy_erm.py and triton_hstu_linear's E-RM wrappers call these.
# ---------------------------------------------------------------------------------------------------


def rm_requested() -> bool:
    """DLRM_HSTU_FP8_BWD_ERM (or its test-only override)."""
    return _RM_REQUESTED if _rm_override is None else _rm_override


def set_rm_requested_for_testing(value: Optional[bool]) -> None:
    """Test-only: override the import-time E-RM knob (None restores it)."""
    global _rm_override
    _rm_override = value


def rm_declined() -> Optional[str]:
    return _rm_declined


def _rm_decline(reason: str) -> None:
    """E-RM-only latch: the row-major entry is off for the process; x21, every shipped entry and
    `_broken` / `_tiled_declined` are untouched (never `_disable`)."""
    global _rm_declined
    if _rm_declined is not None:
        return
    _rm_declined = reason
    log(
        f"wgrad_ck_w3 dout_x row-major OFF: {reason} -> the bottom layer runs the dual cast "
        "from its next call (x21 unchanged)"
    )


def _load_rm() -> bool:
    """Build + bind libwgrad_ck_w3_erm.so (wgrad_tm_ck_w3_erm.cu; same hipcc, flags and cache
    root as the shipped library, its own cache entry) and check the entry's split count. Called by
    prebuild() only, with the knob on; any failure latches the row-major entry only."""
    global _fn_rm, _drain_rm, _lib_rm, _lib_rm_path
    if _fn_rm is not None:
        return True
    if _rm_declined is not None:
        return False
    with _lock:
        if _fn_rm is not None:
            return True
        try:
            so = wgrad_ck.build(_SOURCE_RM, _LIB_NAME_RM)
            lib = ctypes.CDLL(str(so))
            fn = wgrad_ck.bind_entry(lib, so, _ENTRY_RM, _NSPLIT_RM)
            drain = getattr(lib, f"{_ENTRY}_drain_last_error")
            drain.restype = ctypes.c_int
            drain.argtypes = []
        except Exception as error:  # noqa: BLE001
            reason = f"bind {_ENTRY_RM}: {type(error).__name__}: {str(error)[:300]}"
        else:
            _lib_rm_path = so
            _lib_rm = lib
            _drain_rm = drain
            _fn_rm = fn
            return True
    _rm_decline(reason)
    return False


def _index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def rm_inactive_reason(device: torch.device) -> Optional[str]:
    """None iff, with the E-RM knob off, the W3 call of this device would run the ARMED x21 entry
    wgrad_ck_w3_t256s on a step-tiled dout_t, and the row-major entry is bound and not latched.
    Host state only (asked by dy_erm's ticket and again by its consumer check)."""
    if not rm_requested():
        return "ERM knob off"
    if _rm_declined is not None:
        return f"row-major entry latched off ({_rm_declined})"
    if not enabled():
        return f"W3 off or latched ({_broken})"
    if not tiled_requested():
        return "B13 step-tiled dout_t not requested"
    if _tiled_declined is not None:
        return f"step-tiled dout_t declined earlier ({_tiled_declined})"
    if not t256s_requested():
        return "x21 (T256S) not requested"
    if device.type != "cuda":
        return f"device {device}"
    index = _index(device)
    if index not in _tiled_ok:
        return "tiled_active not yet checked on this device"
    if index not in _armed_s:
        return "x21 not armed on this device yet"
    if _fn_rm is None:
        return f"{_ENTRY_RM} not bound (built only by the prebuild, before RUN_START)"
    return None


def rm_decline_reason(
    y8: torch.Tensor,
    x_rm: torch.Tensor,
    scale_y: torch.Tensor,
    scale_d: torch.Tensor,
    m: int,
) -> Optional[Tuple[str, str]]:
    """(category, detail) when this call is outside the row-major entry's contract. The layout is
    explicit: x_rm is float8_e4m3fn [M, 512] with strides (512, 1), never inferred from dim()."""
    m = int(m)
    if m < 1 or m > _M_RM_MAX:
        return "M", f"M={m} outside [1, {_M_RM_MAX}]"
    if y8.dtype != torch.float8_e4m3fn or x_rm.dtype != torch.float8_e4m3fn:
        return "dtype", f"y8 {y8.dtype} x_rm {x_rm.dtype}"
    if x_rm.dim() != 2 or tuple(x_rm.shape) != (m, _KX) or tuple(x_rm.stride()) != (_KX, 1):
        return "x_rm", f"x_rm shape {tuple(x_rm.shape)} strides {tuple(x_rm.stride())} M={m}"
    if (
        y8.dim() != 2
        or int(y8.shape[1]) != _N
        or y8.stride() != (_N, 1)
        or int(y8.shape[0]) < m
    ):
        return "y8", f"y8 shape {tuple(y8.shape)} strides {y8.stride()} M={m}"
    if not (wgrad_ck._aligned(y8) and wgrad_ck._aligned(x_rm)):
        return "align", f"y8 ptr {y8.data_ptr():#x} x_rm ptr {x_rm.data_ptr():#x}"
    device = y8.device
    if device.type != "cuda":
        return "device", f"y8 on {device}"
    for name, t in (("x_rm", x_rm), ("scale_y", scale_y), ("scale_d", scale_d)):
        if t.device != device:
            return "device", f"{name} on {t.device}, y8 on {device}"
    for name, t in (("scale_y", scale_y), ("scale_d", scale_d)):
        if t.dtype != torch.float32 or t.numel() != 1:
            return "scale", f"{name} {t.dtype} numel={t.numel()}"
    return None


def _launch_rm(
    y8: torch.Tensor,
    x_rm: torch.Tensor,
    scale_y: torch.Tensor,
    scale_d: torch.Tensor,
    m: int,
    index: int,
) -> torch.Tensor:
    """wgrad_ck_w3_t256s_xt2 on the caller's stream: ldm = 512 (the row stride), the shared
    21-split workspace, NO padding fill (rows >= M are range-masked in-kernel)."""
    fn = _fn_rm
    drain = _drain_rm
    assert fn is not None and drain is not None
    with torch.cuda.device(index):
        stream = torch.cuda.current_stream(y8.device).cuda_stream
        key = (index, int(stream))
        workspace = _workspaces.get(key)
        if workspace is None or int(workspace.shape[0]) < _NSPLIT_RM:
            workspace = torch.empty(
                (_NSPLIT_RM, _KX, _N), dtype=torch.float32, device=y8.device
            )
            _workspaces[key] = workspace
        stale = int(drain())
        if stale != 0:
            _log_stale(stale)
        out = torch.empty((_KX, _N), dtype=torch.float32, device=y8.device)
        rc = fn(
            y8.data_ptr(),
            x_rm.data_ptr(),
            _KX,
            workspace.data_ptr(),
            out.data_ptr(),
            scale_y.data_ptr(),
            scale_d.data_ptr(),
            int(m),
            stream,
        )
    if rc < 0:
        _log_stale(-rc)
    elif rc != 0:
        raise RuntimeError(f"{_ENTRY_RM} launch returned hip error {rc}")
    return out


def run_w3_rm(
    y8: torch.Tensor,
    x_rm: torch.Tensor,
    scale_y: torch.Tensor,
    scale_d: torch.Tensor,
    m: int,
    retile: Callable[[torch.Tensor, int], torch.Tensor],
    run_x21: Callable[[torch.Tensor], Optional[torch.Tensor]],
    is_oom: Callable[[BaseException], bool],
) -> Optional[torch.Tensor]:
    """dW fp32 [512, 1536] for a row-major x_rm, bitwise the knob-off result (module docstring).

    retile(x_rm, m): the step-tiled [ceil(M/128), 512, 128] copy with a zero tail (rm_to_tiled with
    fp8_tm_alloc.empty_copy: out of memory propagates, the B11 pool is never latched by it).
    run_x21(dout_t): today's run_w3 on it (x21 armed: no guard; None on its own failure, latching W3
    as today). is_oom: fp8_tm_alloc.is_oom. None when that x21 call fails, or when the first-call
    guard is deferred by an out-of-memory re-lay-out and the xt2 result is not all-finite (row-major
    entry latched, that copy not retried); the caller then runs hipBLASLt on rm_to_tok(x_rm), as the
    knob-off path does on x21's failure / non-finite deferral. An out-of-memory fallback re-lay-out
    (after a decline or a launch error) raises: no allocation-free fallback exists for that call."""
    m = int(m)
    device = y8.device
    why = rm_inactive_reason(device)
    if why is None:
        decline = rm_decline_reason(y8, x_rm, scale_y, scale_d, m)
        why = None if decline is None else f"{decline[0]}: {decline[1]}"
    if why is not None:
        _rm_decline(why)
        return run_x21(retile(x_rm, m))
    index = _index(device)
    try:
        out = _launch_rm(y8, x_rm, scale_y, scale_d, m, index)
    except Exception as error:  # noqa: BLE001
        _rm_decline(f"{type(error).__name__}: {error}")
        return run_x21(retile(x_rm, m))
    if index in _armed_rm:
        return out
    layout = f" dout_x=row-major entry={_ENTRY_RM}"
    try:
        dout_t = retile(x_rm, m)
    except Exception as error:
        if not is_oom(error):
            raise
        finite = bool(torch.isfinite(out).all())
        _log_once(
            f"rm_guard_deferred:{index}",
            f"wgrad_ck_w3 guard {_SITE} on cuda:{index}:{layout} deferred (re-lay-out copy OOM) -> "
            + (
                "no verdict, nothing latched; this call uses the xt2 result (all-finite, "
                "unverified), the next call runs the guard again"
                if finite
                else "the xt2 result is not all-finite -> row-major entry latched"
            ),
        )
        if finite:
            return out
        # Not all-finite and no reference: latch the row-major entry and do not retry the copy
        # that just ran out of memory. None -> the caller's hipBLASLt on rm_to_tok(x_rm) (its own
        # copy; out of memory there raises), as x21's non-finite deferral in run_w3.
        _rm_decline("guard deferred and the xt2 result is not all-finite")
        return None
    ref = run_x21(dout_t)
    del dout_t
    if ref is None:
        return None  # x21 itself failed: its shipped semantics (W3 latched by run_w3)
    if not torch.equal(out.view(torch.int32), ref.view(torch.int32)):
        log(f"wgrad_ck_w3 guard {_SITE} on cuda:{index}: N={_N} tokens={m}{layout} bitwise=x21 -> rejected")
        _rm_decline(f"guard: {_ENTRY_RM} output != x21 output (bitwise) on cuda:{index} tokens={m}")
        return ref
    _armed_rm.add(index)
    log(f"wgrad_ck_w3 guard {_SITE} on cuda:{index}: N={_N} tokens={m}{layout} bitwise=x21 -> armed")
    log(f"ARMED wgrad_ck w3 on cuda:{index} (dout_x row-major) entry={_ENTRY_RM}")
    return out


def rm_to_tiled(
    x_rm: torch.Tensor, m: int, empty: Callable[..., torch.Tensor] = torch.empty
) -> torch.Tensor:
    """The step-tiled [ceil(M/128), 512, 128] copy of a row-major x_rm [M, 512]: token t, column c
    at (t // 128, c, t % 128), tokens [M, ceil(M/128)*128) zero, i.e. the dual cast's dout_t bytes
    (bwd_fp8.quantize_row_major_dual(t_tiled=True)). A byte copy; allocated with
    `empty(shape, dtype=, device=)` (triton_hstu_linear passes fp8_tm_alloc.empty_copy)."""
    m = int(m)
    if x_rm.dim() != 2 or tuple(x_rm.shape) != (m, _KX) or not x_rm.is_contiguous():
        raise ValueError(
            f"rm_to_tiled: x_rm {tuple(x_rm.shape)} strides {tuple(x_rm.stride())} is not "
            f"row-major [{m}, {_KX}]"
        )
    kt = (m + _KSTEP - 1) // _KSTEP
    out = empty((kt, _KX, _KSTEP), dtype=x_rm.dtype, device=x_rm.device)
    dst = out.view(torch.uint8)
    src = x_rm.view(torch.uint8)
    full = m // _KSTEP
    if full:
        dst[:full].copy_(src[: full * _KSTEP].view(full, _KSTEP, _KX).transpose(1, 2))
    rem = m - full * _KSTEP
    if rem:
        dst[full, :, :rem].copy_(src[full * _KSTEP :].t())
        dst[full, :, rem:].zero_()
    return out


def rm_to_tok(
    x_rm: torch.Tensor, m: int, empty: Callable[..., torch.Tensor] = torch.empty
) -> torch.Tensor:
    """The row-major token-major operand [512, roundup(M, 16)] of a row-major x_rm, tokens
    [M, roundup(M, 16)) zero: bitwise `untile_to_row_major(rm_to_tiled(x_rm, m), m)`, i.e. what a
    hipBLASLt reader of the knob-off path reads. A byte copy (`empty` as in untile_to_row_major)."""
    m = int(m)
    if x_rm.dim() != 2 or tuple(x_rm.shape) != (m, _KX) or not x_rm.is_contiguous():
        raise ValueError(
            f"rm_to_tok: x_rm {tuple(x_rm.shape)} strides {tuple(x_rm.stride())} is not "
            f"row-major [{m}, {_KX}]"
        )
    mp = (m + _TOKEN_ALIGN - 1) // _TOKEN_ALIGN * _TOKEN_ALIGN
    out = empty((_KX, mp), dtype=x_rm.dtype, device=x_rm.device)
    dst = out.view(torch.uint8)
    dst[:, :m].copy_(x_rm.view(torch.uint8).t())
    if mp > m:
        dst[:, m:].zero_()
    return out
