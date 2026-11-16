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

"""Custom FP8 K=tokens weight-gradient kernel ("wgrad_ck") for the fp8t uvqk / vk wgrad.

DLRM_HSTU_FP8_BWD_WGRAD_CK=1 (default off; Arbor r14 iteration 38; read once at import).

What: `bwd_fp8._run_tm_wgrad` computes dW [512, N] fp32 = gs*as * xT[512, :M] @ dz8[:M, N] for
the uvqk projection (W1, N=2048, once per ordinary layer) and the targets-only vk projection
(W2, N=1024) through the hipBLASLt extension (fp8_wgrad_tm). With this knob it runs
wgrad_tm_ck.cu instead: split-K over XCD cohorts, one 256 x 256 tile per workgroup, each
token chunk of both operands fetched from HBM once per cohort and re-served from the XCD L2
(the lab EA census read 1.20x the minimum bytes vs hipBLASLt's 2.40x), fp32 partials
[NSPLIT, 512, N] in a persistent per-(device, N) workspace (64 MiB) and a fixed-order
reduction kernel that applies gs*as. Kernels in a trace: wgrad_ck_kernel<2048, 16>,
wgrad_ck_reduce<2048, 16> (W1) and wgrad_ck_kernel<1024, 32>, wgrad_ck_reduce<1024, 32> (W2).

How: `build()` compiles the .cu with hipcc (DLRM_HSTU_FP8_BWD_WGRAD_CK_HIPCC, else
$ROCM_PATH/bin/hipcc, else /opt/rocm/bin/hipcc) into
/tmp/dlrm_hstu_fp8_wgrad_ck/<sha16 of source + flags + arch + `hipcc --version`>/libwgrad_ck.so
(DLRM_HSTU_FP8_BWD_WGRAD_CK_CACHE overrides the root). A finished cache entry skips the
build; concurrent ranks build to host-, pid- and thread-unique temporary files and os.replace
them into place.
`prebuild()` runs that before RUN_START (bwd_fp8.prebuild_uvqk_extension, gin
DLRM_HSTU_FP8_PREBUILD). The library is loaded with ctypes and launched on the caller's
current stream. The partial workspace is kept per (device, N, stream), so reuse is
always stream-ordered (training runs one backward stream per device).

Contract (`run` declines, i.e. returns None and the caller runs hipBLASLt, otherwise):
dz8 float8_e5m2, 2-D, strides (N, 1), N in {2048, 1024}, >= M rows (rows >= M are never
read); xT float8_e4m3fn [512, >= roundup(M, 128)], strides (ldm, 1), ldm >= roundup(M, 128),
ldm % 16 == 0, ldm < 2^32 / 256; 16-byte aligned data; gs / as single-element fp32 tensors on
the same device; M >= 1; a gfx950 device. The xT tail tokens [M, roundup(M, 128)) are read
and multiplied by zero rows, so they must be finite: the production producers (LN store_t /
store_t_v2 / recompute_fp8t) zero-fill them.

Guard: the first call per (device, N) also computes the hipBLASLt result (the caller's
`reference`, which is that call's own hipBLASLt wgrad with its own fp64-checked arming) and
requires the custom result to be all-finite with rel_l2 <= 1e-4, then prints
"wgrad_ck guard <site> on cuda:<i>: ... -> armed" and "ARMED wgrad_ck <w1|w2> on cuda:<i>".
A mismatch prints "-> rejected"; a mismatch or ANY exception latches the custom kernel off
for the process ("wgrad_ck DISABLED: <reason>") and the caller uses hipBLASLt for that call
and every later one.

Numerics (declared): fp32-order relative to hipBLASLt. Both kernels feed the same 128-token
chunks to the same v_mfma_f32_16x16x128_f8f6f4 instruction, whose internal 128-wide dot
product dominates the error against an exact fp64 GEMM of the same fp8 operands (rel_l2
~1.2e-5 Gaussian, ~6e-5 heavy-tailed, independent of M and equal for both kernels; r14 iter38
review rv2). Only the fp32 order in which the chunk results are summed differs from hipBLASLt
(its split-K GSU16 + PostGSU16): ck vs hipBLASLt rel_l2 0 to 3.2e-7 at M <= 262205, at most
9.2e-6 at M = 4.19M. Deterministic (fixed reduction order, no atomics): bitwise-reproducible
run to run.

Launch status: the C entries drain a stale HIP error that an earlier call left on the thread
(the last error is sticky) and return it negated; that is logged once and is not a failure.

Memory: two persistent fp32 workspaces per device, [16, 512, 2048] (W1) and [32, 512, 1024]
(W2), 64 MiB each (+128 MiB per device), allocated from the main caching-allocator pool at
the first fp8t step, plus the [512, N] fp32 output of each call.

Step-tiled xT, DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED=1 (default off; Arbor r14 iteration 39;
read once at import; effective only while `tiled_active`: wgrad_ck requested, not latched, the
library loaded (prebuild, or an earlier wgrad_ck call) and a gfx950 device). The LN producers
of the uvqk / vk wgrad operand (triton_layer_norm store_t v2 and recompute_fp8t, called with
tiled_xt=tiled_active(device)) then write xT as a 3-D e4m3 tensor [KT, 512, 128],
KT = Mp / 128, element (token t, feature c) at ((t // 128) * 512 + c) * 128 + t % 128, zero
tail tokens [M, Mp): each 128-token K step of all 512 rows is one contiguous 64 KiB block. The
layout tag is the tensor rank: `run` launches the XT=1 entries wgrad_ck_w1_t / wgrad_ck_w2_t
(same tiles, splits and fp32 order as the row-major entries: bitwise-identical dW) for a 3-D
xT. Every other reader (the first-call guard's hipBLASLt reference, the hipBLASLt fallback of
a decline or a latch) gets `untile_to_row_major(xT)`, the row-major [512, Mp] bytes the knob-off
producers write, so the fallback is bitwise the knob-off hipBLASLt result. No steady-state
memory: the tiled tensor replaces the row-major one byte for byte; only a fallback (and the
first-call guard) allocates the untiled copy. Kernels in a trace: wgrad_ck_kernel<2048, 16, 1>
(W1), wgrad_ck_kernel<1024, 32, 1> (W2), the same reduces; the row-major kernels are
wgrad_ck_kernel<N, NSPLIT, 0> (instruction-identical to iteration 38's wgrad_ck_kernel<N, NSPLIT>).

Robustness (Arbor r14 iteration 40). A decline (`_decline_reason`) of a step-tiled xT turns the
tiled producers off for the rest of the process (`tiled_active` False from then on, logged once):
a steady decline would otherwise re-lay out every call while the producers keep tiling. The
untiled copy for a hipBLASLt reader comes from the caller's allocator (`untile_to_row_major`
`empty`; bwd_fp8 passes fp8_tm_alloc.empty_copy, the B11 pool when active). When that copy is
out of memory, the guard's `reference()` raises `GuardDeferred`: `run` then neither arms nor
latches (no verdict, logged once per site; the next call runs the guard again) and returns None,
and the caller falls back for that call only.
"""

import ctypes
import hashlib
import logging
import os
import socket
import subprocess
import threading
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch

logger: logging.Logger = logging.getLogger(__name__)

_REQUESTED: bool = os.environ.get("DLRM_HSTU_FP8_BWD_WGRAD_CK", "0") == "1"
# Step-tiled xT producers + XT=1 entries (iteration 39; see "Step-tiled xT" above).
_TILED_REQUESTED: bool = (
    os.environ.get("DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED", "0") == "1"
)

_SOURCE: Path = Path(__file__).with_name("wgrad_tm_ck.cu")
_ARCH: str = "gfx950"
# The kernel-lab build (r14 iter37 wgrad, 'pw1'): 190 VGPR, 0 spill, 128 KiB LDS.
_FLAGS: Tuple[str, ...] = (
    "-std=c++20",
    "-O3",
    "-mllvm",
    "-amdgpu-sched-strategy=iterative-ilp",
    "-ffp-contract=off",
    "-shared",
    "-fPIC",
)
_CACHE_ROOT: str = "/tmp/dlrm_hstu_fp8_wgrad_ck"
_LIB_NAME: str = "libwgrad_ck.so"
# A build takes ~1 s (8 concurrent ranks: 3.4 s); a hung hipcc falls back after this.
_BUILD_TIMEOUT_S: int = 300
# Relative-L2 tolerance of the first-call guard against hipBLASLt (same as the
# hipBLASLt fp8t guard): only the fp32 order of the MFMA chunk sums may differ
# (ck vs hipBLASLt 2e-8 .. 3.2e-7 at M <= 262205, 9.2e-6 at M = 4.19M).
_GUARD_TOL: float = 1.0e-4
_KX: int = 512  # output rows (activation features)
_TOKEN_PAD: int = 128  # tokens per K step: xT is read in whole 128-token steps
_LDM_MAX: int = 0xFFFFFFFF // 256  # 32-bit buffer offsets (wgrad_tm_ck.cu WGCK_LDM_MAX)
# N -> (variant, NSPLIT): W1 uvqk wgrad, W2 vk wgrad.
_ENTRIES: Dict[int, Tuple[str, int]] = {2048: ("w1", 16), 1024: ("w2", 32)}
# The step-tiled xT entries (XT=1) of the same variants: N -> (variant, NSPLIT).
_ENTRIES_T: Dict[int, Tuple[str, int]] = {2048: ("w1_t", 16), 1024: ("w2_t", 32)}
# Step-tiled xT [KT, 512, 128]: the XT=1 entries take its step stride as ldm.
_TILED_STEP: int = _KX * _TOKEN_PAD

_lock = threading.Lock()
_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[Path] = None
_funcs: Dict[int, Callable[..., int]] = {}
_funcs_t: Dict[int, Callable[..., int]] = {}
_broken: Optional[str] = None
_override: Optional[bool] = None
_tiled_override: Optional[bool] = None
# (device index, N) that passed the guard; (device index, N, "t") for a step-tiled xT
_armed: set = set()
_workspaces: Dict[Tuple[int, int, int], torch.Tensor] = {}  # (device, N, stream)
_once: set = set()
_toolchains: Dict[str, str] = {}  # hipcc path -> `hipcc --version`
_device_ok: Dict[int, bool] = {}  # device index -> gfx950 (tiled_active)
# Set by `run` on the first decline of a step-tiled xT: the producers stop tiling (S3).
_tiled_declined: Optional[str] = None


class GuardDeferred(Exception):
    """Raised by a first-call guard's `reference()` when the reference cannot be
    computed for this call only (its untiled operand copy is out of memory): `run` /
    wgrad_ck_w3.run_w3 then give no verdict (nothing armed, nothing latched; the next
    call runs the guard again)."""


def _print(message: str) -> None:
    # Same style as bwd_fp8._uvqk_print: rank-local, flushed.
    print(f"[hstu-fp8-bwd-uvqk] {message}", flush=True)


def _print_once(key: str, message: str) -> None:
    if key in _once:
        return
    _once.add(key)
    _print(message)


if _REQUESTED:
    _print("requested: wgrad_ck=1")
if _TILED_REQUESTED:
    _print(
        "requested: wgrad_ck_xt_tiled=1"
        + ("" if _REQUESTED else " -> INERT without DLRM_HSTU_FP8_BWD_WGRAD_CK=1")
    )


def requested() -> bool:
    """The knob (or the test-only override) asks for the custom kernel."""
    return _REQUESTED if _override is None else _override


def tiled_requested() -> bool:
    """DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED (or the test-only override)."""
    return _TILED_REQUESTED if _tiled_override is None else _tiled_override


def set_tiled_requested_for_testing(value: Optional[bool]) -> None:
    """Test-only: override the import-time XT_TILED knob (None restores it)."""
    global _tiled_override
    _tiled_override = value


def tiled_active(device: torch.device) -> bool:
    """Whether the LN producers write the step-tiled xT on `device` now.

    XT_TILED requested, wgrad_ck requested and not latched, its library loaded
    (prebuild, or an earlier wgrad_ck call: a producer never builds) and a
    gfx950 device. Cheap: no tensor access; False at once with the knob off.
    A tiled xT stays consumable after a later latch (`untile_to_row_major`).
    False after `run` declined a tiled xT (iteration 40: no steady re-layout).
    """
    if not tiled_requested() or not enabled() or _lib is None or _tiled_declined is not None:
        return False
    if device.type != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    ok = _device_ok.get(index)
    if ok is None:
        try:
            ok = device_arch(torch.device("cuda", index)) == _ARCH
        except Exception:  # pragma: no cover - defensive
            ok = False
        _device_ok[index] = ok
    return ok


def tiled_shape(d: int, mp: int) -> Optional[Tuple[int, int, int]]:
    """The step-tiled xT shape [Mp / 128, 512, 128] for a [d, Mp] operand, or
    None when the layout does not apply (d != 512 or Mp % 128 != 0)."""
    if d != _KX or mp % _TOKEN_PAD != 0:
        return None
    return (mp // _TOKEN_PAD, _KX, _TOKEN_PAD)


def note_tiled_producer(producer: str, device: torch.device, used: bool, detail: str) -> None:
    """Once per (producer, device, used): a producer wrote a step-tiled xT, or
    was asked to and declined (then it wrote the row-major layout)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    if used:
        _print_once(
            f"xtt_producer:{producer}:{index}",
            f"ARMED wgrad_ck xt_tiled producer {producer} on cuda:{index}: step-tiled "
            f"xT [KT, 512, 128] e4m3 (zero tail), {detail}",
        )
    else:
        _print_once(
            f"xtt_decline:{producer}:{index}",
            f"wgrad_ck xt_tiled declined by {producer} on cuda:{index} ({detail}) "
            "-> row-major xT for such calls",
        )


def untile_to_row_major(
    x_t: torch.Tensor,
    mp: Optional[int] = None,
    empty: Callable[..., torch.Tensor] = torch.empty,
) -> torch.Tensor:
    """Row-major e4m3 [512, Mp] copy of a step-tiled xT [KT, 512, 128]: exactly
    the bytes the knob-off producers write (contiguous, ld = Mp = KT * 128 unless
    `mp` asks for fewer columns). For every non-wgrad_ck reader of a tiled xT;
    allocates (only reached on the guard's reference and on fallbacks) with
    `empty(shape, dtype=, device=)` (bwd_fp8: fp8_tm_alloc.empty_copy)."""
    if x_t.dim() != 3:
        raise ValueError(f"untile_to_row_major: expected a 3-D tiled xT, got {tuple(x_t.shape)}")
    kt, kx, step = (int(s) for s in x_t.shape)
    full = kt * step
    mp = full if mp is None else int(mp)
    if not 0 <= mp <= full:
        raise ValueError(f"untile_to_row_major: Mp={mp} outside [0, {full}]")
    out = empty((kx, kt, step), dtype=x_t.dtype, device=x_t.device)
    out.copy_(x_t.permute(1, 0, 2))
    out = out.view(kx, full)
    return out if mp == full else out[:, :mp].contiguous()


def enabled() -> bool:
    """Requested and not latched off: `_run_tm_wgrad` tries the custom kernel first."""
    return requested() and _broken is None


def broken() -> Optional[str]:
    return _broken


def set_requested_for_testing(value: Optional[bool]) -> None:
    """Test-only: override the import-time knob (None restores it)."""
    global _override
    _override = value


def reset_for_testing(unload: bool = False) -> None:
    """Test-only: clear the latch, the per-(device, N) arming and the once-lines.

    unload=True also forgets the loaded library (the next call builds/loads again,
    e.g. after changing DLRM_HSTU_FP8_BWD_WGRAD_CK_HIPCC / _CACHE).
    """
    global _broken, _lib, _lib_path, _tiled_declined
    _broken = None
    _tiled_declined = None
    _armed.clear()
    _once.clear()
    if unload:
        with _lock:
            _lib = None
            _lib_path = None
            _funcs.clear()
            _funcs_t.clear()


def _disable(reason: str) -> None:
    global _broken
    if _broken is not None:
        return
    _broken = reason
    logger.warning("[hstu-fp8-bwd-uvqk] wgrad_ck disabled: %s", reason)
    _print(f"wgrad_ck DISABLED: {reason}")


def _hipcc() -> str:
    override = os.environ.get("DLRM_HSTU_FP8_BWD_WGRAD_CK_HIPCC", "").strip()
    if override:
        return override
    rocm = os.environ.get("ROCM_PATH", "").strip()
    if rocm:
        candidate = os.path.join(rocm, "bin", "hipcc")
        if os.path.exists(candidate):
            return candidate
    return "/opt/rocm/bin/hipcc"


def _toolchain(hipcc: str) -> str:
    """`hipcc --version` (cached per hipcc path; raises if hipcc cannot run)."""
    version = _toolchains.get(hipcc)
    if version is None:
        proc = subprocess.run(
            [hipcc, "--version"], capture_output=True, text=True, timeout=120
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{hipcc} --version rc={proc.returncode}: "
                f"{((proc.stderr or '') + (proc.stdout or '')).strip()[-400:]}"
            )
        version = proc.stdout
        _toolchains[hipcc] = version
    return version


def _cache_dir(source: Path = _SOURCE) -> Path:
    root = os.environ.get("DLRM_HSTU_FP8_BWD_WGRAD_CK_CACHE", "").strip() or _CACHE_ROOT
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(b"\0" + "\0".join(_FLAGS).encode())
    digest.update(b"\0" + _ARCH.encode())
    # The toolchain too: a .so another hipcc built (a surviving or shared cache)
    # is not reused, so the build's resource usage (190 VGPR, 0 spill) holds.
    digest.update(b"\0" + _toolchain(_hipcc()).encode())
    return Path(root) / digest.hexdigest()[:16]


def device_arch(device: torch.device) -> str:
    return str(torch.cuda.get_device_properties(device).gcnArchName).split(":")[0]


def _check_arch(device: torch.device) -> None:
    arch = device_arch(device)
    if arch != _ARCH:
        raise RuntimeError(f"device arch {arch} is not {_ARCH} (wgrad_tm_ck.cu is {_ARCH}-only)")


def build(source: Path = _SOURCE, lib_name: str = _LIB_NAME) -> Path:
    """Compile wgrad_tm_ck.cu (cache hit: no compile) and return the library path.

    `source` / `lib_name`: another kernel source of the same family and flags
    (wgrad_ck_w3.py builds wgrad_tm_ck_w3.cu this way; own cache directory).
    Raises on any failure (missing hipcc, compile error, timeout).
    """
    so = _cache_dir(source) / lib_name
    if so.is_file():
        return so
    so.parent.mkdir(parents=True, exist_ok=True)
    # Up to 8 ranks (per node, if the cache root is shared across nodes) may race
    # here: each compiles to its own file, then renames it into place atomically
    # (identical bytes, last rename wins).
    tmp = so.parent / (
        f"tmp-{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}-{lib_name}"
    )
    cmd = [_hipcc(), str(source), f"--offload-arch={_ARCH}", *_FLAGS, "-o", str(tmp)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_BUILD_TIMEOUT_S
        )
        if proc.returncode != 0 or not tmp.is_file():
            log = (proc.stderr or "") + (proc.stdout or "")
            raise RuntimeError(
                f"hipcc rc={proc.returncode}: {' '.join(cmd)}: {log.strip()[-1200:]}"
            )
        os.replace(tmp, so)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return so


def bind_entry(lib: ctypes.CDLL, so: Path, name: str, nsplit: int) -> Callable[..., int]:
    """ctypes binding of one entry of the wgrad_ck launcher ABI (wgrad_tm_ck.cu,
    wgrad_tm_ck_w3.cu): int name(G, X, ldm, part, out, gs, as, M, stream), and a
    check that `name`_nsplit() reports the workspace split count `nsplit`."""
    fn = getattr(lib, name)
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_void_p,  # G (dz8 / y8)
        ctypes.c_void_p,  # X (xT / dout_t)
        ctypes.c_longlong,  # ldm
        ctypes.c_void_p,  # part (fp32 workspace)
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # gs
        ctypes.c_void_p,  # as
        ctypes.c_int,  # M
        ctypes.c_void_p,  # hipStream_t
    ]
    getter = getattr(lib, f"{name}_nsplit")
    getter.restype = ctypes.c_int
    getter.argtypes = []
    got = int(getter())
    if got != nsplit:
        raise RuntimeError(f"{so}: {name}_nsplit()={got}, expected {nsplit}")
    return fn


def guard_metrics(out: torch.Tensor, ref: torch.Tensor) -> Tuple[float, bool]:
    """(rel_l2 of `out` vs `ref` in fp64, all-finite(out)): the first-call guard
    measure (accept iff finite and rel <= _GUARD_TOL). Two host syncs."""
    ref64 = ref.double()
    ref_norm = float(ref64.norm())
    rel = float((out.double() - ref64).norm()) / max(ref_norm, 1.0e-300)
    finite = bool(torch.isfinite(out).all())
    return rel, finite


def _load() -> None:
    global _lib, _lib_path
    if _lib is not None:
        return
    with _lock:
        if _lib is not None:
            return
        so = build()
        lib = ctypes.CDLL(str(so))
        funcs: Dict[int, Callable[..., int]] = {}
        for n, (variant, nsplit) in _ENTRIES.items():
            funcs[n] = bind_entry(lib, so, f"wgrad_ck_{variant}", nsplit)
        funcs_t: Dict[int, Callable[..., int]] = {}
        for n, (variant, nsplit) in _ENTRIES_T.items():
            funcs_t[n] = bind_entry(lib, so, f"wgrad_ck_{variant}", nsplit)
        _funcs.clear()
        _funcs.update(funcs)
        _funcs_t.clear()
        _funcs_t.update(funcs_t)
        _lib_path = so
        _lib = lib


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
        _check_arch(torch.device("cuda", torch.cuda.current_device()))
        _load()
    except Exception as error:
        _disable(f"{type(error).__name__}: {error}")
        return False
    return True


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0


def _decline_reason(
    dz8: torch.Tensor,
    x_t: torch.Tensor,
    grad_scale: torch.Tensor,
    act_scale: torch.Tensor,
    m: int,
) -> Optional[Tuple[str, str]]:
    """(category, detail) when this call is outside the kernel's contract."""
    if m < 1:
        return "M", f"M={m}"
    if dz8.dtype != torch.float8_e5m2 or x_t.dtype != torch.float8_e4m3fn:
        return "dtype", f"dz8 {dz8.dtype} xT {x_t.dtype}"
    if dz8.dim() != 2 or x_t.dim() not in (2, 3):
        return "rank", f"dz8 {tuple(dz8.shape)} xT {tuple(x_t.shape)}"
    n = int(dz8.shape[1])
    if n not in _ENTRIES:
        return "N", f"N={n} (supported: {sorted(_ENTRIES)})"
    if dz8.stride() != (n, 1) or int(dz8.shape[0]) < m:
        return "dz8", f"dz8 shape {tuple(dz8.shape)} strides {dz8.stride()} M={m}"
    mp = (m + _TOKEN_PAD - 1) // _TOKEN_PAD * _TOKEN_PAD
    ldm = int(x_t.stride(0))
    if x_t.dim() == 3:
        # step-tiled [>= ceil(M/128), 512, 128], contiguous (the XT=1 entries)
        if (
            tuple(int(s) for s in x_t.shape[1:]) != (_KX, _TOKEN_PAD)
            or x_t.stride() != (_TILED_STEP, _TOKEN_PAD, 1)
            or int(x_t.shape[0]) * _TOKEN_PAD < mp
        ):
            return "xT", f"tiled xT shape {tuple(x_t.shape)} strides {x_t.stride()} M={m} Mp={mp}"
    elif (
        int(x_t.shape[0]) != _KX
        or x_t.stride(1) != 1
        or ldm < mp
        or int(x_t.shape[1]) < mp
        or ldm % 16 != 0
        or ldm > _LDM_MAX
    ):
        return "xT", f"xT shape {tuple(x_t.shape)} strides {x_t.stride()} M={m} Mp={mp}"
    if not (_aligned(dz8) and _aligned(x_t)):
        return "align", f"dz8 ptr {dz8.data_ptr():#x} xT ptr {x_t.data_ptr():#x}"
    device = dz8.device
    for name, t in (("xT", x_t), ("gs", grad_scale), ("as", act_scale)):
        if t.device != device:
            return "device", f"{name} on {t.device}, dz8 on {device}"
    if device.type != "cuda":
        return "device", f"dz8 on {device}"
    for name, t in (("gs", grad_scale), ("as", act_scale)):
        if t.dtype != torch.float32 or t.numel() != 1:
            return "scale", f"{name} {t.dtype} numel={t.numel()}"
    return None


def _launch(
    dz8: torch.Tensor,
    x_t: torch.Tensor,
    grad_scale: torch.Tensor,
    act_scale: torch.Tensor,
    m: int,
    index: int,
) -> torch.Tensor:
    n = int(dz8.shape[1])
    # The layout tag is the rank: a 3-D xT is step-tiled (XT=1 entries).
    variant, nsplit = (_ENTRIES_T if x_t.dim() == 3 else _ENTRIES)[n]
    fn = (_funcs_t if x_t.dim() == 3 else _funcs)[n]
    with torch.cuda.device(index):
        stream = torch.cuda.current_stream(dz8.device).cuda_stream
        # One partial workspace per (device, N, stream): reuse is stream-ordered
        # (training runs one backward stream per device, so this is one workspace
        # per (device, N); a second stream gets its own, never a shared one).
        key = (index, n, int(stream))
        workspace = _workspaces.get(key)
        if workspace is None:
            workspace = torch.empty(
                (nsplit, _KX, n), dtype=torch.float32, device=dz8.device
            )
            _workspaces[key] = workspace
        out = torch.empty((_KX, n), dtype=torch.float32, device=dz8.device)
        rc = fn(
            dz8.data_ptr(),
            x_t.data_ptr(),
            int(x_t.stride(0)),
            workspace.data_ptr(),
            out.data_ptr(),
            grad_scale.data_ptr(),
            act_scale.data_ptr(),
            int(m),
            stream,
        )
    if rc < 0:
        # Both kernels launched; the entry drained a stale error that an earlier
        # call had left on this thread (sticky last error): not a launch failure.
        _print_once(
            f"stale:{-rc}",
            f"wgrad_ck: stale HIP error {-rc} drained before a wgrad_ck_{variant} "
            "launch (left on this thread by an earlier call, not ours)",
        )
    elif rc != 0:
        raise RuntimeError(f"wgrad_ck_{variant} launch returned hip error {rc}")
    return out


def run(
    dz8: torch.Tensor,
    x_t: torch.Tensor,
    grad_scale: torch.Tensor,
    act_scale: torch.Tensor,
    m: int,
    site: str,
    reference: Callable[[], torch.Tensor],
) -> Optional[Tuple[torch.Tensor, str]]:
    """(dW fp32 [512, N], 'wgrad_ck_w1' | 'wgrad_ck_w2'), or None: the caller runs hipBLASLt.

    None when not enabled (knob off or latched), when this call is outside the
    contract (logged once per site and reason), when the first-call guard rejects
    (latched) or on any exception (latched). `reference()` returns the hipBLASLt
    dW of this call (or raises); it is called only by the first-call guard.
    A 3-D (step-tiled) xT runs the XT=1 entries: label 'wgrad_ck_w1_t' |
    'wgrad_ck_w2_t'; the caller must untile it for hipBLASLt (reference and None).
    A declined 3-D xT also turns the tiled producers off (`tiled_active`). A
    `GuardDeferred` from `reference()` returns None without arming or latching.
    """
    global _tiled_declined
    if not enabled():
        return None
    m = int(m)
    why = _decline_reason(dz8, x_t, grad_scale, act_scale, m)
    if why is not None:
        _print_once(
            f"decline:{site}:{why[0]}",
            f"wgrad_ck declined {site} ({why[0]}: {why[1]}) -> hipBLASLt for such calls",
        )
        if x_t.dim() == 3 and _tiled_declined is None:
            _tiled_declined = f"{site} ({why[0]}: {why[1]})"
            _print(
                f"wgrad_ck xt_tiled OFF: a step-tiled xT was declined at {_tiled_declined} "
                "-> the LN producers write the row-major xT from now on"
            )
        return None
    n = int(dz8.shape[1])
    tiled = x_t.dim() == 3
    variant = (_ENTRIES_T if tiled else _ENTRIES)[n][0]
    device = dz8.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        # A step-tiled xT runs other kernels (XT=1): guarded on its own first call.
        arm_key = (index, n, "t") if tiled else (index, n)
        first = arm_key not in _armed
        if first:
            _check_arch(device)
        _load()
        out = _launch(dz8, x_t, grad_scale, act_scale, m, index)
        if first:
            rel, finite = guard_metrics(out, reference())
            line = (
                f"wgrad_ck guard {site} on cuda:{index}: N={n} tokens={m} "
                f"{'xT=tiled ' if tiled else ''}rel={rel:.3e}"
            )
            if not (finite and rel <= _GUARD_TOL):
                _print(line + (" -> rejected" if finite else " finite=0 -> rejected"))
                _disable(
                    f"guard rejected {site} on cuda:{index}: N={n} tokens={m} "
                    f"{'xT=tiled ' if tiled else ''}rel={rel:.3e} finite={int(finite)}"
                )
                return None
            _print(line + " -> armed")
            _armed.add(arm_key)
            _print(f"ARMED wgrad_ck {variant} on cuda:{index}")
            if tiled:
                _print_once(
                    f"xtt_armed:{index}",
                    f"ARMED wgrad_ck xt_tiled on cuda:{index}: step-tiled xT "
                    "[KT, 512, 128] -> wgrad_ck_kernel<N, NSPLIT, 1> (bitwise the "
                    "row-major entries)",
                )
        return out, f"wgrad_ck_{variant}"
    except GuardDeferred as error:
        # The guard's reference could not be computed for this call only (its
        # untiled copy is out of memory): no verdict, nothing armed or latched.
        _print_once(
            f"guard_deferred:{site}",
            f"wgrad_ck guard {site} on cuda:{index}: deferred ({error}) -> no verdict, "
            "nothing latched; this call falls back, the next call runs the guard again",
        )
        return None
    except Exception as error:
        _disable(f"{type(error).__name__}: {error}")
        return None
