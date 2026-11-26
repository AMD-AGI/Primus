# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

# pyre-unsafe

"""K=512 FP8 forward projection GEMM (HSTU V/Q/K and targets V/K) as a hipcc-built HIP library.

Behind ``DLRM_HSTU_FP8_FWD_K512=1``, ``triton_fp8.scaled_mm_fp8_out_preq`` calls this
instead of the hipBLASLt GEMM it runs in the ``input_scale`` backend mode::

    torch._scaled_mm(A, Bw.t(), scale_a=sa, scale_b=sb, bias=bias_bf16,
                     out_dtype=torch.float8_e4m3fn, use_fast_accum=False)

    C[M, N] e4m3 = e4m3_rne(sat448(acc * (sa * sb) + float(bias[n])))
    acc = sum_k A[m, k] * Bw[n, k] in fp32; K = 512; N in {1024, 1536}

The replacement must be bit-identical to that call. ``ops/tests/k512_fp8_gemm_test.py``
compares every output byte through ``scaled_mm_fp8_out_preq`` itself.
That holds on hipBLASLt's data-parallel tiles and on narrow Gaussian data (the test's).
At in-model M hipBLASLt runs a persistent stream-K grid whose tail tiles add split-K
partial sums in fp32 through its workspace, so an order-sensitive row there can round
differently: an fp32 summation-order difference (yambda_5b.gin item (43), NUMERICS).

C ABI of the library (``k512_fp8_gemm.cu``; one source compiled per N with ``-DHK_N=N``)::

    int k512_fp8_gemm_e4m3_bias_N{N}(const void* A,     /* e4m3 [M, 512] row-major */
                                     const void* B,     /* e4m3 [N, 512] row-major */
                                     void* C,           /* e4m3 [M, N] row-major, rows < M */
                                     const void* bias,  /* bf16 [N] */
                                     const float* sa,   /* fp32 device scalar */
                                     const float* sb,   /* fp32 device scalar */
                                     int M,             /* >= 1, any value */
                                     void* stream);     /* hipStream_t; returns hipGetLastError() */
    int k512_fp8_gemm_e4m3_bias_tiled_a_N{N}(...);      /* same signature, A = step-tiled yT (below) */
    const char* k512_fp8_gemm_impl(void);               /* optional implementation tag */

The in-tree ``k512_fp8_gemm.cu`` is a self-contained HIP kernel (raw intrinsics, no
library headers):
- two resident 4-wave workgroups per CU, so one WG's epilogue overlaps the other's MFMA;
- a 128x256 tile and a 5-slot 80 KiB LDS ring fed by LDS-DMA loads;
- operand-swapped ``v_mfma_f32_16x16x128_f8f6f4`` with the same per-element K order as hipBLASLt;
- a packed mul-then-add epilogue whose fp8 converts saturate under MODE.FP16_OVFL, so it needs
  no clamp;
- ``buffer_store_dwordx4`` stores, with M-tail rows dropped by the buffer range;
- XCD-aware tile order.

Build: ``/opt/rocm/bin/hipcc``, once per N per process, into a per-process
directory under ``/tmp/dlrm_hstu_fp8_k512``. The build runs from ``prebuild()``
(``triton_fp8.prebuild_extensions``, before MLPerf RUN_START), or lazily on the
first call. A failed build is latched, so it is not retried on every call.
Optional environment:

* ``DLRM_HSTU_FP8_FWD_K512_SRC``: another ``.cu`` with the same ABI, to A/B a
  candidate kernel without editing the tree.

Every launch is asynchronous on ``torch.cuda.current_stream()``. The scales are
read on the device, so there is no host sync.

Tiled-A entry (Arbor P7 iteration 20, item B15T(L2); not wired into the model here).
``k512_scaled_mm_tiled_a(a_tiled, M, weight_nk, bias, scale_a, scale_b)`` takes A
from the step-tiled token-major yT that the LayerNorm store_t writes for the K=tokens
wgrad (ops/triton/wgrad_ck.py "Step-tiled xT"): e4m3 [ceil(M/128), 512, 128], token t
/ feature c at ((t // 128) * 512 + c) * 128 + t % 128, instead of the row-major y
[M, 512]. Its output is bit-identical to the row-major entry on y (same kernel, same
MFMA operands; the A fragments come from ds_read_b64_tr_b8 transposing LDS reads), so
a layer whose only y reader is this GEMM can skip the y write. The tail tokens [M, Mp)
of the last block do not affect the output. Weight, bias, scales and output are as in
the row-major entry. The entry is in the same per-N library (``prebuild()`` builds it).
``prebuild_tiled_a()`` is its prebuild hook: before RUN_START it runs a startup
self-check against the row-major entry on synthetic data (a private generator; no
data, model state or global RNG is touched). A mismatch latches the entry off for the
process and logs once. ``tiled_a_ready(N)`` reports a passed self-check; a caller
must keep the row-major y unless it is True.
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import hashlib
import os
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch

K: int = 512
SUPPORTED_N: Tuple[int, ...] = (1024, 1536)
HIPCC: str = "/opt/rocm/bin/hipcc"
KNOB: str = "DLRM_HSTU_FP8_FWD_K512"
_SRC_ENV: str = "DLRM_HSTU_FP8_FWD_K512_SRC"
_BUILD_ROOT: Path = Path("/tmp/dlrm_hstu_fp8_k512")
_BUILD_TIMEOUT_S: int = 600
_INT32_MAX: int = 2**31 - 1

_FP8_E4M3: Optional[torch.dtype] = getattr(torch, "float8_e4m3fn", None)


class PrebuildError(RuntimeError):
    """prebuild(): a library failed to build, load or self-check (latched; the
    forward then declines that N). A device fault is not wrapped in this."""


@dataclass(frozen=True)
class Kernel:
    """One loaded library (one N)."""

    n: int
    path: str
    fn: object  # ctypes function pointer
    impl: str
    source: str
    source_sha256: str
    build_s: float
    resources: str


_lock = threading.Lock()  # guards _build_dir
_n_locks: Dict[int, threading.Lock] = {n: threading.Lock() for n in SUPPORTED_N}
_kernels: Dict[int, Kernel] = {}
_failures: Dict[int, str] = {}
_build_dir: Optional[Path] = None
_armed_logged: set = set()
_decline_logged: set = set()


def _print(message: str) -> None:
    print(f"[k512-fwd] {message}", flush=True)


def source_path() -> Path:
    override = os.environ.get(_SRC_ENV)
    return Path(override) if override else Path(__file__).with_name("k512_fp8_gemm.cu")


def entry_symbol(n: int) -> str:
    return f"k512_fp8_gemm_e4m3_bias_N{n}"


def hipcc_command(
    n: int,
    source: Path,
    output: Path,
    *,
    extra_flags: Sequence[str] = (),
) -> list:
    """The exact hipcc invocation: the flags the kernel was gated with
    (iterative-ilp scheduler, -O3) plus -ffp-contract=off."""

    return [
        HIPCC,
        str(source),
        "--offload-arch=gfx950",
        "-std=c++20",
        "-w",
        "-O3",
        # Bit parity with hipBLASLt needs acc*s and +bias as two roundings; the
        # source also says so with '#pragma clang fp contract(off)'.
        "-ffp-contract=off",
        "-I/opt/rocm/include/hip",
        f"-DHK_N={n}",
        f"-DHK_K={K}",
        "-mllvm",
        "-amdgpu-sched-strategy=iterative-ilp",
        "-shared",
        "-fPIC",
        "-Rpass-analysis=kernel-resource-usage",
        *extra_flags,
        "-o",
        str(output),
    ]


def _process_build_dir() -> Path:
    global _build_dir
    with _lock:
        if _build_dir is None:
            _BUILD_ROOT.mkdir(parents=True, exist_ok=True)
            _build_dir = Path(
                tempfile.mkdtemp(prefix=f"pid{os.getpid()}_", dir=str(_BUILD_ROOT))
            )
        return _build_dir


def resource_summary(build_log: str) -> str:
    """Condense the hipcc -Rpass-analysis=kernel-resource-usage remarks to one
    entry per kernel: VGPRs, AGPRs, spills, scratch, occupancy, LDS."""

    per_kernel: Dict[str, Dict[str, str]] = {}
    current = None
    # "<loc>: remark: Function Name: X" or, with --save-temps, "remark: <loc>: ..."
    location = r"(?:\s+\S+:\d+:\d+:)?"
    # ROCm 10.2 (AMD clang 24): "<loc>: Function Name: X", "<loc>:     <Key>: <v>"
    bare = r"\S+:\d+:\d+:\s+"
    bare_key = (
        r"(TotalSGPRs|VGPRs|AGPRs|ScratchSize \[bytes/lane\]|Dynamic Stack|"
        r"Occupancy \[waves/SIMD\]|SGPRs Spill|VGPRs Spill|LDS Size \[bytes/block\])"
    )
    for line in build_log.splitlines():
        m = re.search(r"remark:" + location + r"\s+Function Name:\s+(\S+)", line)
        m = m or re.fullmatch(bare + r"Function Name:\s+(\S+)\s*", line)
        if m:
            current = m.group(1)
            per_kernel[current] = {}
            continue
        m = re.search(
            r"remark:" + location + r"\s+([A-Za-z][A-Za-z /\[\]]*?):\s+(\S+)\s+\[-Rpass",
            line,
        )
        m = m or re.fullmatch(bare + bare_key + r":\s+(\S+)\s*", line)
        if m and current is not None:
            per_kernel[current][m.group(1).strip()] = m.group(2)
    keep = (
        ("VGPRs", "vgpr"),
        ("AGPRs", "agpr"),
        ("VGPRs Spill", "vspill"),
        ("SGPRs Spill", "sspill"),
        ("ScratchSize [bytes/lane]", "scratch"),
        ("Occupancy [waves/SIMD]", "occ"),
        ("LDS Size [bytes/block]", "lds"),
    )
    parts = []
    for name, fields in per_kernel.items():
        m = re.match(r"_Z(\d+)(\w+)", name)
        short = m.group(2)[: int(m.group(1))] if m else name[:40]
        kv = " ".join(f"{label}={fields[key]}" for key, label in keep if key in fields)
        parts.append(f"{short}: {kv}")
    return "; ".join(parts)


def _build(n: int, source: Path) -> Tuple[Path, float, str]:
    out_dir = _process_build_dir()
    output = out_dir / f"libk512_fp8_gemm_N{n}.so"
    temporary = out_dir / f".libk512_fp8_gemm_N{n}.{os.getpid()}.so"
    log_path = out_dir / f"build_N{n}.log"
    command = hipcc_command(n, source, temporary)
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=_BUILD_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"hipcc not found ({HIPCC})") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"hipcc timed out after {_BUILD_TIMEOUT_S}s") from error
    seconds = time.perf_counter() - t0
    log_path.write_text(" ".join(command) + "\n" + result.stdout)
    if result.returncode != 0 or not temporary.exists():
        lines = result.stdout.strip().splitlines()
        errors = [line.strip() for line in lines if "error" in line.lower()]
        tail = " | ".join((errors or lines)[:3])
        raise RuntimeError(
            f"hipcc rc={result.returncode} for N={n} (log {log_path}): {tail[:400]}"
        )
    os.replace(temporary, output)
    return output, seconds, resource_summary(result.stdout)


def _load(n: int) -> Kernel:
    source = source_path()
    blob = source.read_bytes()
    path, seconds, resources = _build(n, source)
    library = ctypes.CDLL(str(path))
    fn = getattr(library, entry_symbol(n))
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_void_p]
    impl = "?"
    impl_fn = getattr(library, "k512_fp8_gemm_impl", None)
    if impl_fn is not None:
        impl_fn.restype = ctypes.c_char_p
        impl_fn.argtypes = []
        impl = (impl_fn() or b"?").decode(errors="replace")
    return Kernel(
        n=n,
        path=str(path),
        fn=fn,
        impl=impl,
        source=str(source),
        source_sha256=hashlib.sha256(blob).hexdigest(),
        build_s=seconds,
        resources=resources,
    )


def kernel(n: int) -> Kernel:
    """The loaded library for N (built on first use). Raises RuntimeError if it
    cannot be built or loaded; the failure is latched for the process."""

    cached = _kernels.get(n)
    if cached is not None:
        return cached
    if n not in SUPPORTED_N:
        raise RuntimeError(f"N={n} is not built (supported: {SUPPORTED_N})")
    with _n_locks[n]:  # per N, so prebuild() compiles both N in parallel
        cached = _kernels.get(n)
        if cached is not None:
            return cached
        if n in _failures:
            raise RuntimeError(_failures[n])
        try:
            loaded = _load(n)
        except Exception as error:
            _failures[n] = f"K=512 FP8 GEMM library for N={n} unavailable: {error}"
            raise RuntimeError(_failures[n]) from error
        _kernels[n] = loaded
        return loaded


def failure(n: int) -> Optional[str]:
    return _failures.get(n)


def ineligible_reason(
    activation: torch.Tensor,
    weight_nk: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> Optional[Tuple[str, str]]:
    """(kind, detail) if the kernel cannot take these operands, else None.

    kind is from a small fixed set (it keys the one-time decline log line)."""

    if _FP8_E4M3 is None:
        return ("dtype", "torch has no float8_e4m3fn")
    if (
        activation.dtype != _FP8_E4M3
        or activation.dim() != 2
        or activation.shape[1] != K
        or not activation.is_contiguous()
    ):
        return (
            "activation",
            f"activation {activation.dtype} {tuple(activation.shape)} stride "
            f"{activation.stride()} (needs contiguous e4m3 [M, {K}])",
        )
    n = weight_nk.shape[0] if weight_nk.dim() == 2 else -1
    if n not in SUPPORTED_N:
        return ("shape", f"N={n} (built for N in {SUPPORTED_N})")
    if (
        weight_nk.dtype != _FP8_E4M3
        or tuple(weight_nk.shape) != (n, K)
        or weight_nk.stride() != (K, 1)
    ):
        return (
            "weight",
            f"weight {weight_nk.dtype} {tuple(weight_nk.shape)} stride "
            f"{weight_nk.stride()} (needs e4m3 [N, {K}] row-major)",
        )
    if bias is None:
        return ("bias", "no bias (the kernel always adds one)")
    if bias.dtype != torch.bfloat16 or tuple(bias.shape) != (n,) or bias.stride() != (1,):
        return (
            "bias",
            f"bias {bias.dtype} {tuple(bias.shape)} stride {bias.stride()} "
            f"(needs contiguous bf16 [{n}])",
        )
    for name, scale in (("scale_a", scale_a), ("scale_b", scale_b)):
        if scale.dtype != torch.float32 or scale.numel() != 1:
            return (
                "scale",
                f"{name} {scale.dtype} {tuple(scale.shape)} (needs one fp32 value)",
            )
    device = activation.device
    if device.type != "cuda" or any(
        t.device != device for t in (weight_nk, bias, scale_a, scale_b)
    ):
        return ("device", "operands are not all on one CUDA device")
    if device.index is not None and device.index != torch.cuda.current_device():
        return (
            "device",
            f"operands on {device}, current device cuda:{torch.cuda.current_device()}",
        )
    if activation.shape[0] > _INT32_MAX:
        return ("rows", f"M={activation.shape[0]} exceeds int32")
    if activation.data_ptr() % 16 or weight_nk.data_ptr() % 16:
        return ("align", "activation/weight not 16-byte aligned")
    return None


def _empty_output(m: int, n: int, device: torch.device) -> torch.Tensor:
    # A separate function so the unit test can hand out a NaN-prefilled buffer.
    return torch.empty((m, n), dtype=_FP8_E4M3, device=device)


def launch(
    kern: Kernel,
    activation: torch.Tensor,
    weight_nk: torch.Tensor,
    bias: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the kernel on operands that passed ineligible_reason(); returns `out`
    or a new e4m3 [M, N] tensor (stream-ordered on the current stream)."""

    m = activation.shape[0]
    if out is None:
        out = _empty_output(m, kern.n, activation.device)
    if m == 0:
        return out
    stream = torch.cuda.current_stream(activation.device).cuda_stream
    rc = kern.fn(
        activation.data_ptr(),
        weight_nk.data_ptr(),
        out.data_ptr(),
        bias.data_ptr(),
        scale_a.data_ptr(),
        scale_b.data_ptr(),
        m,
        stream,
    )
    if rc != 0:
        raise RuntimeError(
            f"{entry_symbol(kern.n)} failed: hipError {rc} (M={m}, lib={kern.path})"
        )
    return out


def k512_gemm_e4m3(
    a_fp8: torch.Tensor,
    weight_nk_fp8: torch.Tensor,
    bias_bf16: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """e4m3 [M, N] = e4m3_rne(sat448(a @ weight_nk.T * (scale_a * scale_b) + bias)).

    a_fp8 e4m3 [M, 512] contiguous; weight_nk_fp8 e4m3 [N, 512] contiguous
    (N in 1024/1536); bias_bf16 [N]; scale_a / scale_b one fp32 value each, on the
    same (current) device. Raises ValueError for anything else."""

    reason = ineligible_reason(a_fp8, weight_nk_fp8, bias_bf16, scale_a, scale_b)
    if reason is not None:
        raise ValueError(f"k512_gemm_e4m3: {reason[1]}")
    return launch(
        kernel(weight_nk_fp8.shape[0]), a_fp8, weight_nk_fp8, bias_bf16, scale_a, scale_b
    )


def log_armed_once(kern: Kernel, m: int) -> None:
    """The one arming line per N per process (printed at the first dispatch)."""

    if kern.n in _armed_logged:
        return
    _armed_logged.add(kern.n)
    _print(
        f"armed: N={kern.n} lib={kern.path} ({KNOB}=1) impl={kern.impl!r} "
        f"src={os.path.basename(kern.source)}@{kern.source_sha256[:12]} "
        f"device=cuda:{torch.cuda.current_device()} first_M={m}"
    )


def log_decline_once(key: str, message: str) -> None:
    """One line per decline kind per process: this call stays on hipBLASLt."""

    if key in _decline_logged:
        return
    _decline_logged.add(key)
    _print(f"declined -> hipBLASLt ({KNOB}=1): {message}")


def prebuild(warm_rows: int = 257) -> bool:
    """Build and load both N before RUN_START, then run each once on synthetic
    zeros (warm_rows rows: one full tile plus a tail row, so the code objects
    are resident and both paths ran) into a 0xFF-prefilled (e4m3 NaN) buffer,
    and check that every output byte became +0.

    No data, model state or RNG is touched. Returns True when both N are ready
    and False when skipped (no gfx950). Raises PrebuildError on a failed build
    or self-check (latched, so the dispatch then declines that N). An error from
    the synchronize is a sticky device fault and propagates unwrapped, as in the
    FlyDSL LayerNorm-backward prebuild."""

    if not torch.cuda.is_available():
        _print("prebuild skipped: no GPU")
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    arch = str(getattr(torch.cuda.get_device_properties(device), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        _print(f"prebuild skipped: arch {arch or 'unknown'} (built for gfx950)")
        return False
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SUPPORTED_N)) as pool:
        futures = {n: pool.submit(kernel, n) for n in SUPPORTED_N}
    errors = []
    for n, future in futures.items():
        try:
            future.result()
        except Exception as error:
            errors.append(f"N={n}: {error}")
    if errors:
        _print("prebuild FAILED: " + " | ".join(errors))
        raise PrebuildError("K=512 FP8 GEMM prebuild failed: " + " | ".join(errors))
    t_build = time.perf_counter() - t0
    u8 = torch.uint8
    a = torch.zeros((warm_rows, K), dtype=u8, device=device).view(_FP8_E4M3)
    one = torch.ones((1, 1), dtype=torch.float32, device=device)
    for n in SUPPORTED_N:
        kern = _kernels[n]
        w = torch.zeros((n, K), dtype=u8, device=device).view(_FP8_E4M3)
        bias = torch.zeros((n,), dtype=torch.bfloat16, device=device)
        out = torch.full((warm_rows, n), 0xFF, dtype=u8, device=device).view(_FP8_E4M3)
        launch(kern, a, w, bias, one, one, out=out)
        torch.cuda.synchronize(device)
        nonzero = int(torch.count_nonzero(out.view(u8)).item())
        if nonzero:
            message = (
                f"N={n}: self-check on zeros wrote {nonzero} nonzero bytes "
                f"({kern.path})"
            )
            with _n_locks[n]:
                _kernels.pop(n, None)
                _failures[n] = f"K=512 FP8 GEMM self-check failed: {message}"
            _print("prebuild FAILED: " + message)
            raise PrebuildError("K=512 FP8 GEMM prebuild failed: " + message)
    detail = "; ".join(
        f"N={n} {_kernels[n].build_s:.1f}s [{_kernels[n].resources}]"
        for n in SUPPORTED_N
    )
    _print(
        f"prebuilt: N={','.join(str(n) for n in SUPPORTED_N)} in {t_build:.1f}s "
        f"(dir {_process_build_dir()}, impl={_kernels[SUPPORTED_N[0]].impl!r}); "
        f"self-check M={warm_rows} OK; {detail}"
    )
    return True


# ---------------------------------------------------------------------------------
# Tiled-A entry (Arbor P7 iteration 20, item B15T(L2); see the module docstring and
# the k512_fp8_gemm.cu header). The caller (not wired here) must keep writing the
# row-major y unless tiled_a_ready(N) is True.

TILED_A_STEP: int = 128  # tokens per step-tiled block
# 2 full blocks + a 44-token block: both tile kinds, a 84-token tail
_TILED_A_SELFCHECK_ROWS: int = 300
_TILED_A_GUARD_ROWS: int = 64

_tiled_a_lock = threading.RLock()  # guards the tiled-A state below
_tiled_a_fns: Dict[int, object] = {}  # N -> ctypes function of the N library
_tiled_a_failures: Dict[int, str] = {}  # N -> latched reason (logged once)
_tiled_a_ok: set = set()  # (device index, N) whose self-check passed
_tiled_a_armed_logged: set = set()


def tiled_a_entry_symbol(n: int) -> str:
    return f"k512_fp8_gemm_e4m3_bias_tiled_a_N{n}"


def tiled_a_shape(m: int) -> Tuple[int, int, int]:
    """The step-tiled yT shape for M tokens: [ceil(M / 128), 512, 128]."""

    return ((m + TILED_A_STEP - 1) // TILED_A_STEP, K, TILED_A_STEP)


def tiled_a_failure(n: int) -> Optional[str]:
    return _tiled_a_failures.get(n)


def _latch_tiled_a(n: int, message: str) -> None:
    with _tiled_a_lock:
        first = n not in _tiled_a_failures
        _tiled_a_failures[n] = message
        _tiled_a_fns.pop(n, None)
        for key in [key for key in _tiled_a_ok if key[1] == n]:
            _tiled_a_ok.discard(key)
    if first:
        _print(f"tiled-A OFF for N={n} (latched for the process): {message}")


def _tiled_a_fn(n: int):
    """The ctypes function of the tiled-A entry for N, from the N library (built or
    loaded by kernel(n)). Raises RuntimeError when it is unavailable (latched)."""

    fn = _tiled_a_fns.get(n)
    if fn is not None:
        return fn
    if n in _tiled_a_failures:
        raise RuntimeError(_tiled_a_failures[n])
    if n not in SUPPORTED_N:
        raise RuntimeError(f"N={n} is not built (supported: {SUPPORTED_N})")
    try:
        kern = kernel(n)
    except RuntimeError as error:
        _latch_tiled_a(n, f"N={n} library unavailable: {error}")
        raise RuntimeError(_tiled_a_failures[n]) from error
    with _tiled_a_lock:
        fn = _tiled_a_fns.get(n)
        if fn is None:
            symbol = tiled_a_entry_symbol(n)
            fn = getattr(ctypes.CDLL(kern.path), symbol, None)
            if fn is None:
                _latch_tiled_a(
                    n,
                    f"{kern.path} (src {os.path.basename(kern.source)}"
                    f"@{kern.source_sha256[:12]}) has no {symbol}",
                )
                raise RuntimeError(_tiled_a_failures[n])
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_void_p]
            _tiled_a_fns[n] = fn
    return fn


def ineligible_reason_tiled_a(
    a_tiled: torch.Tensor,
    m: int,
    weight_nk: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> Optional[Tuple[str, str]]:
    """(kind, detail) if the tiled-A entry cannot take these operands, else None.

    a_tiled: contiguous e4m3 [KT, 512, 128] with KT >= ceil(M / 128), 16-byte
    aligned; M an int >= 0. Weight, bias, scales and device as ineligible_reason()."""

    if _FP8_E4M3 is None:
        return ("dtype", "torch has no float8_e4m3fn")
    if (
        a_tiled.dtype != _FP8_E4M3
        or a_tiled.dim() != 3
        or tuple(a_tiled.shape[1:]) != (K, TILED_A_STEP)
        or a_tiled.stride() != (K * TILED_A_STEP, TILED_A_STEP, 1)
    ):
        return (
            "tiled_activation",
            f"tiled activation {a_tiled.dtype} {tuple(a_tiled.shape)} stride "
            f"{a_tiled.stride()} (needs a contiguous e4m3 [KT, {K}, {TILED_A_STEP}] "
            "step-tiled yT)",
        )
    if isinstance(m, bool) or not isinstance(m, int) or m < 0:
        return ("rows", f"M={m!r} (needs an int >= 0)")
    blocks = int(a_tiled.shape[0])
    if blocks * TILED_A_STEP < m:
        return (
            "rows",
            f"M={m} needs KT >= {tiled_a_shape(m)[0]}, the tiled activation has KT={blocks}",
        )
    # Everything else exactly as the row-major entry, on the same bytes seen as
    # [KT * 128, 512] (rows, alignment, weight, bias, scales, device).
    return ineligible_reason(
        a_tiled.view(blocks * TILED_A_STEP, K), weight_nk, bias, scale_a, scale_b
    )


def _tiled_a_call(fn, a_tiled, m, weight_nk, bias, scale_a, scale_b, out) -> int:
    stream = torch.cuda.current_stream(a_tiled.device).cuda_stream
    return fn(
        a_tiled.data_ptr(),
        weight_nk.data_ptr(),
        out.data_ptr(),
        bias.data_ptr(),
        scale_a.data_ptr(),
        scale_b.data_ptr(),
        m,
        stream,
    )


def launch_tiled_a(
    kern: Kernel,
    a_tiled: torch.Tensor,
    m: int,
    weight_nk: torch.Tensor,
    bias: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the tiled-A entry of kern's library on operands that passed
    ineligible_reason_tiled_a(); returns `out` or a new e4m3 [M, N] tensor
    (stream-ordered on the current stream). Rows >= M of `out`'s storage are not
    written. Does not check readiness: callers use tiled_a_ready() /
    k512_scaled_mm_tiled_a()."""

    fn = _tiled_a_fn(kern.n)
    if out is None:
        out = _empty_output(m, kern.n, a_tiled.device)
    if m == 0:
        return out
    rc = _tiled_a_call(fn, a_tiled, m, weight_nk, bias, scale_a, scale_b, out)
    if rc != 0:
        raise RuntimeError(
            f"{tiled_a_entry_symbol(kern.n)} failed: hipError {rc} (M={m}, lib={kern.path})"
        )
    return out


def tiled_a_to_row_major(
    a_tiled: torch.Tensor, m: int, empty=torch.empty
) -> torch.Tensor:
    """The row-major e4m3 [M, 512] y of a step-tiled yT (a contiguous view of a
    [KT * 128, 512] copy allocated with `empty(shape, dtype=, device=)`): the A of
    the row-major entry or of hipBLASLt when a caller that skipped y falls back."""

    if a_tiled.dim() != 3 or tuple(a_tiled.shape[1:]) != (K, TILED_A_STEP):
        raise ValueError(f"tiled_a_to_row_major: not a [KT, {K}, {TILED_A_STEP}] tensor")
    blocks = int(a_tiled.shape[0])
    if not 0 <= m <= blocks * TILED_A_STEP:
        raise ValueError(f"tiled_a_to_row_major: M={m} outside [0, {blocks * TILED_A_STEP}]")
    out = empty((blocks * TILED_A_STEP, K), dtype=a_tiled.dtype, device=a_tiled.device)
    out.view(blocks, TILED_A_STEP, K).copy_(a_tiled.transpose(1, 2))
    return out[:m]


def _tiled_a_selfcheck_detail(n: int, device: torch.device, rows: int) -> Optional[str]:
    """Bitwise check of the tiled-A entry against the row-major entry on synthetic
    data: None when every byte matches, else the mismatch (or launch error) detail.

    Data: a private torch.Generator (the global RNG is not touched), e4m3 A ~
    2*N(0,1) and W ~ N(0,1), bf16 bias, non-power-of-two scales (outputs are
    spread over the e4m3 range, none saturate). The tiled copy's tail tokens are
    e4m3 NaN (0x7F): they must not reach the output. Both outputs are
    0xFF-prefilled; the tiled one has guard rows after row M that must survive."""

    kern = kernel(n)
    fn = _tiled_a_fn(n)
    u8 = torch.uint8
    generator = torch.Generator(device=device)
    generator.manual_seed(0x0B15)
    y = (torch.randn((rows, K), device=device, generator=generator) * 2.0).to(_FP8_E4M3)
    w = torch.randn((n, K), device=device, generator=generator).to(_FP8_E4M3)
    bias = (torch.randn((n,), device=device, generator=generator) * 0.5).to(torch.bfloat16)
    sa = torch.full((1, 1), 0.3671875, dtype=torch.float32, device=device)
    sb = torch.full((1, 1), 1.0 / 7.0, dtype=torch.float32, device=device)
    blocks = tiled_a_shape(rows)[0]
    padded = torch.full((blocks * TILED_A_STEP, K), 0x7F, dtype=u8, device=device)
    padded[:rows] = y.view(u8)
    a_tiled = (
        padded.view(blocks, TILED_A_STEP, K).transpose(1, 2).contiguous().view(_FP8_E4M3)
    )
    ref = torch.full((rows, n), 0xFF, dtype=u8, device=device)
    got = torch.full((rows + _TILED_A_GUARD_ROWS, n), 0xFF, dtype=u8, device=device)
    got[rows:].fill_(0x5A)
    launch(kern, y, w, bias, sa, sb, out=ref.view(_FP8_E4M3))
    rc = _tiled_a_call(fn, a_tiled, rows, w, bias, sa, sb, got[:rows])
    torch.cuda.synchronize(device)  # a device fault propagates (not a verdict)
    if rc != 0:
        return f"N={n} M={rows}: launch returned hipError {rc}"
    bad_ref = int(((ref & 0x7F) == 0x7F).sum().item())  # NaN / unwritten reference bytes
    differ = int((got[:rows] != ref).sum().item())
    guard = int((got[rows:] != 0x5A).sum().item())
    if bad_ref or differ or guard:
        return (
            f"N={n} M={rows}: {differ} of {rows * n} output bytes differ from the "
            f"row-major entry; {guard} guard bytes after row M changed; "
            f"{bad_ref} NaN/unwritten reference bytes ({kern.path})"
        )
    return None


def tiled_a_selfcheck(
    n: int,
    device: Optional[torch.device] = None,
    rows: int = _TILED_A_SELFCHECK_ROWS,
) -> bool:
    """The startup self-check of the tiled-A entry for N on `device` (default: the
    current device), once per device: builds/loads the N library if needed, then
    compares every output byte with the row-major entry (_tiled_a_selfcheck_detail).
    A failure (no library, no entry, launch error, any differing byte) latches the
    entry off for the process and logs once. Returns True when it passed. A device
    fault (an error from the synchronize) propagates unwrapped."""

    device = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, n)
    with _tiled_a_lock:
        if key in _tiled_a_ok:
            return True
        if n in _tiled_a_failures:
            return False
        if n in _failures:  # the row-major entry of the same library is latched off
            _latch_tiled_a(n, f"N={n} row-major entry latched off: {_failures[n]}")
            return False
        t0 = time.perf_counter()
        try:
            detail = _tiled_a_selfcheck_detail(n, torch.device("cuda", index), rows)
        except RuntimeError as error:
            # A latched build/load failure (kernel(n) or the entry symbol) is a
            # verdict; anything else (e.g. a synchronize error) propagates.
            if n not in _tiled_a_failures and n not in _failures:
                raise
            _latch_tiled_a(n, f"N={n} unavailable: {error}")
            return False
        if detail is not None:
            _latch_tiled_a(n, f"self-check FAILED vs the row-major entry: {detail}")
            return False
        _tiled_a_ok.add(key)
    kern = _kernels[n]
    _print(
        f"tiled-A self-check OK: N={n} M={rows} bitwise == row-major entry "
        f"(NaN tail, guard rows intact) on cuda:{index} in "
        f"{time.perf_counter() - t0:.2f}s; lib={kern.path} "
        f"src={os.path.basename(kern.source)}@{kern.source_sha256[:12]}"
    )
    return True


def tiled_a_ready(n: int, build: bool = False) -> bool:
    """True when the tiled-A entry for N passed its self-check on the current device.
    Cheap (no build, launch or sync) unless build=True and the check has not run
    on this device yet: then it runs it (tiled_a_selfcheck, one device sync)."""

    if n not in SUPPORTED_N or n in _tiled_a_failures or n in _failures:
        return False
    if not torch.cuda.is_available():
        return False
    if (torch.cuda.current_device(), n) in _tiled_a_ok:
        return True
    return build and tiled_a_selfcheck(n)


def prebuild_tiled_a(
    ns: Sequence[int] = (1024,), rows: int = _TILED_A_SELFCHECK_ROWS
) -> bool:
    """Prebuild hook of the tiled-A entry, for before RUN_START (after prebuild();
    a library prebuild() loaded is reused): the startup self-check for each N on
    the current device. Returns True when every N passed, False when skipped (no
    GPU, not gfx950). Raises PrebuildError when an N failed (latched and logged
    once; tiled_a_ready(N) stays False). A device fault propagates unwrapped."""

    if not torch.cuda.is_available():
        _print("tiled-A prebuild skipped: no GPU")
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    arch = str(getattr(torch.cuda.get_device_properties(device), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        _print(f"tiled-A prebuild skipped: arch {arch or 'unknown'} (built for gfx950)")
        return False
    errors = []
    for n in ns:
        if n not in SUPPORTED_N:
            errors.append(f"N={n}: not built (supported: {SUPPORTED_N})")
        elif not tiled_a_selfcheck(n, device, rows):
            errors.append(f"N={n}: {_tiled_a_failures.get(n)}")
    if errors:
        _print("tiled-A prebuild FAILED: " + " | ".join(errors))
        raise PrebuildError("K=512 tiled-A prebuild failed: " + " | ".join(errors))
    return True


def log_tiled_a_armed_once(kern: Kernel, m: int) -> None:
    """The one tiled-A arming line per N per process (for the first dispatch)."""

    if kern.n in _tiled_a_armed_logged:
        return
    _tiled_a_armed_logged.add(kern.n)
    _print(
        f"armed tiled-A: N={kern.n} {tiled_a_entry_symbol(kern.n)} lib={kern.path} "
        f"src={os.path.basename(kern.source)}@{kern.source_sha256[:12]} "
        f"device=cuda:{torch.cuda.current_device()} first_M={m}"
    )


def k512_scaled_mm_tiled_a(
    a_tiled: torch.Tensor,
    m: int,
    weight_nk_fp8: torch.Tensor,
    bias_bf16: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """k512_gemm_e4m3(y, ...) with y = the token rows [0, M) of the step-tiled yT
    a_tiled (e4m3 [ceil(M/128), 512, 128], token t / feature c at
    ((t // 128) * 512 + c) * 128 + t % 128): e4m3 [M, N] =
    e4m3_rne(sat448(y @ weight_nk.T * (scale_a * scale_b) + bias)), bit-identical.

    weight_nk_fp8 e4m3 [N, 512] contiguous (N in 1024/1536); bias_bf16 [N];
    scale_a / scale_b one fp32 value each; `out` (optional) a contiguous e4m3
    [M, N] tensor. Raises ValueError for ineligible operands and RuntimeError when
    the entry is not ready (self-check failed: latched). Without a prior
    prebuild_tiled_a(), the first call on a device runs the self-check (one sync)."""

    reason = ineligible_reason_tiled_a(
        a_tiled, m, weight_nk_fp8, bias_bf16, scale_a, scale_b
    )
    if reason is not None:
        raise ValueError(f"k512_scaled_mm_tiled_a: {reason[1]}")
    n = weight_nk_fp8.shape[0]
    if out is not None and (
        out.dtype != _FP8_E4M3
        or tuple(out.shape) != (m, n)
        or (m > 0 and out.stride() != (n, 1))
        or out.device != a_tiled.device
    ):
        raise ValueError(
            f"k512_scaled_mm_tiled_a: out {out.dtype} {tuple(out.shape)} stride "
            f"{out.stride()} (needs a contiguous e4m3 [{m}, {n}] on {a_tiled.device})"
        )
    if not tiled_a_ready(n, build=True):
        raise RuntimeError(
            f"k512_scaled_mm_tiled_a: the tiled-A entry for N={n} is unavailable: "
            f"{_tiled_a_failures.get(n) or _failures.get(n) or 'not ready'}"
        )
    return launch_tiled_a(
        kernel(n), a_tiled, m, weight_nk_fp8, bias_bf16, scale_a, scale_b, out=out
    )


def reset_tiled_a_for_testing() -> None:
    """Test-only: forget the tiled-A latch, self-check results and once-lines."""

    with _tiled_a_lock:
        _tiled_a_fns.clear()
        _tiled_a_failures.clear()
        _tiled_a_ok.clear()
        _tiled_a_armed_logged.clear()
