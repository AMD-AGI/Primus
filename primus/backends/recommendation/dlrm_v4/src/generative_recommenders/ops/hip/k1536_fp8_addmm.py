# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

# pyre-unsafe

"""K=1536 FP8 output-projection forward GEMM with the fused bf16 residual, as a hipcc-built HIP library.

Behind ``DLRM_HSTU_FP8_FWD_K1536=1``, ``triton_fp8.scaled_addmm_fp8_preq`` calls this
instead of the residual-fused hipBLASLt call of its residual branch
(``ops/triton/hipblaslt_scaled_addmm.cpp``: D = alpha * (scaleA * scaleB * A x B) + beta * C,
alpha = beta = 1, compute 32F, heuristic algorithm index 1, 64 MiB workspace)::

    D[M, 512] bf16 = bf16_rne( fp32( acc * (s_w * s_act) ) + fp32(C[m, n]) )
    acc = sum_k A[m, k] * W[n, k] in fp32 (twelve in-order K=128 MFMA steps); K = 1536

The replacement must be bit-identical to that call. ``ops/tests/k1536_fp8_addmm_test.py``
compares every output word through ``scaled_addmm_fp8_preq`` itself.
That holds on hipBLASLt's data-parallel tiles and on narrow Gaussian data (the test's).
At in-model M hipBLASLt runs a persistent stream-K grid whose tail tiles add split-K
partial sums in fp32 through its workspace, so an order-sensitive row there can round
differently: an fp32 summation-order difference (yambda_5b.gin item (44), NUMERICS).

C ABI of the library (``k1536_fp8_addmm.cu``)::

    int k1536_fp8_addmm_res_bf16_N512(const void* A,      /* e4m3 [M, 1536] row-major */
                                      const void* W,      /* e4m3 [512, 1536] row-major */
                                      const void* C,      /* bf16 [M, 512] residual */
                                      void* D,            /* bf16 [M, 512], rows < M, D != C */
                                      const float* s_act, /* fp32 device scalar */
                                      const float* s_w,   /* fp32 device scalar */
                                      int M,              /* >= 1 */
                                      void* stream);      /* hipStream_t; returns hipGetLastError() */
    const char* k1536_fp8_addmm_impl(void);

The kernel is P7's K=512 forward GEMM family (``k512_fp8_gemm.cu``): two resident 4-wave workgroups
per CU, 128x256 tile, 5-slot 80 KiB LDS-DMA ring, operand-swapped MFMA, XCD-aware tile order; the
residual is LDS-DMA'd into the ring slots the last two K steps free and added in the bf16 epilogue.

Build: ``/opt/rocm/bin/hipcc`` once per process into a per-process directory under
``/tmp/dlrm_hstu_fp8_k1536``, from ``prebuild()`` (``triton_fp8.prebuild_extensions``, before
MLPerf RUN_START) or lazily on the first call. A failed build is latched. Optional environment:
``DLRM_HSTU_FP8_FWD_K1536_SRC`` (another ``.cu`` with the same ABI).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

from generative_recommenders.ops.hip.k512_fp8_gemm import HIPCC, resource_summary

K: int = 1536
N: int = 512
KNOB: str = "DLRM_HSTU_FP8_FWD_K1536"
_SRC_ENV: str = "DLRM_HSTU_FP8_FWD_K1536_SRC"
_ENTRY: str = "k1536_fp8_addmm_res_bf16_N512"
_BUILD_ROOT: Path = Path("/tmp/dlrm_hstu_fp8_k1536")
_BUILD_TIMEOUT_S: int = 600
_INT32_MAX: int = 2**31 - 1

_FP8_E4M3: Optional[torch.dtype] = getattr(torch, "float8_e4m3fn", None)


class PrebuildError(RuntimeError):
    """prebuild(): the library failed to build, load or self-check (latched; the
    forward then keeps the hipBLASLt call). A device fault is not wrapped."""


@dataclass(frozen=True)
class Kernel:
    path: str
    fn: object  # ctypes function pointer
    impl: str
    source: str
    source_sha256: str
    build_s: float
    resources: str


_lock = threading.Lock()
_kernel: Optional[Kernel] = None
_failure: Optional[str] = None
_build_dir: Optional[Path] = None
_armed_logged: bool = False
_decline_logged: set = set()


def _print(message: str) -> None:
    print(f"[k1536-fwd] {message}", flush=True)


def source_path() -> Path:
    override = os.environ.get(_SRC_ENV)
    return Path(override) if override else Path(__file__).with_name("k1536_fp8_addmm.cu")


def hipcc_command(source: Path, output: Path) -> list:
    """The flags the kernel was gated with (iterative-ilp scheduler, -O3) plus
    -ffp-contract=off (acc*s and +C are two roundings, as in hipBLASLt)."""

    return [
        HIPCC,
        str(source),
        "--offload-arch=gfx950",
        "-std=c++20",
        "-w",
        "-O3",
        "-ffp-contract=off",
        "-I/opt/rocm/include/hip",
        "-mllvm",
        "-amdgpu-sched-strategy=iterative-ilp",
        "-shared",
        "-fPIC",
        "-Rpass-analysis=kernel-resource-usage",
        "-o",
        str(output),
    ]


def _process_build_dir() -> Path:
    global _build_dir
    if _build_dir is None:
        _BUILD_ROOT.mkdir(parents=True, exist_ok=True)
        _build_dir = Path(
            tempfile.mkdtemp(prefix=f"pid{os.getpid()}_", dir=str(_BUILD_ROOT))
        )
    return _build_dir


def _load() -> Kernel:
    source = source_path()
    blob = source.read_bytes()
    out_dir = _process_build_dir()
    output = out_dir / "libk1536_fp8_addmm.so"
    temporary = out_dir / f".libk1536_fp8_addmm.{os.getpid()}.so"
    log_path = out_dir / "build.log"
    command = hipcc_command(source, temporary)
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
        raise RuntimeError(f"hipcc rc={result.returncode} (log {log_path}): {tail[:400]}")
    os.replace(temporary, output)
    library = ctypes.CDLL(str(output))
    fn = getattr(library, _ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_void_p]
    impl = "?"
    impl_fn = getattr(library, "k1536_fp8_addmm_impl", None)
    if impl_fn is not None:
        impl_fn.restype = ctypes.c_char_p
        impl_fn.argtypes = []
        impl = (impl_fn() or b"?").decode(errors="replace")
    return Kernel(
        path=str(output),
        fn=fn,
        impl=impl,
        source=str(source),
        source_sha256=hashlib.sha256(blob).hexdigest(),
        build_s=seconds,
        resources=resource_summary(result.stdout),
    )


def kernel() -> Kernel:
    """The loaded library (built on first use). Raises RuntimeError if it cannot
    be built or loaded; the failure is latched for the process."""

    global _kernel, _failure
    if _kernel is not None:
        return _kernel
    with _lock:
        if _kernel is not None:
            return _kernel
        if _failure is not None:
            raise RuntimeError(_failure)
        try:
            _kernel = _load()
        except Exception as error:
            _failure = f"K=1536 FP8 out-projection GEMM library unavailable: {error}"
            raise RuntimeError(_failure) from error
        return _kernel


def failure() -> Optional[str]:
    return _failure


def ineligible_reason(
    activation: torch.Tensor,
    weight_nk: torch.Tensor,
    residual: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> Optional[Tuple[str, str]]:
    """(kind, detail) if the kernel cannot take these operands, else None.
    weight_nk is the [512, 1536] row-major storage of the production
    column-major [1536, 512] weight shadow view."""

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
    if (
        weight_nk.dtype != _FP8_E4M3
        or tuple(weight_nk.shape) != (N, K)
        or weight_nk.stride() != (K, 1)
    ):
        return (
            "weight",
            f"weight {weight_nk.dtype} {tuple(weight_nk.shape)} stride "
            f"{weight_nk.stride()} (needs e4m3 [{N}, {K}] row-major)",
        )
    m = activation.shape[0]
    if (
        residual.dtype != torch.bfloat16
        or tuple(residual.shape) != (m, N)
        or not residual.is_contiguous()
    ):
        return (
            "residual",
            f"residual {residual.dtype} {tuple(residual.shape)} stride "
            f"{residual.stride()} (needs contiguous bf16 [{m}, {N}])",
        )
    for name, scale in (("scale_a", scale_a), ("scale_b", scale_b)):
        if scale.dtype != torch.float32 or scale.numel() != 1:
            return (
                "scale",
                f"{name} {scale.dtype} {tuple(scale.shape)} (needs one fp32 value)",
            )
    device = activation.device
    if device.type != "cuda" or any(
        t.device != device for t in (weight_nk, residual, scale_a, scale_b)
    ):
        return ("device", "operands are not all on one CUDA device")
    if device.index is not None and device.index != torch.cuda.current_device():
        return (
            "device",
            f"operands on {device}, current device cuda:{torch.cuda.current_device()}",
        )
    if m > _INT32_MAX // 2:
        return ("rows", f"M={m} exceeds the int32 grid")
    if activation.data_ptr() % 16 or weight_nk.data_ptr() % 16 or residual.data_ptr() % 16:
        return ("align", "activation/weight/residual not 16-byte aligned")
    return None


def _empty_output(residual: torch.Tensor) -> torch.Tensor:
    # A separate function so the unit test can hand out a NaN-prefilled buffer.
    return torch.empty_like(residual)


def launch(
    kern: Kernel,
    activation: torch.Tensor,
    weight_nk: torch.Tensor,
    residual: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the kernel on operands that passed ineligible_reason(); returns `out`
    or a new bf16 [M, 512] tensor (stream-ordered on the current stream)."""

    m = activation.shape[0]
    if out is None:
        out = _empty_output(residual)
    if m == 0:
        return out
    stream = torch.cuda.current_stream(activation.device).cuda_stream
    rc = kern.fn(
        activation.data_ptr(),
        weight_nk.data_ptr(),
        residual.data_ptr(),
        out.data_ptr(),
        scale_a.data_ptr(),
        scale_b.data_ptr(),
        m,
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"{_ENTRY} failed: hipError {rc} (M={m}, lib={kern.path})")
    return out


def log_armed_once(kern: Kernel, m: int) -> None:
    global _armed_logged
    if _armed_logged:
        return
    _armed_logged = True
    _print(
        f"armed: K={K} N={N} out-projection + residual lib={kern.path} ({KNOB}=1) "
        f"impl={kern.impl!r} src={os.path.basename(kern.source)}@{kern.source_sha256[:12]} "
        f"device=cuda:{torch.cuda.current_device()} first_M={m}"
    )


def log_decline_once(key: str, message: str) -> None:
    """One line per decline kind per process: this call stays on hipBLASLt."""

    if key in _decline_logged:
        return
    _decline_logged.add(key)
    _print(f"declined -> hipBLASLt ({KNOB}=1): {message}")


def prebuild(warm_rows: int = 257) -> bool:
    """Build and load the library before RUN_START, then run it once on
    synthetic zeros (warm_rows rows: one full tile plus a tail row) into a
    0xFFFF-prefilled (bf16 NaN) buffer and check that every output word became
    +0. No data, model state or RNG is touched. Returns True when ready, False
    when skipped (no gfx950). Raises PrebuildError on a failed build or
    self-check (latched). A device fault propagates unwrapped."""

    global _kernel, _failure
    if not torch.cuda.is_available():
        _print("prebuild skipped: no GPU")
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    arch = str(getattr(torch.cuda.get_device_properties(device), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        _print(f"prebuild skipped: arch {arch or 'unknown'} (built for gfx950)")
        return False
    t0 = time.perf_counter()
    try:
        kern = kernel()
    except Exception as error:
        _print(f"prebuild FAILED: {error}")
        raise PrebuildError(f"K=1536 FP8 GEMM prebuild failed: {error}") from error
    t_build = time.perf_counter() - t0
    u8 = torch.uint8
    a = torch.zeros((warm_rows, K), dtype=u8, device=device).view(_FP8_E4M3)
    w = torch.zeros((N, K), dtype=u8, device=device).view(_FP8_E4M3)
    c = torch.zeros((warm_rows, N), dtype=torch.bfloat16, device=device)
    out = torch.full((warm_rows, N), -1, dtype=torch.int16, device=device).view(torch.bfloat16)
    one = torch.ones((1, 1), dtype=torch.float32, device=device)
    launch(kern, a, w, c, one, one, out=out)
    torch.cuda.synchronize(device)
    nonzero = int(torch.count_nonzero(out.view(torch.int16)).item())
    if nonzero:
        message = f"self-check on zeros wrote {nonzero} nonzero words ({kern.path})"
        with _lock:
            _kernel = None
            _failure = f"K=1536 FP8 GEMM self-check failed: {message}"
        _print("prebuild FAILED: " + message)
        raise PrebuildError("K=1536 FP8 GEMM prebuild failed: " + message)
    _print(
        f"prebuilt: K={K} N={N} in {t_build:.1f}s (dir {_process_build_dir()}, "
        f"impl={kern.impl!r}); self-check M={warm_rows} OK; [{kern.resources}]"
    )
    return True
