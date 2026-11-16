# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

# pyre-unsafe

"""Weighted LayerNorm forward producer for the MXFP6 forward K512 GEMMs, as a hipcc-built HIP library.

Behind ``DLRM_HSTU_FP8_FWD_MXFP6=1`` (default off; a DECLARED PRECISION CHANGE of the U / VQK / VK forward
GEMM operands, owned by the integration), the HSTU uvqk / targets-vk forward calls this instead of
``triton_weighted_layer_norm_fwd_fp8_store_t`` (store_t v2, tiled xT). One launch writes::

    mean  fp32 [M]                      == store_t v2 mean (bitwise)
    rstd  fp32 [M]                      == store_t v2 rstd (bitwise)
    xT8   e4m3 [Mp/128, 512, 128]       == store_t v2 yT with tiled_xt=True (bitwise; zero for rows M..Mp-1)
    x6    uint8 [Mp/128, 4, 12288]      MXFP6 E2M3 codes, pr7 "MX6-T" pre-tiled layout (k512 MX6 kernel operand A)
    xs    uint8 [Mp/128, 4, 512]        E8M0 scales, pr7 "MX6-A-v1" layout (byte 127 for pad rows, never 255)
    x8    e4m3 [M, 512] (optional)      == store_t v2 y (bitwise), for an FP8-U fallback
    y     fp32 [M, 512] (debug)         the pre-cast y = LN(x) * inv_scale that all of the above quantize

with Mp = ceil(M / 128) * 128. x6 / xs are byte-identical to pr7's torch reference (proto.py q_e2m3 / enc_e2m3 /
e8m0_exp / pack_rows / tile_A / a_scale_native) applied to y. C ABI of ``ln_store_mx6.cu``::

    int ln_store_mx6_launch(const void* x, const void* w, const void* b,      /* bf16 [M,512], bf16 [512] x2 */
                            void* mean, void* rstd, void* xt8, void* x6, void* xs,
                            void* x8,  /* or NULL */  void* y,  /* fp32 [M,512] or NULL */
                            float eps, float inv_scale, int M, int Mp, void* stream);
        /* 0, this launch's hipError_t, or 100000 + a sticky error an EARLIER HIP call left on the thread */
    const char* ln_store_mx6_impl(void);

Build: ``/opt/rocm/bin/hipcc`` once per process into ``/tmp/dlrm_hstu_ln_store_mx6`` from ``prebuild()``
(before RUN_START) or lazily on the first call; a failed build or self-check is latched. Optional environment:
``DLRM_HSTU_LN_STORE_MX6_SRC`` = another ``.cu`` with the same ABI. Launches are asynchronous on
``torch.cuda.current_stream()``; there is no host sync.

The library holds four kernel instantiations <X8, YD> (x8 written, y written), dispatched on ``x8 / y`` being NULL:
<0,0> is the steady-state call (no x8), <1,0> the FP8-U fallback, <0,1> / <1,1> debug. Every output byte they share
is identical across the four (r14 iter49 review M1; validated in the lab, see RESULT.md).
Optional interface for the integration: ``IMPL`` (str), ``ready(device)`` (None or a reason), ``failure()``,
``ineligible_reason(x, weight, bias)``, ``prebuild()``. xT8 is always written (``xt_out=None`` allocates it).
"""

from __future__ import annotations

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

D: int = 512
TILE: int = 128
KSTEPS: int = 4
X6_TILE_BYTES: int = KSTEPS * 12288  # 128 rows x 512 K x 6 bits
XS_TILE_BYTES: int = KSTEPS * 512  # 128 rows x 16 blocks
HIPCC: str = "/opt/rocm/bin/hipcc"
KNOB: str = "DLRM_HSTU_FP8_FWD_MXFP6"
_SRC_ENV: str = "DLRM_HSTU_LN_STORE_MX6_SRC"
_BUILD_ROOT: Path = Path("/tmp/dlrm_hstu_ln_store_mx6")
_BUILD_TIMEOUT_S: int = 600
_INT32_MAX: int = 2**31 - 1
_ENTRY: str = "ln_store_mx6_launch"
# ln_store_mx6.cu: LNMX6_STALE_ERROR_BASE + code = this launch was clean, but an EARLIER HIP call had left a sticky
# error on the thread (review i3)
_STALE_ERROR_BASE: int = 100000
# Same text as ln_store_mx6_impl() of the in-tree source (a DLRM_HSTU_LN_STORE_MX6_SRC override reports its own text
# in kernel().impl, and its sha in kernel().source_sha256).
IMPL: str = (
    "ln_store_mx6 v3 (r14 iter49 + review fixes): 32-row WG, XCD group 8 (v26.3's store_t v2 map; Triton store_t "
    "v2 now 64-token, XCD group 4), Triton-order statistics, e4m3 xT8 dwords from row-pair converts, MX6-T x6 + "
    "MX6-A-v1 xs via one fp6 convert per 2 rows (s_nop 1 before, s_nop 4 after), LDS-staged copy-out"
)

_FP8_E4M3: Optional[torch.dtype] = getattr(torch, "float8_e4m3fn", None)


class PrebuildError(RuntimeError):
    """prebuild(): the library failed to build, load or self-check (latched)."""


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


def _print(message: str) -> None:
    print(f"[ln-store-mx6] {message}", flush=True)


def source_path() -> Path:
    override = os.environ.get(_SRC_ENV)
    return Path(override) if override else Path(__file__).with_name("ln_store_mx6.cu")


def hipcc_command(source: Path, output: Path, *, extra_flags: Sequence[str] = ()) -> list:
    """-ffp-contract=off: every multiply / add / FMA of the LayerNorm is written out in the source in the
    shipped Triton kernel's order; the compiler must not fuse or reorder them."""

    return [
        HIPCC,
        str(source),
        "--offload-arch=gfx950",
        "-std=c++20",
        "-w",
        "-O3",
        "-ffp-contract=off",
        # The source packs its elementwise math explicitly (float2 = v_pk_*_f32); the SLP vectorizer would also pack
        # the per-row reduction chains of two rows, which needs a v_mov pair per packed add here.
        "-fno-slp-vectorize",
        "-I/opt/rocm/include/hip",
        "-shared",
        "-fPIC",
        "-Rpass-analysis=kernel-resource-usage",
        *extra_flags,
        "-o",
        str(output),
    ]


def _process_build_dir() -> Path:
    global _build_dir
    if _build_dir is None:
        _BUILD_ROOT.mkdir(parents=True, exist_ok=True)
        _build_dir = Path(tempfile.mkdtemp(prefix=f"pid{os.getpid()}_", dir=str(_BUILD_ROOT)))
    return _build_dir


def resource_summary(build_log: str) -> str:
    keep = ("VGPRs", "AGPRs", "VGPRs Spill", "SGPRs Spill", "ScratchSize [bytes/lane]", "Occupancy [waves/SIMD]",
            "LDS Size [bytes/block]")
    per: Dict[str, Dict[str, str]] = {}
    cur = None
    loc = r"(?:\s+\S+:\d+:\d+:)?"
    # ROCm 10.2 (AMD clang 24), bare: "<loc>: Function Name: X" / "<loc>:     <Key>: <v>" (no "remark:", no "[-Rpass")
    bare = r"\S+:\d+:\d+:\s+"
    bare_key = (r"(TotalSGPRs|VGPRs|AGPRs|ScratchSize \[bytes/lane\]|Dynamic Stack|Occupancy \[waves/SIMD\]|"
                r"SGPRs Spill|VGPRs Spill|LDS Size \[bytes/block\])")
    for line in build_log.splitlines():
        m = re.search(r"remark:" + loc + r"\s+Function Name:\s+(\S+)", line)
        m = m or re.fullmatch(bare + r"Function Name:\s+(\S+)\s*", line)
        if m:
            cur = m.group(1)
            per[cur] = {}
            continue
        m = re.search(r"remark:" + loc + r"\s+([A-Za-z][A-Za-z /\[\]]*?):\s+(\S+)\s+\[-Rpass", line)
        m = m or re.fullmatch(bare + bare_key + r":\s+(\S+)\s*", line)
        if m and cur is not None:
            per[cur][m.group(1).strip()] = m.group(2)
    parts = []
    for name, f in per.items():
        parts.append(instantiation_tag(name) + ": " + " ".join(f"{k}={f[k]}" for k in keep if k in f))
    return "; ".join(parts)


def instantiation_tag(mangled: str) -> str:
    """'<X8,YD>=<1,0>' from the Itanium-mangled kernel name ln_store_mx6_kernelILb<X8>ELb<YD>EEEv... (review i1: the
    old suffix match expected S1_/S2_ substitutions, but the names carry S2_/S3_/S4_, so it mislabelled 2 of 4)."""

    m = re.search(r"ln_store_mx6_kernelILb([01])ELb([01])EE", mangled)
    return f"<X8,YD>=<{m.group(1)},{m.group(2)}>" if m else mangled


def _build(source: Path) -> Tuple[Path, float, str]:
    out_dir = _process_build_dir()
    output = out_dir / "libln_store_mx6.so"
    temporary = out_dir / f".libln_store_mx6.{os.getpid()}.so"
    log_path = out_dir / "build.log"
    command = hipcc_command(source, temporary)
    t0 = time.perf_counter()
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                timeout=_BUILD_TIMEOUT_S, check=False)
    except FileNotFoundError as error:
        raise RuntimeError(f"hipcc not found ({HIPCC})") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"hipcc timed out after {_BUILD_TIMEOUT_S}s") from error
    seconds = time.perf_counter() - t0
    log_path.write_text(" ".join(command) + "\n" + result.stdout)
    if result.returncode != 0 or not temporary.exists():
        lines = result.stdout.strip().splitlines()
        errors = [line.strip() for line in lines if "error" in line.lower()]
        raise RuntimeError(f"hipcc rc={result.returncode} (log {log_path}): {' | '.join((errors or lines)[:3])[:400]}")
    os.replace(temporary, output)
    return output, seconds, resource_summary(result.stdout)


def _load() -> Kernel:
    source = source_path()
    blob = source.read_bytes()
    path, seconds, resources = _build(source)
    library = ctypes.CDLL(str(path))
    fn = getattr(library, _ENTRY)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_void_p]
    impl = "?"
    impl_fn = getattr(library, "ln_store_mx6_impl", None)
    if impl_fn is not None:
        impl_fn.restype = ctypes.c_char_p
        impl_fn.argtypes = []
        impl = (impl_fn() or b"?").decode(errors="replace")
    return Kernel(path=str(path), fn=fn, impl=impl, source=str(source),
                  source_sha256=hashlib.sha256(blob).hexdigest(), build_s=seconds, resources=resources)


def kernel() -> Kernel:
    """The loaded library (built on first use). RuntimeError if it cannot be built or loaded (latched)."""

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
            _failure = f"ln_store_mx6 library unavailable: {error}"
            raise RuntimeError(_failure) from error
        return _kernel


def failure() -> Optional[str]:
    return _failure


def ready(device: Optional[torch.device] = None) -> Optional[str]:
    """None when this producer may be used on `device` (default: the current CUDA device), else a short reason: no
    float8_e4m3fn in torch, no GPU, a non-CUDA or non-gfx950 device, or a latched build / self-check failure. Does not
    build: before prebuild() (or the first call) it reports None and the first call builds."""

    if _FP8_E4M3 is None:
        return "torch has no float8_e4m3fn"
    if _failure is not None:
        return _failure
    if not torch.cuda.is_available():
        return "no GPU"
    dev = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    if dev.type != "cuda":
        return f"device {dev} is not a CUDA device"
    index = torch.cuda.current_device() if dev.index is None else dev.index
    arch = str(getattr(torch.cuda.get_device_properties(index), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        return f"arch {arch or 'unknown'} on {dev} (built for gfx950)"
    return None


def padded_rows(m: int) -> int:
    return (m + TILE - 1) // TILE * TILE


def ineligible_reason(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> Optional[str]:
    """None if the kernel takes these operands, else a short reason (the caller keeps the Triton store_t)."""

    if _FP8_E4M3 is None:
        return "torch has no float8_e4m3fn"
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != D or not x.is_contiguous():
        return f"x {x.dtype} {tuple(x.shape)} stride {x.stride()} (needs contiguous bf16 [M, {D}])"
    if x.shape[0] < 1 or padded_rows(x.shape[0]) > _INT32_MAX:
        return f"M={x.shape[0]} out of range"
    for name, t in (("weight", weight), ("bias", bias)):
        if t.dtype != torch.bfloat16 or tuple(t.shape) != (D,) or t.stride() != (1,):
            return f"{name} {t.dtype} {tuple(t.shape)} stride {t.stride()} (needs contiguous bf16 [{D}])"
    dev = x.device
    if dev.type != "cuda" or weight.device != dev or bias.device != dev:
        return "operands are not all on one CUDA device"
    if dev.index is not None and dev.index != torch.cuda.current_device():
        return f"operands on {dev}, current device cuda:{torch.cuda.current_device()}"
    if x.data_ptr() % 16 or weight.data_ptr() % 16 or bias.data_ptr() % 16:
        return "x / weight / bias not 16-byte aligned"
    return None


def _check_out(name: str, t: torch.Tensor, shape: Tuple[int, ...], dtype: torch.dtype, dev: torch.device) -> None:
    if tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous() or t.device != dev or t.data_ptr() % 16:
        raise ValueError(f"ln_store_mx6: {name} must be a contiguous 16-B aligned {dtype} {shape} on {dev}, got "
                         f"{t.dtype} {tuple(t.shape)} stride {t.stride()} on {t.device}")


def ln_store_mx6(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    inv_scale: float,
    mp: Optional[int] = None,
    *,
    want_x8: bool = False,
    return_y: bool = False,
    xt_out: Optional[torch.Tensor] = None,
    x6_out: Optional[torch.Tensor] = None,
    xs_out: Optional[torch.Tensor] = None,
    mean_out: Optional[torch.Tensor] = None,
    rstd_out: Optional[torch.Tensor] = None,
    x8_out: Optional[torch.Tensor] = None,
    y_out: Optional[torch.Tensor] = None,
):
    """LayerNorm forward -> (x6, xs, xT8, mean, rstd, x8 or None) [+ (y,) if return_y].

    x bf16 [M, 512] contiguous; weight / bias bf16 [512]; eps, inv_scale: the host floats the Triton store_t gets
    (both rounded to fp32 as the Triton launcher does); mp must be ceil(M / 128) * 128 when given (the uvqk_tm_pad of
    the call site). Shapes: x6 uint8 [T, 4, 12288], xs uint8 [T, 4, 512], xT8 e4m3 [T, 512, 128], mean / rstd fp32
    [M], x8 e4m3 [M, 512], y fp32 [M, 512]; T = Mp / 128. xt_out / x6_out / xs_out: optional preallocated outputs
    (e.g. xT8 from fp8_tm_alloc.empty_xt); mean_out / rstd_out / x8_out / y_out likewise (x8_out implies want_x8,
    y_out implies return_y). Stream-ordered on the current stream; raises ValueError on operands the
    kernel does not take (see ineligible_reason) and RuntimeError on a build failure or a launch error."""

    reason = ineligible_reason(x, weight, bias)
    if reason is not None:
        raise ValueError(f"ln_store_mx6: {reason}")
    m = x.shape[0]
    mp_ = padded_rows(m)
    if mp is not None and mp != mp_:
        raise ValueError(f"ln_store_mx6: mp={mp} but this kernel writes Mp = ceil(M/128)*128 = {mp_}")
    t = mp_ // TILE
    dev = x.device
    if xt_out is None:
        xt_out = torch.empty((t, D, TILE), dtype=_FP8_E4M3, device=dev)
    if x6_out is None:
        x6_out = torch.empty((t, KSTEPS, 12288), dtype=torch.uint8, device=dev)
    if xs_out is None:
        xs_out = torch.empty((t, KSTEPS, 512), dtype=torch.uint8, device=dev)
    _check_out("xt_out", xt_out, (t, D, TILE), _FP8_E4M3, dev)
    _check_out("x6_out", x6_out, (t, KSTEPS, 12288), torch.uint8, dev)
    _check_out("xs_out", xs_out, (t, KSTEPS, 512), torch.uint8, dev)
    want_x8 = want_x8 or x8_out is not None
    return_y = return_y or y_out is not None
    mean = torch.empty((m,), dtype=torch.float32, device=dev) if mean_out is None else mean_out
    rstd = torch.empty((m,), dtype=torch.float32, device=dev) if rstd_out is None else rstd_out
    x8 = x8_out if x8_out is not None else (torch.empty((m, D), dtype=_FP8_E4M3, device=dev) if want_x8 else None)
    y = y_out if y_out is not None else (torch.empty((m, D), dtype=torch.float32, device=dev) if return_y else None)
    _check_out("mean_out", mean, (m,), torch.float32, dev)
    _check_out("rstd_out", rstd, (m,), torch.float32, dev)
    if x8 is not None:
        _check_out("x8_out", x8, (m, D), _FP8_E4M3, dev)
    if y is not None:
        _check_out("y_out", y, (m, D), torch.float32, dev)
    kern = kernel()
    stream = torch.cuda.current_stream(dev).cuda_stream
    rc = kern.fn(x.data_ptr(), weight.data_ptr(), bias.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
                 xt_out.data_ptr(), x6_out.data_ptr(), xs_out.data_ptr(), x8.data_ptr() if x8 is not None else None,
                 y.data_ptr() if y is not None else None, float(eps), float(inv_scale), m, mp_, stream)
    if rc >= _STALE_ERROR_BASE:
        raise RuntimeError(f"{_ENTRY}: hipError {rc - _STALE_ERROR_BASE} was left pending by an EARLIER HIP call on "
                           f"this thread (not raised by this kernel, whose launch was issued without error; M={m}, "
                           f"lib={kern.path})")
    if rc != 0:
        raise RuntimeError(f"{_ENTRY} failed: hipError {rc} (M={m}, lib={kern.path})")
    res = (x6_out, xs_out, xt_out, mean, rstd, x8)
    return res + (y,) if return_y else res


# ------------------------------------------------------------------ prebuild self-check (r14 iter49 review m9)
# One deterministic, non-zero LayerNorm input whose every output is exact, so that the expected bytes can be pinned:
# * row r: x = mu_r + a_r * s_r(c) with s_r(c) = +-1 (parity of c & mask_r: exactly 256 of each sign per row),
#   mu_r in {-0.75, ..., 0.75}, a_r in {1/8, 1/2, 1, 2, 16}. Every sum is exact, so mean = mu_r and var = a_r^2 (a power
#   of 4); eps = 2^-60 (>= FLT_MIN: the production rsqrt branch) vanishes in var + eps, rstd = 1 / a_r, and
#   (x - mean) * rstd = s_r(c) exactly: y = (s * w + b) * inv_scale with inv_scale = 0.5.
# * 32-column block q with exponent k = _SC_K[q] (-3..8): column 0 zero (w = b = 0); column 1 bias-only y = -2^(k-5),
#   a code that rounds to -0 and must be written +0; column 31 bias-only y = +-7.5 * 2^k, the block amax, so the E8M0
#   byte is k + 127 at the exact ceil-rule boundary; columns 2..30 weight-only y = s * g * 2^k for 29 of the other 30
#   E2M3 magnitudes g (rotated by q). Every y is an e4m3 value or saturates (k = 6..8), every y / 2^k an E2M3 value.
# * Robust to a 1-ulp error of the hardware rsqrt: no weight-column value is at a rounding or an E8M0 boundary.
# * Leg A runs want_x8=True (<1,0>), leg B want_x8=False (<0,0>, the steady-state kernel: its first launch happens
#   here, before RUN_START). The digests are of leg A; leg B must equal leg A byte for byte.
# The digests are recomputed on the CPU by P49/fp6fwd/prod/scripts/selfcheck_ref.py from pr7 proto.py's reference
# functions (the copy embedded in ops/tests/ln_store_mx6_test.py) and checked on gfx950 by the lab validation.
_SC_ROWS: int = 257  # 2 full tiles + 1 row: full blocks, a partial PAD block (row 256) and all-pad blocks
_SC_K: Tuple[int, ...] = (0, 1, -1, 2, -2, 3, -3, 4, 5, 6, 7, 8, 0, -1, 2, 1)
_SC_EPS: float = 2.0**-60
_SC_INV: float = 0.5
_SC_SHA256: Dict[str, str] = {
    "x6": "2ee585f835aff469f0b8bf2f073c15fc78fb93bb10c126c8cf001e85d60e9923",
    "xs": "9cacc00ddeafacf33cf66414974a9fa02f9be6619fe17d35904cebf1d964bb53",
    "xt8": "ac6236643109d42cab419475e506254842aa73ed2544b8dac47558541e6898c8",
    "x8": "4bc242acf93751fe5c32f69e66c149161e2a250a4ae1154a0ebf4763a76564ab",
}


def _e2m3_magnitudes() -> list:
    """The 31 positive E2M3 values in code order: 0.125 .. 0.875, 1 .. 1.875, 2 .. 3.75, 4 .. 7.5."""

    values = [m / 8.0 for m in range(1, 8)]
    for e in (1, 2, 3):
        values += [2.0 ** (e - 1) * (1.0 + m / 8.0) for m in range(8)]
    return values


def selfcheck_case(device) -> tuple:
    """(x, weight, bias, eps, inv_scale, expected mean, expected rstd) of the prebuild self-check (see above)."""

    m = _SC_ROWS
    r = torch.arange(m, dtype=torch.int64, device=device)
    c = torch.arange(D, dtype=torch.int64, device=device)
    v = c[None, :] & ((r * 37) % 511 + 1)[:, None]
    for shift in (8, 4, 2, 1):
        v = v ^ (v >> shift)
    s = 1.0 - 2.0 * (v & 1).to(torch.float32)
    a = torch.tensor([0.125, 0.5, 1.0, 2.0, 16.0], dtype=torch.float32, device=device)[r % 5]
    mu = ((r % 7) - 3).to(torch.float32) * 0.25
    x = (mu[:, None] + a[:, None] * s).to(torch.bfloat16)
    g = _e2m3_magnitudes()[:30]  # 7.5 only as the bias-only block amax
    w = [0.0] * D
    b = [0.0] * D
    for q, k in enumerate(_SC_K):
        base = 32 * q
        b[base + 1] = -(2.0 ** (k - 5)) / _SC_INV
        b[base + 31] = (1.0 if q % 2 == 0 else -1.0) * 7.5 * 2.0**k / _SC_INV
        for f in range(2, 31):
            w[base + f] = g[(f - 2 + q) % 30] * 2.0**k / _SC_INV
    weight = torch.tensor(w, dtype=torch.float64).to(torch.bfloat16).to(device)
    bias = torch.tensor(b, dtype=torch.float64).to(torch.bfloat16).to(device)
    return x, weight, bias, _SC_EPS, _SC_INV, mu, 1.0 / a


def _sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(bytes(t.contiguous().view(torch.uint8).reshape(-1).cpu().tolist())).hexdigest()


def _selfcheck(device: torch.device) -> Tuple[list, str]:
    """Run both legs into poisoned buffers -> (list of failures, note)."""

    u8 = torch.uint8
    x, w, b, eps, inv, mean_exp, rstd_exp = selfcheck_case(device)
    m = x.shape[0]
    t = padded_rows(m) // TILE

    def poisoned(want_x8: bool) -> dict:
        out = dict(
            xt_out=torch.full((t, D, TILE), 0xA5, dtype=u8, device=device).view(_FP8_E4M3),
            x6_out=torch.full((t, KSTEPS, 12288), 0xA5, dtype=u8, device=device),
            xs_out=torch.full((t, KSTEPS, 512), 0xA5, dtype=u8, device=device),
            mean_out=torch.full((m,), float("nan"), dtype=torch.float32, device=device),
            rstd_out=torch.full((m,), float("nan"), dtype=torch.float32, device=device),
        )
        if want_x8:
            out["x8_out"] = torch.full((m, D), 0xA5, dtype=u8, device=device).view(_FP8_E4M3)
        return out

    leg_a = ln_store_mx6(x, w, b, eps, inv, padded_rows(m), want_x8=True, **poisoned(True))  # <1,0>
    leg_b = ln_store_mx6(x, w, b, eps, inv, padded_rows(m), want_x8=False, **poisoned(False))  # <0,0>
    torch.cuda.synchronize(device)
    x6, xs, xt, mean, rstd, x8 = leg_a
    bad = []
    if leg_b[5] is not None:
        bad.append("leg B returned x8")
    for name, i in (("x6", 0), ("xs", 1), ("xT8", 2), ("mean", 3), ("rstd", 4)):
        if not torch.equal(leg_a[i].view(u8), leg_b[i].view(u8)):
            bad.append(f"{name} differs between <1,0> and <0,0>")
    for name, tensor in (("x6", x6), ("xs", xs), ("xt8", xt), ("x8", x8)):
        digest = _sha256(tensor)
        if digest != _SC_SHA256[name]:
            bad.append(f"{name} sha256 {digest[:16]} != pinned {_SC_SHA256[name][:16]}")
    i32 = torch.int32
    if not torch.equal(mean.view(i32), mean_exp.view(i32)):
        bad.append("mean != exact")
    ulp = int((rstd.view(i32).to(torch.int64) - rstd_exp.view(i32).to(torch.int64)).abs().max().item())
    if ulp > 1 or not bool(torch.isfinite(rstd).all().item()):
        bad.append(f"rstd off by {ulp} ulp")
    return bad, f"rstd {'exact' if ulp == 0 else f'within {ulp} ulp'}"


def prebuild(warm_rows: int = _SC_ROWS) -> bool:
    """Build and load before RUN_START, then self-check both production instantiations (leg A want_x8=True, leg B
    want_x8=False) on the deterministic non-zero input above, every output pre-filled with 0xA5 bytes / NaN: leg A's
    x6 / xs / xT8 / x8 against pinned sha256s, its mean bitwise and its rstd within 1 ulp of the exact values, and
    leg B byte-identical to leg A in x6 / xs / xT8 / mean / rstd. Returns False when skipped (no GPU / not gfx950);
    raises PrebuildError on a failure (latched: later calls raise, ready() and failure() report it). warm_rows is
    ignored (kept for callers of the previous signature): the self-check case is fixed at 257 rows."""

    global _kernel, _failure
    del warm_rows
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
    except RuntimeError as error:
        _print(f"prebuild FAILED: {error}")
        raise PrebuildError(str(error)) from error
    t_build = time.perf_counter() - t0
    try:
        bad, note = _selfcheck(device)
    except Exception as error:  # a launch / allocation error of the self-check is a failure too
        bad, note = [f"{type(error).__name__}: {error}"], ""
    if bad:
        message = f"self-check failed: {'; '.join(bad)} ({kern.path})"
        with _lock:
            _kernel = None
            _failure = f"ln_store_mx6 {message}"
        _print("prebuild FAILED: " + message)
        raise PrebuildError("ln_store_mx6 prebuild failed: " + message)
    _print(f"prebuilt in {t_build:.1f}s (dir {_process_build_dir()}, impl={kern.impl!r}, "
           f"src={os.path.basename(kern.source)}@{kern.source_sha256[:12]}); self-check M={_SC_ROWS} <1,0> and <0,0> "
           f"OK (x6/xs/xT8/x8 pinned sha256, mean exact, {note}, legs byte-identical); [{kern.resources}]")
    return True
