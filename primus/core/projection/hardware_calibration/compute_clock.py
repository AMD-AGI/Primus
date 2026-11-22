###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Back-calculate the effective compute clock from measured GEMM times.

Two estimates are produced from the same measurements:

* **Origami inversion** (preferred).  Origami predicts a GEMM's latency in
  cycles for a given clock; the effective clock is the one at which its
  predicted time equals the measured time.  Solved by bisection, since the
  predicted time falls monotonically as the clock rises.
* **MFMA-peak lower bound.**  The clock the part would need to sustain the
  achieved FLOP rate at 100 % matrix-core utilisation.  The real clock is at
  least this; it needs no performance model, so it is always available.
"""

from __future__ import annotations

import contextlib
import io
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from primus.core.projection.hardware_calibration.device_info import run_until_settled
from primus.core.projection.hardware_calibration.gemm_shapes import GemmShape

# Dense matrix-core FLOPs per CU per clock.  MI300X: bf16 1307 TF = 304 CU x
# 2.1 GHz x 2048, fp8 2615 TF (x4096), fp32 163 TF (x256).  MI355X: bf16
# 2.5 PF = 256 CU x 2.4 GHz x 4096, fp8 / MXFP8 5.0 PF (x8192), MXFP4
# 10.1 PF (x16384), fp32 157 TF (x256).
_MFMA_FLOPS_PER_CU_PER_CLK = {
    "gfx942": {"fp32": 256, "bf16": 2048, "fp16": 2048, "fp8": 4096},
    "gfx950": {"fp32": 256, "bf16": 4096, "fp16": 4096, "fp8": 8192, "mx8": 8192, "mx4": 16384},
}

# Origami prices MX / sub-8-bit GEMMs as fp8, so only these are inverted.
ORIGAMI_MODELLED_DTYPES = ("fp32", "bf16", "fp16", "fp8")
# gfx942 has no microscaled matrix cores; elsewhere native MX is attempted.
_NO_MX_ARCHS = ("gfx942",)


def gemm_dtype_candidates(workload_dtype: str, gcn_arch: Optional[str] = None) -> List[str]:
    """Dtypes to time a workload GEMM in, best first; later entries are fallbacks.

    MXFP8 / MXFP4 run natively where the part has microscaled matrix cores
    and fall back to fp8; fp6 (no torch kernel) is timed as fp8; fp8 falls
    back to bf16; anything else is bf16.
    """
    dtype = (workload_dtype or "bf16").lower()
    native_mx = gcn_arch not in _NO_MX_ARCHS
    if dtype in ("mx4", "fp4"):
        return (["mx4"] if native_mx else []) + ["fp8", "bf16"]
    if dtype == "mx8":
        return (["mx8"] if native_mx else []) + ["fp8", "bf16"]
    if dtype in ("mx6", "fp6", "fp8"):
        return ["fp8", "bf16"]
    return [dtype] if dtype in ("fp32", "bf16", "fp16") else ["bf16"]


CLOCK_SEARCH_MIN_MHZ = 100
CLOCK_SEARCH_MAX_MHZ = 4000

# SCLK on MI300-class parts takes ~1 s after launch to fall from its boost to
# the level it holds under sustained MFMA load.
DEFAULT_MIN_SETTLE_S = 1.0

_TARGET_TRIAL_MS = 25.0
_MAX_ITERS_PER_TRIAL = 1000


def time_gemm_ms(
    shape: GemmShape,
    device,
    trials: int = 5,
    reader=None,
    sampler=None,
    min_settle_s: float = DEFAULT_MIN_SETTLE_S,
) -> Tuple[float, Dict]:
    """``(median per-call ms, settle info)`` of a GEMM of *shape* on *device*.

    Runs in ``shape.dtype`` (``fp32`` / ``bf16`` / ``fp16`` / ``fp8`` /
    ``mx8`` / ``mx4``; pick from :func:`gemm_dtype_candidates` first).  fp8
    and MX use ``torch._scaled_mm`` with bf16 output; fp8 in the part's
    native format, MX with 32-element e8m0 block scales.

    The GEMM first runs until the compute clock settles (see
    :func:`run_until_settled`), so the timing reflects the clock the part
    holds under this workload, not its launch boost.  Each trial then times a
    back-to-back loop sized to ~25 ms, so launch overhead is amortised.  When
    a :class:`ClockSampler` is given, the timed trials run inside its window.
    """
    import torch

    call, tensors = _build_gemm(shape, device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def timed_loop(iters: int) -> float:
        start.record()
        for _ in range(iters):
            call()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iters

    call()
    torch.cuda.synchronize(device)
    probe_ms = timed_loop(3)
    iters = int(min(max(_TARGET_TRIAL_MS / max(probe_ms, 1e-4), 3), _MAX_ITERS_PER_TRIAL))
    burst = max(iters // 2, 1)

    def step():
        for _ in range(burst):
            call()
        torch.cuda.synchronize(device)

    settle = run_until_settled(step, reader, "sclk", min_s=min_settle_s)
    with sampler.window() if sampler is not None else contextlib.nullcontext():
        samples = sorted(timed_loop(iters) for _ in range(trials))
    tensors.clear()
    return samples[len(samples) // 2], settle


def _fp8_dtype(device):
    """The fp8 format the part's matrix cores run: fnuz on gfx942, OCP elsewhere."""
    import torch

    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    return torch.float8_e4m3fnuz if arch == "gfx942" else torch.float8_e4m3fn


def _build_gemm(shape: GemmShape, device):
    """``(call, tensors)`` running one ``shape`` GEMM; clearing *tensors* frees them."""
    import torch

    batch = (shape.batch,) if shape.batch > 1 else ()
    if shape.dtype in ("mx8", "mx4"):
        return _build_mx_gemm(shape, device, batch)
    if shape.dtype == "fp8":
        fp8 = _fp8_dtype(device)
        a = torch.randn(*batch, shape.m, shape.k, dtype=torch.bfloat16, device=device).to(fp8)
        # _scaled_mm wants B column-major.
        b = (
            torch.randn(*batch, shape.n, shape.k, dtype=torch.bfloat16, device=device)
            .to(fp8)
            .transpose(-1, -2)
        )
        c = torch.empty(*batch, shape.m, shape.n, dtype=torch.bfloat16, device=device)
        one = torch.ones((), dtype=torch.float32, device=device)
        pairs = [(a[i], b[i], c[i]) for i in range(shape.batch)] if batch else [(a, b, c)]

        def call():
            for ai, bi, ci in pairs:
                torch._scaled_mm(ai, bi, scale_a=one, scale_b=one, out_dtype=torch.bfloat16, out=ci)

        return call, [a, b, c, one, pairs]

    dt = {"fp32": torch.float32, "fp16": torch.float16}.get(shape.dtype, torch.bfloat16)
    a = torch.randn(*batch, shape.m, shape.k, dtype=dt, device=device)
    b = torch.randn(*batch, shape.k, shape.n, dtype=dt, device=device)
    c = torch.empty(*batch, shape.m, shape.n, dtype=dt, device=device)
    mm = torch.bmm if batch else torch.mm

    def call():
        mm(a, b, out=c)

    return call, [a, b, c]


_MX_BLOCK = 32


def _build_mx_gemm(shape: GemmShape, device, batch):
    """MXFP8 (e4m3 + e8m0 scales) or MXFP4 (packed e2m1 + e8m0 scales) via ``_scaled_mm``."""
    import torch

    if shape.k % _MX_BLOCK:
        raise ValueError(f"K={shape.k} is not a multiple of the {_MX_BLOCK}-element MX block")

    def operand(rows):
        if shape.dtype == "mx4":
            packed = torch.randint(0, 256, (*batch, rows, shape.k // 2), dtype=torch.uint8, device=device)
            return packed.view(torch.float4_e2m1fn_x2)
        return torch.randn(*batch, rows, shape.k, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)

    def scales(rows):
        return torch.ones(*batch, rows, shape.k // _MX_BLOCK, device=device).to(torch.float8_e8m0fnu)

    a, b = operand(shape.m), operand(shape.n).transpose(-1, -2)
    sa, sb = scales(shape.m), scales(shape.n)
    c = torch.empty(*batch, shape.m, shape.n, dtype=torch.bfloat16, device=device)
    if batch:
        quads = [(a[i], b[i], sa[i], sb[i], c[i]) for i in range(shape.batch)]
    else:
        quads = [(a, b, sa, sb, c)]

    def call():
        for ai, bi, sai, sbi, ci in quads:
            torch._scaled_mm(ai, bi, scale_a=sai, scale_b=sbi, out_dtype=torch.bfloat16, out=ci)

    return call, [a, b, sa, sb, c, quads]


def _matrix_rates(gcn_arch: str, rates: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """FLOPs per CU per clock by dtype: *rates* (e.g. from the GEMM backend), else the built-in table."""
    return dict(rates) if rates else _MFMA_FLOPS_PER_CU_PER_CLK.get(gcn_arch, {})


def mfma_bound_clock_mhz(
    flops: float,
    time_ms: float,
    gcn_arch: str,
    num_cus: int,
    dtype: str = "bf16",
    rates: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Lowest clock (MHz) that sustains ``flops / time_ms`` of *dtype* at 100 % MFMA use."""
    per_clk = _matrix_rates(gcn_arch, rates).get(dtype)
    if per_clk is None or time_ms <= 0 or num_cus <= 0:
        return None
    flops_per_s = flops / (time_ms / 1e3)
    return flops_per_s / (num_cus * per_clk) / 1e6


def make_origami_predictor(profile_key: str, num_cus: int) -> Optional[Callable[[float, GemmShape], float]]:
    """Return ``predict(clock_mhz, shape) -> ms`` backed by Origami, or *None*.

    *None* when Origami is not installed or does not know *profile_key*.
    """
    from primus.core.projection.simulation_backends.origami_backend import (
        _KNOWN_PROFILES,
        OrigamiGEMMBackend,
    )

    if profile_key not in _KNOWN_PROFILES or not OrigamiGEMMBackend().is_available():
        return None

    backends: Dict[int, OrigamiGEMMBackend] = {}

    def predict(clock_mhz: float, shape: GemmShape) -> float:
        mhz = int(round(clock_mhz))
        with contextlib.redirect_stdout(io.StringIO()):
            backend = backends.get(mhz)
            if backend is None:
                backend = OrigamiGEMMBackend(gpu_arch=profile_key, gpu_clock_mhz=mhz, n_cu_override=num_cus)
                backends[mhz] = backend
            return backend.simulate_gemm(
                shape.m, shape.n, shape.k, shape.dtype, batch=shape.batch
            ).forward_time_ms

    return predict


def make_backend_predictor(backend_name: str, profile_key: str, num_cus: int):
    """``(predict, backend)`` from the registered GEMM backend *backend_name*, else ``(None, None)``.

    ``predict(clock_mhz, shape) -> ms`` like :func:`make_origami_predictor`, for
    backends registered through ``register_gemm_backend`` (including
    out-of-tree plugins), whose hardware profile *profile_key* names.
    *backend* is one instance, for its ``hbm_bandwidth_gbps`` and
    ``matrix_flops_per_cu_per_clk()``.  ``(None, None)`` when the backend is
    not registered, not available, or does not know *profile_key*.
    """
    from primus.core.projection.simulation_backends.base import get_gemm_backend_factory

    factory = get_gemm_backend_factory(backend_name)
    if factory is None:
        return None, None

    def build(mhz):
        with contextlib.redirect_stdout(io.StringIO()):
            return factory(gpu_arch=profile_key, gpu_clock_mhz=mhz, n_cu_override=num_cus)

    try:
        probe = build(None)
        if not probe.is_available():
            return None, None
    except Exception:
        return None, None  # e.g. the backend does not know profile_key

    backends: Dict[int, object] = {}

    def predict(clock_mhz: float, shape: GemmShape) -> float:
        mhz = int(round(clock_mhz))
        backend = backends.get(mhz)
        if backend is None:
            backend = backends[mhz] = build(mhz)
        with contextlib.redirect_stdout(io.StringIO()):
            return backend.simulate_gemm(
                shape.m, shape.n, shape.k, shape.dtype, batch=shape.batch
            ).forward_time_ms

    return predict, probe


def solve_clock_mhz(
    predict_ms: Callable[[float], float],
    measured_ms: float,
    lo_mhz: float = CLOCK_SEARCH_MIN_MHZ,
    hi_mhz: float = CLOCK_SEARCH_MAX_MHZ,
    tol_mhz: float = 1.0,
) -> Tuple[float, bool]:
    """Clock at which ``predict_ms(clock) == measured_ms``; ``(mhz, clamped)``.

    *predict_ms* must decrease with clock.  When the measurement falls outside
    what the model can produce in ``[lo_mhz, hi_mhz]`` the nearest bound is
    returned with ``clamped=True``.
    """
    if measured_ms <= predict_ms(hi_mhz):
        return float(hi_mhz), True
    if measured_ms >= predict_ms(lo_mhz):
        return float(lo_mhz), True
    lo, hi = float(lo_mhz), float(hi_mhz)
    while hi - lo > tol_mhz:
        mid = 0.5 * (lo + hi)
        if predict_ms(mid) > measured_ms:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), False


def _aggregate_mfma_bound_mhz(measurements, total_ms, gcn_arch, num_cus, rates=None) -> Optional[float]:
    """MFMA bound for all GEMMs together: total MFMA cycles needed / total time."""
    rates = _matrix_rates(gcn_arch, rates)
    if total_ms <= 0 or num_cus <= 0 or any(shape.dtype not in rates for shape, _, _ in measurements):
        return None
    cycles = sum(count * shape.flops / (num_cus * rates[shape.dtype]) for shape, count, _ in measurements)
    return cycles / (total_ms / 1e3) / 1e6


def back_calculate_compute_clock(
    measurements: Sequence[Tuple[GemmShape, int, float]],
    gcn_arch: str,
    num_cus: int,
    predictor: Optional[Callable[[float, GemmShape], float]],
    live_sclk_mhz: Optional[float] = None,
    matrix_rates: Optional[Dict[str, float]] = None,
    model: str = "origami",
) -> Dict:
    """Effective clock from ``[(shape, occurrences, measured_ms), ...]``.

    The aggregate solves one clock for the summed time of all GEMMs Origami
    models (:data:`ORIGAMI_MODELLED_DTYPES`), each weighted by its
    occurrences in the workload, so the GEMMs that dominate the workload
    dominate the answer.  Per-GEMM clocks show the spread.  When no GEMM can
    be inverted -- no Origami, or only natively timed MX GEMMs -- the live
    SCLK sampled during the GEMMs is used, else the MFMA-peak lower bound.
    """
    modelled = [m for m in measurements if m[0].dtype in ORIGAMI_MODELLED_DTYPES]
    if predictor is None:
        modelled = []
    rows: List[Dict] = []
    for shape, count, measured_ms in measurements:
        row = {
            **shape.to_dict(),
            "occurrences": count,
            "measured_ms": measured_ms,
            "achieved_tflops": shape.flops / (measured_ms / 1e3) / 1e12,
            "mfma_bound_clock_mhz": mfma_bound_clock_mhz(
                shape.flops, measured_ms, gcn_arch, num_cus, shape.dtype, matrix_rates
            ),
            "origami_clock_mhz": None,
            "origami_clamped": None,
        }
        if predictor is not None and shape.dtype in ORIGAMI_MODELLED_DTYPES:
            mhz, clamped = solve_clock_mhz(lambda f, s=shape: predictor(f, s), measured_ms)
            row["origami_clock_mhz"] = mhz
            row["origami_clamped"] = clamped
        rows.append(row)

    total_ms = sum(count * ms for _, count, ms in measurements)
    result = {
        "gemms": rows,
        "mfma_bound_clock_mhz": _aggregate_mfma_bound_mhz(
            measurements, total_ms, gcn_arch, num_cus, matrix_rates
        ),
        "effective_clock_mhz": None,
        "clamped": None,
        "method": None,
        "model": model if predictor is not None else None,
    }
    if modelled:
        mhz, clamped = solve_clock_mhz(
            lambda f: sum(count * predictor(f, shape) for shape, count, _ in modelled),
            sum(count * ms for _, count, ms in modelled),
        )
        method = "origami-inversion" if model == "origami" else "model-inversion"
        result.update(effective_clock_mhz=mhz, clamped=clamped, method=method)
    elif live_sclk_mhz:
        result.update(effective_clock_mhz=live_sclk_mhz, method="live-sclk")
    elif result["mfma_bound_clock_mhz"] is not None:
        result.update(effective_clock_mhz=result["mfma_bound_clock_mhz"], method="mfma-peak-lower-bound")
    return result
