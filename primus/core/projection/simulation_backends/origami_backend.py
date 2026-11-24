###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Origami GEMM simulation backend.

Origami is an open-source analytical performance model for GEMM kernels on
AMD GPUs (part of the ROCm ecosystem).  It predicts kernel execution time
based on matrix dimensions, data type, tile configuration, and hardware
characteristics.

This is the **default** backend for GEMM simulation in Primus performance
projection.

Installation:
    pip install git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami/python

Environment variables:
    PRIMUS_GEMM_BACKEND  – set to "origami" (or leave unset) to use this backend.
    PRIMUS_GPU_ARCH      – GPU architecture override (e.g. "gfx942", "gfx950").
    PRIMUS_GPU_DEVICE    – GPU device index for hardware detection (default: 0).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from primus.core.projection.simulation_backends.base import (
    GEMMSimulationBackend,
    SimulationResult,
    hbm_bandwidth_override_gbps,
    register_gemm_backend,
)

# ---------------------------------------------------------------------------
# Lazy import – we don't want to fail at module-import time.
# ---------------------------------------------------------------------------
_origami = None
_origami_available: Optional[bool] = None


def _try_import_origami():
    """Try to import origami and cache the result."""
    global _origami, _origami_available
    if _origami_available is not None:
        return _origami_available

    try:
        import origami  # type: ignore[import-untyped]

        _origami = origami
        _origami_available = True
    except ImportError:
        _origami = None
        _origami_available = False

    return _origami_available


# Origami API-capability cache, populated lazily on first use.  Different
# origami builds expose different signatures; we probe once and remember the
# result so the public tree runs against either build without depending on a
# newer origami.  Keys: ``"hw_arch_rf"`` -> whether ``get_hardware_for_arch``
# accepts a register-file (``rf_capacity``) argument.
_ORIGAMI_API_CAPS: Dict[str, bool] = {}


def _get_hardware_for_arch(arch_enum, n_cu, lds_capacity, rf_capacity, l2_capacity, clock_khz):
    """Call ``origami.get_hardware_for_arch`` tolerating builds without rf_capacity.

    Newer origami builds accept a register-file capacity argument (inserted
    between ``lds_capacity`` and ``L2_capacity``); older/public builds expose
    only the 5-argument signature ``(arch, N_CU, lds_capacity, L2_capacity,
    compute_clock_khz)``.
    """
    if _ORIGAMI_API_CAPS.get("hw_arch_rf", True):
        try:
            hw = _origami.get_hardware_for_arch(
                arch_enum, n_cu, lds_capacity, rf_capacity, l2_capacity, clock_khz
            )
            _ORIGAMI_API_CAPS["hw_arch_rf"] = True
            return hw
        except TypeError:
            if _ORIGAMI_API_CAPS.get("hw_arch_rf") is True:
                raise
            _ORIGAMI_API_CAPS["hw_arch_rf"] = False
    return _origami.get_hardware_for_arch(arch_enum, n_cu, lds_capacity, l2_capacity, clock_khz)


# ---------------------------------------------------------------------------
# Known hardware profiles for GPU-less simulation via get_hardware_for_arch.
# ---------------------------------------------------------------------------
@dataclass
class _HardwareProfile:
    """Parameters required by ``origami.get_hardware_for_arch``."""

    arch_enum_name: str  # attribute name on ``origami.architecture_t``
    n_cu: int
    lds_capacity: int  # bytes
    l2_capacity: int  # bytes (per XCD)
    compute_clock_khz: int
    hbm_bandwidth_gbps: float = 5300.0  # peak HBM bandwidth (GB/s)
    rf_capacity: int = 512 * 1024  # Register File (VGPR) capacity per CU in bytes
    # Peak fp32 vector FLOPs per CU per clock (packed FP32), matching the
    # published peaks: MI300X 304 CU x 2.1 GHz x 256 = 163.4 TF,
    # MI355X 256 CU x 2.4 GHz x 256 = 157.3 TF.
    fp32_vector_flops_per_cu_clk: int = 256


_KNOWN_PROFILES: Dict[str, _HardwareProfile] = {
    # MI300X / gfx942: HBM3 ~5.3 TB/s
    "mi300x": _HardwareProfile("gfx942", 304, 65536, 4_194_304, 2_100_000, 5300.0),
    "gfx942": _HardwareProfile("gfx942", 304, 65536, 4_194_304, 2_100_000, 5300.0),
    # MI325X / gfx942 (same die as MI300X, HBM3E upgrade): ~6.0 TB/s
    "mi325x": _HardwareProfile("gfx942", 304, 65536, 4_194_304, 2_100_000, 6000.0),
    # MI350X / MI355X / gfx950: same CDNA4 die (256 CU), HBM3E ~8.0 TB/s.
    # MI350X (air-cooled) and MI355X (liquid-cooled) share the compute profile;
    # without the mi350x key origami falls back to local-device detection and
    # ignores n_cu_override, which breaks the FAv3 1-CU tile model.
    "mi350x": _HardwareProfile("gfx950", 256, 65536, 4_194_304, 2_100_000, 8000.0),
    "mi355x": _HardwareProfile("gfx950", 256, 65536, 4_194_304, 2_100_000, 8000.0),
    "gfx950": _HardwareProfile("gfx950", 256, 65536, 4_194_304, 2_100_000, 8000.0),
    # MI300A
    "mi300a": _HardwareProfile("gfx942", 228, 65536, 4_194_304, 2_100_000, 4000.0),
}

# Dense matrix peaks (TFLOP/s) per datatype.  A GEMM runs at the clock where
# the profile's CUs reach its datatype's peak; a datatype missing here runs as
# its ``_DTYPE_FALLBACK``.  Architectures without an entry keep the profile's
# ``compute_clock_khz`` for every datatype.
_PEAK_TFLOPS: Dict[str, Dict[str, float]] = {
    "mi300x": {"bf16": 579, "fp16": 579, "fp8": 1253, "fp4": 1253, "fp32": 110},
    "mi325x": {"bf16": 827, "fp16": 827, "fp8": 1630},
    "mi350x": {"bf16": 1445, "fp16": 1345, "fp8": 3028, "fp6": 3630, "mxfp4": 4630, "fp32": 130},
    "mi355x": {
        "bf16": 1725,
        "fp8": 3500,
        "mxfp8": 2920,
        "fp6": 4085,
        "mxfp6": 4570,
        "fp4": 5155,
        "mxfp4": 5660,
        "fp32": 137,
    },
}
_PEAK_TFLOPS["gfx942"] = _PEAK_TFLOPS["mi300x"]
_PEAK_TFLOPS["gfx950"] = _PEAK_TFLOPS["mi355x"]

# Dense matrix FLOPs per CU per clock, by Origami arch.
_FLOPS_PER_CU_CLOCK: Dict[str, Dict[str, int]] = {
    "gfx942": {"bf16": 2048, "fp16": 2048, "fp8": 4096, "fp32": 256},
    "gfx950": {"bf16": 4096, "fp16": 4096, "fp8": 8192, "fp32": 256},
}

# Primus dtype names → ``_PEAK_TFLOPS`` keys, and what a datatype runs as where
# the arch has no peak for it (sub-byte types upcast to BF16 on gfx942).
_PEAK_KEY: Dict[str, str] = {"mx8": "mxfp8", "mx6": "mxfp6", "mx4": "mxfp4"}
_DTYPE_FALLBACK: Dict[str, str] = {
    "fp16": "bf16",
    "mxfp8": "fp8",
    "mxfp6": "fp6",
    "fp6": "bf16",
    "fp4": "bf16",
    "mxfp4": "bf16",
}

# ---------------------------------------------------------------------------
# Dtype mapping:  Primus string  →  origami datatype string
# (origami.string_to_datatype accepts these short-hand names)
# ---------------------------------------------------------------------------
_DTYPE_MAP: Dict[str, str] = {
    "bf16": "bf16",
    "fp16": "f16",
    "fp32": "f32",
}
# Origami has no microscaled / sub-8-bit path; these run on the FP8 matrix
# instruction (see ``_resolve_fp8_mi``) at the clock that reaches their peak.
_FP8_CLASS = ("fp8", "mx8", "mx6", "mx4", "fp6", "fp4")

_MAX_STATIC_TILE_K = 64


def _profile_arch(gpu_arch: Optional[str]) -> str:
    return (gpu_arch or os.getenv("PRIMUS_GPU_ARCH", "mi300x")).lower().strip()


def resolve_clock_mhz(gpu_clock_mhz: Optional[int] = None) -> Optional[int]:
    """Explicit compute clock: ``gpu_clock_mhz``, else ``PRIMUS_GPU_CLOCK_MHZ``, else None."""
    if gpu_clock_mhz:
        return int(gpu_clock_mhz)
    env = os.getenv("PRIMUS_GPU_CLOCK_MHZ")
    return int(env) if env else None


def _peak_dtype(arch: str, dtype: str) -> str:
    """The ``_PEAK_TFLOPS`` entry a ``dtype`` matrix kernel runs as on ``arch``."""
    peaks = _PEAK_TFLOPS[arch]
    key = str(dtype or "bf16").lower().strip()
    key = _PEAK_KEY.get(key, key)
    while key not in peaks and key in _DTYPE_FALLBACK:
        key = _DTYPE_FALLBACK[key]
    return key if key in peaks else "bf16"


def _mi_dtype(dtype: str) -> str:
    """The matrix instruction Origami models a ``dtype`` GEMM with."""
    return "fp8" if dtype in _FP8_CLASS else (dtype if dtype in _DTYPE_MAP else "bf16")


def dtype_peak_tflops(
    gpu_arch: Optional[str] = None, dtype: str = "bf16", gpu_clock_mhz: Optional[int] = None
) -> Optional[float]:
    """Dense matrix peak TFLOP/s of a ``dtype`` kernel on ``gpu_arch``, or None without a table.

    The arch's ``_PEAK_TFLOPS`` entry, or at an explicit clock CUs x FLOPs/CU/clock x clock.
    """
    arch = _profile_arch(gpu_arch)
    if arch not in _PEAK_TFLOPS:
        return None
    key = _peak_dtype(arch, dtype)
    clock = resolve_clock_mhz(gpu_clock_mhz)
    if clock is None:
        return float(_PEAK_TFLOPS[arch][key])
    profile = _KNOWN_PROFILES[arch]
    per_cu = _FLOPS_PER_CU_CLOCK[profile.arch_enum_name].get(key)
    if per_cu is None:
        per_cu = _FLOPS_PER_CU_CLOCK[profile.arch_enum_name]["fp8"]
    return profile.n_cu * per_cu * clock / 1e6


def dtype_clock_mhz(
    gpu_arch: Optional[str] = None,
    dtype: str = "bf16",
    gpu_clock_mhz: Optional[int] = None,
    mi_dtype: Optional[str] = None,
) -> Optional[float]:
    """The compute clock a ``dtype`` GEMM is simulated at on ``gpu_arch``.

    An explicit clock applies to every datatype.  Otherwise it is the one at which
    the profile's CUs, issuing the matrix instruction Origami models the dtype
    with, reach the dtype's peak.  None for an arch without a peak table.
    """
    clock = resolve_clock_mhz(gpu_clock_mhz)
    if clock is not None:
        return float(clock)
    arch = _profile_arch(gpu_arch)
    if arch not in _PEAK_TFLOPS:
        return None
    profile = _KNOWN_PROFILES[arch]
    per_cu = _FLOPS_PER_CU_CLOCK[profile.arch_enum_name][mi_dtype or _mi_dtype(dtype)]
    return _PEAK_TFLOPS[arch][_peak_dtype(arch, dtype)] * 1e6 / (profile.n_cu * per_cu)


def _candidate_tile_sizes(mi_k: int) -> List[Tuple[int, int, int]]:
    """Macro-tile candidates, widened so at least one tile can feed the MI.

    Origami only ranks the tiles it is given; if every ``block_K`` is below the
    instruction's K (FP8 on gfx950 is 16x16x128) the instruction is under-fed
    and FP8 shows no gain over BF16.
    """
    if not mi_k or mi_k <= _MAX_STATIC_TILE_K:
        return list(_DEFAULT_TILE_SIZES)
    mn_shapes = sorted({(m, n) for (m, n, _) in _DEFAULT_TILE_SIZES})
    return list(_DEFAULT_TILE_SIZES) + [(m, n, k) for (m, n) in mn_shapes for k in (mi_k, mi_k * 2)]


# ---------------------------------------------------------------------------
# Default candidate tile configurations (macro-tile M×N×K + occupancy).
# A wider search space yields better latency predictions at the cost of
# slightly longer selection time (still << 1 ms per GEMM).
# ---------------------------------------------------------------------------
_DEFAULT_TILE_SIZES: List[Tuple[int, int, int]] = [
    (64, 64, 32),
    (64, 64, 64),
    (64, 128, 32),
    (64, 128, 64),
    (128, 64, 32),
    (128, 64, 64),
    (128, 128, 32),
    (128, 128, 64),
    (128, 256, 32),
    (128, 256, 64),
    (256, 128, 32),
    (256, 128, 64),
    (256, 256, 32),
    (256, 256, 64),
]
_DEFAULT_OCCUPANCIES: List[int] = [1, 2, 4]


class OrigamiGEMMBackend(GEMMSimulationBackend):
    """
    GEMM simulation backend using Origami (open-source).

    Hardware is obtained in one of two ways (in priority order):

    1. **From the local GPU** – ``origami.get_hardware_for_device(device_idx)``
       is called when a ROCm-capable GPU is present.  The device index defaults
       to 0 and can be overridden via the ``PRIMUS_GPU_DEVICE`` env var.
    2. **From a known profile** – when no GPU is available *and* a
       ``--gpu-arch`` / ``PRIMUS_GPU_ARCH`` is provided, the backend falls back
       to ``origami.get_hardware_for_arch`` with hard-coded parameters for
       MI300X, MI355X, etc.
    """

    def __init__(
        self,
        gpu_arch: Optional[str] = None,
        gpu_clock_mhz: Optional[int] = None,
        n_cu_override: Optional[int] = None,
    ):
        """
        Args:
            gpu_arch: Target GPU architecture string (e.g. "gfx942", "mi300x").
                      If *None*, auto-detected from the current GPU or the
                      ``PRIMUS_GPU_ARCH`` env var.
            gpu_clock_mhz: Override the compute clock frequency in MHz.
                      If *None*, uses the profile default or the
                      ``PRIMUS_GPU_CLOCK_MHZ`` env var.
            n_cu_override: Override the number of Compute Units used by
                      Origami's performance model.  Set to ``1`` for
                      per-tile / single-CU simulation (e.g. SDPA tile-level
                      modelling).  If *None*, the profile's default CU
                      count is used.
        """
        self._gpu_arch = gpu_arch or os.getenv("PRIMUS_GPU_ARCH", None)
        if self._gpu_arch is not None:
            self._gpu_arch = self._gpu_arch.lower().strip()

        # Clock override: CLI > env var > profile default
        _env_clock = os.getenv("PRIMUS_GPU_CLOCK_MHZ", None)
        self._clock_override_mhz: Optional[int] = gpu_clock_mhz or (int(_env_clock) if _env_clock else None)

        self._n_cu_override = n_cu_override

        # Lazily initialised origami objects – see ``_ensure_initialized``.
        self._hardware = None  # origami.hardware_t
        self._hardware_by_khz: Dict[int, object] = {}  # profile hardware per compute clock
        self._configs = None  # list[origami.config_t]
        self._clock_ghz: Optional[float] = None
        self._initialized = False
        self._init_dtype: Optional[str] = None  # tracks dtype used for config init
        self._fp8_mi_unavailable = False  # set True if no FP8 / INT8 MI exists
        self._fp8_origami_str: Optional[str] = None  # resolved FP8 MI datatype

    # ------------------------------------------------------------------
    # GEMMSimulationBackend interface
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "origami"

    def is_available(self) -> bool:
        return _try_import_origami()

    @property
    def hbm_bandwidth_gbps(self) -> Optional[float]:
        """Peak HBM bandwidth for the target architecture (GB/s).

        A process-wide override (``--hbm-bandwidth-gbps``, the benchmark-mode
        STREAM calibration, or ``PRIMUS_HBM_BANDWIDTH_GBPS``) wins; otherwise
        resolved from the arch profile (``_KNOWN_PROFILES``) or the
        ``PRIMUS_GPU_ARCH`` env var.  Returns ``None`` only when no
        architecture could be determined.
        """
        override = hbm_bandwidth_override_gbps()
        if override is not None:
            return override
        arch = self._gpu_arch or os.getenv("PRIMUS_GPU_ARCH", "mi300x")
        arch = arch.lower().strip()
        profile = _KNOWN_PROFILES.get(arch)
        return profile.hbm_bandwidth_gbps if profile is not None else None

    def vector_flops(self) -> Optional[float]:
        """Full-chip peak fp32 vector FLOP/s from the arch profile, or *None*.

        ``n_cu x clock x fp32_vector_flops_per_cu_clk``, always for the full chip
        (independent of ``n_cu_override``).  Honours the clock override.
        """
        arch = (self._gpu_arch or os.getenv("PRIMUS_GPU_ARCH", "mi300x")).lower().strip()
        profile = _KNOWN_PROFILES.get(arch)
        if profile is None:
            return None
        clock_khz = (
            self._clock_override_mhz * 1000
            if self._clock_override_mhz is not None
            else profile.compute_clock_khz
        )
        return float(profile.n_cu) * float(clock_khz) * 1e3 * float(profile.fp32_vector_flops_per_cu_clk)

    def peak_flops(self, dtype: str = "bf16") -> Optional[float]:
        """Dense matrix peak FLOP/s of a ``dtype`` kernel on the target, or None without a table."""
        tflops = dtype_peak_tflops(self._gpu_arch, dtype, self._clock_override_mhz)
        return tflops * 1e12 if tflops else None

    def simulate_gemm(
        self,
        m: int,
        n: int,
        k: int,
        dtype: str = "bf16",
        trans_a: bool = False,
        trans_b: bool = False,
        batch: int = 1,
    ) -> SimulationResult:
        if not self.is_available():
            raise RuntimeError(
                "Origami is not installed.  Install with:\n"
                "  pip install git+https://github.com/ROCm/rocm-libraries.git"
                "#subdirectory=shared/origami/python"
            )

        # FP8-class dtypes run on a native FP8 matrix instruction: gfx950 has
        # f8 (16x16x128, fed by the K>=128 tiles of ``_candidate_tile_sizes``);
        # gfx942's FP8 shares the INT8 MFMA unit (16x16x32, 2x BF16), which
        # this Origami build only exposes as i8.  Without either, they run on
        # the BF16 instruction at the clock that reaches their peak.
        fp8_fallback = False
        sim_dtype = "fp8" if dtype in _FP8_CLASS else dtype
        if sim_dtype == "fp8":
            self._ensure_initialized("fp8")
            if self._fp8_mi_unavailable:
                sim_dtype = "bf16"
                fp8_fallback = True
                self._ensure_initialized("bf16")
        else:
            self._ensure_initialized(dtype)
        hardware, clock_ghz = self._hardware_at(dtype, "bf16" if fp8_fallback else None)

        # ----- Build origami problem_t -----
        # NOTE: problem.batch models **batched** GEMM (all sub-problems share
        # the same M, N, K).  For MoE experts this is used as an approximation
        # of grouped GEMM under uniform token distribution.  Origami does not
        # currently expose a native grouped-GEMM model.
        problem = _origami.problem_t()
        problem.size = _origami.dim3_t(m, n, k)
        problem.batch = batch
        problem.a_transpose = _origami.transpose_t.T if trans_a else _origami.transpose_t.N
        problem.b_transpose = _origami.transpose_t.T if trans_b else _origami.transpose_t.N

        if sim_dtype == "fp8":
            origami_dtype = _origami.string_to_datatype(self._fp8_origami_str)
        else:
            origami_dtype = _origami.string_to_datatype(_DTYPE_MAP.get(sim_dtype, "bf16"))
        problem.a_dtype = origami_dtype
        problem.b_dtype = origami_dtype
        problem.c_dtype = origami_dtype
        problem.d_dtype = origami_dtype
        problem.mi_dtype = origami_dtype
        problem.a_mx_block_size = 0
        problem.b_mx_block_size = 0

        # ----- Select best config & predict latency (in clock cycles) -----
        # Newer origami builds take a ``model_t`` selector as a 4th argument;
        # the public/CI build exposes only the 3-argument signature.  Dispatch
        # on availability so the public tree runs against either build.
        try:
            if hasattr(_origami, "model_t"):
                result = _origami.select_config(problem, hardware, self._configs, _origami.model_t.gemm)
            else:
                result = _origami.select_config(problem, hardware, self._configs)
        except Exception as e:
            raise RuntimeError(
                f"Origami select_config failed for " f"(M={m}, N={n}, K={k}, dtype={dtype}): {e}"
            ) from e

        latency_cycles = result.latency
        time_ms = latency_cycles / (clock_ghz * 1e6)

        # Compute achieved TFLOPS for metadata
        flops = 2.0 * m * n * k * batch
        time_s = time_ms / 1e3
        tflops = (flops / time_s / 1e12) if time_s > 0 else 0.0

        return SimulationResult(
            forward_time_ms=time_ms,
            backward_time_ms=0.0,  # Caller computes bwd from fwd
            tflops=tflops,
            metadata={
                "backend": "origami",
                "gpu_arch": self._gpu_arch,
                "latency_cycles": latency_cycles,
                "clock_ghz": clock_ghz,
                "dtype": dtype,
                "batch": batch,
                "fp8_fallback": fp8_fallback,
                "best_tile": (
                    result.config.mt.m,
                    result.config.mt.n,
                    result.config.mt.k,
                ),
                "best_occupancy": result.config.occupancy,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_initialized(self, dtype: str = "bf16") -> None:
        """Lazily initialise hardware and candidate configs.

        If called again with a *different* dtype the candidate config list is
        rebuilt (the matrix-instruction size changes between BF16 and FP8).
        """
        # Hardware only needs to be detected once.
        if self._hardware is None:
            self._hardware = self._get_hardware()
            self._clock_ghz = self._hardware.compute_clock_ghz

        # (Re-)build candidate configs when the dtype changes.
        if self._initialized and self._init_dtype == dtype:
            return

        if dtype == "fp8":
            resolved = self._resolve_fp8_mi()
            if resolved is None:
                self._fp8_mi_unavailable = True
                if int(os.getenv("RANK", "0")) == 0:
                    print(
                        "[Primus:Origami] No FP8 / INT8 matrix instruction for this hardware; "
                        "FP8 runs on the BF16 instruction at the FP8 peak's clock"
                    )
                return
            self._fp8_origami_str, mi = resolved
        else:
            try:
                origami_dtype = _origami.string_to_datatype(_DTYPE_MAP.get(dtype, "bf16"))
            except (ValueError, RuntimeError):
                return  # Origami doesn't know this dtype string
            mi = self._hardware.get_recommended_matrix_instruction(origami_dtype)
            if mi.m == 0 and mi.n == 0 and mi.k == 0:
                return

        configs: list = []
        for mt_m, mt_n, mt_k in _candidate_tile_sizes(mi.k):
            for occ in _DEFAULT_OCCUPANCIES:
                cfg = _origami.config_t()
                cfg.mt = _origami.dim3_t(mt_m, mt_n, mt_k)
                cfg.mi = mi
                cfg.occupancy = occ
                configs.append(cfg)
        self._configs = configs
        self._init_dtype = dtype

        is_rank_0 = int(os.getenv("RANK", "0")) == 0
        if is_rank_0:
            print(
                f"[Primus:Origami] Initialised: "
                f"N_CU={self._hardware.N_CU}, "
                f"NUM_XCD={self._hardware.NUM_XCD}, "
                f"clock={self._clock_ghz} GHz, "
                f"MI={mi.m}x{mi.n}x{mi.k}, "
                f"dtype={dtype}, "
                f"{len(configs)} candidate configs"
            )

        self._initialized = True

    def _resolve_fp8_mi(self):
        """``(origami datatype string, MI)`` of the first FP8-class instruction the arch has, or None.

        Native OCP FP8 (``f8`` / ``bf8``) first; gfx942's FP8 shares the INT8
        MFMA unit (same shape and rate), so ``i8`` is an exact throughput proxy.
        """
        for cand in ("f8", "bf8", "i8"):
            try:
                dt = _origami.string_to_datatype(cand)
            except (ValueError, RuntimeError):
                continue
            mi = self._hardware.get_recommended_matrix_instruction(dt)
            if not (mi.m == 0 and mi.n == 0 and mi.k == 0):
                return cand, mi
        return None

    def _hardware_at(self, dtype: str, mi_dtype: Optional[str] = None):
        """``(hardware, clock GHz)`` for a ``dtype`` GEMM: the profile at that dtype's clock.

        Detected devices and arches without a peak table keep one clock.
        """
        if self._gpu_arch not in _KNOWN_PROFILES:
            return self._hardware, self._clock_ghz
        mhz = dtype_clock_mhz(self._gpu_arch, dtype, self._clock_override_mhz, mi_dtype)
        if mhz is None:
            return self._hardware, self._clock_ghz
        khz = round(mhz * 1000)
        if khz not in self._hardware_by_khz:
            self._hardware_by_khz[khz] = self._profile_hardware(khz)
        return self._hardware_by_khz[khz], khz / 1e6

    def _profile_hardware(self, clock_khz: int):
        profile = _KNOWN_PROFILES[self._gpu_arch]
        n_cu = self._n_cu_override if self._n_cu_override is not None else profile.n_cu
        return _get_hardware_for_arch(
            getattr(_origami.architecture_t, profile.arch_enum_name),
            n_cu,
            profile.lds_capacity,
            profile.rf_capacity,
            profile.l2_capacity,
            clock_khz,
        )

    def _get_hardware(self):
        """
        Obtain an ``origami.hardware_t`` instance.

        Priority:
        1. If ``--gpu-arch`` was explicitly provided AND we have a known
           profile for it, use the profile.  This ensures consistent
           results regardless of the local GPU (important for simulation
           mode targeting a *different* GPU, e.g. simulating MI325X on
           MI300X).
        2. Otherwise, try the local GPU.
        3. Fall back to the arch profile if no GPU is available.
        """
        is_rank_0 = int(os.getenv("RANK", "0")) == 0

        # 1. If an explicit arch was requested AND we have a profile, use it.
        if self._gpu_arch is not None and self._gpu_arch in _KNOWN_PROFILES:
            profile = _KNOWN_PROFILES[self._gpu_arch]
            bf16_mhz = dtype_clock_mhz(self._gpu_arch, "bf16", self._clock_override_mhz)
            clock_khz = round(bf16_mhz * 1000) if bf16_mhz is not None else profile.compute_clock_khz
            n_cu = self._n_cu_override if self._n_cu_override is not None else profile.n_cu
            hw = self._profile_hardware(clock_khz)
            if is_rank_0:
                override_tag = ""
                if self._clock_override_mhz is not None:
                    override_tag = " (overridden via --gpu-clock-mhz, all dtypes)"
                elif bf16_mhz is not None:
                    override_tag = " (BF16; per-dtype defaults)"
                cu_tag = f" (n_cu_override={n_cu})" if self._n_cu_override is not None else ""
                print(
                    f"[Primus:Origami] Using hardware profile for "
                    f"'{self._gpu_arch}': N_CU={n_cu}, "
                    f"clock={clock_khz / 1e6:.3f} GHz{override_tag}{cu_tag}"
                )
            return hw

        # 2. Try local GPU
        device_idx = int(os.getenv("PRIMUS_GPU_DEVICE", "0"))
        try:
            hw = _origami.get_hardware_for_device(device_idx)
            if is_rank_0:
                print(
                    f"[Primus:Origami] Hardware detected from device {device_idx}: "
                    f"N_CU={hw.N_CU}, NUM_XCD={hw.NUM_XCD}, "
                    f"clock={hw.compute_clock_ghz} GHz"
                )
            return hw
        except Exception:
            pass  # No GPU – try arch-based profile

        # 3. Fall back to known profile
        if self._gpu_arch is None:
            raise RuntimeError(
                "Origami could not detect a GPU and no --gpu-arch / "
                "PRIMUS_GPU_ARCH was specified.  Either run on a machine with "
                "a ROCm GPU or provide a target architecture "
                "(e.g. --gpu-arch mi300x)."
            )

        profile = _KNOWN_PROFILES.get(self._gpu_arch)
        if profile is None:
            supported = ", ".join(sorted(_KNOWN_PROFILES.keys()))
            raise RuntimeError(
                f"Unknown GPU architecture '{self._gpu_arch}' for origami. "
                f"Supported architectures: {supported}"
            )

        clock_khz = profile.compute_clock_khz
        if self._clock_override_mhz is not None:
            clock_khz = self._clock_override_mhz * 1000
        n_cu = self._n_cu_override if self._n_cu_override is not None else profile.n_cu
        arch_enum = getattr(_origami.architecture_t, profile.arch_enum_name)
        hw = _get_hardware_for_arch(
            arch_enum,
            n_cu,
            profile.lds_capacity,
            profile.rf_capacity,
            profile.l2_capacity,
            clock_khz,
        )
        if is_rank_0:
            override_tag = ""
            if self._clock_override_mhz is not None:
                override_tag = " (overridden via --gpu-clock-mhz)"
            cu_tag = f" (n_cu_override={n_cu})" if self._n_cu_override is not None else ""
            print(
                f"[Primus:Origami] Using known hardware profile for "
                f"'{self._gpu_arch}': N_CU={n_cu}, "
                f"clock={clock_khz / 1e6:.1f} GHz{override_tag}{cu_tag}"
            )
        return hw


# ---------------------------------------------------------------------------
# Self-registration.  Importing this module registers the built-in ``origami``
# GEMM backend with the shared registry (see ``base.register_gemm_backend``).
# ---------------------------------------------------------------------------
def _make_origami_backend(
    gpu_arch=None,
    gpu_clock_mhz=None,
    n_cu_override=None,
    **_kwargs,
) -> OrigamiGEMMBackend:
    return OrigamiGEMMBackend(
        gpu_arch=gpu_arch,
        gpu_clock_mhz=gpu_clock_mhz,
        n_cu_override=n_cu_override,
    )


register_gemm_backend("origami", _make_origami_backend)
