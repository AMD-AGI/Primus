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
    # Peak *dense* BF16 matrix throughput (TFLOP/s). FP8 is taken as 2x. Used
    # only for the roofline sanity-cap in ``simulate_gemm`` (see below), not for
    # the primary Origami tile prediction.
    peak_tflops_bf16: float = 1307.0
    # Register-file capacity per CU (bytes). CDNA3/CDNA4 expose 512 KiB of VGPR
    # per CU; ``origami.get_hardware_for_arch`` requires this argument (matches
    # ``get_hardware_for_device(...).rf_capacity`` on gfx942/gfx950).
    rf_capacity: int = 524288


_KNOWN_PROFILES: Dict[str, _HardwareProfile] = {
    # MI300X / gfx942: HBM3 ~5.3 TB/s, ~1307 TFLOP/s dense BF16
    "mi300x": _HardwareProfile("gfx942", 304, 65536, 4_194_304, 2_100_000, 5300.0, 1307.0),
    "gfx942": _HardwareProfile("gfx942", 304, 65536, 4_194_304, 2_100_000, 5300.0, 1307.0),
    # MI325X / gfx942 (same die as MI300X, HBM3E upgrade): ~6.0 TB/s
    "mi325x": _HardwareProfile("gfx942", 304, 65536, 4_194_304, 2_100_000, 6000.0, 1307.0),
    # MI355X / gfx950: HBM3E ~8.0 TB/s, ~2500 TFLOP/s dense BF16 (CDNA4)
    "mi355x": _HardwareProfile("gfx950", 256, 65536, 4_194_304, 2_100_000, 8000.0, 2500.0),
    "gfx950": _HardwareProfile("gfx950", 256, 65536, 4_194_304, 2_100_000, 8000.0, 2500.0),
    # MI300A
    "mi300a": _HardwareProfile("gfx942", 228, 65536, 4_194_304, 2_100_000, 4000.0, 981.0),
}

# Achievable fractions of peak for the roofline sanity-cap. The cap only exists
# to trim Origami's small-M (memory-bound decode) over-prediction — a spurious
# ~1/M inflation from tile quantisation. It must NOT override Origami's
# compute-bound tile model for large-M / prefill GEMMs (which is accurate,
# including the native FP8 speedup once K>=MI.k tiles are available).
#
# ``_ROOFLINE_MEM_EFF`` sets how tightly the memory-bound floor caps small-M.
# ``_ROOFLINE_COMPUTE_EFF`` is deliberately *low* so the compute ceiling stays
# well above Origami's real compute-bound predictions (which reach ~55-70% of
# peak) and never clips them; it only guards against a pathological case where
# even the compute roofline is exceeded.
_ROOFLINE_MEM_EFF = 0.70
_ROOFLINE_COMPUTE_EFF = 0.40

# ---------------------------------------------------------------------------
# Dtype mapping:  Primus string  →  origami datatype string
# (origami.string_to_datatype accepts these short-hand names)
# ---------------------------------------------------------------------------
_DTYPE_MAP: Dict[str, str] = {
    "bf16": "bf16",
    "fp16": "f16",
    "fp32": "f32",
    # gfx950 (CDNA4) exposes a native FP8 matrix instruction under the "f8"
    # datatype (16x16x128). The older "bf8_fnuz" name is not accepted by
    # ``string_to_datatype`` in current Origami builds and raised — silently
    # forcing the BF16 fallback on *every* arch. On archs without an FP8 MI
    # (e.g. gfx942) "f8" resolves to a 0x0x0 instruction, which the caller
    # detects and falls back to BF16 (÷2) as before.
    "fp8": "f8",
}

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

# Widest ``block_K`` in the static list above. When a matrix instruction is
# wider than this in the K dimension, none of the static macro-tiles can fully
# feed it and Origami predicts *no* compute speedup (the MI runs half-empty).
_MAX_STATIC_TILE_K = 64


def _candidate_tile_sizes(mi_k: int) -> List[Tuple[int, int, int]]:
    """Macro-tile candidates, widened so at least one tile can feed the MI.

    Origami only ranks the tiles it is given. If every candidate's ``block_K``
    is smaller than the matrix instruction's K dimension, the instruction is
    under-fed and the selected config shows no throughput gain — this is why
    FP8 on gfx950 (MI = 16x16x128) landed at ~BF16 while FP8 on gfx942
    (MI = 16x16x32, fed fine by a K=64 tile) showed the expected speedup.

    We therefore append macro-tiles with ``block_K`` set to a multiple of the
    instruction's K whenever ``mi_k`` exceeds the widest static tile. Smaller
    instructions (all BF16/FP16 MIs, and gfx942 FP8) keep the exact static list
    so their tuned predictions are unchanged.
    """
    if not mi_k or mi_k <= _MAX_STATIC_TILE_K:
        return list(_DEFAULT_TILE_SIZES)

    mn_shapes = sorted({(m, n) for (m, n, _) in _DEFAULT_TILE_SIZES})
    extra = [(m, n, k) for (m, n) in mn_shapes for k in (mi_k, mi_k * 2)]
    return list(_DEFAULT_TILE_SIZES) + extra


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
        self._configs = None  # list[origami.config_t]
        self._clock_ghz: Optional[float] = None
        self._initialized = False
        self._init_dtype: Optional[str] = None  # tracks dtype used for config init
        self._fp8_origami_str: Optional[str] = None  # resolved FP8 MI dtype str

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

        Resolved from the arch profile (``_KNOWN_PROFILES``) or the
        ``PRIMUS_GPU_ARCH`` env var.  Returns ``None`` only when no
        architecture could be determined.
        """
        arch = self._gpu_arch or os.getenv("PRIMUS_GPU_ARCH", "mi300x")
        arch = arch.lower().strip()
        profile = _KNOWN_PROFILES.get(arch)
        return profile.hbm_bandwidth_gbps if profile is not None else None

    @property
    def peak_tflops_bf16(self) -> float:
        """Peak dense BF16 matrix throughput (TFLOP/s) for the roofline cap."""
        arch = self._gpu_arch or os.getenv("PRIMUS_GPU_ARCH", "mi300x")
        arch = arch.lower().strip()
        profile = _KNOWN_PROFILES.get(arch)
        return profile.peak_tflops_bf16 if profile is not None else 1307.0

    def _roofline_ceiling_ms(self, m: int, n: int, k: int, batch: int, sim_dtype: str, native_fp8: bool) -> float:
        """Efficiency-adjusted max(memory, compute) roofline for one (batched) GEMM.

        This is a physical *upper bound* on a well-tuned kernel's time: it can't
        take longer than streaming its operands at achievable HBM bandwidth, nor
        longer than issuing its FLOPs at achievable matrix throughput. Origami's
        tile model is accurate in the compute-bound (large-M) regime but has a
        known small-M over-prediction (a spurious ~1/M inflation for
        memory-bound decode GEMMs); capping at this ceiling removes that
        artifact while leaving compute-bound GEMMs untouched.
        """
        bw = self.hbm_bandwidth_gbps or _FALLBACK_HBM_BW_GBPS  # GB/s
        # FP8 halves operand bytes and doubles matrix throughput; when FP8 is
        # simulated as a BF16 fallback the peak still reflects FP8 silicon.
        is_fp8 = native_fp8 or sim_dtype != "bf16"
        in_bytes = 1 if is_fp8 else 2
        out_bytes = 2  # outputs kept at BF16 for accumulation fidelity
        peak_tflops = self.peak_tflops_bf16 * (2.0 if is_fp8 else 1.0)

        mem_bytes = (float(m) * k + float(n) * k) * in_bytes + float(m) * n * out_bytes
        mem_bytes *= max(1, batch)
        t_mem_ms = mem_bytes / (bw * 1e9 * _ROOFLINE_MEM_EFF) * 1e3

        flops = 2.0 * m * n * k * max(1, batch)
        t_compute_ms = flops / (peak_tflops * 1e12 * _ROOFLINE_COMPUTE_EFF) * 1e3
        return max(t_mem_ms, t_compute_ms)

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

        # FP8 matrix-instruction strategy (see ``_resolve_fp8_mi``)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # FP8 is always modelled with a *native* matrix instruction:
        #   * gfx950 / MI355X (CDNA4): native OCP FP8 MI = 16x16x128. Requires
        #     the K>=128 candidate tiles (``_candidate_tile_sizes``) to actually
        #     feed the instruction, otherwise it runs half-empty and shows no
        #     speedup.
        #   * gfx942 / MI300X (CDNA3): FP8 shares the INT8 MFMA unit (both
        #     16x16x32, 2x BF16, 1-byte operands). This Origami build doesn't
        #     surface an FP8 datatype for gfx942, so we use the INT8 MI as a
        #     physically-exact throughput proxy → native ~1.8x.
        # ``_ensure_initialized`` raises if an arch has neither, which does not
        # happen for any supported AMD GPU.
        sim_dtype = dtype
        self._ensure_initialized(dtype)

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

        # For FP8 the resolved MI datatype may be a native FP8 type (f8/bf8) or
        # an INT8 proxy on archs that share the MFMA unit (see _resolve_fp8_mi).
        if sim_dtype == "fp8" and self._fp8_origami_str:
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
        # ``select_config`` takes a ``model_t`` selecting the GEMM (vs attention)
        # cost model as its 4th argument.
        try:
            result = _origami.select_config(
                problem, self._hardware, self._configs, _origami.model_t.gemm
            )
        except Exception as e:
            raise RuntimeError(
                f"Origami select_config failed for " f"(M={m}, N={n}, K={k}, dtype={dtype}): {e}"
            ) from e

        latency_cycles = result.latency
        time_ms = latency_cycles / (self._clock_ghz * 1e6)

        # Roofline sanity-cap: remove Origami's small-M (memory-bound decode)
        # over-prediction by capping at an efficiency-adjusted max(mem, compute)
        # roofline. Compute-bound (large-M / prefill) GEMMs sit below the
        # ceiling and are unaffected. dtype=="fp8" always earns the FP8 roofline
        # (2x throughput, 1-byte operands) regardless of the BF16 fallback.
        ceiling_ms = self._roofline_ceiling_ms(m, n, k, batch, sim_dtype, native_fp8=(dtype == "fp8"))
        if ceiling_ms > 0:
            time_ms = min(time_ms, ceiling_ms)

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
                "clock_ghz": self._clock_ghz,
                "dtype": dtype,
                "batch": batch,
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

    def _resolve_fp8_mi(self):
        """Resolve the best FP8 matrix instruction for the target arch.

        Tries the native OCP FP8 datatypes first (``f8``/``bf8`` — gfx950/CDNA4
        expose a 16x16x128 MI). On archs whose FP8 shares the INT8 MFMA unit but
        which this Origami build does not surface under an FP8 datatype string
        (gfx942/CDNA3: FP8 and INT8 are both 16x16x32, 2x BF16, 1-byte operands),
        it falls back to the ``i8`` MI as a physically-exact throughput proxy.

        Returns ``(origami_dtype, origami_str, mi)`` for the first datatype that
        yields a non-zero MI, or ``None`` if the arch has no FP8/INT8 MI at all.
        """
        for cand in ("f8", "bf8", "i8"):
            try:
                dt = _origami.string_to_datatype(cand)
            except (ValueError, RuntimeError):
                continue
            mi = self._hardware.get_recommended_matrix_instruction(dt)
            if not (mi.m == 0 and mi.n == 0 and mi.k == 0):
                return dt, cand, mi
        return None

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
                raise RuntimeError(
                    "[Primus:Origami] No FP8/INT8 matrix instruction available for "
                    f"arch '{self._gpu_arch}'. FP8 projection is unsupported on this "
                    "hardware/Origami build."
                )
            origami_dtype, self._fp8_origami_str, mi = resolved
        else:
            origami_str = _DTYPE_MAP.get(dtype, "bf16")
            try:
                origami_dtype = _origami.string_to_datatype(origami_str)
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
            clock_khz = profile.compute_clock_khz
            if self._clock_override_mhz is not None:
                clock_khz = self._clock_override_mhz * 1000
            n_cu = self._n_cu_override if self._n_cu_override is not None else profile.n_cu
            arch_enum = getattr(_origami.architecture_t, profile.arch_enum_name)
            hw = _origami.get_hardware_for_arch(
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
                    f"[Primus:Origami] Using hardware profile for "
                    f"'{self._gpu_arch}': N_CU={n_cu}, "
                    f"clock={clock_khz / 1e6:.1f} GHz{override_tag}{cu_tag}"
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
        hw = _origami.get_hardware_for_arch(
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
