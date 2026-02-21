###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
SDPA simulation backend modelling the **FAv3** (Flash Attention v3) kernels.

The forward and backward kernel parameters are extracted from FAv3 kernel
configurations:

  Forward:
      Config : BF16 ; FMHA FWD ; D128 ; 1TG ; 8W ; 32m×8 ; 64n×1 ; 32×32×16
      • 1 Thread-Group, 8 Wavefronts per workgroup (512 threads)
      • Q-tile  = 256 rows   (32m × 8 wavefronts)
      • KV-tile = 64 columns  per loop iteration
      • 64 MFMAs per loop iteration  (QKᵀ + softmax + PV, pipelined)
      • Workgroups = ⌈S / 256⌉ × B × H

  Backward:
      Config : BF16 ; FMHA BWD ; D128 ; 1TG ; 4W ; 16m×1 ; 64n×4 ; A32
      • 1 Thread-Group, 4 Wavefronts per workgroup (256 threads)
      • Q-tile  = 16 rows   per inner loop step
      • KV-tile = 256 columns  per workgroup  (64n × 4)
      • 256 MFMAs per inner-loop iteration  (dV, dP, dS, dQ, dK phases)
      • Workgroups = ⌈S / 256⌉ × B × H
      • Inner-loop iterations = ⌈S / 16⌉  (over Q blocks)

Tile-level simulation using Origami with a 1-CU backend.
Flash Attention is a fused kernel where sub-operations (QKᵀ, softmax, PV)
execute **sequentially** within each workgroup tile.  By running Origami on
a single CU for each per-tile GEMM and then multiplying by the number of
waves (workgroups / N_CU), we capture the additive cost structure and
per-tile wave quantisation effects that a global roofline misses.

**Origami is required** — this simulator cannot operate without it.

In the backward pass, the dQ gradient is accumulated across KV-workgroups
using ``buffer_atomic_add_f32`` (72 atomic instructions in the kernel).
Each KV-workgroup processes all Q positions and atomically adds its partial
dQ contribution, leading to contention proportional to ⌈S / 256⌉ concurrent
writers per dQ cache line.  The atomic overhead is modelled as an additive
cost on top of the compute/memory time.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

from primus.core.projection.simulation_backends.base import (
    SDPASimulationBackend,
    SimulationResult,
)

# =========================================================================
# FAv3 kernel tile parameters
# =========================================================================


@dataclass(frozen=True)
class _FAv3TileConfig:
    """Tile & occupancy parameters extracted from a FAv3 kernel."""

    q_tile_m: int  # Q rows per workgroup
    kv_tile_n: int  # K/V positions per loop iteration
    n_wavefronts: int  # Wavefronts per workgroup
    mfma_m: int = 32  # MFMA instruction M
    mfma_n: int = 32  # MFMA instruction N
    mfma_k: int = 16  # MFMA instruction K  (BF16 on gfx950)


# Forward:  256 Q-rows, 64 KV-cols/iter, 8 wavefronts
_FAV3_FWD = _FAv3TileConfig(q_tile_m=256, kv_tile_n=64, n_wavefronts=8)

# Backward: 16 Q-rows/inner-iter, 256 KV-cols/workgroup, 4 wavefronts
_FAV3_BWD = _FAv3TileConfig(q_tile_m=16, kv_tile_n=256, n_wavefronts=4)


# =========================================================================
# Backward dQ atomic latencies (for tile-level model)
# =========================================================================
# dQ is accumulated via buffer_atomic_add_f32 across KV-workgroups.
# Latency estimates for CDNA3 (gfx942) at typical clocks:
_ATOMIC_LATENCY_GLOBAL_NS = 400  # HBM read-modify-write latency per atomic op
_ATOMIC_LATENCY_LOCAL_NS = 40  # L1 / LDS atomic latency per op
_WARP_SIZE = 64  # CDNA wavefront width


# =========================================================================
# GPU hardware specs
# =========================================================================


@dataclass
class GPUHardwareSpec:
    """Hardware specification for roofline modelling."""

    # Peak compute throughput in TFLOPS (tera floating-point ops / sec)
    peak_tflops_bf16: float = 1307.0  # MI300X BF16 peak
    peak_tflops_fp16: float = 1307.0
    peak_tflops_fp8: float = 2614.0

    # HBM bandwidth in GB/s
    hbm_bandwidth_gbps: float = 5300.0  # MI300X HBM3

    # Total CUs on the device
    n_cu: int = 304  # MI300X

    # Number of XCDs on the device (cross-die atomics are more expensive)
    n_xcd: int = 8  # MI300X has 8 XCDs


# Pre-defined hardware profiles
_HW_PROFILES: Dict[str, GPUHardwareSpec] = {
    "mi300x": GPUHardwareSpec(
        peak_tflops_bf16=1307.0,
        peak_tflops_fp16=1307.0,
        peak_tflops_fp8=2614.0,
        hbm_bandwidth_gbps=5300.0,
        n_cu=304,
        n_xcd=8,
    ),
    "gfx942": GPUHardwareSpec(  # same as MI300X
        peak_tflops_bf16=1307.0,
        peak_tflops_fp16=1307.0,
        peak_tflops_fp8=2614.0,
        hbm_bandwidth_gbps=5300.0,
        n_cu=304,
        n_xcd=8,
    ),
    "mi325x": GPUHardwareSpec(  # gfx942 die, HBM3E (use --gpu-clock-mhz to override clock)
        peak_tflops_bf16=1307.0,
        peak_tflops_fp16=1307.0,
        peak_tflops_fp8=2614.0,
        hbm_bandwidth_gbps=6000.0,  # HBM3E ~6 TB/s (vs 5.3 on MI300X)
        n_cu=304,
        n_xcd=8,
    ),
    "mi355x": GPUHardwareSpec(
        peak_tflops_bf16=2384.0,
        peak_tflops_fp16=2384.0,
        peak_tflops_fp8=4768.0,
        hbm_bandwidth_gbps=8000.0,
        n_cu=256,
        n_xcd=4,
    ),
    "gfx950": GPUHardwareSpec(  # same as MI355X
        peak_tflops_bf16=2384.0,
        peak_tflops_fp16=2384.0,
        peak_tflops_fp8=4768.0,
        hbm_bandwidth_gbps=8000.0,
        n_cu=256,
        n_xcd=4,
    ),
}


def _get_hardware_spec(
    gpu_arch: Optional[str] = None,
    gpu_clock_mhz: Optional[int] = None,
) -> GPUHardwareSpec:
    """Get hardware spec for the given (or detected) GPU architecture.

    If *gpu_clock_mhz* is provided, the profile's TFLOPS values are scaled
    proportionally (compute throughput is linear in clock frequency).
    """
    arch = gpu_arch or os.getenv("PRIMUS_GPU_ARCH", "mi300x")
    arch = arch.lower().strip()
    spec = _HW_PROFILES.get(arch, _HW_PROFILES["mi300x"])

    # Apply clock override — scale TFLOPS linearly
    clock_override = gpu_clock_mhz or (
        int(v) if (v := os.getenv("PRIMUS_GPU_CLOCK_MHZ")) else None
    )
    if clock_override is not None:
        # Derive the profile's implicit clock from a known reference.
        _PROFILE_CLOCK_MHZ = {
            "mi300x": 2100,
            "gfx942": 2100,
            "mi325x": 2100,  # same gfx942 compute die as MI300X
            "mi355x": 2100,
            "gfx950": 2100,
            "mi300a": 2100,
        }
        base_clock = _PROFILE_CLOCK_MHZ.get(arch, 2100)
        scale = clock_override / base_clock
        spec = GPUHardwareSpec(
            peak_tflops_bf16=spec.peak_tflops_bf16 * scale,
            peak_tflops_fp16=spec.peak_tflops_fp16 * scale,
            peak_tflops_fp8=spec.peak_tflops_fp8 * scale,
            hbm_bandwidth_gbps=spec.hbm_bandwidth_gbps,  # BW doesn't change with clock
            n_cu=spec.n_cu,
            n_xcd=spec.n_xcd,
        )
    return spec


# =========================================================================
# SDPASimulator — FAv3-based analytical model
# =========================================================================


class SDPASimulator(SDPASimulationBackend):
    """
    Analytical SDPA simulation modelling the FAv3 kernel structure.

    Uses an Origami GEMM backend with ``n_cu=1`` to simulate per-tile
    GEMM execution time.  Flash Attention is modelled as a fused kernel
    where QKᵀ, softmax, and PV are sequential within each workgroup.
    The total time = (per-tile-QKᵀ + per-tile-PV) × num_waves.  This
    naturally captures wave quantisation and per-tile efficiency without
    needing an empirical ``compute_efficiency`` parameter.

    Also models the backward dQ atomic overhead from
    ``buffer_atomic_add_f32`` accumulation across KV-workgroups.

    **Origami is required** — instantiation will fail if the Origami
    backend is not available.
    """

    def __init__(
        self,
        gpu_arch: Optional[str] = None,
        hardware_spec: Optional[GPUHardwareSpec] = None,
        gpu_clock_mhz: Optional[int] = None,
    ):
        """
        Args:
            gpu_arch: GPU architecture string (e.g. "mi300x", "gfx942",
                "mi355x", "gfx950").
            hardware_spec: Override hardware spec directly.
            gpu_clock_mhz: Override the GPU compute clock frequency in MHz.
                If provided, the profile's TFLOPS are scaled proportionally.

        Raises:
            RuntimeError: If the Origami backend is not available.
        """
        self._hw = hardware_spec or _get_hardware_spec(gpu_arch, gpu_clock_mhz)

        # Create the Origami 1-CU backend for tile-level simulation.
        # This is required — SDPA simulation cannot proceed without it.
        self._tile_gemm = self._create_tile_gemm_backend(gpu_arch, gpu_clock_mhz)
        if self._tile_gemm is None:
            raise RuntimeError(
                "SDPASimulator requires the Origami backend but it is not "
                "available.  Please install the 'origami' package or ensure "
                "primus.core.projection.simulation_backends.origami_backend "
                "is importable."
            )

    def name(self) -> str:
        return "sdpa_simulator (FAv3)"

    def is_available(self) -> bool:
        return self._tile_gemm is not None and self._tile_gemm.is_available()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_sdpa(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        causal: bool = True,
        dtype: str = "bf16",
        seq_len_kv: Optional[int] = None,
        num_heads_kv: Optional[int] = None,
        head_dim_v: Optional[int] = None,
    ) -> SimulationResult:
        """
        Simulate FAv3 SDPA execution time using Origami 1-CU tile-level
        simulation parameterised by the actual FAv3 tile configuration.

        Args:
            batch_size: Batch size (B).
            num_heads: Number of query heads (H_Q).
            seq_len: Query sequence length (S_Q).
            head_dim: Head dimension for Q·Kᵀ (D_qk).  For standard
                MHA/GQA this is ``kv_channels``; for MLA this is
                ``qk_head_dim + qk_pos_emb_head_dim`` (e.g. 192).
            causal: Whether causal masking is applied.
            dtype: Data type ("bf16", "fp16", "fp8", "fp32").
            seq_len_kv: Key/Value sequence length (S_K).  Defaults to
                ``seq_len`` (self-attention).  Set differently for
                cross-attention or prefill with separate KV cache length.
            num_heads_kv: Number of KV heads.  Defaults to ``num_heads``
                (MHA).  Set lower for GQA/MQA.
            head_dim_v: Value head dimension for P·V (D_v).  Defaults to
                ``head_dim`` (standard attention).  For MLA set to
                ``v_head_dim`` (e.g. 128).
        """
        B = batch_size
        H_Q = num_heads
        S_Q = seq_len
        S_K = seq_len_kv if seq_len_kv is not None else seq_len
        H_K = num_heads_kv if num_heads_kv is not None else num_heads
        D_qk = head_dim
        D_v = head_dim_v if head_dim_v is not None else head_dim
        bpe = self._bytes_per_element(dtype)

        return self._simulate_tile_level(
            B,
            H_Q,
            S_Q,
            S_K,
            H_K,
            D_qk,
            D_v,
            causal,
            dtype,
            bpe,
        )

    # ------------------------------------------------------------------
    # Tile-level simulation (primary mode — requires Origami)
    # ------------------------------------------------------------------

    def _create_tile_gemm_backend(
        self,
        gpu_arch: Optional[str],
        gpu_clock_mhz: Optional[int],
    ):
        """Try to create an Origami backend with 1 CU for per-tile simulation.

        Returns the backend on success, or ``None`` if Origami is not available.
        """
        try:
            from primus.core.projection.simulation_backends.origami_backend import (
                OrigamiGEMMBackend,
            )

            backend = OrigamiGEMMBackend(
                gpu_arch=gpu_arch,
                gpu_clock_mhz=gpu_clock_mhz,
                n_cu_override=1,
            )
            if backend.is_available():
                is_rank_0 = int(os.getenv("RANK", "0")) == 0
                if is_rank_0:
                    print(
                        "[Primus:SDPA] Using Origami 1-CU tile-level simulation "
                        "for Flash Attention"
                    )
                return backend
        except Exception:
            pass
        return None

    def _simulate_tile_level(
        self,
        B: int,
        H_Q: int,
        S_Q: int,
        S_K: int,
        H_K: int,
        D_qk: int,
        D_v: int,
        causal: bool,
        dtype: str,
        bpe: int,
    ) -> SimulationResult:
        """
        Tile-level SDPA simulation using Origami on a single CU.

        Flash Attention is a **fused** kernel — within each workgroup, the
        sub-operations (QKᵀ, softmax, PV) execute **sequentially** and the
        intermediate S/P matrices stay in LDS/registers (never written to
        HBM).  This means the correct timing model is **additive** across
        sub-operations per tile, not a global ``max(compute, memory)``
        roofline.

        Approach:
          1. Simulate each per-tile GEMM using Origami with ``n_cu=1``.
             This captures tile-level wave quantisation, LDS traffic, and
             pipeline effects that a global FLOP-rate model misses.
          2. Sum the per-tile GEMM times (additive, sequential execution).
          3. Multiply by ``num_waves = ⌈workgroups / N_CU⌉`` to account for
             CU-level parallelism across tiles.
          4. Add dQ atomic overhead (backward only).

        Forward per-workgroup (q_tile_m=256 Q rows, sweeps all S_K):
          QKᵀ: Q_tile[256, D_qk] × Kᵀ[D_qk, S_K] → S[256, S_K]
          PV:  P_tile[256, S_K]  × V[S_K, D_v]     → O[256, D_v]
          Workgroups = ⌈S_Q / 256⌉ × B × H_Q

        Backward per-workgroup (kv_tile_n=256 KV cols, sweeps all S_Q):
          5 GEMMs per workgroup (FA backward algorithm):
            1. QKᵀ recompute: Q[S_Q, D_qk] × Kᵀ[D_qk, 256] → S[S_Q, 256]
            2. dP = dO × Vᵀ:  dO[S_Q, D_v]  × Vᵀ[D_v, 256] → dP[S_Q, 256]
            3. dV = Pᵀ × dO:  Pᵀ[256, S_Q]  × dO[S_Q, D_v] → dV[256, D_v]
            4. dQ = dS × K:   dS[S_Q, 256]   × K[256, D_qk]  → dQ[S_Q, D_qk]
            5. dK = dSᵀ × Q:  dSᵀ[256, S_Q]  × Q[S_Q, D_qk] → dK[256, D_qk]
          Workgroups = ⌈S_K / 256⌉ × B × H_Q
        """
        assert self._tile_gemm is not None
        N_CU = self._hw.n_cu
        causal_factor = 0.5 if causal else 1.0

        # ==============================================================
        # FORWARD
        # ==============================================================
        fwd_n_wgs = math.ceil(S_Q / _FAV3_FWD.q_tile_m) * B * H_Q
        fwd_waves = math.ceil(fwd_n_wgs / N_CU)

        # Per-workgroup GEMMs on 1 CU (tile sweeps all S_K positions):
        #   QKᵀ: [q_tile_m, D_qk, S_K]
        r_fwd_qk = self._tile_gemm.simulate_gemm(
            m=_FAV3_FWD.q_tile_m,
            n=S_K,
            k=D_qk,
            dtype=dtype,
        )
        #   PV:  [q_tile_m, S_K, D_v]
        r_fwd_pv = self._tile_gemm.simulate_gemm(
            m=_FAV3_FWD.q_tile_m,
            n=D_v,
            k=S_K,
            dtype=dtype,
        )

        fwd_time_ms = (r_fwd_qk.forward_time_ms + r_fwd_pv.forward_time_ms) * fwd_waves

        # ==============================================================
        # BACKWARD
        # ==============================================================
        kv_tile = _FAV3_BWD.kv_tile_n  # 256
        bwd_n_wgs = math.ceil(S_K / kv_tile) * B * H_Q
        bwd_waves = math.ceil(bwd_n_wgs / N_CU)

        # Per-workgroup GEMMs (5 operations, full Q-sweep on 1 CU):
        # 1. QKᵀ recompute: [S_Q, D_qk, kv_tile]
        r_bwd_qk = self._tile_gemm.simulate_gemm(
            m=S_Q,
            n=kv_tile,
            k=D_qk,
            dtype=dtype,
        )
        # 2. dP = dO × Vᵀ: [S_Q, D_v, kv_tile]
        r_bwd_dp = self._tile_gemm.simulate_gemm(
            m=S_Q,
            n=kv_tile,
            k=D_v,
            dtype=dtype,
        )
        # 3. dV = Pᵀ × dO: [kv_tile, S_Q, D_v]
        r_bwd_dv = self._tile_gemm.simulate_gemm(
            m=kv_tile,
            n=D_v,
            k=S_Q,
            dtype=dtype,
        )
        # 4. dQ = dS × K: [S_Q, kv_tile, D_qk]
        r_bwd_dq = self._tile_gemm.simulate_gemm(
            m=S_Q,
            n=D_qk,
            k=kv_tile,
            dtype=dtype,
        )
        # 5. dK = dSᵀ × Q: [kv_tile, S_Q, D_qk]
        r_bwd_dk = self._tile_gemm.simulate_gemm(
            m=kv_tile,
            n=D_qk,
            k=S_Q,
            dtype=dtype,
        )

        bwd_compute_ms = (
            r_bwd_qk.forward_time_ms
            + r_bwd_dp.forward_time_ms
            + r_bwd_dv.forward_time_ms
            + r_bwd_dq.forward_time_ms
            + r_bwd_dk.forward_time_ms
        ) * bwd_waves

        # ── Backward dQ atomics (latency-based model) ──
        # Each KV-workgroup atomically accumulates dQ via buffer_atomic_add_f32.
        # The latency model counts warp-level reduction updates (global and
        # local) and multiplies by the per-op latency.
        num_k_tiles = math.ceil(kv_tile / kv_tile)  # = 1
        warp_updates_global = math.ceil(num_k_tiles * math.ceil(D_qk / _WARP_SIZE))
        total_updates_global = warp_updates_global * bwd_waves

        warp_updates_local = math.ceil(kv_tile * math.ceil(D_qk / _WARP_SIZE))
        total_updates_local = warp_updates_local * bwd_waves

        bwd_atomic_ms = (
            _ATOMIC_LATENCY_GLOBAL_NS * total_updates_global
            + _ATOMIC_LATENCY_LOCAL_NS * total_updates_local
        ) / 1e6  # ns → ms

        bwd_time_ms = bwd_compute_ms + bwd_atomic_ms

        # ==============================================================
        # METADATA (FLOPs, bytes — for achieved-TFLOPS reporting)
        # ==============================================================
        fwd_flops = (
            2.0 * B * H_Q * S_Q * S_K * D_qk  # QKᵀ
            + 2.0 * B * H_Q * S_Q * S_K * D_v  # PV
            + 5.0 * B * H_Q * S_Q * S_K  # softmax
        ) * causal_factor

        bwd_flops = (
            2.0 * B * H_Q * S_Q * S_K * D_qk  # QKᵀ recomp
            + 2.0 * B * H_Q * S_Q * S_K * D_v  # dP
            + 2.0 * B * H_Q * S_K * S_Q * D_v  # dV
            + 2.0 * B * H_Q * S_Q * S_K * D_qk  # dQ
            + 2.0 * B * H_Q * S_K * S_Q * D_qk  # dK
            + 5.0 * B * H_Q * S_Q * S_K  # softmax bwd
        ) * causal_factor

        fwd_bytes = (
            B * H_Q * S_Q * D_qk * bpe  # Q
            + B * H_K * S_K * D_qk * bpe  # K
            + B * H_K * S_K * D_v * bpe  # V
            + B * H_Q * S_Q * D_v * bpe  # O
            + B * H_Q * S_Q * 4  # logsumexp (fp32)
        )
        bwd_bytes = (
            B * H_Q * S_Q * D_qk * bpe  # Q
            + B * H_K * S_K * D_qk * bpe  # K
            + B * H_K * S_K * D_v * bpe  # V
            + B * H_Q * S_Q * D_v * bpe  # O
            + B * H_Q * S_Q * D_v * bpe  # dO
            + B * H_Q * S_Q * 4  # logsumexp (fp32)
            + B * H_K * S_K * D_qk * bpe  # dK
            + B * H_K * S_K * D_v * bpe  # dV
        )

        fwd_achieved_tflops = (
            (fwd_flops / (fwd_time_ms * 1e-3)) / 1e12 if fwd_time_ms > 0 else 0
        )

        return SimulationResult(
            forward_time_ms=fwd_time_ms,
            backward_time_ms=bwd_time_ms,
            tflops=fwd_achieved_tflops,
            bandwidth_gbps=(
                (fwd_bytes / (fwd_time_ms * 1e-3)) / 1e9 if fwd_time_ms > 0 else 0
            ),
            metadata={
                "backend": "sdpa_simulator (FAv3 tile-level, Origami 1-CU)",
                # Standard metadata keys (for compatibility)
                "fwd_compute_bound": True,
                "fwd_compute_ms": fwd_time_ms,
                "fwd_memory_ms": 0.0,  # included in per-tile Origami model
                "bwd_bottleneck": "compute+atomic",
                "bwd_compute_ms": bwd_compute_ms,
                "bwd_memory_ms": 0.0,  # included in per-tile Origami model
                "bwd_atomic_ms": bwd_atomic_ms,
                "fwd_flops": fwd_flops,
                "bwd_flops": bwd_flops,
                "fwd_bytes": fwd_bytes,
                "bwd_bytes": bwd_bytes,
                "seq_len_q": S_Q,
                "seq_len_kv": S_K,
                "num_heads_q": H_Q,
                "num_heads_kv": H_K,
                "causal": causal,
                # Tile-level details
                "fwd_waves": fwd_waves,
                "fwd_n_workgroups": fwd_n_wgs,
                "fwd_qk_per_tile_ms": r_fwd_qk.forward_time_ms,
                "fwd_pv_per_tile_ms": r_fwd_pv.forward_time_ms,
                "bwd_waves": bwd_waves,
                "bwd_n_workgroups": bwd_n_wgs,
                "bwd_qk_recomp_per_tile_ms": r_bwd_qk.forward_time_ms,
                "bwd_dp_per_tile_ms": r_bwd_dp.forward_time_ms,
                "bwd_dv_per_tile_ms": r_bwd_dv.forward_time_ms,
                "bwd_dq_per_tile_ms": r_bwd_dq.forward_time_ms,
                "bwd_dk_per_tile_ms": r_bwd_dk.forward_time_ms,
                "n_cu": N_CU,
                # FAv3 tile parameters
                "fwd_q_tile_m": _FAV3_FWD.q_tile_m,
                "fwd_kv_tile_n": _FAV3_FWD.kv_tile_n,
                "bwd_q_tile_m": _FAV3_BWD.q_tile_m,
                "bwd_kv_tile_n": _FAV3_BWD.kv_tile_n,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bytes_per_element(self, dtype: str) -> int:
        return {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1}.get(dtype, 2)
