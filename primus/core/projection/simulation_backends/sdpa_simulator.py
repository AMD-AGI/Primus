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
      • MFMA instruction: ``v_mfma_f32_32x32x16_bf16``
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

The model uses a **roofline** approach:
    time = max(compute_time, memory_time, atomic_time)
with FAv3-specific compute/memory efficiency factors and CU utilisation
derived from the tile sizes.

In the backward pass, the dQ gradient is accumulated across KV-workgroups
using ``buffer_atomic_add_f32`` (72 atomic instructions in the kernel).
Each KV-workgroup processes all Q positions and atomically adds its partial
dQ contribution, leading to contention proportional to ⌈S / 256⌉ concurrent
writers per dQ cache line.  The atomic overhead is modelled as a separate
bottleneck dimension in the roofline.
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

    # Max wavefronts per CU (SIMD occupancy limit)
    max_waves_per_cu: int = 8

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
            "mi325x": 1200,
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

    The model captures:
      1. **Total FLOPs** from the SDPA math (QKᵀ, softmax, PV for fwd;
         dV, dP/dS, dQ, dK, softmax-bwd for bwd).
      2. **Flash-Attention memory IO** — Q/K/V are streamed from HBM;
         the full S/P matrices are never materialised.
      3. **CU utilisation** — derived from the FAv3 tile sizes and the
         number of workgroups that can execute concurrently.
      4. **Achieved efficiency** — higher than generic kernels because
         FAv3 is hand-tuned ISA with software pipelining and LDS-based
         data movement.
      5. **Atomic overhead (BWD only)** — dQ is accumulated across
         KV-workgroups via ``buffer_atomic_add_f32`` in FP32.  The model
         accounts for the read-modify-write penalty and contention from
         ⌈S / 256⌉ concurrent writers per dQ cache line.
    """

    def __init__(
        self,
        gpu_arch: Optional[str] = None,
        hardware_spec: Optional[GPUHardwareSpec] = None,
        compute_efficiency: float = 0.51,
        memory_efficiency: float = 0.85,
        atomic_rmw_factor: float = 4.0,
        gpu_clock_mhz: Optional[int] = None,
    ):
        """
        Args:
            gpu_arch: GPU architecture string (e.g. "mi300x", "gfx942",
                "mi355x", "gfx950").
            hardware_spec: Override hardware spec directly.
            compute_efficiency: Fraction of peak TFLOPS achieved (0-1).
                Calibrated against measured FAv3 traces on MI300X:
                  * Measured FA fwd = 5.05 ms, bwd = 10.00 ms
                    (B=3, H_Q=64, S=8192, D=128, H_KV=8, causal, BF16)
                  * 0.51 matches measured within 1%.
                The lower-than-peak efficiency (vs theoretical 0.75-0.85)
                accounts for GQA head broadcasting, LDS bank conflicts,
                barrier synchronisation, and register pressure.
            memory_efficiency: Fraction of peak HBM bandwidth achieved (0-1).
                FAv3 streaming pattern typically achieves 0.80-0.90.
            atomic_rmw_factor: Base slowdown of ``buffer_atomic_add_f32``
                relative to a plain ``buffer_store`` (read-modify-write
                overhead).  Typical range 3-6 on CDNA3.  Contention from
                multiple writers is modelled *on top* of this factor.
            gpu_clock_mhz: Override the GPU compute clock frequency in MHz.
                If provided, the profile's TFLOPS are scaled proportionally.
        """
        self._hw = hardware_spec or _get_hardware_spec(gpu_arch, gpu_clock_mhz)
        self._compute_eff = compute_efficiency
        self._memory_eff = memory_efficiency
        self._atomic_rmw_factor = atomic_rmw_factor

    def name(self) -> str:
        return "sdpa_simulator (FAv3)"

    def is_available(self) -> bool:
        return True  # Pure-Python analytical model, always available

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
    ) -> SimulationResult:
        """
        Simulate FAv3 SDPA execution time using a roofline model
        parameterised by the actual FAv3 tile configuration.

        Args:
            batch_size: Batch size (B).
            num_heads: Number of query heads (H_Q).
            seq_len: Query sequence length (S_Q).
            head_dim: Head dimension (D).
            causal: Whether causal masking is applied.
            dtype: Data type ("bf16", "fp16", "fp8", "fp32").
            seq_len_kv: Key/Value sequence length (S_K).  Defaults to
                ``seq_len`` (self-attention).  Set differently for
                cross-attention or prefill with separate KV cache length.
            num_heads_kv: Number of KV heads.  Defaults to ``num_heads``
                (MHA).  Set lower for GQA/MQA.
        """
        B = batch_size
        H_Q = num_heads
        S_Q = seq_len
        S_K = seq_len_kv if seq_len_kv is not None else seq_len
        H_K = num_heads_kv if num_heads_kv is not None else num_heads
        D = head_dim
        bpe = self._bytes_per_element(dtype)

        # GQA ratio: each KV head serves (H_Q / H_K) query heads.
        # The FLOPs are still per-query-head, so total FLOPs scale with H_Q.
        # Memory for K/V scales with H_K, memory for Q/O scales with H_Q.

        causal_factor = 0.5 if causal else 1.0

        # ==============================================================
        # 1.  COMPUTE  (FLOP counts)
        # ==============================================================
        # Forward  (per query head, then × H_Q)
        #   QKᵀ : 2·B·H_Q·S_Q·S_K·D      (batched GEMM)
        #   softmax : ~5·B·H_Q·S_Q·S_K    (exp, sub-max, sum, div, mul)
        #   PV  : 2·B·H_Q·S_Q·S_K·D      (batched GEMM — P is S_Q×S_K, V is S_K×D)
        # NOTE for PV: output is (S_Q, D), inner dim is S_K.
        # For causal masking, only ~half the S_Q×S_K elements are computed
        # (only valid when S_Q == S_K; for cross-attn causal is usually False).
        fwd_gemm_flops = 2.0 * (2.0 * B * H_Q * S_Q * S_K * D) * causal_factor
        fwd_softmax_flops = 5.0 * B * H_Q * S_Q * S_K * causal_factor
        fwd_flops = fwd_gemm_flops + fwd_softmax_flops

        # Backward (4 batched GEMMs + softmax backward)
        #   dV  = Pᵀ @ dO     : 2·B·H_Q·S_K·S_Q → (S_K, D)  inner dim S_Q
        #   dP  = dO @ Vᵀ     : 2·B·H_Q·S_Q·D → (S_Q, S_K)  inner dim D
        #   dS  = softmax_bwd  : ~5·B·H_Q·S_Q·S_K
        #   dQ  = dS @ K       : 2·B·H_Q·S_Q·S_K → (S_Q, D)  inner dim S_K
        #   dK  = dSᵀ @ Q      : 2·B·H_Q·S_K·S_Q → (S_K, D)  inner dim S_Q
        bwd_gemm_flops = 2.0 * (4.0 * B * H_Q * S_Q * S_K * D) * causal_factor
        bwd_softmax_flops = 5.0 * B * H_Q * S_Q * S_K * causal_factor
        bwd_flops = bwd_gemm_flops + bwd_softmax_flops

        # ==============================================================
        # 2.  MEMORY IO  (Flash Attention – no S/P materialised to HBM)
        # ==============================================================
        # Forward reads:  Q (B·H_Q·S_Q·D), K (B·H_K·S_K·D), V (B·H_K·S_K·D)
        # Forward writes: O (B·H_Q·S_Q·D) + logsumexp (B·H_Q·S_Q, fp32)
        fwd_read_bytes = (
            B * H_Q * S_Q * D * bpe  # Q
            + B * H_K * S_K * D * bpe  # K
            + B * H_K * S_K * D * bpe  # V
        )
        fwd_write_bytes = (
            B * H_Q * S_Q * D * bpe + B * H_Q * S_Q * 4  # O  # logsumexp (fp32)
        )
        fwd_bytes = fwd_read_bytes + fwd_write_bytes

        # Backward reads: Q, K, V, O, dO + logsumexp
        # Backward regular writes: dK (B·H_K·S_K·D) + dV (B·H_K·S_K·D)
        #   NOTE: dQ uses buffer_atomic_add_f32 — accounted separately.
        bwd_read_bytes = (
            B * H_Q * S_Q * D * bpe  # Q
            + B * H_K * S_K * D * bpe  # K
            + B * H_K * S_K * D * bpe  # V
            + B * H_Q * S_Q * D * bpe  # O
            + B * H_Q * S_Q * D * bpe  # dO
            + B * H_Q * S_Q * 4  # logsumexp (fp32)
        )
        bwd_regular_write_bytes = (
            B * H_K * S_K * D * bpe + B * H_K * S_K * D * bpe  # dK  # dV
        )
        bwd_bytes = bwd_read_bytes + bwd_regular_write_bytes

        # ==============================================================
        # 3.  dQ ATOMIC OVERHEAD  (BWD only)
        # ==============================================================
        # In FAv3 backward, each KV-workgroup loops over ALL Q positions
        # and atomically accumulates its partial dQ via buffer_atomic_add_f32.
        #
        # From the FAv3 backward kernel:
        #   - 72 buffer_atomic_add_f32 instructions in the kernel
        #   - 8 atomics per Q-block (per wavefront, 64 threads each)
        #   - 4 wavefronts per workgroup
        #   - Per Q-block: 8 × 64 × 4W = 2048 atomic ops = 8 KB (FP32)
        #     = 16 rows × 128 cols × 4 bytes = 8192 bytes  ✓
        #
        # Contention & L2 coalescing:
        #   ceil(S_K/256) KV-workgroups all write to the same dQ rows.
        #   Workgroups on the SAME XCD can coalesce their atomics in the
        #   local L2 cache (the add is accumulated in L2, only the final
        #   value is flushed to HBM).  So the effective number of HBM
        #   atomic writes per dQ element is min(n_kv_wgs, n_xcd) rather
        #   than the full n_kv_wgs.
        #
        #   Each HBM atomic write is a read-modify-write, which costs
        #   ~rmw_factor × the bandwidth of a regular store.
        n_kv_workgroups = math.ceil(S_K / _FAV3_BWD.kv_tile_n)

        # How many KV-workgroups per XCD (for L2 coalescing estimate)
        hbm_writers_per_element = min(n_kv_workgroups, self._hw.n_xcd)

        # Effective dQ bytes hitting HBM (after L2 coalescing)
        # dQ shape is (B, H_Q, S_Q, D), stored in FP32 (4 bytes)
        dq_atomic_bytes = float(hbm_writers_per_element) * B * H_Q * S_Q * D * 4.0

        # Atomic slowdown = just the RMW factor (contention within-XCD
        # is absorbed by L2; cross-XCD traffic goes to different memory
        # channels and can proceed in parallel)
        atomic_slowdown = self._atomic_rmw_factor

        # ==============================================================
        # 4.  CU UTILISATION  (from FAv3 tile config)
        # ==============================================================
        fwd_cu_util = self._cu_utilisation(B, H_Q, S_Q, _FAV3_FWD)
        bwd_cu_util = self._cu_utilisation(B, H_Q, S_K, _FAV3_BWD)

        # ==============================================================
        # 5.  ROOFLINE:  time = max(compute, memory, atomics)
        # ==============================================================
        peak_tflops = self._peak_tflops(dtype)

        # Effective throughput = peak × efficiency × CU utilisation
        fwd_eff_tflops = peak_tflops * self._compute_eff * fwd_cu_util
        bwd_eff_tflops = peak_tflops * self._compute_eff * bwd_cu_util

        fwd_eff_bw = self._hw.hbm_bandwidth_gbps * self._memory_eff
        bwd_eff_bw = self._hw.hbm_bandwidth_gbps * self._memory_eff

        # Effective atomic bandwidth (HBM BW reduced by RMW + contention)
        bwd_eff_atomic_bw = (
            self._hw.hbm_bandwidth_gbps * self._memory_eff / atomic_slowdown
        )

        # Compute-bound time (ms)
        fwd_compute_ms = (fwd_flops / (fwd_eff_tflops * 1e12)) * 1e3
        bwd_compute_ms = (bwd_flops / (bwd_eff_tflops * 1e12)) * 1e3

        # Memory-bound time (ms)  — regular (non-atomic) IO
        fwd_memory_ms = (fwd_bytes / (fwd_eff_bw * 1e9)) * 1e3
        bwd_memory_ms = (bwd_bytes / (bwd_eff_bw * 1e9)) * 1e3

        # Atomic-bound time (ms)  — dQ accumulation via buffer_atomic_add_f32
        bwd_atomic_ms = (dq_atomic_bytes / (bwd_eff_atomic_bw * 1e9)) * 1e3

        fwd_time_ms = max(fwd_compute_ms, fwd_memory_ms)
        bwd_time_ms = max(bwd_compute_ms, bwd_memory_ms, bwd_atomic_ms)

        # Achieved metrics
        fwd_achieved_tflops = (
            (fwd_flops / (fwd_time_ms * 1e-3)) / 1e12 if fwd_time_ms > 0 else 0
        )

        # Determine what bounds each pass
        bwd_bottleneck = "compute"
        if bwd_atomic_ms >= bwd_compute_ms and bwd_atomic_ms >= bwd_memory_ms:
            bwd_bottleneck = "atomic"
        elif bwd_memory_ms >= bwd_compute_ms:
            bwd_bottleneck = "memory"

        return SimulationResult(
            forward_time_ms=fwd_time_ms,
            backward_time_ms=bwd_time_ms,
            tflops=fwd_achieved_tflops,
            bandwidth_gbps=(
                (fwd_bytes / (fwd_time_ms * 1e-3)) / 1e9 if fwd_time_ms > 0 else 0
            ),
            metadata={
                "backend": "sdpa_simulator (FAv3)",
                "fwd_compute_bound": fwd_compute_ms >= fwd_memory_ms,
                "fwd_compute_ms": fwd_compute_ms,
                "fwd_memory_ms": fwd_memory_ms,
                "bwd_bottleneck": bwd_bottleneck,
                "bwd_compute_ms": bwd_compute_ms,
                "bwd_memory_ms": bwd_memory_ms,
                "bwd_atomic_ms": bwd_atomic_ms,
                "fwd_flops": fwd_flops,
                "bwd_flops": bwd_flops,
                "fwd_bytes": fwd_bytes,
                "bwd_bytes": bwd_bytes,
                "seq_len_q": S_Q,
                "seq_len_kv": S_K,
                "num_heads_q": H_Q,
                "num_heads_kv": H_K,
                # dQ atomic details (buffer_atomic_add_f32)
                "bwd_dq_kv_workgroups": n_kv_workgroups,
                "bwd_dq_hbm_writers_per_elem": hbm_writers_per_element,
                "bwd_dq_atomic_hbm_bytes": dq_atomic_bytes,
                "bwd_dq_rmw_factor": atomic_slowdown,
                "bwd_eff_atomic_bw_gbps": bwd_eff_atomic_bw,
                # CU utilisation
                "fwd_cu_utilisation": fwd_cu_util,
                "bwd_cu_utilisation": bwd_cu_util,
                "causal": causal,
                # FAv3 tile parameters
                "fwd_q_tile_m": _FAV3_FWD.q_tile_m,
                "fwd_kv_tile_n": _FAV3_FWD.kv_tile_n,
                "fwd_wavefronts": _FAV3_FWD.n_wavefronts,
                "bwd_q_tile_m": _FAV3_BWD.q_tile_m,
                "bwd_kv_tile_n": _FAV3_BWD.kv_tile_n,
                "bwd_wavefronts": _FAV3_BWD.n_wavefronts,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bytes_per_element(self, dtype: str) -> int:
        return {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1}.get(dtype, 2)

    def _peak_tflops(self, dtype: str) -> float:
        return {
            "bf16": self._hw.peak_tflops_bf16,
            "fp16": self._hw.peak_tflops_fp16,
            "fp8": self._hw.peak_tflops_fp8,
            "fp32": self._hw.peak_tflops_bf16 / 4,
        }.get(dtype, self._hw.peak_tflops_bf16)

    def _cu_utilisation(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        tile_cfg: _FAv3TileConfig,
    ) -> float:
        """
        Estimate CU utilisation for a FAv3 kernel launch.

        FAv3 forward dispatches one workgroup per Q-tile per (batch, head).
        FAv3 backward dispatches one workgroup per KV-tile per (batch, head).

        Each workgroup occupies ``n_wavefronts`` wavefront slots on a CU.
        If the workgroup uses fewer than ``max_waves_per_cu`` wavefronts,
        multiple workgroups *may* share a CU (higher occupancy).

        CU utilisation = min(active_CUs, N_CU) / N_CU
        """
        # Number of workgroups
        # For FWD: each wg handles q_tile_m rows → ceil(S / q_tile_m) wgs per (B,H)
        # For BWD: each wg handles kv_tile_n cols → ceil(S / kv_tile_n) wgs per (B,H)
        if tile_cfg is _FAV3_FWD:
            n_tiles = math.ceil(seq_len / tile_cfg.q_tile_m)
        else:
            # BWD: workgroups over KV dimension
            n_tiles = math.ceil(seq_len / tile_cfg.kv_tile_n)

        n_workgroups = n_tiles * batch_size * num_heads

        # How many workgroups can share a single CU?
        wgs_per_cu = self._hw.max_waves_per_cu // tile_cfg.n_wavefronts
        wgs_per_cu = max(wgs_per_cu, 1)

        # Effective CU slots
        cu_slots = self._hw.n_cu * wgs_per_cu
        active_slots = min(n_workgroups, cu_slots)

        return active_slots / cu_slots
