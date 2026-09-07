###############################################################################
# HSTU gated-jagged attention simulator (FAv3 tile-level base).
#
# Subclasses the FAv3 ``SDPASimulator`` so the dominant matmul work (QKᵀ and
# A·V, plus the five backward sub-GEMMs) is still priced per-tile on a single
# CU via the origami 1-CU GEMM backend.  On top of that matmul time we add the
# HSTU-specific pointwise epilogue that ordinary softmax-flash does not have:
#
#   * relative positional/temporal bias add per score,
#   * SiLU gating of the scores (instead of softmax),
#   * the U output gate + normalization on the attention output.
#
# The softmax-flash base under-models these because in a plain SDPA kernel the
# epilogue is negligible next to the matmuls; in HSTU it is not.  We price the
# epilogue as a throughput over the (causal) score-matrix elements — a single
# hardware-portable rate in Gelem/s — rather than folding everything into one
# opaque efficiency constant.  The matmul portion continues to scale correctly
# with sequence length, batch and architecture through the tile sweep.
###############################################################################

from __future__ import annotations

import os
from typing import Optional

from primus.core.projection.simulation_backends.base import SimulationResult
from primus.core.projection.simulation_backends.sdpa_simulator import (
    GPUHardwareSpec,
    SDPASimulator,
)

# Default fused-epilogue throughputs (score elements processed per second,
# chip-wide) for HSTU's SiLU-gate + relative-bias + U-gate work.  These are
# hardware throughputs — arch-portable and shape-independent — calibrated once
# against the MI350X flydsl trace (see examples/project_dlrm.py).  They price
# only the pointwise epilogue; the matmuls come from the FAv3 tile sweep.
_DEFAULT_EPILOGUE_GELEM_FWD = 1097.0  # ~1.1 Telem/s forward
_DEFAULT_EPILOGUE_GELEM_BWD = 798.0  # backward gate/bias grads are heavier


class HSTUAttentionSimulator(SDPASimulator):
    """FAv3 tile-level simulator specialised for HSTU gated-jagged attention.

    Reuses the parent's per-tile matmul pricing (origami 1-CU) unchanged and
    adds the HSTU pointwise epilogue as an additive throughput term.  The
    epilogue is *not* present in softmax-flash, so the parent alone
    under-predicts HSTU attention by ~2.6x; this class closes that gap with a
    physical rate instead of a lumped efficiency factor.
    """

    def __init__(
        self,
        gpu_arch: Optional[str] = None,
        hardware_spec: Optional[GPUHardwareSpec] = None,
        gpu_clock_mhz: Optional[int] = None,
        gemm_backend: Optional[str] = None,
        epilogue_gelem_fwd: float = _DEFAULT_EPILOGUE_GELEM_FWD,
        epilogue_gelem_bwd: float = _DEFAULT_EPILOGUE_GELEM_BWD,
    ):
        """
        Args:
            gpu_arch / hardware_spec / gpu_clock_mhz: forwarded to
                ``SDPASimulator``.
            gemm_backend: tile GEMM engine.  Must honour ``n_cu_override=1``
                (origami); gemmologist prices across the full chip and is
                unsuitable for per-tile simulation, so a non-origami request is
                coerced to origami for the tiles.
            epilogue_gelem_fwd / epilogue_gelem_bwd: fused-epilogue throughput
                (Gelem/s) for the SiLU-gate + relative-bias + U-gate work.
        """
        # Only origami honours the 1-CU tile override; force it for the tiles.
        name = (gemm_backend or os.getenv("PRIMUS_GEMM_BACKEND") or "origami").lower().strip()
        if name != "origami":
            if int(os.getenv("RANK", "0")) == 0:
                print(
                    f"[Primus:HSTU-Attn] tile GEMM backend '{name}' cannot price "
                    "per-tile 1-CU GEMMs; using origami for the attention tiles."
                )
            name = "origami"

        super().__init__(
            gpu_arch=gpu_arch,
            hardware_spec=hardware_spec,
            gpu_clock_mhz=gpu_clock_mhz,
            gemm_backend=name,
        )
        self._epi_fwd = float(epilogue_gelem_fwd)
        self._epi_bwd = float(epilogue_gelem_bwd)

    def name(self) -> str:
        return f"hstu_attention_simulator (FAv3 gated-jagged, {self._mode} 1-CU)"

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
        """Price HSTU attention = FAv3 matmul tiles + HSTU pointwise epilogue.

        The base class returns the fused matmul time (QKᵀ + A·V forward; the
        five sub-GEMMs backward).  We add the SiLU-gate/relative-bias/U-gate
        epilogue as ``score_elements / throughput``.  For jagged batches the
        caller passes the effective sqrt(E[L²]) sequence length so the score
        count reflects sequence-length variance.
        """
        base = super().simulate_sdpa(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            causal=causal,
            dtype=dtype,
            seq_len_kv=seq_len_kv,
            num_heads_kv=num_heads_kv,
            head_dim_v=head_dim_v,
        )

        S_K = seq_len_kv if seq_len_kv is not None else seq_len
        causal_factor = 0.5 if causal else 1.0
        # Causal score-matrix elements touched by the epilogue.
        score_elems = batch_size * num_heads * seq_len * S_K * causal_factor

        # Gelem/s -> elements/ms is a factor of 1e6.
        pw_fwd_ms = score_elems / (self._epi_fwd * 1e6) if self._epi_fwd > 0 else 0.0
        pw_bwd_ms = score_elems / (self._epi_bwd * 1e6) if self._epi_bwd > 0 else 0.0

        md = dict(base.metadata)
        md.update(
            {
                "backend": "hstu_attention_simulator (FAv3 gated-jagged, Origami 1-CU)",
                "hstu_matmul_fwd_ms": base.forward_time_ms,
                "hstu_matmul_bwd_ms": base.backward_time_ms,
                "hstu_epilogue_fwd_ms": pw_fwd_ms,
                "hstu_epilogue_bwd_ms": pw_bwd_ms,
                "hstu_score_elems": score_elems,
                "hstu_epilogue_gelem_fwd": self._epi_fwd,
                "hstu_epilogue_gelem_bwd": self._epi_bwd,
            }
        )

        fwd_ms = base.forward_time_ms + pw_fwd_ms
        return SimulationResult(
            forward_time_ms=fwd_ms,
            backward_time_ms=base.backward_time_ms + pw_bwd_ms,
            tflops=(base.metadata.get("fwd_flops", 0.0) / (fwd_ms * 1e-3) / 1e12 if fwd_ms > 0 else 0.0),
            bandwidth_gbps=base.bandwidth_gbps,
            metadata=md,
        )
