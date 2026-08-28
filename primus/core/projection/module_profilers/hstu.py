###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
HSTU (Hierarchical Sequential Transduction Unit) layer profiler.

HSTU is the transformer-like block used by DLRM-v4 rankers (Meta's generative
recommenders / Yambda-5B).  It differs from a standard attention layer in ways
that matter for projection:

* a **single fused UVQK projection** produces four streams (U gate, V value,
  Q query, K key) in one GEMM instead of separate Q/K/V projections;
* the sequence is **jagged** -- padded to ``max_seq_len`` but only a fraction
  (``hstu_fill_factor``, ~0.4 for Yambda) of the positions are valid, so the
  attention core does far less work than a fixed-length model of the same
  padded length;
* the attention output is **SiLU-gated by U** before the output projection.

We price it first-principles with the shared GEMM/SDPA simulation backends:

  1. fused UVQK GEMM:  [T, D] * [D, H*(2*d_qk + 2*d_v)]
  2. attention core:   SDPA over the *effective* (fill-adjusted) sequence,
     head_dim = d_qk, head_dim_v = d_v, causal
  3. output GEMM:      [T, H*d_v] * [H*d_v, D]

Norms / SiLU gating are memory-bound and added as a small HBM term.
"""

from typing import Optional

from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.simulation_backends.base import resolve_hbm_bytes_per_ms
from primus.core.projection.training_config import gemm_dtype_from_config

# Fallback peak HBM bandwidth (MI300-class, ~4 TB/s) when no arch profile is
# reachable; the real value is resolved from the simulation backends so HBM4
# parts are not mis-priced.
_FALLBACK_HBM_GBPS = 4000.0
_ELEMENTWISE_HBM_FRACTION = 0.60  # SiLU/norm streaming efficiency


class HSTULayerProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self._gemm_backend = None
        self._sdpa_backend = None
        self._cached = None
        self._cache_key = None
        self._components = {}

    # -- backend wiring (mirrors DenseTransformerLayerProfiler) ----------------
    def set_simulation_backends(self, gemm_backend=None, sdpa_backend=None):
        self._gemm_backend = gemm_backend
        self._sdpa_backend = sdpa_backend
        self._cached = None
        self._cache_key = None

    def set_gemm_backend(self, backend):
        self._gemm_backend = backend

    def set_sdpa_backend(self, backend):
        self._sdpa_backend = backend

    def set_layer_module(self, module):
        self._cached = None
        self._cache_key = None

    # -- geometry --------------------------------------------------------------
    def _geom(self):
        mc = self.config.model_config
        D = int(mc.hidden_size or mc.embedding_dim or 0)
        H = int(mc.hstu_num_heads or mc.num_attention_heads or 1)
        d_qk = int(mc.hstu_qk_dim or mc.kv_channels or (D // max(1, H)))
        d_v = int(mc.hstu_v_dim or d_qk)
        return D, H, d_qk, d_v

    def _tp(self) -> int:
        return max(1, int(self.config.model_parallel_config.tensor_model_parallel_size or 1))

    def _uvqk_out(self, H, d_qk, d_v) -> int:
        # U, V (d_v each) + Q, K (d_qk each), per head.
        return H * (2 * d_qk + 2 * d_v)

    # -- params ----------------------------------------------------------------
    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        D, H, d_qk, d_v = self._geom()
        uvqk = D * self._uvqk_out(H, d_qk, d_v)
        out = (H * d_v) * D
        total = uvqk + out
        if rank is not None:
            total = total // self._tp()  # heads sharded across TP
        return int(total)

    # -- activation ------------------------------------------------------------
    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        D, H, d_qk, d_v = self._geom()
        tp = self._tp()
        # Padded sequence drives the buffer sizes; UVQK stream is the largest.
        uvqk_width = self._uvqk_out(H, d_qk, d_v) // tp
        return int(batch_size * seq_len * (D + uvqk_width) * 2)  # bf16

    # -- compute ---------------------------------------------------------------
    def _fill(self) -> tuple[float, float]:
        mc = self.config.model_config
        mean = min(1.0, max(0.01, float(getattr(mc, "hstu_fill_factor", 1.0) or 1.0)))
        std = max(0.0, float(getattr(mc, "hstu_fill_factor_std", 0.0) or 0.0))
        return mean, std

    def _mean_seq(self, seq_len: int) -> int:
        """Expected valid tokens E[L] -- drives the (linear) GEMM/elementwise work."""
        mean, _ = self._fill()
        return max(1, int(round(seq_len * mean)))

    def _peak_flops_per_ms(self) -> float:
        """Realizable bf16 matmul peak (flops/ms), probed from the GEMM backend.

        A large square GEMM runs at ~peak, so 2*n^3 / fwd_ms is the backend's
        achievable peak -- backend-agnostic and consistent with how the GEMMs in
        this same projection are priced.  Cached across calls.
        """
        if getattr(self, "_peak_fpm", None):
            return self._peak_fpm
        n = 8192
        g = self._gemm_backend.simulate_gemm(n, n, n, dtype="bf16")
        t = g.forward_time_ms or 0.0
        self._peak_fpm = (2.0 * n * n * n / t) if t > 0 else 1.0e12
        return self._peak_fpm

    def _attn_seq(self, seq_len: int) -> int:
        """Effective sequence for the attention core, which scales as E[L^2].

        E[L^2] = mean^2 + std^2 >= mean^2, so pricing the quadratic term with an
        rms-fill sequence corrects the systematic under-count from squaring the
        mean fill.  With std=0 this reduces to the mean (backward compatible).
        """
        mean, std = self._fill()
        rms = min(1.0, (mean * mean + std * std) ** 0.5)
        return max(1, int(round(seq_len * rms)))

    def _get_simulated_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        mc = self.config.model_config
        D, H, d_qk, d_v = self._geom()
        tp = self._tp()
        dtype = gemm_dtype_from_config(mc)
        mean_seq = self._mean_seq(seq_len)
        attn_seq = self._attn_seq(seq_len)
        tokens = batch_size * mean_seq  # jagged: only valid positions do work
        heads_per_rank = max(1, H // tp)

        # 1. Fused UVQK projection GEMM: [T, D] x [D, H*(2 d_qk + 2 d_v)]/tp
        uvqk_n = self._uvqk_out(heads_per_rank, d_qk, d_v)
        g = self._gemm_backend.simulate_gemm(tokens, uvqk_n, D, dtype=dtype)
        uvqk_fwd = g.forward_time_ms
        uvqk_bwd = g.backward_time_ms or (2.0 * g.forward_time_ms)

        # 2. Attention core (gated dot-product).  HSTU attention is a *gated
        #    jagged* attention (SiLU gate + relative bias, not softmax); the FAv3
        #    SDPA roofline does not model it (it under-predicts by >30x and does
        #    not scale as L^2).  When hstu_attn_flop_efficiency is set we price it
        #    directly from its causal FLOPs at the measured achieved fraction of
        #    realizable matmul peak; otherwise we fall back to the SDPA roofline.
        flop_eff = float(getattr(mc, "hstu_attn_flop_efficiency", 0.0) or 0.0)
        if flop_eff > 0:
            flop_eff = min(1.0, max(0.001, flop_eff))
            # Causal QK^T + A.V: B * heads * L^2 * (d_qk + d_v) flops per layer
            # (the causal 1/2 and the 2-matmul/2-MAC factors cancel to 1).
            attn_flops = batch_size * heads_per_rank * (attn_seq * attn_seq) * (d_qk + d_v)
            peak_fpm = self._peak_flops_per_ms()
            attn_fwd = attn_flops / (peak_fpm * flop_eff)
            bwd_ratio = float(getattr(mc, "hstu_attn_bwd_ratio", 2.0) or 2.0)
            attn_bwd = attn_fwd * max(1.0, bwd_ratio)
        else:
            s = self._sdpa_backend.simulate_sdpa(
                batch_size=batch_size,
                num_heads=heads_per_rank,
                seq_len=attn_seq,
                head_dim=d_qk,
                causal=True,
                dtype="bf16",
                head_dim_v=d_v,
            )
            eff = min(1.0, max(0.01, float(getattr(mc, "hstu_attn_efficiency", 1.0) or 1.0)))
            bwd_eff = float(getattr(mc, "hstu_attn_bwd_efficiency", 0.0) or 0.0)
            bwd_eff = min(1.0, max(0.01, bwd_eff)) if bwd_eff > 0 else eff
            attn_fwd = s.forward_time_ms / eff
            attn_bwd = (s.backward_time_ms or (2.0 * s.forward_time_ms)) / bwd_eff

        # 3. Output projection GEMM: [T, out_in]/tp x [out_in, D].  The gated
        #    attention output is concatenated with residual/gate streams, so the
        #    measured input width is 3*D (1536), not H*d_v -- configurable.
        out_in = int(getattr(mc, "hstu_output_input_dim", 0) or 0) or (heads_per_rank * d_v)
        o = self._gemm_backend.simulate_gemm(tokens, D, out_in, dtype=dtype)
        out_fwd = o.forward_time_ms
        out_bwd = o.backward_time_ms or (2.0 * o.forward_time_ms)

        # Selective activation recomputation: the UVQK projection is recomputed
        # in the backward to regenerate Q/K/V for the attention-backward, so its
        # forward GEMM cost is charged again on the backward path.
        if bool(getattr(mc, "hstu_recompute_attn", False)):
            uvqk_bwd += uvqk_fwd

        # 4. SiLU gating (U) + norms + dropout (input/linear w/ masks) + jagged
        #    pack/unpack: memory-bound elementwise passes over the UVQK + D
        #    activation footprint.  Each logical pass is a read plus a write.
        peak_hbm_bytes_per_ms = resolve_hbm_bytes_per_ms(
            gemm_backend=self._gemm_backend,
            sdpa_backend=self._sdpa_backend,
            fallback_gbps=_FALLBACK_HBM_GBPS,
        )
        passes = max(1.0, float(getattr(mc, "hstu_elementwise_passes", 6.0) or 6.0))
        elem_bytes = tokens * (uvqk_n + D) * 2 * passes  # bf16, read+write per pass
        elem_ms = elem_bytes / (peak_hbm_bytes_per_ms * _ELEMENTWISE_HBM_FRACTION)

        # Per-role split (mirrors the Kineto trace buckets) so a projection can
        # be calibrated against measured kernel time role-by-role.
        self._components = {
            "gemm_fwd": uvqk_fwd + out_fwd,
            "gemm_bwd": uvqk_bwd + out_bwd,
            "attn_fwd": attn_fwd,
            "attn_bwd": attn_bwd,
            "elem_fwd": elem_ms,
            "elem_bwd": elem_ms,
        }
        fwd = uvqk_fwd + attn_fwd + out_fwd + elem_ms
        bwd = uvqk_bwd + attn_bwd + out_bwd + elem_ms
        return (fwd, bwd, self.estimated_activation_memory(batch_size, seq_len))

    def _results(self, batch_size: int, seq_len: int):
        key = (batch_size, seq_len)
        if self._cached is None or self._cache_key != key:
            if self._gemm_backend is None or self._sdpa_backend is None:
                raise RuntimeError(
                    "HSTULayerProfiler requires GEMM and SDPA simulation backends; "
                    "call set_simulation_backends() first."
                )
            self._cached = self._get_simulated_results(batch_size, seq_len)
            self._cache_key = key
        return self._cached

    def measured_forward_time(self, batch_size: int, seq_len: int) -> float:
        return self._results(batch_size, seq_len)[0]

    def measured_backward_time(self, batch_size: int, seq_len: int) -> float:
        return self._results(batch_size, seq_len)[1]

    def measured_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return self._results(batch_size, seq_len)[2]

    def measured_component_times(self, batch_size: int, seq_len: int) -> dict:
        """Per-role fwd/bwd ms for one layer (gemm / attn / elem)."""
        self._results(batch_size, seq_len)  # ensure computed + cached
        return dict(self._components)
