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
from primus.core.projection.training_config import gemm_dtype_from_config

_PEAK_HBM_BYTES_PER_MS = 4.0e9  # ~4 TB/s (MI300-class), bytes/ms
_ELEMENTWISE_HBM_FRACTION = 0.60  # SiLU/norm streaming efficiency


class HSTULayerProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self._gemm_backend = None
        self._sdpa_backend = None
        self._cached = None
        self._cache_key = None

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
    def _effective_seq(self, seq_len: int) -> int:
        fill = float(getattr(self.config.model_config, "hstu_fill_factor", 1.0) or 1.0)
        fill = min(1.0, max(0.01, fill))
        return max(1, int(round(seq_len * fill)))

    def _get_simulated_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        D, H, d_qk, d_v = self._geom()
        tp = self._tp()
        dtype = gemm_dtype_from_config(self.config.model_config)
        eff_seq = self._effective_seq(seq_len)
        tokens = batch_size * eff_seq  # jagged: only valid positions do work
        heads_per_rank = max(1, H // tp)

        fwd = 0.0
        bwd = 0.0

        # 1. Fused UVQK projection GEMM: [T, D] x [D, H*(2 d_qk + 2 d_v)]/tp
        uvqk_n = self._uvqk_out(heads_per_rank, d_qk, d_v)
        g = self._gemm_backend.simulate_gemm(tokens, uvqk_n, D, dtype=dtype)
        fwd += g.forward_time_ms
        bwd += g.backward_time_ms or (2.0 * g.forward_time_ms)

        # 2. Attention core (gated dot-product) over the effective sequence.
        s = self._sdpa_backend.simulate_sdpa(
            batch_size=batch_size,
            num_heads=heads_per_rank,
            seq_len=eff_seq,
            head_dim=d_qk,
            causal=True,
            dtype="bf16",
            head_dim_v=d_v,
        )
        fwd += s.forward_time_ms
        bwd += s.backward_time_ms or (2.0 * s.forward_time_ms)

        # 3. Output projection GEMM: [T, H*d_v]/tp x [H*d_v, D]
        o = self._gemm_backend.simulate_gemm(tokens, D, heads_per_rank * d_v, dtype=dtype)
        fwd += o.forward_time_ms
        bwd += o.backward_time_ms or (2.0 * o.forward_time_ms)

        # 4. SiLU gating (U) + norms: memory-bound elementwise over the UVQK + D
        #    activation footprint.
        elem_bytes = tokens * (uvqk_n + D) * 2  # bf16 read+op
        elem_ms = elem_bytes / (_PEAK_HBM_BYTES_PER_MS * _ELEMENTWISE_HBM_FRACTION)
        fwd += elem_ms
        bwd += elem_ms

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
