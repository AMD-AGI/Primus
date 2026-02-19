###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Abstract base classes for GEMM and SDPA simulation backends.

These backends provide simulated (analytical/model-based) timing for GEMM and
SDPA operations, allowing performance projection without running actual GPU
kernels.  Two concrete GEMM backends are shipped:

- **Origami** (open-source, default) – ``origami_backend.py``

An SDPA simulation backend is provided in ``sdpa_simulator.py``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SimulationResult:
    """Result from a simulation backend."""

    # Predicted time in milliseconds
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0

    # Optional: predicted TFLOPS / bandwidth
    tflops: Optional[float] = None
    bandwidth_gbps: Optional[float] = None

    # Optional: extra metadata from the backend
    metadata: Dict[str, Any] = field(default_factory=dict)


class GEMMSimulationBackend(ABC):
    """Abstract interface for GEMM simulation backends."""

    @abstractmethod
    def name(self) -> str:
        """Return human-readable backend name."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can be used in the current environment."""
        ...

    @property
    def hbm_bandwidth_gbps(self) -> Optional[float]:
        """Peak HBM bandwidth in GB/s for the target GPU, or *None* if unknown.

        Concrete backends should override this when the target architecture is
        known so that downstream profilers (e.g. MoE non-GEMM overhead) can
        derive memory-bandwidth estimates from the actual hardware rather than
        relying on hardcoded absolute numbers.
        """
        return None

    @abstractmethod
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
        """
        Simulate a single GEMM (or batched GEMM) and return predicted timing.

        Args:
            m, n, k: Matrix dimensions  (C = A @ B,  A:[M,K]  B:[K,N]  C:[M,N])
            dtype: Data type string ("bf16", "fp16", "fp8", "fp32")
            trans_a: Whether A is transposed
            trans_b: Whether B is transposed
            batch: Number of independent GEMMs with the **same** (M, N, K) to
                run as a **batched** GEMM.  For MoE experts this is used as an
                approximation of grouped GEMM under the assumption of uniform
                token distribution across experts (so every sub-problem has the
                same M = tokens_per_expert).  Defaults to 1 (single GEMM).

                NOTE: Origami's ``problem.batch`` models batched GEMM, not true
                grouped GEMM (where each sub-problem can have a different M).
                This is an acceptable approximation when token distribution is
                assumed uniform.  If Origami adds native grouped-GEMM support
                in the future, this should be updated.

        Returns:
            SimulationResult with forward_time_ms populated.
        """
        ...

    def simulate_mlp_gemms(
        self,
        batch_tokens: int,
        hidden_size: int,
        ffn_hidden_size: int,
        dtype: str = "bf16",
        swiglu: bool = False,
        num_experts: int = 1,
    ) -> SimulationResult:
        """
        Simulate the GEMM operations in a dense MLP (gate/up/down projections).

        Default implementation calls ``simulate_gemm`` for each projection and
        sums the times.  Backends may override for better accuracy.

        When ``num_experts > 1``, each GEMM is simulated as a **batched** GEMM
        (``batch = num_experts``) with per-expert token count, which serves as
        an approximation of grouped GEMM under uniform token distribution.
        See ``simulate_gemm`` docstring for caveats.

        Args:
            batch_tokens: Number of tokens per expert
                (batch_size * seq_len / TP / CP for dense;
                 topk_tokens / num_local_experts for MoE routed experts)
            hidden_size: Model hidden dimension
            ffn_hidden_size: FFN intermediate dimension
            dtype: Data type string
            swiglu: Whether SwiGLU activation is used (3 projections vs 2)
            num_experts: Number of local experts.  When > 1, each projection
                GEMM uses ``batch = num_experts`` as a batched-GEMM
                approximation of grouped GEMM.  Defaults to 1 (dense MLP).

        Returns:
            SimulationResult with forward_time_ms and backward_time_ms.
        """
        fwd_time = 0.0
        bwd_time = 0.0
        # Use batched GEMM (batch=num_experts) as approximation of grouped GEMM.
        # Valid under uniform token distribution (all experts get the same M).
        # TODO: switch to native grouped-GEMM simulation if/when Origami supports it.
        b = num_experts

        if swiglu:
            # Gate projection fwd:  [tokens, hidden] x [hidden, ffn] -> [tokens, ffn]
            gate_fwd = self.simulate_gemm(batch_tokens, ffn_hidden_size, hidden_size, dtype, batch=b)
            # Up projection fwd:  same shape as gate
            up_fwd = self.simulate_gemm(batch_tokens, ffn_hidden_size, hidden_size, dtype, batch=b)
            # Down projection fwd:  [tokens, ffn] x [ffn, hidden] -> [tokens, hidden]
            down_fwd = self.simulate_gemm(batch_tokens, hidden_size, ffn_hidden_size, dtype, batch=b)

            fwd_time = gate_fwd.forward_time_ms + up_fwd.forward_time_ms + down_fwd.forward_time_ms

            # Backward: simulate actual dgrad + wgrad GEMMs per projection
            # Gate dgrad:  [tokens, ffn] x [ffn, hidden] -> [tokens, hidden]
            gate_dgrad = self.simulate_gemm(batch_tokens, hidden_size, ffn_hidden_size, dtype, batch=b)
            # Gate wgrad:  [hidden, tokens] x [tokens, ffn] -> [hidden, ffn]
            gate_wgrad = self.simulate_gemm(hidden_size, ffn_hidden_size, batch_tokens, dtype, batch=b)
            # Up dgrad + wgrad: same shapes as gate
            up_dgrad = self.simulate_gemm(batch_tokens, hidden_size, ffn_hidden_size, dtype, batch=b)
            up_wgrad = self.simulate_gemm(hidden_size, ffn_hidden_size, batch_tokens, dtype, batch=b)
            # Down dgrad:  [tokens, hidden] x [hidden, ffn] -> [tokens, ffn]
            down_dgrad = self.simulate_gemm(batch_tokens, ffn_hidden_size, hidden_size, dtype, batch=b)
            # Down wgrad:  [ffn, tokens] x [tokens, hidden] -> [ffn, hidden]
            down_wgrad = self.simulate_gemm(ffn_hidden_size, hidden_size, batch_tokens, dtype, batch=b)

            bwd_time = (
                gate_dgrad.forward_time_ms + gate_wgrad.forward_time_ms
                + up_dgrad.forward_time_ms + up_wgrad.forward_time_ms
                + down_dgrad.forward_time_ms + down_wgrad.forward_time_ms
            )
        else:
            # Up projection fwd:  [tokens, hidden] x [hidden, ffn] -> [tokens, ffn]
            up_fwd = self.simulate_gemm(batch_tokens, ffn_hidden_size, hidden_size, dtype, batch=b)
            # Down projection fwd:  [tokens, ffn] x [ffn, hidden] -> [tokens, hidden]
            down_fwd = self.simulate_gemm(batch_tokens, hidden_size, ffn_hidden_size, dtype, batch=b)

            fwd_time = up_fwd.forward_time_ms + down_fwd.forward_time_ms

            # Backward: simulate actual dgrad + wgrad GEMMs per projection
            # Up dgrad:  [tokens, ffn] x [ffn, hidden] -> [tokens, hidden]
            up_dgrad = self.simulate_gemm(batch_tokens, hidden_size, ffn_hidden_size, dtype, batch=b)
            # Up wgrad:  [hidden, tokens] x [tokens, ffn] -> [hidden, ffn]
            up_wgrad = self.simulate_gemm(hidden_size, ffn_hidden_size, batch_tokens, dtype, batch=b)
            # Down dgrad:  [tokens, hidden] x [hidden, ffn] -> [tokens, ffn]
            down_dgrad = self.simulate_gemm(batch_tokens, ffn_hidden_size, hidden_size, dtype, batch=b)
            # Down wgrad:  [ffn, tokens] x [tokens, hidden] -> [ffn, hidden]
            down_wgrad = self.simulate_gemm(ffn_hidden_size, hidden_size, batch_tokens, dtype, batch=b)

            bwd_time = (
                up_dgrad.forward_time_ms + up_wgrad.forward_time_ms
                + down_dgrad.forward_time_ms + down_wgrad.forward_time_ms
            )

        return SimulationResult(forward_time_ms=fwd_time, backward_time_ms=bwd_time)

    def simulate_attention_gemms(
        self,
        batch_tokens: int,
        hidden_size: int,
        num_attention_heads: int,
        kv_channels: int,
        num_query_groups: int,
        dtype: str = "bf16",
    ) -> SimulationResult:
        """
        Simulate the linear projection GEMMs in the attention block
        (QKV projections + output projection).  Does NOT include the SDPA
        computation itself – use SDPASimulationBackend for that.

        Default implementation calls ``simulate_gemm`` for Q, K, V, O projections.

        Returns:
            SimulationResult with forward_time_ms and backward_time_ms.
        """
        fwd_time = 0.0
        bwd_time = 0.0

        # Q projection fwd: [tokens, hidden] x [hidden, heads*kv_channels]
        q_out = num_attention_heads * kv_channels
        q_fwd = self.simulate_gemm(batch_tokens, q_out, hidden_size, dtype)
        fwd_time += q_fwd.forward_time_ms

        # K projection fwd: [tokens, hidden] x [hidden, num_query_groups*kv_channels]
        k_out = num_query_groups * kv_channels
        k_fwd = self.simulate_gemm(batch_tokens, k_out, hidden_size, dtype)
        fwd_time += k_fwd.forward_time_ms

        # V projection fwd: same shape as K
        v_fwd = self.simulate_gemm(batch_tokens, k_out, hidden_size, dtype)
        fwd_time += v_fwd.forward_time_ms

        # Output projection fwd: [tokens, heads*kv_channels] x [heads*kv_channels, hidden]
        o_fwd = self.simulate_gemm(batch_tokens, hidden_size, q_out, dtype)
        fwd_time += o_fwd.forward_time_ms

        # Backward: simulate actual dgrad + wgrad GEMMs per projection
        # Q dgrad:  [tokens, q_out] x [q_out, hidden] -> [tokens, hidden]
        q_dgrad = self.simulate_gemm(batch_tokens, hidden_size, q_out, dtype)
        # Q wgrad:  [hidden, tokens] x [tokens, q_out] -> [hidden, q_out]
        q_wgrad = self.simulate_gemm(hidden_size, q_out, batch_tokens, dtype)
        bwd_time += q_dgrad.forward_time_ms + q_wgrad.forward_time_ms

        # K dgrad:  [tokens, k_out] x [k_out, hidden] -> [tokens, hidden]
        k_dgrad = self.simulate_gemm(batch_tokens, hidden_size, k_out, dtype)
        # K wgrad:  [hidden, tokens] x [tokens, k_out] -> [hidden, k_out]
        k_wgrad = self.simulate_gemm(hidden_size, k_out, batch_tokens, dtype)
        bwd_time += k_dgrad.forward_time_ms + k_wgrad.forward_time_ms

        # V dgrad + wgrad: same shapes as K
        v_dgrad = self.simulate_gemm(batch_tokens, hidden_size, k_out, dtype)
        v_wgrad = self.simulate_gemm(hidden_size, k_out, batch_tokens, dtype)
        bwd_time += v_dgrad.forward_time_ms + v_wgrad.forward_time_ms

        # O dgrad:  [tokens, hidden] x [hidden, q_out] -> [tokens, q_out]
        o_dgrad = self.simulate_gemm(batch_tokens, q_out, hidden_size, dtype)
        # O wgrad:  [q_out, tokens] x [tokens, hidden] -> [q_out, hidden]
        o_wgrad = self.simulate_gemm(q_out, hidden_size, batch_tokens, dtype)
        bwd_time += o_dgrad.forward_time_ms + o_wgrad.forward_time_ms

        return SimulationResult(forward_time_ms=fwd_time, backward_time_ms=bwd_time)


class SDPASimulationBackend(ABC):
    """Abstract interface for Scaled Dot-Product Attention simulation."""

    @abstractmethod
    def name(self) -> str:
        """Return human-readable backend name."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can be used in the current environment."""
        ...

    @abstractmethod
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
        Simulate a Scaled Dot-Product Attention operation.

        Args:
            batch_size: Batch size
            num_heads: Number of query attention heads (per TP rank)
            seq_len: Query sequence length (per CP rank)
            head_dim: Head dimension used in the Q·Kᵀ dot-product.  For
                standard MHA/GQA this equals ``kv_channels``; for MLA it
                equals ``qk_head_dim + qk_pos_emb_head_dim`` (e.g. 192).
            causal: Whether causal masking is used
            dtype: Data type string
            seq_len_kv: Key/Value sequence length.  Defaults to ``seq_len``
                (self-attention).
            num_heads_kv: Number of KV heads.  Defaults to ``num_heads``
                (MHA).  Set lower for GQA / MQA.
            head_dim_v: Value head dimension used in the P·V product and
                for sizing the output O.  Defaults to ``head_dim`` (standard
                attention where Q/K/V all share the same dimension).  For
                MLA this should be set to ``v_head_dim`` (e.g. 128).

        Returns:
            SimulationResult with forward_time_ms and backward_time_ms.
        """
        ...
