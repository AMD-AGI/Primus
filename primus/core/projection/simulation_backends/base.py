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

    @abstractmethod
    def simulate_gemm(
        self,
        m: int,
        n: int,
        k: int,
        dtype: str = "bf16",
        trans_a: bool = False,
        trans_b: bool = False,
    ) -> SimulationResult:
        """
        Simulate a single GEMM operation and return predicted timing.

        Args:
            m, n, k: Matrix dimensions  (C = A @ B,  A:[M,K]  B:[K,N]  C:[M,N])
            dtype: Data type string ("bf16", "fp16", "fp8", "fp32")
            trans_a: Whether A is transposed
            trans_b: Whether B is transposed

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
    ) -> SimulationResult:
        """
        Simulate the GEMM operations in a dense MLP (gate/up/down projections).

        Default implementation calls ``simulate_gemm`` for each projection and
        sums the times.  Backends may override for better accuracy.

        Args:
            batch_tokens: Number of tokens (batch_size * seq_len / TP / CP)
            hidden_size: Model hidden dimension
            ffn_hidden_size: FFN intermediate dimension
            dtype: Data type string
            swiglu: Whether SwiGLU activation is used (3 projections vs 2)

        Returns:
            SimulationResult with forward_time_ms and backward_time_ms.
        """
        fwd_time = 0.0
        bwd_time = 0.0

        if swiglu:
            # Gate projection:  [tokens, hidden] x [hidden, ffn] -> [tokens, ffn]
            gate_res = self.simulate_gemm(batch_tokens, ffn_hidden_size, hidden_size, dtype)
            # Up projection:  same shape
            up_res = self.simulate_gemm(batch_tokens, ffn_hidden_size, hidden_size, dtype)
            # Down projection:  [tokens, ffn] x [ffn, hidden] -> [tokens, hidden]
            down_res = self.simulate_gemm(batch_tokens, hidden_size, ffn_hidden_size, dtype)

            fwd_time = gate_res.forward_time_ms + up_res.forward_time_ms + down_res.forward_time_ms
            # Backward is approximately 2x forward (dgrad + wgrad per projection)
            bwd_time = fwd_time * 2.0
        else:
            # Up projection
            up_res = self.simulate_gemm(batch_tokens, ffn_hidden_size, hidden_size, dtype)
            # Down projection
            down_res = self.simulate_gemm(batch_tokens, hidden_size, ffn_hidden_size, dtype)

            fwd_time = up_res.forward_time_ms + down_res.forward_time_ms
            bwd_time = fwd_time * 2.0

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

        # Q projection: [tokens, hidden] x [hidden, heads*kv_channels]
        q_out = num_attention_heads * kv_channels
        q_res = self.simulate_gemm(batch_tokens, q_out, hidden_size, dtype)
        fwd_time += q_res.forward_time_ms

        # K projection: [tokens, hidden] x [hidden, num_query_groups*kv_channels]
        k_out = num_query_groups * kv_channels
        k_res = self.simulate_gemm(batch_tokens, k_out, hidden_size, dtype)
        fwd_time += k_res.forward_time_ms

        # V projection: same shape as K
        v_res = self.simulate_gemm(batch_tokens, k_out, hidden_size, dtype)
        fwd_time += v_res.forward_time_ms

        # Output projection: [tokens, heads*kv_channels] x [heads*kv_channels, hidden]
        o_res = self.simulate_gemm(batch_tokens, hidden_size, q_out, dtype)
        fwd_time += o_res.forward_time_ms

        bwd_time = fwd_time * 2.0  # dgrad + wgrad

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
    ) -> SimulationResult:
        """
        Simulate a Scaled Dot-Product Attention operation.

        Args:
            batch_size: Batch size
            num_heads: Number of query attention heads (per TP rank)
            seq_len: Query sequence length (per CP rank)
            head_dim: Head dimension (kv_channels)
            causal: Whether causal masking is used
            dtype: Data type string
            seq_len_kv: Key/Value sequence length.  Defaults to ``seq_len``
                (self-attention).
            num_heads_kv: Number of KV heads.  Defaults to ``num_heads``
                (MHA).  Set lower for GQA / MQA.

        Returns:
            SimulationResult with forward_time_ms and backward_time_ms.
        """
        ...
