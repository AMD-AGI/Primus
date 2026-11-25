###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Pluggable analytical op models for the simulate path.

The built-in profilers (attention, dense/MoE MLP, the transformer layer) cover
the architectures Primus already knows.  A new model often ships one op they do
not -- a linear-attention layer, a new gate, a new residual mixer -- and before
this package the projection silently priced it as whatever it resembled.

An :class:`OpModel` is the unit a new op is added as.  It declares which config
keys select and parameterise it (so the coverage check can tell modeled from
unmodeled), which slot of the layer it fills, and returns an :class:`OpCost`
built from the same pricing primitives the built-in profilers use:

* projections and other dense matmuls through the GEMM backend (Origami);
* softmax-attention cores through the FAv3 SDPA simulator, so a new attention
  variant keeps the tile-level, 1-CU wave-quantised cost model;
* any other fused kernel through :meth:`OpContext.fused_tiles`, which prices a
  per-workgroup tile program on one CU and multiplies by the wave count -- the
  same construction the SDPA simulator uses for FAv3;
* elementwise / gather / state traffic as HBM bytes at a sustained fraction of
  peak bandwidth.

Ops live in :mod:`primus.core.projection.op_models.generated` (written by the
tuning agent's op bring-up stage) or in any directory on
``PRIMUS_OP_MODEL_DIRS``, and are discovered on first use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple

from primus.core.projection.simulation_backends.base import (
    resolve_hbm_bytes_per_ms,
    resolve_peak_tflops,
)
from primus.core.projection.training_config import gemm_dtype_from_config

# Slots an op can fill inside one transformer layer.
#   attention        replaces the attention block on ``layer_fraction`` of layers
#   attention_extra  adds to the attention block on ``layer_fraction`` of layers
#   mlp              replaces the MLP block on ``layer_fraction`` of layers
#   layer_extra      adds a per-layer term outside attention and MLP
SLOT_ATTENTION = "attention"
SLOT_ATTENTION_EXTRA = "attention_extra"
SLOT_MLP = "mlp"
SLOT_LAYER_EXTRA = "layer_extra"
SLOTS = (SLOT_ATTENTION, SLOT_ATTENTION_EXTRA, SLOT_MLP, SLOT_LAYER_EXTRA)

# Same sustained-HBM fraction the SwiGLU activation and the layer norm /
# residual model use, so an op's elementwise traffic is priced like theirs.
DEFAULT_STREAM_EFFICIENCY = 0.566
_FALLBACK_HBM_GBPS = 5300.0
_FALLBACK_PEAK_TFLOPS = {"bf16": 1307.0, "fp16": 1307.0, "fp8": 2614.0, "mx8": 2614.0}
_FALLBACK_N_CU = 304

_INACTIVE = (None, False, 0, 0.0, "", "none", "None", "null")


def cfg_value(model_config, key: str, default: Any = None) -> Any:
    """Read *key* from the model config, then from the args it was built from.

    ``ModelConfig`` is a fixed dataclass, so keys a new op introduces are not
    fields on it; :func:`convert_primus_config_to_projection_config` keeps them
    in ``model_config.extra_args`` instead of dropping them.
    """
    val = getattr(model_config, key, None)
    if val is not None:
        return val
    extra = getattr(model_config, "extra_args", None) or {}
    val = extra.get(key)
    return default if val is None else val


def is_active(value: Any) -> bool:
    """True when a config value switches something on."""
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    try:
        return value not in _INACTIVE
    except TypeError:
        return True


def full_num_layers(model_config) -> int:
    """Layer count of the full model, before the simulate path trims the stack."""
    n = cfg_value(model_config, "projection_full_num_layers", None)
    if n:
        return int(n)
    return int(getattr(model_config, "num_layers", 0) or 0)


def interleave_fraction(model_config, every: int) -> float:
    """Share of layers that are *not* the every-``every``-th layer.

    The Qwen3-Next / Qwen3.5 convention: ``linear_attention_freq: 4`` makes
    every 4th layer full attention and the other three linear, so the linear
    share is ``(n - n // 4) / n``.
    """
    n = full_num_layers(model_config)
    every = int(every or 0)
    if n <= 0 or every <= 1:
        return 1.0 if every == 1 else 0.0
    n_full = max(1, n // every)
    return (n - n_full) / n


@dataclass
class OpCost:
    """Per-layer, per-rank cost of one op (milliseconds, FLOPs, HBM bytes)."""

    fwd_ms: float
    bwd_ms: float
    fwd_flops: float
    bwd_flops: float
    fwd_bytes: float
    bwd_bytes: float
    # Optional memory-model terms. ``None`` keeps the built-in estimate.
    activation_bytes: Optional[float] = None
    num_params: Optional[int] = None
    # name -> (fwd_ms, bwd_ms); reported, and checked to sum to the totals.
    breakdown: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class OpContext:
    """Shapes, parallelism and pricing primitives for one op evaluation."""

    config: Any  # TrainingConfig
    batch_size: int
    seq_len: int
    gemm_backend: Any = None
    sdpa_backend: Any = None
    layer_kind: str = "dense"  # "dense" | "moe"

    # ---- config / parallelism ----
    @property
    def mc(self):
        return self.config.model_config

    @property
    def mp(self):
        return self.config.model_parallel_config

    @property
    def tp(self) -> int:
        return max(1, int(getattr(self.mp, "tensor_model_parallel_size", 1) or 1))

    @property
    def cp(self) -> int:
        return max(1, int(getattr(self.mp, "context_model_parallel_size", 1) or 1))

    @property
    def tokens(self) -> int:
        """GEMM ``M`` per rank, with TP folded into it as the attention profiler does."""
        return max(1, self.batch_size * self.seq_len // self.tp // self.cp)

    @property
    def seq_per_cp(self) -> int:
        return max(1, self.seq_len // self.cp)

    def heads_per_rank(self, heads: int) -> int:
        return max(1, int(heads) // self.tp)

    def value(self, key: str, default: Any = None) -> Any:
        return cfg_value(self.mc, key, default)

    @property
    def gemm_dtype(self) -> str:
        return gemm_dtype_from_config(self.mc)

    # ---- hardware ----
    @property
    def hbm_bytes_per_ms(self) -> float:
        return resolve_hbm_bytes_per_ms(self.gemm_backend, self.sdpa_backend, _FALLBACK_HBM_GBPS)

    def peak_flops(self, dtype: Optional[str] = None) -> float:
        dt = dtype or self.gemm_dtype
        tflops = resolve_peak_tflops(self.gemm_backend, self.sdpa_backend, "fp8" if dt == "mx8" else dt)
        return float(tflops or _FALLBACK_PEAK_TFLOPS.get(dt, 1307.0)) * 1e12

    @property
    def n_cu(self) -> int:
        hw = getattr(self.sdpa_backend, "_hw", None)
        return int(getattr(hw, "n_cu", 0) or _FALLBACK_N_CU)

    def roofline_ms(self, flops: float, nbytes: float, dtype: Optional[str] = None) -> float:
        """Lower bound: neither peak compute nor peak HBM can be beaten."""
        return max(flops / self.peak_flops(dtype) * 1e3, nbytes / self.hbm_bytes_per_ms)

    # ---- pricing primitives ----
    def gemm(self, m: int, n: int, k: int, dtype: Optional[str] = None, batch: int = 1) -> float:
        """One (batched) GEMM ``[m,k] x [k,n]``, in ms."""
        m, n, k, batch = max(1, int(m)), max(1, int(n)), max(1, int(k)), max(1, int(batch))
        dt = dtype or self.gemm_dtype
        if self.gemm_backend is not None:
            return self.gemm_backend.simulate_gemm(m, n, k, dt, batch=batch).forward_time_ms
        bpe = 1 if dt in ("fp8", "mx8") else 2
        flops = 2.0 * m * n * k * batch
        nbytes = (m * k + k * n + m * n) * bpe * batch
        return self.roofline_ms(flops, nbytes, dt)

    def linear(self, m: int, n: int, k: int, dtype: Optional[str] = None) -> Tuple[float, float]:
        """A linear layer ``[m,k] x [k,n]``: forward, and dgrad + wgrad backward."""
        fwd = self.gemm(m, n, k, dtype)
        bwd = self.gemm(m, k, n, dtype) + self.gemm(k, n, m, dtype)
        return fwd, bwd

    def sdpa(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        head_dim: int,
        *,
        seq_len_kv: Optional[int] = None,
        head_dim_v: Optional[int] = None,
        num_heads_kv: Optional[int] = None,
        causal: bool = True,
    ) -> Tuple[float, float]:
        """Softmax-attention core on the FAv3 tile-level simulator, ``(fwd, bwd)`` ms."""
        if self.sdpa_backend is not None:
            r = self.sdpa_backend.simulate_sdpa(
                batch_size=int(batch),
                num_heads=int(heads),
                seq_len=int(seq_len),
                head_dim=int(head_dim),
                causal=causal,
                dtype="bf16",
                seq_len_kv=seq_len_kv,
                num_heads_kv=num_heads_kv,
                head_dim_v=head_dim_v,
            )
            return r.forward_time_ms, r.backward_time_ms
        s_k = seq_len_kv or seq_len
        d_v = head_dim_v or head_dim
        cf = 0.5 if causal else 1.0
        fwd_flops = 2.0 * batch * heads * seq_len * s_k * (head_dim + d_v) * cf
        return self.roofline_ms(fwd_flops, 0.0, "bf16"), self.roofline_ms(2.5 * fwd_flops, 0.0, "bf16")

    def fused_tiles(
        self, per_workgroup_gemms: Sequence[Tuple[int, int, int]], n_workgroups: int, dtype: str = "bf16"
    ) -> float:
        """A fused kernel as a per-workgroup GEMM program, priced on one CU.

        ``per_workgroup_gemms`` lists the ``(m, n, k)`` matmuls one workgroup
        runs back to back; the kernel time is their 1-CU sum over
        ``ceil(n_workgroups / N_CU)`` waves -- the FAv3 construction, for any
        fused kernel (chunked recurrences, tiled scans, gated scores).  A loop
        a workgroup runs over the sequence is one matmul with the loop folded
        into a dimension (``(n_chunks * C, n, k)``), as FAv3 folds its KV sweep.
        """
        tiles = [(max(1, int(m)), max(1, int(n)), max(1, int(k))) for m, n, k in per_workgroup_gemms]
        n_wg = max(1, int(n_workgroups))
        sim = getattr(self.sdpa_backend, "simulate_fused_tiles", None)
        if callable(sim):
            return sim(tiles, n_wg, dtype)
        per_wg_flops = sum(2.0 * m * n * k for m, n, k in tiles)
        waves = math.ceil(n_wg / self.n_cu)
        return self.roofline_ms(per_wg_flops * self.n_cu * waves, 0.0, dtype)

    def stream_ms(self, nbytes: float, efficiency: float = DEFAULT_STREAM_EFFICIENCY) -> float:
        """Memory-bound traffic of ``nbytes`` at a sustained fraction of peak HBM."""
        return float(nbytes) / (efficiency * self.hbm_bytes_per_ms)


class OpModel:
    """Analytical model of one op the built-in profilers do not cover.

    Subclasses set the class attributes, implement :meth:`cost`, and are
    registered with :func:`register_op_model`.  Keep :meth:`cost` a function of
    the context alone: it runs inside the projection's per-layer simulation.
    """

    #: Unique snake_case id, e.g. ``"gated_delta_net"``.
    name: ClassVar[str] = ""
    #: One of :data:`SLOTS`.
    slot: ClassVar[str] = SLOT_ATTENTION
    #: Selector key -> config values this op models, e.g.
    #: ``{"experimental_attention_variant": ("gated_delta_net",)}``.
    claims: ClassVar[Dict[str, Tuple[Any, ...]]] = {}
    #: Every config key :meth:`cost` reads beyond the built-in ModelConfig fields.
    consumes: ClassVar[Tuple[str, ...]] = ()
    #: Keys of this op deliberately left unpriced -> why (e.g. a numerics-only
    #: flag).  They count as covered, so state the reason the cost ignores them.
    ignores: ClassVar[Dict[str, str]] = {}
    #: Paper / model card / kernel source the equations come from.
    references: ClassVar[Tuple[str, ...]] = ()

    @classmethod
    def matches(cls, model_config) -> bool:
        """Whether this op is present in *model_config*. Default: any claim holds."""
        for key, values in cls.claims.items():
            val = cfg_value(model_config, key)
            if values == ("*",):
                if is_active(val):
                    return True
                continue
            norm = {str(v).lower() for v in values}
            if val is not None and str(val).lower() in norm:
                return True
        return False

    def layer_fraction(self, model_config) -> float:
        """Share of the model's layers this op runs on."""
        return 1.0

    def cost(self, ctx: OpContext) -> OpCost:
        raise NotImplementedError

    def reference_flops(self, ctx: OpContext) -> Optional[Tuple[float, float]]:
        """Closed-form ``(fwd, bwd)`` FLOPs straight from the paper's equations.

        Written independently of :meth:`cost` so the validator can cross-check
        the kernel decomposition against the math. ``None`` skips the check.
        """
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name}, slot={self.slot})"
