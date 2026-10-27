###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark FLOPs reporting patch.

Megatron's :func:`megatron.training.training.num_floating_point_operations`
takes Muse Spark down its GQA + MoE branch, which already counts the sigmoid
output gate (``attention_output_gate``) and the latent MoE projections and
expert width (``moe_latent_size``). The one term it gets wrong is core
attention on the sliding-window layers: upstream charges every layer the dense
causal ``S / 2`` keys per query, while a sliding-window query attends to
``min(t + 1, W)`` keys. Three of every four layers are sliding-window (46 of
62); at ``S = 4096, W = 2048`` they see 1536.25 keys per query, at 32k 1984.

The patch keeps upstream for everything else and subtracts the difference on
the sliding-window layers.
"""

import copy
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_TAG = "[Patch:megatron.muse_spark.flops_reporting]"
_FORWARD_BACKWARD_FACTOR = 3
_FMA_FACTOR = 2


def mean_keys_per_query(seq_length: int, window: Optional[int] = None) -> float:
    """Mean keys a causal query attends to; ``window`` counts the query itself, as HF does."""
    w = seq_length if not window else min(int(window), seq_length)
    return (w * (w + 1) / 2 + (seq_length - w) * w) / seq_length


@dataclass(frozen=True)
class MuseSparkFlopsBreakdown:
    upstream: float
    swa_attention: float
    num_swa_layers: int
    keys_per_query: float

    @property
    def total(self) -> float:
        return self.upstream + self.swa_attention


def _num_swa_layers(args: Any) -> int:
    from primus.backends.megatron.core.models.muse_spark.muse_spark_ops import (
        global_attention_layer_mask,
    )

    interval = getattr(args, "muse_global_attn_interval", None) or 4
    return sum(not g for g in global_attention_layer_mask(int(args.num_layers), int(interval)))


def compute_muse_spark_flops(args: Any, batch_size: int, upstream_fn) -> MuseSparkFlopsBreakdown:
    """Per-global-batch FLOPs of a Muse Spark model, as upstream's count plus the SWA correction."""
    upstream = upstream_fn(copy.copy(args), batch_size)

    seq = int(args.seq_length)
    window = getattr(args, "muse_sliding_window", None)
    keys = mean_keys_per_query(seq, window)
    num_swa = _num_swa_layers(args) if window else 0

    # upstream's core attention, per token and layer: q_size * (S / 2) * 2 (QK^T and PV);
    # the delta is taken against the exact causal count so it vanishes when W >= S
    q_size = args.kv_channels * args.num_attention_heads
    delta_keys = keys - mean_keys_per_query(seq)
    swa_attention = (
        _FORWARD_BACKWARD_FACTOR * _FMA_FACTOR * batch_size * seq * num_swa * q_size * 2 * delta_keys
    )
    return MuseSparkFlopsBreakdown(
        upstream=upstream,
        swa_attention=swa_attention,
        num_swa_layers=num_swa,
        keys_per_query=keys,
    )


_BREAKDOWN_LOGGED = False


def _emit_breakdown(args: Any, batch_size: int, b: MuseSparkFlopsBreakdown) -> None:
    log_rank_0(
        f"{_TAG} batch_size={batch_size}, seq_length={int(args.seq_length)}, "
        f"num_layers={int(args.num_layers)}, SWA layers={b.num_swa_layers}, "
        f"SWA keys/query={b.keys_per_query:.2f} (dense causal {int(args.seq_length) / 2:.1f})"
    )
    for name, value in (("upstream", b.upstream), ("+swa_attn", b.swa_attention)):
        log_rank_0(f"{_TAG}   {name:<10s} = {value / 1.0e12:12.3f} TFLOP")
    log_rank_0(f"{_TAG}   {'TOTAL':<10s} = {b.total / 1.0e12:12.3f} TFLOP / global-batch")


def _is_muse_spark(args: Any) -> bool:
    return getattr(args, "model_type", None) == "muse_spark"


def make_muse_spark_num_floating_point_operations(original_fn):
    """Wrap ``num_floating_point_operations`` with the sliding-window correction.

    Install it only for Muse Spark. The wrapper cannot check ``args.model_type``
    per call: ``pretrain()`` rebinds it to a ``ModelType`` enum before training.
    """

    def wrapped(args, batch_size):
        breakdown = compute_muse_spark_flops(args, batch_size, original_fn)
        global _BREAKDOWN_LOGGED
        if not _BREAKDOWN_LOGGED:
            _BREAKDOWN_LOGGED = True
            _emit_breakdown(args, batch_size, breakdown)
        return breakdown.total

    wrapped.__wrapped__ = original_fn
    wrapped._muse_spark_flops_patched = True
    return wrapped


@register_patch(
    "megatron.muse_spark.flops_reporting",
    backend="megatron",
    phase="before_train",
    description=(
        "Muse Spark: charge the sliding-window layers the keys each query attends to "
        "instead of dense causal S/2."
    ),
    condition=lambda ctx: _is_muse_spark(get_args(ctx)),
)
def patch_muse_spark_flops_reporting(ctx: PatchContext):
    import megatron.training.training as training_module

    original_fn = training_module.num_floating_point_operations
    if getattr(original_fn, "_muse_spark_flops_patched", False):
        log_rank_0(f"{_TAG} num_floating_point_operations already patched, skip")
        return
    training_module.num_floating_point_operations = make_muse_spark_num_floating_point_operations(
        original_fn
    )
    log_rank_0(f"{_TAG} wrapped num_floating_point_operations with the sliding-window correction")


__all__: Sequence[str] = (
    "MuseSparkFlopsBreakdown",
    "compute_muse_spark_flops",
    "make_muse_spark_num_floating_point_operations",
    "mean_keys_per_query",
    "patch_muse_spark_flops_reporting",
)
