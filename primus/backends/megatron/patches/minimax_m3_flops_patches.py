###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax-M3 FLOPs reporting patch.

Megatron's :func:`megatron.training.training.num_floating_point_operations`
takes M3 down its GQA + MoE branch and gets three things wrong:

* GLU -- M3's ``swigluoai`` runs as Megatron's ``quick_geglu``, a gated linear
  unit (gate, up and down: three GEMMs), but upstream keys the MLP's GEMM count
  on ``args.swiglu`` alone, which M3 leaves False. Every dense MLP, routed
  expert and shared expert is counted at two GEMMs, 2/3 of what runs.
* MSA core attention -- upstream charges every layer the dense causal
  ``S / 2`` keys per query. On the layers ``sparse_attention_freq`` marks, a
  query in block ``t // B`` attends to the causally visible part of the
  ``min(t // B + 1, topk)`` blocks it keeps; at 4k that is 1504.5 keys, at 32k
  1924.5 against upstream's 16384. The other layers are dense and keep
  upstream's count.
* Indexer -- not counted at all: ``linear_index_q`` / ``linear_index_k`` (their
  input is detached, so forward plus weight gradient), the dense causal index
  scores (forward only), and the index-score gradient, which reaches only each
  query's selected slots.

The patch keeps upstream for everything else (projections, logits, MTP, the
x3 forward-backward convention, recompute left out) and corrects those three
terms: it calls upstream with ``swiglu`` set for a GLU activation, then
replaces the MSA layers' core-attention count and adds the indexer.
"""

import copy
import functools
from dataclasses import dataclass
from typing import Any, Sequence, Tuple

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_TAG = "[Patch:megatron.minimax_m3.flops_reporting]"
_FORWARD_BACKWARD_FACTOR = 3
_FMA_FACTOR = 2


@functools.lru_cache(maxsize=None)
def msa_keys_per_query(seq_length: int, block_size: int, topk_blocks: int) -> float:
    """Mean number of keys an MSA query attends to.

    A query at position t keeps ``min(t // B + 1, topk)`` blocks (the indexer
    only picks visible ones): its own block up to and including t, plus that
    many minus one fully visible blocks.
    """
    total = 0
    for qb in range(-(-seq_length // block_size)):
        rows = min(block_size, seq_length - qb * block_size)
        full_blocks = min(qb + 1, topk_blocks) - 1
        total += rows * full_blocks * block_size + rows * (rows + 1) // 2
    return total / seq_length


@dataclass(frozen=True)
class M3FlopsBreakdown:
    upstream: float
    glu: float
    msa_attention: float
    indexer: float
    num_msa_layers: int
    keys_per_query: float

    @property
    def total(self) -> float:
        return self.upstream + self.glu + self.msa_attention + self.indexer


def _msa_arg(args: Any, name: str):
    """An MSA field from ``args``, or MSATransformerConfig's default when the config left it out."""
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )

    value = getattr(args, name, None)
    return getattr(MSATransformerConfig, name) if value is None else value


def _msa_layer_pattern(args: Any) -> Tuple[int, ...]:
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        normalize_sparse_layer_pattern,
    )

    return normalize_sparse_layer_pattern(
        getattr(args, "sparse_attention_freq", None), args.num_layers, field_name="sparse_attention_freq"
    )


def compute_m3_flops(args: Any, batch_size: int, upstream_fn) -> M3FlopsBreakdown:
    """Per-global-batch FLOPs of a MiniMax-M3 model, as upstream's terms plus corrections."""
    upstream = upstream_fn(copy.copy(args), batch_size)

    glu = 0.0
    if getattr(args, "quick_geglu", False) and not getattr(args, "swiglu", False):
        glu_args = copy.copy(args)
        glu_args.swiglu = True
        glu = upstream_fn(glu_args, batch_size) - upstream

    seq = int(args.seq_length)
    tokens = batch_size * seq
    num_msa_layers = sum(_msa_layer_pattern(args))
    topk = int(_msa_arg(args, "sparse_topk_blocks"))
    keys = msa_keys_per_query(seq, int(_msa_arg(args, "sparse_block_size")), topk)

    # upstream's core attention, per token and layer: q_size * (S / 2) * 2 (QK^T and PV)
    q_size = args.kv_channels * args.num_attention_heads
    msa_attention = (
        _FORWARD_BACKWARD_FACTOR * _FMA_FACTOR * tokens * num_msa_layers * q_size * 2 * (keys - seq / 2)
    )

    index_heads = int(_msa_arg(args, "sparse_num_index_heads"))
    index_dim = int(_msa_arg(args, "sparse_index_dim"))
    trained = float(_msa_arg(args, "sparse_indexer_loss_coeff")) > 0.0
    projection = args.hidden_size * (index_heads * index_dim + index_dim)
    scores = index_heads * index_dim * (seq + 1) / 2
    score_grad = 2 * index_heads * index_dim * topk  # dq and dk, selected slots only
    per_token = projection * (2 if trained else 1) + scores + (score_grad if trained else 0)
    indexer = _FMA_FACTOR * tokens * num_msa_layers * per_token

    return M3FlopsBreakdown(
        upstream=upstream,
        glu=glu,
        msa_attention=msa_attention,
        indexer=indexer,
        num_msa_layers=num_msa_layers,
        keys_per_query=keys,
    )


_BREAKDOWN_LOGGED = False


def _emit_breakdown(args: Any, batch_size: int, b: M3FlopsBreakdown) -> None:
    log_rank_0(
        f"{_TAG} batch_size={batch_size}, seq_length={int(args.seq_length)}, "
        f"num_layers={int(args.num_layers)}, MSA layers={b.num_msa_layers}, "
        f"MSA keys/query={b.keys_per_query:.1f} (dense causal {int(args.seq_length) / 2:.1f})"
    )
    for name, value in (
        ("upstream", b.upstream),
        ("+glu", b.glu),
        ("+msa_attn", b.msa_attention),
        ("+indexer", b.indexer),
    ):
        log_rank_0(f"{_TAG}   {name:<10s} = {value / 1.0e12:12.3f} TFLOP")
    log_rank_0(f"{_TAG}   {'TOTAL':<10s} = {b.total / 1.0e12:12.3f} TFLOP / global-batch")


def make_m3_num_floating_point_operations(original_fn):
    """Wrap ``num_floating_point_operations``; non-M3 args fall through unchanged."""

    def wrapped(args, batch_size):
        if not getattr(args, "minimax_sparse_attention", False):
            return original_fn(args, batch_size)
        breakdown = compute_m3_flops(args, batch_size, original_fn)
        global _BREAKDOWN_LOGGED
        if not _BREAKDOWN_LOGGED:
            _BREAKDOWN_LOGGED = True
            _emit_breakdown(args, batch_size, breakdown)
        return breakdown.total

    wrapped.__wrapped__ = original_fn
    wrapped._m3_flops_patched = True
    return wrapped


@register_patch(
    "megatron.minimax_m3.flops_reporting",
    backend="megatron",
    phase="before_train",
    description=(
        "MiniMax-M3: count its GLU MLPs as three GEMMs, charge MSA layers the keys each query "
        "actually attends to instead of dense causal S/2, and add the indexer."
    ),
    condition=lambda ctx: bool(getattr(get_args(ctx), "minimax_sparse_attention", False)),
)
def patch_minimax_m3_flops_reporting(ctx: PatchContext):
    import megatron.training.training as training_module

    original_fn = training_module.num_floating_point_operations
    if getattr(original_fn, "_m3_flops_patched", False):
        log_rank_0(f"{_TAG} num_floating_point_operations already patched, skip")
        return
    training_module.num_floating_point_operations = make_m3_num_floating_point_operations(original_fn)
    log_rank_0(f"{_TAG} wrapped num_floating_point_operations with the MiniMax-M3 corrections")


__all__: Sequence[str] = (
    "M3FlopsBreakdown",
    "compute_m3_flops",
    "make_m3_num_floating_point_operations",
    "msa_keys_per_query",
    "patch_minimax_m3_flops_reporting",
)
