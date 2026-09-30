###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Report MiniMax-M3's MSA indexer distillation loss in the training log.

``MinimaxSparseAttention`` writes the loss into Megatron's MoE aux-loss tracker,
which already handles the pipeline reduction and the per-layer slots. But
``training_log`` passes ``track_moe_metrics`` an explicit ``track_names`` list
built from the MoE router options, and ``reduce_aux_losses_tracker_across_ranks``
iterates exactly that list -- so a key outside it is written every step, zeroed,
and never reported. This appends the key to that list.

Same mechanism as ``deepseek_v4_indexer_loss_patches``. Both predicates read
``args``, which is identical on every rank, so the ranks agree on whether the
key joins the reduction; disagreeing would hang that collective.
``force_initialize=True`` then zero-fills the key on any pipeline stage that
holds only dense-attention layers, keeping the reduced shapes aligned.
"""

from __future__ import annotations

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

# Mirrors indexer_loss.MSA_INDEXER_LOSS_NAME; not imported so this module stays
# importable without Megatron (the registry imports every *_patches.py).
_LOSS_NAME = "msa_indexer_loss"
_PATCHED_MARKER = "_msa_indexer_loss_patched"


def _default_coeff() -> float:
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )

    return MSATransformerConfig.sparse_indexer_loss_coeff


def _indexer_loss_enabled(ctx: PatchContext) -> bool:
    """MSA on, with a non-zero indexer loss coefficient."""
    args = get_args(ctx)
    if not getattr(args, "minimax_sparse_attention", False):
        return False
    coeff = getattr(args, "sparse_indexer_loss_coeff", None)
    # Unset in the preset means the config class's default applies at build time.
    if coeff is None:
        coeff = _default_coeff()
    try:
        return float(coeff) > 0.0
    except (TypeError, ValueError):
        return False


def _make_tracked_with_indexer_loss(original_fn):
    """Wrap ``track_moe_metrics`` so it also reduces and reports our key."""

    def wrapped(*args, track_names=None, **kwargs):
        # None means "every key in the tracker", which already covers ours.
        if track_names is not None and _LOSS_NAME not in track_names:
            track_names = list(track_names) + [_LOSS_NAME]
        return original_fn(*args, track_names=track_names, **kwargs)

    setattr(wrapped, _PATCHED_MARKER, True)
    return wrapped


@register_patch(
    "megatron.minimax_m3.indexer_loss_logging",
    backend="megatron",
    phase="before_train",
    description=(
        "MiniMax-M3: add the MSA indexer distillation loss to the aux-loss keys "
        "training_log reduces and reports, so it appears next to the MoE losses."
    ),
    condition=_indexer_loss_enabled,
)
def patch_minimax_m3_indexer_loss_logging(ctx: PatchContext):
    import megatron.training.training as training_module

    original_fn = getattr(training_module, "track_moe_metrics", None)
    if original_fn is None:
        log_rank_0(
            "[Patch:megatron.minimax_m3.indexer_loss_logging][SKIP] training.track_moe_metrics not found"
        )
        return
    if getattr(original_fn, _PATCHED_MARKER, False):
        return

    training_module.track_moe_metrics = _make_tracked_with_indexer_loss(original_fn)
    log_rank_0(
        f"[Patch:megatron.minimax_m3.indexer_loss_logging] '{_LOSS_NAME}' will be reported "
        "alongside the MoE aux losses"
    )
