###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Make the DeepSeek-V4 indexer distillation loss visible in the training log.

``log_indexer_distill_loss`` writes the per-layer loss into Megatron's MoE
aux-loss tracker, which is the right place for it: the tracker already handles
the pipeline reduction, the per-layer slots and the TensorBoard / W&B plumbing.
What it does not do is report keys nobody asked for. ``training_log`` builds an
explicit list::

    track_names = []
    if "aux_loss" in args.moe_router_load_balancing_type:
        track_names.append("load_balancing_loss")
    ...
    track_moe_metrics(..., track_names=track_names, ...)

and ``reduce_aux_losses_tracker_across_ranks`` iterates exactly that list. A key
outside it is written every step and then silently zeroed by
``clear_aux_losses_tracker`` -- which is why the distillation loss never reached
the log even though it was being computed and tracked correctly.

This patch appends the key to that list. It is deliberately gated twice, on the
model type *and* on the coefficient, so it cannot touch any other model:

* a non-V4 job never installs it, so ``track_moe_metrics`` is the stock one;
* a V4 job with the loss off never installs it either, and in that case
  ``_log_indexer_distill_loss`` does not write the key in the first place.

Both predicates read ``args``, which is identical on every rank, so the ranks
agree on whether the key participates. That matters: the reduction is a
collective, and a key present on some ranks but not others would hang it.
``track_moe_metrics(force_initialize=True)`` then creates a zero entry on any
rank whose pipeline stage happens to hold no CSA layer, which keeps the shapes
aligned.
"""

from __future__ import annotations

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

# Imported lazily inside the predicate/patch so this module stays importable in
# environments without the V4 backend (the patch registry imports every
# ``*_patches.py`` unconditionally).
_LOSS_NAME = "indexer_distill_loss"


def _indexer_distill_enabled(ctx: PatchContext) -> bool:
    """V4 with the distillation loss actually on."""
    args = get_args(ctx)
    if getattr(args, "model_type", None) != "deepseek_v4":
        return False
    coeff = getattr(args, "v4_indexer_distill_loss_coeff", 0.0)
    try:
        return float(coeff or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


def _make_tracked_with_indexer_loss(original_fn):
    """Wrap ``track_moe_metrics`` so it also reduces and reports our key."""

    def wrapped(*args, track_names=None, **kwargs):
        # ``None`` means "every key currently in the tracker", which already
        # covers ours; only an explicit list has to be extended.
        if track_names is not None and _LOSS_NAME not in track_names:
            track_names = list(track_names) + [_LOSS_NAME]
        return original_fn(*args, track_names=track_names, **kwargs)

    wrapped._v4_indexer_loss_patched = True
    return wrapped


@register_patch(
    "megatron.deepseek_v4.indexer_distill_loss_logging",
    backend="megatron",
    phase="before_train",
    description=(
        "DeepSeek-V4: add the indexer distillation loss to the aux-loss keys "
        "training_log reduces and reports, so it appears next to the MoE "
        "losses. Only installed for V4 runs with a non-zero "
        "v4_indexer_distill_loss_coeff."
    ),
    condition=_indexer_distill_enabled,
)
def patch_indexer_distill_loss_logging(ctx: PatchContext):
    """Extend ``track_moe_metrics``'s key list with the distillation loss."""
    import megatron.training.training as training_module

    original_fn = getattr(training_module, "track_moe_metrics", None)
    if original_fn is None:
        log_rank_0(
            "[Patch:megatron.deepseek_v4.indexer_distill_loss_logging][SKIP] "
            "training.track_moe_metrics not found"
        )
        return
    if getattr(original_fn, "_v4_indexer_loss_patched", False):
        return

    training_module.track_moe_metrics = _make_tracked_with_indexer_loss(original_fn)
    log_rank_0(
        "[Patch:megatron.deepseek_v4.indexer_distill_loss_logging] "
        f"'{_LOSS_NAME}' will be reported alongside the MoE aux losses"
    )
