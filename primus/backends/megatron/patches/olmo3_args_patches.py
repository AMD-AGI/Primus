###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 (``model_type: olmo3``) argument patches.

``megatron.olmo3.args`` (phase ``before_train``):

* Window. ``olmo3_sliding_window`` is W (HF ``sliding_window``, the preset sets 4096) and is
  required; ``window_size`` next to it may be W (an int, the preset's form), the Megatron / TE
  pair ``(W - 1, 0)`` or unset (alone it is refused). The patch sets ``window_size = (W - 1, 0)``.
  Rules: :func:`olmo3_validation.olmo3_window_from_settings`.
* ``window_attn_skip_freq`` is required: an int ``2 <= k <= num_layers`` or a 0/1 list of
  ``num_layers`` entries with at least one 0 and one 1; CLI strings (``"4"``,
  ``"([1]*3+[0])*8"``) are parsed as Megatron parses them.
* A single-shard legacy checkpoint (what the native converter writes) loaded with tensor or
  pipeline parallel size > 1 is refused (:func:`olmo3_validation.check_olmo3_checkpoint_layout`).
* Settings the OLMo-3 model does not implement are refused with the shared
  :data:`olmo3_validation.OLMO3_UNSUPPORTED` table (the same table the config and the builder use).
"""

from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
    check_olmo3_checkpoint_layout,
    check_olmo3_unsupported,
    olmo3_window_from_settings,
    parse_olmo3_skip_freq,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

__all__ = ["is_olmo3_model_type", "patch_olmo3_args"]


def is_olmo3_model_type(ctx: PatchContext) -> bool:
    """Patch condition shared by the OLMo-3 patches: ``model_type`` is ``olmo3``."""
    return getattr(get_args(ctx), "model_type", None) == "olmo3"


@register_patch(
    "megatron.olmo3.args",
    backend="megatron",
    phase="before_train",
    description="OLMo-3: window_size = (W - 1, 0) for olmo3_sliding_window W; refuse unsupported settings.",
    condition=is_olmo3_model_type,
)
def patch_olmo3_args(ctx: PatchContext):
    args = get_args(ctx)
    before = getattr(args, "window_size", None)
    ws, w = olmo3_window_from_settings(before, getattr(args, "olmo3_sliding_window", None))
    skip = parse_olmo3_skip_freq(
        getattr(args, "window_attn_skip_freq", None), getattr(args, "num_layers", None)
    )
    check_olmo3_unsupported(args)
    check_olmo3_checkpoint_layout(args)
    args.window_size = ws
    args.olmo3_sliding_window = w
    args.window_attn_skip_freq = skip
    log_rank_0(
        f"[Patch:megatron.olmo3.args] sliding_window={w}: window_size {before!r} -> {ws} (Megatron/TE) "
        f"window_attn_skip_freq={skip}"
    )
