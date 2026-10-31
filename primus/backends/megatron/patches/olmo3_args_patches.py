###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 (``model_type: olmo3``) argument patches.

``megatron.olmo3.args`` (phase ``before_train``):

* Window. The rules live in
  :mod:`primus.backends.megatron.core.models.olmo3.olmo3_validation` (one contract for the args
  patch, ``Olmo3TransformerConfig`` and the builder). ``olmo3_window_size_form`` says which form
  ``window_size`` holds (``hf``: ``[W, 0]``, the model preset; ``megatron``: ``(W - 1, 0)``), so it
  decides whether the window is already converted; ``olmo3_sliding_window`` is the HF window W the
  converted value must agree with. This patch converts an HF window
  once to Megatron / TE ``(W - 1, 0)`` and sets ``olmo3_sliding_window = W`` and
  ``olmo3_window_size_form = megatron``. A window with neither setting is ambiguous and raises
  (never guessed, so a lost form or W cannot shorten the window by one key). The patch is
  idempotent.
* ``window_attn_skip_freq`` is required: an int ``2 <= k <= num_layers`` or a 0/1 list of
  ``num_layers`` entries with at least one 0 and one 1; CLI strings (``"4"``,
  ``"([1]*3+[0])*8"``) are parsed without ``eval``.
* Settings the OLMo-3 model does not implement are refused with the shared
  :data:`OLMO3_UNSUPPORTED` table (the same table the config and the builder use).
"""

from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
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
    description="OLMo-3: HF (W, 0) window -> Megatron (W-1, 0) + olmo3_sliding_window; refuse unsupported settings.",
    condition=is_olmo3_model_type,
)
def patch_olmo3_args(ctx: PatchContext):
    args = get_args(ctx)
    ws, w = olmo3_window_from_settings(
        getattr(args, "window_size", None),
        getattr(args, "olmo3_sliding_window", None),
        getattr(args, "olmo3_window_size_form", None),
    )
    skip = parse_olmo3_skip_freq(
        getattr(args, "window_attn_skip_freq", None), getattr(args, "num_layers", None)
    )
    check_olmo3_unsupported(args)
    before = getattr(args, "window_size", None)
    converted = before is None or isinstance(before, str) or tuple(int(x) for x in before) != tuple(ws)
    args.window_size = ws
    args.olmo3_sliding_window = w
    args.olmo3_window_size_form = "megatron"
    args.window_attn_skip_freq = skip
    if converted:
        log_rank_0(
            f"[Patch:megatron.olmo3.args] sliding_window={w}: window_size {before!r} -> {ws} (Megatron/TE) "
            f"window_attn_skip_freq={skip}"
        )
    else:
        log_rank_0(
            f"[Patch:megatron.olmo3.args] window_size={ws} (already Megatron form, sliding_window={w}) "
            f"window_attn_skip_freq={skip}"
        )
