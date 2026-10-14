###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText NNX train-step fix: keep the updated FP8 delayed-scaling stats.

With ``quantization: "fp8"`` the Flax FP8 ops keep their scales and amax
histories in the ``_overwrite_with_gradient`` collection. ``train_step``
differentiates them as ``custom_params`` and writes the result back with

    nnx.update(state.model, nnx.State.merge(custom_grads, non_param_rest))

``State.merge`` lets later states win, and ``non_param_rest`` still holds the
stale copies of those stats, so the new ones are dropped every step. The scales
stay at 1.0, output gradients underflow in e5m2, and every quantized kernel gets
an exactly-zero gradient (flat loss). Swapping the merge order fixes it; for
non-FP8 runs ``custom_grads`` is empty and the order does not matter.

MaxText's ``train_step`` is re-compiled from its own source with only that call
swapped, so the submodule stays at its upstream pin. If the upstream line
changes, the patch logs a warning and leaves ``train_step`` alone.

  PRIMUS_MAXTEXT_FP8_MERGE_FIX=0   disable
"""

import inspect
import os
import textwrap

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_ENV = "PRIMUS_MAXTEXT_FP8_MERGE_FIX"
_BUGGY = "nnx.State.merge(custom_grads, non_param_rest)"
_FIXED = "nnx.State.merge(non_param_rest, custom_grads)"


def _enabled(_ctx=None) -> bool:
    return os.getenv(_ENV, "1") != "0"


def fixed_train_step(train_module):
    """Return ``train_module.train_step`` rebuilt with the merge order swapped, or None."""
    orig = train_module.train_step
    src = textwrap.dedent(inspect.getsource(orig))
    if src.count(_BUGGY) != 1:
        return None
    first_line = inspect.getsourcelines(orig)[1]
    code = compile("\n" * (first_line - 1) + src.replace(_BUGGY, _FIXED), inspect.getsourcefile(orig), "exec")
    namespace = {}
    exec(code, train_module.__dict__, namespace)  # pylint: disable=exec-used
    return namespace[orig.__name__]


@register_patch(
    patch_id="maxtext.fp8_merge_order",
    backend="maxtext",
    phase="setup",
    description="Keep updated FP8 scales/amax in the NNX train step (State.merge order)",
    condition=_enabled,
)
def patch_fp8_merge_order(ctx: PatchContext) -> None:
    try:
        from maxtext.trainers.pre_train import train as maxtext_train
    except ImportError as e:
        warning_rank_0(f"[Patch:maxtext.fp8_merge_order] MaxText v26.4+ layout not found; skipping: {e}")
        return

    if getattr(maxtext_train.train_step, "_primus_fp8_merge_fixed", False):
        return
    src = inspect.getsource(maxtext_train.train_step)
    if _FIXED in src and _BUGGY not in src:
        log_rank_0("[Patch:maxtext.fp8_merge_order] train_step already merges in the fixed order.")
        return
    patched = fixed_train_step(maxtext_train)
    if patched is None:
        warning_rank_0(
            f"[Patch:maxtext.fp8_merge_order] expected exactly one `{_BUGGY}` in train_step; "
            "MaxText changed, not patching."
        )
        return
    patched._primus_fp8_merge_fixed = True
    maxtext_train.train_step = patched
    warning_rank_0("[Patch:maxtext.fp8_merge_order] train_step patched (FP8 stats merged last).")
