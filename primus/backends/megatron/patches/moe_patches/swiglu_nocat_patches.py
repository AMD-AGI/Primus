###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tier 1B — MoE SwiGLU backward without cat/split.

Wraps the runtime install hook
:mod:`primus.backends.megatron.core.extensions.moe_swiglu_nocat` in a
``@register_patch`` so it runs at the standard ``before_train`` phase via
``run_patches(...)`` instead of being explicitly installed from the trainer
entry point.

Gate:
  * ``PRIMUS_MOE_SWIGLU_NOCAT=1`` — replaces the default
    ``F.silu(gate) * up * probs`` composition (which Inductor lowers to
    ``triton_poi_fused__to_copy_cat_mul_silu_silu_backward_split_1``) with
    a pair of Triton kernels that operate on a single ``[N, 2H]`` ``x``
    buffer, eliminating the bwd ``cat``+``split`` round-trip on stream 0.

The install hook bails gracefully when the MLP is not GLU-gated or when
the activation is not ``torch.nn.functional.silu``, so it's safe to run
regardless of the model.
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _is_swiglu_nocat_enabled(ctx: PatchContext) -> bool:
    return _env_truthy("PRIMUS_MOE_SWIGLU_NOCAT")


@register_patch(
    "megatron.moe.swiglu_nocat",
    backend="megatron",
    phase="before_train",
    description=(
        "Replace GroupedMLP.activation_func_with_probs with Triton fused "
        "SwiGLU+probs (no cat/split in bwd); gated by PRIMUS_MOE_SWIGLU_NOCAT."
    ),
    condition=_is_swiglu_nocat_enabled,
)
def patch_moe_swiglu_nocat(ctx: PatchContext):
    """Install the MoE SwiGLU no-cat runtime monkeypatch."""
    from primus.backends.megatron.core.extensions import moe_swiglu_nocat

    log_rank_0(
        "[Patch:megatron.moe.swiglu_nocat] Installing Triton SwiGLU+probs "
        "(no cat/split in bwd)"
    )
    ok = moe_swiglu_nocat.install()
    if ok:
        log_rank_0(
            "[Patch:megatron.moe.swiglu_nocat]   install() returned True "
            "(GroupedMLP.activation_func_with_probs replaced)"
        )
    else:
        log_rank_0(
            "[Patch:megatron.moe.swiglu_nocat]   install() returned False "
            "(precondition not met)"
        )
