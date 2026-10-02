###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Apply weight decay to every parameter, as the MLPerf flux1 reference does.

The reference (torchtitan) passes all parameters to AdamW in one group with weight_decay 0.1.
Megatron's standard optimizer overrides set wd_mult = 0 for every ``*.bias`` and every 1-D
parameter (for Flux: the linear, AdaLN-modulation and embedder biases and the QK-norm weights).
With ``reference_weight_decay: true`` that override is dropped; every other standard override
(e.g. a decoupled lr) is kept.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return args is not None and getattr(args, "reference_weight_decay", False)


def without_wd_skip(overrides):
    """``overrides`` without the entries whose only effect is wd_mult = 0."""
    return {key: value for key, value in overrides.items() if dict(value) != {"wd_mult": 0.0}}


@register_patch(
    "megatron.optimizer.reference_weight_decay",
    backend="megatron",
    phase="before_train",
    description="Weight decay on every parameter (MLPerf flux1 reference), not skipping biases / 1-D",
    condition=_enabled,
    priority=50,
)
def patch_reference_weight_decay(ctx: PatchContext):
    import megatron.core.optimizer as core_optimizer
    import megatron.training.training as training

    original = core_optimizer.get_standard_config_overrides

    def get_standard_config_overrides(config):
        overrides = original(config=config)
        kept = without_wd_skip(overrides)
        log_rank_0(
            f"[reference_weight_decay] weight decay on all parameters: dropped "
            f"{len(overrides) - len(kept)} wd_mult=0 override(s)"
        )
        return kept

    # training.py imports the function by name; core.optimizer calls it when no overrides are given
    core_optimizer.get_standard_config_overrides = get_standard_config_overrides
    training.get_standard_config_overrides = get_standard_config_overrides
    log_rank_0("[Patch:megatron.optimizer.reference_weight_decay] installed")
