###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Let TE's fused cross entropy reuse the logits buffer for its backward.

TE <= 2.17 always wrote the input gradient into the logits in place. Newer TE
saves a copy of the logits for backward unless ``overwrite_input=True``, which
costs one extra logits-sized buffer per in-flight micro-batch (b * s * vocab *
2 bytes in BF16, ~18 GB for gpt-oss-20B at mbs 12). Megatron's wrapper does not
pass the flag, so this patch does when ``te_ce_overwrite_input`` is set.

With the flag, the logits hold the gradient after the loss backward. Only
enable it when nothing reads the logits after backward (distillation, RL
log-prob reuse, logit logging hooks would see the gradient).
"""

import inspect

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return (
        bool(getattr(args, "te_ce_overwrite_input", False))
        and bool(getattr(args, "cross_entropy_loss_fusion", False))
        and getattr(args, "cross_entropy_fusion_impl", None) == "te"
    )


def supports_overwrite_input(parallel_cross_entropy) -> bool:
    return "overwrite_input" in inspect.signature(parallel_cross_entropy).parameters


def make_overwrite_input_cross_entropy(original, parallel_cross_entropy):
    """Megatron's ``te_parallel_cross_entropy`` with ``overwrite_input=True``."""

    def te_parallel_cross_entropy(logits, labels, tp_group, is_cg_capturable=False):
        # TE rejects overwrite_input for non-contiguous logits.
        if not logits.is_contiguous():
            return original(logits, labels, tp_group, is_cg_capturable)
        return parallel_cross_entropy(
            logits, labels, 0.0, False, tp_group, -100, is_cg_capturable, overwrite_input=True
        )

    te_parallel_cross_entropy._primus_overwrite_input = True
    return te_parallel_cross_entropy


@register_patch(
    "megatron.te.cross_entropy_overwrite_input",
    backend="megatron",
    phase="before_train",
    description="Pass overwrite_input=True to TE parallel_cross_entropy to drop its logits copy",
    condition=_enabled,
)
def patch_te_cross_entropy_overwrite_input(ctx: PatchContext) -> None:
    import megatron.core.extensions.transformer_engine as te_ext
    from megatron.core.models.common.language_module import language_module

    original = te_ext.te_parallel_cross_entropy
    if original is None or getattr(original, "_primus_overwrite_input", False):
        return

    from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

    if not supports_overwrite_input(parallel_cross_entropy):
        log_rank_0(
            "[Patch:megatron.te.cross_entropy_overwrite_input] installed TE has no "
            "overwrite_input; keeping Megatron's cross entropy"
        )
        return

    wrapped = make_overwrite_input_cross_entropy(original, parallel_cross_entropy)
    te_ext.te_parallel_cross_entropy = wrapped
    language_module.te_parallel_cross_entropy = wrapped
    log_rank_0(
        "[Patch:megatron.te.cross_entropy_overwrite_input] TE cross entropy reuses "
        "the logits buffer for backward (overwrite_input=True)"
    )
