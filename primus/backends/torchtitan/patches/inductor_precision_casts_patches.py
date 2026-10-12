###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Preserve eager low-precision casts in ROCm Inductor graphs.

On the pinned ROCm torch 2.12 build, compiled FSDP BF16 llama/qwen recipes
produce a finite first loss but NaN gradients unless
``emulate_precision_casts`` is enabled. Eager and ``aot_eager`` remain finite.

This is not attributed to pytorch#150859: that issue covered float8 rowwise
training with selective activation checkpointing and is no longer reproducible
on upstream 2.12 nightly. The plain-BF16 ROCm reproducer here is different even
though the same compatibility flag fixes it.

The flag is part of Inductor's cache key, so no manual cache invalidation is
required.
"""

from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.module_utils import log_rank_0

_PREFIX = "[Patch:torchtitan.torch.inductor_precision_casts]"


def _compile_enabled(ctx: PatchContext) -> bool:
    if not bool(get_param(ctx, "compile.enable", False)):
        return False

    import torch

    return torch.version.hip is not None


@register_patch(
    "torchtitan.torch.inductor_precision_casts",
    backend="torchtitan",
    phase="setup",  # before parallelize_fn reaches apply_compile
    description="Emulate eager precision casts in ROCm Inductor BF16/FSDP graphs",
    condition=_compile_enabled,
)
def patch_inductor_precision_casts(ctx: PatchContext) -> None:
    """Make inductor round intermediates the way eager does."""
    import torch._inductor.config as inductor_config

    if inductor_config.emulate_precision_casts:
        log_rank_0(f"{_PREFIX} already enabled; leaving it alone")
        return

    inductor_config.emulate_precision_casts = True
    log_rank_0(f"{_PREFIX} torch._inductor.config.emulate_precision_casts = True")
