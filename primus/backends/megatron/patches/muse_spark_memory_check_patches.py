###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark: fail fast when the model states cannot fit (e.g. the full 816B model
with the single-node example defaults) instead of an out-of-memory error after the
model build. Runs before training, when the device and the world size are known.
"""

import torch

from primus.backends.megatron.core.models.muse_spark.memory_check import (
    RAISE_FRACTION,
    WARN_FRACTION,
    GiB,
    gpus_per_node,
    required_state_bytes_or_raise,
    skip_requested,
)
from primus.backends.megatron.core.models.muse_spark.model_types import (
    fail_fast,
    is_muse_spark_model_type,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.env import get_torchrun_env
from primus.core.utils.module_utils import log_rank_0

PATCH_NAME = "megatron.muse_spark.memory_check"


@register_patch(
    PATCH_NAME,
    backend="megatron",
    phase="before_train",
    condition=is_muse_spark_model_type,
    description=(
        "Muse Spark: fail fast when weights + grads + optimizer states exceed RAISE_FRACTION "
        f"({RAISE_FRACTION:.0%}) of GPU memory"
    ),
)
@fail_fast(PATCH_NAME)
def patch_muse_spark_memory_check(ctx: PatchContext):
    args = get_args(ctx)
    if skip_requested(args):
        return
    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    world = int(getattr(args, "world_size", None) or get_torchrun_env()["world_size"])
    # the shared reader (GPUS_PER_NODE, then LOCAL_WORLD_SIZE), as the DeepEP check; a
    # malformed value raises there; with neither set, this node's device count
    n_gpus = gpus_per_node(torch.cuda.device_count())  # >= 1: CUDA is available here
    need = required_state_bytes_or_raise(args, world, total, n_gpus_per_node=n_gpus)
    note = ""
    if need > WARN_FRACTION * total:
        note = (
            f" -- WARNING: over {WARN_FRACTION:.0%}, little room is left for activations "
            "(enable recompute or add GPUs)"
        )
    log_rank_0(
        f"[Patch:{PATCH_NAME}] model states ~{need / GiB:.0f} GiB of "
        f"{total / GiB:.0f} GiB per GPU (activations not counted){note}"
    )
