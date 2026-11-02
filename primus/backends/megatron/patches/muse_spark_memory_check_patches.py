###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark: fail fast when the model states cannot fit (e.g. the full 816B model
with the single-node example defaults) instead of an out-of-memory error after the
model build. Runs before training, when the device and the world size are known.
"""

import os

from primus.backends.megatron.core.models.muse_spark.memory_check import (
    DEFAULT_BUDGET,
    DEFAULT_GPUS_PER_NODE,
    DEFAULT_MAX_NODES,
    RAISE_FRACTION,
    WARN_FRACTION,
)
from primus.backends.megatron.core.models.muse_spark.memory_check import (
    gpus_per_node as read_gpus_per_node,
)
from primus.backends.megatron.core.models.muse_spark.memory_check import (
    state_bytes_per_gpu,
    suggest_layout,
)
from primus.backends.megatron.core.models.muse_spark.model_types import (
    is_muse_spark_model_type,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

GiB = 2**30  # bytes per GiB, for messages
SKIP_ARG = "muse_spark_skip_memory_check"  # config / args key (muse_spark_base.yaml: false)
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off", "")


# What to do when no layout fits: run the cut 4-layer example, or cut the model on the CLI.
CUT_MODEL_HINT = (
    "cut the model (muse_Spark_Open_816B_A42B-proxy-BF16-pretrain.yaml, or --num_layers / --num_experts)"
)


def _layout_hint(args, total_bytes, gpus_per_node):
    """A layout for these GPUs, from :func:`suggest_layout` (states within ``DEFAULT_BUDGET``
    of a GPU)."""
    hint = suggest_layout(args, total_bytes, n_gpus_per_node=gpus_per_node)
    if hint is None:
        return (
            f"No layout up to {DEFAULT_MAX_NODES} nodes keeps the model states under "
            f"{DEFAULT_BUDGET:.0%} of a GPU: {CUT_MODEL_HINT}"
        )
    return (
        f"Use more GPUs, e.g. >= {hint['nodes']} node(s) of {gpus_per_node} such GPUs: PRIMUS_TP={hint['tp']} "
        f"PRIMUS_PP={hint['pp']} PRIMUS_EP={hint['ep']} (~{hint['bytes'] / GiB:.0f} GiB of states per GPU; "
        f"GBS a multiple of DP; recompute helps the activations), or {CUT_MODEL_HINT}"
    )


def skip_requested(args, warn=None) -> bool:
    """``args.muse_spark_skip_memory_check`` as a bool. A bool is used as is; a string is
    true for 1/true/yes/on and false for 0/false/no/off/empty (any case); any other value
    is false with a warning (the check still runs)."""
    value = getattr(
        args, "muse_spark_skip_memory_check", False
    )  # SKIP_ARG; a literal for the config-schema scan
    if value is None or isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text not in _FALSE:
        (warn or log_rank_0)(
            f"[Patch:megatron.muse_spark.memory_check] {SKIP_ARG}={value!r} is not a boolean "
            "(true/yes/on/1 or false/no/off/0/empty); the memory check runs"
        )
    return False


def required_state_bytes_or_raise(args, world_size, total_bytes, gpus_per_node=DEFAULT_GPUS_PER_NODE):
    """The estimated weights + grads + optimizer states per GPU (bytes); raise ValueError when
    they exceed ``RAISE_FRACTION`` of ``total_bytes``. The rest is a fixed headroom for the
    TE GEMM workspaces, RCCL buffers and activations, which are not estimated, so only
    layouts that cannot fit raise.

    Uses ``args.padded_vocab_size`` (the allocated embedding / head rows). The layout in
    the error is computed for this GPU's memory (MI300X and MI355X differ).
    """
    need = state_bytes_per_gpu(args, world_size)
    if need > RAISE_FRACTION * total_bytes:
        raise ValueError(
            f"Muse Spark: weights + grads + optimizer states need ~{need / GiB:.0f} GiB per GPU "
            f"(num_layers={args.num_layers}, num_experts={args.num_experts}, "
            f"TP={args.tensor_model_parallel_size}, PP={args.pipeline_model_parallel_size}, "
            f"EP={args.expert_model_parallel_size}, world={world_size}) but a GPU has "
            f"{total_bytes / GiB:.0f} GiB (limit {RAISE_FRACTION:.0%} of it). {_layout_hint(args, total_bytes, gpus_per_node)}. "
            f"{SKIP_ARG}: true skips this check."
        )
    return need


@register_patch(
    "megatron.muse_spark.memory_check",
    backend="megatron",
    phase="before_train",
    condition=is_muse_spark_model_type,
    description=(
        "Muse Spark: fail fast when weights + grads + optimizer states exceed RAISE_FRACTION "
        f"({RAISE_FRACTION:.0%}) of GPU memory"
    ),
)
def patch_muse_spark_memory_check(ctx: PatchContext):
    args = get_args(ctx)
    if skip_requested(args):
        return
    import torch

    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    world = int(getattr(args, "world_size", None) or os.getenv("WORLD_SIZE", "1"))
    # the shared reader (LOCAL_WORLD_SIZE, then GPUS_PER_NODE), as the DeepEP check; this only
    # sizes the layout hint, so a malformed value is logged and the device count is used
    gpn = read_gpus_per_node(
        torch.cuda.device_count(),  # >= 1: CUDA is available here
        warn=lambda msg: log_rank_0(f"[Patch:megatron.muse_spark.memory_check] {msg}"),
    )
    need = required_state_bytes_or_raise(args, world, total, gpus_per_node=gpn)
    note = ""
    if need > WARN_FRACTION * total:
        note = (
            f" -- WARNING: over {WARN_FRACTION:.0%}, little room is left for activations "
            "(enable recompute or add GPUs)"
        )
    log_rank_0(
        f"[Patch:megatron.muse_spark.memory_check] model states ~{need / GiB:.0f} GiB of "
        f"{total / GiB:.0f} GiB per GPU (activations not counted){note}"
    )
