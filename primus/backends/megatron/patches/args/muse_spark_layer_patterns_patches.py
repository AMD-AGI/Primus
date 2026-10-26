###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark: per-layer pattern args as lists of ``num_layers`` entries.

The Muse Spark presets give ``no_rope_freq`` / ``window_attn_skip_freq`` as the
full 64-entry Python-list expression (e.g. ``"([0,1,0,0]*16)"``). Megatron builds
a plain ``TransformerConfig`` from args in places outside the model builder
(e.g. ``is_dataset_built_on_rank``), and that config checks
``len(no_rope_freq) == num_layers``. Evaluate the strings and cut the lists to
``num_layers`` here, so a depth-reduced run (proxy, trainer test) works.
"""

from primus.core.patches import PatchContext, register_patch
from primus.core.patches.context import get_args
from primus.core.utils.module_utils import log_kv_rank_0

_KEYS = ("no_rope_freq", "window_attn_skip_freq")


_MODEL_TYPES = ("muse_spark", "muse_spark_text")


def _is_muse_spark(ctx: PatchContext) -> bool:
    try:
        if getattr(get_args(ctx), "model_type", None) in _MODEL_TYPES:
            return True
    except AssertionError:
        pass
    return getattr(ctx.extra.get("backend_args"), "model_type", None) in _MODEL_TYPES


def normalize_layer_pattern(value, num_layers):
    if value is None or isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if "[" not in s:
            return int(s)
        value = eval(s, {"__builtins__": {}}, {})  # trusted config expression, as moe_layer_freq
    value = [int(v) for v in value]
    if num_layers and len(value) > num_layers:
        value = value[:num_layers]
    return value


@register_patch(
    "megatron.args.muse_spark_layer_patterns",
    backend="megatron",
    phase="build_args",
    condition=_is_muse_spark,
    description="Muse Spark: evaluate no_rope_freq / window_attn_skip_freq and cut them to num_layers",
)
def patch_muse_spark_layer_patterns(ctx: PatchContext):
    args = ctx.extra.get("backend_args", {})
    num_layers = getattr(args, "num_layers", None)
    for key in _KEYS:
        if not hasattr(args, key):
            continue
        value = normalize_layer_pattern(getattr(args, key), num_layers)
        setattr(args, key, value)
        log_kv_rank_0(f"[Patch:megatron.args.muse_spark_layer_patterns] -{key}", f"{value}")


@register_patch(
    "megatron.args.muse_spark_selective_recompute",
    backend="megatron",
    phase="build_args",
    condition=_is_muse_spark,
    description="Muse Spark: selective recompute takes no recompute_method / recompute_num_layers",
)
def patch_muse_spark_selective_recompute(ctx: PatchContext):
    # The experiment config sets recompute_method/num_layers for full recompute; Megatron
    # requires both unset with recompute_granularity=selective.
    args = ctx.extra.get("backend_args", {})
    if getattr(args, "recompute_granularity", None) != "selective":
        return
    args.recompute_method = None
    args.recompute_num_layers = None
    log_kv_rank_0("[Patch:megatron.args.muse_spark_selective_recompute] -recompute_method", "None")


@register_patch(
    "megatron.args.muse_spark_cp_comm",
    backend="megatron",
    phase="build_args",
    condition=_is_muse_spark,
    description="Muse Spark: sliding-window attention with CP uses TE's all_gather CP (p2p has no SWA backend)",
)
def patch_muse_spark_cp_comm(ctx: PatchContext):
    args = ctx.extra.get("backend_args", {})
    cp = getattr(args, "context_parallel_size", 1) or 1
    comm = getattr(args, "cp_comm_type", None)
    if getattr(args, "window_size", None) is None or cp <= 1:
        return
    if comm in (None, "p2p", ["p2p"]) or (isinstance(comm, list) and all(c == "p2p" for c in comm)):
        args.cp_comm_type = "all_gather"
        log_kv_rank_0("[Patch:megatron.args.muse_spark_cp_comm] -cp_comm_type", "all_gather")
