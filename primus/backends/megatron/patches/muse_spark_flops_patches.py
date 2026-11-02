###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark FLOPs reporting.

Megatron's ``num_floating_point_operations`` already counts the Muse Spark blocks it
knows: GQA with the attention output gate (``attention_output_gate``), the latent MoE
(``moe_latent_size``: experts on the latent width plus the down / up projections), the
untied head. What it gets wrong is the sliding-window attention: it charges every
layer the dense causal ``S / 2`` keys per query, while a sliding layer
(``window_attn_skip_freq`` entry 1) attends to at most ``window`` keys (HF
``sliding_window``, incl. the query). At seq 4096 / window 2048 that over-reports the
core attention of 3 of every 4 layers by ~1.33x; at 1M context by ~256x.

This wrapper changes two terms (the rest stays upstream's formula). It rescales the
sliding-window attention term:

    delta = 6 * B * S * q_size * 2 * (S / 2) * (mean_keys(S, W) / mean_keys(S, S) - 1) * n_sliding

with ``mean_keys(S, W) = mean_i min(i + 1, W)``. The ratio keeps upstream's units
(``S / 2``): it is exactly 1 (no correction) when ``W >= S`` (sliding == full causal).
A layer is a sliding layer by Megatron's own ``is_layer_window_attention``: a list of
0/1 per layer, an int N (every layer except each N-th, which is full attention), or None
(every layer). It also adds the router GEMM (h x num_experts), which
upstream omits, for each MoE layer, counted like upstream (``moe_layer_freq`` plus MTP
layers when the last layer is MoE); dense layers have no router.
Norms / gates are elementwise and not counted (as upstream).
"""

from primus.backends.megatron.core.models.muse_spark.layer_patterns import (
    causal_visible_pairs,
    hf_sliding_window,
    normalize_layer_pattern,
    parse_window_size,
)
from primus.backends.megatron.core.models.muse_spark.model_types import (
    is_muse_spark_model_type,
)
from primus.backends.megatron.patches.kimi_k3_flops_patches import moe_layer_pattern
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_TAG = "[Patch:megatron.muse_spark.flops_reporting]"


def causal_mean_keys(seq_length: int, window) -> float:
    """Mean keys per query of causal attention limited to ``window`` keys (None: full).

    The visible-pair count (:func:`causal_visible_pairs`) divided by the sequence length."""
    s = int(seq_length)
    return causal_visible_pairs(s, window) / s


def sliding_window_keys(args):
    """Keys a sliding-window query sees (incl. itself) from ``args.window_size``, or None
    for no window. Validated like the config (``parse_window_size``): in the presets' HF
    convention (``window_size_is_hf_sliding_window``, default True) the left value is the
    HF ``sliding_window``; in TE's convention it is ``left + 1``."""
    ws = getattr(args, "window_size", None)
    if isinstance(ws, int) and not isinstance(ws, bool):
        ws = (ws, 0)
    if getattr(args, "window_size_is_hf_sliding_window", True):
        return hf_sliding_window(ws)
    ws = parse_window_size(ws)
    return None if ws is None or ws[0] == -1 else ws[0] + 1


def _fallback_is_window_layer(window_size, pattern, layer_number):
    """Pure-Python copy of Megatron's ``is_layer_window_attention`` (1-indexed layer)."""
    if not window_size:
        return False
    if pattern is None:
        return True
    if isinstance(pattern, int):
        return layer_number % pattern != 0
    return bool(pattern[layer_number - 1])


def _window_layer_fn():
    """Megatron's ``is_layer_window_attention``, imported once per count (not per layer)."""
    try:
        from megatron.core.transformer.utils import is_layer_window_attention
    except ImportError:
        return _fallback_is_window_layer
    return is_layer_window_attention


def num_sliding_layers(args) -> int:
    n_layers = int(args.num_layers)
    ws = getattr(args, "window_size", None)
    if isinstance(ws, str):
        ws = parse_window_size(ws)
    pattern = getattr(args, "window_attn_skip_freq", None)
    if pattern is not None:
        pattern = normalize_layer_pattern(pattern, n_layers, "window_attn_skip_freq")
    is_window = _window_layer_fn()
    return sum(1 for n in range(1, n_layers + 1) if is_window(ws, pattern, n))


def num_moe_layers(args) -> int:
    """MoE layers as upstream counts them: ``moe_layer_freq`` (int every-N-th or a 0/1
    list) over ``num_layers``, plus ``mtp_num_layers`` when the last layer is MoE."""
    if not int(getattr(args, "num_experts", None) or 0):
        return 0
    n_layers = int(args.num_layers)
    # unset is upstream's default 1 (every layer); the int / list rules and the length check
    # are the shared moe_layer_pattern (kimi_k3_flops_patches), not a second copy
    freq = getattr(args, "moe_layer_freq", None) or 1
    if isinstance(freq, str):  # yaml / CLI pattern string, e.g. "([1]*3+[0])"
        freq = normalize_layer_pattern(freq, n_layers, "moe_layer_freq")
    pattern = [int(bool(v)) for v in moe_layer_pattern(freq, n_layers)]
    mtp = int(getattr(args, "mtp_num_layers", None) or 0)
    return sum(pattern) + (pattern[-1] * mtp if pattern else 0)


def muse_spark_flops_delta_per_sample(args) -> float:
    """The per-sample correction that :func:`muse_spark_flops_delta` scales by the batch and
    adds to Megatron's count (not the model's total FLOPs): the sliding-window attention
    adjustment plus the router term. It depends only on the run's fixed args (seq_length,
    window, layer pattern, experts), so the wrapper computes it once."""
    s = int(args.seq_length)
    q = int(args.kv_channels) * int(args.num_attention_heads)
    w = sliding_window_keys(args)
    attn = 0.0
    if w is not None and w < s:
        n_sliding = num_sliding_layers(args)
        # upstream per layer per token: 6 * q * 2 * (S/2); scale its S/2 by the key ratio
        ratio = causal_mean_keys(s, w) / causal_mean_keys(s, None)
        attn = 6.0 * q * 2.0 * (s / 2.0) * (ratio - 1.0) * n_sliding
    router = 6.0 * int(args.hidden_size) * int(args.num_experts or 0) * num_moe_layers(args)
    return s * (attn + router)


def muse_spark_flops_delta(args, batch_size, per_sample=None) -> float:
    """FLOPs to add to Megatron's count: a sliding-window correction (negative: upstream
    counts full causal attention on the sliding layers) plus the router GEMM that
    upstream omits (positive). The sign of the sum depends on which term is larger;
    with no window (or a window >= seq_length) it is the router term alone.
    ``per_sample``: a cached :func:`muse_spark_flops_delta_per_sample` value for ``args``."""
    if per_sample is None:
        per_sample = muse_spark_flops_delta_per_sample(args)
    return batch_size * per_sample


def make_muse_spark_num_floating_point_operations(original_fn):
    """Wrap ``original_fn`` so every call adds :func:`muse_spark_flops_delta`.

    The per-sample delta depends only on the run's fixed args, so it is computed on the
    first call for an ``args`` object and reused (``batch_size`` is the only per-call input).

    No ``args.model_type`` check per call: Megatron's ``get_model`` replaces
    ``args.model_type`` with the ``ModelType`` enum before ``train()`` counts FLOPs,
    so a per-call string check would never match. The decision is made once, at
    install time, by the patch condition (``is_muse_spark_model_type``, evaluated in
    ``before_train`` while ``model_type`` is still the yaml string), like
    ``deepseek_v4_flops_patches``. Install this wrapper only for Muse Spark runs.
    """

    cache = {}  # id(args) -> (args, per-sample delta): the fixed args are read once per run

    def wrapped(args, batch_size):
        hit = cache.get(id(args))
        if hit is None or hit[0] is not args:
            hit = cache[id(args)] = (args, muse_spark_flops_delta_per_sample(args))
        return original_fn(args, batch_size) + muse_spark_flops_delta(args, batch_size, hit[1])

    wrapped.__wrapped__ = original_fn
    wrapped._muse_spark_flops_patched = True
    return wrapped


@register_patch(
    "megatron.muse_spark.flops_reporting",
    backend="megatron",
    phase="before_train",
    condition=is_muse_spark_model_type,
    description="Muse Spark: charge sliding-window layers their window, not causal S/2, and add the router GEMM",
)
def patch_muse_spark_flops_reporting(ctx: PatchContext):
    import megatron.training.training as training_module

    original_fn = training_module.num_floating_point_operations
    if getattr(original_fn, "_muse_spark_flops_patched", False):
        return
    training_module.num_floating_point_operations = make_muse_spark_num_floating_point_operations(original_fn)
    log_rank_0(f"{_TAG} wrapped num_floating_point_operations (sliding-window keys, router GEMM)")
