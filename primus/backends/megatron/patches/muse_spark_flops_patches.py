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

from primus.backends.megatron.core.models.muse_spark.layer_freq import moe_layer_pattern
from primus.backends.megatron.core.models.muse_spark.layer_patterns import (
    hf_sliding_window,
    normalize_layer_pattern,
    parse_window_size,
)
from primus.backends.megatron.core.models.muse_spark.model_types import (
    MODEL_TYPE,
    bool_flag,
    is_muse_spark_model_type,
)

# DeepSeek-V4's causal pair count (swa_window keys per query, <= 0 or >= S: full causal):
# the same formula, reused rather than copied
from primus.backends.megatron.patches.deepseek_v4_flops_patches import (
    _local_visible_pairs as causal_visible_pairs_of,
)
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

PATCH_NAME = "megatron.muse_spark.flops_reporting"
_TAG = f"[Patch:{PATCH_NAME}]"


# 2 FLOPs per multiply-add x 3 (forward, plus the two backward GEMMs): Megatron's factor
FWD_BWD_FLOPS_PER_MAC = 6.0


def sliding_window_keys(args):
    """Keys a sliding-window query sees (incl. itself) from ``args.window_size``, or None
    for no window. Validated like the config (``parse_window_size``): in the presets' HF
    convention (``window_size_is_hf_sliding_window``, default True) the left value is the
    HF ``sliding_window``; in TE's convention it is ``left + 1``."""
    ws = getattr(args, "window_size", None)
    if isinstance(ws, int) and not isinstance(ws, bool):
        ws = (ws, 0)
    # strict boolean as the config parses it ("false" / "0" from env-driven yaml are False)
    if bool_flag(
        getattr(args, "window_size_is_hf_sliding_window", None), "window_size_is_hf_sliding_window", True
    ):
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
    # unset (None) is upstream's default 1 (every layer); an int <= 0 is a ValueError (no
    # layout Megatron builds); a string is parsed by normalize_layer_pattern (an empty one is an error). The
    # int / list rules and the length check are the shared moe_layer_pattern (muse_spark/layer_freq.py).
    freq = getattr(args, "moe_layer_freq", None)
    if freq is None:
        freq = 1
    if isinstance(freq, str):  # yaml / CLI pattern string, e.g. "([1]*3+[0])"
        freq = normalize_layer_pattern(freq, n_layers, "moe_layer_freq")
    pattern = [int(bool(v)) for v in moe_layer_pattern(freq, n_layers)]
    mtp = int(getattr(args, "mtp_num_layers", None) or 0)
    return sum(pattern) + (pattern[-1] * mtp if pattern else 0)


class MuseSparkFlopsTerms:
    """The fixed parts of the Muse Spark FLOPs correction for a run (read once from ``args``):
    the attention width ``q``, the sliding window (keys), the number of sliding layers and the
    router FLOPs per token."""

    def __init__(self, args):
        self.seq_length = int(args.seq_length)
        self.q = int(args.kv_channels) * int(args.num_attention_heads)
        self.window = sliding_window_keys(args)
        self.n_sliding = num_sliding_layers(args) if self.window is not None else 0
        self.router_per_token = (
            FWD_BWD_FLOPS_PER_MAC * int(args.hidden_size) * int(args.num_experts or 0) * num_moe_layers(args)
        )

    def sliding_attention_delta(self, seqlen_squared_sum: float, doc_length: float) -> float:
        """The sliding-window correction for attention that upstream charges as
        ``FWD_BWD_FLOPS_PER_MAC * q * seqlen_squared_sum`` per layer (full causal), for
        documents of length ``doc_length``: 0 when they fit in the window."""
        w, L = self.window, int(round(doc_length))
        if self.n_sliding == 0 or L <= w:  # n_sliding is 0 when there is no window
            return 0.0
        ratio = causal_visible_pairs_of(w, L) / causal_visible_pairs_of(0, L)  # = mean_keys ratio
        return FWD_BWD_FLOPS_PER_MAC * self.q * seqlen_squared_sum * (ratio - 1.0) * self.n_sliding


def muse_spark_flops_delta_per_sample(args, terms=None) -> float:
    """The per-sample correction for a padded (BSHD) batch of ``args.seq_length`` tokens per
    sample: the sliding-window attention adjustment plus the router term (``terms``: the
    run's :class:`MuseSparkFlopsTerms`, built from ``args`` when not given)."""
    t = terms or MuseSparkFlopsTerms(args)
    s = t.seq_length
    return t.sliding_attention_delta(s * s, s) + t.router_per_token * s


def muse_spark_flops_delta(
    args,
    batch_size,
    per_sample=None,
    seqlen_squared_sum_in_batch=None,
    total_real_tokens_in_batch=None,
    terms=None,
) -> float:
    """FLOPs to add to Megatron's count: a sliding-window correction (negative: upstream
    counts full causal attention on the sliding layers) plus the router GEMM that
    upstream omits (positive).

    BSHD (no packed stats): ``batch_size`` samples of ``seq_length`` tokens (``per_sample``: a
    cached :func:`muse_spark_flops_delta_per_sample`). Packed / THD batches (Megatron-core 0.19
    passes ``seqlen_squared_sum_in_batch = sum L_i^2`` and ``total_real_tokens_in_batch = sum L_i``):
    the router term is per real token, and the window correction is taken at the
    length-weighted mean document length ``sum L_i^2 / sum L_i`` (exact for equal-length
    documents and for BSHD; 0 when the documents fit in the window, where upstream's count is
    already right). Only the two sums reach the FLOPs count, so per-document lengths are not
    available here."""
    if seqlen_squared_sum_in_batch is None and total_real_tokens_in_batch is None:
        if per_sample is None:
            per_sample = muse_spark_flops_delta_per_sample(args, terms)
        return batch_size * per_sample
    t = terms or MuseSparkFlopsTerms(args)
    s = t.seq_length
    tokens = float(total_real_tokens_in_batch if total_real_tokens_in_batch is not None else batch_size * s)
    sq = float(seqlen_squared_sum_in_batch if seqlen_squared_sum_in_batch is not None else batch_size * s * s)
    doc_length = sq / tokens if tokens > 0 else 0.0
    return t.sliding_attention_delta(sq, doc_length) + t.router_per_token * tokens


# set on the run's args object by the patch: the per-call Muse Spark check of the wrapper
MUSE_SPARK_ARGS_ATTR = "_primus_muse_spark_flops"


def is_muse_spark_args(args) -> bool:
    """``args`` belong to a Muse Spark run: marked by the patch, or ``model_type`` is still
    the Muse Spark string."""
    return bool(getattr(args, MUSE_SPARK_ARGS_ATTR, False)) or getattr(args, "model_type", None) == MODEL_TYPE


def make_muse_spark_num_floating_point_operations(original_fn):
    """Wrap ``original_fn`` so every call adds :func:`muse_spark_flops_delta`.

    The fixed terms depend only on the run's args, so they are computed on the first call for
    an ``args`` object and reused; ``batch_size`` and Megatron-core 0.19's packed-batch sums
    (``seqlen_squared_sum_in_batch`` / ``total_real_tokens_in_batch``) are the per-call inputs.

    Each call checks that ``args`` is a Muse Spark run's (:func:`is_muse_spark_args`); any other
    args get ``original_fn``'s count unchanged, so a later non-Muse model in the same process
    is not corrected. The string ``args.model_type`` alone is not enough: Megatron's
    ``get_model`` replaces it with the ``ModelType`` enum before ``train()`` counts FLOPs. The
    patch therefore marks the run's args object (:data:`MUSE_SPARK_ARGS_ATTR`) at install time
    (``before_train``, while ``model_type`` is still the yaml string), and the check accepts
    either the mark or the string.
    """

    cached_args, cached_per_sample, cached_terms = None, None, None  # the fixed args, read once per run

    def wrapped(
        args, batch_size, seqlen_squared_sum_in_batch=None, total_real_tokens_in_batch=None, **kwargs
    ):
        nonlocal cached_args, cached_per_sample, cached_terms
        packed = {}
        if seqlen_squared_sum_in_batch is not None:
            packed["seqlen_squared_sum_in_batch"] = seqlen_squared_sum_in_batch
        if total_real_tokens_in_batch is not None:
            packed["total_real_tokens_in_batch"] = total_real_tokens_in_batch
        base = original_fn(args, batch_size, **packed, **kwargs)
        if not is_muse_spark_args(args):  # another model in the same process: upstream's count
            return base
        if cached_args is not args:
            cached_terms = MuseSparkFlopsTerms(args)
            cached_args, cached_per_sample = args, muse_spark_flops_delta_per_sample(args, cached_terms)
        return base + muse_spark_flops_delta(
            args, batch_size, per_sample=cached_per_sample, terms=cached_terms, **packed
        )

    wrapped.__wrapped__ = original_fn
    wrapped._muse_spark_flops_patched = True
    return wrapped


@register_patch(
    PATCH_NAME,
    backend="megatron",
    phase="before_train",
    condition=is_muse_spark_model_type,
    description="Muse Spark: charge sliding-window layers their window, not causal S/2, and add the router GEMM",
)
def patch_muse_spark_flops_reporting(ctx: PatchContext):
    import megatron.training.training as training_module

    # mark this run's args (the object Megatron's get_args returns) for the per-call check
    for args in (ctx.extra.get("backend_args"), getattr(ctx.extra.get("module_config"), "params", None)):
        if args is not None:
            setattr(args, MUSE_SPARK_ARGS_ATTR, True)
    original_fn = training_module.num_floating_point_operations
    if getattr(original_fn, "_muse_spark_flops_patched", False):
        return
    training_module.num_floating_point_operations = make_muse_spark_num_floating_point_operations(original_fn)
    log_rank_0(f"{_TAG} wrapped num_floating_point_operations (sliding-window keys, router GEMM)")
