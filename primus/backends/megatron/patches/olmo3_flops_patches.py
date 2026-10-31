###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 FLOPs: correct the core-attention term for sliding-window layers.

Megatron's ``num_floating_point_operations`` counts causal attention over the
full sequence (``seq_length / 2`` keys per query on average) on every layer.
OLMo-3's sliding layers (3 of 4) let a query see at most ``W`` keys
(``olmo3_sliding_window``, HF ``sliding_window`` convention, current token included;
Megatron's ``window_size`` is ``(W - 1, 0)``).
This patch subtracts, per sliding layer, the difference between the causal
count and the windowed count; every other term (projections, MLP, logits) is
the upstream one.

Convention: upstream's. Upstream counts causal attention as ``s / 2`` keys per query, the
continuous form of ``sum_i (i + 1) / s = (s + 1) / 2``. The same continuous form of the windowed
count ``sum_i min(i + 1, W) / s`` is ``W - W**2 / (2 s)`` for ``W < s`` (``s / 2`` for
``W >= s``). The correction is the difference of the two, ``(s - W)**2 / (2 s)`` keys per query
for ``W < s``; for ``W >= s`` the window does not cut anything and the correction is 0 (an explicit
branch in :func:`olmo3_saved_keys`). A sliding layer then counts
``W - W**2 / (2 s)`` keys, which is within half a key of the exact windowed mean, as upstream's
``s / 2`` is within half a key of the exact causal mean. :func:`olmo3_mean_visible_keys` gives
the exact mean, for reference and tests.

Upstream must not count the window itself, or it would be subtracted twice, and it must count
causal attention as ``s / 2`` keys per query, the count the correction subtracts from. The patch
checks both when it installs (and, without built args or before the tokenizer sets
``padded_vocab_size``, at the first call): upstream's count for the run's args must not change
when ``window_size`` / ``window_attn_skip_freq`` are removed (:func:`upstream_counts_window`), and
its ``seq_length**2`` coefficient must be the core-attention term of ``s / 2`` keys per query
(:func:`upstream_causal_keys_ok`). If either check fails (a newer Megatron counts windows, or
counts attention another way), the correction is not applied and a warning says so.

Where the count goes (Megatron at the pinned commit): the throughput log, and the running total
``num_floating_point_operations_so_far`` (training.py, added per iteration), which the progress log
prints and ``save_checkpoint`` stores in the checkpoint; a resumed run continues from the stored
total. So the corrected count is persisted. A run resumed without this patch (or with a Megatron
that counts windows itself) continues the stored total with another count per iteration; the
total then mixes the two counts. The total is reporting only: no schedule, stop rule or optimizer
step reads it, so neither the patch nor the wrapper stops training. An error at install time (in
the upstream check, or in a first try of the correction) is only logged and retried at the first
count; an error at the first count, or in the correction itself, turns the
correction off with a warning (from then on, the count added to the total is upstream's; the
warning names the switch, so a mixed total can be found in the log).

The wrapper stays installed for the rest of the process (Megatron keeps no handle to undo it). It
corrects only the args of an OLMo-3 run: ``olmo3_window_size_form`` is set by the OLMo-3 args patch
alone, so a later non-OLMo-3 run in the same process gets upstream's count. A second OLMo-3 install
in the same process keeps the installed wrapper (around the upstream function it wrapped first)
and resets its per-run state; the new run is then checked at install time, as a first install
does.
"""

import copy
from functools import lru_cache

from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
    olmo3_window_from_settings,
    parse_olmo3_skip_freq,
)
from primus.backends.megatron.patches.olmo3_args_patches import is_olmo3_model_type
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

__all__ = [
    "olmo3_sliding_layer_count",
    "olmo3_mean_visible_keys",
    "olmo3_saved_keys",
    "olmo3_attention_flops_correction",
    "upstream_counts_window",
]


def olmo3_sliding_layer_count(num_layers, megatron_window, skip_freq) -> int:
    """Sliding layers: the layers upstream core attention gives ``megatron_window`` (TE / local
    ``is_layer_window_attention``, 1-indexed, the rule ``olmo3_layer_is_sliding`` uses).

    ``skip_freq`` is parsed and checked by :func:`parse_olmo3_skip_freq` (the args patch rules):
    an int ``2 <= k <= num_layers``, a 0/1 list of ``num_layers`` entries, or their string forms.
    """
    # Imported here: patch modules are imported before Megatron is on sys.path.
    from megatron.core.transformer.utils import is_layer_window_attention

    num_layers = int(num_layers)
    skip = parse_olmo3_skip_freq(skip_freq, num_layers)
    window = None if megatron_window is None else tuple(megatron_window)
    return sum(1 for n in range(1, num_layers + 1) if is_layer_window_attention(window, skip, n))


def _window_keys(window) -> int:
    """HF W (keys incl. the current token); the same W > 1 rule as the args patch."""
    return olmo3_window_from_settings(None, window)[1]


def olmo3_mean_visible_keys(seq_len: int, window: int) -> float:
    """Mean keys per query under a causal window of ``window`` keys (HF W, current token included)."""
    s, w = int(seq_len), _window_keys(window)
    if w >= s:
        return (s + 1) / 2.0
    # i + 1 for i < w, then w
    return (w * (w + 1) / 2.0 + (s - w) * w) / s


def olmo3_saved_keys(seq_len: int, window: int) -> float:
    """Keys per query a causal window of ``window`` keys skips, in upstream's continuous
    convention: ``s / 2 - (W - W**2 / (2 s)) = (s - W)**2 / (2 s)`` for ``W < s``, else 0."""
    s, w = int(seq_len), _window_keys(window)
    if w >= s:
        return 0.0
    return (s - w) ** 2 / (2.0 * s)


def _hashable(v):
    return tuple(_hashable(x) for x in v) if isinstance(v, (list, tuple)) else v


@lru_cache(maxsize=8)
def _correction_per_sample(
    window_size, sliding_window, form, skip_freq, num_layers, seq_length, kv_channels, num_attention_heads
) -> float:
    # The window as the args patch reads it (same function, same forms): (W - 1, 0) and W.
    megatron_window, w = olmo3_window_from_settings(window_size, sliding_window, form)
    n_sliding = olmo3_sliding_layer_count(num_layers, megatron_window, skip_freq)
    s = int(seq_length)
    q_size = int(kv_channels) * int(num_attention_heads)
    # 3 (fwd + bwd) * 2 (FMA) * q_size * keys * 2 (QK^T and PV), per token
    per_token = 3 * 2 * q_size * olmo3_saved_keys(s, w) * 2
    return float(s * per_token * n_sliding)


def olmo3_attention_flops_correction(args, batch_size) -> float:
    """FLOPs to subtract from upstream's count (>= 0): attention work the window skips.

    Per sliding layer, upstream's causal ``s / 2`` keys per query minus the windowed count in
    the same (continuous) convention; see the module docstring. The window is read with
    ``olmo3_window_from_settings`` (the args patch's reader, any declared form); the per-sample
    value is computed once per setting and scaled by ``batch_size``.
    """
    per_sample = _correction_per_sample(
        _hashable(getattr(args, "window_size", None)),
        getattr(args, "olmo3_sliding_window", None),
        getattr(args, "olmo3_window_size_form", None),
        _hashable(getattr(args, "window_attn_skip_freq", None)),
        args.num_layers,
        args.seq_length,
        args.kv_channels,
        args.num_attention_heads,
    )
    return float(batch_size) * per_sample


def upstream_counts_window(original_fn, args) -> bool:
    """True if upstream's ``num_floating_point_operations`` already depends on the attention
    window: its count for ``args`` changes when ``window_size`` and ``window_attn_skip_freq`` are
    removed. Runs on copies (upstream writes to ``args``, e.g. ``num_query_groups``)."""
    with_window, without = copy.copy(args), copy.copy(args)
    without.window_size, without.window_attn_skip_freq = None, None
    return original_fn(with_window, 1) != original_fn(without, 1)


def upstream_causal_keys_ok(original_fn, args) -> bool:
    """True if upstream counts core attention as ``seq_length / 2`` keys per query, the convention
    :func:`olmo3_saved_keys` subtracts from. Upstream's count without the window is
    ``c + a * s + b * s**2`` in the sequence length ``s`` (the second difference below cancels
    ``c`` and ``a``); only core attention gives the ``s**2`` term, so ``b`` must be ``3 (fwd + bwd) * 2 (FMA) * q_size * 1 / 2 (keys per query and token) * 2
    (QK^T and PV)`` per layer, ``q_size = kv_channels * num_attention_heads``. Runs on copies."""

    def count(s):
        a = copy.copy(args)
        a.window_size, a.window_attn_skip_freq, a.seq_length = None, None, s
        return float(original_fn(a, 1))

    s = 1024
    b = (count(3 * s) - 2 * count(2 * s) + count(s)) / (2.0 * s * s)
    layers = int(args.num_layers) + int(getattr(args, "mtp_num_layers", None) or 0)
    want = 6.0 * int(args.kv_channels) * int(args.num_attention_heads) * layers
    return abs(b - want) <= 1e-6 * want


def _check_at_install(check_upstream, run_args):
    """Run the upstream check at install time when the built args allow it. Upstream's count reads
    ``padded_vocab_size``, which the tokenizer sets after this phase: without it the check waits
    for the first count."""
    if run_args is not None and getattr(run_args, "padded_vocab_size", None) is not None:
        check_upstream(run_args, final=False)


def _installed_wrapper(fn):
    """The OLMo-3 wrapper if ``fn`` or any function it wraps (``__wrapped__`` chain) is it, else None."""
    seen = set()
    while fn is not None and id(fn) not in seen:
        if getattr(fn, "_olmo3_flops_patched", False):
            return fn
        seen.add(id(fn))
        fn = getattr(fn, "__wrapped__", None)
    return None


@register_patch(
    "megatron.olmo3.flops",
    backend="megatron",
    phase="before_train",
    description="OLMo-3: count windowed (not full causal) attention on sliding-window layers.",
    condition=is_olmo3_model_type,
)
def patch_olmo3_flops(ctx: PatchContext):
    import megatron.training.training as training_module

    # Try the correction once now so a problem shows in the log at patch time (also on a second
    # install in the same process). The FLOPs count is reporting only (the throughput log and the
    # running total num_floating_point_operations_so_far that the checkpoint stores; see the module
    # docstring), so a failure here never stops the run: the window / skip settings are validated
    # by megatron.olmo3.args, and a correction that still fails at the first count turns itself off
    # with a warning.
    run_args = (getattr(ctx, "extra", None) or {}).get("backend_args")  # Megatron's built args
    if run_args is not None and getattr(run_args, "num_layers", None) is not None:
        try:
            olmo3_attention_flops_correction(run_args, 1)
        except Exception as e:  # noqa: BLE001 - reporting only; must not stop training
            warning_rank_0(
                f"[Patch:megatron.olmo3.flops] cannot compute the correction at install time "
                f"({type(e).__name__}: {e}); retried at the first FLOPs count"
            )

    installed = _installed_wrapper(training_module.num_floating_point_operations)
    if installed is not None:
        # a second install in the same process (a new run): keep the one wrapper (it still wraps
        # the same upstream function: anything installed since then wraps it in turn), reset its
        # per-run state, and check this run now, as a first install does
        installed._olmo3_state.update(checked=False, apply=True)
        _check_at_install(installed._olmo3_check_upstream, run_args)
        return
    original_fn = training_module.num_floating_point_operations

    state = {"checked": False, "apply": True}

    def disable(why):
        state["checked"], state["apply"] = True, False
        warning_rank_0(
            f"[Patch:megatron.olmo3.flops] {why}; the OLMo-3 sliding-window correction is not applied"
        )

    def check_upstream(args, final=True):
        # Once per run: the correction is applied only to an upstream count that does not depend on
        # the window and counts causal attention as s / 2 keys per query (the convention the
        # correction subtracts from). The count is reporting only, so a check never stops training.
        # At install time (final=False) the args may be incomplete: a failure there is logged and
        # the check is retried at the first count. A failure at the first count turns the
        # correction off (the count stays upstream's, never corrected unchecked).
        if not state["checked"]:
            try:
                counts_window = upstream_counts_window(original_fn, args)
                keys = upstream_causal_keys_ok(original_fn, args)
            except Exception as e:  # noqa: BLE001 - reporting only; must not stop training
                why = f"cannot check upstream num_floating_point_operations ({type(e).__name__}: {e})"
                if final:
                    disable(why)
                else:
                    log_rank_0(f"[Patch:megatron.olmo3.flops] {why}; retried at the first FLOPs count")
                return
            state["checked"] = True
            if counts_window:
                disable("upstream num_floating_point_operations already depends on window_size")
            elif not keys:
                disable(
                    "upstream num_floating_point_operations does not count causal attention as "
                    "seq_length / 2 keys per query"
                )
            else:
                log_rank_0(
                    "[Patch:megatron.olmo3.flops] sliding-window layers count windowed attention FLOPs"
                )

    _check_at_install(check_upstream, run_args)

    # The patch is installed only when this run's model_type is olmo3 (the condition). The wrapper
    # stays installed for the rest of the process; it corrects only args objects of an OLMo-3 run:
    # ``olmo3_window_size_form`` is set by the OLMo-3 args patch only (pretrain() later replaces
    # args.model_type with the ModelType enum, so model_type cannot be read here).
    def wrapped(args, batch_size):
        if getattr(args, "olmo3_window_size_form", None) is None:
            return original_fn(args, batch_size)
        check_upstream(args)
        total = original_fn(args, batch_size)
        if not state["apply"]:
            return total
        try:
            correction = olmo3_attention_flops_correction(args, batch_size)
        except Exception as e:  # noqa: BLE001 - reporting only; must not stop training
            disable(f"cannot compute the correction ({type(e).__name__}: {e})")
            return total
        return total - correction

    wrapped.__wrapped__ = original_fn
    wrapped._olmo3_flops_patched = True
    wrapped._olmo3_state = state
    wrapped._olmo3_check_upstream = check_upstream
    training_module.num_floating_point_operations = wrapped
    if not state[
        "checked"
    ]:  # no built args, or padded_vocab_size not set yet: check_upstream runs at the first count
        log_rank_0("[Patch:megatron.olmo3.flops] installed; the upstream check runs at the first FLOPs count")
