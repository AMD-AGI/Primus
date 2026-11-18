###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 hooks for the shared builder (``MODEL_FAMILIES["deepseek_v4"]`` in ``registry.py``).

The shared builder resolves the V4 runtime decoder spec and passes it as
``transformer_layer_spec`` into ``DeepseekV4Model``, which is rooted at
``LanguageModule``. V4 adds the Turbo args plumbing that must run before the
config is built, and the RoPE arguments of its model constructor.
"""

from functools import partial

from megatron.training import print_rank_0

from primus.backends.megatron.core.models.builder import (
    build_model,
    family_model_provider,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_layer_specs import (
    is_v4_turbo_deepep_active,
)

__all__ = [
    "deepseek_v4_builder",
    "model_provider",
    "prepare_deepseek_v4_args",
    "deepseek_v4_model_kwargs",
]


def _maybe_plumb_v4_sink_attention_args(args) -> None:
    """Plan-3 P22: derive Turbo sink-attention args from V4 config.

    :class:`PrimusTurboAttention` reads ``use_sink_attention``,
    ``sink_sliding_window`` and ``sink_window_even_layers_only`` directly
    from the global ``args`` namespace at module-init time.  For V4 we
    can derive all three from the V4 attention configuration:

    * ``use_sink_attention`` follows ``args.attn_sink`` (V4's per-head
      learnable softmax sink — set to ``true`` in the V4-Flash recipe).
    * ``sink_sliding_window`` is derived from ``args.attn_sliding_window``,
      *capped to seq_length*.  V4's dense layers use SWA with window 128
      in the released checkpoint; the released aiter Triton flash-attn
      backend (``aiter/ops/triton/attention/mha.py``) does not yet
      accept ``window_size != (-1, -1)`` and raises ``ValueError:
      Sliding Window is not supported yet in the Triton Backend``.
      We therefore zero out the window when it equals or exceeds
      ``seq_length`` (a window that covers the full causal triangle is
      mathematically equivalent to no window).  When the window is
      strictly shorter than the sequence length we *still* zero it out
      and emit a warning — Turbo can't honor it, and falling back to
      eager-Python every step would defeat the perf goal.  Long-context
      V4-Flash (seq > 128) needs an aiter SWA upgrade before Turbo can
      claim full V4 fidelity; tracked as a P22 follow-up.
    * ``sink_window_even_layers_only`` is hard-set to ``False`` because
      V4 applies SWA on every dense layer (unlike gpt-oss, which only
      windows even-numbered layers).

    The plumbing only fires when all of (a) Turbo is enabled, (b)
    ``use_turbo_attention=True``, (c) the user has not set the sink
    fields explicitly.  This keeps the V4-Flash YAML free of
    Turbo-internal knobs while still producing a correct attention
    forward when the user flips ``use_turbo_attention=true``.
    """
    if not getattr(args, "enable_primus_turbo", False):
        return
    if not getattr(args, "use_turbo_attention", False):
        return
    if not getattr(args, "attn_sink", False):
        # Without the V4 sink, Turbo would not honor SWA either; defer
        # to ``DeepseekV4Attention``'s eager-Python fallback for this
        # configuration (it asserts ``_use_core_attention=False`` when
        # ``attn_sink`` is off and ``attn_sliding_window > 0``).
        return

    if getattr(args, "use_sink_attention", None) in (None, False):
        args.use_sink_attention = True
        print_rank_0(
            "[Primus:deepseek_v4][P22] derived args.use_sink_attention=True "
            "from args.attn_sink=True (Turbo flash-attn sink path)."
        )

    if getattr(args, "sink_sliding_window", None) in (None, 0):
        attn_sw = int(getattr(args, "attn_sliding_window", 0) or 0)
        seq_len = int(getattr(args, "seq_length", 0) or 0)
        if attn_sw <= 0:
            args.sink_sliding_window = 0
        elif seq_len > 0 and attn_sw >= seq_len:
            # Window covers the full causal triangle — equivalent to no
            # window.  Drop it so Turbo's flash kernel runs without
            # window_size (avoids the aiter Triton SWA gap).
            args.sink_sliding_window = 0
            print_rank_0(
                f"[Primus:deepseek_v4][P22] attn_sliding_window={attn_sw} "
                f">= seq_length={seq_len}; dropping window for Turbo "
                "(mathematically equivalent to full causal attention)."
            )
        else:
            args.sink_sliding_window = 0
            print_rank_0(
                f"[Primus:deepseek_v4][P22] WARNING: "
                f"attn_sliding_window={attn_sw} < seq_length={seq_len} "
                "but aiter Triton flash-attn does not support sliding "
                "window yet (raises ValueError).  Setting "
                "sink_sliding_window=0 — V4 dense layers will attend "
                "to *all* causal-prior tokens instead of the windowed "
                "subset.  This deviates from V4-Flash math; for "
                "checkpoint-fidelity training, keep "
                "use_turbo_attention=False until aiter adds SWA support."
            )

    # gpt-oss windows only even layers; V4 windows all dense layers.
    args.sink_window_even_layers_only = False


def _maybe_plumb_v4_turbo_deepep_args(args) -> None:
    """Plan-3 P23: derive Turbo DeepEP MoE args from V4 config.

    When :func:`is_v4_turbo_deepep_active` returns True, the V4 MoE
    layers will be built with :class:`PrimusTurboDeepEPTokenDispatcher`
    (resolved by ``_pick_v4_dispatcher_cls`` in
    ``deepseek_v4_layer_specs.py``).  Two ``args`` fields control
    downstream behaviour:

    * ``moe_enable_deepep`` — asserted by
      :class:`PrimusTurboDeepEPTokenDispatcher.__init__`; without it
      the Turbo dispatcher refuses to build.
    * ``moe_token_dispatcher_type`` — read by
      :class:`DeepseekV4TransformerConfig` to populate the V4 config's
      ``moe_token_dispatcher_type``.  V4 spec build's
      ``_pick_v4_dispatcher_cls`` then sees ``"flex"`` and pairs the
      Turbo class with the right dispatcher-type label so
      ``DeepseekV4MoE._resolve_dispatcher_type_from_spec`` reports
      ``"flex"`` (instead of falling through to the
      ``"unsupported dispatcher module"`` warning + ``"alltoall"``
      fallback).

    The ``before_train`` Turbo MoE patch
    (``primus.backends.megatron.patches.turbo.moe_dispatcher_patches``)
    sets the same two fields, but it fires AFTER ``deepseek_v4_builder``
    has already built ``config`` from ``args``, so V4 needs to plumb
    these values BEFORE ``core_transformer_config_from_args`` runs.

    The plumbing is a no-op when:

    * Turbo is not enabled / DeepEP not requested / TP > 1, OR
    * the user has already set ``args.moe_token_dispatcher_type`` to
      something other than ``"alltoall"`` (we only override the V4
      base.yaml default, never an explicit user opt-in like
      ``"allgather"``).
    """
    if not is_v4_turbo_deepep_active(args):
        return

    if not bool(getattr(args, "moe_enable_deepep", False)):
        args.moe_enable_deepep = True
        print_rank_0(
            "[Primus:deepseek_v4][P23] derived args.moe_enable_deepep=True "
            "from args.use_turbo_deepep=True (Turbo DeepEP dispatcher path)."
        )

    current = str(getattr(args, "moe_token_dispatcher_type", "") or "").lower()
    # Only override when the V4 base default ("alltoall") is in effect
    # or the user already opted into "flex".  Respect explicit
    # "allgather" — that is a different parallelism scheme (tp+ep
    # all-gather/scatter), not a deepep flavour.
    if current in ("", "alltoall", "flex"):
        if current != "flex":
            args.moe_token_dispatcher_type = "flex"
            print_rank_0(
                f"[Primus:deepseek_v4][P23] override "
                f"args.moe_token_dispatcher_type='{current or 'unset'}' -> "
                f"'flex' (Turbo DeepEP dispatcher path)."
            )
    else:
        print_rank_0(
            "[Primus:deepseek_v4][P23] WARNING: "
            f"args.use_turbo_deepep=True but moe_token_dispatcher_type="
            f"'{current}' (not 'flex' / 'alltoall').  Keeping the user's "
            "dispatcher type; the Turbo DeepEP path will not engage for V4 "
            "MoE layers."
        )


def prepare_deepseek_v4_args(args) -> None:
    # Both must run before config construction: PrimusTurboAttention reads the
    # sink args from get_args() in __init__, and the config must already carry
    # the dispatcher type _pick_v4_dispatcher_cls keys off.
    _maybe_plumb_v4_sink_attention_args(args)
    _maybe_plumb_v4_turbo_deepep_args(args)


def deepseek_v4_model_kwargs(args) -> dict:
    return {
        "position_embedding_type": args.position_embedding_type,
        "rotary_percent": args.rotary_percent,
        "rotary_base": args.rotary_base,
        "rope_scaling": args.use_rope_scaling,
    }


deepseek_v4_builder = partial(build_model, "deepseek_v4")
model_provider = family_model_provider(deepseek_v4_builder)
