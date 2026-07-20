###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 model builder + model_provider entry points.

In the upstream Megatron-LM convention there are two pieces:

* ``model_provider``: a thin wrapper that handles ``args.record_memory_history``,
  ModelOpt etc. and delegates to a ``model_builder`` callable. Defined once
  in ``Megatron-LM/model_provider.py``.
* ``<modeltype>_builders.py``: contains ``<modeltype>_builder(args, ...)`` --
  the actual model-class instantiation logic.

For DeepSeek-V4 we keep both in a single primus-owned module so the dispatch
in ``primus/core/utils/import_utils.py`` doesn't have to chase symbols across
``third_party/Megatron-LM``.

Phase 8 contract:
- Resolve V4 runtime decoder spec externally in builder.
- Pass runtime spec as ``transformer_layer_spec`` into ``DeepseekV4Model``.
- DeepseekV4Model is rooted at ``LanguageModule`` and has no GPT placeholder
  spec dependency.
"""

from typing import Optional

from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_layer_specs import (
    get_deepseek_v4_runtime_decoder_spec,
    is_v4_turbo_deepep_active,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_model import (
    DeepseekV4Model,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)


def _resolve_runtime_decoder_spec(
    args,
    config: DeepSeekV4TransformerConfig,
    vp_stage,
):
    """Resolve effective runtime decoder spec for DeepSeek-V4 decoder path."""
    if args.spec is not None:
        return import_module(args.spec)
    return get_deepseek_v4_runtime_decoder_spec(config=config, vp_stage=vp_stage)


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
            "[Primus:DeepSeek-V4][P22] derived args.use_sink_attention=True "
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
                f"[Primus:DeepSeek-V4][P22] attn_sliding_window={attn_sw} "
                f">= seq_length={seq_len}; dropping window for Turbo "
                "(mathematically equivalent to full causal attention)."
            )
        else:
            args.sink_sliding_window = 0
            print_rank_0(
                f"[Primus:DeepSeek-V4][P22] WARNING: "
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
            "[Primus:DeepSeek-V4][P23] derived args.moe_enable_deepep=True "
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
                f"[Primus:DeepSeek-V4][P23] override "
                f"args.moe_token_dispatcher_type='{current or 'unset'}' -> "
                f"'flex' (Turbo DeepEP dispatcher path)."
            )
    else:
        print_rank_0(
            "[Primus:DeepSeek-V4][P23] WARNING: "
            f"args.use_turbo_deepep=True but moe_token_dispatcher_type="
            f"'{current}' (not 'flex' / 'alltoall').  Keeping the user's "
            "dispatcher type; the Turbo DeepEP path will not engage for V4 "
            "MoE layers."
        )


def deepseek_v4_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config: Optional[DeepSeekV4TransformerConfig] = None,
    pg_collection=None,
):
    """Build a DeepSeek-V4 model.

    Phase 8: build from a DeepSeek runtime spec tree only.
    """
    print_rank_0("[Primus:DeepSeek-V4] building DeepseekV4Model...")

    # Plan-3 P22: plumb V4 attn_sink + attn_sliding_window into the
    # Turbo flash-attn sink-attention args before any V4 attention
    # module is constructed (PrimusTurboAttention reads these in
    # ``__init__`` from get_args()).
    _maybe_plumb_v4_sink_attention_args(args)
    # Plan-3 P23: plumb args.moe_enable_deepep + moe_token_dispatcher_type
    # for the Turbo DeepEP MoE dispatcher path BEFORE config construction
    # so the resulting ``config`` carries the right dispatcher type into
    # ``_pick_v4_dispatcher_cls``.
    _maybe_plumb_v4_turbo_deepep_args(args)

    if config is None:
        config = core_transformer_config_from_args(
            args,
            config_class=DeepSeekV4TransformerConfig,
        )

    assert not args.use_legacy_models, "DeepSeek-V4 requires use_legacy_models=False (Mcore-only)."

    runtime_decoder_spec = _resolve_runtime_decoder_spec(args, config, vp_stage)

    model = DeepseekV4Model(
        config=config,
        transformer_layer_spec=runtime_decoder_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )
    return model


def model_provider(
    model_builder=None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config: Optional[DeepSeekV4TransformerConfig] = None,
    pg_collection=None,
):
    """``model_provider`` entry point used by Megatron's ``pretrain()``.

    ``MegatronPretrainTrainer`` will pass ``deepseek_v4_builder`` as the
    first arg via ``functools.partial`` so the upstream ``pretrain()`` can
    call this with the standard ``(pre_process, post_process, vp_stage)``
    signature.
    """
    if model_builder is None:
        model_builder = deepseek_v4_builder

    args = get_args()
    if args.record_memory_history:
        import torch

        torch.cuda.memory._record_memory_history(
            True,
            trace_alloc_max_entries=100000,
            trace_alloc_record_context=True,
        )

    return model_builder(
        args,
        pre_process,
        post_process,
        vp_stage,
        config=config,
        pg_collection=pg_collection,
    )
