###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 builder + ``model_provider`` (dispatched by ``get_model_provider('olmo3')``).

``model_provider`` delegates to Megatron's own ``model_provider`` when Megatron has one
(``record_memory_history``, OOM snapshots, ...), else calls the builder directly (and refuses
``record_memory_history``); the builder is the caller's, :func:`olmo3_builder` by default. ``olmo3_builder`` follows
upstream ``gpt_builder``: validate, build or check the config, pick the spec, build the model.
"""

from typing import Optional

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.olmo3.olmo3_layer_specs import (
    get_olmo3_layer_spec,
)
from primus.backends.megatron.core.models.olmo3.olmo3_model import Olmo3Model
from primus.backends.megatron.core.models.olmo3.olmo3_transformer_config import (
    Olmo3TransformerConfig,
)
from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
    check_olmo3_unsupported,
    olmo3_window_from_settings,
    parse_olmo3_skip_freq,
)
from primus.backends.megatron.core.transformer.olmo3.olmo3_attention import (
    olmo3_layer_is_sliding,
    olmo3_layer_uses_yarn,
)

__all__ = ["olmo3_builder", "model_provider", "check_olmo3_build_args"]


def check_olmo3_build_args(args):
    """Refuse build settings the OLMo-3 builder does not implement (before any config is built)."""
    if getattr(args, "use_legacy_models", False):
        raise ValueError("OLMo-3 requires use_legacy_models=False (Mcore-only).")
    # The OLMo-3 spec owns post-norms, full-width q/k norms and the per-layer RoPE pick from the
    # stacked [2, ...] table; a generic spec would silently drop all of them.
    if getattr(args, "spec", None) is not None:
        raise ValueError(f"OLMo-3 builds its own layer spec; unset spec (got {args.spec!r}).")
    if getattr(args, "mtp_num_layers", None):
        raise ValueError("OLMo-3 has no MTP layers; unset mtp_num_layers.")
    if getattr(args, "yaml_cfg", None) is not None:
        # core_transformer_config_from_yaml builds a plain TransformerConfig without the OLMo-3
        # fields (sliding window W, YaRN on full layers, RoPE layout).
        raise ValueError(
            "OLMo-3 builds its config from args (Olmo3TransformerConfig); yaml_cfg is not supported."
        )
    if getattr(args, "modelopt_enabled", False):
        raise ValueError("OLMo-3 has no ModelOpt builder; unset modelopt_enabled.")
    if getattr(args, "experimental_attention_variant", None) is not None:
        raise ValueError("OLMo-3 attention is fixed; unset experimental_attention_variant.")
    if getattr(args, "heterogeneous_layers_config_path", None) is not None:
        raise ValueError("OLMo-3 layers are homogeneous; unset heterogeneous_layers_config_path.")
    # The same table the args patch and Olmo3TransformerConfig use. The window is checked on the
    # config (Olmo3TransformerConfig.__post_init__, or _check_supplied_config).
    check_olmo3_unsupported(args)


def olmo3_layer_plan(config) -> str:
    """One token per layer, 1-indexed: ``S`` = sliding window (W keys), ``F`` = full attention;
    ``+yarn`` marks layers with the YaRN table. E.g. ``W=4096 rope=per_type S S S F+yarn``."""
    tokens = []
    for n in range(1, config.num_layers + 1):
        sliding = olmo3_layer_is_sliding(config, n)
        with_yarn = olmo3_layer_uses_yarn(config, sliding)
        tokens.append(("S" if sliding else "F") + ("+yarn" if with_yarn else ""))
    return f"W={config.olmo3_sliding_window} rope={config.olmo3_rope_layout} " + " ".join(tokens)


def _check_supplied_config(config):
    """A config passed in may have been changed after ``__post_init__``: check it again."""
    if not isinstance(config, Olmo3TransformerConfig):
        raise TypeError(f"OLMo-3 needs an Olmo3TransformerConfig, got {type(config).__name__}")
    check_olmo3_unsupported(config)
    if config.olmo3_sliding_window is None:
        raise ValueError("OLMo-3 config needs olmo3_sliding_window (the HF sliding window W)")
    # __post_init__ stores the Megatron / TE window (W - 1, 0); a built config must hold exactly
    # that (the int W form is an input form only): anything else was
    # changed after __post_init__ and would hand TE a different window.
    want, _ = olmo3_window_from_settings(None, config.olmo3_sliding_window)
    ws = config.window_size
    have = tuple(ws) if isinstance(ws, (list, tuple)) else ws
    if have != want:
        raise ValueError(
            f"OLMo-3 config window_size {config.window_size!r} is not the Megatron/TE {want} for "
            f"olmo3_sliding_window={config.olmo3_sliding_window} (changed after __post_init__?)"
        )
    parse_olmo3_skip_freq(config.window_attn_skip_freq, config.num_layers)


def olmo3_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config: Optional[Olmo3TransformerConfig] = None,
    pg_collection=None,
):
    """Build an OLMo-3 model."""
    print_rank_0("[Primus:OLMo-3] building Olmo3Model...")
    check_olmo3_build_args(args)
    if config is None:
        config = core_transformer_config_from_args(args, config_class=Olmo3TransformerConfig)
    else:
        _check_supplied_config(config)
    layer_spec = get_olmo3_layer_spec()
    print_rank_0(f"[Primus:OLMo-3] layer plan: {olmo3_layer_plan(config)}")

    model = Olmo3Model(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=getattr(args, "rotary_percent", 1.0),
        rotary_base=args.rotary_base,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )
    return model


def _upstream_model_provider():
    try:
        from model_provider import model_provider as upstream
    except ModuleNotFoundError as e:
        if e.name != "model_provider":  # a dependency of model_provider is missing: a real error
            raise
        return None  # older Megatron: no model_provider module
    return upstream


def model_provider(
    model_builder=None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config: Optional[Olmo3TransformerConfig] = None,
    pg_collection=None,
):
    """``model_provider`` entry point used by Megatron's ``pretrain()``.

    Delegates to upstream ``model_provider`` when Megatron has one (so ``record_memory_history``,
    ... runs); otherwise calls ``model_builder`` (default :func:`olmo3_builder`) directly and raises
    ``ValueError`` if ``record_memory_history`` is set.
    """
    if model_builder is None:
        model_builder = olmo3_builder
    args = get_args()
    # Checked here as well as in olmo3_builder (so twice on the normal path; the scan is cheap):
    # upstream model_provider takes its ModelOpt branch (has_nvidia_modelopt and modelopt_enabled)
    # without calling model_builder, and olmo3_builder can also be called directly (one entry per
    # build path, not one per build).
    check_olmo3_build_args(args)
    upstream = _upstream_model_provider()
    if upstream is not None:
        return upstream(
            model_builder, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection
        )
    if getattr(args, "record_memory_history", False):
        raise ValueError("record_memory_history needs Megatron's model_provider module")
    return model_builder(
        args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection
    )
