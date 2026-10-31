###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 builder + ``model_provider`` (dispatched by ``get_model_provider('olmo3')``).

``model_provider`` delegates to Megatron's own ``model_provider`` (``record_memory_history``,
OOM snapshots, ...) with :func:`olmo3_builder` as the builder. ``olmo3_builder`` follows
upstream ``gpt_builder``: validate, build or check the config, pick the spec, build the model.
"""

import json
import os
import warnings
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
)
from primus.core.utils.arg_utils import parse_optional_switch

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
        # fields (sliding window W and its form, YaRN on full layers, RoPE layout).
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
    yarn = config.olmo3_yarn_enabled
    tokens = []
    for n in range(1, config.num_layers + 1):
        sliding = olmo3_layer_is_sliding(config, n)
        with_yarn = yarn and (not sliding or config.olmo3_rope_layout == "yarn_all")
        tokens.append(("S" if sliding else "F") + ("+yarn" if with_yarn else ""))
    return f"W={config.olmo3_sliding_window} rope={config.olmo3_rope_layout} " + " ".join(tokens)


def _core_attention_window(core_attention):
    """The window the built core attention applies: TE keeps ``window_size``; upstream's local
    DotProductAttention keeps it on its ``scale_mask_softmax``. ``"full"`` = no window (full causal
    attention), ``[left, right]`` = a window, ``"unknown"`` = not readable."""
    for owner in (core_attention, getattr(core_attention, "scale_mask_softmax", None)):
        if owner is not None and hasattr(owner, "window_size"):
            w = owner.window_size
            # None, or TE's "no window" forms (-1, -1) / (-1, 0)
            return "full" if w is None or int(w[0]) < 0 else [int(x) for x in w]
    return "unknown"  # no window attribute (another TE version, a Turbo spec): not an error


def _core_attention_scale(core_attention):
    """The softmax scale the built core attention applies. Local DotProductAttention keeps it on
    itself; TE's DotProductAttention hands it to its backends (flash / fused / unfused), which
    keep it. ``None`` if no module keeps one; ``"mixed"`` if they disagree (reported, not raised)."""
    found = {
        float(m.softmax_scale)
        for m in core_attention.modules()
        if isinstance(getattr(m, "softmax_scale", None), (int, float))
    }
    if len(found) > 1:
        return "mixed"
    return found.pop() if found else None


def _read_or_unknown(read):
    """``read()``, or ``"unknown"`` if it raises (a module tree this report does not know)."""
    try:
        return read()
    except Exception:  # diagnostic only
        return "unknown"


def olmo3_built_layer_report(model) -> list:
    """Diagnostic, opt-in (see olmo3_builder): per decoder layer of this model chunk, the window and
    softmax scale its core attention really uses, and which RoPE table it takes. Read from the
    modules, not the config, so a test or log reader sees what runs.

    Each value is read on its own; a value that cannot be read (missing attribute, unexpected type)
    is reported as ``"unknown"``. A model without ``decoder.layers`` gives ``[]``; without ``config``
    each layer's ``rope`` is ``"unknown"``. So the function does not raise for an unexpected module
    tree."""
    report = []
    yarn = _read_or_unknown(lambda: bool(model.config.olmo3_yarn_enabled))  # factor 1: tables agree
    layers = _read_or_unknown(lambda: list(model.decoder.layers))
    for layer in layers if isinstance(layers, list) else []:
        att = getattr(layer, "self_attention", None)
        core = getattr(att, "core_attention", None)
        scale = _read_or_unknown(lambda: _core_attention_scale(core))
        source = "core_attention"
        if scale is None and isinstance(getattr(att, "softmax_scale", None), (int, float)):
            # Some TE builds keep no scale attribute on the module tree; then report the scale the
            # layer handed to its core attention (the softmax_scale kwarg; see Olmo3SelfAttention).
            scale, source = float(att.softmax_scale), "passed"

        def rope():
            if yarn == "unknown":
                return "unknown"
            return "yarn" if bool(att.olmo3_use_yarn) and yarn else "plain"

        report.append(
            {
                "layer": _read_or_unknown(lambda: int(layer.layer_number)),
                "window": _read_or_unknown(lambda: _core_attention_window(core)),
                "softmax_scale": scale,
                "softmax_scale_source": source,
                "rope": _read_or_unknown(rope),
            }
        )
    return report


def _check_supplied_config(config):
    """A config passed in may have been changed after ``__post_init__``: check it again."""
    if not isinstance(config, Olmo3TransformerConfig):
        raise TypeError(f"OLMo-3 needs an Olmo3TransformerConfig, got {type(config).__name__}")
    check_olmo3_unsupported(config)
    if config.olmo3_sliding_window is None:
        raise ValueError("OLMo-3 config needs olmo3_sliding_window (the HF sliding window W)")
    olmo3_window_from_settings(config.window_size, config.olmo3_sliding_window, config.olmo3_window_size_form)
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
    if layer_report_enabled():
        _print_built_layer_report(model, vp_stage)
    return model


# Opt-in diagnostic, set by tests/trainer/test_megatron_trainer.py::test_olmo_3_1025_7B. A switch
# (arg_utils.parse_optional_switch: 1 / true / yes / on, 0 / false / no / off, any case; unset or "" is off;
# another value is off with a warning: a diagnostic never stops a run).
OLMO3_LAYER_REPORT_ENV = "PRIMUS_OLMO3_LAYER_REPORT"


def layer_report_enabled() -> bool:
    """True when ``PRIMUS_OLMO3_LAYER_REPORT`` is on. A value that is not a switch counts as off
    and logs one warning, so a typo cannot fail the run after the model is built."""
    try:
        return parse_optional_switch(os.environ.get(OLMO3_LAYER_REPORT_ENV), OLMO3_LAYER_REPORT_ENV)
    except ValueError as e:
        warnings.warn(f"{e}; the OLMo-3 layer report is off", stacklevel=2)
        return False


def _print_built_layer_report(model, vp_stage):
    """One JSON line per model chunk built on global rank 0, i.e. the layers of the first pipeline
    stage only; with virtual pipeline stages each chunk prints its own line, labelled with its
    ``vp_stage``. A failure to read the modules is logged; it never stops the build."""
    label = "" if vp_stage is None else f" (vp_stage {vp_stage})"
    try:
        report = json.dumps(olmo3_built_layer_report(model))
    except Exception as e:  # diagnostic only
        print_rank_0(f"[Primus:OLMo-3] built layers{label}: not reported ({type(e).__name__}: {e})")
        return
    print_rank_0(f"[Primus:OLMo-3] built layers{label}: {report}")


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

    Delegates to upstream ``model_provider`` so its setup (``record_memory_history``, ...) runs.
    """
    if model_builder is None:
        model_builder = olmo3_builder
    args = get_args()
    # Checked here too: upstream model_provider takes its ModelOpt branch (has_nvidia_modelopt and
    # modelopt_enabled) without calling model_builder, so olmo3_builder's checks would not run.
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
