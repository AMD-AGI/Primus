###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 model builder + ``model_provider`` entry points.

Mirrors ``deepseek_v4_builders.py``: ``model_provider`` is the thin wrapper
Megatron's ``pretrain()`` calls, and ``kimi_k3_builder`` does the actual
instantiation. Both live in one Primus-owned module so the dispatch in
``primus/core/utils/import_utils.py`` does not have to chase symbols across
``third_party/Megatron-LM``.

``KimiK3TransformerConfig`` carries every Kimi-K3-specific field; the two
plumbing helpers DeepSeek-V4 needs before config construction
(``_maybe_plumb_v4_sink_attention_args`` / ``_maybe_plumb_v4_turbo_deepep_args``)
have no Kimi K3 analogue — K3 has no sink attention and its MoE uses the
stock dispatcher — so there is nothing to derive here.
"""

from typing import Optional

from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.kimi_k3.kimi_k3_model import KimiK3Model
from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
    KimiK3TransformerConfig,
)

__all__ = [
    "kimi_k3_builder",
    "model_provider",
    "describe_latent_moe_resolution",
    "assert_latent_moe_width_reached_the_model",
]


def _resolve_runtime_decoder_spec(args, config: KimiK3TransformerConfig, vp_stage):
    """Resolve the effective runtime decoder spec for the Kimi K3 decoder.

    ``args.spec`` is the user override escape hatch, exactly as in
    ``deepseek_v4_builders.py``; otherwise the Kimi-K3-owned spec
    builder assembles the stage-local layer tree.
    """
    if args.spec is not None:
        return import_module(args.spec)

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_layer_specs import (
        get_kimi_k3_runtime_decoder_spec,
    )

    return get_kimi_k3_runtime_decoder_spec(config, vp_stage=vp_stage)


def describe_latent_moe_resolution(args, config, model) -> str:
    """One line stating the routed-expert latent width at all three layers.

    The stable latent MoE's bottleneck width is named twice --
    ``routed_expert_hidden_size`` (K3 / HF) and ``moe_latent_size`` (upstream) --
    and lives on two objects (``args`` and the config), with a reconciler on
    each. Upstream reads only ``config.moe_latent_size`` (``moe_layer.py``,
    ``experts.py``, ``mlp.py``) while ``training.py`` reads only
    ``args.moe_latent_size``, so the two can disagree and the disagreement is
    invisible: the model is shaped by one and the reported FLOPs/params by the
    other.

    This logs the config, the args copy **and the width the built weights
    actually have**, which is the only one of the three that cannot lie. Always
    on: it is one line per builder call and it is the line that would have made
    the args-phase defect obvious the day it landed.
    """
    parts = [
        f"config.routed_expert_hidden_size={getattr(config, 'routed_expert_hidden_size', None)}",
        f"config.moe_latent_size={getattr(config, 'moe_latent_size', None)}",
        f"args.routed_expert_hidden_size={getattr(args, 'routed_expert_hidden_size', None)}",
        f"args.moe_latent_size={getattr(args, 'moe_latent_size', None)}",
    ]

    parts.append(f"built.fc1_latent_proj={_first_latent_proj_shape(model)}")
    # Both expert paths, because they are *supposed* to differ: report §2.3 has
    # the shared experts keeping "a full-width path for common transformations"
    # while the routed experts "operate in a compact latent space of width l".
    # Reporting only one of them would make the correct asymmetry look like a
    # bug, or hide a real one.
    parts.append(f"built.routed_expert_in={_expert_input_width(model, routed=True)}")
    parts.append(f"built.shared_expert_in={_expert_input_width(model, routed=False)}")
    return "[Primus:Kimi-K3] latent MoE width -- " + ", ".join(parts)


def _first_latent_proj_shape(model):
    """``(name, shape)`` of the first ``fc1_latent_proj`` weight, or ``None``."""
    for name, module in model.named_modules():
        if name.endswith("fc1_latent_proj"):
            weight = getattr(module, "weight", None)
            if weight is not None:
                # TELinear(hidden -> latent), so the weight is [latent, hidden].
                return (name, tuple(weight.shape))
    return None


def _expert_input_width(model, *, routed: bool):
    """Input width of the first routed / shared expert ``linear_fc1``.

    Grouped and sequential experts store the weight differently -- ``TEGroupedMLP``
    keeps per-expert ``weight{i}`` attributes rather than a single ``weight`` --
    so the width is taken from ``in_features`` where the module exposes it and
    from the last dimension of any weight-shaped parameter otherwise.
    """
    for name, module in model.named_modules():
        if not name.endswith("linear_fc1"):
            continue
        is_shared = "shared_expert" in name
        if is_shared == routed:
            continue
        in_features = getattr(module, "in_features", None)
        if in_features is not None:
            return (name, int(in_features))
        for pname, param in module.named_parameters(recurse=False):
            if param.dim() == 2:
                return (f"{name}.{pname}", tuple(param.shape))
    return None


def assert_latent_moe_width_reached_the_model(config, model) -> None:
    """Fail loudly if the configured latent width is not the built width.

    The failure this guards against is silent by construction: a routed-expert
    stack built at ``hidden_size`` instead of ``routed_expert_hidden_size``
    trains perfectly well, converges, and reports a plausible loss -- it is just
    a different model from the one the yaml describes. Nothing downstream
    compares the two.

    ``fc1_latent_proj.weight`` is ``[latent, hidden]`` (``moe_layer.py``)
    and is only built at all when ``config.moe_latent_size`` is truthy, so its
    presence and its first dimension together certify the whole latent path.
    """
    expected = getattr(config, "moe_latent_size", None)
    hidden = int(config.hidden_size)

    projections = [
        (name, tuple(m.weight.shape))
        for name, m in model.named_modules()
        if name.endswith("fc1_latent_proj") and getattr(m, "weight", None) is not None
    ]

    if not expected:
        assert not projections, (
            "config.moe_latent_size is unset but the model built "
            f"{len(projections)} fc1_latent_proj: {projections[:2]}. The routed "
            "experts are running through a bottleneck the config does not declare."
        )
        return

    assert projections, (
        f"config.moe_latent_size={expected} but the model built no "
        "fc1_latent_proj, so every routed expert is running at hidden_size "
        f"({hidden}) instead of the configured latent width. This is the "
        "silent-wrong-shape failure: it trains and converges anyway."
    )
    bad = [(n, s) for n, s in projections if s != (int(expected), hidden)]
    assert not bad, (
        f"config.moe_latent_size={expected}, hidden_size={hidden}, so every "
        f"fc1_latent_proj weight must be {(int(expected), hidden)}; got {bad[:3]}."
    )


def kimi_k3_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config: Optional[KimiK3TransformerConfig] = None,
    pg_collection=None,
):
    """Build a Kimi K3 model."""
    print_rank_0("[Primus:Kimi-K3] building KimiK3Model...")

    if config is None:
        config = core_transformer_config_from_args(args, config_class=KimiK3TransformerConfig)

    # core_transformer_config_from_args silently replaces the requested
    # config_class with plain MLATransformerConfig when
    # args.multi_latent_attention is true (arguments.py), which
    # would drop every Kimi-K3 field without any error. Fail loudly instead.
    assert isinstance(config, KimiK3TransformerConfig), (
        f"Expected a KimiK3TransformerConfig, got {type(config).__name__}. "
        "This happens when the YAML sets multi_latent_attention: true — "
        "megatron/training/arguments.py then overrides config_class. "
        "Kimi K3 builds its own attention specs and must leave the flag false."
    )
    assert not args.use_legacy_models, "Kimi K3 requires use_legacy_models=False (Mcore-only)."

    runtime_decoder_spec = _resolve_runtime_decoder_spec(args, config, vp_stage)

    model = KimiK3Model(
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
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )

    print_rank_0(describe_latent_moe_resolution(args, config, model))
    assert_latent_moe_width_reached_the_model(config, model)

    return model


def model_provider(
    model_builder=None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config: Optional[KimiK3TransformerConfig] = None,
    pg_collection=None,
):
    """``model_provider`` entry point used by Megatron's ``pretrain()``.

    ``get_model_provider`` binds :func:`kimi_k3_builder` as the first arg
    via ``functools.partial`` so upstream ``pretrain()`` can call this with
    the standard ``(pre_process, post_process, vp_stage)`` signature.
    """
    if model_builder is None:
        model_builder = kimi_k3_builder

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
