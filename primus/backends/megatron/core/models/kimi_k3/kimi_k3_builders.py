###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 hooks for the shared builder (``MODEL_FAMILIES["kimi_k3"]`` in ``registry.py``).

The shared builder already checks that ``KimiK3TransformerConfig`` survived
``core_transformer_config_from_args``. K3 has no sink attention and its MoE
uses the stock dispatcher, so it needs no ``prepare_args`` hook; it only adds
the model's positional-embedding argument and a check that the configured
latent MoE width reached the built weights.
"""

from functools import partial

from megatron.training import print_rank_0

from primus.backends.megatron.core.models.builder import build_model, model_provider

__all__ = [
    "kimi_k3_builder",
    "model_provider",
    "kimi_k3_model_kwargs",
    "check_kimi_k3_model",
    "describe_latent_moe_resolution",
    "assert_latent_moe_width_reached_the_model",
]


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


def kimi_k3_model_kwargs(args) -> dict:
    return {"position_embedding_type": args.position_embedding_type}


def check_kimi_k3_model(args, config, model) -> None:
    print_rank_0(describe_latent_moe_resolution(args, config, model))
    assert_latent_moe_width_reached_the_model(config, model)


kimi_k3_builder = partial(build_model, "kimi_k3")
