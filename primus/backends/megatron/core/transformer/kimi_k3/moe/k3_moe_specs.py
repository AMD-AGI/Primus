###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Spec factory for the Kimi K3 Stable Latent MoE FFN.

Mirrors ``_build_ffn_spec`` (``deepseek_v4_layer_specs.py:527-584``): pick
the grouped-expert backend and the shared-expert linears from the spec
provider, assemble the submodule dataclass, return a ``ModuleSpec``. The
layer-spec tree that consumes this lives in ``kimi_k3_layer_specs.py``;
this file is deliberately importable on its own so the MoE block
can be built and tested without the rest of the stack.

Two upstream contracts shape the result and are worth stating:

* the **token dispatcher is not a submodule**. ``MoELayer.__init__``
  selects it from ``config.moe_token_dispatcher_type``
  (``moe_layer.py:224-248``); a ``token_dispatcher`` spec slot would be
  silently ignored.
* the **router is a builder callable, not a ``ModuleSpec``**.
  ``moe_layer.py:192`` calls ``submodules.router(config=...,
  pg_collection=..., is_mtp_layer=...)``, so wrapping :class:`TopKRouter`
  in a ``ModuleSpec`` would raise. The default on
  :class:`MoESubmodules` is already the class itself; leave it alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch  # noqa: F401  # must precede any transformer_engine import
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec

from primus.backends.megatron.core.transformer.kimi_k3.moe.k3_quantile_balancing import (
    quantile_balancing_enabled,
    resolve_quantile_balancing_router,
)
from primus.backends.megatron.core.transformer.kimi_k3.moe.k3_stable_latent_moe import (
    StableLatentMoE,
    StableLatentMoESubmodules,
    resolve_latent_size,
)

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        KimiK3SpecProvider,
    )

__all__ = ["build_stable_latent_moe_spec"]


def build_stable_latent_moe_spec(
    *,
    config: "TransformerConfig",
    provider: Optional["KimiK3SpecProvider"] = None,
) -> ModuleSpec:
    """Return the ``ModuleSpec`` for one Kimi K3 Stable Latent MoE layer.

    Args:
        config: the runtime Kimi K3 config.
        provider: spec provider. Defaults to the one cached on ``config``
            by ``resolve_k3_provider``.

    Returns:
        A ``ModuleSpec`` for :class:`StableLatentMoE`, carrying no
        ``params``: the parent's ``__init__`` takes ``layer_number``, which
        Megatron threads through ``set_layer_number`` after the build
        (``transformer_layer.py:395-397``), not through the spec.
    """
    if provider is None:
        from primus.backends.megatron.core.models.kimi_k3.build_context import (
            resolve_k3_provider,
        )

        provider = resolve_k3_provider(config)

    experts_module, experts_submodules = provider.k3_grouped_mlp_modules(
        moe_use_grouped_gemm=bool(config.moe_grouped_gemm),
        moe_use_legacy_grouped_gemm=bool(getattr(config, "moe_use_legacy_grouped_gemm", False)),
    )
    assert experts_module is not None, "Kimi K3 MoE requires a grouped-expert module."
    experts_spec = (
        ModuleSpec(module=experts_module)
        if experts_submodules is None
        else ModuleSpec(module=experts_module, submodules=experts_submodules)
    )

    # Shared experts stay in MODEL space: MLP passes is_expert=False for them,
    # which is what makes mlp.py:210 skip moe_latent_size. Their width is
    # moe_shared_expert_intermediate_size == moe_intermediate_size *
    # num_shared_experts (modeling_kimi_linear.py:797-801).
    shared_experts_spec = ModuleSpec(
        module=SharedExpertMLP,
        submodules=MLPSubmodules(
            linear_fc1=provider.column_parallel_linear(),
            linear_fc2=provider.row_parallel_linear(),
            activation_func=provider.k3_mlp_activation_func(),
        ),
    )

    submodules = StableLatentMoESubmodules(
        experts=experts_spec,
        shared_experts=shared_experts_spec,
    )
    if resolve_latent_size(config) is not None and bool(config.latent_moe_use_norm):
        submodules.latent_norm = ModuleSpec(module=provider.k3_norm_module())

    # Quantile Balancing needs routing *scores*, which upstream never
    # keeps, so the router has to stash a per-expert margin histogram beside
    # local_tokens_per_expert. Filling the slot here rather than leaving the
    # dataclass default is also what stops the Primus router patch from
    # swapping our class back out: it only replaces a class literally named
    # "TopKRouter" (moe_patches/topk_router_patches.py:185-186), and the QB
    # class is built *on top of* whatever that patch installed.
    if quantile_balancing_enabled(config):
        submodules.router = resolve_quantile_balancing_router()

    return ModuleSpec(module=StableLatentMoE, submodules=submodules)
