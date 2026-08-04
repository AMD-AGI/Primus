###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 Mixture-of-Experts modules.

The K3 FFN is a **Stable Latent MoE**: 896 routed experts living in a
``routed_expert_hidden_size = 3584`` latent space behind a shared
down/up-projection pair, with an RMSNorm on the combined routed output
inside that bottleneck. Shared experts bypass the bottleneck entirely.

See :mod:`.k3_stable_latent_moe` for the module and
:mod:`.k3_moe_specs` for the spec factory that wires it.
"""

from primus.backends.megatron.core.transformer.kimi_k3.moe.k3_moe_specs import (
    build_stable_latent_moe_spec,
)
from primus.backends.megatron.core.transformer.kimi_k3.moe.k3_quantile_balancing import (
    QuantileBalancingMixin,
    compute_quantile_bias,
    make_quantile_balancing_router,
    quantile_balancing_enabled,
    resolve_quantile_balancing_router,
    update_router_expert_bias_quantile,
)
from primus.backends.megatron.core.transformer.kimi_k3.moe.k3_stable_latent_moe import (
    StableLatentMoE,
    StableLatentMoESubmodules,
    resolve_latent_size,
)

__all__ = [
    "StableLatentMoE",
    "StableLatentMoESubmodules",
    "build_stable_latent_moe_spec",
    "resolve_latent_size",
    # Quantile Balancing (WP10)
    "QuantileBalancingMixin",
    "compute_quantile_bias",
    "make_quantile_balancing_router",
    "quantile_balancing_enabled",
    "resolve_quantile_balancing_router",
    "update_router_expert_bias_quantile",
]
