###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Stable Latent MoE — the FFN sub-block of the Kimi K3 text backbone.

Reference: ``KimiSparseMoeBlock`` (``modeling_kimi_linear.py``) and
``KimiMoEGate``.

What "latent" means
-------------------
The 896 routed experts do **not** run in model space. A single pair of
projections, shared by every expert in the layer, bottlenecks the token
into a ``routed_expert_hidden_size = 3584`` latent space; the experts live
there; one shared projection lifts the combined result back to 7168::

    identity = hidden_states
    topk_idx, topk_w = gate(hidden_states)          # FULL-WIDTH router
    h = routed_expert_down_proj(hidden_states)      # 7168 -> 3584
    y = moe_infer(h, topk_idx, topk_w)              # experts in latent
    y = routed_expert_norm(y)                       # the "STABLE" trick
    y = routed_expert_up_proj(y)                    # 3584 -> 7168
    y = y + shared_experts(identity)                # bypasses the latent

Per expert this is ``3 x 3584 x 3072 ~ 33.0 M`` parameters instead of
``3 x 7168 x 3072 ~ 66.1 M`` in model space — the bottleneck halves the
expert cost, which is what makes 896 experts affordable.

Three things are easy to get wrong, and all three are asserted by
``tests/unit_tests/megatron/transformer/kimi_k3/test_stable_latent_moe.py``:

1. The router scores the **full-width** hidden state. ``self.gate(...)``
   runs *before* the down-projection, so the router weight is
   ``[num_experts, hidden_size]`` and never
   ``[num_experts, latent]``.
2. The norm is applied to the **already-combined, top-k-weighted sum** of
   expert outputs, inside the bottleneck, before the up-projection — not
   per-expert and not after the up-projection.
3. The shared experts consume ``identity``, i.e. the **pre**-down-projection
   hidden state, at width ``moe_intermediate_size * num_shared_experts``
   (3072 * 2 = 6144) — note ``moe_intermediate_size``, not
   ``intermediate_size`` (33792).

Why this class is so short
--------------------------
Upstream Megatron **already implements the latent bottleneck**, under the
name ``config.moe_latent_size``:

* ``moe_layer.py`` builds ``fc1_latent_proj`` / ``fc2_latent_proj``
  as duplicated ``TELinear`` (shared across experts, as K3 requires);
* ``moe_layer.py`` applies the down-projection in ``preprocess``,
  which runs **after** ``route`` — so the router already sees
  the full-width hidden state;
* ``moe_layer.py`` applies the up-projection in ``postprocess``,
  after the dispatcher combine;
* ``moe_layer.py`` computes the shared experts on the original
  ``hidden_states``, before ``route`` / ``preprocess`` — the K3 ``identity``
  bypass, for free;
* ``experts.py`` (TEGroupedMLP) and ``mlp.py``
  (SequentialMLP) size the routed experts on ``moe_latent_size`` while
  leaving ``is_expert=False`` shared experts in model space.

The **only** thing upstream is missing is the norm between combine and
up-projection. So ``StableLatentMoE`` subclasses :class:`MoELayer`, maps
``routed_expert_hidden_size`` onto ``moe_latent_size``, and overrides
``postprocess`` to insert that one call. With
``routed_expert_hidden_size=None`` the class runs the parent's code path
untouched, so it is *identical by construction* to a stock
:class:`MoELayer` rather than merely numerically close.

An earlier design sketched hand-rolled ``linear_latent_down`` /
``linear_latent_up`` / ``token_dispatcher`` submodule slots; those turned out
redundant against the upstream this tree actually vendors, so they are not used.

The router
----------
No new router class. ``KimiMoEGate`` is DeepSeek-V3's ``noaux_tc``, and
``moe_utils.topk_routing_with_score_function`` is the same math line for
line: sigmoid over an fp32 gating GEMM, ``expert_bias`` shifts
**selection only** while the returned weights are gathered from the
un-shifted scores, renormalize by the top-k sum with the same ``1e-20``
epsilon, then scale. Configure it with
``moe_router_score_function="sigmoid"`` +
``moe_router_enable_expert_bias=True``. The release has
``num_expert_group == topk_group == 1``, so HF's group-limited branch is
dead code; leave ``moe_router_num_groups`` / ``moe_router_group_topk`` at
``None`` and no machinery is built for it.

Load balancing
--------------
Phase 1 uses ``seq_aux_loss`` at ``1e-3`` plus the ``noaux_tc`` expert
bias, i.e. DeepSeek-V4's configuration. Kimi
K3's published rule is "Quantile Balancing" (tech report §2.3.3, Eq. 14),
which has no public reference implementation; Primus implements it in
``k3_quantile_balancing.py``. It is a
*bias-update rule*, not a loss, so it does not belong in this module: it
replaces ``get_updated_expert_bias`` at
``megatron/core/distributed/finalize_model_grads.py``, which is
the single place the sign-based DeepSeek-V3 update is applied to
``TopKRouter.expert_bias``. See :data:`QUANTILE_BALANCING_HOOK_SITE`.
"""

from __future__ import annotations

import logging
from copy import copy
from dataclasses import dataclass
from typing import Optional, Union

import torch  # noqa: F401  # must precede any transformer_engine import
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)

__all__ = [
    "StableLatentMoE",
    "StableLatentMoESubmodules",
    "QUANTILE_BALANCING_HOOK_SITE",
    "resolve_latent_size",
]


@dataclass
class StableLatentMoESubmodules(MoESubmodules):
    """Spec tree for :class:`StableLatentMoE`.

    Extends upstream :class:`MoESubmodules` (``moe_layer.py``) —
    ``experts`` / ``shared_experts`` / ``router`` keep their upstream
    meaning and are consumed by the parent ``__init__`` — with the one
    slot Kimi K3 adds.

    Note that ``router`` is a *builder callable*, not a ``ModuleSpec``:
    ``moe_layer.py`` invokes it as
    ``submodules.router(config=..., pg_collection=..., is_mtp_layer=...)``.
    Leave it at the :class:`TopKRouter` default.

    There is deliberately no slot for the latent down / up projections or
    for the token dispatcher: upstream owns all three
    (``moe_layer.py``), the latter
    keyed off ``config.moe_token_dispatcher_type``.

    Args:
        latent_norm: RMSNorm over ``routed_expert_hidden_size``, applied to
            the combined routed output before the up-projection. Built with
            ``(config=..., hidden_size=latent, eps=config.layernorm_epsilon)``,
            which is :class:`TENorm`'s signature
            (``transformer_engine.py``). Required when
            ``config.latent_moe_use_norm`` is set; ``IdentityOp`` or ``None``
            both mean "no norm".
    """

    latent_norm: Optional[Union[ModuleSpec, type]] = None


def resolve_latent_size(config: TransformerConfig) -> Optional[int]:
    """Return the routed-expert latent width, or ``None`` for model space.

    ``KimiK3TransformerConfig.routed_expert_hidden_size`` is the K3-native
    field name (HF: ``routed_expert_hidden_size``); ``moe_latent_size`` is
    upstream's name for the same quantity
    (``transformer_config.py``). Either may be set; when both are,
    they must agree.
    """
    k3 = getattr(config, "routed_expert_hidden_size", None)
    upstream = getattr(config, "moe_latent_size", None)
    if k3 is not None and upstream is not None and int(k3) != int(upstream):
        raise ValueError(
            "Kimi K3 routed_expert_hidden_size="
            f"{k3} disagrees with moe_latent_size={upstream}. They name the "
            "same latent width; set one or set both to the same value."
        )
    latent = k3 if k3 is not None else upstream
    if latent is None:
        return None
    latent = int(latent)
    if latent <= 0:
        raise ValueError(f"routed_expert_hidden_size must be > 0 when set, got {latent}")
    return latent


# Where Quantile Balancing replaces the phase-1 rule, recorded so the
# location does not have to be re-derived.
#
# Kimi K3 does not update ``e_score_correction_bias`` with DeepSeek-V3's
# ``+/- rate * sign(violation)``. It sets the bias directly from a per-expert
# quantile of the routing *margin* (an expert's score minus the ``(k+1)``-th
# score of the token), estimated from an all-reduced histogram at the
# ``1 - k/n`` quantile (tech report §2.3.3, Eq. 14).
#
# That is a bias-update rule rather than a loss, so it does not belong in this
# module. The one site to replace is ``get_updated_expert_bias`` as called
# below, which already gathers ``local_tokens_per_expert`` / ``expert_bias``
# from every router in the model and writes the result back in place. Quantile
# Balancing needs the routing *scores* rather than token counts, so it also
# needs a per-step statistic stashed on ``TopKRouter`` alongside
# ``local_tokens_per_expert`` (``router.py``, zeroed at
# ``finalize_model_grads.py``).
QUANTILE_BALANCING_HOOK_SITE = "megatron/core/distributed/finalize_model_grads.py"


class StableLatentMoE(MoELayer):
    """Kimi K3 MoE FFN: an :class:`MoELayer` with a norm inside the bottleneck.

    Layer-assembly note for the K3 layer class. Upstream
    ``TransformerLayer`` decides whether to forward ``pg_collection`` and
    ``is_mtp_layer`` to the mlp with an **identity** check against
    ``(MoELayer, TEGroupedMLP, SequentialMLP)`` (``transformer_layer.py``),
    so a subclass does not match and falls back to
    ``get_default_pg_collection()`` (``moe_layer.py``) — correct, but
    it silently ignores an explicitly-threaded collection. ``KimiK3Layer``
    should pass ``pg_collection`` itself. The neighbouring
    ``isinstance(self.mlp, MoELayer)`` does match, so
    ``is_moe_layer`` comes out right either way.

    Args:
        config: the runtime Kimi K3 config. ``routed_expert_hidden_size``
            selects the latent width and ``latent_moe_use_norm`` the norm;
            everything else is read by the parent.
        submodules: :class:`StableLatentMoESubmodules`. Required, as for
            the parent.
        layer_number: 1-based layer number, threaded by Megatron's spec
            lifecycle. Must be set before a training-mode forward — the
            aux-loss tracker indexes by it (``router.py``).
        pg_collection: Megatron process groups. ``None`` falls back to
            ``get_default_pg_collection()`` (``moe_layer.py``).
        is_mtp_layer: forwarded to the parent. Always ``False`` for K3 —
            the release has no MTP.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[StableLatentMoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ) -> None:
        assert submodules is not None, "StableLatentMoE requires explicit submodules."
        latent = resolve_latent_size(config)
        use_norm = bool(getattr(config, "latent_moe_use_norm", False))

        super().__init__(
            config=self._latent_config(config, latent),
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        self.latent_size = latent
        self.routed_expert_norm = None
        if latent is not None and use_norm:
            norm_spec = getattr(submodules, "latent_norm", None)
            assert norm_spec is not None and norm_spec is not IdentityOp, (
                "config.latent_moe_use_norm is set but submodules.latent_norm is empty. "
                "The aggregated RMSNorm is the 'stable' half of Stable Latent MoE "
                "(modeling_kimi_linear.py) -- build the spec with "
                "build_stable_latent_moe_spec(), or pass latent_norm explicitly."
            )
            self.routed_expert_norm = build_module(
                norm_spec,
                config=self.config,
                hidden_size=latent,
                eps=self.config.layernorm_epsilon,
            )
        elif use_norm:
            # latent_moe_use_norm is meaningless without a bottleneck: HF gates
            # the norm on use_latent_moe too (modeling_kimi_linear.py).
            logger.warning(
                "[Kimi K3] latent_moe_use_norm is set but routed_expert_hidden_size "
                "is None; there is no bottleneck to normalise, so no norm is built. "
                "This matches modeling_kimi_linear.py."
            )

    @staticmethod
    def _latent_config(config: TransformerConfig, latent: Optional[int]) -> TransformerConfig:
        """Return the config the parent should build against.

        Upstream keys the whole latent path off ``config.moe_latent_size``,
        so K3's ``routed_expert_hidden_size`` has to reach it. When the
        caller already set ``moe_latent_size`` the config is used as-is;
        otherwise a shallow copy carries the value, following
        ``SharedExpertMLP.__init__`` rather than mutating a config the
        caller shares with the rest of the layer.

        Two things ride along on the copy:

        * ``moe_shared_expert_overlap`` is forced off. Upstream asserts
          against it in ``preprocess`` (``moe_layer.py``) whenever
          the latent projections are live, and that assert fires at *forward*
          time, i.e. after a full model has been built. Kimi K3's family
          YAML enables the overlap, so leaving it alone turns a config
          detail into a late crash.
        * Nothing else. In particular ``hidden_size`` stays at model width,
          which is what keeps the router at ``[num_experts, hidden_size]``.
        """
        if latent is None:
            return config
        if getattr(config, "moe_latent_size", None) == latent and not bool(config.moe_shared_expert_overlap):
            return config

        latent_config = copy(config)
        latent_config.moe_latent_size = latent
        if config.moe_shared_expert_overlap:
            logger.warning(
                "[Kimi K3] disabling moe_shared_expert_overlap for this MoE layer: "
                "upstream rejects it whenever the MoE latent projections are active "
                "(moe_layer.py). Set moe_shared_expert_overlap: false in the "
                "YAML to make the trade-off explicit."
            )
            latent_config.moe_shared_expert_overlap = False
        return latent_config

    def postprocess(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Norm the combined routed output, lift it back, add the shared experts.

        A copy of ``MoELayer.postprocess`` (``moe_layer.py``) with
        one call inserted. ``combine_postprocess`` has just produced the
        top-k-weighted sum of expert outputs *in latent space*, which is
        exactly HF's ``moe_infer`` return value — so normalising here and
        only here reproduces HF's norm-then-up-projection:

            y = routed_expert_norm(y)      # combined, weighted, in latent
            y = routed_expert_up_proj(y)   # then back to model space
        """
        output = self.token_dispatcher.combine_postprocess(output)

        if self.routed_expert_norm is not None:
            output = self.routed_expert_norm(output)

        if self.config.moe_latent_size:
            output, _ = self.fc2_latent_proj(output)

        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output
