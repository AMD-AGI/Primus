###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Tuple

import torch
from megatron.core.transformer.moe.moe_utils import apply_router_token_dropping
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args

from primus.backends.megatron.core.extensions.logits_processor import fused_softcap


_WARNED_EXPERT_BIAS = False


def _warn_fused_router_without_expert_bias():
    global _WARNED_EXPERT_BIAS
    if not _WARNED_EXPERT_BIAS:
        _WARNED_EXPERT_BIAS = True
        from primus.core.utils.module_utils import warning_rank_0

        warning_rank_0(
            "moe_use_fused_router_with_aux_score is ignored while moe_router_enable_expert_bias is on: "
            "Primus-Turbo's fused router cannot add the expert bias to the selection scores. Using "
            "Megatron's routing instead; set moe_router_fusion: true for its fused (TE) kernels."
        )


def force_balanced_slots(num_tokens: int, topk: int, num_experts: int, spread: bool, device) -> torch.Tensor:
    """[num_tokens, topk] expert ids of ``moe_router_force_load_balancing``'s deterministic modes.

    Both give every expert exactly ``num_tokens * topk / num_experts`` slots, the same every step.
    ``even``: ``(t * topk + k) % E``, each token on ``topk`` consecutive experts -- one EP rank
    whenever a rank holds at least ``topk`` experts, so per-rank-deduplicating dispatch (Mega MoE,
    DeepEP) moves far less than with real routing. ``spread``: ``(t + k * (E // topk)) % E``,
    the ``topk`` experts ``E // topk`` apart, so a token reaches ``min(topk, EP)`` ranks, close to
    the rank fan-out of real routing at that EP.
    """
    if spread:
        assert num_experts % topk == 0, f"spread needs num_experts ({num_experts}) % topk ({topk}) == 0"
        stride = num_experts // topk
        tokens = torch.arange(num_tokens, device=device).unsqueeze(1)
        return (tokens + torch.arange(topk, device=device).unsqueeze(0) * stride) % num_experts
    return torch.arange(num_tokens * topk, device=device).view(num_tokens, topk) % num_experts


class PrimusTopKRouter(TopKRouter):
    """Balanced route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig, *args, **kwargs) -> None:
        super().__init__(config=config, *args, **kwargs)

    def fused_router_and_auxiliary_loss(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            import primus_turbo.pytorch as pt
        except ImportError as e:
            raise ImportError("Failed to import 'primus_turbo'. Please make sure it is installed. ") from e

        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        scores_for_aux_loss, probs, routing_map = pt.ops.fused_group_topk_routing_with_aux_score(
            logits,
            self.config.moe_router_topk,
            self.config.moe_router_num_groups,
            self.config.moe_router_group_topk,
            self.config.moe_router_score_function,
            self.config.moe_router_topk_scaling_factor,
        )

        routing_map = routing_map.bool()

        # Apply token dropping to probs and routing_map.
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
            # todo: fuse the following logic into the turbo fused_group_topk_routing_with_aux_score OP
            # (the routing_map here differs from the one in the OP output is regarding less of group limit)
            if self.config.moe_router_num_groups is None or self.config.moe_router_num_groups <= 1:
                routing_map_for_aux_loss = routing_map
            else:
                _, top_indices_for_aux_loss = torch.topk(scores_for_aux_loss, k=self.topk, dim=1)
                routing_map_for_aux_loss = (
                    torch.zeros_like(logits).int().scatter(1, top_indices_for_aux_loss, 1).bool()
                )
            probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
            probs = self._apply_seq_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
            )
            probs = self._apply_global_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
        # Update expert bias and tokens_per_expert
        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)
        return probs, routing_map

    def routing(self, logits: torch.Tensor, **kwargs):
        args = get_args()

        if args.router_logit_softcapping is not None and args.router_logit_softcapping > 0.0:
            # grok2 router logit softcapping
            fused_softcap(logits, args.router_logit_softcapping)

        if args.enable_primus_turbo and args.moe_use_fused_router_with_aux_score and not self.enable_expert_bias:
            scores, routing_map = self.fused_router_and_auxiliary_loss(logits)
        else:
            if args.enable_primus_turbo and args.moe_use_fused_router_with_aux_score:
                # Primus-Turbo's fused router selects experts without the expert bias, which would
                # silently disable aux-loss-free load balancing; Megatron's routing applies it
                # (moe_router_fusion runs that on TE's fused kernels).
                _warn_fused_router_without_expert_bias()
            scores, routing_map = super().routing(logits, **kwargs)

        assert routing_map.dtype == torch.bool, "routing_map should be boolean"

        # NOTE on ``moe_router_force_load_balancing``:
        #
        # The previous implementation overwrote ``routing_map`` here with a
        # deterministic ``(token_idx * topk + k) % num_experts`` cycle while
        # leaving ``scores`` (the sparse top-k routing probabilities from
        # ``topk_routing_with_score_function``) untouched. Because ``scores``
        # is non-zero only on the *real* top-k experts, the deterministic
        # routing_map rarely overlapped with those non-zero positions, so the
        # MoE combine step multiplied almost every expert output by zero. On
        # a pretrained MoE checkpoint (e.g. DeepSeek-V2-Lite SFT) this
        # collapsed the MoE layer output to essentially just the shared
        # expert path, inflated iter-1 loss by several nats compared to
        # Megatron-Bridge, and broke Native-vs-Bridge A/B loss comparison.
        #
        # The correct (and Bridge-parity) way to do force-load-balancing is
        # to replace the *logits* with random values BEFORE routing, so that
        # both ``scores`` and ``routing_map`` are derived from the same
        # random logits and stay mutually consistent. That replacement is
        # already done in the upstream ``TopKRouter.forward`` via
        # ``apply_random_logits(logits)`` (see
        # third_party/Megatron-LM/megatron/core/transformer/moe/router.py),
        # so by the time we get here ``logits`` is already random when
        # ``args.moe_router_force_load_balancing`` is True. There is nothing
        # extra to do.
        # ``moe_router_force_load_balancing_type`` selects the balancing mode
        # (only relevant when ``moe_router_force_load_balancing`` is on):
        #   * "even"    -> deterministic round-robin so per-expert counts are
        #                  constant every step (constant M_total, no autotune/
        #                  recompile churn); handled by ``_force_even_routing``.
        #                  This must run with DeepEP too: its "even" token_indices
        #                  gather from ``scores``, which are zero off the real
        #                  top-k, and MegaMoE re-derives its top-k from them.
        #   * "spread"  -> as "even", but each token's experts are E // topk apart, so
        #                  a token reaches several EP ranks (see force_balanced_slots).
        #   * "uniform" -> Megatron-LM original random-logits balancing (already
        #                  applied upstream in ``TopKRouter.forward``; nothing to
        #                  do here).
        force_load_balancing_type = getattr(args, "moe_router_force_load_balancing_type", "uniform")
        if args.moe_router_force_load_balancing and force_load_balancing_type in ("even", "spread"):
            scores, routing_map = self._force_even_routing(
                scores, routing_map, spread=force_load_balancing_type == "spread"
            )

        return scores, routing_map

    def _force_even_routing(
        self, scores: torch.Tensor, routing_map: torch.Tensor, spread: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deterministically assign each token's top-k slots to experts in a
        round-robin cycle ``(token_idx * topk + k) % num_experts`` (``spread``:
        ``(token_idx + k * (num_experts // topk)) % num_experts``) so per-expert
        token counts are exactly balanced and step-invariant (constant M_total).

        The original top-k probability magnitudes are carried over to the new
        expert positions (per-slot), keeping ``scores`` and ``routing_map``
        mutually consistent so the MoE combine weights stay correct.
        """
        num_tokens = routing_map.shape[0]
        num_experts = self.config.num_moe_experts
        topk = self.topk
        device = routing_map.device

        # slot[t, k] = deterministic destination expert for token t's k-th slot, distinct
        # within a token while topk <= num_experts.
        slot = force_balanced_slots(num_tokens, topk, num_experts, spread, device)  # [num_tokens, topk]

        # Filled as uint8 and viewed as the (bool) routing map: on gfx1250 a bool fill of
        # [8192, 128] costs ~32 us standalone and ~100 us in a training step, a uint8 one ~5 us.
        new_routing_map = torch.zeros(routing_map.shape, dtype=torch.uint8, device=device)
        new_routing_map.scatter_(1, slot, 1)
        new_routing_map = new_routing_map.view(routing_map.dtype)

        # Carry the real per-token top-k probability magnitudes to the new slots.
        # scores is non-zero only on the real top-k, so topk() extracts exactly them.
        topk_vals, _ = torch.topk(scores, topk, dim=1)  # [num_tokens, topk]
        new_scores = torch.zeros_like(scores)
        new_scores.scatter_(1, slot, topk_vals.to(scores.dtype))

        return new_scores, new_routing_map


def primus_topk_router_for(router):
    """``PrimusTopKRouter`` if ``router`` is Megatron's ``TopKRouter`` class, else ``router`` unchanged.

    The swap rule of the ``megatron.moe.primus_topk_router`` patch, shared with MegaMoE. Matching the
    class by name also catches a ``TopKRouter`` that a module bound before the patch ran; subclasses
    such as ``InferenceTopKRouter`` and builders such as a ``functools.partial`` are left alone.
    """
    if isinstance(router, type) and router.__name__ == "TopKRouter":
        return PrimusTopKRouter
    return router
