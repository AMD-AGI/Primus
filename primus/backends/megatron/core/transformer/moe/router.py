###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial
from typing import Tuple

import torch
from megatron.core.transformer.moe.moe_utils import (
    get_capacity,
    sequence_load_balancing_loss_func,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args
from primus.backends.megatron.core.extensions.logits_processor import fused_softcap


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
        num_tokens, num_experts = logits.shape
        topk = self.config.moe_router_topk
        drop_policy = self.config.moe_token_drop_policy

        scores, probs, routing_map = pt.ops.fused_group_topk_routing_with_aux_score(
            logits,
            self.config.moe_router_topk,
            self.config.moe_router_num_groups,
            self.config.moe_router_group_topk,
            self.config.moe_router_score_function,
            self.config.moe_router_topk_scaling_factor,
        )

        routing_map = routing_map.bool()

        # drop by capacity
        capacity_factor = self.config.moe_expert_capacity_factor
        if capacity_factor is not None:
            # TopK with capacity
            expert_capacity = get_capacity(
                num_tokens=num_tokens * topk, num_experts=num_experts, capacity_factor=capacity_factor
            )

            # Maskout exceeded tokens
            if drop_policy == "probs":
                _, capacity_indices = torch.topk(probs, k=expert_capacity, dim=0, sorted=False)
                capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1).bool()
            elif drop_policy == "position":
                _, capacity_indices = torch.topk(routing_map.int(), k=expert_capacity, dim=0, sorted=False)
                capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1).bool()
            else:
                raise ValueError(f"Invalid drop_policy: {drop_policy}")

            pad_to_capacity = self.config.moe_pad_expert_input_to_capacity
            if pad_to_capacity:
                routing_map = capacity_mask
                probs = probs * routing_map
            else:
                # Get exceed mask and maskout exceeded probs and indices
                routing_map = torch.logical_and(routing_map, capacity_mask)
                probs = probs * routing_map

        # cal auxiliary loss
        aux_loss_func = partial(
            sequence_load_balancing_loss_func,
            probs=scores,
            routing_map=routing_map,
            batch_size=bsz,
            seq_length=seq_length,
            topk=self.topk,
        )

        probs = self.apply_load_balancing_loss(activation=probs, load_balancing_loss_func=aux_loss_func)

        return probs, routing_map

    def routing(self, logits: torch.Tensor):
        args = get_args()

        if args.router_logit_softcapping is not None and args.router_logit_softcapping > 0.0:
            fused_softcap(logits, args.router_logit_softcapping)


        if args.enable_primus_turbo and args.moe_use_fused_router_with_aux_score:
            scores, routing_map = self.fused_router_and_auxiliary_loss(logits)
        else:
            scores, routing_map = super().routing(logits)

        assert routing_map.dtype == torch.bool, "routing_map should be boolean"
        # profile for moe
        if args.moe_router_force_load_balancing:
            indices = (
                torch.arange(routing_map.size(0) * self.topk, device=routing_map.device).view(
                    routing_map.size(0), self.topk
                )
                % self.num_experts
            )
            row = torch.arange(routing_map.size(0), device=routing_map.device).repeat_interleave(self.topk)
            col = indices.view(-1)
            routing_map = torch.zeros_like(routing_map, dtype=torch.bool).index_put_(
                (row, col), torch.ones(1, device=routing_map.device, dtype=torch.bool)
            )

        return scores, routing_map
