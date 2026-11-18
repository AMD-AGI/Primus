###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling's shared-expert-sink router (``InklingTopkRouter`` in ``modeling_inkling.py``).

The checkpoint's ``mlp.gate.weight`` is ``[n_routed + n_shared, hidden]``:
the last ``n_shared`` rows score the shared experts. Here the routed rows stay
Megatron's ``weight`` (so expert-bias updates, sharding and recompute keep
working) and the shared rows become ``shared_weight``. ``expert_bias`` is the
checkpoint's ``e_score_correction_bias`` (aux-loss-free balancing).

Routing, per token::

    logits   = x @ [weight; shared_weight].T
    experts  = topk(sigmoid(logits_routed) + expert_bias, k)
    p        = sigmoid([logits_routed[experts], logits_shared])
    weights  = p / sum(p) * route_scale * global_scale

so the shared experts take part in the normalisation (the "sink") and their
weights are returned next to Megatron's ``(probs, routing_map)``.

The checkpoint stores ``global_scale`` and the bias in fp32; ``expert_bias``
stays fp32 through Megatron's own buffer handling. ``global_scale`` follows
``params_dtype`` like every other parameter (the distributed optimizer
supports one parameter dtype) unless ``inkling_fp32_activations`` and the
``keep_in_fp32`` contract are both on. The product is fp32 either way with
``moe_router_dtype: fp32``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from megatron.core.transformer.moe.moe_utils import router_gating_linear
from megatron.core.transformer.moe.router import TopKRouter

from primus.backends.megatron.core.transformer.inkling.mlp import inkling_global_scale
from primus.backends.megatron.core.transformer.keep_in_fp32 import KeepInFp32Mixin

__all__ = ["InklingTopKRouter"]


class InklingTopKRouter(KeepInFp32Mixin, TopKRouter):
    def __init__(self, config, *args, **kwargs):
        if not config.moe_router_enable_expert_bias:
            raise ValueError("Inkling's router needs moe_router_enable_expert_bias: true")
        self.num_shared_experts = int(config.moe_num_shared_experts)
        super().__init__(config, *args, **kwargs)
        unsupported = {
            "auxiliary load-balancing losses": self.is_aux_loss_enabled(),
            "sinkhorn routing": self.routing_type == "sinkhorn",
            "expert capacity / token dropping": config.moe_expert_capacity_factor is not None,
            "grouped top-k": bool(config.moe_router_num_groups),
            "routing replay": bool(config.moe_enable_routing_replay),
            "router bias (add_bias_linear)": self.bias is not None,
        }
        bad = [name for name, on in unsupported.items() if on]
        if bad:
            raise ValueError(f"Inkling's router does not support: {', '.join(bad)}")
        self.route_scale = float(config.moe_router_topk_scaling_factor or 1.0)

        self.shared_weight = torch.nn.Parameter(
            torch.empty((self.num_shared_experts, config.hidden_size), dtype=torch.float32)
        )
        self.global_scale = inkling_global_scale(config)
        setattr(self.shared_weight, "sequence_parallel", config.sequence_parallel)
        self._reset_inkling_parameters()

    def _reset_inkling_parameters(self):
        if self.config.perform_initialization and self.shared_weight.device.type != "meta":
            self.config.init_method(self.shared_weight)
        self.shared_weight.data = self.shared_weight.data.to(dtype=self.config.params_dtype)

    def _router_dtype(self, input: torch.Tensor) -> torch.dtype:
        return {"fp32": torch.float32, "fp64": torch.float64}.get(self.config.moe_router_dtype, input.dtype)

    def forward(
        self, input: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``input`` ``[s, b, hidden]``; ``padding_mask`` ``[s, b]``, ``True`` = padding.

        Returns ``probs`` ``[tokens, n_routed]`` (zero off the routing map),
        ``routing_map`` ``[tokens, n_routed]`` and ``shared_probs``
        ``[tokens, n_shared]``.
        """
        self._maintain_float32_expert_bias()
        input = self.apply_input_jitter(input)
        weight = torch.cat([self.weight, self.shared_weight], dim=0)
        logits = router_gating_linear(input, weight, None, self._router_dtype(input))
        logits = logits.view(-1, self.num_experts + self.num_shared_experts)
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1)

        routed, shared = logits.split([self.num_experts, self.num_shared_experts], dim=-1)
        routed = self.apply_z_loss(routed, padding_mask=padding_mask)

        scores_for_choice = routed.sigmoid() + self.expert_bias
        top_idx = torch.topk(scores_for_choice, self.topk, dim=-1, sorted=False).indices
        top_logits = torch.cat([routed.gather(-1, top_idx), shared], dim=-1)
        log_p = F.logsigmoid(top_logits)
        weights = torch.exp(log_p - torch.logsumexp(log_p, dim=-1, keepdim=True))
        weights = weights * self.route_scale * self.global_scale.to(weights.dtype)

        probs = torch.zeros_like(routed).scatter(-1, top_idx, weights[:, : self.topk])
        routing_map = torch.zeros_like(routed, dtype=torch.bool).scatter(-1, top_idx, True)
        shared_probs = weights[:, self.topk :]

        self._apply_expert_bias(routing_map, padding_mask=padding_mask)
        return probs, routing_map, shared_probs
