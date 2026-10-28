###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling MoE: routed experts plus router-weighted shared experts.

``InklingMoE`` in ``modeling_inkling.py`` scales each shared expert by its
own per-token weight from the router (the sink), where Megatron's
:class:`MoELayer` runs the shared experts before routing and gates them by a
separate sigmoid. :class:`InklingMoELayer` therefore routes first and hands
the shared weights to :class:`InklingSharedExperts`.

The shared experts are fused into one MLP of width ``n_shared * ffn``:
``linear_fc1`` is ``[gate_e0; gate_e1; ...; up_e0; up_e1; ...]``, so after
the SwiGLU channel block ``e`` belongs to shared expert ``e`` and is scaled
by that expert's weight before ``linear_fc2`` sums the experts.

``inkling_fp32_activations`` moves the SwiGLUs and routing-weight products
(routed and shared) and the sum over the top-k routed outputs plus the shared
output to fp32. Off, they run in the model dtype as in the transformers model.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

from primus.backends.megatron.core.transformer.inkling.mlp import fp32_swiglu

__all__ = ["InklingGroupedMLP", "InklingMoELayer", "InklingSharedExperts"]


class InklingGroupedMLP(TEGroupedMLP):
    """Routed experts whose activation optionally runs in fp32."""

    def bias_act_func(self, intermediate_parallel, bias_parallel, permuted_probs):
        if not self.config.inkling_fp32_activations:
            return super().bias_act_func(intermediate_parallel, bias_parallel, permuted_probs)
        if bias_parallel is not None:
            raise ValueError("Inkling's experts have no biases")
        return fp32_swiglu(intermediate_parallel, permuted_probs)


class InklingSharedExperts(SharedExpertMLP):
    def __init__(self, config, submodules, gate: bool = False, pg_collection=None):
        if gate:
            raise ValueError(
                "Inkling's shared experts are weighted by the router; set moe_shared_expert_gate: false"
            )
        super().__init__(config=config, submodules=submodules, gate=False, pg_collection=pg_collection)
        self.num_shared_experts = int(config.moe_num_shared_experts)
        self.expert_width = int(config.moe_ffn_hidden_size)
        tp_size = self.tp_group.size() if self.tp_group is not None else 1
        tp_rank = self.tp_group.rank() if self.tp_group is not None else 0
        local = self.num_shared_experts * self.expert_width // tp_size
        expert_of_channel = (tp_rank * local + torch.arange(local)) // self.expert_width
        self.register_buffer("expert_of_channel", expert_of_channel, persistent=False)

    def forward(self, hidden_states: torch.Tensor, shared_probs: torch.Tensor):
        """``hidden_states`` ``[s, b, hidden]``, ``shared_probs`` ``[s * b, n_shared]``."""
        fc1, _ = self.linear_fc1(hidden_states)
        scale = shared_probs.view(*fc1.shape[:-1], -1)
        scale = scale.index_select(-1, self.expert_of_channel.to(scale.device))
        if self.config.inkling_fp32_activations:
            intermediate = fp32_swiglu(fc1, scale)
        else:
            gate, up = fc1.chunk(2, dim=-1)
            intermediate = F.silu(gate) * up * scale.to(fc1.dtype)
        output, _ = self.linear_fc2(intermediate)
        return output


class InklingMoELayer(MoELayer):
    def __init__(self, config, submodules=None, layer_number=None, pg_collection=None, is_mtp_layer=False):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        if not self.use_shared_expert:
            raise ValueError("Inkling's MoE needs shared experts")
        if self.shared_expert_overlap:
            raise NotImplementedError("Inkling's MoE does not support moe_shared_expert_overlap")
        if config.moe_latent_size:
            raise NotImplementedError("Inkling's MoE has no latent projection")

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """``hidden_states`` ``[s, b, hidden]``; ``padding_mask`` ``[b, s]``, ``True`` = padding."""
        if intermediate_tensors is not None:
            raise NotImplementedError("Inkling's MoE does not support partial CUDA-graph capture")
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError("MoE with tensor parallelism needs sequence parallelism during training")
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()

        def custom_forward(hidden_states, padding_mask=None):
            probs, routing_map, shared_probs = self.router(hidden_states, padding_mask)
            if self.shared_experts_recompute:
                shared_output = tensor_parallel.checkpoint(
                    self.shared_experts, False, hidden_states, shared_probs
                )
            else:
                shared_output = self.shared_experts(hidden_states, shared_probs)
            routed, probs = self.preprocess(hidden_states, probs, routing_map)
            routed, probs = self.dispatch(routed, probs)
            output, mlp_bias = self.routed_experts_compute(routed, probs)
            output = self.combine(output)
            return self.postprocess(output, shared_output), mlp_bias

        if self.moe_layer_recompute and self.training:
            return tensor_parallel.checkpoint(custom_forward, False, hidden_states, padding_mask)
        return custom_forward(hidden_states, padding_mask)

    def postprocess(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        if not self.config.inkling_fp32_activations:
            return super().postprocess(output, shared_expert_output)
        dtype = output.dtype
        output = self.token_dispatcher.combine_postprocess(output.float())
        return (output + shared_expert_output.float()).to(dtype)
