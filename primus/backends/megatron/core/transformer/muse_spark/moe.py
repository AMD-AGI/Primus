###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark latent MoE and router.

HF ``MuseSparkMoE``::

    w, idx = gate(x)                      # sigmoid(fp32 logits * mult), bias for choice only
    h = pre_expert_norm(pre_expert_proj(x))          # centered RMSNorm on the latent
    y = sum_k w_k * post_expert_norm(expert_k(h))    # input-scaled RMSNorm per expert output
    out = post_expert_proj(y)

Megatron's ``MoELayer`` with ``moe_latent_size`` already has
``fc1_latent_proj`` (= pre_expert_proj) and ``fc2_latent_proj``
(= post_expert_proj). This module adds:

* :class:`MuseSparkTopKRouter`: fp32 gating scaled by ``moe_router_output_multiplier``.
* :class:`MuseSparkMoELayer`: ``pre_expert_norm`` after ``fc1_latent_proj``; it owns
  ``post_expert_norm`` (one weight shared by all experts).
* :class:`MuseSparkGroupedMLP`: ``TEGroupedMLP`` that applies ``post_expert_norm``
  to each expert output and then the routing prob (the backend applies the prob
  inside the activation, before ``linear_fc2``; the norm makes that order wrong
  here). The norm needs the full latent row, so expert tensor parallelism must be 1.
"""

from typing import Optional

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter

from primus.backends.megatron.core.transformer.muse_spark.norms import (
    MuseSparkCenteredRMSNorm,
    MuseSparkInputScaledRMSNorm,
)

__all__ = ["MuseSparkTopKRouter", "MuseSparkMoELayer", "MuseSparkGroupedMLP", "MuseSparkSequentialMLP"]


class _FSDP2LayoutBase(MegatronModule):
    """Empty leading base: lets FSDP2's ``fully_shard`` swap the router's class.

    Megatron's ``Router(ABC, MegatronModule)`` lists ``ABC`` first, so CPython rejects
    ``module.__class__ = FSDP<Router>`` ("object layout differs"). With this base first,
    the class's layout base is ``nn.Module`` like FSDPModule's. The MRO is unchanged otherwise.
    """


class MuseSparkTopKRouter(_FSDP2LayoutBase, TopKRouter):
    """Sigmoid top-k router; logits = fp32 linear * ``moe_router_output_multiplier``.

    The router weight is kept fp32 through Float16Module (HF keeps ``mlp.gate.weight`` fp32).
    Under FSDP2 the fp32 weight is its own parameter group (dtype islands in Primus's wrapper).
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.output_multiplier = float(getattr(config, "moe_router_output_multiplier", 1.0))

    def _apply(self, fn, *args, **kwargs):
        w = getattr(self, "weight", None)
        keep = w.detach().float().clone() if w is not None else None
        module = super()._apply(fn, *args, **kwargs)
        if keep is not None and self.weight.dtype != torch.float32:
            self.weight.data = keep.to(device=self.weight.device)
        return module

    def gating(self, input: torch.Tensor):
        # HF: F.linear(x.float(), W.float()) * output_multiplier. Plain torch fp32 GEMM
        # (TE's general_gemm finds no algorithm on ROCm for bf16 input x fp32 weight).
        if self.weight.device.type == "cpu":
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        bias = None if self.bias is None else self.bias.float()
        logits = torch.nn.functional.linear(input.float(), self.weight.float(), bias)
        return logits * self.output_multiplier


class _PostExpertNormMixin:
    """Run the experts with unit probs, then ``post_expert_norm`` and the routing prob."""

    def _init_post_expert_norm(self):
        etp = getattr(self.config, "expert_tensor_parallel_size", None)
        assert etp in (
            None,
            1,
        ), "Muse Spark post_expert_norm needs the full expert output row: use expert_tensor_parallel_size=1"
        # Set by MuseSparkMoELayer (not registered here: the weight belongs to the MoE layer).
        object.__setattr__(self, "_muse_post_expert_norm", None)

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        output, output_bias = super().forward(
            permuted_local_hidden_states, tokens_per_expert, torch.ones_like(permuted_probs)
        )
        norm = self._muse_post_expert_norm
        if norm is not None:
            output = norm(output)
        output = (output.float() * permuted_probs.float().unsqueeze(-1)).to(output.dtype)
        return output, output_bias


class MuseSparkGroupedMLP(_PostExpertNormMixin, TEGroupedMLP):
    """TEGroupedMLP (grouped GEMM) with the per-expert ``post_expert_norm``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_post_expert_norm()


class MuseSparkSequentialMLP(_PostExpertNormMixin, SequentialMLP):
    """SequentialMLP (one MLP per expert) with the per-expert ``post_expert_norm``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_post_expert_norm()


class MuseSparkMoELayer(MoELayer):
    """MoELayer with Muse Spark latent norms."""

    def __init__(self, config, submodules=None, layer_number: Optional[int] = None, *args, **kwargs):
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        latent = config.moe_latent_size
        assert latent, "Muse Spark MoE needs moe_latent_size"
        self.pre_expert_norm = MuseSparkCenteredRMSNorm(config, latent, eps=config.layernorm_epsilon)
        self.post_expert_norm = MuseSparkInputScaledRMSNorm(
            config, latent, eps=getattr(config, "post_norm_eps", 1e-8)
        )
        if isinstance(self.experts, _PostExpertNormMixin):
            object.__setattr__(self.experts, "_muse_post_expert_norm", self.post_expert_norm)
        else:
            raise TypeError("Muse Spark MoE needs MuseSparkGroupedMLP / MuseSparkSequentialMLP experts")

    def preprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor):
        hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states = self.pre_expert_norm(hidden_states)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(hidden_states, routing_map, probs)
        return hidden_states, probs
