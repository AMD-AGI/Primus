###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark modules layered on upstream Megatron classes.

Each class adds only what stock Megatron lacks for Muse Spark:

* :class:`MuseSparkQueryNorm` -- the upstream weightless query RMSNorm times
  ``muse_qk_scale_factor``.
* :class:`MuseSparkSelfAttention` -- weightless per-head RMSNorm on the core
  attention output before the upstream sigmoid output gate.
* :class:`MuseSparkTopKRouter` -- router logits scaled by ``muse_output_multiplier``.
* :class:`MuseSparkMoELayer` -- RMSNorm after the upstream latent
  down-projection, and a shared input-scaled RMSNorm on every expert output
  before the routing weight is applied. The norm is scale-invariant, so the
  weight must come after it; the experts therefore run with unit probs.
* :class:`MuseSparkTransformerLayer` -- sandwich post-norms and the
  variance-preserving residual gate, installed in place of the two
  bias-dropout-add callables.
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional, Union

import torch
from megatron.core.jit import jit_fuser
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import (
    maybe_skip_or_early_return_by_cudagraph,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

from primus.backends.megatron.core.models.muse_spark.muse_spark_ops import (
    input_scaled_rms_norm,
    residual_gate,
)
from primus.backends.megatron.core.models.muse_spark.muse_spark_transformer_config import (
    MuseSparkTransformerConfig,
)
from primus.backends.megatron.core.transformer.moe.router import PrimusTopKRouter

__all__ = [
    "MuseSparkInputScaledRMSNorm",
    "MuseSparkMoELayer",
    "MuseSparkMoESubmodules",
    "MuseSparkQueryNorm",
    "MuseSparkResidualGate",
    "MuseSparkSelfAttention",
    "MuseSparkTopKRouter",
    "MuseSparkTransformerLayer",
]

_fused_residual_gate = jit_fuser(residual_gate)
_fused_input_scaled_rms_norm = jit_fuser(input_scaled_rms_norm)


@jit_fuser
def _fused_input_scaled_rms_norm_weighted(
    x: torch.Tensor, weight: torch.Tensor, probs: torch.Tensor, eps: float
) -> torch.Tensor:
    return (input_scaled_rms_norm(x, weight, eps).float() * probs.unsqueeze(-1)).to(x.dtype)


def _param_device(config: MuseSparkTransformerConfig) -> torch.device:
    if config.use_cpu_initialization or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda", torch.cuda.current_device())


class MuseSparkInputScaledRMSNorm(torch.nn.Module):
    """``rmsnorm(x * (1 + weight))``; weight initialized to zero."""

    def __init__(self, config: MuseSparkTransformerConfig, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.zeros(hidden_size, dtype=config.params_dtype, device=_param_device(config))
        )
        setattr(self.weight, "sequence_parallel", config.sequence_parallel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _fused_input_scaled_rms_norm(x, self.weight, self.eps)

    def forward_weighted(self, x: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Norm each row of ``x``, then scale it by the matching entry of 1-D ``probs``."""
        return _fused_input_scaled_rms_norm_weighted(x, self.weight, probs, self.eps)


class MuseSparkResidualGate(torch.nn.Module):
    """Per-channel gate merging the residual and the post-normed sublayer output."""

    def __init__(self, config: MuseSparkTransformerConfig) -> None:
        super().__init__()
        self.temperature = config.muse_residual_gate_temperature
        self.eps = config.muse_residual_gate_eps
        self.gate = torch.nn.Parameter(
            torch.zeros(config.hidden_size, dtype=config.params_dtype, device=_param_device(config))
        )
        if config.perform_initialization:
            config.init_method(self.gate)
        setattr(self.gate, "sequence_parallel", config.sequence_parallel)

    def forward(self, residual: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return _fused_residual_gate(residual, hidden_states, self.gate, self.temperature, self.eps)


class MuseSparkQueryNorm(L2Norm):
    """Weightless query RMSNorm, then ``q * muse_qk_scale_factor``."""

    def __init__(self, hidden_size: int, config: MuseSparkTransformerConfig, eps: float = 1e-6, **kwargs):
        super().__init__(hidden_size=hidden_size, eps=eps, **kwargs)
        self.scale = config.muse_qk_scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * self.scale


class MuseSparkSelfAttention(SelfAttention):
    """``SelfAttention`` with Muse's ``o_norm`` between core attention and the output gate."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output_norm = L2Norm(
            hidden_size=self.hidden_size_per_attention_head, eps=self.config.layernorm_epsilon
        )

    def _apply_output_gate(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        heads = x.reshape(*x.shape[:-1], -1, self.hidden_size_per_attention_head)
        x = self.output_norm(heads).reshape(x.shape)
        return super()._apply_output_gate(x, gate)


class MuseSparkTopKRouter(PrimusTopKRouter):
    """Sigmoid top-k router whose fp32 logits are scaled before scoring."""

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        return super().gating(input) * self.config.muse_output_multiplier


@dataclass
class MuseSparkMoESubmodules(MoESubmodules):
    """Upstream MoE slots plus the norm applied after the latent down-projection.

    ``pre_expert_norm`` is built as ``(config=, hidden_size=moe_latent_size, eps=)``,
    which is the ``TENorm`` signature.
    """

    pre_expert_norm: Optional[Union[ModuleSpec, type]] = None


class MuseSparkMoELayer(MoELayer):
    """Latent MoE with Muse's pre-expert and post-expert norms."""

    def __init__(
        self,
        config: MuseSparkTransformerConfig,
        submodules: Optional[MuseSparkMoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ) -> None:
        assert submodules is not None and submodules.pre_expert_norm is not None, (
            "MuseSparkMoELayer needs MuseSparkMoESubmodules with pre_expert_norm set; "
            "build the spec with get_muse_spark_decoder_block_spec()."
        )
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        latent = config.moe_latent_size
        self.pre_expert_norm = build_module(
            submodules.pre_expert_norm,
            config=config,
            hidden_size=latent,
            eps=config.layernorm_epsilon,
        )
        # One weight for all experts. It lives outside ``self.experts``, so its
        # gradient is reduced over the full data-parallel group, EP ranks included.
        self.post_expert_norm = MuseSparkInputScaledRMSNorm(config, latent, config.muse_post_norm_eps)

    @maybe_skip_or_early_return_by_cudagraph("preprocess")
    def preprocess(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ):
        hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states = self.pre_expert_norm(hidden_states)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(hidden_states, routing_map, probs)
        return hidden_states, probs

    def routed_experts_compute(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        if getattr(self, "is_inference_cuda_graphed_iteration", False):
            raise NotImplementedError("Muse Spark does not support the CUDA-graphed inference MoE path.")
        dispatched_input, tokens_per_expert, permuted_probs = self.token_dispatcher.dispatch_postprocess(
            hidden_states, probs
        )
        expert_output, mlp_bias = self.experts(
            dispatched_input, tokens_per_expert, torch.ones_like(permuted_probs)
        )
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        expert_output = self.post_expert_norm.forward_weighted(expert_output, permuted_probs)
        output = self.token_dispatcher.combine_preprocess(expert_output)
        return output, mlp_bias


def _with_moe_kwargs(
    submodules: TransformerLayerSubmodules,
    pg_collection: ProcessGroupCollection,
    is_mtp_layer: bool,
) -> TransformerLayerSubmodules:
    """Thread ``pg_collection`` / ``is_mtp_layer`` to a ``MoELayer`` subclass.

    ``TransformerLayer`` forwards them only when the mlp spec module *is*
    ``MoELayer``, so a subclass would otherwise rebuild default process groups.
    """
    mlp = submodules.mlp
    if not (
        isinstance(mlp, ModuleSpec)
        and isinstance(mlp.module, type)
        and issubclass(mlp.module, MoELayer)
        and mlp.module is not MoELayer
    ):
        return submodules
    params = {**mlp.params, "pg_collection": pg_collection, "is_mtp_layer": is_mtp_layer}
    return dataclasses.replace(submodules, mlp=dataclasses.replace(mlp, params=params))


def _add_bias(x_with_bias):
    x, bias = x_with_bias
    return x if bias is None else x + bias


class MuseSparkTransformerLayer(TransformerLayer):
    """``TransformerLayer`` with post-norms and gated residuals.

    ``x = gate_attn(x, post_attn_norm(attn(input_norm(x))))`` then
    ``x = gate_mlp(x, post_mlp_norm(mlp(pre_mlp_norm(x))))``. The upstream
    forward is kept; only ``self_attn_bda`` / ``mlp_bda`` are replaced, which
    also covers the fine-grained a2a-overlap schedule that calls ``layer.mlp_bda``.
    """

    def __init__(
        self,
        config: MuseSparkTransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        is_mtp_layer: bool = False,
        **kwargs,
    ) -> None:
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        super().__init__(
            config=config,
            submodules=_with_moe_kwargs(submodules, pg_collection, is_mtp_layer),
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
            **kwargs,
        )
        if self.hidden_dropout != 0.0:
            raise ValueError(f"Muse Spark requires hidden_dropout == 0, got {self.hidden_dropout}.")

        self.post_self_attn_norm = L2Norm(hidden_size=config.hidden_size, eps=config.muse_post_norm_eps)
        self.post_self_attn_residual_gate = MuseSparkResidualGate(config)
        self.post_mlp_norm = MuseSparkInputScaledRMSNorm(
            config, config.hidden_size, config.muse_post_norm_eps
        )
        self.post_mlp_residual_gate = MuseSparkResidualGate(config)

        self.self_attn_bda = self._self_attn_gated_residual
        self.mlp_bda = self._mlp_gated_residual

    def _self_attn_gated_residual(self, training: bool, fused: bool):
        return self._apply_self_attn_gated_residual

    def _mlp_gated_residual(self, training: bool, fused: bool):
        return self._apply_mlp_gated_residual

    def _apply_self_attn_gated_residual(self, x_with_bias, residual: torch.Tensor, prob: float):
        hidden_states = self.post_self_attn_norm(_add_bias(x_with_bias))
        return self.post_self_attn_residual_gate(residual, hidden_states)

    def _apply_mlp_gated_residual(self, x_with_bias, residual: torch.Tensor, prob: float):
        hidden_states = self.post_mlp_norm(_add_bias(x_with_bias))
        return self.post_mlp_residual_gate(residual, hidden_states)
