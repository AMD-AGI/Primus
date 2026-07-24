###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fully fused MegaMoE layer, drop-in for Megatron ``MoELayer`` (EP-only, bf16).

Routing is delegated entirely to the Megatron router submodule
(``PrimusTopKRouter``): gating, z-loss, jitter, aux losses, force-load-balancing
and expert bias all live there, and the aux-loss gradient is baked into ``probs``
via ``MoEAuxLossAutoScaler``. This layer only converts the router's dense
``probs`` into sparse top-k and runs the fused expert kernel; the shared expert
reuses Megatron's ``SharedExpertMLP``.

The fused expert weights are exposed as two callable modules:

    experts.fc1_weight (owns w1): called before fused stage1
    experts.fc2_weight (owns w2): called before fused stage2

Calling each module triggers Megatron's DDP forward pre-hook at the matching
weight boundary. This lets w2's parameter all-gather overlap fused stage1 in
forward, while the staged autograd graph lets dW2's reduce-scatter overlap
stage1 backward. The kernel math is unchanged (see primus_turbo
``fused_mega_moe``).
"""

import contextlib
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group
from primus_turbo.pytorch.ops.moe.fused_mega_moe import (
    fused_mega_moe_stage1,
    fused_mega_moe_stage2,
)


class MegaMoEWeightModule(MegatronModule):
    """Callable expert-weight module used as a DDP overlap boundary.

    Calling the module triggers DDP's forward pre-hook before returning its
    weight, allowing the next weight's all-gather to overlap the current fused
    stage. Keeping each weight in a separate module also preserves per-weight
    gradient readiness, so DDP can overlap reduce-scatter with staged backward
    compute.
    """

    def __init__(self, config: TransformerConfig, weight_shape) -> None:
        super().__init__(config)
        device = torch.device("cpu") if config.use_cpu_initialization else torch.cuda.current_device()
        self.weight = torch.nn.Parameter(torch.empty(weight_shape, device=device, dtype=config.params_dtype))

    def forward(self) -> torch.Tensor:
        return self.weight

    def backward_dw(self) -> None:
        # Wgrad is produced inside the custom autograd backward.
        return None


class MegaMoEExperts(MegatronModule):
    """Two-stage fused expert with separately wrapped w1/w2 parameters."""

    def __init__(
        self,
        config: TransformerConfig,
        experts_per_rank: int,
        hidden_size: int,
        intermediate_size: int,
        ep_group,
    ) -> None:
        super().__init__(config)
        self.ep_group = ep_group
        # w1 [g, 2I, H] (gate+up) and w2 [g, H, I] (down projection).
        self.fc1_weight = MegaMoEWeightModule(config, (experts_per_rank, 2 * intermediate_size, hidden_size))
        self.fc2_weight = MegaMoEWeightModule(config, (experts_per_rank, hidden_size, intermediate_size))

    def forward(self, x, topk_idx, topk_weights):
        w1 = self.fc1_weight()
        l1_out, dwib, handle = fused_mega_moe_stage1(x, topk_idx, topk_weights, w1, self.ep_group)
        w2 = self.fc2_weight()
        return fused_mega_moe_stage2(l1_out, dwib, handle, topk_idx, topk_weights, w2, self.ep_group)

    def backward_dw(self) -> None:
        # Match native fc2-then-fc1 order (both no-ops here).
        self.fc2_weight.backward_dw()
        self.fc1_weight.backward_dw()


class PrimusTurboMegaMoELayer(MegatronModule):
    """Fused EP MoE layer: Megatron router -> two-linear fused experts -> shared expert."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[object] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ) -> None:
        super().__init__(config)

        assert (
            pg_collection is not None and pg_collection.ep is not None
        ), "MegaMoE requires an expert-parallel process group"
        assert submodules is not None, "MegaMoE requires MoESubmodules (router/shared_experts)"
        self._assert_supported_config(config)

        self.config = config
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer

        # EP topology: experts sharded evenly across the EP group
        self.ep_group = pg_collection.ep
        self.ep_size = self.ep_group.size()
        self.ep_rank = self.ep_group.rank()
        assert config.num_moe_experts % self.ep_size == 0, "num_experts must be divisible by EP size"
        self.experts_per_rank = config.num_moe_experts // self.ep_size

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_ffn_hidden_size

        # Router owns all routing logic (PrimusTopKRouter via the topk-router patch)
        self.router = submodules.router(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)

        # Separately wrapped w1/w2 provide DDP overlap boundaries.
        self.experts = MegaMoEExperts(
            config,
            self.experts_per_rank,
            self.hidden_size,
            self.intermediate_size,
            self.ep_group,
        )
        if config.perform_initialization:
            self.reset_parameters()

        # experts are EP-sharded: skip DP allreduce hook when EP>1
        expert_parallel = self.ep_size > 1
        for p in (self.experts.fc1_weight.weight, self.experts.fc2_weight.weight):
            setattr(p, "allreduce", not expert_parallel)

        # Optional shared expert: reuse Megatron SharedExpertMLP
        self.use_shared_expert = config.moe_shared_expert_intermediate_size is not None
        if self.use_shared_expert:
            self.shared_experts = build_module(
                submodules.shared_experts,
                config=config,
                pg_collection=pg_collection,
                gate=config.moe_shared_expert_gate,
            )
        else:
            self.shared_experts = None

    def reset_parameters(self) -> None:
        """Init expert weights via init_method, forked into the EP rng region."""
        tracker = get_cuda_rng_tracker() if get_cuda_rng_tracker().is_initialized() else None

        def rng_fork():
            if tracker is None:
                return contextlib.nullcontext()
            return tracker.fork(get_expert_parallel_rng_tracker_name())

        w1 = self.experts.fc1_weight.weight
        w2 = self.experts.fc2_weight.weight
        init1 = self.config.init_method
        init2 = self.config.output_layer_init_method or self.config.init_method
        with torch.no_grad(), rng_fork():
            if init1 is not None:
                init1(w1)
            else:
                w1.normal_(mean=0.0, std=2.0 / math.sqrt(self.hidden_size))
            if init2 is not None:
                init2(w2)
            else:
                w2.normal_(mean=0.0, std=2.0 / math.sqrt(self.intermediate_size))

    @staticmethod
    def _assert_supported_config(config: TransformerConfig) -> None:
        """Only kernel-level constraints; routing features are handled by the router."""
        assert config.tensor_model_parallel_size == 1, "MegaMoE is EP-only (TP==1)"
        assert config.params_dtype == torch.bfloat16, "MegaMoE only supports bf16"
        assert config.gated_linear_unit, "MegaMoE hardcodes a gated SwiGLU MLP"
        assert config.activation_func in (F.silu, torch.nn.SiLU), "MegaMoE hardcodes SiLU"
        assert (
            config.moe_expert_capacity_factor is None
        ), "MegaMoE is dropless; moe_expert_capacity_factor must be None"
        assert not config.add_bias_linear, "MegaMoE fused expert has no bias; set add_bias_linear=False"
        load_balancing_type = config.moe_router_load_balancing_type
        load_balancing_types = (
            load_balancing_type if isinstance(load_balancing_type, (list, tuple)) else [load_balancing_type]
        )
        assert "sinkhorn" not in load_balancing_types, "MegaMoE does not support sinkhorn load balancing"

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        in_shape = hidden_states.shape
        # Megatron convention: transpose [bsz, seq] -> [seq, bsz] to align with hidden_states
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()

        # Router returns dense probs [T, E] (aux-loss grad baked in) + routing_map
        probs, _ = self.router(hidden_states, padding_mask)
        probs = probs.reshape(-1, self.config.num_moe_experts)

        # dense -> sparse: exactly top_k entries are non-zero per token
        topk_weights, topk_idx = probs.topk(self.router.topk, dim=-1)

        x = hidden_states.reshape(-1, self.hidden_size).to(torch.bfloat16)
        y = self.experts(
            x,
            topk_idx,
            topk_weights.to(torch.float32),
        )
        y = y.reshape(in_shape).to(hidden_states.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(hidden_states)
        return y, None

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
        metadata: Optional[dict] = None,
    ):
        # delegated router/shared-expert sharded_state_dict read metadata["dp_cp_group"]
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        prepend_axis_num = len(sharded_offsets)
        edp_rank = parallel_state.get_expert_data_parallel_rank()
        expert_replica_id = (0, 0, edp_rank)

        sharded_sd: dict = {}
        # experts fc1_weight/fc2_weight: EP-sharded on axis 0 (keys match module path)
        for name, weight in (
            ("fc1_weight", self.experts.fc1_weight.weight),
            ("fc2_weight", self.experts.fc2_weight.weight),
        ):
            key = f"{prefix}experts.{name}.weight"
            sharded_sd[key] = ShardedTensor.from_rank_offsets(
                key,
                weight,
                *sharded_offsets,
                (prepend_axis_num, self.ep_rank, self.ep_size),
                replica_id=expert_replica_id,
                prepend_axis_num=prepend_axis_num,
            )
        # router + shared expert: delegate to their own sharded_state_dict
        sharded_sd.update(self.router.sharded_state_dict(f"{prefix}router.", sharded_offsets, metadata))
        if self.shared_experts is not None:
            sharded_sd.update(
                self.shared_experts.sharded_state_dict(f"{prefix}shared_experts.", sharded_offsets, metadata)
            )
        return sharded_sd

    def set_layer_number(self, layer_number: int) -> None:
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)

    def backward_dw(self, *args: object, **kwargs: object) -> None:
        return None
