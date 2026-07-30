###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MegaMoE layer, drop-in for Megatron MoELayer (EP-only, bf16)."""

import contextlib
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
    """Callable expert-weight module used as a DDP overlap boundary."""

    def __init__(self, config: TransformerConfig, weight_shape) -> None:
        super().__init__(config)
        device = torch.device("cpu") if config.use_cpu_initialization else torch.cuda.current_device()
        self.weight = torch.nn.Parameter(torch.empty(weight_shape, device=device, dtype=config.params_dtype))

    def forward(self) -> torch.Tensor:
        return self.weight

    def backward_dw(self) -> None:
        # wgrad produced in custom autograd backward
        return None


class MegaMoEExperts(MegatronModule):
    """Two-stage expert with separately wrapped w1/w2 parameters."""

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
        self.experts_per_rank = experts_per_rank
        # w1 [g, 2I, H] gate+up; w2 [g, H, I] down
        self.fc1_weight = MegaMoEWeightModule(config, (experts_per_rank, 2 * intermediate_size, hidden_size))
        self.fc2_weight = MegaMoEWeightModule(config, (experts_per_rank, hidden_size, intermediate_size))
        # Expert weights are already EP-sharded, so the DP allreduce hook must
        # not touch them.
        expert_parallel = config.expert_model_parallel_size > 1
        for p in (self.fc1_weight.weight, self.fc2_weight.weight):
            setattr(p, "allreduce", not expert_parallel)

    def reset_parameters(self, ep_rank: int) -> None:
        """Init expert weights exactly like Megatron TEGroupedLinear."""
        init_fc1 = self.config.init_method
        init_fc2 = self.config.output_layer_init_method
        assert init_fc1 is not None and init_fc2 is not None, "config init methods are unset"
        # fc1 <- init_method, fc2 <- output_layer_init_method
        weights = (
            (self.fc1_weight.weight, init_fc1),
            (self.fc2_weight.weight, init_fc2),
        )

        if self.config.use_cpu_initialization:
            # cpu rng is rank-identical: draw all experts, keep this rank's shard
            first_expert = ep_rank * self.experts_per_rank
            for weight, init_method in weights:
                master = torch.empty(weight.shape[1:], dtype=torch.float32)
                for e in range(self.config.num_moe_experts):
                    init_method(master)
                    if first_expert <= e < first_expert + self.experts_per_rank:
                        weight.data[e - first_expert].copy_(master)
            return

        tracker = get_cuda_rng_tracker()
        rng_fork = (
            tracker.fork(get_expert_parallel_rng_tracker_name())
            if tracker.is_initialized()
            else contextlib.nullcontext()
        )
        with rng_fork:
            # init each expert as a 2D [out, in] slice, like TE per-gemm weights
            for weight, init_method in weights:
                for expert_weight in weight.data:
                    init_method(expert_weight)

    def forward(self, x, topk_idx, topk_weights):
        w1 = self.fc1_weight()
        l1_out, dwib, handle = fused_mega_moe_stage1(x, topk_idx, topk_weights, w1, self.ep_group)
        w2 = self.fc2_weight()
        return fused_mega_moe_stage2(l1_out, dwib, handle, topk_idx, topk_weights, w2, self.ep_group)

    def backward_dw(self) -> None:
        # match native fc2-then-fc1 order
        self.fc2_weight.backward_dw()
        self.fc1_weight.backward_dw()


class PrimusTurboMegaMoELayer(MegatronModule):
    """EP MoE layer: Megatron router -> two-stage experts -> shared expert."""

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

        # experts sharded evenly across EP group
        self.ep_group = pg_collection.ep
        self.ep_size = self.ep_group.size()
        self.ep_rank = self.ep_group.rank()
        assert config.num_moe_experts % self.ep_size == 0, "num_experts must be divisible by EP size"
        self.experts_per_rank = config.num_moe_experts // self.ep_size

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_ffn_hidden_size

        # router owns all routing logic
        self.router = submodules.router(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)

        # separate w1/w2 modules give DDP overlap boundaries
        self.experts = MegaMoEExperts(
            config,
            self.experts_per_rank,
            self.hidden_size,
            self.intermediate_size,
            self.ep_group,
        )
        if config.perform_initialization:
            self.reset_parameters()

        # optional shared expert
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
        """Init expert weights exactly like Megatron TEGroupedLinear."""
        self.experts.reset_parameters(self.ep_rank)

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
        assert not config.init_model_with_meta_device, "MegaMoE does not support meta-device init"
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
        # transpose [bsz, seq] -> [seq, bsz]
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()

        # dense probs [T, E], aux-loss grad baked in
        probs, _ = self.router(hidden_states, padding_mask)
        probs = probs.reshape(-1, self.config.num_moe_experts)

        # dense -> sparse top-k
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
        # delegates below need metadata["dp_cp_group"]
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        prepend_axis_num = len(sharded_offsets)
        edp_rank = parallel_state.get_expert_data_parallel_rank()
        expert_replica_id = (0, 0, edp_rank)

        sharded_sd: dict = {}
        # expert weights: EP-sharded on axis 0
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
