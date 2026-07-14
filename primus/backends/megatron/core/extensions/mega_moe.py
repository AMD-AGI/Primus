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
from primus_turbo.pytorch.ops.moe.mega_moe_fused import mega_moe_fused


class PrimusTurboMegaMoELayer(MegatronModule):
    """Fused EP MoE layer: Megatron router -> mega_moe_fused experts -> shared expert."""

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

        # Per-rank expert shard: w1 [g, 2I, H] (gate+up), w2 [g, H, I]
        self.w1 = torch.nn.Parameter(
            torch.empty(
                (self.experts_per_rank, 2 * self.intermediate_size, self.hidden_size),
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16,
            )
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(
                (self.experts_per_rank, self.hidden_size, self.intermediate_size),
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16,
            )
        )
        self.reset_parameters()

        # experts are EP-sharded: skip DP allreduce hook when EP>1
        expert_parallel = self.ep_size > 1
        for p in (self.w1, self.w2):
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

        init1 = self.config.init_method
        init2 = self.config.output_layer_init_method or self.config.init_method
        with torch.no_grad(), rng_fork():
            if init1 is not None:
                init1(self.w1)
            else:
                self.w1.normal_(mean=0.0, std=2.0 / math.sqrt(self.hidden_size))
            if init2 is not None:
                init2(self.w2)
            else:
                self.w2.normal_(mean=0.0, std=2.0 / math.sqrt(self.intermediate_size))

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
        y = mega_moe_fused(
            self.ep_group,
            x,
            topk_idx.to(torch.int32),
            topk_weights.to(torch.float32),
            self.w1,
            self.w2,
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
        # experts w1/w2: EP-sharded on axis 0
        for name, weight in (("w1", self.w1), ("w2", self.w2)):
            key = f"{prefix}experts.{name}"
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
