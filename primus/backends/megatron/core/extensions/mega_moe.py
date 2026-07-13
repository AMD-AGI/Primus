###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Adapter: MegaMoE with internal load-balancing aux loss.

The fused ``MegaMoE`` kernel computes the load-balancing aux loss internally and
returns it as a scalar; this adapter logs the unscaled value to Megatron's aux
loss tracker and injects its gradient by scaling the MoE ``output`` (not the
routing weights), keeping the kernel free of framework state.
"""

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
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
)
from primus_turbo.pytorch.modules.moe.mega_moe import MegaMoE


class PrimusTurboMegaMoELayer(MegatronModule):

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
            config.tensor_model_parallel_size == 1
        ), "MegaMoE adapter requires tensor_model_parallel_size == 1 (EP-only)"
        assert (
            config.params_dtype == torch.bfloat16
        ), "MegaMoE adapter only supports bf16 (params_dtype must be torch.bfloat16)"
        assert (
            pg_collection is not None and pg_collection.ep is not None
        ), "MegaMoE adapter requires an expert-parallel process group"

        self._assert_supported_config(config)

        self.config = config
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer
        self.ep_group = pg_collection.ep
        # aux loss coeff; 0 disables. Used for tracker de-scaling
        self.aux_loss_coeff = self.get_aux_loss_coeff(config)
        # aux loss is CP-local; Megatron reduces over tp_cp_group, so guard CP>1
        if self.aux_loss_coeff != 0.0:
            assert getattr(config, "context_parallel_size", 1) == 1, (
                "MegaMoE internal aux loss is CP-local; context_parallel_size>1 "
                "needs a tp_cp_group reduce to match Megatron. Disable aux loss or CP."
            )

        self.mega_moe = MegaMoE(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_ffn_hidden_size,
            num_experts=config.num_moe_experts,
            ep_group=self.ep_group,
            top_k=config.moe_router_topk,
            num_groups=config.moe_router_num_groups,
            topk_group=config.moe_router_group_topk,
            score_function=config.moe_router_score_function,
            routed_scaling_factor=config.moe_router_topk_scaling_factor,
            force_load_balancing=bool(getattr(config, "moe_router_force_load_balancing", False)),
            aux_loss_coeff=self.aux_loss_coeff,
            add_bias_linear=bool(getattr(config, "add_bias_linear", False)),
            enable_expert_bias=False,
            shared_expert_intermediate_size=config.moe_shared_expert_intermediate_size,
            shared_expert_gate=bool(getattr(config, "moe_shared_expert_gate", False)),
            init_method=config.init_method,
            output_layer_init_method=config.output_layer_init_method,
            router_dtype=(torch.float64 if config.moe_router_dtype == "fp64" else torch.float32),
            get_rng_state_tracker=(get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None),
            rng_tracker_name=get_expert_parallel_rng_tracker_name(),
            device=torch.cuda.current_device() if torch.cuda.is_available() else None,
            dtype=torch.bfloat16,
        )

        expert_parallel = self.ep_group.size() > 1
        for p in (self.mega_moe.w1, self.mega_moe.w2):
            setattr(p, "allreduce", not expert_parallel)

    @staticmethod
    def get_aux_loss_coeff(config: TransformerConfig) -> float:
        """Load-balancing ("aux_loss") coeff, ported from Megatron TopKRouter.get_aux_loss_coeff."""
        routing_type = config.moe_router_load_balancing_type
        coeff = getattr(config, "moe_aux_loss_coeff", 0.0)
        if isinstance(routing_type, str):
            if routing_type == "aux_loss":
                return float(coeff)
        if isinstance(routing_type, (list, tuple)):
            try:
                idx = list(routing_type).index("aux_loss")
                return float(coeff[idx])
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    def _assert_supported_config(config: TransformerConfig) -> None:

        # Tolerate configs where these newer fields are absent.
        assert not getattr(config, "moe_seq_aux_loss_coeff", None), (
            "MegaMoE only supports the standard load_balancing aux loss; " "set moe_seq_aux_loss_coeff=0."
        )
        assert not getattr(config, "moe_global_aux_loss_coeff", None), (
            "MegaMoE only supports the standard load_balancing aux loss; " "set moe_global_aux_loss_coeff=0."
        )
        assert (
            not config.moe_z_loss_coeff
        ), "MegaMoE does not implement router z-loss; set moe_z_loss_coeff=None/0."
        load_balancing_type = config.moe_router_load_balancing_type
        load_balancing_types = (
            load_balancing_type if isinstance(load_balancing_type, (list, tuple)) else [load_balancing_type]
        )
        assert "sinkhorn" not in load_balancing_types, "MegaMoE does not support sinkhorn load balancing."
        assert (
            not config.moe_input_jitter_eps
        ), "MegaMoE does not implement moe_input_jitter_eps; set it to None."
        assert (
            config.moe_expert_capacity_factor is None
        ), "MegaMoE is dropless; moe_expert_capacity_factor is not supported (set to None)."
        if config.moe_router_score_function == "softmax":
            assert config.moe_router_pre_softmax, (
                "MegaMoE softmax routing matches Megatron only with "
                "moe_router_pre_softmax=True; enable it or use score_function='sigmoid'."
            )
        assert config.moe_router_dtype in (
            None,
            "fp32",
            "fp64",
        ), f"unsupported moe_router_dtype {config.moe_router_dtype!r}"

        assert config.gated_linear_unit, (
            "MegaMoE hardcodes a gated SwiGLU MLP; set gated_linear_unit=True "
            "or use the non-fused MoE layer."
        )
        assert config.activation_func in (F.silu, torch.nn.SiLU), (
            "MegaMoE hardcodes SiLU activation; set activation_func=F.silu " "or use the non-fused MoE layer."
        )
        assert not config.add_bias_linear, (
            "MegaMoE cannot apply expert FC biases (fused kernel has no bias); "
            "set add_bias_linear=False or use the non-fused MoE layer."
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()
        output, aux_loss = self.mega_moe(hidden_states, padding_mask=padding_mask, return_aux_loss=True)
        if aux_loss is not None:
            # tracker size incl. MTP layers to match Megatron indexing
            num_layers = self.config.num_layers
            if self.config.mtp_num_layers is not None:
                num_layers += self.config.mtp_num_layers
            # log the unscaled loss (de-scale by coeff) then inject grad via output
            save_to_aux_losses_tracker(
                "load_balancing_loss",
                aux_loss.detach() / self.aux_loss_coeff,
                self.layer_number,
                num_layers,
            )
            if self.config.calculate_per_token_loss:
                # match Megatron: rescale aux grad by local token count
                num_tokens = hidden_states.shape[:-1].numel()
                aux_loss = aux_loss * num_tokens
            output = MoEAuxLossAutoScaler.apply(output, aux_loss)
        return output, None

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
        metadata: Optional[dict] = None,
    ):
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        dp_cp_group = metadata["dp_cp_group"]
        prepend_axis_num = len(sharded_offsets)

        mega_moe = self.mega_moe
        sub_prefix = f"{prefix}mega_moe."

        edp_rank = parallel_state.get_expert_data_parallel_rank()
        expert_replica_id = (0, 0, edp_rank)

        module_sd: dict = {}
        mega_moe._save_to_state_dict(module_sd, "", keep_vars=True)

        sharded_sd: dict = {}
        for name in ("w1", "w2"):
            weight = module_sd.pop(name)
            key = f"{sub_prefix}{name}"
            sharded_sd[key] = ShardedTensor.from_rank_offsets(
                key,
                weight,
                *sharded_offsets,
                (prepend_axis_num, mega_moe.ep_rank, mega_moe.ep_size),
                replica_id=expert_replica_id,
                prepend_axis_num=prepend_axis_num,
            )
        sharded_sd.update(
            make_sharded_tensors_for_checkpoint(
                module_sd, sub_prefix, sharded_offsets=sharded_offsets, dp_cp_group=dp_cp_group
            )
        )
        return sharded_sd

    def set_layer_number(self, layer_number: int) -> None:
        self.layer_number = layer_number

    def backward_dw(self, *args: object, **kwargs: object) -> None:
        return None
