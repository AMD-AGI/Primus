from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import get_args


class PrimusGroupedMLP(TEGroupedMLP):
    """An efficient implementation of the Experts layer using TE's GroupedLinear.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        submodules: TEGroupedMLPSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        args = get_args()

        super().__init__(
            num_local_experts,
            config,
            submodules,
            pg_collection,
        )

        # NOTE: use_turbo_fused_act_with_probs is prioritized over use_te_activation_func and bias_activation_fusion
        self.use_turbo_fused_act_with_probs = args.use_turbo_fused_act_with_probs

    def bias_act_func_with_mask(
        self,
        intermediate_parallel: torch.Tensor,
        bias_parallel: torch.Tensor,
        permuted_probs: torch.Tensor,
        tokens_per_experts: Union[torch.Tensor, None] = None,
    ):
        if self.use_turbo_fused_act_with_probs:
            from primus.backends.megatron.core.extensions.primus_turbo import (
                fused_bias_act_with_probs,
            )

            if self.activation_func == F.silu and self.config.gated_linear_unit:
                # dtype is handled inside the fused kernel
                bias_act_with_probs = fused_bias_act_with_probs(
                    intermediate_parallel, bias_parallel, permuted_probs, tokens_per_experts, "silu"
                )
            elif self.activation_func == F.gelu and self.config.gated_linear_unit:
                bias_act_with_probs = fused_bias_act_with_probs(
                    intermediate_parallel, bias_parallel, permuted_probs, tokens_per_experts, "gelu"
                )
            else:
                raise ValueError(
                    "Only support fusion of swiglu and gelu in PrimusGroupedMLP when `use_turbo_fused_act_with_probs` is True."
                )

            return bias_act_with_probs(
                intermediate_parallel, bias_parallel, permuted_probs, tokens_per_experts
            )
        else:
            # use the original bias_act_func from TEGroupedMLP, ignore the tokens_per_experts
            return self.bias_act_func(intermediate_parallel, bias_parallel, permuted_probs)

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of PrimusGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.
            permuted_probs (torch.Tensor): The permuted probs of each token produced by the router.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        permuted_probs = permuted_probs.unsqueeze(-1)

        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = permuted_probs * permuted_local_hidden_states
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        with off_interface(
            self.offload_expert_fc1, permuted_local_hidden_states, "expert_fc1"
        ) as permuted_local_hidden_states:
            fc1_output, bias_parallel = apply_module(self.linear_fc1)(
                permuted_local_hidden_states, tokens_per_expert
            )
        if self.offload_expert_fc1:
            fc1_output = off_interface.group_commit(
                fc1_output,
                name="expert_fc1",
                forced_released_tensors=[permuted_local_hidden_states],
            )

        if self.activation_recompute:
            self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with off_interface(self.offload_moe_act, fc1_output, "moe_act") as fc1_output:
                # NOTE: use the bias_act_func_with_mask instead of the bias_act_func to reduce the extra compute when stage of `sync_free_moe` is 3.
                bias_act_output = self.activation_checkpoint.checkpoint(
                    self.bias_act_func_with_mask, fc1_output, bias_parallel, permuted_probs, tokens_per_expert
                )
        else:
            with off_interface(self.offload_moe_act, fc1_output, "moe_act") as fc1_output:
                bias_act_output = self.bias_act_func(fc1_output, bias_parallel, permuted_probs)
        output, output_bias = apply_module(self.linear_fc2)(bias_act_output, tokens_per_expert)
        if self.activation_recompute:
            self.activation_checkpoint.discard_output_and_register_recompute(output)

        # Delay the offload of the moe act until after the linear_fc2 has been computed
        # to make sure the fc1_output is reloaded to GPU before recomputing moe_act.
        if self.offload_moe_act:
            output = off_interface.group_commit(output, name="moe_act", forced_released_tensors=[fc1_output])
        # NOTE: tokens_per_expert is on GPU, so we need to convert it to a list of ints.
        output = self._apply_bias(output, output_bias, tokens_per_expert.tolist(), permuted_probs)

        output_bias = None

        return output, output_bias
