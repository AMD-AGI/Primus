###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from typing import Optional

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel import all_to_all
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher


class _AsyncAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes, work_handles_dict):
        """Forward function."""
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.work_handles_dict = work_handles_dict

        world_size = group.size()
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input, None

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device(),
            )
        work = torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
        # ctx.mark_non_differentiable(work)
        if os.environ.get('FWD_ASYNC_WORK_DEBUG_EAGER_SYNC') == '1' and world_size > 1:
            if os.environ.get('VERBOSE_DEBUG_PRINT') == '1':
                print("In Forward: FWD_ASYNC_WORK_DEBUG_EAGER_SYNC: Calling work.wait() in _AsyncAllToAll forward")
            work.wait()
        return output, work

    @staticmethod
    def backward(ctx, grad_output, grad_work):
        """Backward function."""
        world_size = ctx.group.size()
        grad_input, work = _AsyncAllToAll.apply(
            ctx.group, grad_output, ctx.input_split_sizes, ctx.output_split_sizes, ctx.work_handles_dict
        )
        if os.environ.get('BWD_ASYNC_WORK_DEBUG_EAGER_SYNC') == '1' and world_size > 1:
            if os.environ.get('VERBOSE_DEBUG_PRINT') == '1':
                print("In Backward: ASYNC_WORK_DEBUG_EAGER_SYNC: Calling work.wait() in _AsyncAllToAll backward")
            work.wait()
        if world_size > 1:
            ctx.work_handles_dict['backward'] = work
        return (
            None,
            grad_input,
            None,
            None,
            None,
        )


def async_all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None):
    """Wrapper for async autograd function"""
    assert group is not None, "group should not be None"
    work_handles_dict = {}
    output, work = _AsyncAllToAll.apply(group, input_, output_split_sizes_, input_split_sizes, work_handles_dict)
    return output, work, work_handles_dict



class AsyncMoEAlltoAllTokenDispatcher(MoEAlltoAllTokenDispatcher):
    """MoE All-to-All token dispatcher with async communication support.

    Extends the base MoEAlltoAllTokenDispatcher to support asynchronous
    all-to-all communication for overlapping computation with communication.
    """

    def token_dispatch(self, permutated_local_input_tokens, permuted_probs, async_op=False):
        """
        Perform all-to-all communication for dispatching tokens.

        Args:
            permutated_local_input_tokens (torch.Tensor): Permuted input tokens.
            permuted_probs (torch.Tensor): Permuted token probabilities.
            async_op (bool): Whether to use async all-to-all communication.

        Returns:
            If async_op=False:
                Tuple[torch.Tensor, torch.Tensor]: Dispatched tokens and probabilities.
            If async_op=True:
                Tuple[torch.Tensor, torch.Tensor, Tuple, Tuple]: Dispatched tokens,
                probabilities, forward handles, and backward handles.
        """
        self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_ep_alltoall", self.tokens_per_expert
        )
        if not async_op:
            global_input_tokens = all_to_all(
                self.ep_group, permutated_local_input_tokens, self.output_splits, self.input_splits
            )
            global_probs = all_to_all(
                self.ep_group, permuted_probs, self.output_splits, self.input_splits
            )
            return global_input_tokens, global_probs
        else:
            global_probs, dist_handle2, handle2_backward_handles = async_all_to_all(
                self.ep_group, permuted_probs, self.output_splits, self.input_splits
            )
            global_input_tokens, dist_handle1, handle1_backward_handles = async_all_to_all(
                self.ep_group, permutated_local_input_tokens, self.output_splits, self.input_splits
            )
            return global_input_tokens, global_probs, (dist_handle1, dist_handle2), (handle1_backward_handles, handle2_backward_handles)

    def token_combine(
        self,
        hidden_states,
        async_finish=True,
        allocate_on_comm_stream=False,
        async_op=False,
    ):
        """Executes fused un-permutation and communication using DeepEP kernels.

        Args:
            hidden_states (torch.Tensor): Expert outputs to combine [SEQL, H].
            async_finish (bool): Whether to finish async operations.
            allocate_on_comm_stream (bool): Whether to allocate on communication stream.
            async_op (bool): Whether to use async all-to-all communication.

        Returns:
            If async_op=False:
                torch.Tensor: Combined local tokens [SEQL, H/TP].
            If async_op=True:
                Tuple[torch.Tensor, handle, backward_handle]: Combined tokens and communication handles.
        """
        if async_op:
            permutated_local_input_tokens, combine_handle, combine_backward_handle = async_all_to_all(
                self.ep_group, hidden_states, self.input_splits, self.output_splits
            )
            return permutated_local_input_tokens, combine_handle, combine_backward_handle
        else:
            permutated_local_input_tokens = all_to_all(
                self.ep_group, hidden_states, self.input_splits, self.output_splits
            )
            return permutated_local_input_tokens


# MoELayer backward compatible method patching (dispatch, combine, forward)
# adding async_op flag arg and returning shared_expert_output in forward
def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor, async_op=False):
    return self.token_dispatcher.token_dispatch(hidden_states, probs, async_op=async_op)


def combine(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor], async_op=False):
    output = self.token_dispatcher.token_combine(output, async_op=async_op)
    output = self.token_dispatcher.combine_postprocess(output)

    if shared_expert_output is not None:
        output = output + shared_expert_output
    return output

def forward(self, hidden_states: torch.Tensor, return_shared_expert=False):
    if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
        raise ValueError(
            "During training, performance may degrade if MoE and tensor parallelism"
            "are enabled without also enabling sequence parallelism."
        )

    # MoE forward: route -> dispatch -> compute -> combine
    def custom_forward(hidden_states):
        hidden_states, probs, residual = self.router_and_preprocess(hidden_states)
        dispatched_input, probs = self.dispatch(hidden_states, probs)
        output, shared_expert_output, mlp_bias = self.experts_compute(
            dispatched_input, probs, residual
        )
        output = self.combine(output, shared_expert_output)
        return output, shared_expert_output, mlp_bias

    if self.moe_layer_recompute:
        if self.config.fp8:
            output, shared_expert_output, mlp_bias = te_checkpoint(
                custom_forward,
                False,
                tensor_parallel.random.get_cuda_rng_tracker,
                parallel_state.get_tensor_model_parallel_group(),
                hidden_states,
            )
        else:
            output,shared_expert_output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
    else:
        output, shared_expert_output, mlp_bias = custom_forward(hidden_states)
    if return_shared_expert:
        return output, shared_expert_output, mlp_bias
    else:
        return output, mlp_bias
