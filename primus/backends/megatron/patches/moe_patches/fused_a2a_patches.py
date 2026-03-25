###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron MoE Fused All-to-All Patches

Patches for replacing Megatron's fused all-to-all dispatch/combine implementations
with Primus versions backed by DeepEP (deep_ep.Buffer) and primus_turbo ops.

The patched hybrid_ep_dispatch and hybrid_ep_combine use DeepEP's HybridEP backend
for inter-rank all-to-all communication, combined with primus_turbo's fused token
permute/unpermute operations (indices_to_multihot, token_permute, token_unpermute).
"""


import torch
from functools import partial

from primus.core.patches import PatchContext, get_args, register_patch

import primus_turbo.pytorch as turbo
from primus.modules.module_utils import log_rank_0
from primus_turbo.pytorch.deep_ep import Buffer


_buffer = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(
                hidden_bytes, group.size()), num_nvl_bytes
        )

        try:
            num_rdma_bytes = max(
                config.get_rdma_buffer_size_hint(
                    hidden_bytes, group.size()), num_rdma_bytes
            )
        except:
            num_rdma_bytes = 0

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


# def dispatch_with_permute(buffer,
#                           x,
#                           group,
#                           num_local_experts=None,
#                           token_indices=None,
#                           token_probs=None,
#                           num_permuted_tokens=None,
#                           handle=None,
#                           ):

#     # If we provide the num_permuted_tokens, we do not need to use sync to
#     # wait for the data in pinned memory ready
#     non_blocking = num_permuted_tokens is not None
#     forward_pass = handle is None
#     if forward_pass:
#         assert num_local_experts is not None
#         assert token_indices is not None
#         assert token_probs is not None

#     else:
#         num_local_experts, _, _, _, _, handle = handle

#     num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank = None, None, None, None
#     # Process the dispatch in forward
#     if forward_pass:
#         num_experts = num_local_experts * group.size()
#         (
#             num_tokens_per_rank,
#             num_tokens_per_rdma_rank,
#             num_tokens_per_expert,
#             is_token_in_rank,
#             _,
#         ) = buffer.get_dispatch_layout(
#             token_indices,
#             num_experts,
#         )

#     # Do MoE dispatch
#     # NOTES: the CPU will wait for GPU's signal to arrive,
#     # so this is not compatible with CUDA graph
#     (
#         recv_x,
#         recv_token_indices,
#         recv_token_probs,
#         _,
#         handle,
#         _,
#     ) = buffer.dispatch(
#         x,
#         topk_idx=token_indices,
#         topk_weights=token_probs,  # DeepEP only supports float32 probs
#         num_tokens_per_rank=num_tokens_per_rank,
#         num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
#         is_token_in_rank=is_token_in_rank,
#         num_tokens_per_expert=num_tokens_per_expert,
#         num_worst_tokens=num_permuted_tokens if non_blocking else 0,
#         handle=handle,
#     )

#     # permute the tokens
#     if forward_pass:
#         dispatched_routing_map, dispatched_probs = turbo.ops.IndicesToMultihot.forward(
#             recv_token_indices, recv_token_probs, num_local_experts, fused=True)
#     else:


#     hidden_shape_before_permute = recv_x.shape

#     permuted_x, permuted_probs, reversed_mapping_for_combine, tokens_per_expert = (
#         turbo.ops.token_permute(
#             recv_x,
#             num_out_tokens=num_permuted_tokens,
#             routing_map=dispatched_routing_map,
#             probs=dispatched_probs,
#             fused=True,
#             return_tokens_per_expert=True,
#         )
#     )

#     handle = (num_local_experts, reversed_mapping_for_combine,
#               dispatched_routing_map, hidden_shape_before_permute, group, handle)

#     return permuted_x, permuted_probs, tokens_per_expert, handle


# def combine_with_unpermute(buffer,
#                            x,
#                            handle,
#                            probs=None):
#     _, reversed_mapping_for_combine, dispatched_routing_map, hidden_shape_before_permute, _, handle = handle
#     unpermuted_x = turbo.ops.token_unpermute(
#         x,
#         reversed_mapping_for_combine,
#         restore_shape=hidden_shape_before_permute,
#         routing_map=dispatched_routing_map,
#         fused=True,
#     )
#     combined_x, combine_probs, _ = buffer.combine(
#         unpermuted_x,
#         handle=handle,
#         topk_weights=probs.float() if probs is not None else None,
#     )
#     return combined_x, combine_probs


# class HybridEPDispatch(torch.autograd.Function):
#     '''
#     Fused dispatch operation for permute + dispatch a2a + permute using the HybridEP backend
#     '''

#     @staticmethod
#     def forward(
#         ctx,
#         x,
#         token_indices,
#         probs,
#         group,
#         num_local_experts,
#         num_permuted_tokens=None,
#     ):
#         '''
#         Forward pass of fused dispatch
#         '''
#         buffer = get_buffer(group, get_hidden_bytes(x))
#         permuted_x, permuted_probs, tokens_per_expert, handle = dispatch_with_permute(
#             buffer, x, group,
#             num_local_experts=num_local_experts,
#             token_indices=token_indices,
#             token_probs=probs,
#             num_permuted_tokens=num_permuted_tokens)

#         ctx.handle = handle
#         ctx.group = group
#         return (
#             permuted_x,
#             permuted_probs,
#             None,
#             tokens_per_expert,
#             handle,
#         )

#     @staticmethod
#     def backward(ctx, grad_x, grad_probs, grad_scaling_factor, grad_tokens_per_expert, grad_handle):
#         '''
#         Backward pass of fused dispatch of the HybridEP backend
#         '''
#         handle = ctx.handle
#         buffer = get_buffer(ctx.group, get_hidden_bytes(grad_x))
#         combined_hidden, combined_probs = combine_with_unpermute(
#             buffer, grad_x, handle, probs=grad_probs
#         )
#         return combined_hidden, None, combined_probs, None, None, None, None, None, None, None


# class HybridEPCombine(torch.autograd.Function):
#     '''
#     Fused combine operation for permute + combine a2a + permute using the HybridEP backend
#     '''

#     @staticmethod
#     def forward(ctx, x, handle, num_permuted_tokens=None):
#         '''
#         Forward pass of fused combine of the HybridEP backend
#         '''
#         assert len(handle) == 6
#         group = handle[-2]
#         buffer = get_buffer(group, get_hidden_bytes(x))
#         combined_hidden, _ = combine_with_unpermute(
#             buffer, x, handle,
#         )

#         ctx.handle = handle
#         ctx.group = group
#         ctx.num_permuted_tokens = num_permuted_tokens
#         return combined_hidden

#     @staticmethod
#     def backward(ctx, grad_x):
#         '''
#         Backward pass of fused combine of the HybridEP backend
#         '''
#         handle = ctx.handle
#         buffer = get_buffer(ctx.group, get_hidden_bytes(grad_x))
#         dispatched_hidden, _, _, _, _ = dispatch_with_permute(
#             buffer, grad_x, ctx.group,
#             num_permuted_tokens=ctx.num_permuted_tokens,
#             handle=handle,
#         )
#         return dispatched_hidden, None, None, None, None


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        num_worst_tokens=0,
    ):
        """Forward pass of fused dispatch."""
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            _,
            handle,
            _,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            num_worst_tokens=num_worst_tokens)

        # Make sure current stream is synchronized

        # Save for backward
        ctx.group = group
        ctx.handle = handle

        return (recv_x, recv_token_indices, recv_token_probs, handle)

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_handle
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        grad_x, grad_token_probs, _ = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
        )

        return grad_x, None, grad_token_probs, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle):
        """Forward pass of fused combine."""
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, _ = buffer.combine(
            x,
            handle=handle,
        )
        ctx.handle = handle
        ctx.group = group
        return combined_x

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of fused combine."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, _ = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
        )
        return grad_x, None, None


def bridge_hybrid_ep_dispatch(
    x,
    routing_map,
    probs,
    group,
    num_local_experts,
    num_sms_dispatch_api=24,
    num_sms_combine_api=24,
    num_permuted_tokens=None,
    pad_multiple=None,
    num_experts=None,
    router_topk=None,
):
    '''
    Perform fused dispatch for "permute + dispatch a2a + permute" using the
    HybridEP backend.

    Args:
        x (torch.Tensor):
            Input hidden states to dispatch.
        routing_map (torch.Tensor):
                Map indicating which expert each token is routed to.
        probs (torch.Tensor):
            Routing probabilities for each token-expert pair.
        group (torch.distributed.ProcessGroup):
            Process group used for communication.
        num_local_experts (int):
            Number of local experts.
        num_sms_dispatch_api (int):
            Number of SMs used by the dispatch API.
        num_sms_combine_api (int):
            Number of SMs used by the combine API.
        num_permuted_tokens (int):
            Number of tokens after permute. HybridEP uses this to allocate buffers.
            If not provided, HybridEP obtains the size from a GPU tensor,
            which causes a D2H synchronization.
        pad_multiple (int):
            Alignment multiple required for FP8 GEMM. If not provided, no padding
            is performed.
    '''
    num_tokens = x.size(0)
    num_worst_tokens = num_tokens * group.size()
    probs = probs.reshape(num_tokens, num_experts)

    probs, token_indices = torch.topk(probs, router_topk, dim=-1)

    recv_x, recv_token_indices, recv_token_probs, handle = FusedDispatch.apply(
        x.contiguous(), token_indices, probs, num_experts, group, num_worst_tokens)

    dispatched_routing_map, dispatched_probs = turbo.ops.indices_to_multihot(
        recv_token_indices, recv_token_probs, num_local_experts, fused=True)

    hidden_shape_before_permute = recv_x.shape

    permuted_x, permuted_probs, reversed_mapping_for_combine, tokens_per_expert = turbo.ops.token_permute(
        recv_x,
        num_out_tokens=num_permuted_tokens,
        routing_map=dispatched_routing_map,
        probs=dispatched_probs,
        fused=True,
        return_tokens_per_expert=True,
    )

    # TODO(zhenhuang12): add overflow_flag to permutation phase
    overflow_flag = torch.zeros(1, dtype=torch.int32, device='cuda')

    handle = (num_local_experts, reversed_mapping_for_combine,
              dispatched_routing_map, hidden_shape_before_permute, group, handle, None, None, overflow_flag)

    return permuted_x, permuted_probs, None, tokens_per_expert, handle


def bridge_hybrid_ep_combine(x, handle, num_permuted_tokens, pad_multiple):
    '''
    Perform fused combine operation for unpermute + combine a2a + unpermute
    using the HybridEP backend

    args:
        x (torch.Tensor):
            Input hidden states to combine
        handle (EventHandle):
            Communication handle from dispatch operation
        num_permuted_tokens (int): The number of tokens before unpermute. HybridEP uses this
            to allocate buffers. If not provided, HybridEP obtains the size from a GPU tensor,
            which causes a D2H synchronization.
        pad_multiple (int):
            The alignment multiple required for FP8 GEMM. If not provided, no padding
            is performed.
    '''
    (_, reversed_mapping_for_combine, dispatched_routing_map,
     hidden_shape_before_permute, group, handle, _, _, _) = handle
    unpermuted_x = turbo.ops.token_unpermute(
        x,
        reversed_mapping_for_combine,
        restore_shape=hidden_shape_before_permute,
        routing_map=dispatched_routing_map,
        fused=True,
    )
    combined_x = FusedCombine.apply(
        unpermuted_x,
        group,
        handle,
    )
    return combined_x


@register_patch(
    "megatron.core.transformer.moe.fused_a2a",
    backend="megatron",
    # Execute early to patch before model building uses the default fused_a2a functions
    phase="build_args",
    description="Replace fused_a2a hybrid_ep_dispatch/combine with DeepEP-backed implementations",
    condition=lambda ctx: not getattr(
        get_args(ctx), "disable_primus_hybridep", False),
)
def patch_fused_a2a(ctx: PatchContext):
    """
    Patch Megatron's fused_a2a module to use Primus implementations.

    Replaces hybrid_ep_dispatch and hybrid_ep_combine in
    megatron.core.transformer.moe.fused_a2a with versions that leverage
    DeepEP buffers for all-to-all communication and primus_turbo ops
    for fused token permutation.
    """
    from megatron.core.transformer.moe import fused_a2a
    from megatron.core.transformer.moe import token_dispatcher

    # step1: replace the hybrid_ep_dispatch and hybrid_ep_combine with the primus implementations
    router_topk = get_args(ctx).moe_router_topk
    num_experts = get_args(ctx).num_experts

    new_bridge_hybrid_ep_dispatch = partial(
        bridge_hybrid_ep_dispatch, num_experts=num_experts, router_topk=router_topk)
    fused_a2a.hybrid_ep_dispatch = new_bridge_hybrid_ep_dispatch
    fused_a2a.hybrid_ep_combine = bridge_hybrid_ep_combine
    log_rank_0(
        f"[Patch:megatron.core.transformer.fused_a2a]   Patched fused_a2a.hybrid_ep_dispatch and fused_a2a.hybrid_ep_combine"
    )

    # step2: replace the hybrid_ep_dispatch and hybrid_ep_combine in the token_dispatcher
    token_dispatcher.hybrid_ep_dispatch = new_bridge_hybrid_ep_dispatch
    token_dispatcher.hybrid_ep_combine = bridge_hybrid_ep_combine
    log_rank_0(
        f"[Patch:megatron.core.transformer.token_dispatcher]   Patched token_dispatcher.hybrid_ep_dispatch and token_dispatcher.hybrid_ep_combine"
    )
