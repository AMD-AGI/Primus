from typing import Optional

import torch
from primus_turbo.pytorch.deep_ep import Buffer, Config

_buffer = None

_dispatch_config: Config = None
_combine_config: Config = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(
    group: torch.distributed.ProcessGroup, hidden_bytes: int, use_default_stream_as_comm_stream: bool = True
):
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
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        
        try:
            num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)
        except:
            pass

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(
            group,
            num_nvl_bytes,
            num_rdma_bytes,
            use_default_stream_as_comm_stream=use_default_stream_as_comm_stream,
        )
    return _buffer


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
        previous_event=None,
        use_cuda_num_token_per_expert: bool = True,
        num_worst_tokens: int = 0,
        config: Optional[Config] = None,
    ):
        global _dispatch_config
        _dispatch_config = config
        """Forward pass of fused dispatch."""
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            tokens_per_expert,
            handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
            num_recv_tokens_per_expert_as_cuda=use_cuda_num_token_per_expert,
            num_worst_tokens=num_worst_tokens,
            config=_dispatch_config,
        )

        ctx.group = group
        ctx.handle = handle
        ctx.event = event

        if not use_cuda_num_token_per_expert:
            tokens_per_expert = torch.tensor(tokens_per_expert)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle):
        """Backward pass of fused dispatch."""
        global _combine_config

        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle

        grad_x, grad_token_probs, event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
            config=_combine_config,
        )
        return grad_x, None, grad_token_probs, None, None, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, previous_event=None, config: Optional[Config] = None):
        """Forward pass of fused combine."""
        global _combine_config
        _combine_config = config

        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, event = buffer.combine(
            x,
            handle=handle,
            async_finish=False,
            previous_event=None,
            allocate_on_comm_stream=False,
            config=_combine_config,
        )
        ctx.handle = handle
        ctx.group = group

        return combined_x, event

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        global _dispatch_config

        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
            config=_dispatch_config,
        )
        return grad_x, None, None, None, None


def fused_dispatch(
    x,
    token_indices,
    token_probs,
    num_experts,
    group,
    previous_event=None,
    use_cuda_num_token_per_expert: bool = True,
    num_worst_tokens: int = 0,
    config: Optional[Config] = None,
):
    """Perform fused dispatch operation if deep_ep is available.

    Args:
        x: Input tensor [num_tokens, hidden_size]
        token_indices: Token routing indices [num_tokens, topk]
        token_probs: Token routing probabilities [num_tokens, topk]
        num_experts: Number of experts
        group: Process group
        previous_event: Previous CUDA event
        num_worst_tokens: the worst number of tokens to receive, if specified, there will be no CPU sync
        config: DeepEP dispatch config for tuning

    Returns:
        Result of FusedDispatch
    """
    return FusedDispatch.apply(
        x.contiguous(),
        token_indices,
        token_probs,
        num_experts,
        group,
        previous_event,
        use_cuda_num_token_per_expert,
        num_worst_tokens,
        config,
    )


def fused_combine(
    x,
    group,
    handle,
    previous_event=None,
    config: Optional[Config] = None,
):
    """Perform fused combine operation if deep_ep is available.

    Args:
        x: Input tensor
        group: Process group
        handle: Communication handle
        previous_event: Previous CUDA event
        config: DeepEP combine config for tuning

    Returns:
        Result of FusedCombine
    """

    return FusedCombine.apply(x, group, handle, previous_event, config)


def set_deepep_num_sms(num_sms):
    """Sets the number of SMs to use for DeepEP"""
    Buffer.set_num_sms(num_sms)
