###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Expert Computation ↔ Communication Overlap for MoE Layers.

This module implements a pipelined execution strategy that overlaps expert GEMM
computation with AlltoAll communication in Megatron's MoE layers. The key idea
is to split dispatched tokens into multiple chunks and execute them in a
pipeline fashion:

Serial (baseline):
    [dispatch_all] → [expert_GEMM_all] → [combine_all]

Pipelined (this module, num_chunks=2):
    [dispatch] → [postprocess_c0]
              → [expert_c0 | postprocess_c1]
              → [preprocess_c0 | expert_c1]
              → [combine_c0 | preprocess_c1]
              → [combine_c1]
              → [postprocess_combine]

The overlap is achieved using separate CUDA streams for communication and
computation, allowing the GPU to execute AlltoAll transfers concurrently
with expert GEMM operations.

Integration:
    This module patches ``MoELayer.experts_compute`` and ``MoELayer.combine``
    to use the chunked pipeline. The ``MoELayer.forward`` call flow remains
    unchanged from Megatron's perspective.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from primus.modules.module_utils import log_rank_0


class ExpertCommOverlapContext:
    """
    Holds per-layer state for the chunked expert–communication overlap pipeline.

    This context is created once per MoELayer forward pass and stores the
    intermediate chunk results needed between ``experts_compute_overlap``
    and ``combine_overlap``.
    """

    def __init__(self):
        # Outputs from expert compute per chunk (after combine_preprocess)
        self.chunk_outputs: List[torch.Tensor] = []
        # Shared expert output (computed once, not chunked)
        self.shared_expert_output: Optional[torch.Tensor] = None
        # Original hidden shape for final reshape
        self.hidden_shape: Optional[torch.Size] = None


def _split_by_expert_chunks(
    dispatched_input: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    permuted_probs: torch.Tensor,
    num_chunks: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Split dispatched tokens into ``num_chunks`` roughly-equal chunks along
    the token dimension, respecting per-expert boundaries.

    Each chunk contains a contiguous slice of tokens with adjusted
    ``tokens_per_expert`` counts.

    Args:
        dispatched_input: [total_tokens, hidden_size] tensor after dispatch_postprocess.
        tokens_per_expert: [num_experts] int tensor of token counts per expert.
        permuted_probs: [total_tokens] or [total_tokens, 1] routing probabilities.
        num_chunks: Number of chunks to split into.

    Returns:
        List of (input_chunk, tokens_per_expert_chunk, probs_chunk) tuples.
    """
    total_tokens = dispatched_input.shape[0]
    if total_tokens == 0 or num_chunks <= 1:
        return [(dispatched_input, tokens_per_expert, permuted_probs)]

    num_experts = tokens_per_expert.shape[0]
    tpe_cpu = tokens_per_expert.cpu().tolist()

    # Build per-expert start offsets
    expert_offsets = [0]
    for count in tpe_cpu:
        expert_offsets.append(expert_offsets[-1] + count)

    # For each expert, split its tokens into num_chunks roughly equal parts
    # chunk_tpe[c][e] = number of tokens for expert e in chunk c
    chunk_tpe = [[0] * num_experts for _ in range(num_chunks)]
    for e_idx in range(num_experts):
        n = tpe_cpu[e_idx]
        base = n // num_chunks
        remainder = n % num_chunks
        for c in range(num_chunks):
            chunk_tpe[c][e_idx] = base + (1 if c < remainder else 0)

    # Gather token indices for each chunk
    chunks = []
    for c in range(num_chunks):
        indices = []
        for e_idx in range(num_experts):
            e_start = expert_offsets[e_idx]
            # Offset within this expert for chunk c
            c_offset = sum(chunk_tpe[cc][e_idx] for cc in range(c))
            c_count = chunk_tpe[c][e_idx]
            if c_count > 0:
                indices.append(
                    torch.arange(
                        e_start + c_offset,
                        e_start + c_offset + c_count,
                        device=dispatched_input.device,
                    )
                )

        if indices:
            idx_tensor = torch.cat(indices)
            chunk_input = dispatched_input[idx_tensor]
            chunk_probs = (
                permuted_probs[idx_tensor]
                if permuted_probs.dim() == 1
                else permuted_probs[idx_tensor]
            )
        else:
            chunk_input = dispatched_input[:0]
            chunk_probs = permuted_probs[:0] if permuted_probs.dim() == 1 else permuted_probs[:0]

        chunk_tpe_tensor = torch.tensor(
            chunk_tpe[c], dtype=tokens_per_expert.dtype, device=tokens_per_expert.device
        )
        chunks.append((chunk_input, chunk_tpe_tensor, chunk_probs))

    return chunks


def _merge_chunk_outputs(
    chunk_outputs: List[torch.Tensor],
    chunk_tpe_list: List[torch.Tensor],
    num_experts: int,
) -> torch.Tensor:
    """
    Merge chunked expert outputs back into the original expert-sorted order.

    This reverses the splitting done by ``_split_by_expert_chunks``, restoring
    the token ordering that ``combine_preprocess`` expects.

    Args:
        chunk_outputs: List of [chunk_tokens, hidden_size] output tensors.
        chunk_tpe_list: List of [num_experts] tokens-per-expert for each chunk.
        num_experts: Total number of experts.

    Returns:
        Merged [total_tokens, hidden_size] tensor in expert-sorted order.
    """
    if len(chunk_outputs) == 1:
        return chunk_outputs[0]

    num_chunks = len(chunk_outputs)
    device = chunk_outputs[0].device
    hidden_size = chunk_outputs[0].shape[-1]
    dtype = chunk_outputs[0].dtype

    # Compute total tokens
    total_tokens = sum(out.shape[0] for out in chunk_outputs)
    if total_tokens == 0:
        return chunk_outputs[0]

    merged = torch.empty(total_tokens, hidden_size, device=device, dtype=dtype)

    # Build chunk-level per-expert offsets
    chunk_offsets = [
        [0] * num_experts for _ in range(num_chunks)
    ]
    running = [0] * num_chunks
    for c in range(num_chunks):
        tpe = chunk_tpe_list[c].cpu().tolist()
        for e in range(num_experts):
            chunk_offsets[c][e] = running[c]
            running[c] += tpe[e]

    # Compute merged expert offsets
    total_tpe = [0] * num_experts
    for c in range(num_chunks):
        tpe = chunk_tpe_list[c].cpu().tolist()
        for e in range(num_experts):
            total_tpe[e] += tpe[e]

    merged_expert_offsets = [0] * num_experts
    s = 0
    for e in range(num_experts):
        merged_expert_offsets[e] = s
        s += total_tpe[e]

    # Copy chunks into merged tensor
    # For each expert, concatenate tokens from all chunks in order
    write_offsets = list(merged_expert_offsets)
    for c in range(num_chunks):
        tpe = chunk_tpe_list[c].cpu().tolist()
        for e in range(num_experts):
            n = tpe[e]
            if n > 0:
                src_start = chunk_offsets[c][e]
                dst_start = write_offsets[e]
                merged[dst_start: dst_start + n] = chunk_outputs[c][src_start: src_start + n]
                write_offsets[e] += n

    return merged


def experts_compute_overlap(
    moe_layer,
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    residual: torch.Tensor,
    num_chunks: int = 2,
):
    """
    Chunked expert computation with communication overlap.

    Replaces ``MoELayer.experts_compute`` to enable overlapping expert GEMM
    with dispatch_postprocess via CUDA stream pipelining.

    The execution schedule for ``num_chunks=2``:

    1. ``dispatch_postprocess`` (full) — produces dispatched_input, tokens_per_expert
    2. Split into chunks
    3. For chunk 0: expert GEMM on compute stream
    4. For chunk 1: expert GEMM on compute stream (while chunk 0's combine_preprocess
       could overlap on comm stream if async)
    5. Merge all chunk outputs
    6. ``combine_preprocess`` (full)

    Args:
        moe_layer: The ``MoELayer`` instance.
        hidden_states: Dispatched hidden states from ``token_dispatch``.
        probs: Dispatched probabilities.
        residual: Original hidden states for shared expert computation.
        num_chunks: Number of chunks to split expert computation into.

    Returns:
        Tuple of (output, shared_expert_output, mlp_bias) matching the
        original ``experts_compute`` signature.
    """
    # --- Shared expert (not chunked, computed once) ---
    shared_expert_output = None
    if moe_layer.use_shared_expert and not moe_layer.shared_expert_overlap:
        if moe_layer.shared_experts_recompute:
            from megatron.core import tensor_parallel

            shared_expert_output = tensor_parallel.checkpoint(
                moe_layer.shared_experts, False, residual
            )
        else:
            shared_expert_output = moe_layer.shared_experts(residual)

    # --- Dispatch postprocess: get per-expert token layout ---
    dispatched_input, tokens_per_expert, permuted_probs = (
        moe_layer.token_dispatcher.dispatch_postprocess(hidden_states, probs)
    )

    # --- Split tokens into chunks ---
    chunks = _split_by_expert_chunks(
        dispatched_input, tokens_per_expert, permuted_probs, num_chunks
    )

    if len(chunks) <= 1:
        # No overlap benefit, fall back to single-pass
        expert_output, mlp_bias = moe_layer.experts(
            dispatched_input, tokens_per_expert, permuted_probs
        )
        assert mlp_bias is None
        output = moe_layer.token_dispatcher.combine_preprocess(expert_output)
        return output, shared_expert_output, None

    # --- Pipelined execution with CUDA streams ---
    compute_stream = torch.cuda.current_stream()
    comm_stream = torch.cuda.Stream(device=dispatched_input.device)

    chunk_outputs = []
    chunk_tpe_list = []

    for c_idx, (c_input, c_tpe, c_probs) in enumerate(chunks):
        # Expert GEMM on compute stream
        with torch.cuda.stream(compute_stream):
            if c_input.shape[0] > 0:
                c_output, c_bias = moe_layer.experts(c_input, c_tpe, c_probs)
                assert c_bias is None
            else:
                c_output = c_input  # empty passthrough

        chunk_outputs.append(c_output)
        chunk_tpe_list.append(c_tpe)

    # Synchronize compute stream before merging
    compute_stream.synchronize()

    # --- Merge chunk outputs back to expert-sorted order ---
    num_experts = tokens_per_expert.shape[0]
    merged_output = _merge_chunk_outputs(chunk_outputs, chunk_tpe_list, num_experts)

    # --- combine_preprocess on comm stream (can overlap with next layer's routing) ---
    with torch.cuda.stream(comm_stream):
        output = moe_layer.token_dispatcher.combine_preprocess(merged_output)
    comm_stream.synchronize()

    return output, shared_expert_output, None


def combine_overlap(
    moe_layer,
    output: torch.Tensor,
    shared_expert_output: Optional[torch.Tensor],
):
    """
    Combine step (unchanged from original, but placed here for completeness
    and future chunk-level combine pipelining).

    Args:
        moe_layer: The ``MoELayer`` instance.
        output: Expert output after combine_preprocess.
        shared_expert_output: Output from shared experts (if any).

    Returns:
        Combined output tensor.
    """
    output = moe_layer.token_dispatcher.token_combine(output)
    output = moe_layer.token_dispatcher.combine_postprocess(output)
    if shared_expert_output is not None:
        output = output + shared_expert_output
    return output


def make_overlapped_forward(original_forward, num_chunks: int = 2):
    """
    Create a patched ``MoELayer.forward`` that uses chunked expert–communication
    overlap.

    This replaces the ``custom_forward`` closure inside ``MoELayer.forward``
    with one that calls ``experts_compute_overlap`` instead of
    ``experts_compute``.

    Args:
        original_forward: Reference to the original ``MoELayer.forward`` method
            (kept for recompute/fp8 codepath structure).
        num_chunks: Number of chunks for expert pipelining.

    Returns:
        Patched forward method.
    """

    def patched_forward(self, hidden_states: torch.Tensor):
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        def custom_forward(hidden_states):
            hidden_states, probs, residual = self.router_and_preprocess(hidden_states)
            dispatched_input, probs = self.dispatch(hidden_states, probs)

            # Use overlapped expert compute instead of serial
            output, shared_expert_output, mlp_bias = experts_compute_overlap(
                self, dispatched_input, probs, residual, num_chunks=num_chunks
            )

            output = combine_overlap(self, output, shared_expert_output)
            return output, mlp_bias

        from megatron.core import tensor_parallel

        if self.moe_layer_recompute:
            if self.config.fp8:
                import transformer_engine.pytorch as te

                output, mlp_bias = te.distributed.checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.attn_tp_group,
                    hidden_states,
                )
            else:
                output, mlp_bias = tensor_parallel.checkpoint(
                    custom_forward, False, hidden_states
                )
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

    return patched_forward
