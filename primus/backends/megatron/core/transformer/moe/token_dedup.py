###############################################################################
# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Ported unchanged from ROCm/Megatron-LM PR #165 (megatron/core/transformer/moe/token_dedup.py).
###############################################################################
"""Fused payload and route-metadata packing for MoE token dedup."""

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


# RCCL is substantially slower for the otherwise 16,392-byte BF16 wire rows
# produced by hidden=8192/topk=2. Round every row to its next 16-byte boundary.
_PAYLOAD_ALIGNMENT_BYTES = 16


def dedup_payload_width(hidden_size: int, topk: int, element_size: int) -> int:
    """Return the activation-plus-route-metadata width with RCCL-safe alignment."""
    # Each route slot carries one exactly-representable local expert ID and one
    # routing probability.  For hidden=8192/topk=2/BF16 these four elements fit
    # inside the padding already required by the 16-byte RCCL alignment.
    payload_bytes = (hidden_size + 2 * topk) * element_size
    aligned_bytes = (
        (payload_bytes + _PAYLOAD_ALIGNMENT_BYTES - 1)
        // _PAYLOAD_ALIGNMENT_BYTES
        * _PAYLOAD_ALIGNMENT_BYTES
    )
    return (aligned_bytes + element_size - 1) // element_size


if triton is not None:

    @triton.jit
    def _pack_token_dedup_kernel(
        hidden_ptr,
        prob_ptr,
        token_ptr,
        expert_ptr,
        output_ptr,
        hidden_stride,
        hidden_size: tl.constexpr,
        topk: tl.constexpr,
        output_width: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        block = tl.program_id(1)
        columns = block * block_size + tl.arange(0, block_size)
        output_mask = columns < output_width
        hidden_mask = columns < hidden_size

        token = tl.load(token_ptr + row)
        hidden = tl.load(
            hidden_ptr + token * hidden_stride + columns,
            mask=hidden_mask,
            other=0.0,
        )

        expert_column = columns - hidden_size
        expert_mask = (columns >= hidden_size) & (
            columns < hidden_size + topk
        )
        expert = tl.load(
            expert_ptr + row * topk + expert_column,
            mask=expert_mask,
            other=0,
        )
        prob_column = columns - hidden_size - topk
        prob_mask = (columns >= hidden_size + topk) & (
            columns < hidden_size + 2 * topk
        )
        prob_expert = tl.load(
            expert_ptr + row * topk + prob_column,
            mask=prob_mask,
            other=-1,
        )
        prob = tl.load(
            prob_ptr + token * topk + prob_column,
            mask=prob_mask & (prob_expert >= 0),
            other=0.0,
        )
        metadata = tl.where(expert_mask, expert.to(tl.float32), prob)
        value = tl.where(hidden_mask, hidden, metadata)
        tl.store(output_ptr + row * output_width + columns, value, mask=output_mask)

    @triton.jit
    def _unpack_token_dedup_payload_grad_kernel(
        grad_output_ptr,
        token_wire_rows_ptr,
        slot_wire_rows_ptr,
        grad_hidden_ptr,
        grad_prob_ptr,
        grad_output_stride_0,
        grad_output_stride_1,
        num_grad_output_rows,
        hidden_size: tl.constexpr,
        topk: tl.constexpr,
        block_size: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        block = tl.program_id(1)
        columns = block * block_size + tl.arange(0, block_size)
        hidden_mask = columns < hidden_size
        accumulator = tl.zeros((block_size,), dtype=tl.float32)

        for slot in tl.static_range(0, topk):
            wire_row = tl.load(token_wire_rows_ptr + token * topk + slot)
            valid_wire_row = (wire_row >= 0) & (
                wire_row < num_grad_output_rows
            )
            safe_wire_row = tl.where(valid_wire_row, wire_row, 0)
            value = tl.load(
                grad_output_ptr
                + safe_wire_row * grad_output_stride_0
                + columns * grad_output_stride_1,
                mask=hidden_mask & valid_wire_row,
                other=0.0,
            )
            accumulator += value

        tl.store(
            grad_hidden_ptr + token * hidden_size + columns,
            accumulator,
            mask=hidden_mask,
        )

        # Every token/route slot occurs in exactly one wire row.  Have the first
        # hidden tile also recover those tiny metadata gradients so packing adds
        # no separate backward kernel.
        prob_mask = (block == 0) & (columns < topk)
        slot_wire_row = tl.load(
            slot_wire_rows_ptr + token * topk + columns,
            mask=prob_mask,
            other=0,
        )
        prob_grad = tl.load(
            grad_output_ptr
            + slot_wire_row * grad_output_stride_0
            + (hidden_size + topk + columns) * grad_output_stride_1,
            mask=prob_mask & (slot_wire_row >= 0) & (slot_wire_row < num_grad_output_rows),
            other=0.0,
        )
        tl.store(
            grad_prob_ptr + token * topk + columns,
            prob_grad,
            mask=prob_mask,
        )

    @triton.jit
    def _unpack_token_dedup_grad_kernel(
        grad_output_ptr,
        token_wire_rows_ptr,
        grad_hidden_ptr,
        grad_output_stride_0,
        grad_output_stride_1,
        grad_hidden_stride_0,
        num_grad_output_rows,
        hidden_size: tl.constexpr,
        topk: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Gather and sum bounded row gradients without metadata handling."""
        token = tl.program_id(0).to(tl.int64)
        block = tl.program_id(1)
        columns = block * block_size + tl.arange(0, block_size)
        hidden_mask = columns < hidden_size
        accumulator = tl.zeros((block_size,), dtype=tl.float32)

        for slot in tl.static_range(0, topk):
            wire_row = tl.load(token_wire_rows_ptr + token * topk + slot)
            valid_wire_row = (wire_row >= 0) & (
                wire_row < num_grad_output_rows
            )
            safe_wire_row = tl.where(valid_wire_row, wire_row, 0)
            value = tl.load(
                grad_output_ptr
                + safe_wire_row * grad_output_stride_0
                + columns * grad_output_stride_1,
                mask=hidden_mask & valid_wire_row,
                other=0.0,
            )
            accumulator += value

        tl.store(
            grad_hidden_ptr + token * grad_hidden_stride_0 + columns,
            accumulator,
            mask=hidden_mask,
        )

    @triton.jit
    def _segment_count_kernel(
        keys_ptr,  # int64 [n] keys in [-1, num_keys); -1 = inactive
        counts_ptr,  # int32 [num_blocks, num_keys] out
        n,
        num_keys: tl.constexpr,
        block: tl.constexpr,
    ):
        """Per block of ``block`` keys, count how often each key occurs."""
        pid = tl.program_id(0)
        offs = pid * block + tl.arange(0, block)
        keys = tl.load(keys_ptr + offs, mask=offs < n, other=-1)
        for key in tl.static_range(num_keys):
            count = tl.sum((keys == key).to(tl.int32), axis=0)
            tl.store(counts_ptr + pid * num_keys + key, count)

    @triton.jit
    def _segment_rank_kernel(
        keys_ptr,  # int64 [n]
        counts_ptr,  # int32 [num_blocks, num_keys] from _segment_count_kernel
        rank_ptr,  # int32 [n] out: exclusive rank of each element among equal keys
        n,
        num_keys: tl.constexpr,
        block: tl.constexpr,
        blocks_pow2: tl.constexpr,
    ):
        """Exclusive rank of every element among the elements with the same key,
        in array order: the block prefix from the per-block counts plus an
        in-block scan. Deterministic, no atomics, no global scan."""
        pid = tl.program_id(0)
        offs = pid * block + tl.arange(0, block)
        keys = tl.load(keys_ptr + offs, mask=offs < n, other=-1)
        blocks = tl.arange(0, blocks_pow2)
        rank = tl.zeros((block,), dtype=tl.int32)
        for key in tl.static_range(num_keys):
            prior = tl.sum(
                tl.load(counts_ptr + blocks * num_keys + key, mask=blocks < pid, other=0), axis=0
            )
            is_key = (keys == key).to(tl.int32)
            exclusive = tl.cumsum(is_key, axis=0) - is_key
            rank = tl.where(keys == key, prior + exclusive, rank)
        tl.store(rank_ptr + offs, rank, mask=offs < n)

    @triton.jit
    def _column_count_kernel(
        x_ptr,  # uint8 [n_rows, ncols]
        counts_ptr,  # int32 [num_blocks, ncols] out
        n_rows,
        ncols: tl.constexpr,
        cols_pow2: tl.constexpr,
        block: tl.constexpr,
    ):
        """Per block of rows, the per-column sum of a row-major 0/1 matrix."""
        pid = tl.program_id(0)
        rows = pid * block + tl.arange(0, block)
        cols = tl.arange(0, cols_pow2)
        mask = (rows[:, None] < n_rows) & (cols[None, :] < ncols)
        x = tl.load(x_ptr + rows[:, None] * ncols + cols[None, :], mask=mask, other=0).to(tl.int32)
        tl.store(counts_ptr + pid * ncols + cols, tl.sum(x, axis=0), mask=cols < ncols)

    @triton.jit
    def _column_scan_kernel(
        x_ptr,  # uint8 [n_rows, ncols]
        counts_ptr,  # int [num_blocks, ncols] (row stride counts_stride)
        out_ptr,  # int32 [n_rows, ncols] out: inclusive per-column prefix over rows
        n_rows,
        counts_stride,
        ncols: tl.constexpr,
        cols_pow2: tl.constexpr,
        block: tl.constexpr,
        blocks_pow2: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * block + tl.arange(0, block)
        cols = tl.arange(0, cols_pow2)
        mask = (rows[:, None] < n_rows) & (cols[None, :] < ncols)
        x = tl.load(x_ptr + rows[:, None] * ncols + cols[None, :], mask=mask, other=0).to(tl.int32)
        blocks = tl.arange(0, blocks_pow2)
        bmask = (blocks[:, None] < pid) & (cols[None, :] < ncols)
        prior = tl.sum(
            tl.load(counts_ptr + blocks[:, None] * counts_stride + cols[None, :], mask=bmask, other=0), axis=0
        ).to(tl.int32)
        inclusive = tl.cumsum(x, axis=0) + prior[None, :]
        tl.store(out_ptr + rows[:, None] * ncols + cols[None, :], inclusive, mask=mask)

    @triton.jit
    def _build_wire_send_index_kernel(
        routing_map_ptr,  # uint8 [T, E], slot-space routing bits
        order_ptr,  # int32 [T, EP], inclusive per-destination token prefix of hits
        send_counts_ptr,  # int64 [EP], wire rows per destination (exclusive prefix = row base)
        indices_ptr,  # int64 [T, K] out: set-bit expert ids, ascending
        slot_wire_rows_ptr,  # int64 [T, K] out: wire row of every route slot
        token_wire_rows_ptr,  # int64 [T, K] out: wire row at its first slot, else -1
        wire_tokens_ptr,  # int64 [Wmax] out: source token of each wire row
        wire_experts_ptr,  # int64 [Wmax, K] out: local expert per slot (prefilled -1)
        num_experts: tl.constexpr,
        experts_pow2: tl.constexpr,
        num_local_experts: tl.constexpr,
        ep_size: tl.constexpr,
        ep_pow2: tl.constexpr,
        topk: tl.constexpr,
    ):
        """One program per token builds every send-side dedup index in one pass.

        Replaces the ~28-launch torch composition in ``_build_send_buffers``
        (compactions, remainders, index_copy chains). Wire-row numbering is
        destination-major with tokens ascending inside a destination -- the
        exact order ``_compact_mask_indices`` produced -- via
        ``base[dest] + order[token, dest] - 1``. Slot order within a token is
        ascending expert id (torch.topk's tie order was arbitrary; any
        consistent order is valid because the probabilities ride in the same
        slots).
        """
        token = tl.program_id(0)
        cols = tl.arange(0, experts_pow2)
        emask = cols < num_experts
        bits = tl.load(
            routing_map_ptr + token * num_experts + cols, mask=emask, other=0
        ).to(tl.int32)
        set_mask = emask & (bits != 0)
        # Conditional top-1 leaves unused slots in the fixed-width top-2
        # payload. Initialize only those slots; their backward maps must be -1.
        ks = tl.arange(0, triton.next_power_of_2(topk))
        unused = (ks < topk) & (ks >= tl.sum(set_mask.to(tl.int32), axis=0))
        tl.store(indices_ptr + token * topk + ks, 0, mask=unused)
        tl.store(slot_wire_rows_ptr + token * topk + ks, -1, mask=unused)
        tl.store(token_wire_rows_ptr + token * topk + ks, -1, mask=unused)
        slot = tl.cumsum(bits, axis=0) - 1
        slot_ok = set_mask & (slot < topk)
        dest = cols // num_local_experts
        local_expert = (cols % num_local_experts).to(tl.int64)
        order_t = tl.load(order_ptr + token * ep_size + dest, mask=slot_ok, other=0).to(tl.int64)
        dests = tl.arange(0, ep_pow2)
        send_counts = tl.load(send_counts_ptr + dests, mask=dests < ep_size, other=0)
        base_d = tl.sum(tl.where(dests[None, :] < dest[:, None], send_counts[None, :], 0), axis=1)
        wire = base_d + order_t - 1

        tl.store(indices_ptr + token * topk + slot, cols.to(tl.int64), mask=slot_ok)
        tl.store(slot_wire_rows_ptr + token * topk + slot, wire, mask=slot_ok)

        # A slot is the wire row's representative iff it is the token's FIRST
        # set bit within that destination block (destinations are e // nle, so
        # blocks are contiguous in expert id).
        inblk_pref = tl.zeros((experts_pow2,), dtype=tl.int32)
        for d in tl.static_range(ep_size):
            blk = tl.where(dest == d, bits, 0)
            inblk_pref += tl.where(dest == d, tl.cumsum(blk, axis=0), 0)
        rep = slot_ok & (inblk_pref == 1)

        tl.store(
            token_wire_rows_ptr + token * topk + slot,
            tl.where(rep, wire, -1),
            mask=slot_ok,
        )
        tl.store(wire_tokens_ptr + wire, token.to(tl.int64), mask=rep)
        tl.store(wire_experts_ptr + wire * topk + slot, local_expert, mask=slot_ok)

    @triton.jit
    def _routing_counts_kernel(
        routing_ptr,  # uint8 [T, E] routing bits (slot space)
        hits_ptr,  # uint8 [T, EP] out: token routes to destination rank
        partial_ptr,  # int64 [num_blocks, E + EP] out: per-block expert counts | destination counts
        num_tokens,
        num_experts: tl.constexpr,
        experts_pow2: tl.constexpr,
        num_local_experts: tl.constexpr,
        ep_size: tl.constexpr,
        block: tl.constexpr,
    ):
        """One pass over a block of tokens: expert counts, destination hit map and
        per-destination wire-row counts (replaces sum / reshape-any / sum / cat)."""
        pid = tl.program_id(0)
        rows = pid * block + tl.arange(0, block)
        cols = tl.arange(0, experts_pow2)
        mask = (rows[:, None] < num_tokens) & (cols[None, :] < num_experts)
        bits = tl.load(routing_ptr + rows[:, None] * num_experts + cols[None, :], mask=mask, other=0).to(tl.int32)
        width = num_experts + ep_size
        tl.store(
            partial_ptr + pid * width + cols, tl.sum(bits, axis=0).to(tl.int64), mask=cols < num_experts
        )
        dest = cols // num_local_experts
        for d in tl.static_range(ep_size):
            hit = tl.sum(tl.where(dest[None, :] == d, bits, 0), axis=1) > 0
            tl.store(hits_ptr + rows * ep_size + d, hit.to(tl.uint8), mask=rows < num_tokens)
            tl.store(partial_ptr + pid * width + num_experts + d, tl.sum(hit.to(tl.int64), axis=0))

    @triton.jit
    def _recv_count_kernel(
        experts_ptr,  # payload dtype [R, K] (strided), received local expert ids, -1 inactive
        counts_ptr,  # int32 [num_blocks, nle] out
        num_rows,
        experts_stride_0,
        experts_stride_1,
        num_local_experts: tl.constexpr,
        rows_per_block: tl.constexpr,
        topk: tl.constexpr,
        block_pow2: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, block_pow2)
        in_block = offs < rows_per_block * topk
        row = (pid * rows_per_block + offs // topk).to(tl.int64)
        slot = offs % topk
        valid = in_block & (row < num_rows)
        expert = tl.load(
            experts_ptr + row * experts_stride_0 + slot * experts_stride_1, mask=valid, other=-1
        ).to(tl.int32)
        for key in tl.static_range(num_local_experts):
            tl.store(counts_ptr + pid * num_local_experts + key, tl.sum((expert == key).to(tl.int32), axis=0))

    @triton.jit
    def _recv_rank_index_kernel(
        experts_ptr,  # payload dtype [R, K] (strided)
        counts_ptr,  # int32 [num_blocks, nle] from _recv_count_kernel
        expert_counts_ptr,  # int64 [nle], rows per local expert (exclusive prefix = row base)
        probs_ptr,  # payload dtype [R, K] (strided), received routing probabilities
        wire_to_expert_rows_ptr,  # int64 [R, K] out: expert row per slot, else -1
        expert_wire_rows_ptr,  # int64 [A] out: expert-major row -> wire row
        expert_probs_ptr,  # fp32 [A] out: routing probability of each expert-major row
        num_rows,
        experts_stride_0,
        experts_stride_1,
        probs_stride_0,
        probs_stride_1,
        num_local_experts: tl.constexpr,
        experts_pow2: tl.constexpr,
        rows_per_block: tl.constexpr,
        topk: tl.constexpr,
        block_pow2: tl.constexpr,
        blocks_pow2: tl.constexpr,
    ):
        """Rank every received slot among the slots of its expert (block prefix +
        in-block scan), and write the expert-major indices and probabilities in the
        same pass, straight from the bf16 payload columns."""
        pid = tl.program_id(0)
        offs = tl.arange(0, block_pow2)
        in_block = offs < rows_per_block * topk
        row = (pid * rows_per_block + offs // topk).to(tl.int64)
        slot = offs % topk
        in_rows = in_block & (row < num_rows)
        expert = tl.load(
            experts_ptr + row * experts_stride_0 + slot * experts_stride_1, mask=in_rows, other=-1
        ).to(tl.int32)
        valid = in_rows & (expert >= 0)
        blocks = tl.arange(0, blocks_pow2)
        rank = tl.zeros((block_pow2,), dtype=tl.int32)
        for key in tl.static_range(num_local_experts):
            prior = tl.sum(
                tl.load(counts_ptr + blocks * num_local_experts + key, mask=blocks < pid, other=0), axis=0
            )
            is_key = (expert == key).to(tl.int32)
            rank = tl.where(expert == key, prior + tl.cumsum(is_key, axis=0) - is_key, rank)
        experts = tl.arange(0, experts_pow2)
        expert_counts = tl.load(expert_counts_ptr + experts, mask=experts < num_local_experts, other=0)
        safe_expert = tl.where(valid, expert, 0)
        base = tl.sum(
            tl.where(experts[None, :] < safe_expert[:, None], expert_counts[None, :], 0), axis=1
        )
        expert_row = base + rank.to(tl.int64)
        flat = row * topk + slot
        tl.store(wire_to_expert_rows_ptr + flat, tl.where(valid, expert_row, -1), mask=in_rows)
        tl.store(expert_wire_rows_ptr + expert_row, row, mask=valid)
        prob = tl.load(probs_ptr + row * probs_stride_0 + slot * probs_stride_1, mask=valid, other=0.0)
        tl.store(expert_probs_ptr + expert_row, prob.to(tl.float32), mask=valid)

    @triton.jit
    def _build_wire_recv_index_kernel(
        experts_ptr,  # int64 [R, K], received local experts (-1 inactive)
        rank_ptr,  # int32 [R*K], exclusive rank of each slot among slots of the same expert
        expert_counts_ptr,  # int64 [nle], rows per local expert (exclusive prefix = row base)
        probs_ptr,  # payload dtype, received routing probabilities (strided [R, K])
        wire_to_expert_rows_ptr,  # int64 [R, K] out: expert row per slot, else -1
        expert_wire_rows_ptr,  # int64 [A] out: expert-major row -> wire row
        expert_probs_ptr,  # fp32 [A] out: routing probability of each expert-major row
        probs_stride_0,
        probs_stride_1,
        num_local_experts: tl.constexpr,
        experts_pow2: tl.constexpr,
        topk: tl.constexpr,
        topk_pow2: tl.constexpr,
    ):
        """One program per received wire row builds the receive-side indices.

        Replaces the ~19-launch torch composition in ``_expand_received``.
        Expert-major numbering: ``base[e] + rank[flat_slot]`` reproduces the
        membership-compaction order (rows ascending within an expert). The
        same pass scatters each slot's routing probability to its expert-major
        row, so the dispatcher needs no separate probability gather.
        """
        row = tl.program_id(0)
        ks = tl.arange(0, topk_pow2)
        kmask = ks < topk
        expert = tl.load(experts_ptr + row * topk + ks, mask=kmask, other=-1)
        valid = kmask & (expert >= 0)
        safe_expert = tl.where(valid, expert, 0)
        flat = row * topk + ks
        rank = tl.load(rank_ptr + flat, mask=valid, other=0).to(tl.int64)
        experts = tl.arange(0, experts_pow2)
        expert_counts = tl.load(expert_counts_ptr + experts, mask=experts < num_local_experts, other=0)
        base_e = tl.sum(
            tl.where(experts[None, :] < safe_expert[:, None], expert_counts[None, :], 0), axis=1
        )
        expert_row = base_e + rank
        tl.store(
            wire_to_expert_rows_ptr + flat,
            tl.where(valid, expert_row, -1),
            mask=kmask,
        )
        tl.store(expert_wire_rows_ptr + expert_row, row.to(tl.int64), mask=valid)
        prob = tl.load(
            probs_ptr + row * probs_stride_0 + ks * probs_stride_1, mask=valid, other=0.0
        )
        tl.store(expert_probs_ptr + expert_row, prob.to(tl.float32), mask=valid)

    @triton.jit
    def _weighted_reduce_token_dedup_kernel(
        expert_output_ptr,
        wire_to_expert_rows_ptr,
        wire_prob_ptr,
        output_ptr,
        wire_prob_stride,
        hidden_size: tl.constexpr,
        topk: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """One program per (wire row, hidden tile): probability-weighted sum of the
        wire row's expert rows (the paper-era combine; experts then skip the multiply)."""
        wire_row = tl.program_id(0).to(tl.int64)
        block = tl.program_id(1)
        columns = block * block_size + tl.arange(0, block_size)
        hidden_mask = columns < hidden_size
        accumulator = tl.zeros((block_size,), dtype=tl.float32)
        for slot in tl.static_range(0, topk):
            expert_row = tl.load(wire_to_expert_rows_ptr + wire_row * topk + slot)
            valid = expert_row >= 0
            safe_expert_row = tl.where(valid, expert_row, 0)
            expert_value = tl.load(
                expert_output_ptr + safe_expert_row * hidden_size + columns,
                mask=hidden_mask & valid,
                other=0.0,
            )
            probability = tl.load(
                wire_prob_ptr + wire_row * wire_prob_stride + slot, mask=valid, other=0.0
            )
            # Round each weighted product to the activation dtype before the sum,
            # matching the eager expression the stock dispatcher produces.
            product = (expert_value * probability).to(output_ptr.dtype.element_ty)
            accumulator += product.to(tl.float32)
        tl.store(output_ptr + wire_row * hidden_size + columns, accumulator, mask=hidden_mask)

    @triton.jit
    def _weighted_reduce_token_dedup_backward_kernel(
        grad_output_ptr,
        expert_output_ptr,
        wire_to_expert_rows_ptr,
        wire_prob_ptr,
        grad_expert_ptr,
        grad_prob_ptr,
        wire_prob_stride,
        grad_prob_stride,
        hidden_size: tl.constexpr,
        topk: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """One program per (wire row, slot): grad of the expert row = grad_out * p, and
        grad of p = <grad_out, expert row>. Every expert row has exactly one owner slot."""
        assignment = tl.program_id(0).to(tl.int64)
        wire_row = assignment // topk
        slot = assignment % topk
        expert_row = tl.load(wire_to_expert_rows_ptr + assignment)
        valid = expert_row >= 0
        safe_expert_row = tl.where(valid, expert_row, 0)
        probability = tl.load(
            wire_prob_ptr + wire_row * wire_prob_stride + slot, mask=valid, other=0.0
        )
        accumulator = tl.zeros((block_size,), dtype=tl.float32)
        columns = tl.arange(0, block_size)
        for block in tl.static_range(0, (hidden_size + block_size - 1) // block_size):
            offsets = block * block_size + columns
            hidden_mask = offsets < hidden_size
            grad_output = tl.load(
                grad_output_ptr + wire_row * hidden_size + offsets, mask=hidden_mask & valid, other=0.0
            )
            expert_value = tl.load(
                expert_output_ptr + safe_expert_row * hidden_size + offsets,
                mask=hidden_mask & valid,
                other=0.0,
            )
            tl.store(
                grad_expert_ptr + safe_expert_row * hidden_size + offsets,
                (grad_output * probability).to(grad_expert_ptr.dtype.element_ty),
                mask=hidden_mask & valid,
            )
            accumulator += (grad_output * expert_value).to(tl.float32)
        tl.store(
            grad_prob_ptr + wire_row * grad_prob_stride + slot, tl.sum(accumulator, axis=0), mask=valid
        )

    @triton.jit
    def _restore_token_dedup_kernel(
        returned_ptr,
        token_wire_rows_ptr,
        output_ptr,
        hidden_size: tl.constexpr,
        topk: tl.constexpr,
        block_size: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        block = tl.program_id(1)
        columns = block * block_size + tl.arange(0, block_size)
        hidden_mask = columns < hidden_size
        accumulator = tl.zeros((block_size,), dtype=tl.float32)

        for slot in tl.static_range(0, topk):
            wire_row = tl.load(token_wire_rows_ptr + token * topk + slot)
            valid = wire_row >= 0
            safe_wire_row = tl.where(valid, wire_row, 0)
            value = tl.load(
                returned_ptr + safe_wire_row * hidden_size + columns,
                mask=hidden_mask & valid,
                other=0.0,
            )
            accumulator += value.to(tl.float32)

        tl.store(
            output_ptr + token * hidden_size + columns,
            accumulator,
            mask=hidden_mask,
        )


# Row-wise variants of the bandwidth-bound dedup kernels: one program per output row
# holding the whole hidden row (<= _ROWWISE_MAX_HIDDEN elements) instead of a grid of
# 256/1024-wide tiles. Same results bit for bit, ~40% less time per call for the pack
# forward/backward and ~25% for the bounded gather-sum at hidden=2048 (MI300X).
_ROWWISE_MAX_HIDDEN = 4096


def _rowwise_ok(hidden_size: int) -> bool:
    return triton is not None and triton.next_power_of_2(hidden_size) <= _ROWWISE_MAX_HIDDEN


if triton is not None:

    @triton.jit
    def _pack_rowwise_kernel(
        hidden_ptr, prob_ptr, token_ptr, expert_ptr, output_ptr, hidden_stride,
        hidden_size: tl.constexpr, topk: tl.constexpr, output_width: tl.constexpr,
        hidden_pow2: tl.constexpr, meta_pow2: tl.constexpr,
    ):
        """One program per wire row: copy the token's activation row, then write the
        expert-id | probability | zero-padding metadata columns."""
        row = tl.program_id(0).to(tl.int64)
        token = tl.load(token_ptr + row)
        cols = tl.arange(0, hidden_pow2)
        hmask = cols < hidden_size
        hidden = tl.load(hidden_ptr + token * hidden_stride + cols, mask=hmask, other=0.0)
        tl.store(output_ptr + row * output_width + cols, hidden, mask=hmask)
        mc = tl.arange(0, meta_pow2)
        mcols = hidden_size + mc
        omask = mcols < output_width
        expert_mask = mc < topk
        expert = tl.load(expert_ptr + row * topk + mc, mask=expert_mask, other=0)
        pc = mc - topk
        prob_mask = (mc >= topk) & (mc < 2 * topk)
        prob_expert = tl.load(expert_ptr + row * topk + pc, mask=prob_mask, other=-1)
        prob = tl.load(prob_ptr + token * topk + pc, mask=prob_mask & (prob_expert >= 0), other=0.0)
        meta = tl.where(expert_mask, expert.to(tl.float32), prob)
        tl.store(output_ptr + row * output_width + mcols, meta, mask=omask)

    @triton.jit
    def _gather_sum_rowwise_kernel(
        src_ptr, rows_ptr, out_ptr, src_stride_0, src_stride_1, out_stride_0, num_src_rows,
        hidden_size: tl.constexpr, topk: tl.constexpr, hidden_pow2: tl.constexpr,
    ):
        """out[row] = sum over the row's up to ``topk`` source rows (-1 / out of range = skip)."""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, hidden_pow2)
        hmask = cols < hidden_size
        acc = tl.zeros((hidden_pow2,), dtype=tl.float32)
        for slot in tl.static_range(0, topk):
            src_row = tl.load(rows_ptr + row * topk + slot)
            valid = (src_row >= 0) & (src_row < num_src_rows)
            safe = tl.where(valid, src_row, 0)
            acc += tl.load(src_ptr + safe * src_stride_0 + cols * src_stride_1, mask=hmask & valid, other=0.0)
        tl.store(out_ptr + row * out_stride_0 + cols, acc, mask=hmask)

    @triton.jit
    def _pack_bwd_rowwise_kernel(
        grad_output_ptr, token_wire_rows_ptr, slot_wire_rows_ptr, grad_hidden_ptr, grad_prob_ptr,
        go_stride_0, go_stride_1, num_go_rows,
        hidden_size: tl.constexpr, topk: tl.constexpr, hidden_pow2: tl.constexpr, topk_pow2: tl.constexpr,
    ):
        """One program per token: sum of its wire rows' activation gradients plus the
        per-slot probability gradients (each slot occurs in exactly one wire row)."""
        token = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, hidden_pow2)
        hmask = cols < hidden_size
        acc = tl.zeros((hidden_pow2,), dtype=tl.float32)
        for slot in tl.static_range(0, topk):
            wr = tl.load(token_wire_rows_ptr + token * topk + slot)
            valid = (wr >= 0) & (wr < num_go_rows)
            safe = tl.where(valid, wr, 0)
            acc += tl.load(grad_output_ptr + safe * go_stride_0 + cols * go_stride_1, mask=hmask & valid, other=0.0)
        tl.store(grad_hidden_ptr + token * hidden_size + cols, acc, mask=hmask)
        ks = tl.arange(0, topk_pow2)
        kmask = ks < topk
        swr = tl.load(slot_wire_rows_ptr + token * topk + ks, mask=kmask, other=0)
        pg = tl.load(
            grad_output_ptr + swr * go_stride_0 + (hidden_size + topk + ks) * go_stride_1,
            mask=kmask & (swr >= 0) & (swr < num_go_rows), other=0.0,
        )
        tl.store(grad_prob_ptr + token * topk + ks, pg, mask=kmask)


def _gather_sum_rows(src, rows, out, hidden_size: int, topk: int):
    """``out[r, :hidden_size] = sum_k src[rows[r, k]]`` for valid ``rows``; tiled or row-wise launch."""
    num_rows = rows.shape[0]
    if num_rows == 0:
        return out
    if _rowwise_ok(hidden_size):
        _gather_sum_rowwise_kernel[(num_rows,)](
            src, rows, out, src.stride(0), src.stride(1), out.stride(0), src.shape[0],
            hidden_size=hidden_size, topk=topk, hidden_pow2=triton.next_power_of_2(hidden_size),
        )
    else:
        block_size = 1024
        _unpack_token_dedup_grad_kernel[(num_rows, triton.cdiv(hidden_size, block_size))](
            src, rows, out, src.stride(0), src.stride(1), out.stride(0), src.shape[0],
            hidden_size=hidden_size, topk=topk, block_size=block_size,
        )
    return out


class _PackTokenDedup(torch.autograd.Function):
    """Pack activations and routes while preserving hidden/probability gradients."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        wire_tokens: torch.Tensor,
        wire_experts: torch.Tensor,
        token_wire_rows: torch.Tensor,
        slot_wire_rows: torch.Tensor,
    ) -> torch.Tensor:
        num_wire_rows = wire_tokens.numel()
        hidden_size = hidden_states.shape[1]
        topk = wire_experts.shape[1]
        output_width = dedup_payload_width(
            hidden_size, topk, hidden_states.element_size()
        )
        ctx.hidden_shape = hidden_states.shape
        ctx.prob_shape = probs.shape
        ctx.prob_dtype = probs.dtype
        ctx.save_for_backward(wire_tokens, token_wire_rows, slot_wire_rows)

        if num_wire_rows == 0:
            return hidden_states.new_empty((0, output_width))

        if triton is None or not hidden_states.is_cuda:
            payload = torch.cat(
                (
                    hidden_states.index_select(0, wire_tokens),
                    wire_experts.to(dtype=hidden_states.dtype),
                    torch.where(
                        wire_experts >= 0,
                        probs.index_select(0, wire_tokens),
                        torch.zeros((), dtype=probs.dtype, device=probs.device),
                    ).to(dtype=hidden_states.dtype),
                ),
                dim=1,
            )
            if payload.shape[1] != output_width:
                payload = torch.nn.functional.pad(
                    payload, (0, output_width - payload.shape[1])
                )
            return payload

        output = hidden_states.new_empty((num_wire_rows, output_width))
        if _rowwise_ok(hidden_size):
            _pack_rowwise_kernel[(num_wire_rows,)](
                hidden_states, probs, wire_tokens, wire_experts, output, hidden_states.stride(0),
                hidden_size=hidden_size, topk=topk, output_width=output_width,
                hidden_pow2=triton.next_power_of_2(hidden_size),
                meta_pow2=max(triton.next_power_of_2(output_width - hidden_size), 16),
            )
            return output
        # Hidden=8192 otherwise launches 33 programs per wire row and reloads
        # the row's token/expert metadata in every program. A wider tile cuts
        # that launch/grid and metadata overhead while retaining enough rows to
        # saturate the device for the dispatcher workloads.
        block_size = 1024
        grid = (
            num_wire_rows,
            triton.cdiv(output_width, block_size),
        )
        _pack_token_dedup_kernel[grid](
            hidden_states,
            probs,
            wire_tokens,
            wire_experts,
            output,
            hidden_states.stride(0),
            hidden_size=hidden_size,
            topk=topk,
            output_width=output_width,
            block_size=block_size,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        wire_tokens, token_wire_rows, slot_wire_rows = ctx.saved_tensors
        num_tokens, hidden_size = ctx.hidden_shape
        if triton is None or not grad_output.is_cuda:
            grad_hidden = grad_output.new_zeros((num_tokens, hidden_size))
            grad_hidden.index_add_(0, wire_tokens, grad_output[:, :hidden_size])
            topk = ctx.prob_shape[1]
            grad_prob = torch.empty(
                ctx.prob_shape,
                dtype=ctx.prob_dtype,
                device=grad_output.device,
            )
            slots = torch.arange(topk, device=grad_output.device).view(1, topk)
            # Inactive slots carry -1; clamp for the gather and zero them after, matching
            # the masked store of the triton backward.
            grad_prob.copy_(
                grad_output[
                    slot_wire_rows.clamp(min=0),
                    hidden_size + topk + slots,
                ]
            )
            grad_prob.mul_((slot_wire_rows >= 0).to(dtype=grad_prob.dtype))
            return grad_hidden, grad_prob, None, None, None, None

        grad_hidden = grad_output.new_empty((num_tokens, hidden_size))
        grad_prob = torch.empty(
            ctx.prob_shape,
            dtype=ctx.prob_dtype,
            device=grad_output.device,
        )
        if _rowwise_ok(hidden_size) and num_tokens:
            topk = token_wire_rows.shape[1]
            _pack_bwd_rowwise_kernel[(num_tokens,)](
                grad_output, token_wire_rows, slot_wire_rows, grad_hidden, grad_prob,
                grad_output.stride(0), grad_output.stride(1), grad_output.shape[0],
                hidden_size=hidden_size, topk=topk, hidden_pow2=triton.next_power_of_2(hidden_size),
                topk_pow2=triton.next_power_of_2(topk),
            )
            return grad_hidden, grad_prob, None, None, None, None
        block_size = 256
        grid = (num_tokens, triton.cdiv(hidden_size, block_size))
        _unpack_token_dedup_payload_grad_kernel[grid](
            grad_output,
            token_wire_rows,
            slot_wire_rows,
            grad_hidden,
            grad_prob,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.shape[0],
            hidden_size=hidden_size,
            topk=token_wire_rows.shape[1],
            block_size=block_size,
        )
        return grad_hidden, grad_prob, None, None, None, None


class _ExpandTokenDedup(torch.autograd.Function):
    """Expand wire rows while gathering their gradients without atomics."""

    @staticmethod
    def forward(
        ctx,
        received: torch.Tensor,
        wire_rows: torch.Tensor,
        wire_to_expert_rows: torch.Tensor,
    ) -> torch.Tensor:
        ctx.input_shape = received.shape
        ctx.save_for_backward(wire_rows, wire_to_expert_rows)
        return received.index_select(0, wire_rows)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        wire_rows, wire_to_expert_rows = ctx.saved_tensors
        num_wire_rows, hidden_size = ctx.input_shape
        grad_received = _expanded_token_dedup_grad(
            grad_output,
            wire_to_expert_rows,
            num_wire_rows,
            hidden_size,
            wire_rows,
        )
        return grad_received, None, None


def _expanded_token_dedup_grad(
    grad_output: torch.Tensor,
    wire_to_expert_rows: torch.Tensor,
    num_wire_rows: int,
    hidden_size: int,
    wire_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather one expert-major gradient back to its deduplicated wire rows."""
    if triton is None or not grad_output.is_cuda:
        if wire_rows is None:
            active = wire_to_expert_rows >= 0
            wire_rows = torch.nonzero(active, as_tuple=False)[:, 0]
        grad_received = grad_output.new_zeros((num_wire_rows, hidden_size))
        grad_received.index_add_(0, wire_rows, grad_output)
        return grad_received

    grad_received = grad_output.new_empty((num_wire_rows, hidden_size))
    # Every output row is written by exactly one program, so multiple peer
    # sources can be gathered without atomics.
    return _gather_sum_rows(grad_output, wire_to_expert_rows, grad_received, hidden_size, wire_to_expert_rows.shape[1])


class _PermuteTokenDedup(torch.autograd.Function):
    """Apply a known row permutation with a gather-only backward."""

    @staticmethod
    def forward(
        ctx,
        expert_output: torch.Tensor,
        wire_to_expert_row: torch.Tensor,
        expert_row_to_wire_rank: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(expert_row_to_wire_rank)
        return expert_output.index_select(0, wire_to_expert_row)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (expert_row_to_wire_rank,) = ctx.saved_tensors
        grad_expert_output = grad_output.index_select(
            0, expert_row_to_wire_rank
        )
        return grad_expert_output, None, None


class _ReduceTokenDedup(torch.autograd.Function):
    """Sum each wire row's expert rows; the adjoint of ``_ExpandTokenDedup``."""

    @staticmethod
    def forward(ctx, expert_output, expert_wire_rows, wire_to_expert_rows):
        ctx.save_for_backward(expert_wire_rows)
        num_wire_rows, hidden_size = wire_to_expert_rows.shape[0], expert_output.shape[1]
        if num_wire_rows == 0:
            return expert_output.new_empty((0, hidden_size))
        return _expanded_token_dedup_grad(
            expert_output, wire_to_expert_rows, num_wire_rows, hidden_size, expert_wire_rows
        )

    @staticmethod
    def backward(ctx, grad_output):
        (expert_wire_rows,) = ctx.saved_tensors
        return grad_output.index_select(0, expert_wire_rows), None, None


class _RestoreTokenDedup(torch.autograd.Function):
    """Sum destination partials into tokens with a gather-only backward."""

    @staticmethod
    def forward(
        ctx,
        returned: torch.Tensor,
        token_wire_rows: torch.Tensor,
        wire_tokens: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, topk = token_wire_rows.shape
        hidden_size = returned.shape[1]
        ctx.save_for_backward(wire_tokens)
        if num_tokens == 0:
            return returned.new_empty((0, hidden_size))

        if triton is None or not returned.is_cuda:
            output = returned.new_zeros((num_tokens, hidden_size))
            output.index_add_(0, wire_tokens, returned)
            return output

        output = returned.new_empty((num_tokens, hidden_size))
        block_size = 1024
        grid = (num_tokens, triton.cdiv(hidden_size, block_size))
        _restore_token_dedup_kernel[grid](
            returned,
            token_wire_rows,
            output,
            hidden_size=hidden_size,
            topk=topk,
            block_size=block_size,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (wire_tokens,) = ctx.saved_tensors
        return grad_output.index_select(0, wire_tokens), None, None


class _WeightedReduceTokenDedup(torch.autograd.Function):
    """Probability-weighted sum of each wire row's expert rows (paper-era combine).
    The experts receive no probabilities in this mode; the weights are applied here
    and their gradient flows back into the received payload's probability columns."""

    @staticmethod
    def forward(ctx, expert_output, wire_to_expert_rows, wire_probs):
        num_wire_rows, topk = wire_to_expert_rows.shape
        hidden_size = expert_output.shape[1]
        wire_probs = wire_probs.contiguous()
        ctx.save_for_backward(expert_output, wire_to_expert_rows, wire_probs)
        output = expert_output.new_empty((num_wire_rows, hidden_size))
        if num_wire_rows:
            block_size = 1024
            grid = (num_wire_rows, triton.cdiv(hidden_size, block_size))
            _weighted_reduce_token_dedup_kernel[grid](
                expert_output, wire_to_expert_rows, wire_probs, output, wire_probs.stride(0),
                hidden_size=hidden_size, topk=topk, block_size=block_size,
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        expert_output, wire_to_expert_rows, wire_probs = ctx.saved_tensors
        num_wire_rows, topk = wire_to_expert_rows.shape
        hidden_size = expert_output.shape[1]
        grad_expert = torch.empty_like(expert_output)
        grad_prob = torch.zeros_like(wire_probs)
        if num_wire_rows:
            grad_output = grad_output.contiguous()
            _weighted_reduce_token_dedup_backward_kernel[(num_wire_rows * topk,)](
                grad_output, expert_output, wire_to_expert_rows, wire_probs, grad_expert, grad_prob,
                wire_probs.stride(0), grad_prob.stride(0),
                hidden_size=hidden_size, topk=topk, block_size=1024,
            )
        return grad_expert, None, grad_prob


def weighted_reduce_token_dedup_expert_output(expert_output, wire_to_expert_rows, wire_probs):
    """Return one probability-weighted expert-output row per wire row."""
    return _WeightedReduceTokenDedup.apply(expert_output, wire_to_expert_rows, wire_probs)


class _UnpackTokenDedupPayload(torch.autograd.Function):
    """Expand the received payload's activations to expert-major rows and hand out the
    per-expert-row probabilities, writing the payload gradient in ONE pass.

    The payload is sliced three ways (activation | expert ids | probabilities); letting
    autograd differentiate the slices costs a zero-filled payload-shaped gradient plus an
    add per slice. Here the bounded gather-sum writes the activation columns directly
    into the payload gradient and the probability columns are filled by a gather.
    """

    @staticmethod
    def forward(
        ctx,
        payload: torch.Tensor,
        expert_wire_rows: torch.Tensor,
        wire_to_expert_rows: torch.Tensor,
        expert_probs: torch.Tensor,
        hidden_size: int,
        topk: int,
        return_wire_probs: bool = False,
    ):
        ctx.save_for_backward(expert_wire_rows, wire_to_expert_rows)
        ctx.hidden_size, ctx.topk = hidden_size, topk
        ctx.payload_shape, ctx.payload_dtype = payload.shape, payload.dtype
        ctx.return_wire_probs = return_wire_probs
        expanded = payload[:, :hidden_size].index_select(0, expert_wire_rows)
        if return_wire_probs:
            # The [R, K] probability columns themselves (weighted-combine mode).
            return expanded, payload[:, hidden_size + topk : hidden_size + 2 * topk].contiguous()
        return expanded, expert_probs

    @staticmethod
    def backward(ctx, grad_expanded: torch.Tensor, grad_expert_probs: torch.Tensor):
        expert_wire_rows, wire_to_expert_rows = ctx.saved_tensors
        hidden_size, topk = ctx.hidden_size, ctx.topk
        num_rows, width = ctx.payload_shape
        grad_payload = torch.empty(ctx.payload_shape, dtype=ctx.payload_dtype, device=grad_expanded.device)
        if num_rows:
            _gather_sum_rows(grad_expanded, wire_to_expert_rows, grad_payload, hidden_size, topk)
        # Expert-id columns and alignment padding carry no gradient; probabilities do.
        grad_payload[:, hidden_size:].zero_()
        if ctx.return_wire_probs:
            if grad_expert_probs is not None and num_rows:
                grad_payload[:, hidden_size + topk : hidden_size + 2 * topk].copy_(grad_expert_probs)
        elif grad_expert_probs is not None and num_rows:
            valid = wire_to_expert_rows >= 0
            grad = grad_expert_probs.index_select(0, wire_to_expert_rows.clamp(min=0).reshape(-1))
            grad = torch.where(valid, grad.view_as(wire_to_expert_rows), torch.zeros((), dtype=grad.dtype, device=grad.device))
            grad_payload[:, hidden_size + topk : hidden_size + 2 * topk].copy_(grad)
        return grad_payload, None, None, None, None, None, None


def unpack_token_dedup_payload(
    payload: torch.Tensor,
    expert_wire_rows: torch.Tensor,
    wire_to_expert_rows: torch.Tensor,
    expert_probs: torch.Tensor,
    hidden_size: int,
    topk: int,
    return_wire_probs: bool = False,
):
    """Return ``(expanded, probs)`` from the received payload with a single fused
    backward into the payload gradient. ``probs`` is the per-expert-row probability
    vector, or with ``return_wire_probs`` the ``[R, K]`` wire-row probabilities."""
    return _UnpackTokenDedupPayload.apply(
        payload, expert_wire_rows, wire_to_expert_rows, expert_probs, hidden_size, topk,
        return_wire_probs,
    )


def pack_token_dedup_payload(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    wire_tokens: torch.Tensor,
    wire_experts: torch.Tensor,
    token_wire_rows: torch.Tensor,
    slot_wire_rows: torch.Tensor,
) -> torch.Tensor:
    """Return contiguous ``[activation | expert IDs | probabilities]`` rows."""
    return _PackTokenDedup.apply(
        hidden_states,
        probs,
        wire_tokens,
        wire_experts,
        token_wire_rows,
        slot_wire_rows,
    )


def expand_token_dedup_received(
    received: torch.Tensor,
    wire_rows: torch.Tensor,
    wire_to_expert_rows: torch.Tensor,
) -> torch.Tensor:
    """Fan wire rows out to experts with a bounded gather-sum backward."""
    return _ExpandTokenDedup.apply(received, wire_rows, wire_to_expert_rows)


def permute_token_dedup_expert_output(
    expert_output: torch.Tensor,
    wire_to_expert_row: torch.Tensor,
    expert_row_to_wire_rank: torch.Tensor,
) -> torch.Tensor:
    """Restore wire order without the atomic backward of ``index_select``."""
    return _PermuteTokenDedup.apply(
        expert_output, wire_to_expert_row, expert_row_to_wire_rank
    )


def reduce_token_dedup_expert_output(
    expert_output: torch.Tensor,
    expert_wire_rows: torch.Tensor,
    wire_to_expert_rows: torch.Tensor,
) -> torch.Tensor:
    """Return one summed expert-output row per wire row."""
    return _ReduceTokenDedup.apply(expert_output, expert_wire_rows, wire_to_expert_rows)


def restore_token_dedup_output(
    returned: torch.Tensor,
    token_wire_rows: torch.Tensor,
    wire_tokens: torch.Tensor,
) -> torch.Tensor:
    """Sum returned destination rows into the original local token order."""
    return _RestoreTokenDedup.apply(returned, token_wire_rows, wire_tokens)


def fused_index_build_available(device: torch.device) -> bool:
    """True when the single-kernel (Triton) index builders can run on ``device``;
    otherwise the equivalent multi-op torch composition is used."""
    return triton is not None and device.type == "cuda"


def column_prefix_sum(
    x: torch.Tensor, counts: torch.Tensor | None = None, block: int = 1024
) -> torch.Tensor:
    """Inclusive prefix sum over rows of a row-major bool/uint8 ``[n_rows, ncols]``
    matrix, as int32, with a two-pass block scan (no global scan kernel).
    ``counts`` may supply the per-``block`` column sums (``[num_blocks, ncols]``,
    any integer dtype, any row stride) to skip the counting pass."""
    n_rows, ncols = x.shape
    num_blocks = max(triton.cdiv(n_rows, block), 1)
    x8 = x.contiguous().view(torch.uint8)
    out = torch.empty((n_rows, ncols), dtype=torch.int32, device=x.device)
    if n_rows:
        cols_pow2 = triton.next_power_of_2(ncols)
        if counts is None:
            counts = torch.empty((num_blocks, ncols), dtype=torch.int32, device=x.device)
            _column_count_kernel[(num_blocks,)](
                x8, counts, n_rows, ncols=ncols, cols_pow2=cols_pow2, block=block
            )
        _column_scan_kernel[(num_blocks,)](
            x8, counts, out, n_rows, counts.stride(0), ncols=ncols, cols_pow2=cols_pow2, block=block,
            blocks_pow2=triton.next_power_of_2(num_blocks),
        )
    return out


def routing_counts(routing_map: torch.Tensor, num_local_experts: int, ep_size: int, block: int = 256):
    """One fused pass over the routing map. Returns ``(packed, hits, partial)``:
    ``packed`` int64 ``[E + EP]`` = per-expert token counts followed by per-destination
    wire-row counts (the metadata all-gather payload), ``hits`` uint8 ``[T, EP]`` and
    ``partial`` int64 ``[num_blocks, E + EP]`` the per-``block`` partial counts (its
    destination columns feed :func:`column_prefix_sum`)."""
    num_tokens, num_experts = routing_map.shape
    device = routing_map.device
    num_blocks = max(triton.cdiv(num_tokens, block), 1)
    hits = torch.empty((num_tokens, ep_size), dtype=torch.uint8, device=device)
    partial = torch.zeros((num_blocks, num_experts + ep_size), dtype=torch.long, device=device)
    if num_tokens:
        _routing_counts_kernel[(num_blocks,)](
            routing_map.contiguous().view(torch.uint8),
            hits,
            partial,
            num_tokens,
            num_experts=num_experts,
            experts_pow2=triton.next_power_of_2(num_experts),
            num_local_experts=num_local_experts,
            ep_size=ep_size,
            block=block,
        )
    return partial.sum(dim=0), hits, partial


def segment_ranks(keys: torch.Tensor, num_keys: int) -> torch.Tensor:
    """Exclusive rank of every element of ``keys`` (int64, -1 = inactive) among the
    elements with the same key, in array order; int32. Two-pass block scan."""
    n = keys.numel()
    block = 1024
    num_blocks = max(triton.cdiv(n, block), 1)
    counts = torch.empty((num_blocks, num_keys), dtype=torch.int32, device=keys.device)
    rank = torch.empty((n,), dtype=torch.int32, device=keys.device)
    if n:
        _segment_count_kernel[(num_blocks,)](keys, counts, n, num_keys=num_keys, block=block)
        _segment_rank_kernel[(num_blocks,)](
            keys, counts, rank, n, num_keys=num_keys, block=block,
            blocks_pow2=triton.next_power_of_2(num_blocks),
        )
    return rank


def build_wire_send_index(
    routing_map: torch.Tensor,
    hits: torch.Tensor,
    dedup_send_counts: torch.Tensor,
    num_local_experts: int,
    ep_size: int,
    topk: int,
    max_wire_rows: int,
    hit_block_counts: torch.Tensor | None = None,
    hit_block: int = 1024,
):
    """Build every send-side dedup index tensor with one kernel launch.

    ``hits`` is the ``[tokens, ep_size]`` destination hit map and
    ``dedup_send_counts`` its per-destination column sums, both pure GPU
    tensors -- so this needs NO host split values and can run while the
    metadata all-gather is still in flight. The wire tensors are sized
    ``max_wire_rows`` (an upper bound); the caller narrows them to the real
    count once the splits have landed on the host.

    Returns ``(indices, slot_wire_rows, token_wire_rows, wire_tokens,
    wire_experts)`` with the same semantics the torch path in
    ``_build_send_buffers`` produces, except slot order is ascending expert id
    rather than torch.topk's tie order (equivalent: probabilities and expert
    ids ride in the same slots).
    """
    num_tokens, num_experts = routing_map.shape
    device = routing_map.device
    indices = torch.empty((num_tokens, topk), dtype=torch.long, device=device)
    slot_wire_rows = torch.empty((num_tokens, topk), dtype=torch.long, device=device)
    token_wire_rows = torch.empty((num_tokens, topk), dtype=torch.long, device=device)
    wire_tokens = torch.empty((max_wire_rows,), dtype=torch.long, device=device)
    # Inactive slots of a wire row must read -1 (the pack kernel masks
    # probabilities on them); the kernel writes active slots only.
    wire_experts = torch.full((max_wire_rows, topk), -1, dtype=torch.long, device=device)
    if num_tokens:
        order = column_prefix_sum(hits, hit_block_counts, hit_block)
        _build_wire_send_index_kernel[(num_tokens,)](
            routing_map.contiguous().view(torch.uint8),
            order,
            dedup_send_counts,
            indices,
            slot_wire_rows,
            token_wire_rows,
            wire_tokens,
            wire_experts,
            num_experts=num_experts,
            experts_pow2=triton.next_power_of_2(num_experts),
            num_local_experts=num_local_experts,
            ep_size=ep_size,
            ep_pow2=triton.next_power_of_2(ep_size),
            topk=topk,
        )
    return indices, slot_wire_rows, token_wire_rows, wire_tokens, wire_experts


def build_wire_recv_index(
    received_experts: torch.Tensor,
    tokens_per_local_expert: torch.Tensor,
    num_assignments: int,
    received_probs: torch.Tensor,
):
    """Build the receive-side expert-major indices with one kernel launch.

    Returns ``(wire_to_expert_rows, expert_wire_rows, expert_probs)``: the
    first maps each received (wire row, slot) to its expert-major row (-1
    inactive), the second maps each expert-major row back to its wire row --
    exactly what the torch membership-compaction path in ``_expand_received``
    produced -- and the third is the fp32 routing probability of every
    expert-major row. ``received_experts`` and ``received_probs`` are the
    ``[R, K]`` column slices of the payload (any dtype holding the integer ids
    exactly, -1 = inactive); two launches, no host values beyond the sizes.
    """
    num_rows, topk = received_experts.shape
    device = received_experts.device
    num_local_experts = tokens_per_local_expert.shape[0]
    wire_to_expert_rows = torch.empty((num_rows, topk), dtype=torch.long, device=device)
    expert_wire_rows = torch.empty((num_assignments,), dtype=torch.long, device=device)
    expert_probs = torch.empty((num_assignments,), dtype=torch.float32, device=device)
    rows_per_block = max(1024 // topk, 1)
    num_blocks = max(triton.cdiv(num_rows, rows_per_block), 1)
    counts = torch.empty((num_blocks, num_local_experts), dtype=torch.int32, device=device)
    if num_rows:
        common = dict(
            num_local_experts=num_local_experts,
            rows_per_block=rows_per_block,
            topk=topk,
            block_pow2=triton.next_power_of_2(rows_per_block * topk),
        )
        _recv_count_kernel[(num_blocks,)](
            received_experts, counts, num_rows,
            received_experts.stride(0), received_experts.stride(1), **common,
        )
        _recv_rank_index_kernel[(num_blocks,)](
            received_experts, counts, tokens_per_local_expert, received_probs,
            wire_to_expert_rows, expert_wire_rows, expert_probs, num_rows,
            received_experts.stride(0), received_experts.stride(1),
            received_probs.stride(0), received_probs.stride(1),
            experts_pow2=triton.next_power_of_2(num_local_experts),
            blocks_pow2=triton.next_power_of_2(num_blocks), **common,
        )
    return wire_to_expert_rows, expert_wire_rows, expert_probs
