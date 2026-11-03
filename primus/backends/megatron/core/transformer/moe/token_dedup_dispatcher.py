###############################################################################
# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""All-to-all MoE token dispatchers with custom expert placement and token dedup.

Ported from ROCm/Megatron-LM PR #165 (``--moe-token-dedup``) and PR #166
(``--moe-expert-placement``) onto Megatron-core v0.19's
``MoEAlltoAllTokenDispatcher`` as subclasses, so ``third_party/Megatron-LM``
stays untouched:

* :class:`PrimusAlltoAllTokenDispatcher` -- the stock dispatcher plus a custom
  expert placement. The routing map / probs columns are permuted into rank-major
  "slot space" at dispatch time, so the upstream contiguous split logic sends
  each token to the EP rank that owns its expert. The base class keeps working
  in slot space (its ``local_expert_indices`` are the contiguous slot block);
  the true global expert ids owned by this rank live in
  ``placed_expert_indices``.
* :class:`PrimusDedupAlltoAllTokenDispatcher` -- on top of that, one wire row per
  unique (token, destination EP rank) instead of one per (token, expert), in
  both the dispatch and the combine direction.

The ``megatron.moe.token_dedup`` patch binds these in place of
``MoEAlltoAllTokenDispatcher`` when ``moe_token_dedup`` or
``moe_expert_placement`` is set; otherwise the upstream dispatcher is untouched.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
from megatron.core import utils
from megatron.core.tensor_parallel import all_to_all
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig

from primus.backends.megatron.core.transformer.moe.token_dedup import (
    build_wire_recv_index,
    build_wire_send_index,
    dedup_payload_width,
    expand_token_dedup_received,
    fused_index_build_available,
    pack_token_dedup_payload,
    permute_token_dedup_expert_output,
    reduce_token_dedup_expert_output,
    restore_token_dedup_output,
    routing_counts,
    unpack_token_dedup_payload,
    weighted_reduce_token_dedup_expert_output,
)


def _compact_mask_indices(mask: torch.Tensor, count: int) -> torch.Tensor:
    """Return set-bit indices without ``nonzero`` or sorting.

    ``count`` is already known by the caller from collective metadata, so this
    compaction remains device-side and does not need to discover a dynamic
    output shape.
    """
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=mask.device)
    prefix = mask.reshape(-1).cumsum(0, dtype=torch.long)
    ordinal = torch.arange(1, count + 1, dtype=torch.long, device=mask.device)
    return torch.searchsorted(prefix, ordinal)


def _token_destination_hit_map(
    indices: torch.Tensor, num_local_experts: int, ep_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the sort-free token-to-destination map.

    Returns destination ranks with shape ``[tokens, topk]`` and a destination
    major boolean map with shape ``[ep_size, tokens]``. Multiple experts for one
    token on the same destination set only one bit in the latter.
    """
    destinations = torch.div(indices, num_local_experts, rounding_mode="floor")
    ep_ranks = torch.arange(ep_size, device=indices.device).view(ep_size, 1, 1)
    hits = (destinations.unsqueeze(0) == ep_ranks).any(dim=-1)
    return destinations, hits


def dedup_weighted_combine_enabled(config) -> bool:
    """Whether the dedup combine applies the routing weights itself.

    Mirrors the PR's condition: experts-side weighting stays for
    ``moe_apply_probs_on_input``, the TE op fuser, and fused weighted activations.
    """
    return bool(
        getattr(config, "moe_token_dedup_weighted_combine", False)
        and not getattr(config, "moe_apply_probs_on_input", False)
        and not getattr(config, "use_transformer_engine_op_fuser", False)
        and not getattr(config, "use_fused_weighted_squared_relu", False)
        and (
            getattr(config, "use_te_activation_func", False)
            or not getattr(config, "bias_activation_fusion", False)
        )
    )


class PrimusAlltoAllTokenDispatcher(MoEAlltoAllTokenDispatcher):
    """Upstream all-to-all dispatcher plus a custom expert placement.

    ``local_expert_indices`` passed in are the TRUE global expert ids owned by this
    rank (possibly non-contiguous). The base class is initialised with the
    contiguous slot block instead, because everything it does with
    ``local_expert_indices`` (the ``preprocess`` count slice) is in slot space
    after the column permutation applied in :meth:`dispatch_preprocess`.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        expert_order: Optional[List[int]] = None,
    ) -> None:
        ep_rank = utils.get_pg_rank(pg_collection.ep)
        slot_indices = [ep_rank * num_local_experts + i for i in range(num_local_experts)]
        super().__init__(
            num_local_experts=num_local_experts,
            local_expert_indices=slot_indices,
            config=config,
            pg_collection=pg_collection,
        )
        self.set_expert_order(
            expert_order if expert_order is not None else list(range(self.num_experts)),
            local_expert_indices,
        )

    def set_expert_order(self, expert_order: List[int], local_expert_indices: List[int]) -> None:
        """(Re)configure the custom expert placement.

        Called at construction and again from ``MoELayer.set_layer_number()`` once
        the (per-layer) placement is known. Placement only permutes which global
        expert maps to each local slot; all shapes and ``num_local_experts`` are
        unchanged, so this is a metadata-only update.

        Args:
            expert_order: global expert ids in rank-major slot order.
            local_expert_indices: global expert ids owned by this rank.
        """
        assert (
            len(expert_order) == self.num_experts
        ), f"expert_order has {len(expert_order)} entries, expected {self.num_experts}."
        assert len(local_expert_indices) == self.num_local_experts, "Invalid local expert indices"
        self.expert_order = list(expert_order)
        self.placed_expert_indices = list(local_expert_indices)
        # In slot space this rank always owns the contiguous block
        # [ep_rank * num_local_experts, (ep_rank + 1) * num_local_experts).
        self.local_slot_start = self.ep_rank * self.num_local_experts
        self.local_slot_end = self.local_slot_start + self.num_local_experts
        assert self.local_expert_indices == list(range(self.local_slot_start, self.local_slot_end))
        # Default contiguous placement -> no column permutation (fast path). Otherwise the
        # gather index tensor is created lazily on the routing map's device at first use.
        self.expert_order_is_identity = self.expert_order == list(range(self.num_experts))
        self._expert_order_index_cache = None

    def _get_expert_order_index(self, device):
        """Lazily build/cache the column-permutation index tensor on ``device``."""
        if self.expert_order_is_identity:
            return None
        cache = self._expert_order_index_cache
        if cache is None or cache.device != device:
            self._expert_order_index_cache = torch.tensor(self.expert_order, dtype=torch.long, device=device)
        return self._expert_order_index_cache

    def _to_slot_space(self, routing_map: torch.Tensor, probs: torch.Tensor):
        """Permute the expert (column) axis into rank-major slot order."""
        expert_order_index = self._get_expert_order_index(routing_map.device)
        if expert_order_index is None:
            return routing_map, probs
        return (
            routing_map.index_select(1, expert_order_index),
            probs.index_select(1, expert_order_index),
        )

    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        routing_map, probs = self._to_slot_space(routing_map, probs)
        return super().dispatch_preprocess(hidden_states, routing_map, probs)


class PrimusDedupAlltoAllTokenDispatcher(PrimusAlltoAllTokenDispatcher):
    """All-to-all dispatcher with one wire row per token and destination rank.

    The regular alltoall dispatcher sends one hidden row for every token/expert
    assignment. This opt-in implementation (``moe_token_dedup``) sends a hidden
    row only once when several selected experts are on the same EP rank. A small
    expert-id record accompanies each wire row and the receiver expands directly
    into expert-major order. The combine direction is deduplicated too: local
    expert outputs for the same wire row are summed to one row before the
    combine all-to-all.

    Combined with a co-activation expert placement (``moe_expert_placement``),
    which co-locates experts that tend to fire together, the number of wire rows
    per token drops further because more of a token's experts share a rank.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        expert_order: Optional[List[int]] = None,
    ) -> None:
        super().__init__(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            pg_collection=pg_collection,
            expert_order=expert_order,
        )
        if self.ep_size <= 1:
            raise ValueError("Token dedup requires expert parallel size greater than one.")
        if self.tp_size > 1:
            raise ValueError(
                "Token dedup does not support expert tensor parallelism; set "
                "expert_tensor_parallel_size: 1."
            )
        if self.drop_and_pad or self.config.moe_expert_capacity_factor is not None:
            raise ValueError(
                "Token dedup currently supports dropless routing only; disable expert capacity "
                "and expert-input padding."
            )
        if getattr(self.config, "moe_router_padding_for_quantization", False):
            raise ValueError(
                "Token dedup requires exactly top-k routing rows; disable "
                "moe_router_padding_for_quantization."
            )
        if getattr(self.config, "cuda_graph_impl", "none") != "none":
            raise ValueError("Token dedup does not support CUDA graphs.")
        if self.num_local_experts > 256:
            raise ValueError(
                "Token dedup supports at most 256 local experts so expert IDs remain exact "
                "when carried in a BF16 activation row."
            )
        self.weighted_combine = dedup_weighted_combine_enabled(self.config)
        # Per-peer wire-row counts of the deduplicated all-to-all (host lists).
        self.dedup_input_splits = None
        self.dedup_output_splits = None
        self._dedup_meta_work = None
        self._dedup_hits = None
        self._dedup_destinations = None
        self._dedup_token_wire_rows = None
        self._dedup_wire_tokens = None
        self._dedup_wire_to_expert_rows = None
        self._dedup_expert_wire_rows = None
        self._dedup_wire_probs = None
        self._dedup_num_tokens = 0
        self._dedup_topk = 0

    def _metadata_gather_launch(self, packed: torch.Tensor) -> torch.Tensor:
        """Launch the one packed metadata all-gather; do NOT touch the host.

        All host values needed by the all-to-all-v interface ride one collective,
        and the D2H is deferred to :meth:`_metadata_finish` so the local index
        build and payload pack can overlap the gather's in-flight time.
        """
        gathered_all = packed.new_empty(self.ep_size * packed.numel())
        # async_op=True so the kernels launched "during" the gather do not queue
        # behind the collective and the peer-skew wait.
        self._dedup_meta_work = torch.distributed.all_gather_into_tensor(
            gathered_all, packed, group=self.ep_group, async_op=True
        )
        return gathered_all.view(self.ep_size, packed.numel())

    def _metadata_finish(self, gathered_all: torch.Tensor) -> torch.Tensor:
        """Wait for the gather, copy metadata to the host, publish split lists."""
        work = self._dedup_meta_work
        self._dedup_meta_work = None
        if work is not None:
            work.wait()
        counts_width = self.num_experts + self.ep_size
        gathered = gathered_all[:, :counts_width]
        gathered_cpu = gathered.to(torch.device("cpu"), non_blocking=False)
        all_expert_counts_cpu = gathered_cpu[:, : self.num_experts]
        local_expert_counts_cpu = all_expert_counts_cpu[
            :, self.local_slot_start : self.local_slot_end
        ].contiguous()
        self.input_splits = (
            all_expert_counts_cpu[self.ep_rank].view(self.ep_size, self.num_local_experts).sum(dim=1).tolist()
        )
        self.output_splits = local_expert_counts_cpu.sum(dim=1).tolist()
        self.dedup_input_splits = gathered_cpu[self.ep_rank, self.num_experts : counts_width].tolist()
        self.dedup_output_splits = gathered_cpu[:, self.num_experts + self.ep_rank].tolist()
        self.num_global_tokens_per_local_expert_cpu = local_expert_counts_cpu
        self.cuda_sync_point = "no_sync"
        self.tokens_per_expert_gpu = gathered[:, self.local_slot_start : self.local_slot_end].sum(dim=0)
        return local_expert_counts_cpu.sum(dim=0)

    def _build_send_buffers(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Pack destination-major activations and route metadata without sorting.

        Torch fallback for devices without the fused Triton index build.
        """
        num_tokens, topk = indices.shape
        hidden_size = hidden_states.shape[-1]
        # A wire row is wider than the activation because it carries expert IDs,
        # routing probabilities, and alignment.
        self.dedup_row_elems = dedup_payload_width(hidden_size, topk, hidden_states.element_size())
        if num_tokens == 0:
            self._dedup_num_tokens = 0
            self._dedup_topk = topk
            self._dedup_token_wire_rows = torch.empty((0, topk), dtype=torch.long, device=indices.device)
            self._dedup_wire_tokens = torch.empty(0, dtype=torch.long, device=indices.device)
            return hidden_states.new_empty((0, self.dedup_row_elems))
        num_wire_rows = sum(self.dedup_input_splits)
        entries = _compact_mask_indices(self._dedup_hits, num_wire_rows)
        wire_destinations = torch.div(entries, num_tokens, rounding_mode="floor")
        wire_tokens = entries.remainder(num_tokens)
        token_destinations = self._dedup_destinations.index_select(0, wire_tokens)
        token_local_experts = indices.remainder(self.num_local_experts).index_select(0, wire_tokens)
        wire_experts = torch.where(
            token_destinations == wire_destinations.unsqueeze(1),
            token_local_experts,
            token_local_experts.new_full((), -1),
        ).to(dtype=torch.long)
        # Retain the exact token/top-k slot for every expert assignment. These
        # maps drive probability gradients and the final destination-partial
        # reduction without sorting.
        active = wire_experts >= 0
        active_entries = _compact_mask_indices(active, sum(self.input_splits))
        wire_token_slots = (
            wire_tokens.unsqueeze(1) * topk + torch.arange(topk, device=indices.device).unsqueeze(0)
        ).reshape(-1)
        return_slots = wire_token_slots.index_select(0, active_entries)
        wire_row_slots = (
            torch.arange(num_wire_rows, device=indices.device).unsqueeze(1).expand(-1, topk).reshape(-1)
        )
        slot_wire_rows = torch.full((num_tokens * topk,), -1, dtype=torch.long, device=indices.device)
        slot_wire_rows.index_copy_(0, return_slots, wire_row_slots.index_select(0, active_entries))
        slot_wire_rows = slot_wire_rows.view(num_tokens, topk)
        # One wire-row lookup per unique token/destination lets both output
        # restoration and payload backward gather bounded rows without atomics.
        representative_slots = active.to(dtype=torch.int8).argmax(dim=1)
        token_wire_rows = torch.full((num_tokens * topk,), -1, dtype=torch.long, device=indices.device)
        token_wire_rows.index_copy_(
            0,
            wire_tokens * topk + representative_slots,
            torch.arange(num_wire_rows, device=indices.device),
        )
        token_wire_rows = token_wire_rows.view(num_tokens, topk)
        self._dedup_num_tokens = num_tokens
        self._dedup_topk = topk
        self._dedup_token_wire_rows = token_wire_rows
        self._dedup_wire_tokens = wire_tokens
        return pack_token_dedup_payload(
            hidden_states, probs, wire_tokens, wire_experts, token_wire_rows, slot_wire_rows
        )

    def _expand_received(self, received: torch.Tensor, received_experts: torch.Tensor) -> torch.Tensor:
        """Fan all received wire rows into expert-major order.

        Also records the wire-row -> expert-row map that the deduplicated combine
        uses to collapse the local expert outputs back to one row per wire row.
        """
        num_assignments = sum(self.output_splits)
        if num_assignments == 0:
            self._dedup_wire_to_expert_rows = torch.full_like(received_experts, -1)
            self._dedup_expert_wire_rows = received_experts.new_empty(0)
            return received[:0]
        if fused_index_build_available(received.device):
            wire_to_expert_rows, expert_wire_rows, _ = build_wire_recv_index(
                received_experts,
                self.tokens_per_expert_gpu,
                num_assignments,
                received_experts.new_zeros(received_experts.shape, dtype=received.dtype),
            )
            if num_assignments == received.shape[0]:
                # No destination collisions: a permutation, gather-only backward.
                inverse = wire_to_expert_rows.amax(dim=1)
                expanded = permute_token_dedup_expert_output(received, expert_wire_rows, inverse)
            else:
                expanded = expand_token_dedup_received(received, expert_wire_rows, wire_to_expert_rows)
            self._dedup_wire_to_expert_rows = wire_to_expert_rows
            self._dedup_expert_wire_rows = expert_wire_rows
            return expanded
        flat_experts = received_experts.reshape(-1)
        expert_ids = torch.arange(self.num_local_experts, device=received.device).unsqueeze(1)
        expert_membership = expert_ids == flat_experts.unsqueeze(0)
        # Compact the expert-major membership matrix. Each selected entry gives
        # the wire row to duplicate into that expert's contiguous input region.
        membership_entries = _compact_mask_indices(expert_membership, num_assignments)
        flat_wire_slots = membership_entries.remainder(flat_experts.numel())
        wire_rows = torch.div(flat_wire_slots, received_experts.shape[1], rounding_mode="floor")
        # Expert outputs later need to collapse to wire-major rows before the
        # deduplicated combine. Construct the inverse mapping with a scatter,
        # not an argsort.
        wire_rank = (flat_experts >= 0).cumsum(0, dtype=torch.long) - 1
        expert_row_to_wire_rank = wire_rank.index_select(0, flat_wire_slots)
        wire_to_expert_row = torch.empty_like(expert_row_to_wire_rank)
        wire_to_expert_row.scatter_(
            0, expert_row_to_wire_rank, torch.arange(num_assignments, device=received.device)
        )
        wire_to_expert_rows = torch.full_like(received_experts, -1)
        wire_to_expert_rows.view(-1).index_copy_(
            0, flat_wire_slots, torch.arange(num_assignments, device=received.device)
        )
        if num_assignments == received.shape[0]:
            # With no destination collisions every wire row has exactly one
            # expert row, so this is a permutation rather than a fan-out.
            expanded = permute_token_dedup_expert_output(received, wire_rows, wire_to_expert_row)
        else:
            expanded = expand_token_dedup_received(received, wire_rows, wire_to_expert_rows)
        self._dedup_wire_to_expert_rows = wire_to_expert_rows
        self._dedup_expert_wire_rows = wire_rows
        return expanded

    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """Build one wire row per unique (token, destination EP rank) pair.

        The wire row carries the activation plus per-slot local expert ids and
        routing probabilities, so no separate metadata or probability all-to-all
        is needed.
        """
        self.hidden_shape = hidden_states.shape
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        # Custom expert placement: the dedup hit map is built in slot space, so
        # placements that co-locate co-activated experts on one rank directly
        # increase dedup opportunities.
        routing_map, probs = self._to_slot_space(routing_map, probs)
        self.probs = probs
        self.routing_map = routing_map
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        topk = self.config.moe_router_topk
        num_tokens = routing_map.shape[0]
        hidden_size = self.hidden_shape[-1]
        use_fused = fused_index_build_available(routing_map.device)

        if use_fused:
            # One fused pass: expert counts | destination counts (the all-gather payload),
            # the [tokens, ep_size] destination hit map, and its per-block partial sums.
            packed, hits, hit_partial = routing_counts(routing_map, self.num_local_experts, self.ep_size)
            dedup_send_counts = packed[self.num_experts :]
            hit_block_counts, hit_block = hit_partial[:, self.num_experts :], 256
        else:
            # [num_experts] in slot space; equivalent to bincount but sync-free.
            expert_counts = routing_map.sum(dim=0, dtype=torch.long)
            # [tokens, ep_size]: does the token route to that destination rank?
            hits = routing_map.reshape(num_tokens, self.ep_size, self.num_local_experts).any(dim=-1)
            dedup_send_counts = hits.sum(dim=0, dtype=torch.long)
            packed = torch.cat((expert_counts, dedup_send_counts))
        gathered = self._metadata_gather_launch(packed)

        if use_fused:
            # Local-only GPU work sized by an upper bound (a token has at most
            # min(topk, ep_size) wire rows), so it overlaps the metadata gather
            # still in flight above. Only the a2a itself needs host split values.
            max_wire_rows = num_tokens * min(topk, self.ep_size)
            (indices, slot_wire_rows, token_wire_rows, wire_tokens_full, wire_experts_full) = (
                build_wire_send_index(
                    routing_map,
                    hits,
                    dedup_send_counts,
                    self.num_local_experts,
                    self.ep_size,
                    topk,
                    max_wire_rows,
                    hit_block_counts=hit_block_counts,
                    hit_block=hit_block,
                )
            )
            # Differentiable slot-probability gather: routing-probability
            # gradients flow back to the router through this op.
            slot_probs = probs.gather(1, indices)
        else:
            # Torch fallback (CPU / no triton): derive fixed-width top-k indices
            # from the dense map; slot order within a row is irrelevant as long
            # as the probs use the same order.
            indices = torch.topk(routing_map.int(), k=topk, dim=1).indices
            slot_probs = probs.gather(1, indices)
            self._dedup_destinations = torch.div(indices, self.num_local_experts, rounding_mode="floor")
            self._dedup_destinations.masked_fill_(~routing_map.gather(1, indices), -1)
            self._dedup_hits = hits.t()

        if self.shared_experts is not None:
            self.shared_experts.pre_forward_comm(hidden_states.view(self.hidden_shape))

        self.tokens_per_expert = self._metadata_finish(gathered)

        if use_fused:
            num_wire_rows = sum(self.dedup_input_splits)
            self.dedup_row_elems = dedup_payload_width(hidden_size, topk, hidden_states.element_size())
            self._dedup_num_tokens = num_tokens
            self._dedup_topk = topk
            self._dedup_token_wire_rows = token_wire_rows
            self._dedup_wire_tokens = wire_tokens_full.narrow(0, 0, num_wire_rows)
            if num_tokens == 0:
                payload = hidden_states.new_empty((0, self.dedup_row_elems))
            else:
                payload = pack_token_dedup_payload(
                    hidden_states,
                    slot_probs,
                    self._dedup_wire_tokens,
                    wire_experts_full.narrow(0, 0, num_wire_rows),
                    token_wire_rows,
                    slot_wire_rows,
                )
        else:
            payload = self._build_send_buffers(hidden_states, slot_probs, indices)
        # Probabilities travel inside the payload; there is no separate probs tensor to dispatch.
        return payload, None

    def token_dispatch(self, permutated_local_input_tokens, permuted_probs=None):
        """Single deduplicated all-to-all: one wire row per (token, destination rank)."""
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()
        received_payload = all_to_all(
            self.ep_group,
            permutated_local_input_tokens,
            self.dedup_output_splits,
            self.dedup_input_splits,
            use_nccl_stream=self.use_nccl_stream,
        )
        return received_payload, None

    def dispatch_postprocess(self, global_input_tokens, global_probs=None):
        """Unpack the received payload and fan wire rows into expert-major order."""
        hidden_size = self.hidden_shape[-1]
        received = global_input_tokens[:, :hidden_size]
        received_expert_cols = global_input_tokens[:, hidden_size : hidden_size + self._dedup_topk]
        received_probs = global_input_tokens[
            :, hidden_size + self._dedup_topk : hidden_size + 2 * self._dedup_topk
        ]
        num_assignments = sum(self.output_splits)
        if fused_index_build_available(received.device) and num_assignments > 0:
            # One kernel builds the expert-major indices and gathers each expert row's
            # probability; one autograd node expands the payload and owns the whole
            # payload gradient (no per-slice zero-fill + add in backward).
            wire_to_expert_rows, expert_wire_rows, expert_probs = build_wire_recv_index(
                received_expert_cols.detach(),
                self.tokens_per_expert_gpu,
                num_assignments,
                received_probs.detach(),
            )
            self._dedup_wire_to_expert_rows = wire_to_expert_rows
            self._dedup_expert_wire_rows = expert_wire_rows
            # Weighted combine: the routing weights are applied in the combine's
            # reduce kernel and the experts get no probabilities.
            weighted = self.weighted_combine
            expanded, permuted_probs = unpack_token_dedup_payload(
                global_input_tokens,
                expert_wire_rows,
                wire_to_expert_rows,
                expert_probs,
                hidden_size,
                self._dedup_topk,
                weighted,
            )
            if weighted:
                self._dedup_wire_probs = permuted_probs
                permuted_probs = None
            else:
                self._dedup_wire_probs = None
                permuted_probs = permuted_probs.to(self.probs.dtype)
            if self.shared_experts is not None:
                self.shared_experts.linear_fc1_forward_and_act(expanded)
        else:
            self._dedup_wire_probs = None
            received_experts = received_expert_cols.to(dtype=torch.long)
            expanded = self._expand_received(received, received_experts)
            if self.shared_experts is not None:
                self.shared_experts.linear_fc1_forward_and_act(expanded)
            # Weight inside the experts as the stock dispatcher does; the combine then only sums.
            wire_rows = self._dedup_expert_wire_rows
            rows = torch.arange(wire_rows.numel(), device=wire_rows.device).unsqueeze(1)
            wire_slots = self._dedup_wire_to_expert_rows.index_select(0, wire_rows) == rows
            flat_slots = wire_rows * self._dedup_topk + wire_slots.to(torch.int8).argmax(dim=1)
            permuted_probs = received_probs.reshape(-1)[flat_slots].to(self.probs.dtype)
        tokens_per_expert = self.tokens_per_expert
        self.tokens_per_expert = None
        if os.environ.get("NVTE_USE_GROUPED_GEMM_TRITON") == "1":
            tokens_per_expert_gpu = self.tokens_per_expert_gpu.to(torch.int32)
            self.tokens_per_expert_gpu = None
            return expanded, (tokens_per_expert, tokens_per_expert_gpu), permuted_probs
        return expanded, tokens_per_expert, permuted_probs

    def combine_preprocess(self, hidden_states):
        """Sum the local expert outputs to one row per wire row (weighted if the
        experts skipped the routing-weight multiply)."""
        if self._dedup_wire_probs is not None:
            return weighted_reduce_token_dedup_expert_output(
                hidden_states, self._dedup_wire_to_expert_rows, self._dedup_wire_probs
            )
        return reduce_token_dedup_expert_output(
            hidden_states, self._dedup_expert_wire_rows, self._dedup_wire_to_expert_rows
        )

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """Deduplicated combine all-to-all: one row per wire row."""
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()
        return all_to_all(
            self.ep_group,
            hidden_states,
            self.dedup_input_splits,
            self.dedup_output_splits,
            use_nccl_stream=self.use_nccl_stream,
        )

    def combine_postprocess(self, permutated_local_input_tokens):
        """Scatter returned wire rows back to token order and restore the input shape."""
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(permutated_local_input_tokens)
            self.shared_experts.post_forward_comm()
        output = restore_token_dedup_output(
            permutated_local_input_tokens, self._dedup_token_wire_rows, self._dedup_wire_tokens
        )
        output = output.view(self.hidden_shape)
        # Autograd functions saved the tensors needed by backward. Drop the
        # dispatcher's extra references now so route maps and the received
        # payload view do not stay live beyond this forward.
        self._dedup_token_wire_rows = None
        self._dedup_wire_tokens = None
        self._dedup_wire_to_expert_rows = None
        self._dedup_expert_wire_rows = None
        self._dedup_wire_probs = None
        self._dedup_destinations = None
        self._dedup_hits = None
        if self.shared_experts is not None:
            output += self.shared_experts.get_output()
        return output


def build_alltoall_token_dispatcher(
    num_local_experts: int,
    local_expert_indices: List[int],
    config: TransformerConfig = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    **kwargs,
):
    """Drop-in replacement for the ``MoEAlltoAllTokenDispatcher`` constructor call in
    ``MoELayer.__init__``: the dedup dispatcher when ``moe_token_dedup`` is on
    (and EP > 1, where there is something to dedup), else the placement-capable
    stock dispatcher."""
    use_dedup = bool(getattr(config, "moe_token_dedup", False)) and utils.get_pg_size(pg_collection.ep) > 1
    cls = PrimusDedupAlltoAllTokenDispatcher if use_dedup else PrimusAlltoAllTokenDispatcher
    return cls(num_local_experts, local_expert_indices, config=config, pg_collection=pg_collection, **kwargs)
