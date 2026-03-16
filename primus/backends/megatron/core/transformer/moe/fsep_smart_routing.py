###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""
FSEP Smart Routing: token assignment across expert replicas.

Provides GPU-accelerated (Triton) and Python-fallback implementations of:
  1. smart_routing_map  — per-slot token allocation with intra-node priority (Algorithm 3)
  2. compute_smart_routing — token→slot assignment with capacity-aware allocation
  3. new_routing_map_with_gradients — differentiable routing with Triton backward

Triton kernels ported from:
  laer_moe/galvatron/core/runtime/moe/fused_kernel.py
    _smart_routing_kernel, _token_assignment_kernel, _gradient_mapping_kernel

Reference: LAER-MoE (ASPLOS '26), Algorithms 3 & 4
"""

from typing import Tuple, Optional

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Triton import with graceful fallback
# ---------------------------------------------------------------------------
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Triton Kernels
# ═══════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:

    # -------------------------------------------------------------------
    # Kernel 1: Smart Routing Map (Algorithm 3 — Lite Routing)
    # Reference: fused_kernel.py::_smart_routing_kernel
    # -------------------------------------------------------------------
    @triton.jit
    def _smart_routing_kernel(
        # Input tensors
        tokens_per_expert_ptr,   # [tp_size, ep_size, num_global_experts]
        expert_locations_ptr,    # [origin_expert_num, max_locations]
        # Output tensors
        output_ptr,              # [tp_size, ep_size, ep_size * num_local_experts]
        # Configuration (compile-time constants)
        tp_size: tl.constexpr,
        ep_size: tl.constexpr,
        num_global_experts: tl.constexpr,
        num_local_experts: tl.constexpr,
        max_locations: tl.constexpr,
        gpus_per_node: tl.constexpr,
        # Meta parameters
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton kernel: compute per-slot token allocation with intra-node priority.

        Each (tp_idx, src_ep_rank) program processes all experts, distributing
        tokens to replica slots. Intra-node replicas are preferred to minimize
        cross-node bandwidth.

        Grid: (tp_size, ep_size)
        """
        tp_idx = tl.program_id(0)
        src_ep_rank = tl.program_id(1)

        src_node = src_ep_rank // gpus_per_node

        # Base pointers for this (tp_idx, src_ep_rank)
        tokens_offset = tp_idx * ep_size * num_global_experts + src_ep_rank * num_global_experts
        tokens_base = tokens_per_expert_ptr + tokens_offset
        output_base = (
            output_ptr
            + tp_idx * ep_size * ep_size * num_local_experts
            + src_ep_rank * ep_size * num_local_experts
        )

        # Process each expert
        for expert_id in range(num_global_experts):
            tokens_for_expert = tl.load(tokens_base + expert_id)
            if tokens_for_expert != 0:
                expert_locations_base = expert_locations_ptr + expert_id * max_locations

                # Phase 1: Count intra-node and inter-node replicas
                intra_count = 0
                inter_count = 0
                for loc_idx in range(max_locations):
                    location = tl.load(expert_locations_base + loc_idx)
                    if location >= 0:
                        target_node = location // gpus_per_node // num_local_experts
                        if target_node == src_node:
                            intra_count += 1
                        else:
                            inter_count += 1

                remaining_tokens = tokens_for_expert

                # Phase 2: Distribute to intra-node replicas first
                if intra_count > 0:
                    tokens_per_location = remaining_tokens // intra_count
                    extra_tokens = remaining_tokens % intra_count

                    assigned = 0
                    for loc_idx in range(max_locations):
                        location = tl.load(expert_locations_base + loc_idx)
                        if location >= 0:
                            target_node = location // gpus_per_node // num_local_experts
                            if target_node == src_node:
                                tokens_to_assign = tokens_per_location
                                if assigned < extra_tokens:
                                    tokens_to_assign += 1
                                tl.atomic_add(output_base + location, tokens_to_assign)
                                assigned += 1
                    remaining_tokens = 0

                # Phase 3: Distribute remaining to inter-node replicas
                if remaining_tokens > 0 and inter_count > 0:
                    tokens_per_location = remaining_tokens // inter_count
                    extra_tokens = remaining_tokens % inter_count

                    assigned = 0
                    for loc_idx in range(max_locations):
                        location = tl.load(expert_locations_base + loc_idx)
                        if location >= 0:
                            tokens_to_assign = tokens_per_location
                            if assigned < extra_tokens:
                                tokens_to_assign += 1
                            tl.atomic_add(output_base + location, tokens_to_assign)
                            assigned += 1

    # -------------------------------------------------------------------
    # Kernel 2: Token Assignment (capacity-aware, atomic)
    # Reference: fused_kernel.py::_token_assignment_kernel
    # -------------------------------------------------------------------
    @triton.jit
    def _token_assignment_kernel(
        # Input tensors
        routing_map_ptr,         # [token_num, origin_expert_num]
        probs_ptr,               # [token_num, origin_expert_num]
        expert_locations_ptr,    # [origin_expert_num, max_locations]
        copy_num_ptr,            # [num_global_experts] — mutable capacity (atomic)
        # Output tensors
        new_routing_map_ptr,     # [token_num, num_global_experts]
        new_probs_ptr,           # [token_num, num_global_experts]
        # Dimension parameters (compile-time constants)
        token_num: tl.constexpr,
        origin_expert_num: tl.constexpr,
        num_global_experts: tl.constexpr,
        max_locations: tl.constexpr,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton kernel: assign each token to exactly one replica slot per expert.

        Each program processes one token. For each expert the token is routed to,
        it tries each replica slot in order. Uses atomic_add(-1) on copy_num to
        claim capacity; if old_count > 0, allocation succeeds. Otherwise, the
        counter is restored and the next slot is tried.

        Grid: (token_num,)
        """
        token_id = tl.program_id(0)

        if token_id >= token_num:
            return

        routing_base = routing_map_ptr + token_id * origin_expert_num
        probs_base = probs_ptr + token_id * origin_expert_num
        new_routing_base = new_routing_map_ptr + token_id * num_global_experts
        new_probs_base = new_probs_ptr + token_id * num_global_experts

        for expert_idx in range(origin_expert_num):
            is_routed = tl.load(routing_base + expert_idx)
            if is_routed:
                prob_val = tl.load(probs_base + expert_idx)

                expert_locations_base = expert_locations_ptr + expert_idx * max_locations

                allocated = 0
                for loc_idx in range(max_locations):
                    if allocated == 0:
                        location = tl.load(expert_locations_base + loc_idx)
                        if location >= 0:
                            # Atomic allocation attempt
                            old_count = tl.atomic_add(copy_num_ptr + location, -1)
                            if old_count > 0:
                                # Success: claim this slot
                                tl.store(new_routing_base + location, 1)
                                tl.store(new_probs_base + location, prob_val)
                                allocated = 1
                            else:
                                # Slot full: restore counter
                                tl.atomic_add(copy_num_ptr + location, 1)

    # -------------------------------------------------------------------
    # Kernel 3: Gradient Mapping (backward pass)
    # Reference: fused_kernel.py::_gradient_mapping_kernel
    # -------------------------------------------------------------------
    @triton.jit
    def _gradient_mapping_kernel(
        # Input tensors
        grad_new_probs_ptr,        # [token_num, num_global_experts]
        new_routing_map_ptr,       # [token_num, num_global_experts]
        inverse_expert_map_ptr,    # [num_global_experts]
        # Output tensors
        grad_probs_ptr,            # [token_num, origin_expert_num]
        # Dimensions (compile-time constants)
        token_num: tl.constexpr,
        origin_expert_num: tl.constexpr,
        num_global_experts: tl.constexpr,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Triton kernel: map gradients from new_probs back to original probs.

        For each active (token, slot) pair, looks up the original expert index
        via inverse_expert_map and writes the gradient.

        Grid: (token_num,)
        """
        token_idx = tl.program_id(0)

        if token_idx >= token_num:
            return

        for location in range(num_global_experts):
            map_value = tl.load(
                new_routing_map_ptr + token_idx * num_global_experts + location
            )
            grad_value = tl.load(
                grad_new_probs_ptr + token_idx * num_global_experts + location
            )

            if map_value != 0:
                expert_idx = tl.load(inverse_expert_map_ptr + location)
                tl.store(
                    grad_probs_ptr + token_idx * origin_expert_num + expert_idx,
                    grad_value,
                )


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: GPU wrapper functions (dispatch to Triton or Python fallback)
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# smart_routing_map — public API
# ---------------------------------------------------------------------------

def smart_routing_map(
    num_global_tokens_per_expert: torch.Tensor,
    expert_locations: torch.Tensor,
    num_local_experts: int,
    ep_size: int = 0,
    ep_rank: int = 0,
    gpus_per_node: int = 8,
) -> torch.Tensor:
    """
    Compute balanced per-slot token allocation (Algorithm 3: Lite Routing).

    Dispatches to Triton kernel on CUDA tensors, Python fallback otherwise.

    Args:
        num_global_tokens_per_expert: [tp_size, ep_size, N_E] or [N_E]
        expert_locations: [N_E, max_S], -1 for invalid
        num_local_experts: experts per GPU
        ep_size / ep_rank / gpus_per_node: topology info

    Returns:
        3D input → [tp_size, ep_size, N_slots]
        1D input → [N_slots]
    """
    if num_global_tokens_per_expert.dim() == 3:
        return _smart_routing_map_3d(
            num_global_tokens_per_expert, expert_locations,
            num_local_experts, gpus_per_node,
        )
    else:
        return _smart_routing_map_1d(
            num_global_tokens_per_expert, expert_locations,
            num_local_experts, ep_rank, gpus_per_node,
        )


@torch.no_grad()
def _smart_routing_map_3d(
    tokens_per_expert: torch.Tensor,    # [tp_size, ep_size, N_E]
    expert_locations: torch.Tensor,     # [N_E, max_S]
    num_local_experts: int,
    gpus_per_node: int = 8,
) -> torch.Tensor:
    """3D smart routing: Triton kernel on CUDA, Python fallback on CPU."""
    tp_size, ep_size, N_E = tokens_per_expert.shape
    max_S = expert_locations.shape[1]
    N_slots = ep_size * num_local_experts
    device = tokens_per_expert.device

    # Use Triton kernel if available and tensors are on CUDA
    if _TRITON_AVAILABLE and device.type == "cuda":
        output = torch.zeros(
            (tp_size, ep_size, N_slots),
            dtype=tokens_per_expert.dtype, device=device,
        )
        tokens_tensor = tokens_per_expert.contiguous()
        locations_tensor = expert_locations.contiguous()
        grid = (tp_size, ep_size)
        _smart_routing_kernel[grid](
            tokens_tensor,
            locations_tensor,
            output,
            tp_size=tp_size,
            ep_size=ep_size,
            num_global_experts=N_E,
            num_local_experts=num_local_experts,
            max_locations=max_S,
            gpus_per_node=gpus_per_node,
            BLOCK_SIZE=1,
        )
        return output

    # ── Python fallback ──
    return _smart_routing_map_3d_python(
        tokens_per_expert, expert_locations, num_local_experts, gpus_per_node,
    )


def _smart_routing_map_3d_python(
    tokens_per_expert: torch.Tensor,
    expert_locations: torch.Tensor,
    num_local_experts: int,
    gpus_per_node: int = 8,
) -> torch.Tensor:
    """Pure-Python 3D smart routing (CPU fallback)."""
    tp_size, ep_size, N_E = tokens_per_expert.shape
    max_S = expert_locations.shape[1]
    N_slots = ep_size * num_local_experts
    device = tokens_per_expert.device

    output = torch.zeros(
        (tp_size, ep_size, N_slots), dtype=tokens_per_expert.dtype, device=device,
    )

    for tp_idx in range(tp_size):
        for src_rank in range(ep_size):
            src_node = src_rank // gpus_per_node

            for expert_id in range(N_E):
                tokens = tokens_per_expert[tp_idx, src_rank, expert_id].item()
                if tokens == 0:
                    continue

                intra_slots = []
                inter_slots = []
                for loc_idx in range(max_S):
                    location = expert_locations[expert_id, loc_idx].item()
                    if location < 0:
                        continue
                    target_node = (location // num_local_experts) // gpus_per_node
                    if target_node == src_node:
                        intra_slots.append(location)
                    else:
                        inter_slots.append(location)

                remaining = tokens
                if intra_slots:
                    n = len(intra_slots)
                    per_slot = remaining // n
                    extra = remaining % n
                    for idx, slot in enumerate(intra_slots):
                        output[tp_idx, src_rank, slot] += per_slot + (1 if idx < extra else 0)
                    remaining = 0

                if remaining > 0 and inter_slots:
                    n = len(inter_slots)
                    per_slot = remaining // n
                    extra = remaining % n
                    for idx, slot in enumerate(inter_slots):
                        output[tp_idx, src_rank, slot] += per_slot + (1 if idx < extra else 0)

    return output


def _smart_routing_map_1d(
    tokens_per_expert: torch.Tensor,
    expert_locations: torch.Tensor,
    num_local_experts: int,
    ep_rank: int = 0,
    gpus_per_node: int = 8,
) -> torch.Tensor:
    """
    1D smart routing: allocates tokens across slots for a single source rank.

    Infers N_slots from the max valid slot index in expert_locations,
    then uses the Python fallback (since 1D is typically small-scale / CPU).
    """
    device = tokens_per_expert.device
    N_E = tokens_per_expert.shape[0]
    max_S = expert_locations.shape[1]

    # Infer N_slots from expert_locations content
    valid_mask = expert_locations >= 0
    if valid_mask.any():
        N_slots = int(expert_locations[valid_mask].max().item()) + 1
    else:
        N_slots = N_E

    slot_allocation = torch.zeros(N_slots, dtype=tokens_per_expert.dtype, device=device)
    src_node = ep_rank // gpus_per_node

    for e in range(N_E):
        tokens = tokens_per_expert[e].item()
        if tokens == 0:
            continue

        intra_slots = []
        inter_slots = []
        for j in range(max_S):
            slot = expert_locations[e, j].item()
            if slot < 0:
                continue
            target_node = (slot // num_local_experts) // gpus_per_node
            if target_node == src_node:
                intra_slots.append(slot)
            else:
                inter_slots.append(slot)

        remaining = tokens
        if intra_slots:
            n = len(intra_slots)
            per_slot = remaining // n
            extra = remaining % n
            for idx, slot in enumerate(intra_slots):
                slot_allocation[slot] += per_slot + (1 if idx < extra else 0)
            remaining = 0

        if remaining > 0 and inter_slots:
            n = len(inter_slots)
            per_slot = remaining // n
            extra = remaining % n
            for idx, slot in enumerate(inter_slots):
                slot_allocation[slot] += per_slot + (1 if idx < extra else 0)

    return slot_allocation


# ---------------------------------------------------------------------------
# compute_smart_routing — public API (token-level assignment)
# ---------------------------------------------------------------------------

def compute_smart_routing(
    routing_map: torch.Tensor,            # [T, N_E] bool
    probs: torch.Tensor,                  # [T, N_E] float
    expert_locations: torch.Tensor,       # [N_E, max_S]
    inverse_expert_map: torch.Tensor,     # [N_slots]
    tokens_per_slot: torch.Tensor,        # [N_slots] — capacity from smart_routing_map
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assign tokens to expert replica slots with capacity-aware allocation.

    Dispatches to Triton kernel on CUDA, Python fallback otherwise.

    Args:
        routing_map: [T, N_E] bool — original expert assignments
        probs: [T, N_E] float — routing probabilities
        expert_locations: [N_E, max_S] — replica slot indices (-1 = invalid)
        inverse_expert_map: [N_slots] — slot→expert mapping
        tokens_per_slot: [N_slots] — per-slot capacity budget

    Returns:
        new_routing_map: [T, N_slots] bool
        new_probs: [T, N_slots] float
    """
    T, N_E = routing_map.shape
    N_slots = tokens_per_slot.shape[0]
    max_S = expert_locations.shape[1]
    device = routing_map.device

    if _TRITON_AVAILABLE and device.type == "cuda":
        return _compute_smart_routing_triton(
            routing_map, probs, expert_locations, tokens_per_slot,
            T, N_E, N_slots, max_S,
        )

    return _compute_smart_routing_python(
        routing_map, probs, expert_locations, tokens_per_slot,
        T, N_E, N_slots, max_S,
    )


def _compute_smart_routing_triton(
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    expert_locations: torch.Tensor,
    tokens_per_slot: torch.Tensor,
    T: int, N_E: int, N_slots: int, max_S: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated token→slot assignment."""
    device = routing_map.device

    new_routing_map = torch.zeros((T, N_slots), dtype=routing_map.dtype, device=device)
    new_probs = torch.zeros((T, N_slots), dtype=probs.dtype, device=device)

    # Mutable capacity — cloned so atomic decrements don't corrupt the input
    copy_num = tokens_per_slot.clone().to(dtype=torch.int32, device=device)

    # Ensure int32 routing_map for Triton (bool not supported for arithmetic)
    routing_int = routing_map.to(dtype=torch.int32).contiguous()

    grid = (T,)
    _token_assignment_kernel[grid](
        routing_int,
        probs.contiguous(),
        expert_locations.contiguous(),
        copy_num.contiguous(),
        new_routing_map,
        new_probs,
        token_num=T,
        origin_expert_num=N_E,
        num_global_experts=N_slots,
        max_locations=max_S,
        BLOCK_SIZE=1,
    )
    return new_routing_map, new_probs


def _compute_smart_routing_python(
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    expert_locations: torch.Tensor,
    tokens_per_slot: torch.Tensor,
    T: int, N_E: int, N_slots: int, max_S: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-Python token→slot assignment (CPU fallback)."""
    device = routing_map.device

    new_routing_map = torch.zeros((T, N_slots), dtype=torch.bool, device=device)
    new_probs = torch.zeros((T, N_slots), dtype=probs.dtype, device=device)

    cap = tokens_per_slot.clone().long().to(device)

    for e in range(N_E):
        token_mask = routing_map[:, e]
        if not token_mask.any():
            continue

        token_ids = token_mask.nonzero(as_tuple=True)[0]
        token_probs = probs[token_ids, e]

        valid_mask = expert_locations[e] >= 0
        if not valid_mask.any():
            continue
        valid_slot_indices = expert_locations[e][valid_mask]

        slot_ptr = 0
        n_slots = len(valid_slot_indices)

        for i in range(len(token_ids)):
            if slot_ptr >= n_slots:
                break
            tid = token_ids[i].item()
            while slot_ptr < n_slots:
                slot = valid_slot_indices[slot_ptr].item()
                if cap[slot].item() > 0:
                    cap[slot] -= 1
                    new_routing_map[tid, slot] = True
                    new_probs[tid, slot] = token_probs[i]
                    break
                slot_ptr += 1

    return new_routing_map, new_probs


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: Differentiable routing (autograd.Function with Triton backward)
# ═══════════════════════════════════════════════════════════════════════════

class _NewRoutingMapFunction(torch.autograd.Function):
    """
    Differentiable token→slot routing with Triton-accelerated backward.

    Forward: uses _token_assignment_kernel (Triton) for capacity-aware assignment
    Backward: uses _gradient_mapping_kernel (Triton) for fast gradient routing

    Reference: fused_kernel.py::NewRoutingMapWithGradients
    """

    @staticmethod
    def forward(
        ctx,
        routing_map: torch.Tensor,          # [T, N_E] bool
        probs: torch.Tensor,                # [T, N_E] float
        expert_locations: torch.Tensor,      # [N_E, max_S] long
        inverse_expert_map: torch.Tensor,    # [N_slots] long
        slot_capacity: torch.Tensor,         # [N_slots] long
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T, N_E = routing_map.shape
        N_slots = slot_capacity.shape[0]
        max_S = expert_locations.shape[1]
        device = routing_map.device

        new_routing_map = torch.zeros((T, N_slots), dtype=routing_map.dtype, device=device)
        new_probs = torch.zeros((T, N_slots), dtype=probs.dtype, device=device)

        if _TRITON_AVAILABLE and device.type == "cuda":
            copy_num = slot_capacity.clone().to(dtype=torch.int32, device=device)
            routing_int = routing_map.to(dtype=torch.int32).contiguous()

            grid = (T,)
            _token_assignment_kernel[grid](
                routing_int,
                probs.contiguous(),
                expert_locations.contiguous(),
                copy_num.contiguous(),
                new_routing_map,
                new_probs,
                token_num=T,
                origin_expert_num=N_E,
                num_global_experts=N_slots,
                max_locations=max_S,
                BLOCK_SIZE=1,
            )
        else:
            # Python fallback
            cap = slot_capacity.clone().long()
            for t in range(T):
                for e in range(N_E):
                    if not routing_map[t, e]:
                        continue
                    prob_val = probs[t, e]
                    for j in range(max_S):
                        slot = expert_locations[e, j].item()
                        if slot < 0:
                            continue
                        if cap[slot].item() > 0:
                            cap[slot] -= 1
                            new_routing_map[t, slot] = 1
                            new_probs[t, slot] = prob_val
                            break

        ctx.save_for_backward(new_routing_map, inverse_expert_map)
        ctx.T = T
        ctx.N_E = N_E

        return new_routing_map, new_probs

    @staticmethod
    def backward(ctx, grad_new_routing_map, grad_new_probs):
        """Map gradients from new_probs → probs via inverse_expert_map."""
        new_routing_map, inverse_expert_map = ctx.saved_tensors
        T, N_E = ctx.T, ctx.N_E

        if grad_new_probs is None:
            return None, None, None, None, None

        N_slots = grad_new_probs.shape[1]
        device = grad_new_probs.device

        grad_probs = torch.zeros(
            (T, N_E), dtype=grad_new_probs.dtype, device=device,
        )

        if _TRITON_AVAILABLE and device.type == "cuda":
            # Triton-accelerated gradient mapping
            # Convert new_routing_map to int32 for Triton
            routing_int = new_routing_map.to(dtype=torch.int32).contiguous()

            grid = (T,)
            _gradient_mapping_kernel[grid](
                grad_new_probs.contiguous(),
                routing_int,
                inverse_expert_map.contiguous(),
                grad_probs,
                token_num=T,
                origin_expert_num=N_E,
                num_global_experts=N_slots,
                BLOCK_SIZE=1,
            )
        else:
            # Vectorized Python fallback
            active_mask = new_routing_map.bool()
            if active_mask.any():
                t_indices, slot_indices = active_mask.nonzero(as_tuple=True)
                expert_indices = inverse_expert_map[slot_indices]
                grad_values = grad_new_probs[t_indices, slot_indices]
                grad_probs.index_put_(
                    (t_indices, expert_indices), grad_values, accumulate=True,
                )

        return None, grad_probs, None, None, None


def new_routing_map_with_gradients(
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    expert_locations: torch.Tensor,
    inverse_expert_map: torch.Tensor,
    slot_capacity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute new token routing map for FSEP dispatch (differentiable).

    Uses Triton kernels for both forward (token assignment) and
    backward (gradient mapping) when available.

    Args:
        routing_map: [T, N_E] bool
        probs: [T, N_E] float
        expert_locations: [N_E, max_S] long
        inverse_expert_map: [N_slots] long
        slot_capacity: [N_slots] long

    Returns:
        new_routing_map: [T, N_slots]
        new_probs: [T, N_slots]
    """
    return _NewRoutingMapFunction.apply(
        routing_map, probs, expert_locations, inverse_expert_map, slot_capacity,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Fast vectorized path (no capacity tracking)
# ═══════════════════════════════════════════════════════════════════════════

def compute_smart_routing_fast(
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    expert_locations: torch.Tensor,
    inverse_expert_map: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast scatter-based routing without capacity tracking.
    Assigns each token to the FIRST valid replica of its target expert.

    Useful for static FSEP (uniform sharding, no dynamic rebalancing).
    No Python loops — pure gather/scatter.
    """
    T, N_E = routing_map.shape
    N_slots = inverse_expert_map.shape[0]
    device = routing_map.device

    new_routing_map = torch.zeros((T, N_slots), dtype=torch.bool, device=device)
    new_probs = torch.zeros((T, N_slots), dtype=probs.dtype, device=device)

    primary_slot = expert_locations[:, 0]
    valid_experts = (primary_slot >= 0)

    if valid_experts.any():
        valid_e_idx = valid_experts.nonzero(as_tuple=True)[0]
        valid_slots = primary_slot[valid_e_idx]

        sub_map = routing_map[:, valid_e_idx]
        sub_probs = probs[:, valid_e_idx]

        slots_expanded = valid_slots.unsqueeze(0).expand(T, -1)

        new_routing_map.scatter_(1, slots_expanded, sub_map)
        new_probs.scatter_(1, slots_expanded, sub_probs * sub_map.float())

    return new_routing_map, new_probs
