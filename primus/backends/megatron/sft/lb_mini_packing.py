###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""LB-Mini sequence-length load balancing for Megatron-native SFT (ODC path).

This is the Megatron-side port of the load-balancing core that ODC's official
example (``odc_rocm_dev/examples/llm_training/packing.py``) uses. The algorithm
itself is unchanged (it is already validated on the ODC example); we only strip
the example's ``from args import get_args`` dependency so it can be driven by
plain function arguments from a Primus monkey-patch.

What it does
------------
Given the *effective* sequence lengths of one global minibatch (``minibatch_size
* dp`` samples), it produces a balanced assignment so that:

  1. ``get_seqlen_balanced_partitions`` (Karmarkar-Karp differencing) splits the
     samples across the ``dp`` ranks so each rank's TOTAL token workload is as
     equal as possible -- this is the "LB" (load balance) part.

  2. Within a rank, ``rearrange_micro_batches`` re-packs that rank's samples into
     micro-batches each capped at ``max_token_len``. With ``same_num_in_dp=False``
     (the LB-Mini / ODC mode) each rank ends up with a DIFFERENT number of
     micro-batches -- the short-workload ranks get fewer, the long ones get more,
     and crucially NO rank pads up to the global max. This is the "Mini"
     (variable micro-batch count) part that only ODC's point-to-point comm can
     drive without a collective deadlock.

The baseline (``same_num_in_dp=True``) instead all-reduce(MAX)es the micro-batch
count so every rank runs the same number of steps -- exactly Megatron's current
behaviour, kept here so the two paths share one code base.

IMPORTANT: this module is pure CPU index math; it never touches CUDA or the
process group except for the optional ``all_reduce`` used by the baseline
``same_num_in_dp=True`` path.
"""

import heapq
import os
from typing import Callable, List, Optional

import torch
from torch import distributed as dist


# ---------------------------------------------------------------------------
# Cost models: map a sequence length to a scalar "workload".
# ---------------------------------------------------------------------------
def get_seq_costs_linear(seq_len: List[int]) -> List[float]:
    """Linear cost == token count. Good default for MLP-bound SFT."""
    return list(seq_len)


# Default fit coefficients (a, b) + normalizer = ODC example's DeepSeek-1.5B fit.
# A 7B/8B-class model has a different attention/MLP FLOP ratio, so the relative
# weighting of long vs short sequences differs. To keep the cost model
# model-appropriate WITHOUT forking this shared function (existing 1.5B runs stay
# byte-for-byte identical), the coefficients are overridable per-run via env:
#   LB_MINI_FIT_A / LB_MINI_FIT_B / LB_MINI_FIT_NORM.
# Unset -> the 1.5B defaults below. For a 7B-class model use a=4.348357, b=8.768377.
_FIT_A_DEFAULT = 1.982122
_FIT_B_DEFAULT = 2.611821
_FIT_NORM_DEFAULT = 32000.0


def get_seq_costs_fit(seq_len: List[int]) -> List[float]:
    """Quadratic-ish fit (attention-aware): a*s^2 + b*s with s normalized.

    Coefficients default to the ODC example's DeepSeek-1.5B fit and may be
    overridden per-run via env (LB_MINI_FIT_A/B/NORM) so a 7B/8B-class model can
    use its own (a=4.348357, b=8.768377). Only meaningful when
    use_packed_attention=true (segmented thd attn); under full O(total^2)
    attention the linear model is the correct target.
    """
    a = float(os.environ.get("LB_MINI_FIT_A", _FIT_A_DEFAULT))
    b = float(os.environ.get("LB_MINI_FIT_B", _FIT_B_DEFAULT))
    norm = float(os.environ.get("LB_MINI_FIT_NORM", _FIT_NORM_DEFAULT))
    normalized = [s / norm for s in seq_len]
    return [a * s * s + b * s for s in normalized]


def resolve_cost_func(name: str) -> Callable[[List[int]], List[float]]:
    if name == "linear":
        return get_seq_costs_linear
    if name == "fit":
        return get_seq_costs_fit
    raise ValueError(f"Unknown LB-Mini cost model: {name!r}. Supported: linear, fit")


# ---------------------------------------------------------------------------
# Karmarkar-Karp largest-differencing partitioning.
# https://en.wikipedia.org/wiki/Largest_differencing_method
# ---------------------------------------------------------------------------
def karmarkar_karp(seq_cost_list: List[float], k_partitions: int, equal_size: bool):
    """Partition indices of ``seq_cost_list`` into ``k_partitions`` balanced sets."""

    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: float):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items, k: int) -> None:
            self.k = k
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur = [idx for idx, _ in self.sets[i].items]
                partitions.append(cur)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> float:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

    sorted_seq_cost_list = sorted([(seqlen, i) for i, seqlen in enumerate(seq_cost_list)])
    states_pq = []
    if equal_size:
        assert len(seq_cost_list) % k_partitions == 0
        for offset in range(0, len(sorted_seq_cost_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seq_cost_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seq_cost_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    partitions = states_pq[0].get_partitions()
    if equal_size:
        for partition in partitions:
            assert len(partition) * k_partitions == len(seq_cost_list)
    return partitions


# ---------------------------------------------------------------------------
# Local-search refinement to tighten the KK partitions.
# ---------------------------------------------------------------------------
_EPS = 1e-6


def _swap_max_partition(seq_cost_list, partitions):
    max_cost = sum(seq_cost_list[_] for _ in partitions[0])
    min_cost = sum(seq_cost_list[_] for _ in partitions[-1])
    for i, item_idx in enumerate(partitions[0]):
        item_cost = seq_cost_list[item_idx]
        for j in range(1, len(partitions)):
            cost_j = sum(seq_cost_list[_] for _ in partitions[j])
            for k, swap_idx in enumerate(partitions[j]):
                swap_cost = seq_cost_list[swap_idx]
                if item_cost - swap_cost <= _EPS:
                    continue
                if cost_j - swap_cost + item_cost + _EPS < max_cost and (
                    max_cost - item_cost + swap_cost > min_cost + _EPS
                ):
                    partitions[j][k] = item_idx
                    partitions[0][i] = swap_idx
                    return True
    return False


def _swap_min_partition(seq_cost_list, partitions):
    max_cost = sum(seq_cost_list[_] for _ in partitions[0])
    min_cost = sum(seq_cost_list[_] for _ in partitions[-1])
    for i, item_idx in enumerate(partitions[-1]):
        item_cost = seq_cost_list[item_idx]
        for j in range(0, len(partitions) - 1):
            cost_j = sum(seq_cost_list[_] for _ in partitions[j])
            for k, swap_idx in enumerate(partitions[j]):
                swap_cost = seq_cost_list[swap_idx]
                if swap_cost - item_cost <= _EPS:
                    continue
                if cost_j - swap_cost + item_cost > min_cost + _EPS and (
                    min_cost - item_cost + swap_cost + _EPS < max_cost
                ):
                    partitions[j][k] = item_idx
                    partitions[-1][i] = swap_idx
                    return True
    return False


def _balance_partition(seq_cost_list, partitions):
    while True:
        partitions = sorted(
            partitions,
            key=lambda x: (sum(seq_cost_list[i] for i in x), min(x) if x else 0),
            reverse=True,
        )
        if _swap_max_partition(seq_cost_list, partitions):
            continue
        if _swap_min_partition(seq_cost_list, partitions):
            continue
        break
    return partitions


def get_seqlen_balanced_partitions(
    seqlen_list: List[int],
    k_partitions: int,
    equal_size: bool,
    get_seq_costs_func: Optional[Callable] = None,
) -> List[List[int]]:
    """Balance ``seqlen_list`` indices into ``k_partitions`` by total workload.

    equal_size=True  -> every partition has the same item count (baseline).
    equal_size=False -> partitions may have different item counts (LB-Mini).
    """
    if get_seq_costs_func is None:
        get_seq_costs_func = get_seq_costs_linear
    seq_cost_list = get_seq_costs_func(seqlen_list)
    assert len(seq_cost_list) >= k_partitions, f"{len(seq_cost_list)} < {k_partitions}"

    def _check_and_sort(partitions):
        assert len(partitions) == k_partitions
        seen = set()
        out = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"partition {i} empty"
            seen.update(partition)
            out[i] = sorted(partition)
        assert seen == set(range(len(seq_cost_list)))
        return out

    def _diff(partitions):
        loads = [sum(seq_cost_list[i] for i in p) for p in partitions]
        return max(loads) - min(loads)

    partitions = karmarkar_karp(seq_cost_list, k_partitions, equal_size)
    partitions = _balance_partition(seq_cost_list, partitions)
    if not equal_size and len(seq_cost_list) % k_partitions == 0:
        eq = karmarkar_karp(seq_cost_list, k_partitions, equal_size=True)
        eq = _balance_partition(seq_cost_list, eq)
        if _diff(partitions) > _diff(eq):
            partitions = eq
    return _check_and_sort(partitions)


def _ceildiv(a, b):
    return -(a // -b)


def rearrange_micro_batches(
    seq_len_effective: List[int],
    max_token_len: int,
    dp_group=None,
    same_num_in_dp: bool = True,
    sort_partition_workload: bool = True,
    get_seq_costs_func: Optional[Callable] = None,
) -> List[List[int]]:
    """Split one rank's samples into micro-batches each <= ``max_token_len``.

    same_num_in_dp=True  -> all DP ranks agree on the micro-batch count via
                            all_reduce(MAX). Megatron's current behaviour.
    same_num_in_dp=False -> each rank uses its own count (LB-Mini; ODC only).
    """
    if get_seq_costs_func is None:
        get_seq_costs_func = get_seq_costs_linear
    total_seqlen = sum(seq_len_effective)
    num_micro_batches = min(len(seq_len_effective), _ceildiv(total_seqlen, max_token_len))
    if dist.is_initialized() and same_num_in_dp:
        t = torch.tensor([num_micro_batches]).cuda()
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro_batches = t.cpu().item()

    while True:
        assert num_micro_batches <= len(seq_len_effective)
        micro_bsz_idx = get_seqlen_balanced_partitions(
            seq_len_effective,
            num_micro_batches,
            equal_size=False,
            get_seq_costs_func=get_seq_costs_func,
        )
        check_failed = False
        for partition in micro_bsz_idx:
            if sum(seq_len_effective[i] for i in partition) > max_token_len:
                check_failed = True
                break
        actual_size = num_micro_batches + 1 if check_failed else len(micro_bsz_idx)
        if dist.is_initialized() and same_num_in_dp:
            t = torch.tensor([actual_size]).cuda()
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=dp_group)
            actual_size = t.cpu().item()
        if actual_size == num_micro_batches:
            break
        num_micro_batches += 1

    if sort_partition_workload:
        micro_bsz_idx.sort(
            key=lambda partition: (
                sum(get_seq_costs_func([seq_len_effective[idx] for idx in partition])),
                min(partition) if partition else 0,
            ),
            reverse=True,
        )
    return micro_bsz_idx


def plan_minibatch(
    lengths: List[int],
    rank: int,
    world_size: int,
    max_token_len: int,
    same_micro_num: bool = False,
    get_seq_costs_func: Optional[Callable] = None,
) -> List[List[int]]:
    """Top-level LB-Mini planner for ONE global minibatch.

    Args:
        lengths: effective seq lengths of ALL samples in this global minibatch
                 (length == minibatch_size * world_size), identical on every rank.
        rank/world_size: this rank within the DP group.
        max_token_len: per-micro-batch token cap (memory constraint).
        same_micro_num: False -> LB-Mini (ODC, ranks differ); True -> baseline.

    Returns:
        local_idx: list of micro-batches for THIS rank; each micro-batch is a list
                   of indices into ``lengths``. ``len(local_idx)`` is this rank's
                   micro-batch count (may differ across ranks when same_micro_num
                   is False).
    """
    if get_seq_costs_func is None:
        get_seq_costs_func = get_seq_costs_linear
    assert max(lengths) <= max_token_len, f"{max(lengths)} > max_token_len={max_token_len}"

    # Step 1: KK-balance the whole minibatch across DP ranks.
    mini_partitions = get_seqlen_balanced_partitions(
        lengths,
        world_size,
        equal_size=same_micro_num,
        get_seq_costs_func=get_seq_costs_func,
    )
    local_index = mini_partitions[rank]
    local_lengths = [lengths[i] for i in local_index]

    # Step 2: split this rank's slice into micro-batches.
    micro_indexes = rearrange_micro_batches(
        local_lengths,
        max_token_len,
        same_num_in_dp=same_micro_num,
        sort_partition_workload=True,
        get_seq_costs_func=get_seq_costs_func,
    )
    # Map rank-local positions back to global minibatch indices.
    local_idx = [[local_index[p] for p in micro] for micro in micro_indexes]
    return local_idx


__all__ = [
    "get_seq_costs_linear",
    "get_seq_costs_fit",
    "resolve_cost_func",
    "karmarkar_karp",
    "get_seqlen_balanced_partitions",
    "rearrange_micro_batches",
    "plan_minibatch",
]
