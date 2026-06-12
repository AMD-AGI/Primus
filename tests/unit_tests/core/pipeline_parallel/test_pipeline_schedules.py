###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for Primus pipeline-parallel schedule algorithms.

These exercise the scheduler factory and every registered schedule algorithm
(1F1B, interleaved 1F1B, zero-bubble, zero-bubble-heuristic, ZBV formatted, ZBV
greedy v-half/v-min). Schedule-table generation is pure CPU combinatorics (no
GPU / no distributed init), so the tests are fast and deterministic.
"""

import pytest

from primus.core.pipeline_parallel.scheduler.algorithms.base import (
    PipelineScheduleAlgo,
)
from primus.core.pipeline_parallel.scheduler.schedule_table_factory import (
    produce_schedule_instance,
)
from primus.core.pipeline_parallel.scheduler.scheduler_node import (
    FuncType,
    SchedulerNode,
)
from primus.core.pipeline_parallel.utils import find_prev_node_with_type

PP_SIZE = 4
MICRO_BATCHES = 8
FWD = FuncType.F
BWD_TYPES = (FuncType.B, FuncType.BW)

# (algorithm, vpp_size) — vpp constraints differ per algorithm.
SCHEDULES = [
    ("1f1b", 1),
    ("zero-bubble", 1),
    ("zero-bubble-heuristic", 1),
    ("1f1b-interleaved", 2),
    ("zbv-formatted", 2),
    ("v-half", 2),
    ("v-min", 2),
]


def _make(algo, vpp, pp=PP_SIZE, mb=MICRO_BATCHES):
    return produce_schedule_instance(algo, pp_size=pp, vpp_size=vpp, micro_batches=mb)


# ----------------------- factory -----------------------


def test_invalid_algorithm_raises():
    with pytest.raises(ValueError):
        produce_schedule_instance("does-not-exist", pp_size=4, vpp_size=1, micro_batches=8)


def test_factory_returns_algo_instance_and_caches():
    a = _make("1f1b", 1, pp=2, mb=4)
    b = _make("1f1b", 1, pp=2, mb=4)
    assert isinstance(a, PipelineScheduleAlgo)
    assert a is b  # identical args hit the instance cache


# ----------------------- vpp constraints -----------------------


def test_interleaved_requires_vpp_gt_1():
    with pytest.raises(AssertionError):
        _make("1f1b-interleaved", vpp=1)


@pytest.mark.parametrize("algo", ["zbv-formatted", "v-half", "v-min"])
def test_zbv_requires_vpp_2(algo):
    with pytest.raises(AssertionError):
        _make(algo, vpp=1)


# ----------------------- schedule-table generation -----------------------


@pytest.mark.parametrize("algo,vpp", SCHEDULES)
def test_schedule_table_structure(algo, vpp):
    table = _make(algo, vpp).generate_schedule_table()
    assert isinstance(table, list)
    assert len(table) == PP_SIZE, "one node list per pipeline stage"
    for rank_nodes in table:
        assert isinstance(rank_nodes, list) and rank_nodes, "each rank must have nodes"
        assert all(isinstance(n, SchedulerNode) for n in rank_nodes)
        # every stage runs forward compute and some backward compute
        assert any(n.func_type == FWD for n in rank_nodes)
        assert any(n.func_type in BWD_TYPES or n.func_type == FuncType.W for n in rank_nodes)


@pytest.mark.parametrize("algo,vpp", SCHEDULES)
def test_all_microbatches_have_forward(algo, vpp):
    table = _make(algo, vpp).generate_schedule_table()
    # union of forward mini-batch ids on the first stage should cover every microbatch
    fwd_mbs = {n.mini_batch for n in table[0] if n.func_type == FWD}
    assert fwd_mbs >= set(range(MICRO_BATCHES))


def test_1f1b_forward_backward_balance():
    """In plain 1F1B each stage runs exactly one F and one BW per microbatch."""
    table = _make("1f1b", 1).generate_schedule_table()
    for rank_nodes in table:
        f = sum(1 for n in rank_nodes if n.func_type == FuncType.F)
        bw = sum(1 for n in rank_nodes if n.func_type == FuncType.BW)
        assert f == MICRO_BATCHES
        assert bw == MICRO_BATCHES


def test_zero_bubble_splits_backward_into_b_and_w():
    """Zero-bubble splits backward into B (compute) and W (weight grad)."""
    table = _make("zero-bubble", 1).generate_schedule_table()
    has_b = any(n.func_type == FuncType.B for rank in table for n in rank)
    has_w = any(n.func_type == FuncType.W for rank in table for n in rank)
    assert has_b and has_w


# ----------------------- utils.find_prev_node_with_type (pure helper) -----------------------


def _single_rank_nodes():
    return [
        SchedulerNode(FuncType.F, mini_batch=0, chunk=0),
        SchedulerNode(FuncType.F, mini_batch=1, chunk=0),
        SchedulerNode(FuncType.BW, mini_batch=0, chunk=0),
    ]


def test_find_prev_node_defaults_to_current_mb_chunk():
    nodes = _single_rank_nodes()
    # from BW(mb=0) at idx 2, the previous F with the same mb/chunk is idx 0
    assert find_prev_node_with_type(nodes, 2, [FuncType.F]) == 0


def test_find_prev_node_explicit_mb_chunk():
    nodes = _single_rank_nodes()
    assert find_prev_node_with_type(nodes, 2, [FuncType.F], mini_batch=1, chunk=0) == 1


def test_find_prev_node_returns_none_when_absent():
    nodes = _single_rank_nodes()
    assert find_prev_node_with_type(nodes, 2, [FuncType.F], mini_batch=9) is None
