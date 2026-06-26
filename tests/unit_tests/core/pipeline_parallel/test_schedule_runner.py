###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the pipeline-schedule runtime *orchestration*.

`ScheduleRunner.run` walks a schedule table and dispatches each node to the
matching handler. The orchestration is exercised with mock handlers (no real
forward/backward/comm) — the handlers themselves are covered by E2E PP training.

Importing `scheduler` pulls in `offload_handler`, which initializes GPU state at
import time, so this module is skipped cleanly on hosts without a ROCm GPU.
"""

import pytest

try:
    from primus.core.pipeline_parallel.scheduler.scheduler import ScheduleRunner
    from primus.core.pipeline_parallel.scheduler.scheduler_node import (
        FuncType,
        SchedulerNode,
    )
except Exception as exc:  # noqa: BLE001 - import pulls in GPU-initializing offload buffer
    pytest.skip(f"pipeline ScheduleRunner requires a GPU at import: {exc}", allow_module_level=True)


@pytest.fixture(autouse=True)
def _stub_offload_buffer(monkeypatch):
    """run() touches the global OFFLOAD_BUFFER at the end; stub it so the
    orchestration test stays isolated from offload state."""

    class _Buf:
        record_offload_memory_info = False

        def check_empty(self):
            return None

    monkeypatch.setattr("primus.core.pipeline_parallel.scheduler.scheduler.OFFLOAD_BUFFER", _Buf())


def _recording_handlers(calls):
    def make(key):
        def handler(node, idx, table):
            calls.append(key)

        return handler

    return {ft: make(ft) for ft in FuncType}


def test_runner_dispatches_each_node_in_order():
    calls = []
    runner = ScheduleRunner(handle_func_dict=_recording_handlers(calls))
    table = [[SchedulerNode(FuncType.F, 0, 0), SchedulerNode(FuncType.BW, 0, 0)]]
    runner.run(table, rank=0)
    assert calls == [FuncType.F, FuncType.BW]


def test_runner_combined_group_routes_to_fb_handler():
    calls = []
    runner = ScheduleRunner(handle_func_dict=_recording_handlers(calls))
    combined = SchedulerNode(FuncType.F, 0, 0, args={"combined_node": True, "combined_group": ["x"]})
    runner.run([[combined]], rank=0)
    assert calls == [FuncType.FB]


def test_runner_invokes_pre_and_post_process_hooks():
    pre, post = [], []
    handlers = {ft: (lambda node, idx, table: None) for ft in FuncType}
    runner = ScheduleRunner(
        handle_func_dict=handlers,
        pre_process_func=lambda n, i, t: pre.append(n),
        post_process_func=lambda n, i, t: post.append(n),
    )
    table = [[SchedulerNode(FuncType.F, 0, 0), SchedulerNode(FuncType.BW, 1, 0)]]
    runner.run(table, rank=0)
    assert len(pre) == 2 and len(post) == 2
