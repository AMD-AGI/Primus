###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest
import torch

from primus.backends.common import mori_allgather


def test_deferred_work_waits_on_device_event_without_host_sync(monkeypatch):
    waited_events = []

    class ConsumerStream:
        def wait_event(self, event):
            waited_events.append(event)

        def wait_stream(self, stream):
            raise AssertionError("event path should not wait on the producer stream")

    class Event:
        def synchronize(self):
            raise AssertionError("wait must not synchronize the host")

    event = Event()
    monkeypatch.setattr(mori_allgather.torch.cuda, "current_stream", lambda _: ConsumerStream())
    work = mori_allgather._DeviceDeferredEventWork(
        stream=SimpleNamespace(),
        device=torch.device("cuda", 0),
        event=event,
    )

    assert work.wait()
    assert work.wait()
    assert waited_events == [event]
    assert work.is_completed()


def test_dense_node_defaults_to_async_completion(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.delenv("MORI_HIER_DEBUG_SYNC", raising=False)

    mori_allgather.MoriAllGather()

    assert mori_allgather.os.environ["MORI_HIER_DEBUG_SYNC"] == "0"


def test_compact_workspace_sizes_cover_sliced_and_fallback_paths():
    mib = 1 << 20

    input_bytes, output_bytes = mori_allgather._compact_workspace_sizes(
        cap_bytes=381 * mib,
        world_size=16,
        ranks_per_node=8,
        slice_min_bytes=8 * mib,
    )

    assert input_bytes == 8 * mib
    assert output_bytes == 2 * 381 * mib


def test_compact_workspace_sizes_keep_full_output_for_small_messages():
    mib = 1 << 20

    input_bytes, output_bytes = mori_allgather._compact_workspace_sizes(
        cap_bytes=4 * mib,
        world_size=16,
        ranks_per_node=8,
        slice_min_bytes=8 * mib,
    )

    assert input_bytes == 4 * mib
    assert output_bytes == 16 * 4 * mib


def test_compact_workspace_sizes_reject_invalid_topology():
    with pytest.raises(ValueError, match="must be divisible"):
        mori_allgather._compact_workspace_sizes(
            cap_bytes=1024,
            world_size=16,
            ranks_per_node=6,
            slice_min_bytes=1024,
        )


def test_auto_shmem_heap_uses_two_gib_for_compact_workspace():
    mib = 1 << 20
    gib = 1 << 30

    heap_bytes = mori_allgather._auto_shmem_heap_bytes(
        input_buffer_size=8 * mib,
        output_buffer_size=762 * mib,
        world_size=16,
        ranks_per_node=8,
    )

    assert heap_bytes == 2 * gib


def test_auto_shmem_heap_grows_for_large_workspace():
    mib = 1 << 20
    gib = 1 << 30

    heap_bytes = mori_allgather._auto_shmem_heap_bytes(
        input_buffer_size=8 * mib,
        output_buffer_size=6 * gib,
        world_size=16,
        ranks_per_node=8,
    )

    assert heap_bytes == 15 * gib


def test_observe_fsdp_param_group_uses_effective_dtype_and_rounds_up():
    adapter = mori_allgather.MoriAllGather.__new__(mori_allgather.MoriAllGather)
    adapter._observed_max_shard_bytes = 0
    group = SimpleNamespace(
        mp_policy=SimpleNamespace(param_dtype=torch.bfloat16),
        fsdp_params=[
            SimpleNamespace(_sharded_param_data=torch.empty(100, dtype=torch.float32)),
            SimpleNamespace(_sharded_param_data=torch.empty(20, dtype=torch.int32)),
        ],
    )

    assert adapter.observe_fsdp_param_group(group) == 1 << 20
    assert adapter._observed_max_shard_bytes == 1 << 20


def test_registration_output_uses_persistent_backing_extent():
    adapter = mori_allgather.MoriAllGather.__new__(mori_allgather.MoriAllGather)
    adapter._host_proxy = False
    adapter._output_buffer = torch.empty(1024)
    output_view = adapter._output_buffer.narrow(0, 0, 512)

    registration_output = adapter._registration_output(output_view)

    assert registration_output is adapter._output_buffer
    assert registration_output.numel() == 1024


def test_registration_output_rejects_offset_view_and_host_proxy():
    adapter = mori_allgather.MoriAllGather.__new__(mori_allgather.MoriAllGather)
    adapter._host_proxy = False
    adapter._output_buffer = torch.empty(1024)
    offset_view = adapter._output_buffer.narrow(0, 1, 512)

    assert adapter._registration_output(offset_view) is offset_view

    adapter._host_proxy = True
    prefix_view = adapter._output_buffer.narrow(0, 0, 512)
    assert adapter._registration_output(prefix_view) is prefix_view


def test_observed_capacity_builds_compact_collective_once(monkeypatch):
    mib = 1 << 20
    calls = []
    collective = object()
    fake_shmem = SimpleNamespace(shmem_mype=lambda: 0, shmem_npes=lambda: 16)

    def hier_all_gather(*args, **kwargs):
        calls.append((args, kwargs))
        return collective

    fake_ccl = SimpleNamespace(HierAllGather=hier_all_gather)

    def import_module(name):
        if name == "mori.shmem":
            return fake_shmem
        if name == "mori.ccl":
            return fake_ccl
        raise AssertionError(f"unexpected import: {name}")

    initialized_heaps = []
    monkeypatch.setenv("MORI_FSDP_COMPACT_WORKSPACE", "1")
    monkeypatch.delenv("MORI_SHMEM_HEAP_SIZE", raising=False)
    monkeypatch.setattr(
        mori_allgather,
        "ensure_mori_shmem_initialized",
        lambda _: initialized_heaps.append(mori_allgather.os.environ["MORI_SHMEM_HEAP_SIZE"]),
    )
    monkeypatch.setattr(mori_allgather.importlib, "import_module", import_module)
    monkeypatch.setattr(mori_allgather, "_safe_log_rank_0", lambda _: None)

    adapter = mori_allgather.MoriAllGather.__new__(mori_allgather.MoriAllGather)
    adapter._ranks_per_node = 8
    adapter._collective = None
    adapter._rank = None
    adapter._world_size = None
    adapter._cap_bytes = 0
    adapter._observed_max_shard_bytes = 381 * mib
    adapter._host_proxy = False
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 16)

    assert adapter._get_collective(group, 380 * mib) is collective
    assert adapter._get_collective(group, 128 * mib) is collective
    assert len(calls) == 1
    assert initialized_heaps == ["2G"]

    args, kwargs = calls[0]
    assert args == (0, 16)
    assert kwargs["input_buffer_size"] == 8 * mib
    assert kwargs["output_buffer_size"] == 2 * 381 * mib
    assert kwargs["slice_min_bytes"] == 8 * mib
    assert kwargs["slice_direct"] is True
