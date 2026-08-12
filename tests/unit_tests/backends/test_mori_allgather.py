###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest
import torch

from primus.backends.common import mori_allgather


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

    monkeypatch.setenv("MORI_FSDP_COMPACT_WORKSPACE", "1")
    monkeypatch.setattr(mori_allgather, "ensure_mori_shmem_initialized", lambda _: None)
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

    args, kwargs = calls[0]
    assert args == (0, 16)
    assert kwargs["input_buffer_size"] == 8 * mib
    assert kwargs["output_buffer_size"] == 2 * 381 * mib
    assert kwargs["slice_min_bytes"] == 8 * mib
    assert kwargs["slice_direct"] is True
