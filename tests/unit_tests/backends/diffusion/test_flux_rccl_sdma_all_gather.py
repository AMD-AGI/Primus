###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import warnings
from types import SimpleNamespace

import pytest
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard import _fsdp_collectives

import primus.backends.diffusion.trainers.fsdp2 as fsdp2
import primus.backends.megatron.core.distributed.sdma_param_gather as sdma_param_gather
from primus.backends.diffusion.trainers.fsdp2 import FSDP2Trainer


class FakeOptions:
    def __init__(self):
        self.config = SimpleNamespace(cta_policy=None)


class FakePG:
    def __init__(self, ranks):
        self.ranks = tuple(ranks)


class FakeModule:
    def __init__(self, group):
        self.group = group
        self.comm = None

    def _get_fsdp_state(self):
        return SimpleNamespace(_fsdp_param_groups=[SimpleNamespace(_all_gather_process_group=self.group)])

    def set_custom_all_gather(self, comm):
        self.comm = comm


@pytest.fixture
def fake_dist(monkeypatch):
    """World of 4 ranks seen from rank 0: FSDP groups (0, 1) and (2, 3)."""
    state = {"hosts": ["a", "a", "a", "a"], "new_group_calls": []}
    fsdp2._RCCL_SDMA_GROUPS.clear()
    monkeypatch.setattr(fsdp2, "_RCCL_SDMA_ALL_GATHER_CLS", None)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 4)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: list(group.ranks))

    def all_gather_object(entries, obj):
        groups = [(0, 1), (0, 1), (2, 3), (2, 3)]
        for rank in range(4):
            entries[rank] = (groups[rank], state["hosts"][rank])

    def new_group(ranks, backend=None, pg_options=None, group_desc=None):
        state["new_group_calls"].append((tuple(ranks), backend, pg_options.config.cta_policy, group_desc))
        return FakePG(ranks)

    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(dist, "new_group", new_group)
    monkeypatch.setattr(dist, "is_backend_available", lambda name: name == "nccl-lazy")
    monkeypatch.setattr(
        dist,
        "ProcessGroupNCCL",
        SimpleNamespace(Options=FakeOptions, NCCL_CTA_POLICY_ZERO="zero"),
        raising=False,
    )
    monkeypatch.setattr(fsdp2, "_CopyEngineAllGather", lambda: "copy-comm")
    monkeypatch.setattr(fsdp2, "_rccl_sdma_all_gather", lambda group: ("symm-comm", group.ranks))
    monkeypatch.setenv("FSDP_ALL_GATHER_BACKEND", "rccl_sdma")
    monkeypatch.delenv("FSDP_SDMA_MODULES", raising=False)
    monkeypatch.delenv("FSDP_ALL_GATHER_IMPL", raising=False)
    yield state
    fsdp2._RCCL_SDMA_GROUPS.clear()


def make_trainer():
    trainer = FSDP2Trainer.__new__(FSDP2Trainer)
    trainer.rank = 0
    return trainer


def test_twin_groups_are_zero_cta_nccl_lazy_in_sorted_order(fake_dist):
    fsdp2._prepare_rccl_sdma_groups(FakePG((0, 1)))
    assert fake_dist["new_group_calls"] == [
        ((0, 1), "nccl-lazy", "zero", "FLUX_RCCL_SDMA_ALL_GATHER"),
        ((2, 3), "nccl-lazy", "zero", "FLUX_RCCL_SDMA_ALL_GATHER"),
    ]
    fsdp2._prepare_rccl_sdma_groups(FakePG((0, 1)))
    assert len(fake_dist["new_group_calls"]) == 2


def test_default_uses_rccl_twin_group_within_one_node(fake_dist):
    module = FakeModule(FakePG((0, 1)))
    make_trainer()._maybe_attach_sdma_all_gather(module, "single_blocks.0")
    assert module.comm == ("symm-comm", (0, 1))


def test_default_uses_rccl_twin_group_across_nodes(fake_dist):
    fake_dist["hosts"] = ["a", "b", "a", "b"]
    module = FakeModule(FakePG((0, 1)))
    make_trainer()._maybe_attach_sdma_all_gather(module, "single_blocks.0")
    assert module.comm == ("symm-comm", (0, 1))


def test_copy_within_one_node_uses_hip_ipc(fake_dist, monkeypatch):
    monkeypatch.setenv("FSDP_ALL_GATHER_IMPL", "copy")
    module = FakeModule(FakePG((0, 1)))
    make_trainer()._maybe_attach_sdma_all_gather(module, "single_blocks.0")
    assert module.comm == "copy-comm"


def test_copy_across_nodes_is_rejected(fake_dist, monkeypatch):
    fake_dist["hosts"] = ["a", "b", "a", "b"]
    monkeypatch.setenv("FSDP_ALL_GATHER_IMPL", "copy")
    with pytest.raises(ValueError, match="cannot reach GPUs on other nodes"):
        make_trainer()._maybe_attach_sdma_all_gather(FakeModule(FakePG((0, 1))), "single_blocks.0")


def test_unknown_impl_is_rejected(fake_dist, monkeypatch):
    monkeypatch.setenv("FSDP_ALL_GATHER_IMPL", "auto")
    with pytest.raises(ValueError, match="symm or copy"):
        make_trainer()._maybe_attach_sdma_all_gather(FakeModule(FakePG((0, 1))), "single_blocks.0")


def test_rccl_all_gather_runs_on_its_own_group(monkeypatch):
    calls = []

    class FakeSymmMemAllGather:
        def __init__(self, group, backend):
            self._group = group

        def __call__(self, output_tensor, input_tensor, group, async_op=False):
            calls.append((group, async_op))

    monkeypatch.setattr(_fsdp_collectives, "SymmMemAllGather", FakeSymmMemAllGather, raising=False)
    monkeypatch.setattr(fsdp2, "_RCCL_SDMA_ALL_GATHER_CLS", None)
    comm = fsdp2._rccl_sdma_all_gather("twin")
    comm("out", "in", "fsdp-mesh-group", True)
    assert calls == [("twin", True)]


def test_sdma_fallback_warns_once_per_reason(monkeypatch):
    monkeypatch.setattr(sdma_param_gather, "_FALLBACK_REASONS_WARNED", set())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sdma_param_gather._warn_fallback_once("no symmetric memory")
        sdma_param_gather._warn_fallback_once("no symmetric memory")
        sdma_param_gather._warn_fallback_once("no group name")
    assert [str(w.message) for w in caught] == [
        "SDMA all-gather is using torch.distributed.all_gather_into_tensor: no symmetric memory",
        "SDMA all-gather is using torch.distributed.all_gather_into_tensor: no group name",
    ]
