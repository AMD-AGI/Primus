###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch.distributed.fsdp as fsdp_pkg
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as ffsc
from torch.distributed.fsdp._fully_shard import _fully_shard as ffs_mod

import primus.backends.torchtitan.patches.sdma_symm_mem_collectives as sdma_patch
from primus.core.patches.patch_registry import PatchRegistry

PATCH_ID = "torchtitan.fsdp.sdma_symm_mem_collectives"


class FakeSymmMemAllGather:
    def __init__(self, group, backend):
        self.group = group
        self.backend = backend


class FakeFsdpModule:
    def __init__(self, n_groups=1):
        self.groups = [SimpleNamespace(_all_gather_process_group=f"pg{i}") for i in range(n_groups)]
        self.custom_all_gather = None

    def _get_fsdp_state(self):
        return SimpleNamespace(_fsdp_param_groups=self.groups)

    def set_custom_all_gather(self, comm):
        self.custom_all_gather = comm


@pytest.fixture
def patched(monkeypatch):
    """Run the patch over a stub fully_shard; returns (orig, early-imported module)."""

    def orig_fully_shard(module, **kwargs):
        return module

    orig_fully_shard.state = lambda module: "state"
    monkeypatch.setattr(fsdp_pkg, "fully_shard", orig_fully_shard)
    monkeypatch.setattr(ffs_mod, "fully_shard", orig_fully_shard)
    monkeypatch.setattr(ffsc, "SymmMemAllGather", FakeSymmMemAllGather, raising=False)
    monkeypatch.setattr(sdma_patch, "log_rank_0", lambda *args, **kwargs: None)
    early = ModuleType("fake_torchtitan_parallelize")
    early.fully_shard = orig_fully_shard
    monkeypatch.setitem(sys.modules, early.__name__, early)

    sdma_patch.patch_torchtitan_fsdp_sdma_symm_mem(None)
    return orig_fully_shard, early


class TestSdmaSymmMemPatchRegistration:
    def test_patch_registered_before_weight_tying(self):
        patch = PatchRegistry.get(PATCH_ID)
        assert patch is not None
        assert patch.backend == "torchtitan"
        assert patch.phase == "setup"
        assert patch.priority < PatchRegistry.get("torchtitan.fsdp.weight_tying").priority

    @pytest.mark.parametrize(
        "env, enabled",
        [
            ({}, False),
            ({"SDMA_ALL_GATHER": "1"}, True),
            ({"FSDP_ALL_GATHER_BACKEND": "rccl_sdma"}, True),
            ({"FSDP_ALL_GATHER_BACKEND": "nccl"}, False),
        ],
    )
    def test_gate(self, monkeypatch, env, enabled):
        monkeypatch.delenv("SDMA_ALL_GATHER", raising=False)
        monkeypatch.delenv("FSDP_ALL_GATHER_BACKEND", raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert sdma_patch._sdma_all_gather_enabled(None) is enabled


class TestSdmaSymmMemPatchAttach:
    def test_rebinds_already_imported_alias(self, patched):
        orig, early = patched
        assert early.fully_shard is fsdp_pkg.fully_shard
        assert early.fully_shard is not orig
        assert early.fully_shard.state is orig.state

        module = FakeFsdpModule()
        assert early.fully_shard(module) is module
        assert isinstance(module.custom_all_gather, FakeSymmMemAllGather)
        assert module.custom_all_gather.group == "pg0"
        assert module.custom_all_gather.backend == "NCCL"

    def test_list_input_attaches_to_shared_state(self, patched):
        first, second = FakeFsdpModule(), FakeFsdpModule()
        result = fsdp_pkg.fully_shard([first, second])
        assert result == [first, second]
        assert isinstance(first.custom_all_gather, FakeSymmMemAllGather)

    def test_skips_multi_group_module(self, patched):
        module = FakeFsdpModule(n_groups=2)
        fsdp_pkg.fully_shard(module)
        assert module.custom_all_gather is None
