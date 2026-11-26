###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the Megatron-FSDP dense-model expert data-parallel group fallback."""

from types import SimpleNamespace

import pytest

pytest.importorskip("megatron.core.distributed.fsdp.src.megatron_fsdp.utils")

from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import (  # noqa: E402
    FSDPDistributedIndex,
)

import primus.backends.megatron.patches.megatron_fsdp_patches as patch_mod  # noqa: E402
from primus.core.patches.context import PatchContext  # noqa: E402


@pytest.fixture
def patched_get_dp_group(monkeypatch):
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)
    original = FSDPDistributedIndex.get_dp_group
    patch_mod.patch_megatron_fsdp_dense_expert_dp_group(None)
    yield
    FSDPDistributedIndex.get_dp_group = original


def _index(expt_fsdp_group):
    index = object.__new__(FSDPDistributedIndex)
    index.use_hybrid_fsdp = False
    index.fsdp_group = "dp_group"
    index.expt_fsdp_group = expt_fsdp_group
    return index


def test_dense_model_falls_back_to_dp_group(patched_get_dp_group):
    index = _index(expt_fsdp_group=None)
    assert index.get_dp_group(is_expert_parallel=True) == "dp_group"
    assert index.get_dp_group() == "dp_group"


def test_moe_model_keeps_expert_group(patched_get_dp_group):
    index = _index(expt_fsdp_group="expt_group")
    assert index.get_dp_group(is_expert_parallel=True) == "expt_group"


@pytest.mark.parametrize(
    "use_megatron_fsdp,num_experts,expected",
    [(True, None, True), (True, 0, True), (True, 8, False), (False, None, False)],
)
def test_applies_only_to_dense_megatron_fsdp(use_megatron_fsdp, num_experts, expected):
    args = SimpleNamespace(use_megatron_fsdp=use_megatron_fsdp, num_experts=num_experts)
    ctx = PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=args)},
    )
    assert patch_mod._is_dense_megatron_fsdp(ctx) is expected


def test_patch_is_idempotent(patched_get_dp_group):
    patched = FSDPDistributedIndex.get_dp_group
    patch_mod.patch_megatron_fsdp_dense_expert_dp_group(None)
    assert FSDPDistributedIndex.get_dp_group is patched
