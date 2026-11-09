###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import sys
from types import ModuleType

import pytest

from primus.backends.megatron.training.hylo_utils import (
    hybrid_batch_getter,
    hybrid_pretrain_module,
)

LEGACY = ("tokens", "labels", "loss_mask", "attention_mask", "position_ids")


def _module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install(monkeypatch, mamba, hybrid=None):
    monkeypatch.setitem(sys.modules, "pretrain_mamba", mamba)
    if hybrid is not None:
        monkeypatch.setitem(sys.modules, "pretrain_hybrid", hybrid)


def test_old_megatron_reads_pretrain_mamba_positionally(monkeypatch):
    # Before 0.19 get_batch returns the legacy five first, then any packing metadata.
    mamba = _module(
        "pretrain_mamba",
        train_valid_test_datasets_provider=object(),
        get_batch=lambda it: [f"{key}:{next(it)}" for key in LEGACY] + ["cu_seqlens", "max_seqlen"],
    )
    _install(monkeypatch, mamba)
    assert hybrid_pretrain_module() is mamba
    assert hybrid_batch_getter()(iter([7] * 5)) == tuple(f"{key}:7" for key in LEGACY)


def test_megatron_0_19_reads_pretrain_hybrid_by_batch_key(monkeypatch):
    # Megatron-core 0.19 returns [batch[key] for key in BATCH_KEYS], in alphabetical order.
    keys = (
        "attention_mask",
        "cu_seqlens",
        "cu_seqlens_padded",
        "hybrid_cp_group",
        "labels",
        "local_cp_size",
        "loss_mask",
        "max_seqlen",
        "position_ids",
        "tokens",
    )
    hybrid = _module(
        "pretrain_hybrid",
        train_valid_test_datasets_provider=object(),
        BATCH_KEYS=keys,
        get_batch=lambda it: [key for key in keys],
    )
    _install(monkeypatch, _module("pretrain_mamba"), hybrid)
    assert hybrid_pretrain_module() is hybrid
    assert hybrid_batch_getter()(iter(())) == LEGACY


def test_megatron_0_19_get_batch_must_still_carry_the_legacy_keys(monkeypatch):
    hybrid = _module(
        "pretrain_hybrid",
        train_valid_test_datasets_provider=object(),
        BATCH_KEYS=("tokens", "labels"),
        get_batch=lambda it: ["t", "l"],
    )
    _install(monkeypatch, _module("pretrain_mamba"), hybrid)
    with pytest.raises(KeyError):
        hybrid_batch_getter()(iter(()))
