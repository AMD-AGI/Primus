###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Which pipeline stages ``get_batch_func`` (zero-bubble DataLoaderStore) loads a batch on."""

from types import SimpleNamespace

import pytest
import torch

import primus.backends.megatron.data_loader_store as store

_KEYS = ("tokens", "labels", "loss_mask", "attention_mask", "position_ids")


@pytest.fixture
def stage(monkeypatch):
    calls = []

    def fake_tp_batch(data_iterator, mtp_on_this_rank):
        calls.append(mtp_on_this_rank)
        return {key: torch.zeros(1, 4) for key in _KEYS}

    args = SimpleNamespace(
        patch_zero_bubble=False, context_parallel_size=1, enable_primus_turbo=False, use_turbo_attention=False
    )
    monkeypatch.setattr(store, "get_args", lambda: args)
    monkeypatch.setattr(store, "_get_batch_on_this_tp_rank", fake_tp_batch)
    monkeypatch.setattr(store, "get_batch_on_this_cp_rank", lambda batch, **kwargs: batch)
    monkeypatch.setattr(store, "get_context_parallel_group", lambda: None)

    def set_stage(first_or_last, mtp):
        monkeypatch.setattr(store, "is_first_or_last_pipeline_stage", lambda vp_stage: first_or_last)
        monkeypatch.setattr(store, "_mtp_on_this_rank", lambda vp_stage: mtp)

    return set_stage, calls


def test_middle_stage_without_mtp_skips_the_batch(stage):
    set_stage, calls = stage
    set_stage(first_or_last=False, mtp=False)
    assert store.get_batch_func(iter([]), vp_stage=0) == (None,) * 5
    assert calls == []


def test_middle_stage_with_mtp_loads_the_batch(stage):
    set_stage, calls = stage
    set_stage(first_or_last=False, mtp=True)
    tokens, *_ = store.get_batch_func(iter([]), vp_stage=1)
    assert tokens is not None
    assert calls == [True]


def test_first_or_last_stage_loads_the_batch(stage):
    set_stage, calls = stage
    set_stage(first_or_last=True, mtp=False)
    tokens, *_ = store.get_batch_func(iter([]), vp_stage=0)
    assert tokens is not None
    assert calls == [False]
