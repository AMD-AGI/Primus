###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Regression test: ``patch_grad_zero_and_data_prefetch`` must not wrap a
``None`` data iterator.

With PP>=3, Megatron builds no dataset on the middle pipeline stages, so
``train_step`` receives ``data_iterator=None`` there. Wrapping it in a
``CudaPrefetchIterator`` raised ``TypeError: 'NoneType' object is not an
iterator`` on the first step (#1335).

Level 1: Pure Python, no GPU (Megatron and CUDA streams are stubbed).

Run:
    python -m pytest \
      tests/unit_tests/backends/megatron/test_delayed_fp8_prefetch_none_iterator.py -v
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from primus.backends.megatron.patches import delayed_fp8_scaling_patches as dfp
from primus.core.patches import PatchContext


class _FakeStream:
    def wait_stream(self, stream):
        pass


@pytest.fixture
def stub_megatron(monkeypatch):
    """Install stub Megatron modules and CUDA streams; return the stub
    ``megatron.training.training`` module."""

    def _original_train_step(
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        config,
        forward_backward_func,
        iteration=None,
    ):
        return forward_backward_func(data_iterator=data_iterator)

    mods = {
        name: ModuleType(name)
        for name in (
            "megatron",
            "megatron.training",
            "megatron.training.training",
            "megatron.core",
            "megatron.core.distributed",
        )
    }
    mods["megatron"].training = mods["megatron.training"]
    mods["megatron"].core = mods["megatron.core"]
    mods["megatron.training"].training = mods["megatron.training.training"]
    mods["megatron.core"].distributed = mods["megatron.core.distributed"]
    mods["megatron.training.training"].train_step = _original_train_step
    mods["megatron.core.distributed"].DistributedDataParallel = type("DistributedDataParallel", (), {})
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _FakeStream())
    monkeypatch.setitem(dfp._PREFETCH_HANDLE, "state", None)
    return mods["megatron.training.training"]


def test_none_data_iterator_skips_prefetch(stub_megatron):
    """A middle pipeline stage (data_iterator=None, TP=1) must pass None
    through to forward_backward_func without building a prefetcher."""
    params = SimpleNamespace(tensor_model_parallel_size=1, bf16=True)
    ctx = PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=params)},
    )
    dfp.patch_grad_zero_and_data_prefetch(ctx)

    seen = {}

    def forward_backward_func(**kwargs):
        seen["data_iterator"] = kwargs["data_iterator"]
        return "ok"

    result = stub_megatron.train_step(None, None, None, None, None, None, forward_backward_func, iteration=1)

    assert result == "ok"
    assert seen["data_iterator"] is None
    assert "iter" not in dfp.get_prefetch_state()
