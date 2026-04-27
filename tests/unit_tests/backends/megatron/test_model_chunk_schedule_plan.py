###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


class _FakeNode:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def forward(self, value):
        self.calls.append(f"{self.name}.forward")
        return f"{self.name}.forward({value})"

    def backward(self, value):
        self.calls.append(f"{self.name}.backward")
        return f"{self.name}.backward({value})"


class _FakeLayer:
    def __init__(self, prefix, calls, early_attn_release):
        self.config = SimpleNamespace(ep_overlap_early_attn_memory_release=early_attn_release)
        self.attn = _FakeNode(f"{prefix}.attn", calls)
        self.mlp = _FakeNode(f"{prefix}.mlp", calls)
        self.moe_dispatch = _FakeNode(f"{prefix}.moe_dispatch", calls)
        self.moe_combine = _FakeNode(f"{prefix}.moe_combine", calls)
        self.mtp_post_process = _FakeNode(f"{prefix}.mtp_post_process", calls)

    def get_fp8_context(self):
        return nullcontext()


def _import_model_chunk_schedule_plan(monkeypatch: pytest.MonkeyPatch):
    fake_package_name = "primus.backends.megatron.core.pipeline_parallel.zerobubble"
    fake_module_name = f"{fake_package_name}.zbpp_utils"
    target_module_name = "primus.backends.megatron.core.models.common.model_chunk_schedule_plan"

    fake_package = types.ModuleType(fake_package_name)
    fake_package.__path__ = []

    fake_zbpp_utils = types.ModuleType(fake_module_name)

    class _FakeWeightGradStore:
        @staticmethod
        def split_bw():
            return False

        @staticmethod
        def flush(num=None):
            return None

        @staticmethod
        def queue_size():
            return 0

        @staticmethod
        def pop():
            return None

    fake_zbpp_utils.WeightGradStore = _FakeWeightGradStore
    fake_package.zbpp_utils = fake_zbpp_utils

    monkeypatch.setitem(sys.modules, fake_package_name, fake_package)
    monkeypatch.setitem(sys.modules, fake_module_name, fake_zbpp_utils)
    monkeypatch.delitem(sys.modules, target_module_name, raising=False)
    return importlib.import_module(target_module_name)


@pytest.mark.parametrize("early_attn_release", [False, True])
def test_execute_overlapped_1f1b_matches_upstream_schedule_plan_interface(
    monkeypatch: pytest.MonkeyPatch,
    early_attn_release: bool,
):
    """Repro for #665: current upstream layers no longer expose a post_attn node."""

    model_chunk_schedule_plan = _import_model_chunk_schedule_plan(monkeypatch)
    calls = []
    monkeypatch.setattr(
        model_chunk_schedule_plan,
        "pop_weight_grad",
        lambda num=None: calls.append(f"pop_weight_grad({num})"),
    )

    f_layer = _FakeLayer("f", calls, early_attn_release)
    b_layer = _FakeLayer("b", calls, early_attn_release)

    f_input, b_grad = model_chunk_schedule_plan.execute_overlapped_1f1b(
        f_layer,
        b_layer,
        f_input="forward-input",
        b_grad="backward-grad",
        is_last_layer_in_bwd=False,
    )

    assert "post_attn" not in "".join(calls)
    assert calls.count("pop_weight_grad(1)") == 2
    assert calls.index("b.moe_dispatch.backward") < calls.index("b.attn.backward")
    assert calls.index("f.attn.forward") < calls.index("f.moe_dispatch.forward")
    if early_attn_release:
        assert calls.index("b.attn.backward") < calls.index("f.mlp.forward")
    else:
        assert calls.index("f.mtp_post_process.forward") < calls.index("b.attn.backward")
    assert f_input.startswith("f.mtp_post_process.forward(")
    assert "forward-input" in f_input
    assert b_grad.startswith("b.attn.backward(")
