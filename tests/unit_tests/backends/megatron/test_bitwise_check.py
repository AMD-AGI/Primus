###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus.backends.megatron.core.fusions import bitwise_check


@pytest.fixture(autouse=True)
def fresh_checks(monkeypatch):
    monkeypatch.setattr(bitwise_check, "_results", {})


def test_same_bits():
    a = torch.tensor([1.0, float("nan"), -0.0], dtype=torch.bfloat16)
    assert bitwise_check.same_bits(a, a.clone())
    assert not bitwise_check.same_bits(a, torch.tensor([1.0, float("nan"), 0.0], dtype=torch.bfloat16))
    assert not bitwise_check.same_bits(a, a.float())
    assert not bitwise_check.same_bits(a, None)
    assert bitwise_check.same_bits(None, None)
    assert bitwise_check.same_bits(torch.arange(3), torch.arange(3))


def test_outputs_and_grads_leaves_the_rng_and_other_leaves_alone():
    w = torch.randn(4, requires_grad=True)
    x = torch.randn(4)
    state = torch.get_rng_state()
    out = bitwise_check.outputs_and_grads(lambda x, u, i: (x * w)[i], x, torch.ones(3), torch.tensor([0, 2]))
    assert torch.equal(torch.get_rng_state(), state)
    assert w.grad is None
    y, dx, du = out
    assert y.shape == (2,) and dx.shape == (4,) and du is None


def test_passed_is_decided_once_per_name_and_key():
    calls = []

    def run():
        calls.append(1)
        return [torch.ones(2)]

    assert bitwise_check.passed("same", run, lambda: [torch.ones(2)])
    assert bitwise_check.passed("same", run, lambda: [torch.zeros(2)])
    assert calls == [1]
    assert bitwise_check.passed("same", run, lambda: [torch.ones(2)], key=(4, 8))
    assert not bitwise_check.passed("same", run, lambda: [torch.zeros(2)], key=(4, 16))
    assert calls == [1, 1, 1]
    assert bitwise_check._results == {("same", None): True, ("same", (4, 8)): True, ("same", (4, 16)): False}


def test_a_mismatch_warns_on_every_rank(monkeypatch):
    warned = []
    monkeypatch.setattr(bitwise_check, "warning_rank_all", warned.append)

    assert not bitwise_check.passed("differs", lambda: [torch.ones(2)], lambda: [torch.zeros(2)])
    assert not bitwise_check.passed("lengths", lambda: [torch.ones(2)], lambda: [torch.ones(2)] * 2)
    assert len(warned) == 2 and "keeping that path" in warned[0]


def test_an_error_keeps_the_original_path():
    def run():
        raise ImportError("missing helper")

    assert not bitwise_check.passed("raises", run, lambda: [torch.ones(2)])
    assert bitwise_check._results == {("raises", None): False}


def test_running_out_of_memory_keeps_the_original_path():
    def run():
        raise torch.cuda.OutOfMemoryError("HIP out of memory. Tried to allocate 2.00 GiB")

    assert not bitwise_check.passed("oom", run, lambda: [torch.ones(2)])
    assert bitwise_check._results == {("oom", None): False}


def test_a_gpu_fault_is_raised():
    def run():
        raise RuntimeError("HIP error: an illegal memory access was encountered")

    with pytest.raises(RuntimeError, match="illegal memory access"):
        bitwise_check.passed("faults", run, lambda: [torch.ones(2)])
    assert bitwise_check._results == {}


def test_nothing_to_compare_decides_nothing():
    assert not bitwise_check.passed("empty", lambda: [torch.ones(0)], lambda: [torch.ones(0)])
    assert bitwise_check._results == {}
    assert bitwise_check.passed("empty", lambda: [torch.ones(2)], lambda: [torch.ones(2)])


def test_inference_mode_decides_nothing():
    with torch.inference_mode():
        assert not bitwise_check.passed("same", lambda: [torch.ones(2)], lambda: [torch.ones(2)])
    assert bitwise_check._results == {}
    assert bitwise_check.passed("same", lambda: [torch.ones(2)], lambda: [torch.ones(2)])
