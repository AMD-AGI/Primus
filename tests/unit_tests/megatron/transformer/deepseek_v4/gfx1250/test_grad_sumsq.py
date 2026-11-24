###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Grad-norm sum-of-squares (``MultiSumsq``): one launch over many ranges and extra FP32 buffers.

* The value matches the per-range ``grad_sumsq`` chain plus the FP32 extras (fp32 accumulation tolerance), and
  matches an fp64 reference.
* The plan is CUDA-graph capturable: replay after mutating the data gives the new value; NaN / Inf in any range
  propagate.
* Ranges and extras live in different allocations placed more than 2 GB apart (a multi-allocation address trap on
  gfx1250 when pointers are given as offsets from one base).
* Misaligned ranges and FP32 extras with numel % 16 != 0 are reported through ``plan.ok`` instead of being summed.
"""

import pytest
import torch

from .helpers import requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"
GB = 1 << 30


@pytest.fixture
def data():
    torch.manual_seed(0)
    spacers = []
    big = (torch.randn(40_000_128, device=DEV) * 1e-3).to(torch.bfloat16)
    spacers.append(torch.empty(int(2.5 * GB), dtype=torch.uint8, device=DEV))
    big2 = (torch.randn(12_345_680, device=DEV) * 1e-3).to(torch.bfloat16)
    spacers.append(torch.empty(int(2.5 * GB), dtype=torch.uint8, device=DEV))
    cuts = [0, 1024, 5008, 7_000_000, 7_000_128, 15_000_000, 39_999_008, 40_000_128]
    ranges = [big2[16:4_000_016]] + [big[a:b] for a, b in zip(cuts[:-1], cuts[1:])] + [big2[4_000_016:]]
    extras = [torch.rand(4096, device=DEV), torch.rand(768, device=DEV)]
    spacers.append(torch.empty(int(2.5 * GB), dtype=torch.uint8, device=DEV))
    extras.append(torch.rand(3_000_000, device=DEV))
    yield big, ranges, extras
    del spacers


def _ref(ranges, extras):
    from primus.backends.megatron.core.distributed.grad_sumsq import grad_sumsq

    acc = None
    for r in ranges:
        n = grad_sumsq(r)
        acc = n if acc is None else acc + n
    return acc + torch.cat(extras).sum()


def test_multi_sumsq_value_and_graph_replay(data):
    from primus.backends.megatron.core.distributed.grad_sumsq import MultiSumsq

    big, ranges, extras = data
    plan = MultiSumsq(ranges, extras)
    assert plan.ok
    a, b = _ref(ranges, extras).item(), plan().item()
    exact = sum(r.double().pow(2).sum().item() for r in ranges) + sum(x.double().sum().item() for x in extras)
    assert abs(a - b) / abs(a) < 1e-5
    assert abs(b - exact) / exact < 1e-5

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        plan()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            out = plan()
    big.mul_(0.5)
    extras[0].mul_(2)
    g.replay()
    torch.cuda.synchronize()
    a, b = _ref(ranges, extras).item(), out.item()
    assert abs(a - b) / abs(a) < 1e-5
    for bad, check in ((float("nan"), torch.isnan), (float("inf"), torch.isinf)):
        big[12345] = bad
        g.replay()
        torch.cuda.synchronize()
        assert check(out).item()


def test_unsupported_inputs_are_reported(data):
    from primus.backends.megatron.core.distributed.grad_sumsq import MultiSumsq

    big, _, _ = data
    assert not MultiSumsq([big[3:1000]], []).ok  # misaligned range start
    x = (torch.randn(4096, device=DEV) * 1e-2).to(torch.bfloat16)
    for n in (3, 17, 48, 1024, 16384):
        e = torch.rand(n, device=DEV)
        m = MultiSumsq([x], [e])
        if n % 16:
            assert not m.ok
        else:
            assert m.ok
            ref = x.double().pow(2).sum().item() + e.double().sum().item()
            assert abs(m().item() - ref) / ref < 1e-6
