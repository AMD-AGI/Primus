###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""turbo_grouped_gemm_schedule reaches the BF16 grouped GEMM of PrimusTurboGroupedLinear."""

from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.core.extensions import primus_turbo as pt


@pytest.mark.parametrize("schedule", ["static", "work_steal"])
def test_bf16_grouped_gemm_gets_the_schedule(monkeypatch, schedule):
    calls = []
    monkeypatch.setattr(
        pt.primus_turbo_torch.ops, "grouped_gemm", lambda *args, **kwargs: calls.append(kwargs) or "out"
    )
    for name in ("is_turbo_fp8_enabled", "is_turbo_fp4_enabled"):
        monkeypatch.setattr(pt.PrimusTurboLowPrecisionGlobalStateManager, name, staticmethod(lambda: False))
    layer = SimpleNamespace(prepare_weights=lambda x: (x, "weights", None), grouped_gemm_schedule=schedule)
    out, bias = pt.PrimusTurboGroupedLinear.forward_internal(layer, torch.zeros(4, 2), torch.tensor([4]))
    assert out == "out" and bias is None
    assert calls == [{"trans_b": True, "schedule": schedule, "fuse_bgrad_accum_pattern": None}]
