###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The default grid of the per-tensor Triton Adam kernel depends on the GPU architecture."""

from types import SimpleNamespace

import pytest

from primus.core.kernels import triton_adam as ta


@pytest.mark.parametrize(
    "arch, warp_size, expected",
    [
        ("gfx950:sramecc+:xnack-", 64, 2),
        ("gfx942:sramecc+:xnack-", 64, 2),
        ("gfx1250", 32, 16),
        ("gfx9xx-unknown", 64, 2),
        ("", 32, 16),
    ],
)
def test_workgroups_per_cu(arch, warp_size, expected):
    assert ta.workgroups_per_cu(arch, warp_size) == expected


def test_default_grid_size_scales_with_cu_count(monkeypatch):
    props = {
        0: SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-", warp_size=64, multi_processor_count=256),
        1: SimpleNamespace(gcnArchName="gfx1250", warp_size=32, multi_processor_count=256),
    }
    monkeypatch.setattr(ta.torch.cuda, "get_device_properties", lambda i: props[i])
    ta.default_grid_size.cache_clear()
    try:
        assert ta.default_grid_size(0) == 512
        assert ta.default_grid_size(1) == 4096
    finally:
        ta.default_grid_size.cache_clear()
