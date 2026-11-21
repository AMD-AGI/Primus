###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Replication with a Primus-Turbo whose grouped_gemm does not take weight segments fails at startup."""

import sys
import types

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("importing the Primus-Turbo extension needs a GPU", allow_module_level=True)

pytest.importorskip("primus_turbo")

from primus.backends.megatron.core.extensions import primus_turbo  # noqa: E402

_IMPL = "primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl"


def test_turbo_without_segments_is_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, _IMPL, types.ModuleType(_IMPL))
    assert not primus_turbo._turbo_grouped_gemm_takes_segments()
    with pytest.raises(RuntimeError, match="Primus-Turbo#675"):
        primus_turbo._require_turbo_grouped_gemm_segments()


def test_turbo_with_segments_passes():
    if not primus_turbo._turbo_grouped_gemm_takes_segments():
        pytest.skip("this Primus-Turbo has no segmented grouped_gemm")
    primus_turbo._require_turbo_grouped_gemm_segments()
