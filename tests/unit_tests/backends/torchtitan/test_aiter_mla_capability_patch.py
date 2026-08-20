###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the gfx942 DeepSeek MLA fmha_v3 capability correction."""

from unittest.mock import MagicMock

import pytest
import torch

import primus.backends.torchtitan.patches.turbo.aiter_mla_capability_patches as capability_patch
from primus.core.patches.patch_registry import PatchRegistry

PATCH_ID = "torchtitan.primus_turbo.aiter_mla_capability"


def test_patch_registered_for_torchtitan_setup():
    patch = PatchRegistry.get(PATCH_ID)
    assert patch is not None
    assert patch.backend == "torchtitan"
    assert patch.phase == "setup"


def _kwargs(qk_head_dim=128, v_head_dim=128):
    q = torch.empty(1, 8, 2, qk_head_dim)
    k = torch.empty_like(q)
    v = torch.empty(1, 8, 2, v_head_dim)
    return {
        "dout": torch.empty_like(v),
        "q": q,
        "k": k,
        "v": v,
        "out": torch.empty_like(v),
        "softmax_lse": torch.empty(1, 2, 8),
        "dq": torch.empty_like(q),
        "dk": torch.empty_like(k),
        "dv": torch.empty_like(v),
        "dbias": None,
        "dropout_p": 0.0,
        "softmax_scale": 0.1,
        "causal": True,
        "window_size_left": -1,
        "window_size_right": -1,
        "bias": None,
        "alibi_slopes": None,
        "deterministic": False,
        "rng_state": torch.tensor([3, 7], dtype=torch.int64),
        "sink": None,
        "dsink": None,
        "qkv_format": "bshd",
    }


@pytest.mark.parametrize("qk_head_dim", [128, 192])
def test_deepseek_mla_shapes_route_to_ck(monkeypatch, qk_head_dim):
    monkeypatch.setattr(
        capability_patch,
        "_is_gfx942_tensor",
        lambda tensor: True,
    )
    original = MagicMock(return_value="original")
    mha = MagicMock()
    mha.mha_bwd.return_value = "softmax_d"
    execute = capability_patch._make_execute_wrapper(original, mha)
    kwargs = _kwargs(qk_head_dim=qk_head_dim)

    result = execute(**kwargs)
    assert result == (
        "softmax_d",
        kwargs["dq"],
        kwargs["dk"],
        kwargs["dv"],
        None,
        None,
    )
    original.assert_not_called()
    assert mha.mha_bwd.call_args.args[0] is kwargs["dout"]


def test_other_shapes_keep_original_backend(monkeypatch):
    monkeypatch.setattr(
        capability_patch,
        "_is_gfx942_tensor",
        lambda tensor: True,
    )
    original = MagicMock(return_value="original")
    execute = capability_patch._make_execute_wrapper(original, MagicMock())
    kwargs = _kwargs(qk_head_dim=64, v_head_dim=64)

    assert execute(**kwargs) == "original"
    original.assert_called_once_with(**kwargs)
