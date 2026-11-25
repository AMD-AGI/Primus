###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import sys
from types import ModuleType, SimpleNamespace

import pytest

import primus.backends.megatron.patches.turbo_rmsnorm_bwd_pin_patches as mod


def _ctx(**args):
    defaults = dict(turbo_rmsnorm_bwd_config=[4, 1], use_turbo_rms_norm=True)
    defaults.update(args)
    return SimpleNamespace(extra={"args": SimpleNamespace(**defaults)})


@pytest.fixture(autouse=True)
def _args(monkeypatch):
    monkeypatch.setattr(mod, "get_args", lambda ctx: ctx.extra["args"])


def _fake_turbo(monkeypatch, kernels):
    rk = ModuleType("primus_turbo.triton.normalization.rmsnorm_kernel")
    for name, tuner in kernels.items():
        setattr(rk, name, tuner)
    norm = ModuleType("primus_turbo.triton.normalization")
    norm.rmsnorm_kernel = rk
    for name, m in (
        ("primus_turbo", ModuleType("primus_turbo")),
        ("primus_turbo.triton", ModuleType("primus_turbo.triton")),
        ("primus_turbo.triton.normalization", norm),
        ("primus_turbo.triton.normalization.rmsnorm_kernel", rk),
    ):
        monkeypatch.setitem(sys.modules, name, m)


def _tuner():
    configs = [SimpleNamespace(num_warps=w, num_stages=s) for w in (4, 8) for s in (1, 2)]
    return SimpleNamespace(configs=configs, cache={"key": configs[0]})


def test_missing_turbo_raises_when_turbo_rmsnorm_is_on(monkeypatch):
    monkeypatch.setitem(sys.modules, "primus_turbo.triton.normalization", None)
    with pytest.raises(RuntimeError, match="does not import"):
        mod.patch_turbo_rmsnorm_bwd_config(_ctx())


def test_missing_turbo_is_skipped_when_turbo_rmsnorm_is_off(monkeypatch):
    monkeypatch.setitem(sys.modules, "primus_turbo.triton.normalization", None)
    mod.patch_turbo_rmsnorm_bwd_config(_ctx(use_turbo_rms_norm=False))


def test_pins_both_grid_stride_kernels(monkeypatch):
    tuners = {
        "rmsnorm_bwd_kernel_grid_stride": _tuner(),
        "rmsnorm_bwd_residual_kernel_grid_stride": _tuner(),
    }
    _fake_turbo(monkeypatch, tuners)
    mod.patch_turbo_rmsnorm_bwd_config(_ctx(turbo_rmsnorm_bwd_config=[8, 2]))
    for tuner in tuners.values():
        assert [(c.num_warps, c.num_stages) for c in tuner.configs] == [(8, 2)]
        assert not tuner.cache


def test_unknown_config_or_kernels_raise(monkeypatch):
    _fake_turbo(monkeypatch, {"rmsnorm_bwd_kernel_grid_stride": _tuner()})
    with pytest.raises(ValueError, match="not one of"):
        mod.patch_turbo_rmsnorm_bwd_config(_ctx(turbo_rmsnorm_bwd_config=[16, 1]))
    _fake_turbo(monkeypatch, {})
    with pytest.raises(RuntimeError, match="needs a new pin"):
        mod.patch_turbo_rmsnorm_bwd_config(_ctx())
