###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import inspect
from types import SimpleNamespace

import megatron.core.transformer.multi_token_prediction as mtp_mod
import pytest

from primus.backends.megatron.core.utils import (
    mtp_on_this_rank_compat,
    process_mtp_loss_compat,
)

_CONFIG = SimpleNamespace(
    pipeline_model_parallel_layout=None, mtp_num_layers=1, virtual_pipeline_model_parallel_size=None
)


def _pin_mtp_on_this_rank(config, ignore_virtual=True, vp_stage=None):
    return {"config": config, "ignore_virtual": ignore_virtual, "vp_stage": vp_stage}


def _core019_mtp_on_this_rank(layout=None, mtp_num_layers=None, ignore_virtual=True, vp_stage=None):
    return {"layout": layout, "mtp_num_layers": mtp_num_layers, "vp_stage": vp_stage}


def _main_mtp_on_this_rank(
    layout=None, mtp_num_layers=None, ignore_virtual=True, vp_stage=None, pp_group=None, vp_size=None
):
    return {"mtp_num_layers": mtp_num_layers, "pp_group": pp_group, "vp_size": vp_size}


@pytest.mark.parametrize("fake", [_pin_mtp_on_this_rank, _core019_mtp_on_this_rank, _main_mtp_on_this_rank])
def test_mtp_on_this_rank_compat_matches_each_signature(monkeypatch, fake):
    monkeypatch.setattr(mtp_mod, "mtp_on_this_rank", fake)
    pg_collection = SimpleNamespace(pp="pp-group")
    result = mtp_on_this_rank_compat(_CONFIG, vp_stage=0, pg_collection=pg_collection)
    if fake is _pin_mtp_on_this_rank:
        assert result["config"] is _CONFIG
    elif fake is _main_mtp_on_this_rank:
        assert result == {"mtp_num_layers": 1, "pp_group": "pp-group", "vp_size": None}
    else:
        assert result == {"layout": None, "mtp_num_layers": 1, "vp_stage": 0}


def _recorder(real):
    calls = []

    def fake(*args, **kwargs):
        inspect.signature(real).bind(*args, **kwargs)
        calls.append(kwargs)
        return True

    fake.__signature__ = inspect.signature(real)
    return fake, calls


def test_mtp_on_this_rank_compat_binds_installed_signature(monkeypatch):
    fake, calls = _recorder(mtp_mod.mtp_on_this_rank)
    monkeypatch.setattr(mtp_mod, "mtp_on_this_rank", fake)
    assert mtp_on_this_rank_compat(_CONFIG, vp_stage=None) is True
    assert len(calls) == 1


def test_process_mtp_loss_compat_binds_installed_signature(monkeypatch):
    real = mtp_mod.process_mtp_loss
    fake, calls = _recorder(real)
    monkeypatch.setattr(mtp_mod, "process_mtp_loss", fake)
    required = {
        name: None
        for name, p in inspect.signature(real).parameters.items()
        if p.default is inspect.Parameter.empty and name not in ("runtime_gather_output", "tp_group")
    }
    assert process_mtp_loss_compat(runtime_gather_output=False, tp_group=None, **required) is True
    assert set(calls[0]) <= set(inspect.signature(real).parameters)
