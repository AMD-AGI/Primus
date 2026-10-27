###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Optional Megatron-LM args: the FLA switches must read cleanly when Megatron-Bridge leaves them unset."""

from types import SimpleNamespace

import pytest

pytest.importorskip("megatron.training")

import megatron.training.global_vars as global_vars

import primus.backends.megatron.core.transformer.fla_flash_attention as ffa
from primus.backends.megatron.core.utils import get_megatron_arg


@pytest.fixture
def no_megatron_args(monkeypatch):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)


def _set_megatron_args(monkeypatch, **fields):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(**fields))


def test_reads_initialized_args(monkeypatch):
    _set_megatron_args(monkeypatch, use_fla_short_conv=True)
    assert get_megatron_arg("use_fla_short_conv", False) is True


def test_missing_attribute_returns_default(monkeypatch):
    _set_megatron_args(monkeypatch)
    assert get_megatron_arg("use_fla_short_conv", False) is False


def test_uninitialized_args_return_default(no_megatron_args):
    assert get_megatron_arg("use_fla_fused_rmsnorm", False) is False
    assert get_megatron_arg("fla_mla_attn", "") == ""


def test_fla_mla_attn_setting_wins_and_auto_applies_without_args(monkeypatch):
    monkeypatch.setattr(ffa, "_flash_attn_exceeds_te_range", lambda: True)
    _set_megatron_args(monkeypatch, fla_mla_attn="0")
    assert ffa.is_enabled() is False

    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    assert ffa.is_enabled() is True


def test_fla_flash_attention_builds_without_megatron_args(no_megatron_args, monkeypatch, capsys):
    monkeypatch.setattr(ffa, "_BANNER_PRINTED", False)
    monkeypatch.setattr(ffa, "_load_flash_attn", lambda: None)

    ffa.FLAFlashAttention(config=SimpleNamespace(), layer_number=1)

    assert "reason=auto-enabled" in capsys.readouterr().err
    assert ffa._BANNER_PRINTED
