###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ZAYA_* switch parsing and the line that logs the switches in effect."""

import pytest
from test_zaya1_math import _cfg  # noqa: E402  (pytest puts this dir on sys.path)

from primus.backends.megatron.core.models.zaya1 import zaya1_modules as zm


@pytest.mark.parametrize("value", ["1", "true", "True", "ON", "yes", " on "])
def test_switch_values_that_mean_on(monkeypatch, value):
    monkeypatch.setenv("ZAYA_TEST_SWITCH", value)
    assert zm._env_on("ZAYA_TEST_SWITCH", "0")


@pytest.mark.parametrize("value", ["0", "false", "False", "FALSE", "off", "No", ""])
def test_switch_values_that_mean_off(monkeypatch, value):
    monkeypatch.setenv("ZAYA_TEST_SWITCH", value)
    assert not zm._env_on("ZAYA_TEST_SWITCH", "1")


@pytest.mark.parametrize("value", ["2", "enabled", "fasle"])
def test_unknown_switch_values_raise(monkeypatch, value):
    monkeypatch.setenv("ZAYA_TEST_SWITCH", value)
    with pytest.raises(ValueError, match="ZAYA_TEST_SWITCH"):
        zm._env_on("ZAYA_TEST_SWITCH")


def test_moe_gemm_takes_split_or_turbo(monkeypatch):
    monkeypatch.delenv("ZAYA_MOE_GEMM", raising=False)
    assert zm._env_choice("ZAYA_MOE_GEMM", "split", ("split", "turbo")) == "split"
    monkeypatch.setenv("ZAYA_MOE_GEMM", " Turbo ")
    assert zm._env_choice("ZAYA_MOE_GEMM", "split", ("split", "turbo")) == "turbo"
    monkeypatch.setenv("ZAYA_MOE_GEMM", "trubo")
    with pytest.raises(ValueError, match="trubo"):
        zm._env_choice("ZAYA_MOE_GEMM", "split", ("split", "turbo"))


def test_switch_summary_names_the_experts_that_run(monkeypatch):
    monkeypatch.setattr(zm, "MOE_GEMM", "turbo")
    assert "ZAYA_MOE_GEMM=turbo (experts: turbo)" in zm.switch_summary(_cfg())
    assert "ZAYA_MOE_GEMM=turbo (experts: split, turbo is BF16-only)" in zm.switch_summary(_cfg(fp8="hybrid"))
    monkeypatch.setattr(zm, "ATTN_FLASH", False)
    monkeypatch.setattr(zm, "FUSE_NORM", False)
    line = zm.switch_summary(_cfg())
    assert "ZAYA_ATTN_FLASH=off (dense mask)" in line and "ZAYA_FUSE_NORM=off" in line
