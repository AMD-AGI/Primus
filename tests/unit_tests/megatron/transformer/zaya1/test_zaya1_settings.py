###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The kernel-path config fields: their defaults, the modules that read them, and the build-log line."""

from pathlib import Path
from types import SimpleNamespace

import yaml
from test_zaya1_math import _cfg  # noqa: E402  (pytest puts this dir on sys.path)

import primus
from primus.backends.megatron.core.models.zaya1 import zaya1_modules as zm
from primus.backends.megatron.core.models.zaya1.zaya1_transformer_config import (
    Zaya1TransformerConfig,
)

_PATHS = (
    "zaya_attn_flash",
    "zaya_fuse_norm",
    "zaya_fuse_cca",
    "zaya_conv_gemm",
    "zaya_fuse_moe",
    "zaya_router_defer_ar",
)
_CHUNKED_CE = ("zaya_chunked_ce", "zaya_chunked_ce_tokens", "zaya_chunked_ce_compile")


def test_base_yaml_and_modules_default_like_the_config():
    base = Path(primus.__file__).parent / "configs" / "models" / "megatron" / "zaya1_base.yaml"
    values = yaml.safe_load(base.read_text())
    for name in _PATHS + _CHUNKED_CE:
        assert values[name] == getattr(Zaya1TransformerConfig, name), name
    for name in _PATHS:
        assert zm._PATH_DEFAULTS[name] == getattr(Zaya1TransformerConfig, name), name


def test_modules_read_the_fields_when_built():
    for flag in (True, False):
        stack = zm.ZayaStack(_cfg(num_layers=2, **{name: flag for name in _PATHS}))
        cca, moe = stack.layers[0].self_attn, stack.layers[1].zaya_block
        assert stack.fuse_norm is stack.attn_flash is stack.layers[0].fuse_norm is flag
        assert cca.attn_flash is cca.fuse_cca is cca._conv_gemm_ok is flag
        assert moe.fuse_moe is moe.router.fuse_moe is moe.router.defer_bias_update is flag
        assert all(expert.fuse_moe is flag for expert in moe.local_experts)


def test_switch_summary_names_the_paths_that_run(monkeypatch):
    turbo = SimpleNamespace(
        enable_primus_turbo=True, use_turbo_grouped_gemm=True, turbo_grouped_gemm_backend="triton"
    )
    monkeypatch.setattr(zm, "_megatron_args", lambda: turbo)
    assert zm.switch_summary(_cfg()).endswith("; experts: Primus Turbo grouped GEMM (backend: triton)")
    assert zm.switch_summary(_cfg(fp8="hybrid")).endswith(
        "; experts: split (Turbo's FP8 grouped GEMM isn't wired up for ZAYA1 yet)"
    )
    monkeypatch.setattr(zm, "_megatron_args", lambda: SimpleNamespace(enable_primus_turbo=True))
    assert zm.switch_summary(_cfg()).endswith("; experts: split")
    line = zm.switch_summary(_cfg(zaya_attn_flash=False, zaya_fuse_norm=False, zaya_router_defer_ar=True))
    assert "zaya_attn_flash=off (dense mask)" in line and "zaya_fuse_norm=off" in line
    assert "zaya_router_defer_ar=on" in line
