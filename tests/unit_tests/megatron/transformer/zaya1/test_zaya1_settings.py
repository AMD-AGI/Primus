###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The kernel-path config fields: their defaults, the values they take, and the modules that read them."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_zaya1_math import _cfg  # noqa: E402  (pytest puts this dir on sys.path)
from test_zaya1_model import _tiny_config  # noqa: E402

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
    "zaya_sort_dispatch",
    "zaya_router_defer_ar",
)
_CHUNKED_CE = ("fused_ce_mode", "fused_ce_chunk_tokens", "fused_ce_compile")


def test_every_path_is_off_by_default():
    base = Path(primus.__file__).parent / "configs" / "models" / "megatron" / "zaya1_base.yaml"
    values = yaml.safe_load(base.read_text())
    for name in _PATHS + _CHUNKED_CE:
        assert values[name] == getattr(Zaya1TransformerConfig, name), name
    for name in _PATHS + ("fused_ce_compile",):
        assert getattr(Zaya1TransformerConfig, name) is False, name
    # Unset rather than 0, which would also run the FLA runtime-knob patch.
    assert values["fused_ce_mode"] is None
    # The unit tests' stand-in configs have none of the fields.
    assert not any(zm._path_on(SimpleNamespace(), name) for name in _PATHS)


def test_on_off_fields_take_only_true_or_false():
    # A CLI override such as --zaya_fuse_norm off arrives as the string "off".
    for name in _PATHS + ("fused_ce_compile",):
        assert getattr(_tiny_config(**{name: 1}), name) is True
        assert getattr(_tiny_config(**{name: 0}), name) is False
        for value in ("off", "false", "", None, 2, 1.0):
            with pytest.raises(ValueError, match=f"{name} must be true or false"):
                _tiny_config(**{name: value})


def test_modules_read_the_fields_when_built():
    for flag in (True, False):
        stack = zm.ZayaStack(_cfg(num_layers=2, **{name: flag for name in _PATHS}))
        cca, moe = stack.layers[0].self_attn, stack.layers[1].zaya_block
        assert stack.fuse_norm is stack.attn_flash is stack.layers[0].fuse_norm is flag
        assert cca.attn_flash is cca.fuse_cca is cca._conv_gemm_ok is flag
        assert moe.fuse_moe is moe.router.fuse_moe is moe.router.defer_bias_update is flag
        assert moe.sort_dispatch is flag
        assert all(expert.fuse_moe is flag for expert in moe.local_experts)
    # The GEMM convolutions cover kernel 2 only; CCA falls back to nn.Conv1d.
    assert zm.CCA(_cfg(zaya_conv_gemm=True, cca_time1=3))._conv_gemm_ok is False
