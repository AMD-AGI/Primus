# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
The WAN trainer has to hand the model config the precision recipe it was given.

The MXFP4 local linears run their preshuffle-contract check only when
``config.fp4 == "mxfp4"``, so collapsing the YAML value to ``True`` turned the
check off for every WAN MXFP4 run.

The attention layout is threaded the same way, and the trainer logs which
attention kernels the resulting config will run.
"""

from types import SimpleNamespace

import pytest

from primus.backends.megatron.wan_pretrain_trainer import (
    WanPretrainTrainer,
    describe_attention_path,
)


def _build(**params):
    trainer = SimpleNamespace(backend_args=SimpleNamespace(transformer_impl="local", **params))
    return WanPretrainTrainer._build_wan_config_from_yaml(trainer)


def test_fp4_recipe_value_reaches_the_model_config():
    config = _build(fp4="mxfp4", fp4_recipe="mxfp4")

    assert config.fp4 == "mxfp4"
    assert config.fp4_recipe == "mxfp4"


def test_fp4_unset_leaves_the_model_config_without_fp4():
    config = _build()

    assert config.fp4 is None


def test_local_thd_attention_reaches_the_model_config():
    assert _build(local_thd_attention=True).local_thd_attention is True
    assert _build().local_thd_attention is False


@pytest.mark.parametrize(
    "transformer_impl,local_thd,fp8,expected",
    [
        ("transformer_engine", False, False, "TEDotProductAttention"),
        ("local", False, False, "flash_attn_func"),
        ("local", False, True, "flash_attn_fp8_func"),
        ("local", True, False, "flash_attn_varlen_func"),
    ],
)
def test_the_attention_path_names_its_kernel(transformer_impl, local_thd, fp8, expected):
    config = SimpleNamespace(transformer_impl=transformer_impl, local_thd_attention=local_thd)

    assert expected in describe_attention_path(config, fp8_attention=fp8)


def test_local_thd_attention_rejects_fp8_attention():
    """Turbo has no fp8 varlen kernel, so this would otherwise fail on the first forward."""
    config = SimpleNamespace(transformer_impl="local", local_thd_attention=True)

    with pytest.raises(ValueError, match="fp8"):
        describe_attention_path(config, fp8_attention=True)
