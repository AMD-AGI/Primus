# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
The WAN trainer has to hand the model config the precision recipe it was given.

The MXFP4 local linears run their preshuffle-contract check only when
``config.fp4 == "mxfp4"``, so collapsing the YAML value to ``True`` turned the
check off for every WAN MXFP4 run.
"""

from types import SimpleNamespace

from primus.backends.megatron.wan_pretrain_trainer import WanPretrainTrainer


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
