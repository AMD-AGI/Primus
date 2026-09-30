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
import torch.nn as nn

from primus.backends.megatron.wan_pretrain_trainer import (
    WanPretrainTrainer,
    block_linear_types,
    describe_attention_path,
    describe_linear_precision,
)


def _build(**params):
    params.setdefault("transformer_impl", "local")
    trainer = SimpleNamespace(backend_args=SimpleNamespace(**params))
    return WanPretrainTrainer._build_wan_config_from_yaml(trainer)


def test_fp4_recipe_value_reaches_the_model_config():
    config = _build(fp4="mxfp4", fp4_recipe="mxfp4")

    assert config.fp4 == "mxfp4"
    assert config.fp4_recipe == "mxfp4"


def test_fp4_unset_leaves_the_model_config_without_fp4():
    config = _build()

    assert config.fp4 is None


def test_fp8_settings_reach_the_model_config():
    config = _build(
        fp8="e4m3",
        fp8_recipe="tensorwise",
        fp8_amax_history_len=16,
        fp8_amax_compute_algo="max",
        fp8_scaling_strategy="delayed",
        fp8_force_nt_layout=True,
        fp8_reduce_amax=True,
    )

    assert config.fp8 == "e4m3"
    assert config.fp8_recipe == "tensorwise"
    assert config.fp8_amax_history_len == 16
    assert config.fp8_amax_compute_algo == "max"
    assert config.fp8_scaling_strategy == "delayed"
    assert config.fp8_force_nt_layout is True
    assert config.fp8_reduce_amax is True
    # The Float8 local linears raise at construction if it is on.
    assert config.gradient_accumulation_fusion is False


def test_fp8_unset_leaves_the_model_config_in_bf16():
    config = _build()

    assert config.fp8 is None
    assert describe_linear_precision(config) == "bf16"


def test_fp8_on_the_te_path_is_rejected():
    """The TE linears would need an FP8 autocast the WAN forward never enters."""
    with pytest.raises(ValueError, match="fp8 on WAN requires transformer_impl='local'"):
        _build(transformer_impl="transformer_engine", fp8="e4m3", fp8_recipe="tensorwise")


_PARAM_SHARDING = {"use_megatron_fsdp": True, "data_parallel_sharding_strategy": "optim_grads_params"}


@pytest.mark.parametrize(
    "scaling", [{"fp8_recipe": "delayed"}, {"fp8_recipe": "tensorwise", "fp8_scaling_strategy": "delayed"}]
)
def test_delayed_fp8_with_megatron_fsdp_param_sharding_is_rejected(scaling):
    """The delayed-scaling update reads weights that the sharding has emptied."""
    with pytest.raises(ValueError, match="delayed FP8 scaling does not work with Megatron-FSDP"):
        _build(fp8="hybrid", **scaling, **_PARAM_SHARDING)


def test_fp8_that_avoids_delayed_scaling_or_param_sharding_builds():
    assert _build(fp8="hybrid", fp8_recipe="tensorwise", **_PARAM_SHARDING).fp8 == "hybrid"
    assert _build(fp8="hybrid", fp8_recipe="delayed", use_megatron_fsdp=False).fp8 == "hybrid"


def test_fp8_on_the_local_path_builds_the_turbo_float8_linears():
    pytest.importorskip("primus_turbo")
    from primus.backends.megatron.core.models.diffusion.common.backend_resolution import (
        resolve_diffusion_backend,
    )

    backend, sensitive_backend = resolve_diffusion_backend(_build(fp8="e4m3", fp8_recipe="tensorwise"))

    assert type(backend).__name__ == "PrimusTurboFloat8LocalSpecProvider"
    assert sensitive_backend is None


def test_sensitive_layer_settings_reach_the_model_config():
    config = _build(
        fp4="mxfp4",
        fp4_recipe="mxfp4",
        num_dit_layers=4,
        sensitive_layers_enabled=True,
        sensitive_layers_start=1,
        sensitive_layers_end=2,
        sensitive_layer_precision="bf16",
    )

    assert config.sensitive_layers_enabled is True
    assert (config.sensitive_layers_start, config.sensitive_layers_end) == (1, 2)
    assert config.sensitive_layer_precision == "bf16"
    assert describe_linear_precision(config) == "MXFP4 (mxfp4), first 1 / last 2 blocks in bf16"


def test_sensitive_layers_without_fp4_are_rejected():
    """Only the MXFP4 backend builds the first/last blocks at another precision."""
    with pytest.raises(ValueError, match="needs fp4 set"):
        _build(
            fp8="e4m3",
            fp8_recipe="tensorwise",
            num_dit_layers=4,
            sensitive_layers_enabled=True,
            sensitive_layers_start=1,
            sensitive_layers_end=1,
        )


def test_the_linear_precision_names_the_fp8_recipe():
    config = _build(fp8="e4m3", fp8_recipe="tensorwise")

    assert describe_linear_precision(config) == "FP8 e4m3 (tensorwise recipe)"


def test_block_linear_types_lists_the_built_linear_classes():
    class Float8ColumnParallelLinear(nn.Linear):
        pass

    class Float8RowParallelLinear(nn.Linear):
        pass

    model = nn.Sequential(
        Float8ColumnParallelLinear(4, 4),
        Float8RowParallelLinear(4, 4),
        Float8ColumnParallelLinear(4, 4),
        nn.Linear(4, 4),
    )

    assert block_linear_types(model) == ["Float8ColumnParallelLinear", "Float8RowParallelLinear"]


def test_context_parallelism_is_rejected():
    """Nothing splits the sequence, yet the local attention would switch to its CP kernel."""
    with pytest.raises(ValueError, match="set context_parallel_size: 1"):
        _build(context_parallel_size=2)


def test_context_parallel_size_one_builds():
    assert _build(context_parallel_size=1).context_parallel_size == 1


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
