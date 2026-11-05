###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU tests for smollm3_norm_epsilon_patches: a SmolLM3 run whose layernorm_epsilon and
norm_epsilon differ stops before the trainer is built; matching values and other models pass."""

from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches import smollm3_norm_epsilon_patches as eps
from primus.core.patches import run_patches


def _run(model, **params):
    module_config = SimpleNamespace(params=SimpleNamespace(**params))
    return run_patches(
        backend="megatron",
        phase="build_args",
        model_name=model,
        extra={"module_config": module_config, "backend_args": module_config.params},
        enabled_ids=[eps.KEY],
    )


def test_matching_values_pass():
    assert _run("smollm3_3B_Base.yaml", layernorm_epsilon=1e-6, norm_epsilon=1e-6) == 1


@pytest.mark.parametrize("override", [dict(layernorm_epsilon=1e-5), dict(norm_epsilon=1e-5)])
def test_one_key_overridden_stops_the_run(override):
    params = {"layernorm_epsilon": 1e-6, "norm_epsilon": 1e-6, **override}
    with pytest.raises(SystemExit, match="layernorm_epsilon=.*norm_epsilon="):
        _run("smollm3_3B_Base.yaml", **params)


def test_renamed_preset_is_found_by_its_tokenizer():
    params = dict(tokenizer_model="HuggingFaceTB/SmolLM3-3B-Base", layernorm_epsilon=1e-6, norm_epsilon=1e-5)
    with pytest.raises(SystemExit):
        _run("my_3b.yaml", **params)
    with pytest.raises(SystemExit):
        _run(None, **params)


def test_other_models_are_not_checked():
    assert _run("llama3.1_8B.yaml", layernorm_epsilon=1e-5, norm_epsilon=1e-6) == 0


def test_the_smollm3_preset_and_recipes_agree():
    """The preset as merged (llama3_base's 1e-5 under norm_epsilon is overridden)."""
    from primus.core.config.preset_loader import PresetLoader

    preset = PresetLoader.load("smollm3_3B_Base.yaml", "megatron", config_type="models")
    eps.check_epsilons(preset["layernorm_epsilon"], preset["norm_epsilon"])
    assert float(preset["layernorm_epsilon"]) == 1e-6
