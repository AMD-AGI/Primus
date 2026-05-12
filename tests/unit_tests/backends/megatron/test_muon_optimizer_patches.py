###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import dataclasses
import sys
import types
from types import SimpleNamespace

import pytest

from primus.core.patches.context import PatchContext


def _install_fake_megatron_training(monkeypatch: pytest.MonkeyPatch):
    megatron_mod = types.ModuleType("megatron")
    megatron_mod.__path__ = []

    training_pkg = types.ModuleType("megatron.training")
    training_pkg.__path__ = []

    training_mod = types.ModuleType("megatron.training.training")
    original_calls = []

    def original_get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=None,
        use_gloo_process_groups=True,
        pg_collection=None,
        dump_param_to_param_group_map=None,
    ):
        original_calls.append(
            {
                "config": config,
                "model_chunks": model_chunks,
                "config_overrides": config_overrides,
                "use_gloo_process_groups": use_gloo_process_groups,
                "pg_collection": pg_collection,
                "dump_param_to_param_group_map": dump_param_to_param_group_map,
            }
        )
        return "original-result"

    training_mod.get_megatron_optimizer = original_get_megatron_optimizer
    training_pkg.training = training_mod
    megatron_mod.training = training_pkg

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.training", training_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", training_mod)

    return training_mod, original_calls


def _install_fake_muon_dependencies(monkeypatch: pytest.MonkeyPatch):
    muon_calls = []

    moun_mod = types.ModuleType("primus.backends.megatron.core.optimizer.moun")

    def fake_get_megatron_muon_optimizer(
        config,
        model_chunks,
        config_overrides=None,
        use_gloo_process_groups=True,
        layer_wise_distributed_optimizer=False,
        pg_collection=None,
        dump_param_to_param_group_map=None,
    ):
        muon_calls.append(
            {
                "config": config,
                "model_chunks": model_chunks,
                "config_overrides": config_overrides,
                "use_gloo_process_groups": use_gloo_process_groups,
                "layer_wise_distributed_optimizer": layer_wise_distributed_optimizer,
                "pg_collection": pg_collection,
                "dump_param_to_param_group_map": dump_param_to_param_group_map,
            }
        )
        return "muon-result"

    moun_mod.get_megatron_muon_optimizer = fake_get_megatron_muon_optimizer

    moun_config_mod = types.ModuleType("primus.backends.megatron.core.optimizer.moun_optimizer_config")

    @dataclasses.dataclass
    class FakeMounOptimizerConfig:
        optimizer: str = "muon"
        muon_tp_mode: str = "blockwise"
        timers: object = None

    moun_config_mod.MounOptimizerConfig = FakeMounOptimizerConfig

    monkeypatch.setitem(
        sys.modules,
        "primus.backends.megatron.core.optimizer.moun",
        moun_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "primus.backends.megatron.core.optimizer.moun_optimizer_config",
        moun_config_mod,
    )

    return muon_calls


def _call_get_megatron_optimizer(
    optimizer_fn,
    config,
    model_chunks,
    config_overrides,
    use_gloo_process_groups,
    pg_collection,
    dump_param_to_param_group_map,
    positional_config_overrides,
):
    args = [config, model_chunks]
    kwargs = {
        "use_gloo_process_groups": use_gloo_process_groups,
        "pg_collection": pg_collection,
        "dump_param_to_param_group_map": dump_param_to_param_group_map,
    }
    if positional_config_overrides:
        args.append(config_overrides)
    else:
        kwargs["config_overrides"] = config_overrides
    return optimizer_fn(*args, **kwargs)


@pytest.mark.parametrize(
    "positional_config_overrides",
    [False, True],
    ids=["keyword_config_overrides", "positional_config_overrides"],
)
def test_patch_get_megatron_optimizer_muon_matches_runtime_signature(
    monkeypatch: pytest.MonkeyPatch,
    positional_config_overrides: bool,
):
    training_mod, original_calls = _install_fake_megatron_training(monkeypatch)
    muon_calls = _install_fake_muon_dependencies(monkeypatch)

    monkeypatch.setattr(
        "primus.backends.megatron.patches.muon_optimizer_patches.log_rank_0",
        lambda *args, **kwargs: None,
    )

    from primus.backends.megatron.patches.muon_optimizer_patches import (
        patch_get_megatron_optimizer_muon,
    )

    original_fn = training_mod.get_megatron_optimizer
    ctx = PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"backend_args": SimpleNamespace(muon_tp_mode="blockwise")},
    )

    patch_get_megatron_optimizer_muon(ctx)

    assert training_mod.get_megatron_optimizer is not original_fn

    adam_config = SimpleNamespace(optimizer="adam", timers="adam-timer")
    result = _call_get_megatron_optimizer(
        training_mod.get_megatron_optimizer,
        adam_config,
        ["chunk-0"],
        config_overrides={"dense": "group"},
        use_gloo_process_groups=False,
        pg_collection="pg-0",
        dump_param_to_param_group_map="dense-map",
        positional_config_overrides=positional_config_overrides,
    )

    assert result == "original-result"
    assert original_calls == [
        {
            "config": adam_config,
            "model_chunks": ["chunk-0"],
            "config_overrides": {"dense": "group"},
            "use_gloo_process_groups": False,
            "pg_collection": "pg-0",
            "dump_param_to_param_group_map": "dense-map",
        }
    ]

    muon_config = SimpleNamespace(optimizer="muon-dist", timers="muon-timer")
    result = _call_get_megatron_optimizer(
        training_mod.get_megatron_optimizer,
        muon_config,
        ["chunk-1"],
        config_overrides={"sparse": "group"},
        use_gloo_process_groups=False,
        pg_collection="pg-1",
        dump_param_to_param_group_map="muon-map",
        positional_config_overrides=positional_config_overrides,
    )

    assert result == "muon-result"
    assert len(muon_calls) == 1
    muon_call = muon_calls[0]
    assert {key: value for key, value in muon_call.items() if key != "config"} == {
        "model_chunks": ["chunk-1"],
        "config_overrides": {"sparse": "group"},
        "use_gloo_process_groups": False,
        "layer_wise_distributed_optimizer": True,
        "pg_collection": "pg-1",
        "dump_param_to_param_group_map": "muon-map",
    }
    assert muon_call["config"].timers == "muon-timer"
    assert muon_call["config"].muon_tp_mode == "blockwise"
