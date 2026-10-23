###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for MegatronBridgePosttrainTrainer forward step selection.

Text-only models must use ``gpt_step``: ``vlm_step`` pads tokens without
updating ``cu_seqlens`` and crashes packed SFT with ``pad_to_max_length=False``.

The ``megatron.bridge.training.*`` modules are replaced with stubs so the tests
run on CPU without Megatron-Bridge installed. Import is deferred into a fixture
(like test_config_utils.py) so the package's torch-pulling __init__ doesn't run
at collection time under the coverage C tracer.
"""

import sys
import types
from types import SimpleNamespace

import pytest


def _gpt_forward_step(*args, **kwargs):
    pass


def _vlm_forward_step(*args, **kwargs):
    pass


@pytest.fixture
def trainer_mod():
    import primus.backends.megatron_bridge.megatron_bridge_posttrain_trainer as m

    return m


@pytest.fixture
def stub_step_modules(monkeypatch):
    for name, fn in (
        ("megatron.bridge.training.gpt_step", _gpt_forward_step),
        ("megatron.bridge.training.vlm_step", _vlm_forward_step),
    ):
        mod = types.ModuleType(name)
        mod.forward_step = fn
        monkeypatch.setitem(sys.modules, name, mod)


class TestIsVlmModel:
    def test_text_only_provider(self, trainer_mod):
        assert not trainer_mod._is_vlm_model(SimpleNamespace(num_layers=2, hidden_size=64))

    def test_vision_config_provider(self, trainer_mod):
        # Qwen2.5-VL / Qwen3-VL / Gemma3-VL / GLM-4.5V style
        assert trainer_mod._is_vlm_model(SimpleNamespace(vision_config=object(), freeze_vision_model=False))

    def test_vision_config_none_is_text_only(self, trainer_mod):
        assert not trainer_mod._is_vlm_model(SimpleNamespace(vision_config=None))

    def test_freeze_vision_model_without_vision_config(self, trainer_mod):
        # Ministral3 / Nemotron-VL style
        assert trainer_mod._is_vlm_model(SimpleNamespace(freeze_vision_model=False))


class TestSelectForwardStep:
    def test_text_only_model_uses_gpt_step(self, trainer_mod, stub_step_modules):
        model_cfg = SimpleNamespace(num_layers=2, hidden_size=64)
        assert trainer_mod._select_forward_step(model_cfg) is _gpt_forward_step

    def test_vlm_model_uses_vlm_step(self, trainer_mod, stub_step_modules):
        model_cfg = SimpleNamespace(vision_config=object(), freeze_vision_model=False)
        assert trainer_mod._select_forward_step(model_cfg) is _vlm_forward_step


class TestPosttrainTrain:
    @pytest.fixture
    def finetune_calls(self, trainer_mod, stub_step_modules, monkeypatch):
        calls = []
        config_mod = types.ModuleType("megatron.bridge.training.config")
        config_mod.runtime_config_update = lambda cfg: None
        finetune_mod = types.ModuleType("megatron.bridge.training.finetune")
        finetune_mod.finetune = lambda cfg, forward_step_func: calls.append(forward_step_func)
        monkeypatch.setitem(sys.modules, "megatron.bridge.training.config", config_mod)
        monkeypatch.setitem(sys.modules, "megatron.bridge.training.finetune", finetune_mod)
        monkeypatch.setattr(trainer_mod, "log_rank_0", lambda *a, **k: None)
        monkeypatch.setattr(trainer_mod, "log_dict_aligned", lambda *a, **k: None)
        return calls

    @staticmethod
    def _make_trainer(trainer_mod, model_cfg):
        trainer = object.__new__(trainer_mod.MegatronBridgePosttrainTrainer)
        trainer.cfg_container = SimpleNamespace(model=model_cfg, to_dict=lambda: {})
        return trainer

    def test_text_only_model_finetunes_with_gpt_step(self, trainer_mod, finetune_calls):
        self._make_trainer(trainer_mod, SimpleNamespace(num_layers=2)).train()
        assert finetune_calls == [_gpt_forward_step]

    def test_vlm_model_finetunes_with_vlm_step(self, trainer_mod, finetune_calls):
        self._make_trainer(trainer_mod, SimpleNamespace(vision_config=object())).train()
        assert finetune_calls == [_vlm_forward_step]
