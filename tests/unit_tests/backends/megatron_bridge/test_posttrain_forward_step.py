###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for MegatronBridgePosttrainTrainer forward step selection."""

import inspect
import sys
import types
from types import SimpleNamespace

import pytest


def _gpt_forward_step(state, data_iterator, model, return_schedule_plan: bool = False):
    return None if data_iterator is None else next(data_iterator)


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
        assert trainer_mod._is_vlm_model(SimpleNamespace(vision_config=object()))

    def test_vision_config_none_is_text_only(self, trainer_mod):
        assert not trainer_mod._is_vlm_model(SimpleNamespace(vision_config=None))

    def test_freeze_vision_model_without_vision_config(self, trainer_mod):
        assert trainer_mod._is_vlm_model(SimpleNamespace(freeze_vision_model=False))


class TestSelectForwardStep:
    def test_text_only_model_uses_gpt_step(self, trainer_mod, stub_step_modules):
        model_cfg = SimpleNamespace(num_layers=2, hidden_size=64)
        assert trainer_mod._select_forward_step(model_cfg).__wrapped__ is _gpt_forward_step

    def test_vlm_model_uses_vlm_step(self, trainer_mod, stub_step_modules):
        model_cfg = SimpleNamespace(vision_config=object(), freeze_vision_model=False)
        assert trainer_mod._select_forward_step(model_cfg) is _vlm_forward_step


class TestDefaultAttentionMask:
    @pytest.fixture
    def gpt_step(self, trainer_mod, stub_step_modules):
        return trainer_mod._select_forward_step(SimpleNamespace(num_layers=2))

    def test_adds_missing_attention_mask(self, gpt_step):
        # Chat SFT collate output with get_attention_mask_from_fusion=True.
        batch = gpt_step(None, iter([{"tokens": 1, "labels": 2}]), None)
        assert batch == {"tokens": 1, "labels": 2, "attention_mask": None}

    def test_keeps_existing_attention_mask(self, gpt_step):
        mask = object()
        assert gpt_step(None, iter([{"attention_mask": mask}]), None)["attention_mask"] is mask

    def test_passes_none_iterator_through(self, gpt_step):
        assert gpt_step(None, None, None) is None

    def test_keeps_signature_for_state_injection(self, gpt_step):
        assert inspect.signature(gpt_step) == inspect.signature(_gpt_forward_step)


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
        assert [f.__wrapped__ for f in finetune_calls] == [_gpt_forward_step]

    def test_vlm_model_finetunes_with_vlm_step(self, trainer_mod, finetune_calls):
        self._make_trainer(trainer_mod, SimpleNamespace(vision_config=object())).train()
        assert finetune_calls == [_vlm_forward_step]
