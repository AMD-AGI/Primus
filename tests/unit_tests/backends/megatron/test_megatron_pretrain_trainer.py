###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for MegatronPretrainTrainer.

Focus:
    - Construction behavior (delegates to MegatronBaseTrainer and initializes fields)
    - train wiring to Megatron pretrain entrypoint
"""

import sys
import types
from types import SimpleNamespace
from typing import Any, List, Tuple

import pytest

from primus.backends.megatron.megatron_pretrain_trainer import MegatronPretrainTrainer


def _build_trainer(monkeypatch: pytest.MonkeyPatch) -> MegatronPretrainTrainer:
    """Helper to build MegatronPretrainTrainer with a stubbed MegatronBaseTrainer."""

    # Stub out MegatronBaseTrainer.__init__ to avoid real Megatron imports/patching.
    def dummy_init(self, backend_args: Any = None):
        self.backend_args = backend_args

    monkeypatch.setattr(
        "primus.backends.megatron.megatron_base_trainer.MegatronBaseTrainer.__init__",
        dummy_init,
    )

    # Silence logging from the trainer module.
    monkeypatch.setattr(
        "primus.backends.megatron.megatron_pretrain_trainer.log_rank_0",
        lambda *args, **kwargs: None,
    )

    backend_args = SimpleNamespace()

    return MegatronPretrainTrainer(backend_args=backend_args)


class TestMegatronPretrainTrainer:
    """Tests for MegatronPretrainTrainer wiring and behavior."""

    def test_init_sets_expected_attributes(self, monkeypatch: pytest.MonkeyPatch):
        trainer = _build_trainer(monkeypatch)

        # MegatronBaseTrainer stub should have stored backend_args.
        assert trainer.backend_args is not None

    def test_train_invokes_megatron_pretrain_with_expected_arguments(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        trainer = _build_trainer(monkeypatch)

        # Prepare fake Megatron and helper modules used inside train().
        calls: List[Tuple[tuple, dict]] = []

        # 1) megatron.core.enums.ModelType
        model_type = SimpleNamespace(encoder_or_decoder="ENCODER_OR_DECODER")
        enums_mod = types.SimpleNamespace(ModelType=model_type)
        monkeypatch.setitem(sys.modules, "megatron.core.enums", enums_mod)

        # 2) megatron.training with inprocess_restart and pretrain
        def fake_pretrain(*args, **kwargs):
            # This should not be called directly; wrapped_pretrain is used instead.
            raise AssertionError("fake_pretrain was called directly; expected wrapped_pretrain to be used")

        def wrapped_pretrain(*args, store=None, **kwargs):
            # In newer Megatron versions, the inprocess_restart wrapper can accept `store=...`.
            # Our trainer only forwards `store` when the wrapped callable explicitly supports it.
            calls.append((args, {"store": store, **kwargs}))

        class DummyInprocessRestart:
            @staticmethod
            def maybe_wrap_for_inprocess_restart(fn):
                # Ensure we received the original pretrain function.
                assert fn is fake_pretrain
                return wrapped_pretrain, "STORE"

        training_mod = types.SimpleNamespace(
            inprocess_restart=DummyInprocessRestart,
            pretrain=fake_pretrain,
        )
        monkeypatch.setitem(sys.modules, "megatron.training", training_mod)

        # 3) pretrain_gpt with forward_step and train_valid_test_datasets_provider
        train_valid_test_datasets_provider = SimpleNamespace(is_distributed=False)
        pretrain_gpt_mod = types.SimpleNamespace(
            forward_step="FORWARD_STEP",
            train_valid_test_datasets_provider=train_valid_test_datasets_provider,
        )
        monkeypatch.setitem(sys.modules, "pretrain_gpt", pretrain_gpt_mod)

        # 4) primus.core.utils.import_utils.get_model_provider
        model_provider = object()
        provider_calls = []

        def fake_get_model_provider(**kwargs):
            provider_calls.append(kwargs)
            return model_provider

        import_utils_mod = types.SimpleNamespace(get_model_provider=fake_get_model_provider)
        monkeypatch.setitem(sys.modules, "primus.core.utils.import_utils", import_utils_mod)

        # Execute training wiring.
        trainer.train()

        assert provider_calls == [{"model_type": "gpt"}]

        # Train datasets provider should be marked distributed.
        assert train_valid_test_datasets_provider.is_distributed is True

        # wrapped_pretrain should have been called exactly once.
        assert len(calls) == 1
        (args, kwargs) = calls[0]

        # Positional arguments:
        assert args[0] is train_valid_test_datasets_provider
        assert args[1] is model_provider
        assert args[2] is model_type.encoder_or_decoder
        assert args[3] == "FORWARD_STEP"

        # Keyword arguments:
        assert kwargs == {"store": "STORE"}

    def test_train_marks_function_provider_distributed_when_attribute_is_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        trainer = _build_trainer(monkeypatch)

        calls: List[Tuple[tuple, dict]] = []

        model_type = SimpleNamespace(encoder_or_decoder="ENCODER_OR_DECODER")
        enums_mod = types.SimpleNamespace(ModelType=model_type)
        monkeypatch.setitem(sys.modules, "megatron.core.enums", enums_mod)

        def fake_pretrain(*args, **kwargs):
            raise AssertionError("fake_pretrain was called directly; expected wrapped_pretrain to be used")

        def wrapped_pretrain(*args, store=None, **kwargs):
            calls.append((args, {"store": store, **kwargs}))

        class DummyInprocessRestart:
            @staticmethod
            def maybe_wrap_for_inprocess_restart(fn):
                assert fn is fake_pretrain
                return wrapped_pretrain, "STORE"

        training_mod = types.SimpleNamespace(
            inprocess_restart=DummyInprocessRestart,
            pretrain=fake_pretrain,
        )
        monkeypatch.setitem(sys.modules, "megatron.training", training_mod)

        # This matches upstream more closely: the provider is a plain function and only gets the
        # marker attribute when the wrapper sets it explicitly.
        def train_valid_test_datasets_provider(train_val_test_num_samples):
            raise AssertionError("dataset provider should not be executed in the wiring test")

        pretrain_gpt_mod = types.SimpleNamespace(
            forward_step="FORWARD_STEP",
            train_valid_test_datasets_provider=train_valid_test_datasets_provider,
        )
        monkeypatch.setitem(sys.modules, "pretrain_gpt", pretrain_gpt_mod)

        model_provider = object()
        import_utils_mod = types.SimpleNamespace(get_model_provider=lambda **_: model_provider)
        monkeypatch.setitem(sys.modules, "primus.core.utils.import_utils", import_utils_mod)

        trainer.train()

        assert getattr(train_valid_test_datasets_provider, "is_distributed", None) is True
        assert len(calls) == 1
        (args, kwargs) = calls[0]
        assert args[0] is train_valid_test_datasets_provider
        assert args[1] is model_provider
        assert args[2] is model_type.encoder_or_decoder
        assert args[3] == "FORWARD_STEP"
        assert kwargs == {"store": "STORE"}

    def test_non_gpt_family_takes_components_from_its_entrypoint_marked_distributed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        trainer = _build_trainer(monkeypatch)

        def forward_step():
            pass

        def train_valid_test_datasets_provider():
            pass

        entrypoint = types.ModuleType("pretrain_mamba")
        entrypoint.forward_step = forward_step
        entrypoint.train_valid_test_datasets_provider = train_valid_test_datasets_provider
        monkeypatch.setitem(sys.modules, "pretrain_mamba", entrypoint)
        monkeypatch.setattr(trainer, "get_forward_step", lambda: pytest.fail("mamba trains like GPT"))

        components = trainer._get_training_components("mamba")

        assert components == (forward_step, train_valid_test_datasets_provider)
        assert train_valid_test_datasets_provider.is_distributed is True

    @pytest.mark.parametrize("model_type", ["gpt", "kimi_k3", "inkling", "diffusion_model", None])
    def test_gpt_trained_families_use_the_overridable_components(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_type,
    ):
        trainer = _build_trainer(monkeypatch)
        monkeypatch.setattr(trainer, "get_forward_step", lambda: "FORWARD_STEP")
        monkeypatch.setattr(trainer, "get_datasets_provider", lambda: "DATASETS_PROVIDER")

        assert trainer._get_training_components(model_type) == ("FORWARD_STEP", "DATASETS_PROVIDER")
