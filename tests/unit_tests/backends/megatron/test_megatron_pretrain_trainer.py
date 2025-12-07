###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for MegatronPretrainTrainer.

Focus:
    - Construction behavior (delegates to MegatronBaseTrainer and initializes fields)
    - run_train wiring to Megatron pretrain entrypoint
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
    def dummy_init(self, primus_config: Any, module_config: Any, backend_args: Any):
        self.primus_config = primus_config
        self.module_config = module_config
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

    primus_config = SimpleNamespace()
    module_config = SimpleNamespace(model="gpt2", framework="megatron")
    backend_args = SimpleNamespace()

    return MegatronPretrainTrainer(primus_config, module_config, backend_args)


class TestMegatronPretrainTrainer:
    """Tests for MegatronPretrainTrainer wiring and behavior."""

    def test_init_sets_expected_attributes(self, monkeypatch: pytest.MonkeyPatch):
        trainer = _build_trainer(monkeypatch)

        # MegatronBaseTrainer stub should have stored these attributes.
        assert trainer.primus_config is not None
        assert trainer.module_config is not None
        assert trainer.backend_args is not None

        # Pretrain trainer adds training components placeholders.
        assert hasattr(trainer, "model")
        assert hasattr(trainer, "optimizer")
        assert hasattr(trainer, "opt_param_scheduler")
        assert trainer.model is None
        assert trainer.optimizer is None
        assert trainer.opt_param_scheduler is None

    def test_run_train_invokes_megatron_pretrain_with_expected_arguments(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        trainer = _build_trainer(monkeypatch)

        # Prepare fake Megatron and helper modules used inside run_train().
        calls: List[Tuple[tuple, dict]] = []

        # 1) megatron.core.enums.ModelType
        model_type = SimpleNamespace(encoder_or_decoder="ENCODER_OR_DECODER")
        enums_mod = types.SimpleNamespace(ModelType=model_type)
        monkeypatch.setitem(sys.modules, "megatron.core.enums", enums_mod)

        # 2) megatron.training with inprocess_restart and pretrain
        def fake_pretrain(*args, **kwargs):
            # This should not be called directly; wrapped_pretrain is used instead.
            raise AssertionError("fake_pretrain was called directly; expected wrapped_pretrain to be used")

        def wrapped_pretrain(*args, **kwargs):
            calls.append((args, kwargs))

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
        import_utils_mod = types.SimpleNamespace(
            get_model_provider=lambda: model_provider,
        )
        monkeypatch.setitem(sys.modules, "primus.core.utils.import_utils", import_utils_mod)

        # Execute training wiring.
        trainer.run_train()

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
