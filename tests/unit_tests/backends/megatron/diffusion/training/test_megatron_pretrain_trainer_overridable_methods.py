# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for overridable methods in MegatronPretrainTrainer.

Tests verify that get_forward_step() and get_datasets_provider() methods
can be overridden by subclasses while maintaining backward compatibility.
"""

import sys
import types
from types import SimpleNamespace
from typing import Any, List, Tuple

import pytest
import torch

from primus.backends.megatron.megatron_pretrain_trainer import MegatronPretrainTrainer

# train() imports the real megatron.core/training stack (transitively primus_turbo/TE), which
# initializes CUDA at import time. Gate the two tests that call train() so they run only on GPU.
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="train() imports real megatron.core/training which initialize CUDA at import",
)


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


class TestMegatronPretrainTrainerOverridableMethods:
    """Tests for overridable methods in MegatronPretrainTrainer."""

    @requires_gpu
    def test_backward_compatibility_gpt_training_still_works(self, monkeypatch: pytest.MonkeyPatch):
        """Test that existing GPT training still works (backward compatibility)."""
        trainer = _build_trainer(monkeypatch)

        calls: List[Tuple[tuple, dict]] = []

        # Use the real megatron modules that train() imports; patch only the integration seams.
        import megatron.core.pipeline_parallel as mpp
        import megatron.training
        import megatron.training.inprocess_restart as inprocess_restart
        import megatron.training.training as mt_training
        from megatron.core.enums import ModelType

        def fake_pretrain(*args, **kwargs):
            raise AssertionError("fake_pretrain was called directly; expected wrapped_pretrain to be used")

        def wrapped_pretrain(*args, store=None, **kwargs):
            calls.append((args, {"store": store, **kwargs}))

        def fake_maybe_wrap(fn):
            assert fn is fake_pretrain
            return wrapped_pretrain, "STORE"

        monkeypatch.setattr(megatron.training, "pretrain", fake_pretrain)
        monkeypatch.setattr(inprocess_restart, "maybe_wrap_for_inprocess_restart", fake_maybe_wrap)

        # train() reassigns get_forward_backward_func on both real modules; snapshot both through
        # monkeypatch so they are restored and the mutation does not leak into later tests.
        monkeypatch.setattr(mpp, "get_forward_backward_func", mpp.get_forward_backward_func)
        monkeypatch.setattr(mt_training, "get_forward_backward_func", mt_training.get_forward_backward_func)

        model_provider = object()
        monkeypatch.setattr(
            "primus.core.utils.import_utils.get_model_provider",
            lambda *args, **kwargs: model_provider,
        )

        # pretrain_gpt is a Megatron example script, not importable here -> mock it via sys.modules.
        train_valid_test_datasets_provider = SimpleNamespace(is_distributed=False)
        pretrain_gpt_mod = types.SimpleNamespace(
            forward_step="FORWARD_STEP",
            train_valid_test_datasets_provider=train_valid_test_datasets_provider,
        )
        monkeypatch.setitem(sys.modules, "pretrain_gpt", pretrain_gpt_mod)

        # Execute training
        trainer.train()

        # Verify backward compatibility: should work exactly as before
        assert train_valid_test_datasets_provider.is_distributed is True
        assert len(calls) == 1
        (args, kwargs) = calls[0]

        # Positional arguments should match expected GPT training pattern
        assert args[0] is train_valid_test_datasets_provider
        assert args[1] is model_provider
        assert args[2] is ModelType.encoder_or_decoder
        assert args[3] == "FORWARD_STEP"

        assert kwargs == {"store": "STORE"}
