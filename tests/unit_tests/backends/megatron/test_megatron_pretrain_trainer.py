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
from primus.core.utils.import_utils import _PRIMUS_MODEL_PROVIDERS

# The types the trainer routes to pretrain_gpt: every key of the provider table. Which types the
# table holds is pinned once, in tests/unit_tests/core/utils/test_import_utils.py.
_PRIMUS_MODEL_TYPES = sorted(_PRIMUS_MODEL_PROVIDERS)


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


class _MegatronStubs:
    """Stand-ins for the Megatron modules train() imports, shared by every wiring test: the
    ModelType enum, ``megatron.training`` (pretrain + an inprocess_restart wrapper that records
    each call), ``megatron.training.training``, ``megatron.core.pipeline_parallel`` and the
    pretrain_gpt entrypoint."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, datasets_provider: Any):
        self.calls: List[Tuple[tuple, dict]] = []
        self.model_type = SimpleNamespace(encoder_or_decoder="ENCODER_OR_DECODER")
        self.datasets_provider = datasets_provider
        self.requested: List[Any] = []

        def fake_pretrain(*args, **kwargs):
            raise AssertionError("fake_pretrain was called directly; expected wrapped_pretrain to be used")

        def wrapped_pretrain(*args, store=None, **kwargs):
            # the trainer forwards `store` only because this signature names it
            self.calls.append((args, {"store": store, **kwargs}))

        class DummyInprocessRestart:
            @staticmethod
            def maybe_wrap_for_inprocess_restart(fn):
                assert fn is fake_pretrain
                return wrapped_pretrain, "STORE"

        training_pkg = types.ModuleType("megatron.training")
        training_pkg.__path__ = []  # a package, so `import megatron.training.training` resolves
        training_pkg.inprocess_restart = DummyInprocessRestart
        training_pkg.pretrain = fake_pretrain
        training_pkg.get_args = lambda: SimpleNamespace(dump_pp_data=False)
        training_mod = types.ModuleType("megatron.training.training")
        training_pkg.training = training_mod
        pp_mod = types.ModuleType("megatron.core.pipeline_parallel")
        pp_mod.get_forward_backward_func = lambda *a, **k: "SCHEDULE"
        self.pp_mod = pp_mod

        def get_model_provider(model_type="gpt"):
            self.requested.append(model_type)
            return f"PROVIDER[{model_type}]"

        for name, mod in {
            "megatron.core.enums": SimpleNamespace(ModelType=self.model_type),
            "megatron.core.pipeline_parallel": pp_mod,
            "megatron.training": training_pkg,
            "megatron.training.training": training_mod,
            "pretrain_gpt": SimpleNamespace(
                forward_step="GPT_FORWARD_STEP", train_valid_test_datasets_provider=datasets_provider
            ),
        }.items():
            monkeypatch.setitem(sys.modules, name, mod)
        # the trainer imports get_model_provider at module level
        monkeypatch.setattr(
            "primus.backends.megatron.megatron_pretrain_trainer.get_model_provider", get_model_provider
        )
        # When megatron.core is not imported yet (a CPU host, or a test run alone), stand in for
        # the parent packages too, so `import megatron.core.pipeline_parallel as mpp` binds the
        # stubs above instead of importing (or failing to import) the real package.
        if "megatron.core" not in sys.modules:
            root, core = types.ModuleType("megatron"), types.ModuleType("megatron.core")
            root.__path__, core.__path__ = [], []
            root.core, root.training = core, training_pkg
            core.pipeline_parallel, core.enums = pp_mod, sys.modules["megatron.core.enums"]
            monkeypatch.setitem(sys.modules, "megatron", root)
            monkeypatch.setitem(sys.modules, "megatron.core", core)

    def single_call(self):
        assert len(self.calls) == 1
        return self.calls[0]


def _function_provider():
    """The upstream shape: a plain function, without the is_distributed attribute."""

    def train_valid_test_datasets_provider(train_val_test_num_samples):
        raise AssertionError("dataset provider should not be executed in the wiring test")

    return train_valid_test_datasets_provider


class TestMegatronPretrainTrainer:
    """Tests for MegatronPretrainTrainer wiring and behavior."""

    def test_init_sets_expected_attributes(self, monkeypatch: pytest.MonkeyPatch):
        trainer = _build_trainer(monkeypatch)
        assert trainer.backend_args is not None

    def test_train_invokes_megatron_pretrain_with_expected_arguments(self, monkeypatch: pytest.MonkeyPatch):
        trainer = _build_trainer(monkeypatch)
        provider = SimpleNamespace(is_distributed=False)
        stubs = _MegatronStubs(monkeypatch, provider)

        trainer.train()

        assert provider.is_distributed is True
        args, kwargs = stubs.single_call()
        assert args == (provider, "PROVIDER[gpt]", stubs.model_type.encoder_or_decoder, "GPT_FORWARD_STEP")
        assert kwargs == {"store": "STORE"}
        assert stubs.requested == ["gpt"]

    def test_train_marks_function_provider_distributed_when_attribute_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        trainer = _build_trainer(monkeypatch)
        provider = _function_provider()
        stubs = _MegatronStubs(monkeypatch, provider)

        trainer.train()

        assert getattr(provider, "is_distributed", None) is True
        args, kwargs = stubs.single_call()
        assert args[0] is provider and args[2] is stubs.model_type.encoder_or_decoder
        assert kwargs == {"store": "STORE"}

    @pytest.mark.parametrize("model_type_name", _PRIMUS_MODEL_TYPES)
    def test_gpt_data_model_types_reuse_pretrain_gpt_with_their_own_provider(
        self, monkeypatch: pytest.MonkeyPatch, model_type_name: str
    ):
        """Each Primus-owned causal-LM type trains with pretrain_gpt's forward_step and datasets
        provider (marked distributed), its own model provider, ModelType.encoder_or_decoder and
        the inprocess_restart-wrapped pretrain (with its store)."""
        trainer = _build_trainer(monkeypatch)
        trainer.backend_args.model_type = model_type_name
        provider = _function_provider()
        stubs = _MegatronStubs(monkeypatch, provider)

        def fallback():
            raise AssertionError("the fallback forward_step / datasets provider was used")

        monkeypatch.setattr(trainer, "get_forward_step", fallback, raising=False)
        monkeypatch.setattr(trainer, "get_datasets_provider", fallback, raising=False)

        trainer.train()

        assert stubs.requested == [model_type_name]
        assert getattr(provider, "is_distributed", None) is True
        args, kwargs = stubs.single_call()
        assert args[0] is provider
        assert args[1] == f"PROVIDER[{model_type_name}]"
        assert args[2] is stubs.model_type.encoder_or_decoder
        assert args[3] == "GPT_FORWARD_STEP"
        assert kwargs == {"store": "STORE"}
