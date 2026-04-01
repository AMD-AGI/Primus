###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for MegatronSFTTrainer."""

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from primus.backends.megatron.megatron_sft_trainer import MegatronSFTTrainer
from primus.backends.megatron.sft import runtime as sft_runtime


def _build_trainer(monkeypatch: pytest.MonkeyPatch, model_type: str = "gpt") -> MegatronSFTTrainer:
    """Build trainer with MegatronBaseTrainer init stubbed out."""

    def dummy_init(self, backend_args: Any = None):
        self.backend_args = backend_args

    monkeypatch.setattr(
        "primus.backends.megatron.megatron_base_trainer.MegatronBaseTrainer.__init__",
        dummy_init,
    )
    monkeypatch.setattr(
        "primus.backends.megatron.megatron_sft_trainer.log_rank_0",
        lambda *args, **kwargs: None,
    )

    backend_args = SimpleNamespace(model_type=model_type, lora=None)
    return MegatronSFTTrainer(backend_args=backend_args)


def _stub_sft_runtime_and_forward_step(monkeypatch: pytest.MonkeyPatch):
    """Patch SFT helper factories used by trainer.train()."""
    datasets_provider = object()
    forward_step = object()
    calls = []

    def fake_run_sft_pretrain(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "primus.backends.megatron.sft.runtime.create_sft_datasets_provider",
        lambda: datasets_provider,
    )
    monkeypatch.setattr(
        "primus.backends.megatron.sft.runtime.run_sft_pretrain",
        fake_run_sft_pretrain,
    )
    monkeypatch.setattr(
        "primus.backends.megatron.sft.forward_step.create_sft_forward_step",
        lambda: forward_step,
    )

    return datasets_provider, forward_step, calls


@pytest.mark.parametrize(
    ("model_type", "expected_provider_kwargs"),
    [
        pytest.param("gpt", {}, id="gpt-default-provider"),
        pytest.param("mamba", {"model_type": "mamba"}, id="non-gpt-provider"),
    ],
)
def test_train_selects_model_provider_by_model_type(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    expected_provider_kwargs: dict[str, str],
):
    trainer = _build_trainer(monkeypatch, model_type=model_type)

    pretrain_fn = object()
    monkeypatch.setitem(sys.modules, "megatron.training", types.SimpleNamespace(pretrain=pretrain_fn))
    datasets_provider, forward_step, run_calls = _stub_sft_runtime_and_forward_step(monkeypatch)

    provider_calls = []
    model_provider = object()

    def fake_get_model_provider(**kwargs):
        provider_calls.append(kwargs)
        return model_provider

    monkeypatch.setattr("primus.core.utils.import_utils.get_model_provider", fake_get_model_provider)

    trainer.train()

    assert provider_calls == [expected_provider_kwargs]
    assert len(run_calls) == 1
    assert run_calls[0] == {
        "pretrain_fn": pretrain_fn,
        "datasets_provider": datasets_provider,
        "model_provider": model_provider,
        "forward_step": forward_step,
    }


def test_create_sft_datasets_provider_uses_runtime_args(monkeypatch: pytest.MonkeyPatch):
    calls = []
    fake_args = SimpleNamespace(
        sft_dataset_name="my-org/my-sft-dataset",
        sft_conversation_format="chatml",
        seq_length=2048,
        seed=123,
    )

    training_mod = types.SimpleNamespace(
        get_args=lambda: fake_args,
        get_tokenizer=lambda: "TOKENIZER",
    )
    monkeypatch.setitem(sys.modules, "megatron.training", training_mod)
    monkeypatch.setattr(sft_runtime, "log_rank_0", lambda *args, **kwargs: None)

    def fake_build_train_valid_test_datasets(**kwargs):
        calls.append(kwargs)
        return "TRAIN_DS", "VALID_DS", "TEST_DS"

    monkeypatch.setattr(sft_runtime, "build_train_valid_test_datasets", fake_build_train_valid_test_datasets)

    provider = sft_runtime.create_sft_datasets_provider()
    assert provider.is_distributed is True

    result = provider([100, 20, 10], vp_stage=0)

    assert result == ("TRAIN_DS", "VALID_DS", "TEST_DS")
    assert calls == [
        {
            "dataset_name": "my-org/my-sft-dataset",
            "tokenizer": "TOKENIZER",
            "max_seq_length": 2048,
            "train_val_test_num_samples": [100, 20, 10],
            "formatter": "chatml",
            "seed": 123,
        }
    ]


def test_run_sft_pretrain_uses_new_megatron_entrypoint_signature(monkeypatch: pytest.MonkeyPatch):
    calls = []

    model_type = SimpleNamespace(encoder_or_decoder="ENCODER_OR_DECODER")
    enums_mod = types.SimpleNamespace(ModelType=model_type)
    monkeypatch.setitem(sys.modules, "megatron.core.enums", enums_mod)

    def fake_pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        model_type_arg,
        forward_step,
        process_non_loss_data_func=None,
        extra_args_provider=None,
        args_defaults=None,
        get_embedding_ranks=None,
        get_position_embedding_ranks=None,
        non_loss_data_func=None,
        store=None,
    ):
        raise AssertionError("Expected inprocess wrapper to be used")

    def wrapped_pretrain(*args, **kwargs):
        calls.append((args, kwargs))

    class DummyInprocessRestart:
        @staticmethod
        def maybe_wrap_for_inprocess_restart(fn):
            assert fn is fake_pretrain
            return wrapped_pretrain, "STORE"

    training_mod = types.SimpleNamespace(inprocess_restart=DummyInprocessRestart)
    monkeypatch.setitem(sys.modules, "megatron.training", training_mod)

    datasets_provider = object()
    model_provider = object()
    forward_step = object()

    sft_runtime.run_sft_pretrain(
        pretrain_fn=fake_pretrain,
        datasets_provider=datasets_provider,
        model_provider=model_provider,
        forward_step=forward_step,
    )

    assert len(calls) == 1
    args, kwargs = calls[0]

    assert args == (
        datasets_provider,
        model_provider,
        model_type.encoder_or_decoder,
        forward_step,
    )
    assert kwargs == {
        "args_defaults": {"tokenizer_type": "GPT2BPETokenizer"},
        "extra_args_provider": None,
        "store": "STORE",
    }


def test_run_sft_pretrain_falls_back_to_direct_pretrain_call(monkeypatch: pytest.MonkeyPatch):
    calls = []

    model_type = SimpleNamespace(encoder_or_decoder="ENCODER_OR_DECODER")
    enums_mod = types.SimpleNamespace(ModelType=model_type)
    monkeypatch.setitem(sys.modules, "megatron.core.enums", enums_mod)
    monkeypatch.setattr(sft_runtime, "log_rank_0", lambda *args, **kwargs: None)

    def fake_pretrain(train_valid_test_datasets_provider, model_provider, model_type_arg, forward_step):
        calls.append(
            (
                train_valid_test_datasets_provider,
                model_provider,
                model_type_arg,
                forward_step,
            )
        )

    training_mod = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "megatron.training", training_mod)

    datasets_provider = object()
    model_provider = object()
    forward_step = object()

    sft_runtime.run_sft_pretrain(
        pretrain_fn=fake_pretrain,
        datasets_provider=datasets_provider,
        model_provider=model_provider,
        forward_step=forward_step,
    )

    assert calls == [
        (
            datasets_provider,
            model_provider,
            model_type.encoder_or_decoder,
            forward_step,
        )
    ]
