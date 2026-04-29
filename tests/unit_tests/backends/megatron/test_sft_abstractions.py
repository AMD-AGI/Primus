###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for Megatron-local SFT abstractions."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.sft.formatters import AlpacaFormatter
from primus.backends.megatron.sft.forward_step import create_sft_forward_step
from primus.backends.megatron.sft.preprocessing import (
    normalize_sft_sample,
    tokenize_formatted_sft_sample,
)


class MockTokenizer:
    """Tokenizer stub compatible with Megatron SFT helpers."""

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return [hash(token) % 10000 for token in tokens]


def _install_fake_megatron_training(monkeypatch: pytest.MonkeyPatch, **args) -> None:
    """Install a minimal megatron.training stub for forward_step tests."""
    megatron_mod = types.ModuleType("megatron")
    training_mod = types.ModuleType("megatron.training")
    training_mod.get_args = lambda: SimpleNamespace(**args)
    megatron_mod.training = training_mod

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.training", training_mod)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_normalize_sft_sample_supports_common_single_turn_fields():
    sample = normalize_sft_sample(
        {
            "prompt": "Explain GPUs",
            "output": "GPUs accelerate parallel workloads.",
            "input": "Use one sentence",
            "system": "Be concise",
        }
    )

    assert sample.instruction == "Explain GPUs"
    assert sample.response == "GPUs accelerate parallel workloads."
    assert sample.input_text == "Use one sentence"
    assert sample.system_prompt == "Be concise"
    assert not sample.is_multi_turn


def test_tokenize_formatted_sft_sample_marks_only_supervised_segments():
    formatter = AlpacaFormatter()
    formatted = formatter.format_sample(
        normalize_sft_sample(
            {
                "instruction": "What is SFT?",
                "response": "SFT is supervised fine-tuning.",
            }
        )
    )

    input_ids, labels, loss_mask = tokenize_formatted_sft_sample(
        formatted_sample=formatted,
        tokenizer=MockTokenizer(),
        max_seq_length=512,
    )

    assert torch.equal(input_ids, labels)
    assert loss_mask.sum().item() > 0
    assert loss_mask.sum().item() < len(loss_mask)


def test_forward_step_passes_new_megatron_kwargs_and_computes_masked_loss(monkeypatch):
    _install_fake_megatron_training(monkeypatch, use_legacy_models=False)
    forward_step = create_sft_forward_step()
    packed_seq_params = object()

    batch = {
        "input_ids": torch.tensor([1, 2, 3]),
        "labels": torch.tensor([4, 5, 6]),
        "loss_mask": torch.tensor([0.0, 1.0, 1.0]),
        "packed_seq_params": packed_seq_params,
    }

    class RecordingModel:
        def __init__(self):
            self.calls = []

        def __call__(self, tokens, position_ids, attention_mask, **kwargs):
            self.calls.append(
                {
                    "tokens": tokens,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "kwargs": kwargs,
                }
            )
            return torch.tensor([[1.0, 2.0, 3.0]])

    model = RecordingModel()

    output_tensor, loss_fn = forward_step(iter([batch]), model)
    model_call = model.calls[0]

    assert torch.equal(model_call["tokens"], torch.tensor([[1, 2, 3]]))
    assert torch.equal(model_call["position_ids"], torch.tensor([[0, 1, 2]]))
    assert model_call["attention_mask"] is None
    assert torch.equal(model_call["kwargs"]["labels"], torch.tensor([[4, 5, 6]]))
    assert torch.equal(model_call["kwargs"]["loss_mask"], torch.tensor([[0.0, 1.0, 1.0]]))
    assert model_call["kwargs"]["packed_seq_params"] is packed_seq_params

    loss, num_tokens, metrics = loss_fn(output_tensor)

    assert loss.item() == pytest.approx(5.0)
    assert num_tokens.item() == 2
    assert torch.allclose(metrics["lm loss"], torch.tensor([5.0, 2.0]))


def test_forward_step_returns_schedule_plan_for_new_megatron(monkeypatch):
    _install_fake_megatron_training(monkeypatch, use_legacy_models=False)
    forward_step = create_sft_forward_step()
    expected_schedule_plan = object()

    batch = {
        "input_ids": torch.tensor([7, 8]),
        "labels": torch.tensor([9, 10]),
        "loss_mask": torch.tensor([1.0, 0.0]),
    }

    class ScheduleModel:
        def __init__(self):
            self.schedule_calls = []

        def build_schedule_plan(self, tokens, position_ids, attention_mask, **kwargs):
            self.schedule_calls.append(
                {
                    "tokens": tokens,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "kwargs": kwargs,
                }
            )
            return expected_schedule_plan

    model = ScheduleModel()

    schedule_plan, loss_fn = forward_step(iter([batch]), model, return_schedule_plan=True)
    model_call = model.schedule_calls[0]

    assert schedule_plan is expected_schedule_plan
    assert torch.equal(model_call["tokens"], torch.tensor([[7, 8]]))
    assert torch.equal(model_call["position_ids"], torch.tensor([[0, 1]]))
    assert model_call["attention_mask"] is None
    assert torch.equal(model_call["kwargs"]["labels"], torch.tensor([[9, 10]]))
    assert torch.equal(model_call["kwargs"]["loss_mask"], torch.tensor([[1.0, 0.0]]))

    loss, num_tokens, metrics = loss_fn(torch.tensor([[2.0, 4.0]]))

    assert loss.item() == pytest.approx(2.0)
    assert num_tokens.item() == 1
    assert torch.allclose(metrics["lm loss"], torch.tensor([2.0, 1.0]))


def test_forward_step_requires_model_schedule_builder(monkeypatch):
    _install_fake_megatron_training(monkeypatch, use_legacy_models=False)
    forward_step = create_sft_forward_step()

    batch = {
        "input_ids": torch.tensor([1, 2]),
        "labels": torch.tensor([1, 2]),
        "loss_mask": torch.tensor([1.0, 1.0]),
    }

    with pytest.raises(AttributeError, match="build_schedule_plan"):
        forward_step(iter([batch]), model=object(), return_schedule_plan=True)


def test_forward_step_supports_legacy_megatron_models(monkeypatch):
    _install_fake_megatron_training(monkeypatch, use_legacy_models=True)
    forward_step = create_sft_forward_step()

    batch = {
        "input_ids": torch.tensor([3, 4, 5]),
        "labels": torch.tensor([6, 7, 8]),
        "loss_mask": torch.tensor([1.0, 1.0, 0.0]),
        "packed_seq_params": object(),
    }

    class LegacyModel:
        def __init__(self):
            self.calls = []

        def __call__(self, tokens, position_ids, attention_mask, **kwargs):
            self.calls.append(
                {
                    "tokens": tokens,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "kwargs": kwargs,
                }
            )
            return torch.tensor([[3.0, 4.0, 5.0]])

    model = LegacyModel()

    output_tensor, loss_fn = forward_step(iter([batch]), model, return_schedule_plan=True)
    model_call = model.calls[0]

    assert torch.equal(model_call["tokens"], torch.tensor([[3, 4, 5]]))
    assert torch.equal(model_call["position_ids"], torch.tensor([[0, 1, 2]]))
    assert model_call["attention_mask"] is None
    assert torch.equal(model_call["kwargs"]["labels"], torch.tensor([[6, 7, 8]]))
    assert "loss_mask" not in model_call["kwargs"]
    assert "packed_seq_params" not in model_call["kwargs"]

    loss, num_tokens, metrics = loss_fn(output_tensor)

    assert loss.item() == pytest.approx(7.0)
    assert num_tokens.item() == 2
    assert torch.allclose(metrics["lm loss"], torch.tensor([7.0, 2.0]))
