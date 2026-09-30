# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Sensitive layers (``sensitive_layers_enabled``) must stay out of FP8/FP4.

Megatron keeps a first/last layer in bf16 by giving it ``nullcontext()``, which
cannot switch off an autocast already active around it. Flux used to wrap the
whole transformer in one FP8/FP4 context on the TransformerEngine spec, so the
sensitive layers were quantized like every other layer. These pin the per-layer
contexts, the config rejections and, on GPU, what TE's linears see.
"""

import contextlib
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.transformer.diffusion_transformer_block import (
    DiffusionTransformerBlock,
    uses_per_layer_quantization_context,
)

SENSITIVE = dict(
    sensitive_layers_enabled=True,
    sensitive_layers_start=1,
    sensitive_layers_end=1,
    sensitive_layer_precision="bf16",
)


def _te_config(**overrides):
    params = dict(
        num_joint_layers=2,
        num_single_layers=2,
        transformer_impl="transformer_engine",
        fp8="hybrid",
        fp8_recipe="tensorwise",
    )
    params.update(overrides)
    return FluxConfig.flux_535m(**params)


# ----------------------------------------------------------------------------
# When quantization is entered per layer
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transformer_impl,fp8,fp4,first_last,expected",
    [
        ("transformer_engine", "hybrid", None, True, True),
        ("transformer_engine", None, "mxfp4", True, True),
        ("transformer_engine", "hybrid", None, False, False),
        ("transformer_engine", None, None, True, False),
        ("local", "hybrid", None, True, False),
    ],
)
def test_per_layer_quantization_only_with_bf16_first_last_layers(
    transformer_impl, fp8, fp4, first_last, expected
):
    config = SimpleNamespace(
        transformer_impl=transformer_impl, fp8=fp8, fp4=fp4, first_last_layers_bf16=first_last
    )

    assert uses_per_layer_quantization_context(config) is expected


def test_sensitive_layers_on_the_te_spec_select_per_layer_contexts():
    assert uses_per_layer_quantization_context(_te_config(**SENSITIVE))
    assert not uses_per_layer_quantization_context(_te_config())


@pytest.mark.parametrize("fp8,fp4", [("hybrid", None), (None, "mxfp4")])
def test_the_model_level_context_steps_aside_for_sensitive_layers(monkeypatch, fp8, fp4):
    import megatron.core.fp4_utils as fp4_utils
    import megatron.core.fp8_utils as fp8_utils

    from primus.backends.megatron.core.models.diffusion.flux.model import Flux

    built = []
    monkeypatch.setattr(fp8_utils, "get_fp8_context", lambda *a, **k: built.append("fp8"))
    monkeypatch.setattr(fp4_utils, "get_fp4_context", lambda *a, **k: built.append("fp4"))
    model = SimpleNamespace(
        config=SimpleNamespace(
            transformer_impl="transformer_engine", fp8=fp8, fp4=fp4, first_last_layers_bf16=True
        )
    )

    assert isinstance(Flux.get_fp8_context(model), contextlib.nullcontext)
    assert built == []


# ----------------------------------------------------------------------------
# DiffusionTransformerBlock's non-recompute loop
# ----------------------------------------------------------------------------


class _Layer:
    def __init__(self, layer_number, events):
        self.layer_number = layer_number
        self.events = events

    def __call__(self, hidden_states, **kwargs):
        self.events.append(("run", self.layer_number))
        return hidden_states


QUANTIZATION = {"fp8": dict(fp8="hybrid", fp4=None), "mxfp4": dict(fp8=None, fp4="mxfp4")}


def _run_block(monkeypatch, quantization, first_last_layers_bf16):
    import megatron.core.fp4_utils as fp4_utils
    import megatron.core.fp8_utils as fp8_utils

    events = []

    def fake_context(kind):
        @contextlib.contextmanager
        def context(config, layer_no=-1):
            events.append((kind, layer_no))
            yield

        return context

    monkeypatch.setattr(fp8_utils, "get_fp8_context", fake_context("fp8"))
    monkeypatch.setattr(fp4_utils, "get_fp4_context", fake_context("fp4"))
    block = SimpleNamespace(
        config=SimpleNamespace(
            model_type="flux",
            recompute_granularity=None,
            transformer_impl="transformer_engine",
            first_last_layers_bf16=first_last_layers_bf16,
            **QUANTIZATION[quantization],
        ),
        training=True,
        layers=[_Layer(n, events) for n in (1, 2, 3)],
    )
    DiffusionTransformerBlock.forward(block, torch.zeros(2), attention_mask=None, timestep_emb=torch.zeros(1))
    return events


@pytest.mark.parametrize("quantization,kind", [("fp8", "fp8"), ("mxfp4", "fp4")])
def test_the_loop_enters_each_layers_own_context(monkeypatch, quantization, kind):
    events = _run_block(monkeypatch, quantization, first_last_layers_bf16=True)

    assert events == [(kind, 0), ("run", 1), (kind, 1), ("run", 2), (kind, 2), ("run", 3)]


@pytest.mark.parametrize("quantization", ["fp8", "mxfp4"])
def test_the_loop_enters_no_context_without_sensitive_layers(monkeypatch, quantization):
    """The model-level context still covers every layer there."""
    events = _run_block(monkeypatch, quantization, first_last_layers_bf16=False)

    assert events == [("run", 1), ("run", 2), ("run", 3)]


# ----------------------------------------------------------------------------
# Config rejections
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["stack", "double_stack", "single_stack", "full_dit"])
def test_stack_runners_are_rejected_with_sensitive_layers(strategy):
    with pytest.raises(ValueError, match="bypass the per-layer"):
        _te_config(torch_compile_strategy=strategy, **SENSITIVE)


@pytest.mark.parametrize("strategy", ["per_block", "whole_model"])
def test_block_strategies_build_with_sensitive_layers(strategy):
    assert _te_config(torch_compile_strategy=strategy, **SENSITIVE).first_last_layers_bf16


def test_stack_runners_still_build_without_sensitive_layers():
    assert _te_config(torch_compile_strategy="full_dit").torch_compile_strategy == "full_dit"


def test_non_bf16_sensitive_precision_is_rejected_on_the_te_spec():
    with pytest.raises(ValueError, match="can only keep sensitive layers in bf16"):
        _te_config(**{**SENSITIVE, "sensitive_layer_precision": "tw_fp8"})


# ----------------------------------------------------------------------------
# GPU: what TE's linears see while each layer runs
# ----------------------------------------------------------------------------


def _fp8_state_manager():
    try:
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    except ImportError:
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    return FP8GlobalStateManager


def _record_fp8_per_layer(model):
    state = _fp8_state_manager()
    seen = defaultdict(set)
    for index, layer in enumerate(model.transformer.layers):
        for module in layer.modules():
            name = type(module).__name__
            if name.startswith("TE") and "Linear" in name:
                module.register_forward_pre_hook(
                    lambda m, a, index=index: seen[index].add(bool(state.is_fp8_enabled()))
                )
    return seen


def _flux_inputs():
    from primus.backends.megatron.core.models.diffusion.flux.utils import (
        generate_image_position_ids,
        pack_latents,
    )

    batch, height, width, text = 2, 16, 16, 8
    img = pack_latents(torch.randn(batch, 16, height, width)).transpose(0, 1).cuda().to(torch.bfloat16)
    txt = torch.randn(text, batch, 4096).cuda().to(torch.bfloat16)
    y = torch.randn(batch, 768).cuda().to(torch.bfloat16)
    timesteps = torch.rand(batch).cuda()
    img_ids = generate_image_position_ids(batch, height, width, device="cuda")
    txt_ids = torch.zeros(batch, text, 3).cuda()
    return img, txt, y, timesteps, img_ids, txt_ids


RECOMPUTE = dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)


class TestTESpecFP8PerLayer:
    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pytest.importorskip("transformer_engine")

    def _fp8_seen(self, recompute, sensitive):
        from primus.backends.megatron.core.models.diffusion.flux.model import Flux

        overrides = {**(RECOMPUTE if recompute else {}), **(SENSITIVE if sensitive else {})}
        model = Flux(_te_config(**overrides)).cuda().to(torch.bfloat16)
        seen = _record_fp8_per_layer(model)
        model.train()
        output = model(*_flux_inputs())
        if recompute:
            output.float().sum().backward()
        return {index: sorted(states) for index, states in seen.items()}

    @pytest.mark.parametrize("recompute", [False, True], ids=["no_recompute", "full_recompute"])
    def test_sensitive_layers_run_in_bf16(self, recompute):
        assert self._fp8_seen(recompute, sensitive=True) == {0: [False], 1: [True], 2: [True], 3: [False]}

    @pytest.mark.parametrize("recompute", [False, True], ids=["no_recompute", "full_recompute"])
    def test_every_layer_runs_in_fp8_without_sensitive_layers(self, recompute):
        assert self._fp8_seen(recompute, sensitive=False) == {0: [True], 1: [True], 2: [True], 3: [True]}
