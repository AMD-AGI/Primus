###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling model build: full size on the meta device, tiny on the GPU.

The P0 gate is :func:`test_full_size_meta_build_matches_checkpoint`: the
full Inkling-Small model builds without allocating memory and every
parameter maps one-to-one onto the released checkpoint.
"""

from __future__ import annotations

import copy
from functools import partial

import pytest
import torch

from primus.backends.megatron.core.models.inkling.inkling_builders import (
    build_inkling_model,
    inkling_builder,
    inkling_config_from_model_yaml,
    megatron_args_from_model_yaml,
    model_provider,
)
from primus.backends.megatron.core.models.inkling.inkling_hf_mapping import (
    InklingGeometry,
    module_tensor_shapes,
    release_to_megatron,
)
from tests.unit_tests.backends.megatron.inkling.inkling_test_utils import (
    RELEASE_VOCAB,
    numel,
    preset,
    random_release_state,
    release_config,
    release_meta_tensors,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="TE modules need a GPU")


@pytest.fixture(autouse=True)
def _parallel_state(init_parallel_state):
    yield


def _converted_shapes(config, release):
    geo = InklingGeometry.from_megatron_config(config)
    return {k: tuple(v.shape) for k, v in release_to_megatron(release, geo).items()}


def test_full_size_meta_build_matches_checkpoint():
    config = inkling_config_from_model_yaml(preset("inkling_small"))
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    model = build_inkling_model(config, vocab_size=RELEASE_VOCAB, max_sequence_length=4096, meta=True)
    grown = torch.cuda.memory_allocated() - before
    assert grown < 1 << 20, f"meta build allocated {grown} bytes of GPU memory"
    assert {p.device.type for p in model.parameters()} == {"meta"}

    release = release_meta_tensors()
    assert module_tensor_shapes(model) == _converted_shapes(config, release)
    assert (
        numel(module_tensor_shapes(model).values())
        == numel(t.shape for t in release.values())
        == 263_655_199_274
    )


def test_proxy_meta_build_matches_checkpoint_slice():
    config = inkling_config_from_model_yaml(preset("inkling_small_12L"))
    model = build_inkling_model(config, vocab_size=RELEASE_VOCAB, max_sequence_length=4096, meta=True)
    release = {
        k: v
        for k, v in release_meta_tensors().items()
        if not k.startswith("model.llm.layers.") or int(k.split(".")[3]) < 12
    }
    assert module_tensor_shapes(model) == _converted_shapes(config, release)
    assert numel(module_tensor_shapes(model).values()) == 67_517_036_044


def test_tiny_model_loads_converted_checkpoint():
    config = inkling_config_from_model_yaml(preset("inkling_small_tiny"))
    model = build_inkling_model(config, vocab_size=512, max_sequence_length=4096)
    release = random_release_state(config, vocab=512, dtype=torch.bfloat16)
    converted = release_to_megatron(release, InklingGeometry.from_megatron_config(config))

    # the converter's on-disk form: TE extra state as None, loaded strictly
    state = dict(converted)
    state.update({k: None for k in model.state_dict() if k.endswith("_extra_state")})
    model.load_state_dict(state, strict=True)

    loaded = model.state_dict()
    for key, value in converted.items():
        torch.testing.assert_close(loaded[key].float().cpu(), value.float(), rtol=0, atol=0, msg=key)


def test_tiny_bf16_training_step_tracks_fp32():
    """The bf16 training dtype path runs end to end and stays close to fp32."""
    fp32 = inkling_config_from_model_yaml(
        preset("inkling_small_tiny"), params_dtype=torch.float32, gradient_accumulation_fusion=False
    )
    bf16 = inkling_config_from_model_yaml(preset("inkling_small_tiny"), gradient_accumulation_fusion=False)
    ref = build_inkling_model(fp32, vocab_size=512, max_sequence_length=4096).cuda()
    model = build_inkling_model(bf16, vocab_size=512, max_sequence_length=4096).cuda()
    model.load_state_dict(ref.state_dict(), strict=True)
    model.train()
    ref.train()

    ids = torch.randint(0, 512, (2, 96), generator=torch.Generator().manual_seed(0)).cuda()
    labels = torch.roll(ids, -1, dims=1)
    losses = []
    for m in (ref, model):
        loss = m(ids, None, None, labels=labels).float().mean()
        loss.backward()
        losses.append(loss.item())
    torch.testing.assert_close(losses[1], losses[0], rtol=2e-2, atol=0)
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    router = model.decoder.layers[1].mlp.router
    assert router.expert_bias.dtype == torch.float32 and router.local_tokens_per_expert.sum() > 0


def test_init_matches_reference_statistics():
    """``InklingPreTrainedModel._init_weights``: every weight -- projections,
    output ones and LM head included, experts, relative projection, short
    convs, router rows -- is normal(0, initializer_range); norms and
    ``global_scale`` are 1 and the correction bias is 0."""
    config = inkling_config_from_model_yaml(
        preset("inkling_small_tiny"), params_dtype=torch.float32, gradient_accumulation_fusion=False
    )
    std = release_config()["text_config"].get("initializer_range", 0.02)
    assert config.init_method_std == std
    model = build_inkling_model(config, vocab_size=512, max_sequence_length=4096).cuda()

    checked = 0
    for name, p in model.named_parameters():
        p = p.detach().float()
        if "norm" in name or name.endswith("global_scale"):
            assert torch.equal(p, torch.ones_like(p)), f"{name} is not initialised to 1"
            continue
        n = p.numel()
        assert abs(p.mean().item()) < 5 * std / n**0.5, f"{name}: mean {p.mean().item():.2e}"
        tol = max(0.05, 5 / (2 * n) ** 0.5)
        assert abs(p.std().item() / std - 1) < tol, f"{name}: std {p.std().item():.4f}, expected {std}"
        checked += 1
    assert checked > 0
    for name, b in model.named_buffers():
        if name.endswith("expert_bias"):
            assert torch.count_nonzero(b) == 0, name


def _checkpointed(function) -> str:
    if type(function).__name__ == "InklingCoreAttention":
        return "core_attn"
    if isinstance(function, partial) and isinstance(function.func, torch.nn.Module):
        return "mlp"
    return type(function).__name__


@pytest.mark.parametrize(
    "modules",
    [["core_attn"], ["layernorm"], ["mlp"], ["core_attn", "layernorm", "mlp"]],
    ids=["core_attn", "layernorm", "mlp", "all"],
)
@pytest.mark.parametrize("backend", ["eager", "flex"])
def test_selective_recompute(backend, modules, monkeypatch):
    """Selective recompute checkpoints what ``recompute_modules`` names and leaves loss and
    gradients unchanged. The tiny model has one dense and one MoE layer, so ``core_attn`` is
    checkpointed twice, ``mlp`` once (dense layers only) and ``layernorm`` four times."""
    from megatron.core import tensor_parallel

    calls, norm_calls, norm_recomputes = [], [], []
    checkpoint = tensor_parallel.checkpoint
    norm_checkpoint = tensor_parallel.CheckpointWithoutOutput.checkpoint
    norm_recompute = tensor_parallel.CheckpointWithoutOutput._recompute

    def counting_checkpoint(function, *args):
        calls.append(_checkpointed(function))
        return checkpoint(function, *args)

    def counting_norm_checkpoint(self, function, *args):
        norm_calls.append(type(function).__name__)
        return norm_checkpoint(self, function, *args)

    def counting_norm_recompute(self, grad):
        norm_recomputes.append(self.ctx is not None)
        return norm_recompute(self, grad)

    monkeypatch.setattr(tensor_parallel, "checkpoint", counting_checkpoint)
    monkeypatch.setattr(tensor_parallel.CheckpointWithoutOutput, "checkpoint", counting_norm_checkpoint)
    monkeypatch.setattr(tensor_parallel.CheckpointWithoutOutput, "_recompute", counting_norm_recompute)
    results = []
    for recompute in (None, "selective"):
        config = inkling_config_from_model_yaml(
            preset("inkling_small_tiny"),
            params_dtype=torch.float32,
            gradient_accumulation_fusion=False,
            inkling_attention_backend=backend,
            recompute_granularity=recompute,
            recompute_modules=modules,
        )
        torch.manual_seed(0)
        model = build_inkling_model(config, vocab_size=512, max_sequence_length=4096).cuda()
        if results:
            model.load_state_dict(results[0][2], strict=True)
        model.train()
        ids = torch.randint(0, 512, (2, 160), generator=torch.Generator().manual_seed(0)).cuda()
        loss = model(ids, None, None, labels=torch.roll(ids, -1, dims=1)).float().mean()
        loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        results.append((loss.item(), grads, model.state_dict()))

    expected = ["core_attn", "core_attn"] if "core_attn" in modules else []
    if "mlp" in modules:
        expected.append("mlp")
    assert sorted(calls) == sorted(expected)
    norms = 4 if "layernorm" in modules else 0
    assert len(norm_calls) == norms and set(norm_calls) <= {"RMSNorm", "TENorm"}
    assert norm_recomputes == [True] * norms, "every dropped norm output must be rebuilt in backward"
    (loss_off, grads_off, _), (loss_on, grads_on, _) = results
    torch.testing.assert_close(loss_on, loss_off, rtol=1e-6, atol=0)
    assert grads_on.keys() == grads_off.keys()
    assert grads_on["decoder.layers.0.self_attention.rel_proj"].abs().sum() > 0
    for name, grad in grads_off.items():
        torch.testing.assert_close(grads_on[name], grad, rtol=1e-5, atol=1e-6, msg=name)


def test_deferred_lm_head_wgrad_matches_immediate():
    """``defer_embedding_wgrad_compute`` on a last pipeline stage: the LM head buffers its input
    and output gradient, and the schedule's drain gives the immediate weight gradient."""
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel import schedules

    config = inkling_config_from_model_yaml(
        preset("inkling_small_tiny"), params_dtype=torch.float32, gradient_accumulation_fusion=True
    )
    gen = torch.Generator().manual_seed(0)
    # Two microbatches: Megatron's drain needs at least two buffered entries, as any PP > 1 run has.
    micro = [
        (torch.randn(96, 1, config.hidden_size, generator=gen), torch.randint(0, 512, (1, 96), generator=gen))
        for _ in range(2)
    ]
    grads, state = [], None
    for defer in (False, True):
        cfg = copy.copy(config)
        # Megatron only allows this flag with PP > 1; the LM-head buffers do not depend on PP.
        cfg.defer_embedding_wgrad_compute = defer
        # A last stage holds no embedding, so the drain's weight lookup returns the LM head.
        model = build_inkling_model(cfg, vocab_size=512, max_sequence_length=4096, pre_process=False).cuda()
        if state is None:
            state = model.state_dict()
        model.load_state_dict(state, strict=True)
        model.train()
        for p in model.parameters():
            p.main_grad = torch.zeros_like(p, dtype=torch.float32)
        head = model.output_layer.weight

        embedding_module = schedules.clear_embedding_activation_buffer(cfg, model, is_last_stage=True)
        for hidden, labels in micro:
            model.set_input_tensor(hidden.cuda().requires_grad_())
            model(None, None, None, labels=labels.cuda()).float().mean().backward()
        if defer:
            assert embedding_module is model
            assert len(model.embedding_activation_buffer) == len(model.grad_output_buffer) == 2
            assert head.main_grad.count_nonzero() == 0
            schedules.finish_embedding_wgrad_compute(
                cfg, embedding_module, True, parallel_state.get_tensor_model_parallel_group()
            )
            assert not model.embedding_activation_buffer and not model.grad_output_buffer
        else:
            assert embedding_module is None and model.embedding_activation_buffer is None
        grads.append(head.main_grad.clone())

    assert grads[0].count_nonzero() > 0
    torch.testing.assert_close(grads[1], grads[0], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("keep_fp32", ["0", "1"])
def test_fp32_activations_global_scale_storage(keep_fp32, monkeypatch):
    """``global_scale`` is stored in fp32 only with the flag and the ``keep_in_fp32`` contract."""
    monkeypatch.setenv("PRIMUS_V4_KEEP_FP32", keep_fp32)
    for flag in (False, True):
        config = inkling_config_from_model_yaml(
            preset("inkling_small_tiny"), gradient_accumulation_fusion=False, inkling_fp32_activations=flag
        )
        model = build_inkling_model(config, vocab_size=512, max_sequence_length=4096)
        expected = torch.float32 if flag and keep_fp32 == "1" else torch.bfloat16
        for layer in (model.decoder.layers[0].mlp, model.decoder.layers[1].mlp.router):
            assert layer.global_scale.dtype == expected
            assert layer.cuda().bfloat16().global_scale.dtype == expected


def test_trainer_entry_points_build_inkling():
    from primus.core.utils.import_utils import get_model_provider

    provider = get_model_provider("inkling")
    assert isinstance(provider, partial)
    assert provider.func is model_provider and provider.args == (inkling_builder,)

    args = megatron_args_from_model_yaml(preset("inkling_small_tiny"), padded_vocab_size=512)
    model = inkling_builder(args, pre_process=True, post_process=True)
    assert model.output_layer.weight.shape == (512, 256)
    assert model.decoder.layers[1].mlp.router.shared_weight.shape == (2, 256)
