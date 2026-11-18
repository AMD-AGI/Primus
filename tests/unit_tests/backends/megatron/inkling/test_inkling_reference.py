###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The official transformers Inkling against the importer and the Megatron model.

Skips unless ``transformers.models.inkling`` imports; see ``inkling_reference.py``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import torch

from primus.backends.megatron.core.models.inkling.inkling_builders import (
    build_inkling_model,
    inkling_config_from_model_yaml,
)
from primus.backends.megatron.core.models.inkling.inkling_hf_mapping import (
    InklingGeometry,
    megatron_to_release,
    module_tensor_shapes,
    release_to_megatron,
    transformers_to_release,
)
from tests.unit_tests.backends.megatron.inkling.inkling_reference import (
    build_reference_model,
    hf_text_config,
    megatron_forward_backward,
    reference_forward_backward,
    reference_modeling,
    reference_weights_for_megatron,
    release_to_transformers,
    transformers_inkling_available,
)
from tests.unit_tests.backends.megatron.inkling.inkling_test_utils import (
    preset,
    release_schema,
    release_shapes,
)

pytestmark = pytest.mark.skipif(
    not transformers_inkling_available(), reason="transformers.models.inkling not importable"
)

VOCAB = 512


@pytest.fixture(scope="module")
def tiny():
    return inkling_config_from_model_yaml(preset("inkling_small_tiny"), params_dtype=torch.float32)


def test_reference_layout_matches_released_checkpoint():
    """transformers' module layout, mapped back to release names, is the real checkpoint (text part)."""
    from transformers.models.inkling.modeling_inkling import InklingForCausalLM

    config = inkling_config_from_model_yaml(preset("inkling_small"))
    with torch.device("meta"):
        model = InklingForCausalLM(hf_text_config(config, 201024))
    mapped = {k: tuple(v.shape) for k, v in transformers_to_release(model.state_dict()).items()}
    real = {k: shape for k, (_, shape) in release_shapes().items() if k.startswith("model.llm.")}
    assert mapped == real


def test_reference_weights_round_trip(tiny):
    model = build_reference_model(tiny, VOCAB)
    release = transformers_to_release(model.state_dict())
    assert {k: tuple(v.shape) for k, v in release.items()} == {
        k: shape for k, (shape, _) in release_schema(tiny, VOCAB).items()
    }
    weights = reference_weights_for_megatron(model, tiny)
    back = megatron_to_release(weights, InklingGeometry.from_megatron_config(tiny))
    for k, v in release.items():
        torch.testing.assert_close(back[k], v, rtol=0, atol=0, msg=k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE modules need a GPU")
def test_reference_weights_load_into_megatron(tiny, init_parallel_state):
    ref = build_reference_model(tiny, VOCAB)
    weights = reference_weights_for_megatron(ref, tiny)
    model = build_inkling_model(tiny, vocab_size=VOCAB, max_sequence_length=4096)
    state = dict(weights)
    state.update({k: None for k in model.state_dict() if k.endswith("_extra_state")})
    model.load_state_dict(state, strict=True)
    loaded = model.state_dict()
    for key, value in weights.items():
        torch.testing.assert_close(loaded[key].cpu(), value, rtol=0, atol=0, msg=key)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE modules need a GPU")
def test_reference_gradients_land_in_megatron_layout(tiny, init_parallel_state):
    ref = build_reference_model(tiny, VOCAB)
    ids = torch.randint(0, VOCAB, (2, 96), generator=torch.Generator().manual_seed(0))
    logits, loss, grads = reference_forward_backward(ref, ids, tiny)
    assert logits.shape == (2, 96, VOCAB) and torch.isfinite(loss)
    model = build_inkling_model(tiny, vocab_size=VOCAB, max_sequence_length=4096, meta=True)
    assert {k: tuple(v.shape) for k, v in grads.items()} == module_tensor_shapes(model)
    # the bias reaches every attention layer (eager is the only backend that applies it)
    for i in range(tiny.num_layers):
        assert grads[f"decoder.layers.{i}.self_attention.rel_proj"].abs().sum() > 0


# ---------------------------------------------------------------------------
# Numerical parity (P1): the Megatron forward/backward against the reference
# ---------------------------------------------------------------------------

UNPADDED = 500  # < VOCAB, so the padded-vocab mask is exercised
SEQ = 96  # past the sliding window (32), the global bias extent (64) and the log-scaling floor (64)


@pytest.fixture(scope="module")
def tiny_fp32():
    return inkling_config_from_model_yaml(
        preset("inkling_small_tiny"),
        params_dtype=torch.float32,
        unpadded_vocab_size=UNPADDED,
        gradient_accumulation_fusion=False,
        inkling_attention_backend="eager",
    )


def _paired_models(config, seed: int = 0):
    ref = build_reference_model(config, VOCAB, device="cuda", seed=seed, numeric=True)
    model = build_inkling_model(config, vocab_size=VOCAB, max_sequence_length=4096).cuda()
    state = dict(reference_weights_for_megatron(ref, config))
    state.update({k: None for k in model.state_dict() if k.endswith("_extra_state")})
    model.load_state_dict(state, strict=True)
    return ref, model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE modules need a GPU")
@pytest.mark.parametrize("fp32_activations", [False, True])
@pytest.mark.parametrize("backend", ["eager", "flex"])
def test_tiny_fp32_forward_backward_matches_reference(
    tiny_fp32, backend, fp32_activations, init_parallel_state
):
    """P1 gate: logits, loss and every gradient match the official implementation in fp32.

    Both attention backends, and in an fp32 model the ``inkling_fp32_activations``
    path computes the same function, so it must match too.
    """
    config = copy.copy(tiny_fp32)
    config.inkling_fp32_activations = fp32_activations
    config.inkling_attention_backend = backend
    ref, model = _paired_models(config)
    ids = torch.randint(0, UNPADDED, (2, SEQ), generator=torch.Generator().manual_seed(1)).cuda()

    ref_logits, ref_loss, ref_grads = reference_forward_backward(ref, ids, config)
    logits, loss, grads = megatron_forward_backward(model, ids)

    assert ref_logits.shape[-1] == UNPADDED
    torch.testing.assert_close(logits[..., :UNPADDED], ref_logits, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)
    assert set(grads) == {n for n, _ in model.named_parameters()}
    for name, grad in grads.items():
        expected = ref_grads[name].to(grad.device)
        scale = expected.abs().max().clamp(min=1e-12)
        err = (grad - expected).abs().max() / scale
        assert err < 1e-5, f"{name}: max |diff| / max |ref| = {err:.2e}"


def _bf16_errors(logits, loss, grads, ref_logits, ref_loss, ref_grads):
    """Median per-token logits error, loss error and median gradient error against fp32.

    Per-token medians, not maxima: in bf16 a few near-tie tokens pick another
    expert, in the official model as well, and dominate any max.
    """
    scale = ref_logits.float().abs().max()
    token = ((logits.float() - ref_logits.float()).abs().amax(-1) / scale).flatten()
    grad = [
        ((grads[n].float() - g.float()).abs().max() / g.float().abs().max()).item()
        for n, g in ref_grads.items()
        if n in grads and g.abs().max() > 0
    ]
    return token.median().item(), abs(loss.item() - ref_loss.item()), sorted(grad)[len(grad) // 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE modules need a GPU")
def test_tiny_bf16_tracks_reference_as_closely_as_transformers_bf16(tiny_fp32, init_parallel_state):
    """bf16 Megatron, with and without ``inkling_fp32_activations``, is no further
    from the fp32 reference than the official model run in bf16 (short convs
    kept in fp32, as ``from_pretrained`` does)."""
    ref = build_reference_model(tiny_fp32, VOCAB, device="cuda", seed=0, numeric=True)
    ids = torch.randint(0, UNPADDED, (2, SEQ), generator=torch.Generator().manual_seed(1)).cuda()
    expected = reference_forward_backward(ref, ids, tiny_fp32)
    weights = reference_weights_for_megatron(ref, tiny_fp32)

    ref16 = copy.deepcopy(ref).to(torch.bfloat16)
    for name, module in ref16.named_modules():
        if (
            name.rsplit(".", 1)[-1]
            in reference_modeling().InklingPreTrainedModel._keep_in_fp32_modules_strict
        ):
            module.float()
    baseline = _bf16_errors(*reference_forward_backward(ref16, ids, tiny_fp32), *expected)
    del ref, ref16

    outputs = {}
    for flag in (False, True):
        config = inkling_config_from_model_yaml(
            preset("inkling_small_tiny"),
            unpadded_vocab_size=UNPADDED,
            gradient_accumulation_fusion=False,
            inkling_attention_backend="eager",
            inkling_fp32_activations=flag,
        )
        model = build_inkling_model(config, vocab_size=VOCAB, max_sequence_length=4096).cuda()
        state = dict(weights)
        state.update({k: None for k in model.state_dict() if k.endswith("_extra_state")})
        model.load_state_dict(state, strict=True)
        logits, loss, grads = megatron_forward_backward(model, ids)
        outputs[flag] = logits
        token, loss_err, grad = _bf16_errors(logits[..., :UNPADDED], loss, grads, *expected)
        assert (
            token <= 1.25 * baseline[0]
        ), f"fp32_activations={flag}: token error {token:.2e} vs {baseline[0]:.2e}"
        assert loss_err <= max(2 * baseline[1], 1e-4), f"fp32_activations={flag}: loss error {loss_err:.2e}"
        assert (
            grad <= 1.25 * baseline[2]
        ), f"fp32_activations={flag}: gradient error {grad:.2e} vs {baseline[2]:.2e}"
    assert not torch.equal(outputs[False], outputs[True]), "inkling_fp32_activations changed nothing"


RELEASE_LAYERS = 6
RELEASE_LAYERS_DIR = Path(os.environ.get("INKLING_SMALL_FIRST_LAYERS", "/tmp/inkling_ref/small_6L"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE modules need a GPU")
@pytest.mark.skipif(
    not (RELEASE_LAYERS_DIR / "model.safetensors").exists(),
    reason="first Inkling-Small layers not fetched; see fetch_release_layers.py",
)
def test_inkling_small_first_layers_match_reference(init_parallel_state):
    """P1 gate: the first six released layers, imported, give the official outputs (fp32).

    Layers 0-5 of Inkling-Small are one full 5:1 attention period (sliding
    0-4, global 5) and cover both MLP kinds (dense 0-1, MoE 2-5). The sequence
    runs past the 512-token window and the 1024-token global bias extent; it
    stays below the 128k log-scaling floor, which the tiny test covers. The
    reference loads the release names through transformers' own conversion
    ops, the Megatron model through the importer, and runs with both attention
    backends. Each fp32 model holds ~110 GB, so the two run one after the other.
    """
    from safetensors.torch import load_file

    release = load_file(str(RELEASE_LAYERS_DIR / "model.safetensors"))
    fetched = {int(k.split(".")[3]) for k in release if k.startswith("model.llm.layers.")}
    assert fetched == set(range(RELEASE_LAYERS)), f"{RELEASE_LAYERS_DIR} holds layers {sorted(fetched)}"
    config = inkling_config_from_model_yaml(
        preset("inkling_small"),
        params_dtype=torch.float32,
        num_layers=RELEASE_LAYERS,
        moe_layer_freq="([0]*2+[1]*4)",
        gradient_accumulation_fusion=False,
        inkling_attention_backend="eager",
    )
    assert config.local_layer_ids == [0, 1, 2, 3, 4]
    vocab = release["model.llm.embed.weight"].shape[0]
    unpadded = config.unpadded_vocab_size
    ids = torch.randint(0, unpadded, (1, 1536), generator=torch.Generator().manual_seed(0)).cuda()

    with torch.device("cuda"):
        ref = reference_modeling().InklingForCausalLM(hf_text_config(config, vocab))
    ref.load_state_dict(release_to_transformers(release), strict=True)
    ref.eval()
    with torch.no_grad():
        expected = ref(input_ids=ids).logits
    del ref
    torch.cuda.empty_cache()

    model = build_inkling_model(config, vocab_size=vocab, max_sequence_length=4096).cuda().eval()
    state = dict(release_to_megatron(release, InklingGeometry.from_megatron_config(config)))
    state.update({k: None for k in model.state_dict() if k.endswith("_extra_state")})
    model.load_state_dict(state, strict=True)
    del release, state
    assert [layer.mlp.__class__.__name__ for layer in model.decoder.layers] == ["InklingMLP"] * 2 + [
        "InklingMoELayer"
    ] * 4
    assert [layer.self_attention.core_attention.is_sliding for layer in model.decoder.layers] == [
        True
    ] * 5 + [False]

    for backend in ("eager", "flex"):
        for layer in model.decoder.layers:
            layer.self_attention.core_attention.backend = backend
        with torch.no_grad():
            logits = model(ids, None, None)[..., :unpadded]
        assert expected.shape == logits.shape
        err = ((logits - expected).abs().max() / expected.abs().max()).item()
        assert err < 1e-5, f"{backend}: max |diff| / max |ref| = {err:.2e}"
        assert (logits.argmax(-1) == expected.argmax(-1)).float().mean() > 0.999, backend
