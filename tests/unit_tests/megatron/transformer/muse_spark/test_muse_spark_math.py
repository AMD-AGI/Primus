###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU checks of the Muse Spark ops against the reference formulas.

The ``_hf_*`` references are copied from transformers'
``models/muse_spark/modeling_muse_spark.py`` (``MuseSparkResidualGate``,
``MuseSparkTextInputScaledRMSNorm``, the LM-head softcap in
``MuseSparkForConditionalGeneration``) and ``layer_types`` from
``configuration_muse_spark.py``.

The reference keeps the residual gates in fp32 (``_keep_in_fp32_modules_strict``)
and computes ``1 + weight`` in the weight's dtype. The Primus ops compute in
fp32 whatever the parameter dtype, so the bit-exact comparisons use fp32
gate and norm weights.
"""

import pytest
import torch

from primus.backends.megatron.core.models.muse_spark.muse_spark_ops import (
    global_attention_layer_mask,
    input_scaled_rms_norm,
    residual_gate,
    softcap_logits,
)


def _hf_residual_gate(residual, hidden_states, gate, temperature, eps):
    w = gate / temperature
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=eps))
    return (alpha * residual + beta * hidden_states).to(residual.dtype)


def _hf_input_scaled_rms_norm(x, weight, eps):
    output = x.float() * (1.0 + weight).float()
    output = output.type_as(x).float()
    output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + eps)
    return output.type_as(x)


def _hf_layer_types(num_hidden_layers):
    return [
        "full_attention" if (num_hidden_layers - 1 - i) % 4 == 0 else "sliding_attention"
        for i in range(num_hidden_layers)
    ]


def _rms_norm(x, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


@pytest.mark.parametrize("num_layers", [1, 4, 5, 62])
def test_global_layers_match_reference_layer_types(num_layers):
    mask = global_attention_layer_mask(num_layers, 4)
    assert mask == [t == "full_attention" for t in _hf_layer_types(num_layers)]


def test_global_layers_count_back_from_the_last():
    mask = global_attention_layer_mask(62, 4)
    assert [i for i, g in enumerate(mask) if g] == list(range(1, 62, 4))
    assert mask[-1] is True
    assert sum(mask) == 16
    # A 4-layer proxy keeps the last group of the full model: 3 sliding, 1 global.
    assert global_attention_layer_mask(4, 4) == [False, False, False, True]
    assert global_attention_layer_mask(3, 1) == [True, True, True]
    with pytest.raises(ValueError):
        global_attention_layer_mask(4, 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_residual_gate_matches_reference(dtype):
    torch.manual_seed(0)
    hidden = 64
    residual = torch.randn(3, 2, hidden, dtype=dtype)
    hidden_states = torch.randn(3, 2, hidden, dtype=dtype)
    gate = torch.randn(hidden) * 0.5
    got = residual_gate(residual, hidden_states, gate, 0.3, 1e-3)
    ref = _hf_residual_gate(residual, hidden_states, gate, 0.3, 1e-3)
    assert got.dtype == dtype
    torch.testing.assert_close(got, ref, rtol=0, atol=0)

    # Megatron casts the gate to the bf16 param dtype; that rounds only the gate.
    got_bf16_gate = residual_gate(residual, hidden_states, gate.bfloat16(), 0.3, 1e-3)
    ref_bf16_gate = _hf_residual_gate(residual, hidden_states, gate.bfloat16().float(), 0.3, 1e-3)
    torch.testing.assert_close(got_bf16_gate, ref_bf16_gate, rtol=0, atol=0)


def test_residual_gate_preserves_variance():
    """alpha^2 + beta^2 == 1 below the clamp, so unit-variance inputs stay unit variance."""
    gate = torch.linspace(-0.6, 0.6, 257)
    weight = gate / 0.3
    beta = torch.sigmoid(weight)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-weight) * (1 + beta), min=1e-3))
    torch.testing.assert_close(alpha.pow(2) + beta.pow(2), torch.ones_like(gate))

    torch.manual_seed(0)
    residual = torch.randn(200_000, 1)
    hidden_states = torch.randn(200_000, 1)
    out = residual_gate(residual, hidden_states, torch.tensor([0.25]), 0.3, 1e-3)
    assert abs(out.var().item() - 1.0) < 0.02


def test_residual_gate_clamps_alpha():
    """A saturated gate keeps alpha at sqrt(eps) instead of dropping the residual."""
    residual = torch.ones(1, 1)
    hidden_states = torch.zeros(1, 1)
    out = residual_gate(residual, hidden_states, torch.tensor([30.0]), 0.3, 1e-3)
    torch.testing.assert_close(out, torch.full((1, 1), 1e-3**0.5))


def test_input_scaled_rms_norm():
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    eps = 1e-8
    # Zero weight (the init) is a plain weightless RMSNorm.
    torch.testing.assert_close(input_scaled_rms_norm(x, torch.zeros(32), eps), _rms_norm(x, eps))
    # The weight scales the input, so any uniform weight cancels out...
    uniform = torch.full((32,), 0.7)
    torch.testing.assert_close(input_scaled_rms_norm(x, uniform, eps), _rms_norm(x, eps))
    # ...and only the per-channel shape of the weight survives.
    weight = torch.randn(32) * 0.1
    torch.testing.assert_close(input_scaled_rms_norm(x, weight, eps), _rms_norm(x * (1 + weight), eps))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_input_scaled_rms_norm_matches_reference(dtype):
    """Both round ``x * (1 + w)`` to the input dtype before normalising."""
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=dtype)
    weight = torch.randn(64) * 0.1
    got = input_scaled_rms_norm(x, weight, 1e-8)
    assert got.dtype == dtype
    torch.testing.assert_close(got, _hf_input_scaled_rms_norm(x, weight, 1e-8), rtol=0, atol=0)


def test_routing_weight_must_follow_the_post_expert_norm():
    """The post-expert norm is scale-invariant, so it would erase a weight applied before it.

    This is why MuseSparkMoELayer runs the experts with unit probs and applies
    the routing weight after ``post_expert_norm``.
    """
    torch.manual_seed(0)
    expert_out = torch.randn(6, 16)
    probs = torch.rand(6) + 0.05
    weight = torch.randn(16) * 0.1
    norm_then_prob = input_scaled_rms_norm(expert_out, weight, 1e-8) * probs[:, None]
    prob_then_norm = input_scaled_rms_norm(expert_out * probs[:, None], weight, 1e-8)
    torch.testing.assert_close(prob_then_norm, input_scaled_rms_norm(expert_out, weight, 1e-8))
    assert not torch.allclose(norm_then_prob, prob_then_norm)
    torch.testing.assert_close(norm_then_prob.norm(dim=-1) / probs, torch.full((6,), 16**0.5))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_softcap_logits(dtype):
    multiplier = 0.1767766952966369
    cap = 20.0
    logits = torch.tensor([-1e4, -300.0, -1.0, 0.0, 1.0, 300.0, 1e4], dtype=dtype)
    got = softcap_logits(logits, multiplier, cap)
    ref = logits.float() * multiplier
    ref = ref / cap
    ref = torch.tanh(ref)
    ref = (ref * cap).to(dtype)
    assert got.dtype == dtype
    torch.testing.assert_close(got, ref)
    assert got.float().abs().max() <= cap
    # Near zero the cap is inactive and only the multiplier remains.
    small = torch.tensor([0.5, -2.0])
    torch.testing.assert_close(softcap_logits(small, multiplier, cap), small * multiplier, rtol=1e-3, atol=0)
