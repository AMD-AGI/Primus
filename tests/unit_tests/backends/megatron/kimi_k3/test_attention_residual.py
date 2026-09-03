###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for ``AttentionResidualMixer.forward``.

The mixer replaces ``x = x + sublayer(x)`` with a learned softmax mixture over
the running residual stream (``prefix_sum``) and a set of cross-layer
checkpoints (``block_residual``); see the module docstring at
``primus/backends/megatron/core/transformer/kimi_k3/attention_residual.py``
for the three easy-to-get-wrong details this file pins:

1. the output mixes the *un-normalised* candidates -- RMS normalisation feeds
   the softmax scores only;
2. the scorer is rank-1, ``norm_weight (elementwise*) proj_weight``;
3. everything runs in fp32 (or wider) internally and casts back to
   ``prefix_sum``'s dtype at the end.

``_reference_attn_res_mix`` below is an independent transcription of
``_apply_attn_res`` (``modeling_kimi_linear.py``) -- written without looking at
``attn_res_kernels._eager.reference.eager_attn_res_mix`` -- so a regression in
either implementation has a real chance of being caught rather than the test
merely checking the module against its own backend.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

mcore_module = pytest.importorskip(
    "megatron.core.transformer.module",
    reason="megatron.core is not importable in this environment",
)

from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

from primus.backends.megatron.core.transformer.kimi_k3.attention_residual import (  # noqa: E402
    AttentionResidualMixer,
)


def _make_config(hidden_size=8, eps=1e-6, params_dtype=torch.float32, attn_res_backend="eager"):
    config = TransformerConfig(
        hidden_size=hidden_size,
        num_attention_heads=1,
        num_layers=1,
        layernorm_epsilon=eps,
        params_dtype=params_dtype,
        use_cpu_initialization=True,
    )
    config.attn_res_backend = attn_res_backend
    return config


def _make_mixer(hidden_size=8, eps=1e-6, params_dtype=torch.float32, seed=0):
    mixer = AttentionResidualMixer(_make_config(hidden_size, eps, params_dtype))
    # Reseed *after* construction (reset_parameters may or may not draw from
    # the RNG depending on config.init_method) so the overwritten values below
    # -- and everything a caller subsequently draws with torch.randn -- are
    # deterministic regardless of that detail.
    torch.manual_seed(seed)
    with torch.no_grad():
        mixer.norm_weight.normal_(mean=1.0, std=0.1)
        mixer.proj_weight.normal_(mean=0.0, std=1.0)
    return mixer


def _reference_attn_res_mix(prefix_sum, block_residual, norm_weight, proj_weight, eps):
    """Independent transcription of ``_apply_attn_res``.

    Deliberately written candidate-by-candidate with an explicit RMSNorm and
    an explicit weighted sum, rather than the batched ``cat`` /
    ``matmul`` formulation the production eager kernel uses, so this does not
    just re-run the same code under a different name.
    """
    compute_dtype = torch.promote_types(prefix_sum.dtype, torch.float32)
    prefix_sum_f = prefix_sum.to(compute_dtype)
    block_residual_f = block_residual.to(compute_dtype)
    score_vec = (norm_weight.to(compute_dtype) * proj_weight.to(compute_dtype).squeeze(0)).to(compute_dtype)

    num_blocks = block_residual_f.shape[-2]
    candidates = [block_residual_f[..., i, :] for i in range(num_blocks)] + [prefix_sum_f]

    raw_scores = []
    for cand in candidates:
        variance = cand.pow(2).mean(dim=-1, keepdim=True)
        normed = cand * torch.rsqrt(variance + eps)
        raw_scores.append((normed * score_vec).sum(dim=-1))

    scores = torch.stack(raw_scores, dim=-1)  # [*, num_candidates]
    probs = scores.softmax(dim=-1)

    mixed = torch.zeros_like(prefix_sum_f)
    for i, cand in enumerate(candidates):
        mixed = mixed + probs[..., i : i + 1] * cand

    return mixed.to(prefix_sum.dtype)


# ---------------------------------------------------------------------------
# Correctness against the independent reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_blocks", [1, 3])
@pytest.mark.parametrize("leading_shape", [(5,), (2, 3)])
def test_forward_matches_independent_reference(num_blocks, leading_shape):
    """Bit-close parity against a from-scratch transcription of the reference.

    Covers both a flat token axis and Megatron's ``[seq, batch]`` layout: the
    module docstring notes ``torch.matmul`` broadcasts over leading dims
    identically either way, so both shapes should hit the same arithmetic.
    """
    hidden_size = 8
    mixer = _make_mixer(hidden_size=hidden_size)

    prefix_sum = torch.randn(*leading_shape, hidden_size)
    block_residual = torch.randn(*leading_shape, num_blocks, hidden_size)

    actual = mixer(prefix_sum, block_residual)
    expected = _reference_attn_res_mix(
        prefix_sum, block_residual, mixer.norm_weight, mixer.proj_weight, mixer.eps
    )

    assert actual.shape == prefix_sum.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_forward_output_dtype_matches_prefix_sum():
    """Detail 3: fp32 internally, cast back once to ``prefix_sum``'s dtype."""
    mixer = _make_mixer(hidden_size=8, params_dtype=torch.float32)

    prefix_sum = torch.randn(4, 8, dtype=torch.float64)
    block_residual = torch.randn(4, 2, 8, dtype=torch.float64)

    out = mixer(prefix_sum, block_residual)

    assert out.dtype == torch.float64
    expected = _reference_attn_res_mix(
        prefix_sum, block_residual, mixer.norm_weight, mixer.proj_weight, mixer.eps
    )
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# Detail 1: mixing the un-normalised candidates, not the RMS-normalised ones
# ---------------------------------------------------------------------------


def test_forward_mixes_unnormalised_candidates():
    """A candidate with huge norm must not be rescaled away by the mixer.

    If the module mixed the RMS-normalised ``k`` instead of the raw ``v`` (the
    bug the module docstring calls out), every candidate would end up at unit
    RMS and a large-magnitude block checkpoint would look identical to a small
    one at the output. Constructing one candidate 100x the others and checking
    the output scales with it (rather than collapsing to O(1) RMS) catches
    that swap.
    """
    hidden_size = 8
    mixer = _make_mixer(hidden_size=hidden_size)

    prefix_sum = torch.randn(1, hidden_size)
    big_block = torch.randn(1, 1, hidden_size) * 100.0

    out = mixer(prefix_sum, big_block)
    expected = _reference_attn_res_mix(prefix_sum, big_block, mixer.norm_weight, mixer.proj_weight, mixer.eps)

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
    # The un-normalised mix can only reach this magnitude if the raw (not
    # RMS-normalised) big_block leaked into the weighted sum.
    assert out.norm() > 5.0


# ---------------------------------------------------------------------------
# num_blocks == 0: a legal no-op softmax over the stream alone
# ---------------------------------------------------------------------------


def test_forward_with_zero_blocks_is_identity():
    mixer = _make_mixer(hidden_size=8)
    prefix_sum = torch.randn(3, 8)
    block_residual = torch.empty(3, 0, 8)

    out = mixer(prefix_sum, block_residual)

    torch.testing.assert_close(out, prefix_sum, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


def test_forward_raises_on_hidden_size_mismatch():
    mixer = _make_mixer(hidden_size=8)
    prefix_sum = torch.randn(2, 8)
    block_residual = torch.randn(2, 1, 4)

    with pytest.raises(ValueError, match="hidden"):
        mixer(prefix_sum, block_residual)


# ---------------------------------------------------------------------------
# Differentiability (module docstring: gradcheck-safe end to end)
# ---------------------------------------------------------------------------


def test_forward_gradcheck():
    hidden_size = 4
    mixer = _make_mixer(hidden_size=hidden_size, params_dtype=torch.float64)
    mixer = mixer.to(torch.float64)

    prefix_sum = torch.randn(2, hidden_size, dtype=torch.float64, requires_grad=True)
    block_residual = torch.randn(2, 2, hidden_size, dtype=torch.float64, requires_grad=True)
    mixer.norm_weight.requires_grad_(True)
    mixer.proj_weight.requires_grad_(True)

    # norm_weight / proj_weight are passed through so gradcheck perturbs the
    # exact tensors the mixer reads (they are the same objects, not copies).
    assert torch.autograd.gradcheck(
        lambda ps, br, nw, pw: mixer(ps, br),
        (prefix_sum, block_residual, mixer.norm_weight, mixer.proj_weight),
        eps=1e-6,
        atol=1e-4,
    )
