###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for Kimi K3 attention residuals and the layer / block assembly.

Four things are under test, in increasing order of scope.

**The mixer** (:class:`AttentionResidualMixer` vs ``_apply_attn_res``,
``modeling_kimi_linear.py:1075-1088``). Parity is asserted bit-exactly in fp32
against a transcription that shares no code with the implementation. Three
*negative controls* accompany it, because a softmax over a handful of similar
candidates is forgiving enough that a wrong implementation can still look
plausible:

* mixing the RMS-normalised ``k`` instead of the raw ``v`` — the easiest thing
  to get wrong, and the one the reference is most explicit about (``:1083``
  builds ``k``, ``:1087`` mixes ``v_float``);
* dropping the RMSNorm gain from the rank-1 ``score_weight`` (``:1084``);
* concatenating the running stream *first* instead of last, which permutes the
  softmax and silently misroutes every weight.

Each control must differ from the reference by a wide margin, so a passing run
also demonstrates that the test discriminates.

**The per-layer bookkeeping** (:class:`KimiK3Layer` vs
``KimiDecoderLayer._forward_attn_residual``, ``:973-1046``). An 8-layer stack
built with *stub* attention / MLP sub-blocks is compared against a
transcription of the reference loop that reuses those very same sub-block
modules, so any difference is bookkeeping rather than numerics. The checkpoint
count and the ``prefix_sum`` reset are asserted at every layer.

Note that the checkpoint trace on *entry* to each layer is
``[0, 1, 1, 1, 1, 2, 2, 2]`` for ``attn_res_block_size = 4``, not the
``[0, 1, 1, 1, 2, 2, 2, 2]`` written in ``DESIGN.md`` §7.2. Appends land at
layers 0 and 4, and layer 4's own append happens *after* its pre-attention mix
(``:987`` runs before ``:995``), so layer 4 still enters with one checkpoint.

**The pipeline seam** (``_lift_res_in`` / ``_lower_res_out``). PP > 1 is a
later work package, but the carrier round-trip and the "fill count is a pure
function of the layer index" property it rests on are cheap to pin now.

**The spec tree** (``get_kimi_k3_runtime_decoder_spec``): layer pattern, MoE
pattern, the filled ``activation_func`` slot, and construction of the 8-layer
debug model.

The first three groups are pure PyTorch and run on CPU. The spec-tree group
needs Transformer Engine, and building the MoE layers additionally needs a
visible GPU — not because Kimi K3 wants one, but because upstream's
``TopKRouter`` hard-codes ``device=torch.cuda.current_device()`` for its
expert-bias buffers (``router.py:172-189``).
"""

from __future__ import annotations

import math
import os
from typing import List

import pytest
import torch  # must precede any transformer_engine import
import torch.nn as nn
import torch.nn.functional as F

from primus.backends.megatron.core.models.kimi_k3.kimi_k3_block import (
    KimiK3Layer,
    KimiK3LayerSubmodules,
    KimiK3TransformerBlock,
    KimiK3TransformerBlockSubmodules,
    _lift_res_in,
    _lower_res_out,
    attn_res_num_blocks_before,
)
from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
    KimiK3TransformerConfig,
)
from primus.backends.megatron.core.transformer.kimi_k3.attention_residual import (
    AttentionResidualHead,
    AttentionResidualMixer,
)

HIDDEN = 32
SEQ = 6
BATCH = 2
NUM_LAYERS = 8
BLOCK_SIZE = 4
EPS = 1e-5

# Checkpoints in flight on ENTRY to each layer, for block_size 4 over 8 layers.
ENTRY_TRACE = [0, 1, 1, 1, 1, 2, 2, 2]


# ===========================================================================
# Reference transcriptions (share no code with the implementation)
# ===========================================================================


def reference_apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """``_apply_attn_res`` (``modeling_kimi_linear.py:1075-1088``), verbatim.

    ``prefix_sum`` is ``[num_tokens, hidden]``, ``block_residual`` is
    ``[num_tokens, num_blocks, hidden]``, ``proj_weight`` is the
    ``nn.Linear(hidden, 1, bias=False)`` weight ``[1, hidden]`` and
    ``norm_weight`` is the RMSNorm gain ``[hidden]``.
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


def _control_mix_normalised(prefix_sum, block_residual, proj_weight, norm_weight, eps):
    """Negative control: mix the RMS-normalised candidates instead of the raw ones."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    probs = (k * score_weight).sum(-1).softmax(-1).unsqueeze(1)
    return torch.matmul(probs, k).squeeze(1).to(v.dtype)


def _control_score_without_norm_gain(prefix_sum, block_residual, proj_weight, norm_weight, eps):
    """Negative control: score with the projection alone, dropping the RMSNorm gain."""
    del norm_weight
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)
    probs = (k * proj_weight.squeeze(0).float()).sum(-1).softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v_float).squeeze(1).to(v.dtype)


def _control_score_unnormalised(prefix_sum, block_residual, proj_weight, norm_weight, eps):
    """Negative control: score the raw candidates, skipping the RMS normalisation.

    The mirror image of ``_control_mix_normalised``: this one keeps the right
    thing in the mixture and puts the wrong thing in the scores.
    """
    del eps
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    probs = (v_float * score_weight).sum(-1).softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v_float).squeeze(1).to(v.dtype)


def _mix_with_stream_first(prefix_sum, block_residual, proj_weight, norm_weight, eps):
    """The mixture with the running stream concatenated first instead of last."""
    v = torch.cat((prefix_sum.unsqueeze(1), block_residual), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)
    score_weight = norm_weight.float() * proj_weight.squeeze(0).float()
    probs = (k * score_weight).sum(-1).softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v_float).squeeze(1).to(v.dtype)


# ===========================================================================
# Helpers
# ===========================================================================


def _base_config(**overrides):
    """A minimal Kimi K3 config; enough for the mixer and the stubbed layers."""
    kwargs = dict(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        kv_channels=8,
        ffn_hidden_size=2 * HIDDEN,
        attn_res_block_size=BLOCK_SIZE,
        linear_attention_freq=[1, 1, 1, 0, 1, 1, 1, 0],
        layernorm_epsilon=EPS,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        perform_initialization=True,
        activation_func=F.silu,
        add_bias_linear=False,
        normalization="RMSNorm",
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    kwargs.update(overrides)
    return KimiK3TransformerConfig(**kwargs)


def _make_mixer(config=None, *, seed=0, cls=AttentionResidualMixer):
    """A mixer with both rank-1 factors randomised.

    ``norm_weight`` is ones at init, which would make the controls that touch
    it degenerate, so both factors get real values.
    """
    config = config or _base_config()
    mixer = cls(config=config)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        mixer.norm_weight.copy_(torch.randn(HIDDEN, generator=generator) * 0.3 + 1.0)
        mixer.proj_weight.copy_(torch.randn(1, HIDDEN, generator=generator) * 0.5)
    return mixer


def _random_candidates(num_tokens: int, num_blocks: int, *, seed=1, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    prefix_sum = torch.randn(num_tokens, HIDDEN, generator=generator, dtype=torch.float32) * 3.0
    block_residual = torch.randn(
        num_tokens, num_blocks, HIDDEN, generator=generator, dtype=torch.float32
    )
    # A spread of magnitudes, so RMS normalisation is far from a no-op and
    # "mix v" vs "mix k" genuinely diverge.
    scales = torch.tensor([0.25, 1.0, 4.0][:num_blocks]).view(1, -1, 1)
    return prefix_sum.to(dtype), (block_residual * scales).to(dtype)


def _relative_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


# ===========================================================================
# 1. The mixer
# ===========================================================================


@pytest.mark.parametrize("num_blocks", [1, 2, 3])
def test_mixer_matches_reference_bit_exactly(num_blocks):
    """``AttentionResidualMixer`` == ``_apply_attn_res`` in fp32."""
    mixer = _make_mixer()
    prefix_sum, block_residual = _random_candidates(SEQ * BATCH, num_blocks)

    got = mixer(prefix_sum, block_residual)
    want = reference_apply_attn_res(
        prefix_sum, block_residual, mixer.proj_weight, mixer.norm_weight, EPS
    )

    assert got.shape == prefix_sum.shape
    assert torch.equal(got, want), f"max |diff| = {(got - want).abs().max().item():.3e}"


def test_mixer_with_zero_blocks_is_the_identity():
    """With no checkpoints the softmax has one candidate, so the mix is a no-op.

    The caller skips the call in that case (``:987``); this pins that the
    skipped and the taken branch agree, which is what makes the skip an
    optimisation rather than a semantic difference.
    """
    mixer = _make_mixer()
    prefix_sum, _ = _random_candidates(SEQ * BATCH, 1)
    empty = prefix_sum.new_zeros(prefix_sum.shape[0], 0, HIDDEN)

    torch.testing.assert_close(mixer(prefix_sum, empty), prefix_sum, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "control,name",
    [
        (_control_mix_normalised, "mixes the normalised k instead of v"),
        (_control_score_without_norm_gain, "drops the RMSNorm gain from score_weight"),
        (_control_score_unnormalised, "scores the raw candidates instead of the normalised ones"),
    ],
)
def test_negative_controls_differ_measurably(control, name):
    """Each plausible mistake must move the output well outside numerical noise."""
    mixer = _make_mixer()
    prefix_sum, block_residual = _random_candidates(SEQ * BATCH, 2)

    correct = reference_apply_attn_res(
        prefix_sum, block_residual, mixer.proj_weight, mixer.norm_weight, EPS
    )
    wrong = control(prefix_sum, block_residual, mixer.proj_weight, mixer.norm_weight, EPS)

    relative = _relative_difference(wrong, correct)
    assert relative > 1e-2, f"control '{name}' is indistinguishable (relative diff {relative:.3e})"
    # ...and the implementation is on the correct side of the split.
    assert torch.equal(mixer(prefix_sum, block_residual), correct)


def test_candidate_order_does_not_matter():
    """Concatenation order is *not* a bug surface, and that is worth recording.

    Scores are computed per candidate, so permuting the candidate list permutes
    the softmax with it and the weighted sum is unchanged. Anyone reviewing this
    code will wonder whether ``cat((block_residual, prefix_sum))`` could have
    been the other way round; it could, up to summation order.
    """
    mixer = _make_mixer()
    prefix_sum, block_residual = _random_candidates(SEQ * BATCH, 2)

    torch.testing.assert_close(
        _mix_with_stream_first(prefix_sum, block_residual, mixer.proj_weight, mixer.norm_weight, EPS),
        mixer(prefix_sum, block_residual),
        rtol=1e-6,
        atol=1e-6,
    )


def test_output_is_a_convex_combination_of_the_raw_candidates():
    """The mix lands inside the candidate hull, elementwise.

    A convex combination of ``v`` cannot leave the per-coordinate min/max
    envelope of ``v``. Mixing ``k`` would, as soon as the candidates differ in
    RMS, so this is a second independent guard on the same bug.
    """
    mixer = _make_mixer()
    prefix_sum, block_residual = _random_candidates(SEQ * BATCH, 3)

    out = mixer(prefix_sum, block_residual)
    candidates = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)

    tol = 1e-5
    assert (out >= candidates.amin(dim=1) - tol).all()
    assert (out <= candidates.amax(dim=1) + tol).all()


def test_probabilities_sum_to_one():
    """Recompute the scores the way the module does and check the softmax."""
    mixer = _make_mixer()
    prefix_sum, block_residual = _random_candidates(SEQ * BATCH, 2)

    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1).float()
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)
    probs = (k * mixer.score_weight()).sum(-1).softmax(-1)

    assert probs.shape == (SEQ * BATCH, 3)
    torch.testing.assert_close(probs.sum(-1), torch.ones(SEQ * BATCH), rtol=1e-6, atol=1e-6)


def test_score_weight_is_the_rank_one_product():
    """``score_weight`` == RMSNorm gain * projection direction (``:1084``)."""
    mixer = _make_mixer()
    torch.testing.assert_close(
        mixer.score_weight(),
        mixer.norm_weight.float() * mixer.proj_weight.squeeze(0).float(),
        rtol=0,
        atol=0,
    )
    assert mixer.score_weight().shape == (HIDDEN,)
    assert sum(p.numel() for p in mixer.parameters()) == 2 * HIDDEN


def test_sequence_first_layout_matches_the_flattened_reference():
    """``[s, b, ...]`` and the reference's flat ``[t, ...]`` agree bit for bit.

    The reference flattens batch and sequence into one token axis; Megatron is
    sequence-first. The mixer indexes the candidate axis with ``dim=-2`` so the
    two layouts are the same arithmetic, and this is the assertion that says so.
    """
    mixer = _make_mixer()
    flat_prefix, flat_blocks = _random_candidates(SEQ * BATCH, 2)

    got = mixer(
        flat_prefix.view(SEQ, BATCH, HIDDEN), flat_blocks.view(SEQ, BATCH, 2, HIDDEN)
    ).reshape(SEQ * BATCH, HIDDEN)
    want = reference_apply_attn_res(flat_prefix, flat_blocks, mixer.proj_weight, mixer.norm_weight, EPS)

    assert torch.equal(got, want)


def test_bf16_inputs_compute_in_fp32_and_cast_back():
    """fp32 internals, bf16 in and out (``:1081``, ``:1088``)."""
    config = _base_config(params_dtype=torch.bfloat16, bf16=True)
    mixer = _make_mixer(config)
    prefix_sum, block_residual = _random_candidates(SEQ * BATCH, 2, dtype=torch.bfloat16)

    out = mixer(prefix_sum, block_residual)
    assert out.dtype == torch.bfloat16
    assert torch.equal(
        out,
        reference_apply_attn_res(
            prefix_sum, block_residual, mixer.proj_weight, mixer.norm_weight, EPS
        ),
    )

    # A bf16-internal implementation would lose the low bits of the score sum.
    fp32_reference = reference_apply_attn_res(
        prefix_sum.float(),
        block_residual.float(),
        mixer.proj_weight.float(),
        mixer.norm_weight.float(),
        EPS,
    )
    assert _relative_difference(out.float(), fp32_reference) < 1e-2


def test_head_is_a_distinct_class_with_its_own_parameters():
    """``AttentionResidualHead`` is the post-stack mix, not a shared mixer.

    The block builds exactly one, on ``post_process`` only, so a type check is
    the cheapest way to assert that placement.
    """
    config = _base_config()
    head = _make_mixer(config, cls=AttentionResidualHead)
    mixer = _make_mixer(config, seed=7)

    assert isinstance(head, AttentionResidualMixer)
    assert type(head) is not type(mixer)
    assert head.norm_weight is not mixer.norm_weight

    prefix_sum, block_residual = _random_candidates(SEQ * BATCH, 2)
    assert torch.equal(
        head(prefix_sum, block_residual),
        reference_apply_attn_res(prefix_sum, block_residual, head.proj_weight, head.norm_weight, EPS),
    )


def test_gradients_reach_both_rank_one_factors():
    """Both scorer factors must be trainable; a detached one is silently inert."""
    mixer = _make_mixer()
    prefix_sum, block_residual = _random_candidates(SEQ * BATCH, 2)
    prefix_sum.requires_grad_(True)

    mixer(prefix_sum, block_residual).square().sum().backward()

    for name, param in mixer.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient is not finite"
    assert prefix_sum.grad is not None


# ===========================================================================
# 2. Per-layer bookkeeping
# ===========================================================================


class _StubNorm(nn.Module):
    """Deterministic RMSNorm stand-in with the Megatron norm build signature."""

    def __init__(self, config=None, hidden_size: int = HIDDEN, eps: float = EPS, **_kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class _StubSubBlock(nn.Module):
    """A sub-block with the Megatron ``(output, bias)`` contract.

    ``tanh``-bounded and small, so an 8-layer stack of them stays numerically
    tame. The same *instance* backs both the layer under test and the reference
    loop, so the comparison isolates bookkeeping from numerics.
    """

    def __init__(self, config=None, scale: float = 0.1, seed: int = 0, **_kwargs):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(HIDDEN, HIDDEN, generator=generator) * scale)

    def forward(self, hidden_states, *args, **kwargs):
        return torch.tanh(hidden_states @ self.weight), None


class _FakePgCollection:
    """Just enough of ``ProcessGroupCollection`` for a stubbed CPU layer."""

    tp = None
    cp = None
    pp = None


def _stub_layer_submodules(seed: int) -> KimiK3LayerSubmodules:
    from megatron.core.transformer.spec_utils import ModuleSpec

    return KimiK3LayerSubmodules(
        input_layernorm=_StubNorm,
        self_attention=ModuleSpec(module=_StubSubBlock, params={"seed": seed}),
        pre_mlp_layernorm=_StubNorm,
        mlp=ModuleSpec(module=_StubSubBlock, params={"seed": seed + 100}),
        attn_res_mixer=ModuleSpec(module=AttentionResidualMixer),
        mlp_res_mixer=ModuleSpec(module=AttentionResidualMixer),
    )


def _build_stub_block(config) -> KimiK3TransformerBlock:
    from megatron.core.transformer.spec_utils import ModuleSpec

    layer_specs = [
        ModuleSpec(
            module=KimiK3Layer,
            params={"layer_idx": i, "is_kda_layer": bool(config.is_kda_layer(i))},
            submodules=_stub_layer_submodules(seed=i),
        )
        for i in range(int(config.num_layers))
    ]
    block = KimiK3TransformerBlock(
        config=config,
        submodules=KimiK3TransformerBlockSubmodules(
            layer_specs=layer_specs,
            attn_res_head=ModuleSpec(module=AttentionResidualHead),
            final_layernorm=_StubNorm,
        ),
        pre_process=True,
        post_process=True,
        pg_collection=_FakePgCollection(),
    )
    # Randomise every scorer, so a swapped or shared mixer cannot pass by symmetry.
    generator = torch.Generator().manual_seed(11)
    for module in block.modules():
        if isinstance(module, AttentionResidualMixer):
            with torch.no_grad():
                module.norm_weight.copy_(torch.randn(HIDDEN, generator=generator) * 0.2 + 1.0)
                module.proj_weight.copy_(torch.randn(1, HIDDEN, generator=generator) * 0.4)
    return block


def _reference_stack_forward(block, hidden_states: torch.Tensor):
    """``KimiLinearModel.forward``'s loop (``:1188-1217``) in flat ``[t, h]`` form.

    Reuses the block's own submodules, so the only thing reimplemented here is
    the residual bookkeeping.
    """
    seq, batch, hidden = hidden_states.shape
    flat = hidden_states.reshape(seq * batch, hidden)
    block_residual = flat.new_zeros(seq * batch, 0, hidden)
    trace: List[int] = []

    for layer in block.layers:
        trace.append(block_residual.shape[1])
        prefix_sum = flat

        if block_residual.shape[1] > 0:
            flat = reference_apply_attn_res(
                prefix_sum,
                block_residual,
                layer.attn_res_mixer.proj_weight,
                layer.attn_res_mixer.norm_weight,
                EPS,
            )

        if layer.layer_idx % BLOCK_SIZE == 0:
            block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
            prefix_sum = None

        attn_out, _ = layer.self_attention(layer.input_layernorm(flat))
        prefix_sum = attn_out if prefix_sum is None else prefix_sum + attn_out

        flat = reference_apply_attn_res(
            prefix_sum,
            block_residual,
            layer.mlp_res_mixer.proj_weight,
            layer.mlp_res_mixer.norm_weight,
            EPS,
        )
        mlp_out, _ = layer.mlp(layer.pre_mlp_layernorm(flat))
        flat = prefix_sum + mlp_out

    flat = reference_apply_attn_res(
        flat,
        block_residual,
        block.attn_res_head.proj_weight,
        block.attn_res_head.norm_weight,
        EPS,
    )
    return block.final_layernorm(flat).view(seq, batch, hidden), trace


def _record_block_residual_trace(block):
    """Capture ``block_residual``'s checkpoint count on entry to every layer."""
    trace: List[int] = []

    def _hook(_module, _args, kwargs):
        residual = kwargs.get("block_residual")
        trace.append(0 if residual is None else residual.shape[-2])

    handles = [layer.register_forward_pre_hook(_hook, with_kwargs=True) for layer in block.layers]
    return trace, handles


def test_layer_stack_matches_the_reference_loop():
    """The whole 8-layer bookkeeping, end to end, against the HF transcription."""
    block = _build_stub_block(_base_config())
    hidden_states = torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(3))

    got = block(hidden_states)
    want, reference_trace = _reference_stack_forward(block, hidden_states)

    assert reference_trace == ENTRY_TRACE
    assert got.shape == (SEQ, BATCH, HIDDEN)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_block_residual_checkpoint_trace():
    """Appends land at ``L % 4 == 0``, so entry counts are 0,1,1,1,1,2,2,2.

    Layer 4 still enters with one checkpoint: its own append happens after its
    pre-attention mix. ``DESIGN.md`` §7.2 records ``[0,1,1,1,2,2,2,2]``, which
    is off by one at layer 4.
    """
    config = _base_config()
    block = _build_stub_block(config)
    trace, handles = _record_block_residual_trace(block)
    try:
        block(torch.randn(SEQ, BATCH, HIDDEN))
    finally:
        for handle in handles:
            handle.remove()

    assert trace == ENTRY_TRACE
    # The same numbers, derived without running anything -- the property the PP
    # carrier relies on.
    assert trace == [attn_res_num_blocks_before(i, BLOCK_SIZE) for i in range(NUM_LAYERS)]
    assert block.attn_res_block_count_trace() == ENTRY_TRACE
    assert attn_res_num_blocks_before(NUM_LAYERS, BLOCK_SIZE) == config.attn_res_num_blocks_max == 2


def test_prefix_sum_resets_at_a_checkpoint_layer():
    """After an append, the running sum restarts from the attention output.

    Layer 0 appends, so its output is ``attn_out + mlp_out`` with the layer
    *input* nowhere in the sum. Layer 1 does not append, so its input survives.
    Feeding a large-magnitude input separates the two cases unambiguously,
    because the stub sub-blocks are ``tanh``-bounded.
    """
    block = _build_stub_block(_base_config())
    layer0, layer1 = block.layers[0], block.layers[1]

    hidden_states = torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(5)) * 50.0
    empty = hidden_states.new_zeros(SEQ, BATCH, 0, HIDDEN)

    out0, residual0 = layer0(hidden_states, block_residual=empty)
    assert residual0.shape[-2] == 1
    assert out0.abs().max() < 4.0, "layer 0 did not reset prefix_sum after its checkpoint append"
    torch.testing.assert_close(residual0[:, :, 0, :], hidden_states, rtol=0, atol=0)

    out1, residual1 = layer1(out0, block_residual=residual0)
    assert residual1.shape[-2] == 1, "layer 1 must not append a checkpoint"
    assert residual1 is residual0, "layer 1 must forward the same checkpoint tensor"
    # Layer 1 keeps its input in the running sum, so its output differs from the
    # input only by two bounded sub-block outputs.
    assert (out1 - out0).abs().max() < 4.0


def test_layer_zero_has_no_pre_attention_mixer():
    """Layer 0's pre-attention mix is skipped (``:987``), so it builds no mixer.

    Building one anyway would leave two parameters that can never receive a
    gradient, which permanently disarms the "every parameter gets a grad" check
    below. Every later layer must have one.
    """
    block = _build_stub_block(_base_config())

    assert block.layers[0].attn_res_mixer is None
    assert all(layer.attn_res_mixer is not None for layer in block.layers[1:])
    assert all(layer.mlp_res_mixer is not None for layer in block.layers)


def test_perturbing_a_pre_attention_mixer_changes_the_output():
    """Layer 1's pre-attention mixer is genuinely on the forward path."""
    block = _build_stub_block(_base_config())
    hidden_states = torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(9))
    empty = hidden_states.new_zeros(SEQ, BATCH, 0, HIDDEN)

    out0, residual0 = block.layers[0](hidden_states, block_residual=empty)
    baseline1, _ = block.layers[1](out0, block_residual=residual0)

    with torch.no_grad():
        block.layers[1].attn_res_mixer.proj_weight.mul_(-5.0)
    perturbed1, _ = block.layers[1](out0, block_residual=residual0)

    assert not torch.equal(baseline1, perturbed1)


def test_layer_returns_the_checkpoint_tensor_not_a_context():
    """The second return slot carries ``block_residual``, per ``:1046``."""
    block = _build_stub_block(_base_config())
    hidden_states = torch.randn(SEQ, BATCH, HIDDEN)
    empty = hidden_states.new_zeros(SEQ, BATCH, 0, HIDDEN)

    out, residual = block.layers[0](hidden_states, block_residual=empty)
    assert out.shape == (SEQ, BATCH, HIDDEN)
    assert residual.shape == (SEQ, BATCH, 1, HIDDEN)


def test_layer_rejects_a_checkpoint_count_that_drifts_from_its_index():
    """The runtime state and the index-derived schedule must agree."""
    block = _build_stub_block(_base_config())
    hidden_states = torch.randn(SEQ, BATCH, HIDDEN)
    wrong = hidden_states.new_zeros(SEQ, BATCH, 2, HIDDEN)

    with pytest.raises(AssertionError, match="checkpoints"):
        block.layers[1](hidden_states, block_residual=wrong)


def test_disabled_attention_residuals_fall_back_to_a_plain_residual():
    """``attn_res_block_size = None`` gives ordinary pre-norm residuals."""
    from megatron.core.transformer.spec_utils import ModuleSpec

    config = _base_config(attn_res_block_size=None)
    layer = KimiK3Layer(
        config=config,
        submodules=KimiK3LayerSubmodules(
            input_layernorm=_StubNorm,
            self_attention=ModuleSpec(module=_StubSubBlock, params={"seed": 1}),
            pre_mlp_layernorm=_StubNorm,
            mlp=ModuleSpec(module=_StubSubBlock, params={"seed": 2}),
        ),
        layer_idx=0,
        pg_collection=_FakePgCollection(),
    )

    x = torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(13))
    out, residual = layer(x, block_residual=None)

    attn_out, _ = layer.self_attention(layer.input_layernorm(x))
    after_attn = x + attn_out
    mlp_out, _ = layer.mlp(layer.pre_mlp_layernorm(after_attn))

    assert residual is None
    torch.testing.assert_close(out, after_attn + mlp_out, rtol=1e-6, atol=1e-7)


def test_gradients_reach_every_layer_parameter():
    """No unwired submodule: every parameter in the stack gets a gradient.

    This is the check that catches a spec-tree wiring bug -- a mixer that is
    built but never called still holds parameters, and only a gradient sweep
    notices.
    """
    block = _build_stub_block(_base_config())
    hidden_states = torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(17))

    block(hidden_states).square().mean().backward()

    missing = [name for name, p in block.named_parameters() if p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"
    non_finite = [name for name, p in block.named_parameters() if not torch.isfinite(p.grad).all()]
    assert not non_finite, f"parameters with non-finite gradient: {non_finite}"


def test_output_head_runs_once_after_the_stack():
    """Perturbing ``attn_res_head`` must change the block output (``:1215-1217``)."""
    block = _build_stub_block(_base_config())
    hidden_states = torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(19))

    baseline = block(hidden_states)
    with torch.no_grad():
        block.attn_res_head.proj_weight.mul_(-3.0)
    assert not torch.equal(baseline, block(hidden_states))

    assert len([m for m in block.modules() if isinstance(m, AttentionResidualHead)]) == 1


# ===========================================================================
# 3. The pipeline seam
# ===========================================================================


@pytest.mark.parametrize("num_filled", [0, 1, 2])
def test_pp_carrier_round_trip(num_filled):
    """``_lower_res_out`` then ``_lift_res_in`` is the identity, at any fill level.

    The carrier is padded to ``num_blocks_max`` so every stage boundary sees one
    static 3-D shape; the receiving stage slices the padding off using a fill
    count it derives from its own layer offset, never from the wire.
    """
    num_blocks_max = 2
    generator = torch.Generator().manual_seed(23)
    hidden_states = torch.randn(SEQ, BATCH, HIDDEN, generator=generator)
    block_residual = torch.randn(SEQ, BATCH, num_filled, HIDDEN, generator=generator)

    packed = _lower_res_out(
        hidden_states, block_residual, post_process=False, num_blocks_max=num_blocks_max
    )
    assert packed.shape == ((1 + num_blocks_max) * SEQ, BATCH, HIDDEN)

    got_hidden, got_residual = _lift_res_in(
        packed, pre_process=False, num_blocks=num_filled, num_blocks_max=num_blocks_max
    )
    assert torch.equal(got_hidden, hidden_states)
    assert torch.equal(got_residual, block_residual)


def test_pp_carrier_is_a_passthrough_on_the_final_stage():
    """The final stage has already collapsed the candidate axis."""
    hidden_states = torch.randn(SEQ, BATCH, HIDDEN)
    residual = torch.randn(SEQ, BATCH, 2, HIDDEN)
    out = _lower_res_out(hidden_states, residual, post_process=True, num_blocks_max=2)
    assert out is hidden_states


def test_first_stage_starts_with_a_genuinely_empty_checkpoint_tensor():
    """``:1190-1192`` -- zero-width, not a zero-filled slot."""
    hidden_states = torch.randn(SEQ, BATCH, HIDDEN)
    got_hidden, residual = _lift_res_in(
        hidden_states, pre_process=True, num_blocks=0, num_blocks_max=2
    )
    assert got_hidden is hidden_states
    assert residual.shape == (SEQ, BATCH, 0, HIDDEN)


def test_pp_carrier_rejects_an_overfull_checkpoint_tensor():
    with pytest.raises(ValueError, match="exceeds"):
        _lower_res_out(
            torch.randn(SEQ, BATCH, HIDDEN),
            torch.randn(SEQ, BATCH, 3, HIDDEN),
            post_process=False,
            num_blocks_max=2,
        )


def test_pp_carrier_rejects_an_unpacked_boundary_tensor():
    with pytest.raises(ValueError, match="divisible"):
        _lift_res_in(
            torch.randn(SEQ + 1, BATCH, HIDDEN), pre_process=False, num_blocks=1, num_blocks_max=2
        )


@pytest.mark.parametrize(
    "num_layers,block_size,expected",
    [
        (8, 4, ENTRY_TRACE),
        (93, 12, None),
        (6, 1, [0, 1, 2, 3, 4, 5]),
    ],
)
def test_fill_count_is_a_pure_function_of_the_layer_index(num_layers, block_size, expected):
    counts = [attn_res_num_blocks_before(i, block_size) for i in range(num_layers)]
    if expected is not None:
        assert counts == expected
    # The count after the whole stack is what the carrier must be sized for.
    assert attn_res_num_blocks_before(num_layers, block_size) == math.ceil(num_layers / block_size)
    assert counts == sorted(counts)
    assert counts[0] == 0


def test_production_shape_appends_eight_checkpoints():
    """93 layers at block size 12 -> appends at 0,12,...,84, i.e. 8 checkpoints."""
    appends = [layer_idx for layer_idx in range(93) if layer_idx % 12 == 0]
    assert appends == [0, 12, 24, 36, 48, 60, 72, 84]
    assert attn_res_num_blocks_before(93, 12) == 8


# ===========================================================================
# 4. The spec tree and model construction
# ===========================================================================

_DEBUG_SHAPE = dict(
    num_layers=8,
    hidden_size=1024,
    num_attention_heads=8,
    kv_channels=32,
    ffn_hidden_size=2048,
    q_lora_rank=256,
    kv_lora_rank=128,
    qk_head_dim=32,
    qk_pos_emb_head_dim=16,
    v_head_dim=32,
    rope_type="rope",
    mscale=1.0,
    mscale_all_dim=0.0,
    apply_rope_fusion=False,
    mla_use_nope=True,
    mla_use_output_gate=True,
    linear_num_key_heads=8,
    linear_num_value_heads=8,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    linear_attention_freq="[1, 1, 1, 0, 1, 1, 1, 0]",
    attn_res_block_size=4,
    num_moe_experts=8,
    moe_router_topk=2,
    moe_ffn_hidden_size=512,
    moe_shared_expert_intermediate_size=512,
    moe_shared_expert_overlap=False,
    routed_expert_hidden_size=512,
    latent_moe_use_norm=True,
    moe_layer_freq=[0] + [1] * 7,
    moe_router_score_function="sigmoid",
    moe_router_enable_expert_bias=True,
    moe_router_pre_softmax=False,
    moe_router_load_balancing_type="seq_aux_loss",
    moe_aux_loss_coeff=1e-3,
    moe_token_dispatcher_type="alltoall",
    moe_grouped_gemm=True,
    moe_permute_fusion=False,
    moe_router_dtype="fp32",
    gated_linear_unit=True,
    activation_func=F.silu,
    bias_activation_fusion=False,
    use_te_activation_func=True,
    activation_situ_beta=4.0,
    activation_situ_linear_beta=25.0,
    add_bias_linear=False,
    normalization="RMSNorm",
    layernorm_epsilon=1e-5,
    qk_layernorm=False,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    params_dtype=torch.float32,
    use_cpu_initialization=True,
    perform_initialization=True,
    kda_backend="eager",
)


def _debug_config(**overrides):
    kwargs = dict(_DEBUG_SHAPE)
    kwargs.update(overrides)
    return KimiK3TransformerConfig(**kwargs)


def _decoder_spec(config):
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_layer_specs import (
        get_kimi_k3_runtime_decoder_spec,
    )

    return get_kimi_k3_runtime_decoder_spec(config)


@pytest.fixture()
def unset_nvte_attention_env(monkeypatch):
    """Clear the TE attention-backend env vars before constructing a model.

    The ROCm image bakes ``NVTE_FLASH_ATTN=0``, and ``LanguageModule.__init__``
    -> ``_set_attention_backend`` asserts those three vars are unset-or-1 for
    the default ``auto`` backend (``language_module.py:103``). Same mitigation
    the diffusion suite already uses
    (``tests/unit_tests/backends/megatron/diffusion/conftest.py:28-47``) and
    that Megatron's own harness applies in ``Utils.initialize_distributed``.
    """
    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def single_rank_parallel_state():
    """A 1-rank gloo group plus Megatron model-parallel state.

    gloo rather than nccl: these tests are about structure, so there is no
    reason to claim a device for collectives.
    """
    import socket

    import torch.distributed as dist
    from megatron.core import parallel_state as ps

    created = False
    if not dist.is_initialized():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
        created = True

    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=1,
    )
    if torch.cuda.is_available():
        from megatron.core.tensor_parallel import random as tp_random

        tp_random.model_parallel_cuda_manual_seed(1234)

    try:
        yield
    finally:
        if ps.model_parallel_is_initialized():
            ps.destroy_model_parallel()
        if created:
            dist.destroy_process_group()


def test_spec_tree_attention_pattern_is_kkkf_kkkf():
    """6 KDA layers and 2 full-attention layers, in the order K K K F K K K F."""
    from primus.backends.megatron.core.transformer.kimi_k3.kimi_delta_attention import (
        KimiDeltaAttention,
    )
    from primus.backends.megatron.core.transformer.kimi_k3.kimi_k3_mla_attention import (
        KimiK3MLASelfAttention,
    )

    layer_specs = _decoder_spec(_debug_config()).submodules.layer_specs
    assert len(layer_specs) == 8

    letters = []
    for i, layer_spec in enumerate(layer_specs):
        assert layer_spec.module is KimiK3Layer
        assert layer_spec.params["layer_idx"] == i
        attention = layer_spec.submodules.self_attention.module
        if attention is KimiDeltaAttention:
            letters.append("K")
        elif attention is KimiK3MLASelfAttention:
            letters.append("F")
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected attention module {attention}")

    assert "".join(letters) == "KKKFKKKF"
    assert letters.count("K") == 6
    assert letters.count("F") == 2
    assert [i for i, s in enumerate(layer_specs) if s.params["is_kda_layer"]] == [0, 1, 2, 4, 5, 6]


def test_spec_tree_mlp_pattern_is_dense_then_moe():
    """Layer 0 is a dense MLP; layers 1-7 are the Stable Latent MoE."""
    from megatron.core.transformer.mlp import MLP

    from primus.backends.megatron.core.transformer.kimi_k3.moe.k3_stable_latent_moe import (
        StableLatentMoE,
    )

    layer_specs = _decoder_spec(_debug_config()).submodules.layer_specs
    assert layer_specs[0].submodules.mlp.module is MLP
    for layer_spec in layer_specs[1:]:
        assert layer_spec.submodules.mlp.module is StableLatentMoE


def test_every_mlp_spec_fills_the_activation_func_slot():
    """``use_te_activation_func`` makes the module slot load-bearing.

    With the slot empty, ``mlp.py:226-229`` falls back to
    ``config.activation_func`` (``F.silu``) applied to the fused
    ``[gate | up]`` tensor, and ``linear_fc2`` then receives double width. The
    shared experts and the grouped routed experts have the same requirement.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.situ_activation import (
        SituActivation,
    )

    layer_specs = _decoder_spec(_debug_config()).submodules.layer_specs

    assert layer_specs[0].submodules.mlp.submodules.activation_func is SituActivation
    moe_submodules = layer_specs[1].submodules.mlp.submodules
    assert moe_submodules.shared_experts.submodules.activation_func is SituActivation
    assert moe_submodules.experts.submodules.activation_func is SituActivation


def test_spec_tree_wires_the_mixers_and_exactly_one_head():
    spec = _decoder_spec(_debug_config())

    # Layer 0 skips the pre-attention mix, so it carries no attn_res_mixer.
    assert spec.submodules.layer_specs[0].submodules.attn_res_mixer is None
    for layer_spec in spec.submodules.layer_specs[1:]:
        assert layer_spec.submodules.attn_res_mixer.module is AttentionResidualMixer
    for layer_spec in spec.submodules.layer_specs:
        assert layer_spec.submodules.mlp_res_mixer.module is AttentionResidualMixer

    assert spec.submodules.attn_res_head.module is AttentionResidualHead
    assert spec.module is KimiK3TransformerBlock


def test_spec_tree_omits_the_mixers_when_the_mechanism_is_off():
    spec = _decoder_spec(_debug_config(attn_res_block_size=None))
    assert spec.submodules.attn_res_head is None
    assert spec.submodules.layer_specs[0].submodules.mlp_res_mixer is None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason=(
        "The MoE layers cannot be built without a visible GPU: upstream's TopKRouter "
        "hard-codes device=torch.cuda.current_device() for its expert-bias buffers "
        "(router.py:172-189)."
    ),
)
def test_debug_model_constructs_on_cpu(single_rank_parallel_state, unset_nvte_attention_env):
    """The 8-layer debug model builds with ``use_cpu_initialization``.

    No ``.cuda()`` on the model and no launcher — the same property that makes
    the DeepSeek-V4 modules CPU-instantiable (``deepseek_v4_block.py:825-830``).

    The device assertion is scoped to the modules this work package owns. Two
    upstream components place their own storage on the accelerator regardless:
    ``TopKRouter``'s expert-bias buffers (``router.py:172-189``) and KDA's
    ``A_log`` / ``dt_bias`` (``kimi_delta_attention.py:98-105``).
    """
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_model import KimiK3Model
    from primus.backends.megatron.core.transformer.kimi_k3.kimi_delta_attention import (
        KimiDeltaAttention,
    )
    from primus.backends.megatron.core.transformer.kimi_k3.kimi_k3_mla_attention import (
        KimiK3MLASelfAttention,
    )

    config = _debug_config()
    model = KimiK3Model(
        config=config,
        transformer_layer_spec=_decoder_spec(config),
        vocab_size=1024,
        max_sequence_length=512,
        pre_process=True,
        post_process=True,
        share_embeddings_and_output_weights=False,
    )

    assert len(model.decoder.layers) == 8
    kinds = "".join(
        "K" if isinstance(layer.self_attention, KimiDeltaAttention) else "F"
        for layer in model.decoder.layers
    )
    assert kinds == "KKKFKKKF"
    assert all(
        isinstance(model.decoder.layers[i].self_attention, KimiK3MLASelfAttention) for i in (3, 7)
    )
    assert isinstance(model.decoder.attn_res_head, AttentionResidualHead)
    assert model.decoder.num_blocks_max == 2
    assert model.decoder.attn_res_block_count_trace() == ENTRY_TRACE

    # NoPE invariant: every rotary module in the stack has a zero-width table.
    for name, module in model.named_modules():
        inv_freq = getattr(module, "inv_freq", None)
        if inv_freq is not None:
            assert inv_freq.numel() == 0, f"{name} would apply a rotation"

    for name, module in model.named_modules():
        if isinstance(module, AttentionResidualMixer):
            for param_name, param in module.named_parameters():
                assert param.device.type == "cpu", f"{name}.{param_name} is on {param.device}"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TE's linear modules want a device context even under use_cpu_initialization",
)
def test_dense_only_debug_model_constructs_without_moe(
    single_rank_parallel_state, unset_nvte_attention_env
):
    """The same stack with the MoE disabled, i.e. no upstream router at all."""
    from megatron.core.transformer.mlp import MLP

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_model import KimiK3Model

    config = _debug_config(
        num_moe_experts=None,
        moe_layer_freq=1,
        routed_expert_hidden_size=None,
        moe_router_enable_expert_bias=False,
        moe_shared_expert_intermediate_size=None,
        moe_ffn_hidden_size=None,
        moe_grouped_gemm=False,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
    )
    model = KimiK3Model(
        config=config,
        transformer_layer_spec=_decoder_spec(config),
        vocab_size=1024,
        max_sequence_length=512,
    )

    assert len(model.decoder.layers) == 8
    assert all(isinstance(layer.mlp, MLP) for layer in model.decoder.layers)
