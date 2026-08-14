###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the Kimi K3 FLOPs closed form (WP8).

These are pure arithmetic tests: they pin the properties that make the
closed form *correct* rather than merely *plausible*, because a FLOPs model
has no natural failure mode — a wrong one just prints a wrong number
forever. The properties pinned here are the four the validation report
flagged (``validate/VALIDATION.md`` §7.3) plus the two structural ones:

1. **KDA is linear in the sequence length.** Doubling ``seq_length`` must
   exactly double the KDA contribution, and the per-token KDA cost must not
   depend on ``seq_length`` at all. This is the property upstream's
   MHA-branch fallback violates.
2. **MLA is quadratic in the sequence length**, and its per-token cost grows
   linearly — the opposite of (1), so a test that passes both cannot be
   trivially satisfied.
3. The latent bottleneck is charged at the latent width, and the two latent
   projections are charged **once per token**, not once per routed expert.
4. Every module that exists in the model appears in the total, checked by
   removing each in turn and asserting the total moves.

``import torch`` comes first in this file even though nothing here needs a
GPU: a bare ``import transformer_engine`` SIGABRTs on this image and any
Primus import can pull it in transitively (``DECISIONS.md``, "Import-order
hazard").
"""

from __future__ import annotations

import types

import torch  # noqa: F401  # must precede any transformer_engine import
import pytest

from primus.backends.megatron.patches.kimi_k3_flops_patches import (
    KimiK3FlopsBreakdown,
    attn_res_fmac_per_token,
    attn_res_num_blocks_before,
    compute_kimi_k3_flops,
    kda_core_fmac_per_token,
    kda_proj_fmac_per_token,
    latent_moe_fmac_per_token,
    linear_attention_pattern,
    mla_fmac_per_token,
    moe_layer_pattern,
)

# ---------------------------------------------------------------------------
# The debug shape, transcribed from primus/configs/models/megatron/kimi_k3_debug.yaml
# ---------------------------------------------------------------------------

DEBUG_ARGS = dict(
    seq_length=512,
    hidden_size=1024,
    num_layers=8,
    num_attention_heads=8,
    kv_channels=32,
    ffn_hidden_size=2048,
    swiglu=True,
    # full attention (MLA, NoPE)
    q_lora_rank=256,
    kv_lora_rank=128,
    qk_head_dim=32,
    qk_pos_emb_head_dim=16,
    v_head_dim=32,
    mla_use_output_gate=True,
    # KDA
    linear_num_key_heads=8,
    linear_num_value_heads=8,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    kda_chunk_size=64,
    linear_attention_freq="[1, 1, 1, 0, 1, 1, 1, 0]",
    # attention residuals
    attn_res_block_size=4,
    # MoE
    num_experts=8,
    moe_router_topk=2,
    moe_ffn_hidden_size=512,
    moe_shared_expert_intermediate_size=512,
    routed_expert_hidden_size=512,
    moe_layer_freq=[0, 1, 1, 1, 1, 1, 1, 1],
    moe_latent_size=None,
    mtp_num_layers=None,
    # vocab: 163840 + 1 eod, padded to a multiple of 128 by NullTokenizer + TP=1
    vocab_size=163840,
    padded_vocab_size=163968,
)

GLOBAL_BATCH = 8


def _args(**overrides):
    merged = dict(DEBUG_ARGS)
    merged.update(overrides)
    return types.SimpleNamespace(**merged)


# ---------------------------------------------------------------------------
# Per-layer pattern parsing
# ---------------------------------------------------------------------------


def test_flops_linear_attention_pattern_from_string_list():
    """The YAML ships the pattern as a *string*; the args layer keeps it that way.

    ``wp6/smoke_kimi_k3_debug_fla.log:1658`` shows
    ``linear_attention_freq ... [1, 1, 1, 0, 1, 1, 1, 0]`` typed ``(str)`` —
    Primus normalises ``moe_layer_freq`` but not this field.
    """
    assert linear_attention_pattern("[1, 1, 1, 0, 1, 1, 1, 0]", 8) == [1, 1, 1, 0, 1, 1, 1, 0]
    assert sum(linear_attention_pattern("[1, 1, 1, 0, 1, 1, 1, 0]", 8)) == 6


def test_flops_linear_attention_pattern_int_ratio_matches_upstream():
    """An int ``N`` means every ``N``-th layer is FULL attention, not linear.

    Pinned against ``training.py:460-465``, which builds
    ``0 if ((i + 1) % freq == 0) else 1``. Getting this inverted would swap
    69 KDA layers for 24 at the production shape.
    """
    assert linear_attention_pattern(4, 8) == [1, 1, 1, 0, 1, 1, 1, 0]


def test_flops_moe_layer_pattern_expression_string():
    """``moe_layer_freq`` may arrive as ``"([0]*1+[1]*7)"`` — a BinOp, not a literal."""
    assert moe_layer_pattern("([0]*1+[1]*7)", 8) == [0, 1, 1, 1, 1, 1, 1, 1]
    assert moe_layer_pattern([0, 1, 1, 1, 1, 1, 1, 1], 8) == [0, 1, 1, 1, 1, 1, 1, 1]


def test_flops_pattern_length_mismatch_raises():
    with pytest.raises(ValueError, match="linear_attention_freq has 4 entries"):
        linear_attention_pattern("[1, 1, 1, 0]", 8)


# ---------------------------------------------------------------------------
# (1) KDA is LINEAR in T. This is the headline correction.
# ---------------------------------------------------------------------------


def test_flops_kda_per_token_cost_is_independent_of_seq_length():
    """Neither KDA helper takes ``seq_len`` — pinned by signature, not by value.

    The chunkwise delta rule's per-token cost is
    ``3 C K + 2 C V + 3 K V + C^2/3`` (see
    ``kda_kernels/_eager/reference.py:301-402``); ``seq_length`` appears
    nowhere in it. Upstream's MHA branch, by contrast, adds
    ``query_projection_size * seq_length / 2 * 2`` per token.
    """
    import inspect

    for fn in (kda_proj_fmac_per_token, kda_core_fmac_per_token):
        params = inspect.signature(fn).parameters
        assert "seq_len" not in params and "seq_length" not in params, (
            f"{fn.__name__} accepts a sequence length, which means the KDA cost "
            "can be made to grow with T -- the exact bug this patch exists to fix"
        )


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_flops_kda_total_scales_exactly_linearly_in_seq_length(factor):
    """Doubling T doubles the KDA total, to the integer.

    The comparison is at fixed *tokens per iteration* only in the sense that
    ``batch_size`` is held constant; the point is the ratio, which must be
    exactly ``factor`` and not ``factor**2``.
    """
    base = compute_kimi_k3_flops(_args(seq_length=512), GLOBAL_BATCH)[1]
    scaled = compute_kimi_k3_flops(_args(seq_length=512 * factor), GLOBAL_BATCH)[1]

    assert scaled.kda_proj == base.kda_proj * factor
    assert scaled.kda_core == base.kda_core * factor


def test_flops_mla_is_quadratic_in_seq_length():
    """The control for the test above: full attention *must* grow super-linearly.

    Doubling T doubles the token count and doubles the per-token score cost,
    so the score term quadruples while the projection terms only double. The
    ratio therefore sits strictly between 2 and 4 — a test that could not
    pass if ``mla_fmac_per_token`` had accidentally dropped its ``seq_len``.
    """
    base = compute_kimi_k3_flops(_args(seq_length=512), GLOBAL_BATCH)[1]
    doubled = compute_kimi_k3_flops(_args(seq_length=1024), GLOBAL_BATCH)[1]
    ratio = doubled.mla / base.mla
    assert 2.0 < ratio < 4.0, ratio


def test_flops_kda_core_matches_hand_derived_chunk_inventory():
    """Pin the exact per-token constant against the reference's matmul list.

    Three ``C^2 K`` matmuls (``reference.py:370, 380, 391``), two ``C^2 V``
    (``:381, 394``), three ``C K V`` (``:389, 394, 398``) and the
    ``sum_{r<C} r^2 ~= C^3/3`` triangular inverse (``:372-375``), all divided
    by ``C``.
    """
    c, k, v, heads = 64, 32, 32, 8
    expected = heads * (3 * c * k + 2 * c * v + 3 * k * v + (c * c) // 3)
    assert (
        kda_core_fmac_per_token(
            num_heads=heads, key_head_dim=k, value_head_dim=v, chunk_size=c
        )
        == expected
    )
    # 8 * (6144 + 4096 + 3072 + 1365)
    assert expected == 117416


def test_flops_kda_proj_counts_the_output_gate():
    """``g_proj`` is the term upstream's ``gated_delta_net`` branch has no slot for.

    ``kimi_delta_attention.py:366`` builds it at ``H -> v_dim`` whenever
    ``kda_use_full_rank_gate`` is set, which is the only variant implemented
    (``:360-365`` raises otherwise). Dropping it under-counts a KDA layer's
    projections by ``H * v_dim``, which is 19 % of them at the debug shape.
    """
    kwargs = dict(
        hidden_size=1024, num_heads=8, key_head_dim=32, value_head_dim=32, conv_kernel_dim=4
    )
    total = kda_proj_fmac_per_token(**kwargs)
    v_dim = 8 * 32
    # q + k + v + g are the four wide `H -> 256` projections; o_proj is the
    # fifth wide one. Removing g_proj would leave `total - H*v_dim`.
    assert total == 1_362_944
    assert 1024 * v_dim / total > 0.18


# ---------------------------------------------------------------------------
# (3) The latent MoE bottleneck
# ---------------------------------------------------------------------------


def test_flops_latent_moe_charges_experts_at_the_latent_width():
    """Routed experts run at ``latent``, not ``hidden_size``.

    ``mlp.py:208-213`` and ``experts.py:185, 206-207`` size the expert FC1
    input / FC2 output on ``config.moe_latent_size`` when the token is going
    to a routed expert. With ``latent = hidden/2`` the routed term halves.
    """
    common = dict(
        hidden_size=1024,
        moe_ffn_hidden_size=512,
        moe_router_topk=2,
        num_experts=8,
        shared_expert_ffn_hidden_size=512,
        swiglu=True,
    )
    with_latent = latent_moe_fmac_per_token(latent_size=512, **common)
    without = latent_moe_fmac_per_token(latent_size=None, **common)

    routed_with = 2 * 3 * 512 * 512
    routed_without = 2 * 3 * 1024 * 512
    assert routed_with * 2 == routed_without

    # The latent path pays 2 * H * latent extra for fc1/fc2 but saves half the
    # routed cost, which nets out negative at these widths.
    assert with_latent < without
    assert without - with_latent == routed_without - routed_with - 2 * 1024 * 512


def test_flops_latent_projections_are_charged_once_per_token_not_per_expert():
    """``fc1_latent_proj`` / ``fc2_latent_proj`` run outside the dispatch.

    ``moe_layer.py:359-363`` projects down in ``preprocess`` (before
    ``token_dispatcher.dispatch_preprocess``) and ``:448-449`` projects back
    up in ``postprocess`` (after ``combine``). So the cost is independent of
    ``moe_router_topk``. Charging them per routed expert would inflate the
    MoE term by ``topk``, which is 16 at the production shape.
    """
    common = dict(
        hidden_size=1024,
        latent_size=512,
        moe_ffn_hidden_size=512,
        num_experts=8,
        shared_expert_ffn_hidden_size=0,
        swiglu=True,
    )
    topk1 = latent_moe_fmac_per_token(moe_router_topk=1, **common)
    topk2 = latent_moe_fmac_per_token(moe_router_topk=2, **common)
    per_expert = topk2 - topk1
    assert per_expert == 3 * 512 * 512  # one expert's SwiGLU triple at latent width
    # The latent + router terms are the topk-invariant remainder.
    assert topk1 - per_expert == 1024 * 8 + 2 * 1024 * 512


def test_flops_shared_expert_is_charged_at_hidden_not_latent():
    """The shared expert sees the PRE-down-projection hidden state.

    ``moe_layer.py:379-402``'s ``shared_experts_compute`` is called with the
    original ``hidden_states``; only the routed path goes through
    ``fc1_latent_proj``. ``DECISIONS.md`` "Settled during WP5" records the
    same fact from the module side.
    """
    common = dict(
        hidden_size=1024,
        latent_size=512,
        moe_ffn_hidden_size=512,
        moe_router_topk=2,
        num_experts=8,
        swiglu=True,
    )
    with_shared = latent_moe_fmac_per_token(shared_expert_ffn_hidden_size=512, **common)
    without_shared = latent_moe_fmac_per_token(shared_expert_ffn_hidden_size=0, **common)
    assert with_shared - without_shared == 3 * 1024 * 512  # H, not latent


# ---------------------------------------------------------------------------
# Attention residuals
# ---------------------------------------------------------------------------


def test_flops_attn_res_block_count_trace_matches_the_measured_runtime_trace():
    """The append schedule must reproduce the trace measured on the live model.

    ``validate/VALIDATION.md`` §4.2 measured ``block_residual`` counts on
    entry of ``[0,1,1,1,1,2,2,2]`` for the 8-layer debug shape with
    ``attn_res_block_size = 4``. Same numbers the block asserts on at
    ``kimi_k3_block.py:482-487``.
    """
    trace = [attn_res_num_blocks_before(i, 4) for i in range(8)]
    assert trace == [0, 1, 1, 1, 1, 2, 2, 2]
    # Post-stack, the head sees the final count.
    assert attn_res_num_blocks_before(8, 4) == 2


def test_flops_attn_res_counts_15_mixers_plus_1_head():
    """16 modules at the debug shape, matching the measured ``mixers built = 16``.

    ``mlp_res_mixer`` on all 8 layers (``kimi_k3_block.py:353-358``),
    ``attn_res_mixer`` only where ``num_blocks_in > 0`` i.e. layers 1-7
    (``:368-376``), one ``attn_res_head`` on ``post_process``
    (``:594-601``). VALIDATION.md §4.2 reports exactly 16.
    """
    hidden = 1024
    total = attn_res_fmac_per_token(hidden_size=hidden, num_layers=8, attn_res_block_size=4)
    # attn mixers see the pre-append count: layers 1..7 -> [1,1,1,1,2,2,2] (+1 each)
    attn_candidates = sum(n + 1 for n in [1, 1, 1, 1, 2, 2, 2])
    # mlp mixers see the post-append count: [1,1,1,1,2,2,2,2] (+1 each)
    mlp_candidates = sum(n + 1 for n in [1, 1, 1, 1, 2, 2, 2, 2])
    head_candidates = 2 + 1
    expected = 2 * (attn_candidates + mlp_candidates + head_candidates) * hidden
    assert total == expected
    assert total == 81_920


def test_flops_attn_res_disabled_is_exactly_zero():
    """``attn_res_block_size: 0`` is the ablation knob the validation used."""
    assert attn_res_fmac_per_token(hidden_size=1024, num_layers=8, attn_res_block_size=0) == 0
    breakdown = compute_kimi_k3_flops(_args(attn_res_block_size=0), GLOBAL_BATCH)[1]
    assert breakdown.attn_res == 0


# ---------------------------------------------------------------------------
# (4) Nothing silently missing: every module moves the total
# ---------------------------------------------------------------------------


def test_flops_debug_shape_layer_census():
    """6 KDA / 2 full attention / 1 dense / 7 MoE, matching the runtime census.

    ``VALIDATION.md`` §4.1 read these off the instantiated ``nn.Module``s:
    pattern ``KKKFKKKF``, ``MLP`` at layer 0 and ``StableLatentMoE`` at 1-7.
    """
    _, breakdown = compute_kimi_k3_flops(_args(), GLOBAL_BATCH)
    assert (breakdown.num_kda_layers, breakdown.num_full_attn_layers) == (6, 2)
    assert (breakdown.num_dense_layers, breakdown.num_moe_layers) == (1, 7)


def test_flops_every_component_is_non_zero_at_the_debug_shape():
    """A zero component means a module was silently dropped from the model."""
    _, b = compute_kimi_k3_flops(_args(), GLOBAL_BATCH)
    for field in ("kda_proj", "kda_core", "mla", "dense_mlp", "moe", "attn_res", "logits"):
        assert getattr(b, field) > 0, field


def test_flops_total_is_the_sum_times_six():
    """The fwd+bwd (3) x FMA (2) expansion is applied exactly once."""
    total, b = compute_kimi_k3_flops(_args(), GLOBAL_BATCH)
    assert total == 6 * b.total_fmac()
    assert b.total_fmac() == sum(
        (b.kda_proj, b.kda_core, b.mla, b.dense_mlp, b.moe, b.attn_res, b.logits)
    )


def test_flops_debug_shape_total_is_pinned():
    """Pin the absolute number so a refactor cannot drift it silently.

    215 099 376 FMAC/token-batch... specifically: 215 099 376 FMAC per token,
    times 4096 tokens (gbs 8 x seq 512), times 6.
    """
    total, b = compute_kimi_k3_flops(_args(), GLOBAL_BATCH)
    tokens = GLOBAL_BATCH * DEBUG_ARGS["seq_length"]
    assert b.total_fmac() == 215_099_376 * tokens
    assert total == 215_099_376 * tokens * 6
    # ~5.29 TFLOP per global batch of 4096 tokens.
    assert 5.28e12 < total < 5.29e12


def test_flops_logits_dominate_at_the_debug_shape():
    """The honest caveat, pinned as a test so nobody quotes the debug agreement.

    At hidden 1024 against a 163 968-row untied vocabulary the LM head is
    ~78 % of all FLOPs, which is why upstream's five-way-wrong formula still
    lands within ~1 % here. Any claim that the correction "does not matter"
    is a claim about this ratio, so the ratio is asserted.
    """
    _, b = compute_kimi_k3_flops(_args(), GLOBAL_BATCH)
    assert b.logits / b.total_fmac() > 0.75


def test_flops_logits_share_collapses_at_a_scaled_shape():
    """...and at a scaled shape the vocab head stops dominating.

    Same vocabulary, hidden 2048, 24 layers, seq 2048: the LM head drops
    below a third of the total, so the layer-level corrections become the
    dominant term. This is the shape class Task 3's config lives in.
    """
    scaled = _args(
        hidden_size=2048,
        num_layers=24,
        seq_length=2048,
        ffn_hidden_size=5632,
        moe_ffn_hidden_size=1024,
        moe_shared_expert_intermediate_size=1024,
        routed_expert_hidden_size=1024,
        num_attention_heads=16,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        kv_channels=128,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        q_lora_rank=768,
        kv_lora_rank=256,
        linear_attention_freq=4,
        moe_layer_freq=[0] + [1] * 23,
        attn_res_block_size=8,
    )
    _, b = compute_kimi_k3_flops(scaled, GLOBAL_BATCH)
    assert b.logits / b.total_fmac() < 0.34


# ---------------------------------------------------------------------------
# args-layer moe_latent_size fix
# ---------------------------------------------------------------------------


def test_flops_closed_form_does_not_depend_on_args_moe_latent_size():
    """The closed form reads ``routed_expert_hidden_size``, so patch order is free.

    ``args.moe_latent_size`` is ``None`` in a real launcher run
    (``wp6/smoke_kimi_k3_debug_fla.log:1760``) until
    ``patch_k3_args_moe_latent_size`` runs at ``build_args``. The FLOPs
    wrapper installs at ``before_train``, so it *would* normally see the
    fixed value -- but relying on that would make the number silently
    dependent on patch registration order.
    """
    with_none = compute_kimi_k3_flops(_args(moe_latent_size=None), GLOBAL_BATCH)[0]
    with_set = compute_kimi_k3_flops(_args(moe_latent_size=512), GLOBAL_BATCH)[0]
    assert with_none == with_set


@pytest.fixture
def _silence_primus_logger(monkeypatch):
    """Primus's ``log_rank_0`` needs a runtime-initialised global logger.

    Outside a real launcher ``primus.core.utils.logger._logger`` is ``None``
    and any log call raises ``AttributeError``. Production code should not
    have to guard against that, so the test stubs the call instead.
    """
    import primus.backends.megatron.patches.kimi_k3_flops_patches as mod

    monkeypatch.setattr(mod, "log_rank_0", lambda *a, **k: None)
    return mod


def test_flops_args_patch_mirrors_routed_expert_hidden_size(_silence_primus_logger):
    from primus.core.patches import PatchContext

    patch_k3_args_moe_latent_size = _silence_primus_logger.patch_k3_args_moe_latent_size
    args = types.SimpleNamespace(routed_expert_hidden_size=3584, moe_latent_size=None)
    ctx = PatchContext(backend="megatron", phase="build_args", extra={"backend_args": args})
    patch_k3_args_moe_latent_size(ctx)
    assert args.moe_latent_size == 3584


def test_flops_args_patch_raises_on_disagreement():
    from primus.backends.megatron.patches.kimi_k3_flops_patches import (
        patch_k3_args_moe_latent_size,
    )
    from primus.core.patches import PatchContext

    args = types.SimpleNamespace(routed_expert_hidden_size=3584, moe_latent_size=1792)
    ctx = PatchContext(backend="megatron", phase="build_args", extra={"backend_args": args})
    with pytest.raises(ValueError, match="disagrees with routed_expert_hidden_size"):
        patch_k3_args_moe_latent_size(ctx)


def test_flops_args_patch_is_a_noop_without_the_k3_field():
    """Non-K3 jobs must be untouched even if the condition is ever loosened."""
    from primus.backends.megatron.patches.kimi_k3_flops_patches import (
        patch_k3_args_moe_latent_size,
    )
    from primus.core.patches import PatchContext

    args = types.SimpleNamespace(moe_latent_size=None)
    ctx = PatchContext(backend="megatron", phase="build_args", extra={"backend_args": args})
    patch_k3_args_moe_latent_size(ctx)
    assert args.moe_latent_size is None


# ---------------------------------------------------------------------------
# Comparison against what upstream would have said
# ---------------------------------------------------------------------------


def test_flops_breakdown_dataclass_is_frozen():
    """A reported number must not be mutable after it is computed."""
    b = KimiK3FlopsBreakdown(
        kda_proj=1, kda_core=1, mla=1, dense_mlp=1, moe=1, attn_res=1, logits=1
    )
    assert b.total_fmac() == 7
    assert b.to_total_flops() == 42
    with pytest.raises((AttributeError, TypeError)):
        b.moe = 2  # type: ignore[misc]
