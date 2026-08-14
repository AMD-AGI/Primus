###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The Kimi K3 layer / block assembly and the attention-residual bookkeeping.

:mod:`test_attention_residual` pins the mixer's arithmetic; this file pins
the wiring around it -- which is where an integration bug actually lives:

* the spec tree puts the right attention module on every layer
  (``K K K F K K K F`` at the debug shape) and the right FFN;
* the per-layer checkpoint bookkeeping matches
  ``KimiDecoderLayer._forward_attn_residual``
  (``modeling_kimi_linear.py:984-1046``) -- the append schedule, the
  ``prefix_sum`` reset, and ``block_residual.shape``'s growth;
* every parameter receives a gradient, which is the cheapest test for an
  unwired submodule and the one this work package is most likely to break;
* the pipeline seam round-trips.

The bookkeeping tests carry their own transcription of the reference's
control flow (:func:`reference_layer_trace`) rather than reading it off the
implementation, so a change to the append schedule fails rather than
silently redefining the expectation.

Shape: the 8-layer ``kimi_k3_debug.yaml`` geometry with a short sequence.
``attn_res_block_size=4`` over 8 layers is the minimum that exercises a
*growing* ``block_residual`` axis -- two appends, so the mixer sees 1, 2
and finally 3 candidates.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import pytest
import torch

NUM_LAYERS = 8
HIDDEN = 256
SEQ = 64
BATCH = 2
ATTN_RES_BLOCK_SIZE = 4
KDA_PATTERN = [1, 1, 1, 0, 1, 1, 1, 0]
EXPECTED_ATTENTION_PATTERN = "KKKFKKKF"


# ---------------------------------------------------------------------------
# The oracle for the bookkeeping: modeling_kimi_linear.py:984-1046
# ---------------------------------------------------------------------------


def reference_layer_trace(
    num_layers: int, block_size: int
) -> Tuple[List[int], List[int], List[bool], List[bool]]:
    """Replay ``_forward_attn_residual``'s control flow symbolically.

    Tracks only the shapes and the ``None``-ness of ``prefix_sum``, which
    is all the schedule is:

    ```
    prefix_sum = hidden_states                       # :985
    if block_residual.shape[1] > 0:                  # :987
        <pre-attention mix>
    if layer_idx % attn_res_block_size == 0:         # :995
        block_residual = cat([block_residual, ...])  # :996
        prefix_sum = None                            # :998
    ```

    Returns:
        ``(nb_on_entry, nb_on_exit, appends, pre_attn_mix_runs)``.
    """
    nb_on_entry: List[int] = []
    nb_on_exit: List[int] = []
    appends: List[bool] = []
    mixes: List[bool] = []

    num_blocks = 0  # block_residual starts genuinely empty (:1190-1192)
    for layer_idx in range(num_layers):
        nb_on_entry.append(num_blocks)
        mixes.append(num_blocks > 0)
        appended = layer_idx % block_size == 0
        appends.append(appended)
        if appended:
            num_blocks += 1
        nb_on_exit.append(num_blocks)
    return nb_on_entry, nb_on_exit, appends, mixes


def test_reference_trace_is_the_documented_one():
    """Guard the oracle itself against a transcription slip."""
    entry, exit_, appends, mixes = reference_layer_trace(NUM_LAYERS, ATTN_RES_BLOCK_SIZE)
    assert entry == [0, 1, 1, 1, 1, 2, 2, 2]
    assert exit_ == [1, 1, 1, 1, 2, 2, 2, 2]
    assert appends == [True, False, False, False, True, False, False, False]
    assert mixes == [False, True, True, True, True, True, True, True]


def test_production_shape_appends_eight_checkpoints():
    """93 layers at ``attn_res_block_size=12`` -> 8 blocks (DESIGN.md 3.5.3)."""
    _, exit_, appends, _ = reference_layer_trace(93, 12)
    assert [i for i, a in enumerate(appends) if a] == [0, 12, 24, 36, 48, 60, 72, 84]
    assert exit_[-1] == 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mpu_tp1():
    """A 1-rank process group plus Megatron model-parallel state.

    Required, not optional: ``KimiDeltaAttention`` asserts it was given a
    ``pg_collection`` (``kimi_delta_attention.py:251``) and every
    projection needs a TP group. Same fixture shape as
    ``test_kda_module.py:42-76``.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29583")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo", world_size=1, rank=0
        )
        created = True
    try:
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )
        if torch.cuda.is_available():
            model_parallel_cuda_manual_seed(1234)
        yield
    finally:
        if created:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def _make_config(
    *,
    num_layers: int = NUM_LAYERS,
    attn_res_block_size: Optional[int] = ATTN_RES_BLOCK_SIZE,
    kda_pattern: Optional[List[int]] = None,
    num_moe_experts: int = 8,
    params_dtype: Optional[torch.dtype] = None,
):
    """The ``kimi_k3_debug.yaml`` geometry, narrowed for test speed.

    ``hidden_size`` is 256 rather than the yaml's 1024 and the head dims
    scale with it; every *structural* field (the interleave, the block
    size, the dense-then-MoE split, the latent bottleneck ratio) is the
    yaml's, because those are what is under test.
    """
    import torch.nn.functional as F

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
        KimiK3TransformerConfig,
    )

    if params_dtype is None:
        params_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if kda_pattern is None:
        kda_pattern = KDA_PATTERN[:num_layers]

    return KimiK3TransformerConfig(
        num_layers=num_layers,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        ffn_hidden_size=512,
        kv_channels=32,
        # MLA geometry, at the debug yaml's ratios
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_head_dim=32,
        v_head_dim=32,
        qk_pos_emb_head_dim=16,
        rope_type="rope",
        mscale=1.0,
        mscale_all_dim=0.0,
        apply_rope_fusion=False,
        multi_latent_attention=False,
        # KDA geometry
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        linear_attention_freq=list(kda_pattern),
        kda_backend="eager",
        kda_chunk_size=64,
        attn_res_block_size=attn_res_block_size,
        # MoE: layer 0 dense, the rest routed, latent bottleneck at hidden/2
        num_moe_experts=num_moe_experts or None,
        moe_router_topk=2,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        routed_expert_hidden_size=HIDDEN // 2,
        latent_moe_use_norm=True,
        moe_layer_freq=[0] + [1] * (num_layers - 1),
        moe_router_score_function="sigmoid",
        moe_router_pre_softmax=False,
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=1e-3,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_shared_expert_overlap=False,
        moe_permute_fusion=False,
        # situ reaches the MLP only through the activation_func module slot
        gated_linear_unit=True,
        activation_func=F.silu,
        use_te_activation_func=True,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        bias_activation_fusion=False,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        params_dtype=params_dtype,
        bf16=params_dtype is torch.bfloat16,
        init_method_std=0.02,
        use_cpu_initialization=not torch.cuda.is_available(),
        perform_initialization=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


def _build_block(config, *, pre_process: bool = True, post_process: bool = True):
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.spec_utils import build_module

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_layer_specs import (
        get_kimi_k3_runtime_decoder_spec,
    )

    spec = get_kimi_k3_runtime_decoder_spec(config)
    return build_module(
        spec,
        config=config,
        pre_process=pre_process,
        post_process=post_process,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# The spec tree
# ---------------------------------------------------------------------------


def test_spec_tree_attention_pattern(mpu_tp1):
    """6 KDA + 2 full-attention layers, in the order ``K K K F K K K F``."""
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_layer_specs import (
        get_kimi_k3_runtime_decoder_spec,
    )

    config = _make_config()
    spec = get_kimi_k3_runtime_decoder_spec(config)
    layer_specs = spec.submodules.layer_specs

    assert len(layer_specs) == NUM_LAYERS
    pattern = "".join("K" if s.params["is_kda_layer"] else "F" for s in layer_specs)
    assert pattern == EXPECTED_ATTENTION_PATTERN, pattern
    assert pattern.count("K") == 6
    assert pattern.count("F") == 2
    assert [s.params["layer_idx"] for s in layer_specs] == list(range(NUM_LAYERS))


def test_spec_tree_attention_modules_and_ffn(mpu_tp1):
    """The pattern has to reach the built modules, not just the spec params."""
    from primus.backends.megatron.core.transformer.kimi_k3.kimi_delta_attention import (
        KimiDeltaAttention,
    )
    from primus.backends.megatron.core.transformer.kimi_k3.kimi_k3_mla_attention import (
        KimiK3MLASelfAttention,
    )
    from primus.backends.megatron.core.transformer.kimi_k3.moe.k3_stable_latent_moe import (
        StableLatentMoE,
    )

    block = _build_block(_make_config())

    assert [layer.layer_idx for layer in block.layers] == list(range(NUM_LAYERS))
    for idx, layer in enumerate(block.layers):
        want_kda = EXPECTED_ATTENTION_PATTERN[idx] == "K"
        assert layer.is_kda_layer is want_kda, idx
        expected_cls = KimiDeltaAttention if want_kda else KimiK3MLASelfAttention
        assert isinstance(layer.self_attention, expected_cls), (idx, type(layer.self_attention))
        # first_k_dense_replace == 1: layer 0 dense, layers 1-7 routed.
        assert isinstance(layer.mlp, StableLatentMoE) is (idx > 0), (idx, type(layer.mlp))
        # 1-based, as Megatron's aux-loss tracker expects (router.py:464-479).
        assert layer.layer_number == idx + 1


def test_every_mlp_fills_the_activation_func_slot(mpu_tp1):
    """``use_te_activation_func`` makes the module slot the only live hook.

    With the slot empty, ``MLP.__init__`` falls back to
    ``config.activation_func`` -- ``F.silu`` -- applied to the fused
    ``[gate | up]`` tensor (``mlp.py:226-229``), which is the wrong
    activation and hands ``linear_fc2`` double the width. It builds and
    trains, so only an explicit check catches it.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.situ_activation import (
        SituActivation,
    )

    config = _make_config()
    assert config.use_te_activation_func
    block = _build_block(config)

    dense_mlp = block.layers[0].mlp
    assert isinstance(dense_mlp.activation_func, SituActivation), type(dense_mlp.activation_func)
    assert dense_mlp.activation_func.beta == 4.0
    assert dense_mlp.activation_func.linear_beta == 25.0

    # The routed experts read the same slot (experts.py:196-199).
    routed = block.layers[1].mlp.experts
    assert isinstance(routed.activation_func, SituActivation), type(routed.activation_func)


def test_no_rotary_rotation_anywhere(mpu_tp1):
    """NoPE invariant: any rotary module present must be zero-width.

    K3 keeps ``qk_pos_emb_head_dim`` at its released width and disables the
    rotation with a zero-width frequency table instead, so the assertion
    is not "no rotary module exists" but "no rotary module rotates".
    """
    block = _build_block(_make_config())
    rotary = [(n, m) for n, m in block.named_modules() if "rotary" in n.lower()]
    # Only the two full-attention layers carry one.
    assert len(rotary) == 2, [n for n, _ in rotary]
    for name, module in rotary:
        # inv_freq is arange(0, dim, 2), so a zero-width table has no entries
        # (rotary_pos_embedding.py:79-81). This is the same spelling WP4's own
        # NoPE test uses (test_k3_mla_nope.py:504).
        inv_freq = getattr(module, "inv_freq", None)
        assert inv_freq is not None, name
        assert inv_freq.numel() == 0, (name, inv_freq.numel())


def test_post_stack_head_is_built_on_post_process_only(mpu_tp1):
    from primus.backends.megatron.core.transformer.kimi_k3.attention_residual import (
        AttentionResidualHead,
    )

    config = _make_config()
    final = _build_block(config, pre_process=True, post_process=True)
    assert isinstance(final.attn_res_head, AttentionResidualHead)
    assert final.final_layernorm is not None

    middle = _build_block(config, pre_process=False, post_process=False)
    assert middle.attn_res_head is None
    assert middle.final_layernorm is None


# ---------------------------------------------------------------------------
# The per-layer bookkeeping
# ---------------------------------------------------------------------------


def test_static_append_schedule_matches_the_reference(mpu_tp1):
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_block import (
        attn_res_num_blocks_before,
    )

    entry, _, appends, _ = reference_layer_trace(NUM_LAYERS, ATTN_RES_BLOCK_SIZE)
    assert [
        attn_res_num_blocks_before(i, ATTN_RES_BLOCK_SIZE) for i in range(NUM_LAYERS)
    ] == entry

    block = _build_block(_make_config())
    assert block.attn_res_block_count_trace() == entry
    assert [layer.appends_checkpoint for layer in block.layers] == appends


def test_pre_attention_mixer_is_built_exactly_where_it_runs(mpu_tp1):
    """Layer 0 has no pre-attention mixer, every other layer does.

    The reference allocates ``self_attention_res_{norm,proj}`` on every
    layer (``:908-917``) but never reaches layer 0's, because
    ``block_residual`` starts genuinely empty (``:1190-1192``) and the mix
    is guarded on ``shape[1] > 0`` (``:987``). Building them would leave
    two parameters that can never receive a gradient, which would disarm
    :func:`test_every_parameter_receives_a_finite_gradient` for good.
    """
    _, _, _, mixes = reference_layer_trace(NUM_LAYERS, ATTN_RES_BLOCK_SIZE)
    block = _build_block(_make_config())

    assert [layer.attn_res_mixer is not None for layer in block.layers] == mixes
    # The pre-MLP mix runs on every layer, including layer 0 -- which has
    # just appended its own checkpoint, so block_residual is never empty
    # there (:1028-1033).
    assert all(layer.mlp_res_mixer is not None for layer in block.layers)


def test_attention_residual_parameter_count(mpu_tp1):
    """``2 * hidden`` per live mixer, and nothing else."""
    block = _build_block(_make_config())
    _, _, _, mixes = reference_layer_trace(NUM_LAYERS, ATTN_RES_BLOCK_SIZE)
    num_live = sum(mixes) + NUM_LAYERS + 1  # pre-attn mixers + pre-mlp mixers + head

    counted = sum(
        p.numel()
        for name, p in block.named_parameters()
        if "res_mixer" in name or "attn_res_head" in name
    )
    assert counted == num_live * 2 * HIDDEN, (counted, num_live)


def test_runtime_block_residual_trace_matches_the_reference(mpu_tp1):
    """Instrument the layers and check the shapes the forward really sees.

    ``block.attn_res_block_count_trace()`` is derived from the layer
    indices; this observes the tensor. The two are computed independently,
    so agreement means the append schedule and the index arithmetic have
    not drifted.
    """
    config = _make_config()
    block = _build_block(config).to(_device())

    entry_observed: List[int] = []
    exit_observed: List[int] = []

    for layer in block.layers:
        original = layer.forward

        def instrumented(hs, am=None, *, block_residual=None, _orig=original, **kwargs):
            entry_observed.append(0 if block_residual is None else block_residual.shape[-2])
            out_hidden, out_blocks = _orig(hs, am, block_residual=block_residual, **kwargs)
            exit_observed.append(out_blocks.shape[-2])
            return out_hidden, out_blocks

        layer.forward = instrumented

    x = torch.randn(SEQ, BATCH, HIDDEN, dtype=config.params_dtype, device=_device())
    block.train()
    out = block(x, None)

    entry, exit_, _, _ = reference_layer_trace(NUM_LAYERS, ATTN_RES_BLOCK_SIZE)
    assert entry_observed == entry, entry_observed
    assert exit_observed == exit_, exit_observed
    assert out.shape == (SEQ, BATCH, HIDDEN)
    assert torch.isfinite(out.float()).all()


def test_layer_drift_between_schedule_and_state_is_caught(mpu_tp1):
    """Handing a layer the wrong number of checkpoints must raise.

    The guard exists because a silent mismatch would not crash -- the
    mixer accepts any ``num_blocks`` -- it would just mix a different
    candidate set, which no loss curve would reveal.
    """
    config = _make_config()
    block = _build_block(config).to(_device())
    layer = block.layers[5]  # enters with 2 checkpoints
    assert layer.num_blocks_in == 2

    hidden = torch.randn(SEQ, BATCH, HIDDEN, dtype=config.params_dtype, device=_device())
    wrong = torch.randn(SEQ, BATCH, 1, HIDDEN, dtype=config.params_dtype, device=_device())
    with pytest.raises(AssertionError, match="checkpoints"):
        layer(hidden, None, block_residual=wrong)


def test_prefix_sum_resets_only_on_appending_layers(mpu_tp1):
    """The reset is what makes the mechanism more than a residual rename.

    On an appending layer the running sum restarts from the attention
    output (``:1023-1026``); elsewhere it accumulates. Detected
    behaviourally: zero the MLP's contribution and the layer's output is
    ``attn_out`` alone on an appending layer, and ``prefix_sum + attn_out``
    otherwise.
    """
    config = _make_config()
    block = _build_block(config).to(_device())
    dtype = config.params_dtype

    for layer_idx in (0, 4, 1, 5):
        layer = block.layers[layer_idx]
        hidden = torch.randn(SEQ, BATCH, HIDDEN, dtype=dtype, device=_device())
        blocks = torch.randn(
            SEQ, BATCH, layer.num_blocks_in, HIDDEN, dtype=dtype, device=_device()
        )

        captured = {}
        original_attn = layer.self_attention.forward

        def capture(*args, _orig=original_attn, **kwargs):
            out = _orig(*args, **kwargs)
            captured["attn"] = out[0]
            return out

        layer.self_attention.forward = capture
        # Make the MLP contribute exactly zero so prefix_sum's value is
        # fully determined by the reset semantics.
        original_mlp = layer.mlp.forward
        layer.mlp.forward = lambda x, *a, **k: (torch.zeros_like(x), None)
        try:
            out, _ = layer(hidden, None, block_residual=blocks)
        finally:
            layer.self_attention.forward = original_attn
            layer.mlp.forward = original_mlp

        attn_out = captured["attn"]
        if layer.appends_checkpoint:
            torch.testing.assert_close(out, attn_out, atol=1e-2, rtol=1e-2)
        else:
            torch.testing.assert_close(out, hidden + attn_out, atol=1e-2, rtol=1e-2)
            assert not torch.allclose(out, attn_out, atol=1e-2, rtol=1e-2), (
                f"layer {layer_idx} does not append, so prefix_sum must carry the "
                "incoming hidden state into the sum"
            )


def test_disabling_attention_residuals_gives_a_plain_residual_stack(mpu_tp1):
    """``attn_res_block_size=None`` must degrade cleanly, not half-apply."""
    config = _make_config(attn_res_block_size=None)
    assert config.attn_res_num_blocks_max == 0
    block = _build_block(config).to(_device())

    assert block.attn_res_head is None
    assert all(layer.attn_res_mixer is None for layer in block.layers)
    assert all(layer.mlp_res_mixer is None for layer in block.layers)
    assert not any("res_mixer" in n or "attn_res_head" in n for n, _ in block.named_parameters())

    x = torch.randn(SEQ, BATCH, HIDDEN, dtype=config.params_dtype, device=_device())
    out = block(x, None)
    assert out.shape == (SEQ, BATCH, HIDDEN)
    assert torch.isfinite(out.float()).all()


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_every_parameter_receives_a_finite_gradient(mpu_tp1):
    """DESIGN.md 7.2 item 10: the test that catches an unwired submodule.

    A spec-tree bug most often shows up as a module that is built but
    never called -- the model trains, the loss falls, and one projection
    stays at its initialisation forever.
    """
    config = _make_config()
    block = _build_block(config).to(_device())
    block.train()

    x = torch.randn(SEQ, BATCH, HIDDEN, dtype=config.params_dtype, device=_device())
    block(x, None).float().square().mean().backward()

    missing = [n for n, p in block.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"{len(missing)} parameters got no gradient: {missing[:10]}"

    nonfinite = [
        n for n, p in block.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not nonfinite, f"non-finite gradients: {nonfinite[:10]}"

    for critical in ("A_log", "dt_bias", "conv1d", "res_mixer", "attn_res_head"):
        matched = [
            n for n, p in block.named_parameters() if critical in n and p.grad is not None
        ]
        assert matched, f"no parameter matching {critical!r} received a gradient"


# ---------------------------------------------------------------------------
# The pipeline seam
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_blocks", [0, 1, 2])
def test_pipeline_seam_round_trip(num_blocks):
    """Packing into the sequence axis must be lossless.

    This is the mechanism pipeline parallelism will use: the boundary
    carries ``[(1 + nb_max) * s, b, h]``, a constant 3-D shape that
    standard P2P kernels handle unchanged (the V4 trick,
    ``deepseek_v4_block.py:802-806``). Needs no process group -- it is
    pure reshaping.
    """
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_block import (
        _lift_res_in,
        _lower_res_out,
    )

    nb_max = 2
    hidden = torch.randn(SEQ, BATCH, HIDDEN)
    blocks = torch.randn(SEQ, BATCH, num_blocks, HIDDEN)

    packed = _lower_res_out(hidden, blocks, post_process=False, num_blocks_max=nb_max)
    assert packed.shape == ((1 + nb_max) * SEQ, BATCH, HIDDEN)

    got_hidden, got_blocks = _lift_res_in(
        packed, pre_process=False, num_blocks=num_blocks, num_blocks_max=nb_max
    )
    assert torch.equal(got_hidden, hidden)
    assert got_blocks.shape == blocks.shape
    assert torch.equal(got_blocks, blocks)


def test_pipeline_seam_final_stage_returns_the_plain_hidden_state():
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_block import _lower_res_out

    hidden = torch.randn(SEQ, BATCH, HIDDEN)
    blocks = torch.randn(SEQ, BATCH, 2, HIDDEN)
    out = _lower_res_out(hidden, blocks, post_process=True, num_blocks_max=2)
    assert out is hidden


def test_first_stage_starts_with_a_zero_width_block_residual():
    """``hidden_states.new_zeros(num_tokens, 0, hidden)`` (``:1190-1192``)."""
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_block import _lift_res_in

    hidden = torch.randn(SEQ, BATCH, HIDDEN)
    got_hidden, got_blocks = _lift_res_in(
        hidden, pre_process=True, num_blocks=0, num_blocks_max=2
    )
    assert got_hidden is hidden
    assert got_blocks.shape == (SEQ, BATCH, 0, HIDDEN)
    assert got_blocks.numel() == 0


def test_unpacked_boundary_tensor_is_rejected():
    """A stage that forgot to pack must fail loudly, not reinterpret shapes."""
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_block import _lift_res_in

    unpacked = torch.randn(SEQ, BATCH, HIDDEN)  # not a multiple of 1 + nb_max
    with pytest.raises(ValueError, match="not divisible"):
        _lift_res_in(unpacked, pre_process=False, num_blocks=1, num_blocks_max=4)


def test_pipeline_parallel_builds_now_that_the_shape_patch_exists(mpu_tp1):
    """PP > 1 used to raise ``NotImplementedError`` here.

    The guard was removed in WP7 once
    ``primus/backends/megatron/patches/kimi_k3_pp_shape_patches.py`` landed
    to teach the scheduler the folded ``[(1 + nb_max) * s, b, h]`` wire
    shape. This test is the previous refusal test inverted, so a
    re-introduced guard fails rather than passing silently.

    The seam's numerics are pinned by
    ``test_kimi_k3_pp_shapes.py::test_two_stage_split_matches_one_stage_bit_exactly``.

    Note what the stage slicing does here: ``get_num_layers_to_build`` reads
    ``config.pipeline_model_parallel_size`` rather than the live
    ``parallel_state``, so with PP=2 on the config this single-rank build
    correctly yields stage 0's four layers even though the process group is
    1-rank.
    """
    config = _make_config()
    config.pipeline_model_parallel_size = 2
    block = _build_block(config)
    assert len(block.layers) == NUM_LAYERS // 2
    assert block.global_layer_indices == [0, 1, 2, 3]
    assert block.layer_offset == 0
    assert block.num_blocks_max == config.attn_res_num_blocks_max == 2
