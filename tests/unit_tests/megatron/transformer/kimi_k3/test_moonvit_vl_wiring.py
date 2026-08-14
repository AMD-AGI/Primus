###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The multimodal seam: MoonViT-V2 features spliced into the K3 text stream.

``test_moonvit_tower.py`` pins the tower. This file pins the ten lines that
join it to the language model, which is where an integration bug lives:
whether the features land on the right positions, in the right order, in the
right layout, and whether a gradient gets back through them.

Megatron is **sequence-first** (``[seq, batch, hidden]``) and ``input_ids``
is **batch-first** (``[batch, seq]``). That transpose is the single most
likely thing to get wrong here, and at ``batch == seq`` it is invisible, so
the ordering tests deliberately use a non-square batch.
"""

from __future__ import annotations

import os

import pytest
import torch  # must precede any transformer_engine import

PLACEHOLDER = 7
VISION = dict(
    vt_num_hidden_layers=2,
    vt_hidden_size=64,
    vt_intermediate_size=128,
    vt_num_attention_heads=4,
    vt_qkv_hidden_size=96,
    vt_patch_size=4,
    vt_init_pos_emb_height=8,
    vt_init_pos_emb_width=8,
    vt_init_pos_emb_time=4,
    vt_rope_max_height=32,
    vt_rope_max_width=32,
    vt_media_placeholder_token_id=PLACEHOLDER,
    vt_attention_backend="eager",
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the K3 decoder and the MoonViT spec tree both build Transformer Engine modules",
)


@pytest.fixture(autouse=True)
def _unset_nvte_attention_env(monkeypatch):
    """Clear the TE attention-backend env vars.

    ``rocm/primus:v26.4`` bakes ``NVTE_FLASH_ATTN=0`` (it targets the
    fused/CK path). ``LanguageModule.__init__`` calls
    ``_set_attention_backend`` (``language_module.py:48``), whose ``auto``
    branch asserts those vars are unset-or-1, so every ``KimiK3Model``
    construction fails with ``NVTE_FLASH_ATTN set to 0, but expected 1``.

    Same fixture, same reason, as
    ``tests/unit_tests/backends/megatron/diffusion/conftest.py:28-47``.
    That docstring says "non-diffusion Primus megatron tests never construct
    a model that runs this assertion" -- true until this file, because the
    other Kimi K3 tests build a ``KimiK3TransformerBlock`` directly and
    never reach ``LanguageModule``.

    ``monkeypatch.delenv`` rather than ``os.environ.pop``:
    ``_set_attention_backend`` writes the var back after its check, and
    monkeypatch's teardown is what contains that leak.
    """
    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="module")
def mpu_tp1():
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29593")
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


def make_config(**overrides):
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
        KimiK3TransformerConfig,
    )

    kwargs = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=8,
        ffn_hidden_size=256,
        kv_channels=16,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_head_dim=16,
        v_head_dim=16,
        qk_pos_emb_head_dim=8,
        rope_type="rope",
        mscale=1.0,
        mscale_all_dim=0.0,
        apply_rope_fusion=False,
        multi_latent_attention=False,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        linear_attention_freq=[1, 0],
        kda_backend="eager",
        kda_chunk_size=64,
        num_moe_experts=None,
        moe_layer_freq=0,
        params_dtype=torch.float32,
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,
        use_cpu_initialization=not torch.cuda.is_available(),
        # Kimi K3 trains without dropout, and Megatron's 0.1 defaults would
        # make every test here stochastic. That is not merely inconvenient:
        # it would make test_vision_features_change_the_logits pass on noise
        # whether or not the splice does anything.
        # test_forward_is_deterministic is the guard on that.
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    kwargs.update(VISION)
    kwargs.update(overrides)
    return KimiK3TransformerConfig(**kwargs)


def build_vl(**overrides):
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_layer_specs import (
        get_kimi_k3_runtime_decoder_spec,
    )
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_vl_model import (
        KimiK3VisionLanguageModel,
    )

    cfg = make_config(**overrides)
    spec = get_kimi_k3_runtime_decoder_spec(cfg)
    model = KimiK3VisionLanguageModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=64,
        max_sequence_length=64,
        share_embeddings_and_output_weights=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device=device, dtype=torch.float32), cfg


def make_media(model, grid, device):
    grid_thws = torch.tensor(grid, dtype=torch.long, device=device)
    total = int(grid_thws.prod(dim=-1).sum())
    g = torch.Generator().manual_seed(5)
    px = torch.randn(total, 3, VISION["vt_patch_size"], VISION["vt_patch_size"], generator=g)
    return px.to(device=device), grid_thws


# ===========================================================================
# expand_media_placeholders
# ===========================================================================


def test_expand_media_placeholders_matches_the_token_counts():
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_vl_model import (
        expand_media_placeholders,
    )

    ids = torch.tensor([[1, 2, PLACEHOLDER, 3, PLACEHOLDER, 4]])
    out = expand_media_placeholders(ids, [4, 2], PLACEHOLDER)
    assert out.tolist() == [[1, 2] + [PLACEHOLDER] * 4 + [3] + [PLACEHOLDER] * 2 + [4]]
    assert int((out == PLACEHOLDER).sum()) == 6


def test_expand_media_placeholders_rejects_a_count_mismatch():
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_vl_model import (
        expand_media_placeholders,
    )

    ids = torch.tensor([[1, PLACEHOLDER, 2]])
    with pytest.raises(ValueError, match="placeholder"):
        expand_media_placeholders(ids, [4, 4], PLACEHOLDER)


# ===========================================================================
# The splice
# ===========================================================================


@requires_gpu
def test_splice_writes_features_to_placeholder_positions_only(mpu_tp1):
    model, cfg = build_vl()
    device = "cuda"
    seq, batch = 6, 3  # non-square: a [seq, batch] / [batch, seq] swap shows up
    input_ids = torch.full((batch, seq), 1, dtype=torch.long, device=device)
    input_ids[0, 2] = PLACEHOLDER
    input_ids[1, 0] = PLACEHOLDER
    input_ids[1, 4] = PLACEHOLDER
    input_ids[2, 5] = PLACEHOLDER

    decoder_input = torch.zeros(seq, batch, cfg.hidden_size, device=device)
    features = torch.arange(4, dtype=torch.float32, device=device).unsqueeze(-1).expand(
        4, cfg.hidden_size
    ) + 1.0

    out = model.splice(decoder_input, input_ids, features.contiguous())

    # Media order is (batch, then seq): sample 0's one slot, then sample 1's
    # two in sequence order, then sample 2's.
    assert out[2, 0, 0].item() == 1.0
    assert out[0, 1, 0].item() == 2.0
    assert out[4, 1, 0].item() == 3.0
    assert out[5, 2, 0].item() == 4.0
    # Everything else untouched.
    written = {(2, 0), (0, 1), (4, 1), (5, 2)}
    for s in range(seq):
        for b in range(batch):
            if (s, b) not in written:
                assert out[s, b].abs().max().item() == 0.0


@requires_gpu
def test_splice_rejects_a_placeholder_count_mismatch(mpu_tp1):
    model, cfg = build_vl()
    device = "cuda"
    input_ids = torch.tensor([[1, PLACEHOLDER, 2]], device=device)
    decoder_input = torch.zeros(3, 1, cfg.hidden_size, device=device)
    with pytest.raises(ValueError, match="placeholder"):
        model.splice(decoder_input, input_ids, torch.zeros(4, cfg.hidden_size, device=device))


@requires_gpu
def test_splice_is_a_no_op_without_placeholders(mpu_tp1):
    model, cfg = build_vl()
    device = "cuda"
    input_ids = torch.full((2, 4), 1, dtype=torch.long, device=device)
    decoder_input = torch.randn(4, 2, cfg.hidden_size, device=device)
    out = model.splice(decoder_input, input_ids, torch.zeros(0, cfg.hidden_size, device=device))
    assert out is decoder_input


@requires_gpu
def test_injected_bug_transposed_splice_is_caught(mpu_tp1):
    """Write at ``[batch, seq]`` instead of ``[seq, batch]``.

    Megatron is sequence-first and ``input_ids`` is batch-first, so this is
    the most likely mistake in the splice. It is silent whenever both index
    orders are in bounds, which is why the placeholder sits at ``(1, 2)`` --
    valid either way round on a 6 x 3 buffer -- rather than somewhere the
    wrong order would raise ``IndexError`` and give the game away.
    """
    model, cfg = build_vl()
    device = "cuda"
    seq, batch = 6, 3
    input_ids = torch.full((batch, seq), 1, dtype=torch.long, device=device)
    input_ids[1, 2] = PLACEHOLDER  # batch 1, position 2
    decoder_input = torch.zeros(seq, batch, cfg.hidden_size, device=device)
    features = torch.ones(1, cfg.hidden_size, device=device)

    good = model.splice(decoder_input, input_ids, features)
    bad = decoder_input.clone()
    bad[1, 2] = features[0]  # the transposed write: in bounds, wrong slot

    assert good[2, 1, 0].item() == 1.0, "the feature must land at [seq=2, batch=1]"
    assert bad[1, 2, 0].item() == 1.0
    assert (good - bad).abs().max().item() > 0.5


# ===========================================================================
# End to end
# ===========================================================================


@requires_gpu
def test_multimodal_forward_runs_and_matches_the_manual_splice(mpu_tp1):
    model, cfg = build_vl()
    device = "cuda"
    grid = [(1, 4, 4), (1, 2, 4)]
    px, grid_thws = make_media(model, grid, device)
    counts = model.token_counts(grid_thws)
    assert counts == [4, 2]

    seq = 12
    input_ids = torch.full((1, seq), 1, dtype=torch.long, device=device)
    input_ids[0, 2 : 2 + sum(counts)] = PLACEHOLDER

    logits = model(input_ids=input_ids, position_ids=None, attention_mask=None,
                   pixel_values=px, grid_thws=grid_thws)
    assert logits.shape == (1, seq, 64)
    assert torch.isfinite(logits).all()


@requires_gpu
def test_text_only_path_is_unchanged_by_the_vision_tower(mpu_tp1):
    """``pixel_values=None`` must be bit-identical to the plain language model."""
    model, cfg = build_vl()
    device = "cuda"
    input_ids = torch.full((1, 8), 1, dtype=torch.long, device=device)

    with torch.no_grad():
        via_vl = model(input_ids=input_ids, position_ids=None, attention_mask=None)
        direct = model.language_model(
            input_ids=input_ids, position_ids=None, attention_mask=None
        )
    torch.testing.assert_close(via_vl, direct, rtol=0, atol=0)


@requires_gpu
def test_gradient_reaches_every_vision_parameter_from_the_language_loss(mpu_tp1):
    """The property that says the tower is actually wired, not merely present."""
    model, cfg = build_vl()
    device = "cuda"
    grid = [(1, 4, 4)]
    px, grid_thws = make_media(model, grid, device)
    counts = model.token_counts(grid_thws)

    seq = 10
    input_ids = torch.full((1, seq), 1, dtype=torch.long, device=device)
    input_ids[0, 1 : 1 + sum(counts)] = PLACEHOLDER
    labels = torch.full((1, seq), 2, dtype=torch.long, device=device)

    loss = model(
        input_ids=input_ids, position_ids=None, attention_mask=None,
        pixel_values=px, grid_thws=grid_thws, labels=labels,
    ).mean()
    loss.backward()

    vision = [(n, p) for n, p in model.named_parameters() if n.startswith("vision_tower.")]
    assert vision, "no vision parameters found"
    missing = [n for n, p in vision if p.grad is None]
    assert missing == [], f"no gradient for {missing}"
    dead = [n for n, p in vision if p.grad.abs().max().item() == 0.0]
    assert dead == [], f"zero gradient for {dead}"


@requires_gpu
def test_forward_is_deterministic(mpu_tp1):
    """Two identical multimodal forwards must be bit-identical.

    This is the guard that makes
    :func:`test_vision_features_change_the_logits` mean anything. Megatron's
    ``hidden_dropout`` / ``attention_dropout`` default to 0.1, and with them
    live *every* pair of forwards differs, so a perturbation test would pass
    on dropout noise whether or not the splice was wired to anything.
    """
    model, cfg = build_vl()
    device = "cuda"
    grid = [(1, 4, 4)]
    px, grid_thws = make_media(model, grid, device)
    counts = model.token_counts(grid_thws)
    seq = 10
    input_ids = torch.full((1, seq), 1, dtype=torch.long, device=device)
    input_ids[0, 1 : 1 + sum(counts)] = PLACEHOLDER

    with torch.no_grad():
        a = model(input_ids=input_ids, position_ids=None, attention_mask=None,
                  pixel_values=px, grid_thws=grid_thws)
        b = model(input_ids=input_ids, position_ids=None, attention_mask=None,
                  pixel_values=px, grid_thws=grid_thws)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@requires_gpu
def test_vision_features_change_the_logits(mpu_tp1):
    """Perturbing the image must move the output.

    A splice that writes nowhere, or writes into a detached copy, still
    produces finite logits and a decreasing loss.
    """
    model, cfg = build_vl()
    device = "cuda"
    grid = [(1, 4, 4)]
    px, grid_thws = make_media(model, grid, device)
    counts = model.token_counts(grid_thws)
    seq = 10
    input_ids = torch.full((1, seq), 1, dtype=torch.long, device=device)
    input_ids[0, 1 : 1 + sum(counts)] = PLACEHOLDER

    with torch.no_grad():
        base = model(input_ids=input_ids, position_ids=None, attention_mask=None,
                     pixel_values=px, grid_thws=grid_thws)
        moved = model(input_ids=input_ids, position_ids=None, attention_mask=None,
                      pixel_values=px * 3.0 + 1.0, grid_thws=grid_thws)
    assert (base - moved).abs().max().item() > 1e-4


# ===========================================================================
# Guards
# ===========================================================================


@requires_gpu
def test_sequence_parallel_is_refused_rather_than_mis_spliced(mpu_tp1):
    """Under SP the splice would write to the wrong rank; refuse, do not guess.

    The flag is set *after* construction because
    ``ModelParallelConfig.__post_init__`` rejects ``sequence_parallel``
    without tensor parallelism (``model_parallel_config.py:403``), and this
    test is about the vision guard, not about that one.
    """
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_layer_specs import (
        get_kimi_k3_runtime_decoder_spec,
    )
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_vl_model import (
        KimiK3VisionLanguageModel,
    )

    cfg = make_config()
    spec = get_kimi_k3_runtime_decoder_spec(cfg)
    cfg.sequence_parallel = True
    with pytest.raises(NotImplementedError, match="sequence_parallel"):
        KimiK3VisionLanguageModel(
            config=cfg,
            transformer_layer_spec=spec,
            vocab_size=64,
            max_sequence_length=64,
            share_embeddings_and_output_weights=False,
        )


@requires_gpu
def test_missing_placeholder_id_is_refused(mpu_tp1):
    with pytest.raises(ValueError, match="media_placeholder_token_id"):
        build_vl(vt_media_placeholder_token_id=None)


@requires_gpu
def test_text_only_config_cannot_build_a_vl_model(mpu_tp1):
    with pytest.raises(ValueError, match="vision tower"):
        build_vl(vt_num_hidden_layers=None)
