###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The Megatron MoonViT-V2 tower: config, spec tree, parity, wiring.

``test_moonvit_reference.py`` pins the eager reference against the official
implementation. This file pins :class:`MoonViTModel` -- the Megatron-spec
port that actually trains -- against that reference, plus the things a
reference cannot have: the derived configs, the ``ModuleSpec`` tree, tensor
shapes and dtypes across both attention backends, gradient coverage, and
distributed-checkpoint coverage of every new parameter slot.

Config tests run anywhere. Everything that builds a module needs Transformer
Engine and a visible GPU, because the spec's linears are TE's and
``TEDotProductAttention`` has no CPU path; those are skipped rather than
faked, matching ``test_attention_residual.py:1068``.

The parity tolerance is stated and argued rather than tuned: **fp32,
``max_abs <= 5e-6`` over the whole 4-layer tower**, which is ~40 float32
ULPs at unit RMS. The two implementations do the same arithmetic in a
different order -- the reference runs ``nn.Linear``, the port runs TE's
column/row-parallel linears with fp32 accumulation, and the reference fuses
the residual add differently from ``bias_dropout_add`` -- so bit-equality is
not available and is not claimed. bf16 is deliberately **not** given a
tolerance here: ``vision/scripts/parity_megatron_vs_reference.py`` measures
both implementations against the *fp32* reference instead, which separates
"our port is wrong" from "bf16 is bf16".
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import pytest
import torch  # must precede any transformer_engine import

GRID = [(1, 8, 8), (1, 4, 6), (3, 2, 4)]
VISION = dict(
    vt_num_hidden_layers=4,
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
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the MoonViT spec tree builds Transformer Engine linears, which need a GPU",
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mpu_tp1():
    """A 1-rank process group plus Megatron model-parallel state.

    Same shape as ``test_kimi_k3_block.py:113-148``. TE's linears need a TP
    group even at size 1.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
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


def make_k3_config(**overrides):
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
        KimiK3TransformerConfig,
    )

    kwargs = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=8,
        ffn_hidden_size=256,
        kv_channels=16,
        params_dtype=torch.float32,
        use_cpu_initialization=not torch.cuda.is_available(),
    )
    kwargs.update(VISION)
    kwargs.update(overrides)
    return KimiK3TransformerConfig(**kwargs)


def build_tower(*, dtype=torch.float32, backend="eager", **overrides):
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_vision_config import (
        build_moonvit_configs,
    )
    from primus.backends.megatron.core.models.kimi_k3.moonvit_model import MoonViTModel

    cfg = make_k3_config(params_dtype=dtype, vt_attention_backend=backend, **overrides)
    tower_cfg, proj_cfg = build_moonvit_configs(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoonViTModel(tower_cfg, proj_cfg).to(device=device, dtype=dtype)
    return model, tower_cfg, proj_cfg


def build_reference(tower_cfg, proj_cfg, *, dtype=torch.float32):
    from primus.backends.megatron.core.transformer.kimi_k3.vision.moonvit_reference import (
        MoonViTReference,
        MoonViTReferenceConfig,
    )

    cfg = MoonViTReferenceConfig(
        hidden_size=tower_cfg.hidden_size,
        intermediate_size=tower_cfg.ffn_hidden_size,
        num_hidden_layers=tower_cfg.num_layers,
        num_attention_heads=tower_cfg.num_attention_heads,
        qkv_hidden_size=tower_cfg.qkv_hidden_size,
        patch_size=tower_cfg.patch_size,
        init_pos_emb_height=tower_cfg.init_pos_emb_height,
        init_pos_emb_width=tower_cfg.init_pos_emb_width,
        init_pos_emb_time=tower_cfg.init_pos_emb_time,
        mm_hidden_size=tower_cfg.hidden_size,
        text_hidden_size=proj_cfg.hidden_size,
        projector_ln_eps=proj_cfg.projector_ln_eps,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MoonViTReference(cfg).to(device=device, dtype=dtype)


#: Reference parameter -> Megatron parameter, per encoder layer. The fused
#: ``wqkv`` is handled separately: the released ``[q | k | v]`` block layout
#: (``modeling_kimi_k3.py:519-528``) splits into three column-parallel
#: projections by ``chunk(3, dim=0)``.
_BLOCK_MAP = {
    "norm0.weight": "input_layernorm.weight",
    "norm1.weight": "pre_mlp_layernorm.weight",
    "attn.wo.weight": "self_attention.linear_proj.weight",
    "mlp.fc0.weight": "mlp.linear_fc1.weight",
    "mlp.fc1.weight": "mlp.linear_fc2.weight",
}
_QKV_TARGETS = (
    "self_attention.linear_q.weight",
    "self_attention.linear_k.weight",
    "self_attention.linear_v.weight",
)
_TOP_MAP = {
    "patch_embed.proj.weight": "patch_embed.proj.weight",
    "patch_embed.pos_emb.weight": "patch_embed.pos_emb.weight",
    "encoder.final_layernorm.weight": "decoder.final_layernorm.weight",
    "projector.proj.0.weight": "projector.encoder_projector.encoder.linear_fc1.weight",
    "projector.proj.2.weight": "projector.encoder_projector.encoder.linear_fc2.weight",
    "projector.post_norm.weight": "projector.post_norm.weight",
}


def mirror_reference_into_tower(reference, tower, *, swap_qkv: bool = False) -> int:
    """Copy the reference's parameters into the Megatron tower.

    ``swap_qkv`` is the bug-injection hook: it feeds the ``k`` block where
    ``q`` belongs, which is a shape-preserving, silent corruption.
    """
    ref_sd = dict(reference.state_dict())
    meg_sd = dict(tower.state_dict())
    order = (1, 0, 2) if swap_qkv else (0, 1, 2)
    copied = 0
    with torch.no_grad():
        for ref_key, meg_key in _TOP_MAP.items():
            if ref_key in ref_sd and meg_key in meg_sd:
                meg_sd[meg_key].copy_(ref_sd[ref_key].to(meg_sd[meg_key].dtype))
                copied += 1
        for i in range(len(reference.encoder.blocks)):
            for tail, target in _BLOCK_MAP.items():
                src = ref_sd[f"encoder.blocks.{i}.{tail}"]
                dst = meg_sd[f"decoder.layers.{i}.{target}"]
                dst.copy_(src.to(dst.dtype))
                copied += 1
            fused = ref_sd[f"encoder.blocks.{i}.attn.wqkv.weight"].chunk(3, dim=0)
            for slot, chunk_idx in zip(_QKV_TARGETS, order):
                dst = meg_sd[f"decoder.layers.{i}.{slot}"]
                dst.copy_(fused[chunk_idx].to(dst.dtype))
                copied += 1
    return copied


def make_inputs(tower_cfg, grid=GRID, dtype=torch.float32, seed: int = 11):
    g = torch.Generator().manual_seed(seed)
    total = sum(t * h * w for t, h, w in grid)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    px = torch.randn(total, 3, tower_cfg.patch_size, tower_cfg.patch_size, generator=g)
    return (
        px.to(device=device, dtype=dtype),
        torch.tensor(grid, dtype=torch.long, device=device),
    )


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().double() - b.detach().double()).abs().max())


#: fp32, whole tower. ~40 float32 ULPs at unit RMS; see the module docstring
#: for why bit-equality is not available.
FP32_TOL = 5e-6


# ===========================================================================
# 1. Config: validation and derivation (no GPU needed)
# ===========================================================================


def test_text_only_config_is_unaffected():
    """A config with no ``vt_num_hidden_layers`` must behave exactly as before."""
    cfg = make_k3_config(vt_num_hidden_layers=None)
    assert cfg.has_vision_tower is False
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_vision_config import (
        build_moonvit_configs,
    )

    with pytest.raises(ValueError, match="no vision tower"):
        build_moonvit_configs(cfg)


def test_config_reports_a_vision_tower_when_configured():
    assert make_k3_config().has_vision_tower is True


@pytest.mark.parametrize(
    "override,match",
    [
        ({"vt_qkv_hidden_size": 98}, "not divisible"),
        ({"vt_qkv_hidden_size": 8, "vt_num_attention_heads": 4}, "divisible by 4"),
        ({"mm_hidden_size": 999}, "must equal"),
        ({"mm_projector_type": "patchmerger"}, "mm_projector_type"),
        ({"vt_pos_emb_interpolation_mode": "cubic"}, "interpolation_mode"),
        ({"vt_merge_kernel_size": (2, 0)}, "positive"),
        ({"vt_attention_backend": "flash"}, "auto|eager|te"),
        ({"projector_ln_eps": 0.0}, "projector_ln_eps"),
        ({"vt_layernorm_epsilon": -1.0}, "vt_layernorm_epsilon"),
        ({"vt_init_pos_emb_time": 0}, "vt_init_pos_emb_time"),
    ],
)
def test_config_rejects_bad_vision_geometry(override, match):
    with pytest.raises(ValueError, match=match):
        make_k3_config(**override)


def test_head_dim_divisible_by_four_is_a_rope_requirement_not_a_style_rule():
    """``Rope2DPosEmbRepeated`` asserts it (``:370``): 4 channels per frequency.

    ``96 / 4 = 24`` is fine; ``72 / 4 = 18`` is not divisible by 4 and must
    be rejected even though it divides the head count cleanly.
    """
    make_k3_config(vt_qkv_hidden_size=96, vt_num_attention_heads=4)  # 24, ok
    with pytest.raises(ValueError, match="divisible by 4"):
        make_k3_config(vt_qkv_hidden_size=72, vt_num_attention_heads=4)  # 18, not


def test_derived_tower_config_matches_the_released_shape():
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_vision_config import (
        build_moonvit_configs,
        gelu_tanh,
    )

    cfg = make_k3_config(
        hidden_size=7168,
        vt_num_hidden_layers=27,
        vt_hidden_size=1024,
        vt_intermediate_size=4096,
        vt_num_attention_heads=12,
        vt_qkv_hidden_size=1536,
        vt_patch_size=14,
        vt_init_pos_emb_height=64,
        vt_init_pos_emb_width=64,
        vt_rope_max_height=512,
        vt_rope_max_width=512,
    )
    tower, proj = build_moonvit_configs(cfg)

    assert (tower.num_layers, tower.hidden_size, tower.ffn_hidden_size) == (27, 1024, 4096)
    assert tower.num_attention_heads == 12 and tower.num_query_groups == 12
    # The attention inner width is WIDER than the residual stream.
    assert tower.kv_channels == 128 and tower.qkv_hidden_size == 1536
    assert tower.qkv_hidden_size != tower.hidden_size
    assert tower.normalization == "RMSNorm"
    assert tower.gated_linear_unit is False
    assert tower.activation_func is gelu_tanh
    assert tower.add_bias_linear is False and tower.add_qkv_bias is False
    # The eps a default Megatron port gets wrong: nn.RMSNorm's eps=None
    # default, not Megatron's 1e-5.
    from primus.backends.megatron.core.transformer.kimi_k3.vision.moonvit_reference import (
        MOONVIT_TOWER_NORM_EPS,
    )

    assert tower.layernorm_epsilon == MOONVIT_TOWER_NORM_EPS
    assert tower.layernorm_epsilon < torch.finfo(torch.float32).eps
    assert tower.pipeline_model_parallel_size == 1

    # 4096 -> 4096 -> 7168, with the projector's own 1e-5.
    assert proj.projector_input_size == 1024 * 4
    assert proj.ffn_hidden_size == 4096
    assert proj.hidden_size == 7168
    assert proj.layernorm_epsilon == 1e-5 and proj.projector_ln_eps == 1e-5
    assert proj.gated_linear_unit is False
    assert proj.activation_func is torch.nn.functional.gelu  # erf, not tanh


def test_released_parameter_count_is_reproduced():
    """~401 M tower + ~46 M projector, as ``research/KIMI_K3_ARCH.md`` 1.4 says.

    Counted from the derived config rather than by building 447 M parameters.
    """
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_vision_config import (
        build_moonvit_configs,
    )

    cfg = make_k3_config(
        hidden_size=7168,
        vt_num_hidden_layers=27,
        vt_hidden_size=1024,
        vt_intermediate_size=4096,
        vt_num_attention_heads=12,
        vt_qkv_hidden_size=1536,
        vt_patch_size=14,
        vt_init_pos_emb_height=64,
        vt_init_pos_emb_width=64,
    )
    tower, proj = build_moonvit_configs(cfg)
    h, q, f = tower.hidden_size, tower.qkv_hidden_size, tower.ffn_hidden_size
    per_layer = 3 * h * q + q * h + 2 * h * f + 2 * h
    total_tower = (
        h * tower.in_channels * tower.patch_size**2
        + tower.init_pos_emb_height * tower.init_pos_emb_width * h
        + tower.num_layers * per_layer
        + h
    )
    total_proj = (
        proj.projector_input_size * proj.ffn_hidden_size
        + proj.ffn_hidden_size * proj.hidden_size
        + proj.hidden_size
    )
    assert 4.00e8 < total_tower < 4.03e8
    assert 4.6e7 < total_proj < 4.7e7
    # The exact figure the released shards carry, both halves together.
    assert total_tower + total_proj == 447_358_976


# ===========================================================================
# 2. The spec tree
# ===========================================================================


@requires_gpu
def test_spec_tree_shape(mpu_tp1):
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_vision_config import (
        build_moonvit_configs,
    )
    from primus.backends.megatron.core.models.kimi_k3.moonvit_layer_specs import (
        get_moonvit_block_spec,
    )
    from primus.backends.megatron.core.transformer.kimi_k3.vision.moonvit_attention import (
        MoonViTEagerAttention,
        MoonViTSelfAttention,
    )

    tower_cfg, _ = build_moonvit_configs(make_k3_config(vt_attention_backend="eager"))
    spec = get_moonvit_block_spec(tower_cfg)

    assert spec.module is TransformerBlock
    submodules = spec.params["spec"]
    assert len(submodules.layer_specs) == tower_cfg.num_layers
    layer = submodules.layer_specs[0]
    assert layer.module is TransformerLayer
    assert layer.submodules.self_attention.module is MoonViTSelfAttention
    assert layer.submodules.self_attention.submodules.core_attention.module is MoonViTEagerAttention
    assert layer.submodules.mlp.module is MLP
    # Non-gated: the activation is config.activation_func, so this slot stays
    # empty -- the opposite of the text backbone's rule, and deliberate.
    assert layer.submodules.mlp.submodules.activation_func is None


@requires_gpu
def test_attention_mask_type_is_no_mask(mpu_tp1):
    from megatron.core.transformer.enums import AttnMaskType

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_vision_config import (
        build_moonvit_configs,
    )
    from primus.backends.megatron.core.models.kimi_k3.moonvit_layer_specs import (
        get_moonvit_block_spec,
    )

    tower_cfg, _ = build_moonvit_configs(make_k3_config())
    spec = get_moonvit_block_spec(tower_cfg)
    attn = spec.params["spec"].layer_specs[0].submodules.self_attention
    assert attn.params["attn_mask_type"] is AttnMaskType.no_mask


def test_attention_backend_resolution():
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_vision_config import (
        build_moonvit_configs,
    )
    from primus.backends.megatron.core.models.kimi_k3.moonvit_layer_specs import (
        resolve_moonvit_attention_backend,
    )

    for requested in ("eager", "te"):
        cfg, _ = build_moonvit_configs(make_k3_config(vt_attention_backend=requested))
        assert resolve_moonvit_attention_backend(cfg) == requested
    auto, _ = build_moonvit_configs(make_k3_config(vt_attention_backend="auto"))
    assert resolve_moonvit_attention_backend(auto) == ("te" if torch.cuda.is_available() else "eager")


@requires_gpu
def test_parameter_names_and_no_unexpected_buffers(mpu_tp1):
    model, tower_cfg, _ = build_tower()
    names = {n for n, _ in model.named_parameters()}
    for expected in (
        "patch_embed.proj.weight",
        "patch_embed.pos_emb.weight",
        "decoder.layers.0.input_layernorm.weight",
        "decoder.layers.0.self_attention.linear_q.weight",
        "decoder.layers.0.self_attention.linear_k.weight",
        "decoder.layers.0.self_attention.linear_v.weight",
        "decoder.layers.0.self_attention.linear_proj.weight",
        "decoder.layers.0.pre_mlp_layernorm.weight",
        "decoder.layers.0.mlp.linear_fc1.weight",
        "decoder.layers.0.mlp.linear_fc2.weight",
        "decoder.final_layernorm.weight",
        "projector.encoder_projector.encoder.linear_fc1.weight",
        "projector.encoder_projector.encoder.linear_fc2.weight",
        "projector.post_norm.weight",
    ):
        assert expected in names, f"missing parameter {expected}"
    # No biases anywhere -- report section 2.4's stability argument.
    assert not any(n.endswith(".bias") for n in names)
    # The temporal code must stay non-persistent, as upstream registers it
    # (``:248-253``); a persistent buffer would break a released-state load.
    assert "patch_embed.pos_emb.time_weight" not in dict(model.named_buffers(), **{}) or (
        "patch_embed.pos_emb.time_weight" not in model.state_dict()
    )
    assert "patch_embed.pos_emb.time_weight" not in model.state_dict()
    assert "rotary_pos_emb.freqs_cis" not in model.state_dict()


# ===========================================================================
# 3. Parity against the reference
# ===========================================================================


@requires_gpu
def test_tower_matches_the_reference_in_fp32(mpu_tp1):
    model, tower_cfg, proj_cfg = build_tower(dtype=torch.float32, backend="eager")
    reference = build_reference(tower_cfg, proj_cfg, dtype=torch.float32)
    mirror_reference_into_tower(reference, model)
    px, grid = make_inputs(tower_cfg)

    with torch.no_grad():
        got = model(px, grid, return_stages=True)
        want = reference(px, grid, return_stages=True)

    assert max_abs(model.patch_embed(px, grid), want.patch_embed) == 0.0
    assert max_abs(got["encoded"], want.encoder) < FP32_TOL
    assert max_abs(
        got["merged"], torch.cat([m.reshape(m.shape[0], -1) for m in want.merged], dim=0)
    ) < FP32_TOL
    assert max_abs(got["projected"], torch.cat(want.projected, dim=0)) < FP32_TOL


@requires_gpu
@pytest.mark.parametrize(
    "grid",
    [
        [(1, 8, 8)],                      # one image at the pos-emb's own size
        [(1, 4, 6)],                      # one image, interpolation path
        [(3, 2, 4)],                      # video
        [(1, 8, 8), (1, 4, 6), (3, 2, 4)],  # mixed batch
        [(1, 2, 2)],                      # the smallest legal grid
    ],
    ids=["identity_grid", "interpolated", "video", "mixed", "minimal"],
)
def test_variable_resolution_and_video_shapes(mpu_tp1, grid):
    model, tower_cfg, proj_cfg = build_tower(dtype=torch.float32, backend="eager")
    reference = build_reference(tower_cfg, proj_cfg, dtype=torch.float32)
    mirror_reference_into_tower(reference, model)
    px, grid_thws = make_inputs(tower_cfg, grid)

    with torch.no_grad():
        got = model(px, grid_thws)
        want = torch.cat(reference(px, grid_thws), dim=0)

    expected_tokens = sum((h // 2) * (w // 2) for _, h, w in grid)
    assert got.shape == (expected_tokens, proj_cfg.hidden_size)
    assert model.token_counts(grid_thws) == [(h // 2) * (w // 2) for _, h, w in grid]
    assert max_abs(got, want) < FP32_TOL


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("backend", ["eager", "te"])
def test_dtype_and_backend_coverage(mpu_tp1, dtype, backend):
    """Shapes, dtypes and finiteness across the dtype x backend grid.

    ``te`` in fp32 is expected to be unavailable -- ``TEDotProductAttention``
    dispatches to a fused or flash kernel in ``thd`` format and neither has
    an fp32 path -- so it is reported as a skip rather than silently passing
    on a fallback.
    """
    model, tower_cfg, proj_cfg = build_tower(dtype=dtype, backend=backend)
    px, grid = make_inputs(tower_cfg, dtype=dtype)
    try:
        with torch.no_grad():
            out = model(px, grid)
    except (RuntimeError, AssertionError) as exc:
        if backend == "te" and dtype is torch.float32:
            pytest.skip(f"TE thd attention has no fp32 path on this build: {exc}")
        raise
    assert out.dtype == dtype
    assert out.shape == (sum(model.token_counts(grid)), proj_cfg.hidden_size)
    assert torch.isfinite(out.float()).all()


@requires_gpu
def test_every_parameter_receives_a_gradient(mpu_tp1):
    model, tower_cfg, _ = build_tower(dtype=torch.float32, backend="eager")
    px, grid = make_inputs(tower_cfg)
    model(px, grid).float().pow(2).mean().backward()

    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == [], f"no gradient for {missing}"
    dead = [n for n, p in model.named_parameters() if p.grad.abs().max().item() == 0.0]
    assert dead == [], f"zero gradient for {dead}"


@requires_gpu
def test_sharded_state_dict_covers_every_parameter(mpu_tp1):
    """Distributed-checkpoint coverage of the new slots.

    ``DECISIONS.md``'s WP7 section lists unverified ``sharded_state_dict``
    coverage of new parameter slots as a standing risk. The tower adds two
    that no upstream module owns -- the ``Conv2d`` patch projection and the
    learnable position grid -- so they are checked here rather than
    discovered at the first reshard.
    """
    model, tower_cfg, _ = build_tower(dtype=torch.float32, backend="eager")
    sharded = model.sharded_state_dict(prefix="vision.")

    for name, _ in model.named_parameters():
        key = f"vision.{name}"
        assert key in sharded, f"{key} missing from sharded_state_dict"
    assert "vision.patch_embed.proj.weight" in sharded
    assert "vision.patch_embed.pos_emb.weight" in sharded
    assert "vision.projector.post_norm.weight" in sharded
    # Non-persistent buffers must stay out of the checkpoint.
    assert "vision.patch_embed.pos_emb.time_weight" not in sharded
    assert "vision.rotary_pos_emb.freqs_cis" not in sharded


@requires_gpu
def test_attention_rejects_missing_packing_information(mpu_tp1):
    """No ``cu_seqlens`` must raise, not silently attend across media items."""
    from primus.backends.megatron.core.transformer.kimi_k3.vision.moonvit_attention import (
        MoonViTEagerAttention,
    )

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_vision_config import (
        build_moonvit_configs,
    )

    tower_cfg, _ = build_moonvit_configs(make_k3_config())
    attn = MoonViTEagerAttention(tower_cfg)
    q = torch.randn(8, 4, tower_cfg.kv_channels)
    with pytest.raises(ValueError, match="cu_seqlens"):
        attn(q, q, q, packed_seq_params=None)


@requires_gpu
def test_self_attention_requires_rope(mpu_tp1):
    """A MoonViT layer with no positional information is a different model."""
    model, tower_cfg, _ = build_tower(dtype=torch.float32, backend="eager")
    layer = model.decoder.layers[0].self_attention
    x = torch.randn(8, 1, tower_cfg.hidden_size, device="cuda")
    with pytest.raises(ValueError, match="2-D RoPE"):
        layer(x, rotary_pos_emb=None)


@requires_gpu
def test_tower_is_block_diagonal_over_media_items(mpu_tp1):
    """The end-to-end version of the reference's block-diagonality test."""
    model, tower_cfg, _ = build_tower(dtype=torch.float32, backend="eager")
    grid = [(1, 4, 4), (1, 4, 6)]
    px, grid_thws = make_inputs(tower_cfg, grid)

    with torch.no_grad():
        base = model(px, grid_thws)
        perturbed = px.clone()
        perturbed[0] += 10.0
        after = model(perturbed, grid_thws)

    first_item_tokens = (4 // 2) * (4 // 2)
    assert max_abs(after[first_item_tokens:], base[first_item_tokens:]) == 0.0
    assert max_abs(after[:first_item_tokens], base[:first_item_tokens]) > 1e-4


# ===========================================================================
# 4. Norm epsilons -- the MLA-class hazard, audited structurally
# ===========================================================================
#
# A sibling lane found a real defect in the Kimi K3 text backbone's MLA:
# ``kv_a_layernorm`` ran at ``config.layernorm_epsilon`` (1e-5) where the
# release gives that norm ``KimiRMSNorm``'s class default of 1e-6 -- a
# DeepSeek-V3 inheritance that every other call site overrides and those two
# did not. Fixing it moved MLA parity from 2.91e-05 to 3.82e-07, and **no
# existing test could have caught it, because every test built both sides
# from the same config**.
#
# The vision tower is exposed to the same hazard, and in a sharper form: the
# release uses *two different* epsilons -- the tower's norms take
# ``nn.RMSNorm``'s ``eps=None`` default and only the projector's ``post_norm``
# takes ``projector_ln_eps = 1e-5`` (``modeling_kimi_k3.py:490-491``, ``:591``
# vs ``:795``). Getting that split backwards is exactly the shape of the MLA
# defect.
#
# The parity tests here already run against the released weights, so they do
# not have the same-config blind spot. These tests are the structural
# complement: they assert the epsilon each norm carries directly, so a wrong
# value fails by name rather than by a tolerance that might be too loose.


def _norm_eps(module) -> float:
    """The epsilon a norm module carries, whichever attribute holds it.

    TE's RMSNorm keeps it in ``eps``; the torch and apex flavours use
    ``layer_norm_eps`` or ``variance_epsilon``. There is no common accessor.
    """
    for attr in ("eps", "layer_norm_eps", "variance_epsilon"):
        if hasattr(module, attr):
            value = getattr(module, attr)
            if value is not None:
                return float(value)
    raise AssertionError(f"{type(module).__name__} exposes no epsilon attribute")


def _tower_norms(model):
    """``(name, module)`` for every norm inside the tower, excluding the projector."""
    out = []
    for name, mod in model.named_modules():
        if name.startswith("projector"):
            continue
        if name.endswith(("input_layernorm", "pre_mlp_layernorm", "final_layernorm")):
            out.append((name, mod))
    return out


@requires_gpu
def test_every_tower_norm_carries_the_released_epsilon(mpu_tp1):
    """3 norms per layer-pair plus the final one, all at the release's value.

    The count is asserted too: adding a norm to the tower without deciding
    its epsilon should fail here rather than drift into a parity number.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.vision.moonvit_reference import (
        MOONVIT_TOWER_NORM_EPS,
    )

    model, tower_cfg, proj_cfg = build_tower()
    norms = _tower_norms(model)
    # two per layer (input_layernorm, pre_mlp_layernorm) + one final
    assert len(norms) == 2 * tower_cfg.num_layers + 1, [n for n, _ in norms]

    for name, mod in norms:
        eps = _norm_eps(mod)
        assert eps == MOONVIT_TOWER_NORM_EPS, f"{name} has eps={eps}"
        assert eps < torch.finfo(torch.float32).eps, f"{name} eps={eps} is visible in fp32"

    # ...and the projector's is a genuinely different number.
    post = _norm_eps(model.projector.post_norm)
    assert post == proj_cfg.projector_ln_eps == 1e-5
    assert post != MOONVIT_TOWER_NORM_EPS


@requires_gpu
def test_norm_epsilons_match_the_reference_pairwise(mpu_tp1):
    """Every Megatron norm agrees with its reference counterpart.

    The Megatron port reads ``config.layernorm_epsilon`` through ``TENorm``;
    the reference reads ``nn.RMSNorm``'s ``eps=None`` default. Two different
    code paths reading two different sources, which is precisely how the MLA
    defect happened, so the agreement is asserted rather than assumed.
    """
    model, tower_cfg, proj_cfg = build_tower()
    reference = build_reference(tower_cfg, proj_cfg)

    ref_norms = [
        reference.encoder.blocks[0].norm0,
        reference.encoder.blocks[0].norm1,
        reference.encoder.final_layernorm,
    ]
    meg_norms = [
        model.decoder.layers[0].input_layernorm,
        model.decoder.layers[0].pre_mlp_layernorm,
        model.decoder.final_layernorm,
    ]
    for ref, meg in zip(ref_norms, meg_norms):
        # nn.RMSNorm stores eps=None and lets ATen substitute; compare
        # against the measured substitution rather than against None.
        from primus.backends.megatron.core.transformer.kimi_k3.vision.moonvit_reference import (
            MOONVIT_TOWER_NORM_EPS,
        )

        ref_eps = MOONVIT_TOWER_NORM_EPS if ref.eps is None else float(ref.eps)
        assert _norm_eps(meg) == ref_eps

    assert _norm_eps(model.projector.post_norm) == float(reference.projector.post_norm.eps)


@requires_gpu
def test_injected_bug_projector_norm_taking_the_tower_epsilon_is_caught(mpu_tp1):
    """The MLA defect, transplanted: a norm silently on the *other* epsilon.

    The release uses two epsilons and the projector's is the one that is
    explicitly 1e-5. Running it at the tower's near-zero value instead is
    shape-preserving, dtype-preserving, raises nothing, and is the exact
    mistake the sibling lane found in ``kv_a_layernorm``.
    """
    model, tower_cfg, proj_cfg = build_tower()
    reference = build_reference(tower_cfg, proj_cfg)
    mirror_reference_into_tower(reference, model)
    px, grid = make_inputs(tower_cfg)

    with torch.no_grad():
        want = torch.cat(reference(px, grid), dim=0)
        good = model(px, grid)
    assert max_abs(good, want) < FP32_TOL

    from primus.backends.megatron.core.transformer.kimi_k3.vision.moonvit_reference import (
        MOONVIT_TOWER_NORM_EPS,
    )

    model.projector.post_norm.eps = MOONVIT_TOWER_NORM_EPS
    with torch.no_grad():
        bad = model(px, grid)
    gap = max_abs(bad, want)
    assert gap > FP32_TOL, (
        f"swapping the projector's 1e-5 for the tower's {MOONVIT_TOWER_NORM_EPS} moved the "
        f"output by only {gap}, which is inside the tolerance -- the parity test would not "
        "have caught the MLA-class defect here"
    )


# ===========================================================================
# 5. Deliberate bug injection on the Megatron port
# ===========================================================================


@requires_gpu
def test_injected_bug_swapped_q_and_k_projections_is_caught(mpu_tp1):
    """Load the ``k`` block into ``q``: same shapes, same dtypes, wrong model.

    The most plausible mistake in the ``wqkv`` -> three-projection split, and
    the reason that split is worth a test at all.
    """
    model, tower_cfg, proj_cfg = build_tower(dtype=torch.float32, backend="eager")
    reference = build_reference(tower_cfg, proj_cfg, dtype=torch.float32)
    mirror_reference_into_tower(reference, model, swap_qkv=True)
    px, grid = make_inputs(tower_cfg)

    with torch.no_grad():
        got = model(px, grid)
        want = torch.cat(reference(px, grid), dim=0)
    assert max_abs(got, want) > 1e-3, "swapping q and k is invisible; the parity test has no power"


@requires_gpu
def test_injected_bug_shared_rope_across_media_items_is_caught(mpu_tp1):
    """Hand every media item the first item's frequencies.

    This is precisely what reusing Megatron's ``thd`` rotary path would have
    produced (``rope_utils.py:236-238``), and it is the single reason
    :class:`MoonViTSelfAttention` exists rather than a ``SelfAttention``
    subclass. It changes no shape and raises nothing.
    """
    model, tower_cfg, proj_cfg = build_tower(dtype=torch.float32, backend="eager")
    reference = build_reference(tower_cfg, proj_cfg, dtype=torch.float32)
    mirror_reference_into_tower(reference, model)
    grid = [(1, 8, 8), (1, 4, 6)]
    px, grid_thws = make_inputs(tower_cfg, grid)

    with torch.no_grad():
        good = model(px, grid_thws)

    original = model.rotary_pos_emb.forward

    def broken(grid_arg, device=None):
        freqs = original(grid_arg, device=device)
        first_len = int(grid_arg[0].prod())
        head = freqs[:first_len]
        # Every subsequent item re-reads the first item's rows, which is the
        # `freqs[0:len]` re-slicing the thd path does.
        return torch.cat([head, head[: freqs.shape[0] - first_len]], dim=0)

    model.rotary_pos_emb.forward = broken
    try:
        with torch.no_grad():
            bad = model(px, grid_thws)
    finally:
        model.rotary_pos_emb.forward = original

    assert max_abs(good, bad) > 1e-3


@requires_gpu
def test_injected_bug_default_layernorm_epsilon_is_caught(mpu_tp1):
    """1e-5 on the tower norms instead of the released 0.

    The default a Megatron port lands on, and the reason
    ``vt_layernorm_epsilon`` is a field with a non-default default.
    """
    good_model, tower_cfg, proj_cfg = build_tower(dtype=torch.float32, backend="eager")
    reference = build_reference(tower_cfg, proj_cfg, dtype=torch.float32)
    mirror_reference_into_tower(reference, good_model)
    px, grid = make_inputs(tower_cfg)
    with torch.no_grad():
        want = torch.cat(reference(px, grid), dim=0)
        good = good_model(px, grid)
    assert max_abs(good, want) < FP32_TOL

    bad_model, bad_cfg, bad_proj = build_tower(
        dtype=torch.float32, backend="eager", vt_layernorm_epsilon=1e-5
    )
    assert bad_cfg.layernorm_epsilon == 1e-5
    mirror_reference_into_tower(reference, bad_model)
    with torch.no_grad():
        bad = bad_model(px, grid)
    gap = max_abs(bad, want)
    assert gap > FP32_TOL, f"eps 0 -> 1e-5 moved the tower by only {gap}, below the tolerance"


@requires_gpu
def test_injected_bug_full_attention_across_items_is_caught(mpu_tp1):
    """One packed segment covering the whole batch.

    The block-diagonality test must fail on it.
    """
    model, tower_cfg, _ = build_tower(dtype=torch.float32, backend="eager")
    grid = [(1, 4, 4), (1, 4, 6)]
    px, grid_thws = make_inputs(tower_cfg, grid)

    import primus.backends.megatron.core.models.kimi_k3.moonvit_model as model_mod
    from megatron.core.packed_seq_params import PackedSeqParams

    original = model_mod.moonvit_packed_seq_params

    def broken(g):
        total = int(g.prod(dim=-1).sum())
        cu = torch.tensor([0, total], dtype=torch.int32, device=g.device)
        return PackedSeqParams(
            qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
            max_seqlen_q=total, max_seqlen_kv=total,
        )

    model_mod.moonvit_packed_seq_params = broken
    try:
        with torch.no_grad():
            base = model(px, grid_thws)
            perturbed = px.clone()
            perturbed[0] += 10.0
            after = model(perturbed, grid_thws)
    finally:
        model_mod.moonvit_packed_seq_params = original

    first_item_tokens = (4 // 2) * (4 // 2)
    leak = max_abs(after[first_item_tokens:], base[first_item_tokens:])
    assert leak > 1e-6, "cross-item leakage is invisible; the block-diagonality test has no power"
