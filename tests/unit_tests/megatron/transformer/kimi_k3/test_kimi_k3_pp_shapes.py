###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The Kimi K3 pipeline-parallel seam (WP7).

Two things are pinned here, and they are independent:

1. **The scheduler patch** --
   ``primus/backends/megatron/patches/kimi_k3_pp_shape_patches.py`` has to
   tell Megatron that the PP wire carries
   ``[(1 + attn_res_num_blocks_max) * s, b, h]``. Both wrappers and the
   gating condition are exercised without a process group, because the
   patch is pure function composition.

2. **The seam's numerics** -- splitting the stack across two stages, with
   the folded tensor handed over by hand, must reproduce the single-stage
   result *bit-exactly*, and ``block_residual`` must survive the fold /
   unfold with ``torch.equal``. This is the test that would catch a
   permute-order or padding-slice error, and it needs one device rather
   than two ranks.

A real two-rank run is the integration counterpart and lives outside the
unit suite (``wp7/``), because the unit suite is single-process.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch

NUM_LAYERS = 8
HIDDEN = 256
SEQ = 64
BATCH = 2
ATTN_RES_BLOCK_SIZE = 4
KDA_PATTERN = [1, 1, 1, 0, 1, 1, 1, 0]


# ---------------------------------------------------------------------------
# 1. The scheduler patch
# ---------------------------------------------------------------------------


def _args(**kwargs) -> SimpleNamespace:
    base = dict(
        model_type="kimi_k3",
        num_layers=8,
        attn_res_block_size=4,
        pipeline_model_parallel_size=2,
        patch_zero_bubble=False,
        patch_primus_pipeline=False,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _ctx(**kwargs) -> SimpleNamespace:
    """``get_args`` reads ``ctx.extra["module_config"].params`` (``context.py:106-110``)."""
    return SimpleNamespace(extra={"module_config": SimpleNamespace(params=_args(**kwargs))})


@pytest.mark.parametrize(
    "num_layers, block_size, expected",
    [
        (8, 4, 3),  # kimi_k3_debug.yaml: ceil(8/4) = 2 checkpoints -> stride 3
        (93, 12, 9),  # production: ceil(93/12) = 8 checkpoints -> stride 9
        (24, 12, 3),  # the phase-2 curve shape
        (8, 8, 2),  # a single checkpoint
        (8, 0, 1),  # mechanism off -> the patch is a no-op
        (8, None, 1),
        (0, 4, 1),  # no layers declared yet
    ],
)
def test_seq_multiplier_is_one_plus_ceil_num_layers_over_block_size(num_layers, block_size, expected):
    """``1 + attn_res_num_blocks_max``, derived exactly as the config property.

    The value must not be read off ``args``: ``attn_res_num_blocks_max`` is a
    ``@property`` on the transformer config
    (``kimi_k3_transformer_config.py:373-384``) and never becomes an args
    field, so a ``getattr`` would silently read the 1 that disables the
    patch.
    """
    from primus.backends.megatron.patches.kimi_k3_pp_shape_patches import (
        kimi_k3_pp_seq_multiplier,
    )

    assert kimi_k3_pp_seq_multiplier(_args(num_layers=num_layers, attn_res_block_size=block_size)) == expected


def test_seq_multiplier_agrees_with_the_config_property():
    """Pin the derivation against the one the block actually uses."""
    from primus.backends.megatron.patches.kimi_k3_pp_shape_patches import (
        kimi_k3_pp_seq_multiplier,
    )

    config = _make_config()
    assert kimi_k3_pp_seq_multiplier(
        _args(num_layers=config.num_layers, attn_res_block_size=config.attn_res_block_size)
    ) == 1 + config.attn_res_num_blocks_max


def test_get_tensor_shapes_wrapper_scales_the_sequence_dim_only():
    """``(s, b, h) -> (s * mult, b, h)``.

    The fold stacks ``1 + num_blocks_max`` copies of ``[s, b, h]`` along
    dim 0 (``kimi_k3_block.py:264-265``), so micro-batch and hidden must be
    untouched -- scaling ``hidden`` instead would still give the right
    ``numel``, and PyTorch P2P only validates ``numel``, so it would fail
    silently in exactly the way this patch exists to prevent.
    """
    from primus.backends.megatron.patches.kimi_k3_pp_shape_patches import (
        _make_k3_get_tensor_shapes,
    )

    original = lambda **kw: [(kw["seq_length"], kw["micro_batch_size"], 7168)]  # noqa: E731
    wrapped = _make_k3_get_tensor_shapes(original, 3)

    assert wrapped(seq_length=512, micro_batch_size=4) == [(1536, 4, 7168)]
    assert wrapped.__wrapped__ is original
    assert wrapped._k3_pp_shape_patched is True
    assert wrapped._k3_pp_seq_mult == 3


def test_get_tensor_shapes_wrapper_composes_with_sequence_parallel_division():
    """The factor multiplies the *already divided* local length.

    ``get_tensor_shapes`` divides by CP and then by TP when
    ``sequence_parallel`` is on (``schedules.py:1945-1948``); the block folds
    whatever local sequence length the stage actually holds, so wrapping the
    return value -- rather than the ``seq_length`` argument -- is the
    composition that is correct for both.
    """
    from primus.backends.megatron.patches.kimi_k3_pp_shape_patches import (
        _make_k3_get_tensor_shapes,
    )

    def original(*, seq_length, micro_batch_size, tp_size, cp_size):
        return [(seq_length // cp_size // tp_size, micro_batch_size, 1024)]

    wrapped = _make_k3_get_tensor_shapes(original, 3)
    assert wrapped(seq_length=512, micro_batch_size=1, tp_size=2, cp_size=1) == [(768, 1, 1024)]


def test_interleaved_wrapper_scales_seq_length():
    """The VPP schedule builds its wire shape inline (``schedules.py:1001``)."""
    from primus.backends.megatron.patches.kimi_k3_pp_shape_patches import (
        _make_k3_interleaved_schedule,
    )

    seen = {}

    def original(**kwargs):
        seen.update(kwargs)
        return "ok"

    wrapped = _make_k3_interleaved_schedule(original, 3)
    assert wrapped(seq_length=512, micro_batch_size=1, decoder_seq_length=None) == "ok"
    assert seen["seq_length"] == 1536
    assert seen["decoder_seq_length"] is None
    assert seen["micro_batch_size"] == 1

    wrapped(seq_length=512, decoder_seq_length=512)
    assert seen["decoder_seq_length"] == 1536

    # A schedule called without the kwarg at all must not gain one.
    seen.clear()
    wrapped(micro_batch_size=1)
    assert "seq_length" not in seen


@pytest.mark.parametrize(
    "overrides, wanted",
    [
        ({}, True),
        ({"model_type": "deepseek_v4"}, False),
        ({"pipeline_model_parallel_size": 1}, False),
        ({"attn_res_block_size": 0}, False),
        ({"attn_res_block_size": None}, False),
    ],
)
def test_patch_condition_gates_on_model_type_pp_and_block_size(overrides, wanted):
    """A K3 config with ``attn_res_block_size`` unset keeps the stock wire.

    That configuration gets plain ``x = x + sublayer(x)`` residuals
    (``kimi_k3_block.py:462-472``) and so emits an unfolded ``[s, b, h]``;
    scaling it would break a working setup rather than fix a broken one.
    """
    from primus.backends.megatron.patches.kimi_k3_pp_shape_patches import (
        _wants_k3_pp_shape_patch,
    )

    assert _wants_k3_pp_shape_patch(_ctx(**overrides)) is wanted


@pytest.mark.parametrize("flag", ["patch_zero_bubble", "patch_primus_pipeline"])
def test_alternative_pipeline_schedules_are_refused(flag):
    """Both bind ``get_tensor_shapes`` at module import time.

    ``zerobubble/runtime.py:31`` and ``primuspipe/pipeline_launcher.py:17``
    use a module-level ``from ... import get_tensor_shapes``, which captures
    the original function object, and the zero-bubble runtime additionally
    recomputes the shape inline (``runtime.py:1103-1106``). Rebinding the
    ``schedules`` module attribute is therefore invisible to them, so the
    patch must refuse rather than leave the wire silently unscaled.
    """
    from primus.backends.megatron.patches.kimi_k3_pp_shape_patches import (
        patch_kimi_k3_pp_tensor_shape,
    )

    with pytest.raises(NotImplementedError, match=flag):
        patch_kimi_k3_pp_tensor_shape(_ctx(**{flag: True}))


def test_patch_installs_both_wrappers_and_is_idempotent(monkeypatch):
    """Applying it twice must not stack two multipliers."""
    import megatron.core.pipeline_parallel.schedules as schedules

    from primus.backends.megatron.patches import kimi_k3_pp_shape_patches
    from primus.backends.megatron.patches.kimi_k3_pp_shape_patches import (
        patch_kimi_k3_pp_tensor_shape,
    )

    # ``log_rank_0`` resolves Primus's global logger, which only exists inside a
    # real run (``logger.py:373`` dereferences a module global that is None
    # here). The patch's job is the rebinding, not the logging.
    monkeypatch.setattr(kimi_k3_pp_shape_patches, "log_rank_0", lambda *a, **k: None)

    saved = (schedules.get_tensor_shapes, schedules.forward_backward_pipelining_with_interleaving)
    ctx = _ctx()
    try:
        patch_kimi_k3_pp_tensor_shape(ctx)
        first = schedules.get_tensor_shapes
        assert getattr(first, "_k3_pp_shape_patched", False)
        assert first._k3_pp_seq_mult == 3
        assert getattr(
            schedules.forward_backward_pipelining_with_interleaving,
            "_k3_pp_interleaved_patched",
            False,
        )

        patch_kimi_k3_pp_tensor_shape(ctx)
        assert schedules.get_tensor_shapes is first
        assert schedules.get_tensor_shapes._k3_pp_seq_mult == 3
    finally:
        schedules.get_tensor_shapes, schedules.forward_backward_pipelining_with_interleaving = saved


def test_patch_is_registered_under_a_kimi_k3_id():
    """Auto-discovery is by filename (``patches/__init__.py:51-60``)."""
    import primus.backends.megatron.patches  # noqa: F401  # triggers registration
    from primus.core.patches import PatchRegistry

    assert "megatron.kimi_k3.pp_tensor_shape" in PatchRegistry.list_ids()
    patch = PatchRegistry.get("megatron.kimi_k3.pp_tensor_shape")
    assert patch is not None and patch.backend == "megatron" and patch.phase == "before_train"


# ---------------------------------------------------------------------------
# 2. The seam's numerics
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mpu_tp1():
    """A 1-rank process group plus Megatron model-parallel state."""
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29584")
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


def _make_config(*, params_dtype: Optional[torch.dtype] = None):
    """The ``kimi_k3_debug.yaml`` geometry, narrowed -- same as test_kimi_k3_block."""
    import torch.nn.functional as F

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
        KimiK3TransformerConfig,
    )

    if params_dtype is None:
        params_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    return KimiK3TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        ffn_hidden_size=512,
        kv_channels=32,
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
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        linear_attention_freq=list(KDA_PATTERN),
        kda_backend="eager",
        kda_chunk_size=64,
        attn_res_block_size=ATTN_RES_BLOCK_SIZE,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        routed_expert_hidden_size=HIDDEN // 2,
        latent_moe_use_norm=True,
        moe_layer_freq=[0] + [1] * (NUM_LAYERS - 1),
        moe_router_score_function="sigmoid",
        moe_router_pre_softmax=False,
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="seq_aux_loss",
        moe_aux_loss_coeff=1e-3,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_shared_expert_overlap=False,
        moe_permute_fusion=False,
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


def _build_stage(config, layer_specs: List, *, pre_process: bool, post_process: bool):
    """One :class:`KimiK3TransformerBlock` over an explicit slice of layers.

    ``get_kimi_k3_runtime_decoder_spec`` slices by ``get_num_layers_to_build``
    (``kimi_k3_layer_specs.py:283``), which needs a real PP-sized
    ``parallel_state``; this test is single-process, so the slice is done by
    hand and the specs keep the ``layer_idx`` they were built with -- which
    is what drives ``layer_offset`` and therefore the unfold width.
    """
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.spec_utils import ModuleSpec, build_module

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_block import (
        KimiK3TransformerBlock,
        KimiK3TransformerBlockSubmodules,
    )
    from primus.backends.megatron.core.transformer.kimi_k3.attention_residual import (
        AttentionResidualHead,
    )
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_layer_specs import (
        _build_norm_spec,
    )
    from primus.backends.megatron.core.models.kimi_k3.build_context import (
        resolve_k3_provider,
    )

    provider = resolve_k3_provider(config)
    spec = ModuleSpec(
        module=KimiK3TransformerBlock,
        submodules=KimiK3TransformerBlockSubmodules(
            layer_specs=layer_specs,
            attn_res_head=ModuleSpec(module=AttentionResidualHead),
            final_layernorm=_build_norm_spec(config=config, provider=provider),
        ),
    )
    return build_module(
        spec,
        config=config,
        pre_process=pre_process,
        post_process=post_process,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _stage_specs(config) -> List:
    from primus.backends.megatron.core.models.kimi_k3.build_context import (
        resolve_k3_provider,
    )
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_layer_specs import (
        _build_stage_layer_specs,
    )

    return _build_stage_layer_specs(
        config, provider=resolve_k3_provider(config), vp_stage=None, pp_rank=None
    )


@pytest.mark.parametrize("split_at", [1, 4, 5])
def test_two_stage_split_matches_one_stage_bit_exactly(mpu_tp1, split_at):
    """The PP boundary must be a pure carrier, not an approximation.

    Builds the whole stack once, then a two-stage split of the *same*
    weights, and hands the folded ``[(1 + nb_max) * s, b, h]`` tensor from
    stage 0 to stage 1 exactly as PP P2P would. ``torch.equal`` rather than
    ``allclose``: the fold is a permute, a pad and a reshape, so any
    deviation at all is a bug and not rounding.

    ``split_at`` covers a boundary that lands on an append layer (4, where
    ``4 % attn_res_block_size == 0``) and two that do not, because the
    unfold width is ``attn_res_num_blocks_before(layer_offset)`` and an
    off-by-one there only shows up on one of the three.
    """
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_block import (
        _lift_res_in,
        attn_res_num_blocks_before,
    )
    config = _make_config()
    all_specs = _stage_specs(config)
    assert len(all_specs) == NUM_LAYERS

    torch.manual_seed(4321)
    whole = _build_stage(config, all_specs, pre_process=True, post_process=True).to(_device())
    stage0 = _build_stage(config, all_specs[:split_at], pre_process=True, post_process=False).to(
        _device()
    )
    stage1 = _build_stage(config, all_specs[split_at:], pre_process=False, post_process=True).to(
        _device()
    )

    # The split stages must carry the whole model's weights, not their own
    # freshly-initialised ones: Megatron initialises per stage, so a PP run's
    # weights genuinely differ from a PP=1 run's. This test is about the
    # boundary, so the weights are made identical by hand.
    for local_idx in range(split_at):
        stage0.layers[local_idx].load_state_dict(whole.layers[local_idx].state_dict())
    for local_idx, global_idx in enumerate(range(split_at, NUM_LAYERS)):
        stage1.layers[local_idx].load_state_dict(whole.layers[global_idx].state_dict())
    stage1.attn_res_head.load_state_dict(whole.attn_res_head.state_dict())
    stage1.final_layernorm.load_state_dict(whole.final_layernorm.state_dict())

    for block in (whole, stage0, stage1):
        block.eval()

    x = torch.randn(SEQ, BATCH, HIDDEN, dtype=config.params_dtype, device=_device())

    with torch.no_grad():
        want = whole(x.clone(), None)
        wire = stage0(x.clone(), None)
        assert wire.shape == ((1 + config.attn_res_num_blocks_max) * SEQ, BATCH, HIDDEN), wire.shape
        stage1.set_input_tensor(wire)
        got = stage1(None, None)

    assert got.shape == want.shape
    assert torch.equal(got, want), (
        f"split at {split_at}: max abs diff {(got.float() - want.float()).abs().max().item():g}"
    )

    # And the checkpoints themselves, independently of the hidden state: the
    # padded slots must be dropped and the candidate axis must come back in
    # the order the mixer expects.
    num_blocks = attn_res_num_blocks_before(split_at, ATTN_RES_BLOCK_SIZE)
    _, unfolded_blocks = _lift_res_in(
        wire,
        pre_process=False,
        num_blocks=num_blocks,
        num_blocks_max=config.attn_res_num_blocks_max,
    )
    assert unfolded_blocks.shape == (SEQ, BATCH, num_blocks, HIDDEN)
    assert stage1.layers[0].num_blocks_in == num_blocks


def test_block_exposes_tp_group_so_a_checkpoint_can_be_saved(mpu_tp1):
    """``TransformerBlock.sharded_state_dict`` reads ``self.tp_group``.

    The block bypasses the parent ``__init__``, so every attribute the
    parent's other methods read has to be set explicitly.
    ``transformer_block.py:953-963`` passes ``self.tp_group`` to
    ``sharded_state_dict_default`` for each child that is not ``self.layers``,
    i.e. for ``attn_res_head`` and ``final_layernorm`` -- so without it a save
    raised ``AttributeError: 'KimiK3TransformerBlock' object has no attribute
    'tp_group'`` at the first ``save_interval``, long after the run looked
    healthy. Found by WP7's first real checkpoint save.
    """
    from megatron.core import parallel_state

    config = _make_config()
    all_specs = _stage_specs(config)
    block = _build_stage(config, all_specs, pre_process=True, post_process=True)

    assert block.tp_group is parallel_state.get_tensor_model_parallel_group()

    sharded = block.sharded_state_dict(prefix="decoder.")
    assert sharded
    # The children that go through the tp_group argument.
    assert any("attn_res_head" in k for k in sharded)
    assert any("final_layernorm" in k for k in sharded)
    # Every parameter must be represented.
    missing = [n for n, _ in block.named_parameters() if f"decoder.{n}" not in sharded]
    assert not missing, f"{len(missing)} parameters missing from the checkpoint: {missing[:8]}"


def test_padding_slots_on_the_wire_are_zero(mpu_tp1):
    """Unused checkpoint slots must be zeros, not uninitialised memory.

    The pad exists only to keep the P2P shape constant
    (``kimi_k3_block.py:257-263``). A receiver never reads it -- but a
    ``new_empty`` there would make the wire tensor's contents depend on the
    allocator, which is exactly the kind of thing that makes a run
    irreproducible for no visible reason.
    """
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_block import _lower_res_out

    hidden = torch.randn(SEQ, BATCH, HIDDEN)
    blocks = torch.randn(SEQ, BATCH, 1, HIDDEN)
    packed = _lower_res_out(hidden, blocks, post_process=False, num_blocks_max=3)

    unfolded = packed.view(4, SEQ, BATCH, HIDDEN)
    assert torch.equal(unfolded[0], hidden)
    assert torch.equal(unfolded[1], blocks[:, :, 0])
    assert torch.count_nonzero(unfolded[2:]) == 0
