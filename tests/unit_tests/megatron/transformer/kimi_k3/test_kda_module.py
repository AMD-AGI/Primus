###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Module-level tests for :class:`KimiDeltaAttention` at ``tp_size == 1``.

The kernel numerics are pinned by ``test_kda_collapse_to_gated_delta_rule``
and ``test_kda_eager_reference``; what is left to verify is the *wiring*:
that the module assembles the projections, the three separate causal
convolutions, the gate, the recurrence and the gated output norm into the
same function the HF reference computes.

The comparison is against :class:`HFKdaReference`, a self-contained
transcription of HF's ``KimiDeltaAttention`` that shares no code with the
implementation under test, with weights copied by name.

Multi-rank TP is out of scope for this work package; ``tp_size == 1`` is
what these tests exercise (see the ``KimiDeltaAttention`` docstring).
"""

from __future__ import annotations

import os

import pytest
import torch

from tests.unit_tests.megatron.transformer.kimi_k3.kda_reference_impls import (
    HFKdaReference,
    assert_close_scaled,
)

HIDDEN_SIZE = 64
NUM_HEADS = 4
HEAD_DIM = 16
CONV_SIZE = 4
NORM_EPS = 1e-5


@pytest.fixture(scope="module")
def tp1_process_group():
    """A 1-rank gloo process group with Megatron model-parallel state.

    ``model_parallel_cuda_manual_seed`` is required, not optional:
    ``KimiDeltaAttention.reset_parameters`` forks the TP RNG tracker (as
    ``GatedDeltaNet`` does), and the tracker raises
    ``"cuda rng state model-parallel-rng is not added"`` until a seed has
    been registered.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29571")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
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


def _make_config():
    from megatron.core.transformer import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        ffn_hidden_size=4 * HIDDEN_SIZE,
        linear_num_key_heads=NUM_HEADS,
        linear_num_value_heads=NUM_HEADS,
        linear_key_head_dim=HEAD_DIM,
        linear_value_head_dim=HEAD_DIM,
        linear_conv_kernel_dim=CONV_SIZE,
        layernorm_epsilon=NORM_EPS,
        params_dtype=torch.float32,
        init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
        output_layer_init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
        use_cpu_initialization=True,
        perform_initialization=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


def _make_submodules():
    from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

    from primus.backends.megatron.core.transformer.kimi_k3 import (
        KimiDeltaAttentionSubmodules,
        KimiGatedRMSNorm,
    )

    return KimiDeltaAttentionSubmodules(
        q_proj=ColumnParallelLinear,
        k_proj=ColumnParallelLinear,
        v_proj=ColumnParallelLinear,
        f_a_proj=ColumnParallelLinear,
        f_b_proj=ColumnParallelLinear,
        b_proj=ColumnParallelLinear,
        g_proj=ColumnParallelLinear,
        out_norm=KimiGatedRMSNorm,
        o_proj=RowParallelLinear,
    )


def _build_kda_with_config(tp1, config, **kwargs):
    from megatron.core.process_groups_config import ProcessGroupCollection

    from primus.backends.megatron.core.transformer.kimi_k3 import KimiDeltaAttention

    return KimiDeltaAttention(
        config=config,
        submodules=_make_submodules(),
        layer_number=1,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        **kwargs,
    )


def _build_kda(tp1, **kwargs):
    return _build_kda_with_config(tp1, _make_config(), **kwargs)


def _copy_weights_to_reference(kda, ref: HFKdaReference) -> None:
    """Move ``KimiDeltaAttention``'s weights onto the HF-shaped reference.

    At ``tp_size == 1`` every parallel linear holds the full weight, so
    the mapping is a straight copy. ``ColumnParallelLinear`` and
    ``nn.Linear`` share the ``[out, in]`` layout.
    """
    with torch.no_grad():
        for name in ("q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj", "b_proj", "g_proj", "o_proj"):
            getattr(ref, name).weight.copy_(getattr(kda, name).weight)
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            getattr(ref, name).weight.copy_(getattr(kda, name).weight)
            if getattr(kda, name).bias is not None:
                getattr(ref, name).bias.copy_(getattr(kda, name).bias)
        ref.A_log.copy_(kda.A_log)
        ref.dt_bias.copy_(kda.dt_bias)
        ref.o_norm_weight.copy_(kda.out_norm.weight)


def test_forward_matches_the_hf_reference(tp1_process_group, kda_device):
    """Identical weights and inputs => identical output, to fp32 tolerance."""
    torch.manual_seed(3)
    kda = _build_kda(tp1_process_group).to(kda_device).eval()
    ref = HFKdaReference(HIDDEN_SIZE, NUM_HEADS, HEAD_DIM, conv_size=CONV_SIZE, rms_norm_eps=NORM_EPS).to(
        kda_device
    )
    _copy_weights_to_reference(kda, ref)

    batch, seq_len = 2, 96
    x_bsh = torch.randn(batch, seq_len, HIDDEN_SIZE, device=kda_device)
    # KimiDeltaAttention is sequence-first; the HF reference is batch-first.
    got, bias = kda(x_bsh.transpose(0, 1).contiguous(), attention_mask=None)
    got = got.transpose(0, 1)
    if bias is not None:
        got = got + bias
    want = ref(x_bsh)

    assert got.shape == want.shape == (batch, seq_len, HIDDEN_SIZE)
    assert_close_scaled(got, want, 1e-5, "module vs HF reference")


def test_forward_shape_and_causality(tp1_process_group, kda_device):
    """Output keeps the ``[s, b, h]`` layout and does not look ahead."""
    torch.manual_seed(4)
    kda = _build_kda(tp1_process_group).to(kda_device).eval()

    seq_len, batch, cut = 96, 2, 50
    x = torch.randn(seq_len, batch, HIDDEN_SIZE, device=kda_device)
    base, _ = kda(x, attention_mask=None)
    assert base.shape == (seq_len, batch, HIDDEN_SIZE)

    x2 = x.clone()
    x2[cut + 1 :] = torch.randn(seq_len - cut - 1, batch, HIDDEN_SIZE, device=kda_device)
    perturbed, _ = kda(x2, attention_mask=None)

    prefix = (base[: cut + 1] - perturbed[: cut + 1]).abs().max().item()
    suffix = (base[cut + 1 :] - perturbed[cut + 1 :]).abs().max().item()
    print(f"[module causality] prefix max|d|={prefix:.3e}  suffix max|d|={suffix:.3e}")
    assert prefix == 0.0, f"output before the cut moved by {prefix:.3e}; the module looks ahead"
    assert suffix > 0.0, "the perturbation had no effect; the test would be vacuous"


def test_backward_reaches_every_parameter(tp1_process_group, kda_device):
    torch.manual_seed(5)
    kda = _build_kda(tp1_process_group).to(kda_device)
    x = torch.randn(64, 2, HIDDEN_SIZE, device=kda_device, requires_grad=True)
    out, _ = kda(x, attention_mask=None)
    out.float().sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in kda.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name}.grad has non-finite entries"


def test_parameter_shapes_follow_the_hf_reference(tp1_process_group):
    """Guard the parameter geometry against silent drift from the HF layout."""
    kda = _build_kda(tp1_process_group)
    projection_size = NUM_HEADS * HEAD_DIM
    expected = {
        "q_proj.weight": (projection_size, HIDDEN_SIZE),
        "k_proj.weight": (projection_size, HIDDEN_SIZE),
        "v_proj.weight": (projection_size, HIDDEN_SIZE),
        "f_a_proj.weight": (HEAD_DIM, HIDDEN_SIZE),
        "f_b_proj.weight": (projection_size, HEAD_DIM),
        "b_proj.weight": (NUM_HEADS, HIDDEN_SIZE),
        "g_proj.weight": (projection_size, HIDDEN_SIZE),
        "o_proj.weight": (HIDDEN_SIZE, projection_size),
        "A_log": (NUM_HEADS,),
        "dt_bias": (projection_size,),
        "out_norm.weight": (HEAD_DIM,),
        "q_conv1d.weight": (projection_size, 1, CONV_SIZE),
        "k_conv1d.weight": (projection_size, 1, CONV_SIZE),
        "v_conv1d.weight": (projection_size, 1, CONV_SIZE),
    }
    actual = {name: tuple(p.shape) for name, p in kda.named_parameters()}
    assert actual == expected, f"parameter geometry drifted:\n got {actual}\n want {expected}"


def test_default_initialisation_keeps_retention_near_one(tp1_process_group):
    """``A_log = 0`` and the Mamba ``dt_bias`` must start ``alpha`` close to 1.

    The two initialisation choices are documented (and contested) in the
    ``KimiDeltaAttention`` docstring; this test states what the defaults
    are *for*, so changing them cannot go unnoticed.
    """
    kda = _build_kda(tp1_process_group)
    assert torch.allclose(kda.A_log, torch.zeros_like(kda.A_log)), "default A_log must be 0 (report §2.1.1)"
    # g = -5 * sigmoid(exp(0) * (0 + dt_bias)) with the zero-mean gate
    # pre-activation, so alpha = exp(g) at init depends only on dt_bias.
    alpha = torch.exp(-5.0 * torch.sigmoid(kda.dt_bias))
    print(
        f"[init] dt_bias in [{kda.dt_bias.min():.3f}, {kda.dt_bias.max():.3f}]  "
        f"alpha in [{alpha.min():.4f}, {alpha.max():.4f}]"
    )
    assert (kda.dt_bias < 0).all(), "dt_bias must be negative so initial retention is near 1"
    assert alpha.min() > 0.5, f"initial retention {alpha.min():.4f} is too forgetful"


def test_sharded_state_dict_covers_every_parameter(tp1_process_group):
    """``sharded_state_dict`` must build and name every parameter.

    Only the TP=1 path is exercised here; the axis-0 sharding of
    ``A_log`` / ``dt_bias`` / the conv weights is declarative and needs a
    multi-rank run to validate, which belongs with the parallelism work
    package.
    """
    kda = _build_kda(tp1_process_group)
    sharded = kda.sharded_state_dict(prefix="kda.")

    assert sharded, "sharded_state_dict returned nothing"
    for name, _ in kda.named_parameters():
        key = f"kda.{name}"
        assert key in sharded, f"{key} missing from sharded_state_dict (got {sorted(sharded)})"
    for key in ("kda.A_log", "kda.dt_bias", "kda.q_conv1d.weight"):
        assert getattr(sharded[key], "replica_id", None) is not None or hasattr(
            sharded[key], "global_shape"
        ), f"{key} was not wrapped as a sharded tensor"


def test_backend_selector_is_validated_and_resolved_at_construction(tp1_process_group):
    """A bad ``kda_backend`` must fail while building, not on the first forward.

    Mirrors how ``DeepseekV4Attention`` validates its string selector
    against a whitelist in ``__init__`` and eagerly loads the selected
    backend (``deepseek_v4_attention.py:695-730, 737-771``).
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        eager_chunk_kda,
    )

    kda = _build_kda(tp1_process_group)
    assert kda.backend_name == "eager"
    assert kda.kda_backend is eager_chunk_kda, "the backend must be resolved once, at construction"

    config = _make_config()
    config.kda_backend = "triton_v99"
    with pytest.raises(ValueError, match="kda_backend must be one of"):
        _build_kda_with_config(tp1_process_group, config)

    # ``flydsl`` is hardware-gated (gfx950 / CDNA4). Either way the outcome must
    # be decided at construction: the kernel is resolved eagerly where it is
    # available, and where it is not the failure is an actionable ImportError
    # naming the fallbacks — never a crash on the first forward.
    have_flydsl = True
    try:
        import flydsl  # noqa: F401
    except ImportError:
        have_flydsl = False
    on_gfx950 = torch.cuda.is_available() and str(
        getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    ).startswith("gfx950")

    config = _make_config()
    config.kda_backend = "flydsl"
    if have_flydsl and on_gfx950:
        kda_fly = _build_kda_with_config(tp1_process_group, config)
        assert kda_fly.backend_name == "flydsl"
        assert callable(kda_fly.kda_backend)
        assert kda_fly.kda_backend is not eager_chunk_kda
    else:
        with pytest.raises(ImportError, match="eager | eager_recurrent | fla"):
            _build_kda_with_config(tp1_process_group, config)


def test_low_rank_output_gate_is_rejected(tp1_process_group):
    """Kimi Linear's low-rank output gate is out of scope; fail loudly."""
    config = _make_config()
    config.kda_use_full_rank_gate = False
    with pytest.raises(NotImplementedError, match="full-rank KDA output gate"):
        _build_kda_with_config(tp1_process_group, config)


# ---------------------------------------------------------------------------
# Tensor parallelism above one rank (WP7)
# ---------------------------------------------------------------------------
#
# The suite is single-process, so ``pg_collection.tp.size()`` is always 1 here.
# These tests therefore exercise the two TP-only code paths by calling them
# directly with the state a multi-rank build would have. The multi-rank
# numerical validation that motivated both is in ``wp7/RESULTS.md``:
# ``f_b_proj`` under sequence parallelism raised
# ``RuntimeError: shape '[2, 128, 4, 32]' is invalid for input of size 65536``,
# and ``out_norm.weight``'s gradient deviated from the TP=1 gradient by
# rel_rms 0.78 (TP=2) / 0.91 (TP=4) in fp32 while its TP *sum* matched to
# 1.4e-7.


def test_f_b_proj_is_built_with_sequence_parallel_off(tp1_process_group):
    """``f_b_proj`` must not all-gather the sequence a second time.

    ``f_a_proj`` is a ``ColumnParallelLinear(gather_output=True)``, so under
    sequence parallelism it gathers the sequence *and* the output width and
    hands ``f_b_proj`` a full-sequence, full-width, TP-replicated tensor. A
    sequence-parallel ``f_b_proj`` would gather the sequence again and emit
    ``[s * tp, b, ...]``.
    """
    kda = _build_kda(tp1_process_group)

    # sequence_parallel cannot be set on the config at construction time --
    # TransformerConfig rejects it at tensor_model_parallel_size == 1 -- so it
    # is set here, which is exactly the state a TP>1 build sees.
    kda.config.sequence_parallel = True
    f_b_config = kda._f_b_proj_config()
    assert f_b_config is not kda.config, "must be a copy, not a mutation"
    assert f_b_config.sequence_parallel is False
    assert kda.config.sequence_parallel is True, "the shared config was mutated"
    # Nothing else may differ: the copy exists for one flag.
    differing = [
        f.name
        for f in kda.config.__dataclass_fields__.values()
        if getattr(kda.config, f.name, None) is not getattr(f_b_config, f.name, None)
    ]
    assert differing == ["sequence_parallel"], differing


def test_f_b_proj_config_is_the_shared_config_when_sp_is_off(tp1_process_group):
    """No copy when there is nothing to override."""
    kda = _build_kda(tp1_process_group)
    assert kda.config.sequence_parallel is False
    assert kda._f_b_proj_config() is kda.config


def test_out_norm_gain_is_flagged_so_its_gradient_is_summed_over_tp(tp1_process_group):
    """``out_norm``'s gain is shared across heads while the heads are sharded.

    Each TP rank therefore computes only the sum over *its* heads. The
    ``sequence_parallel`` attribute is what makes
    ``_allreduce_non_tensor_model_parallel_grads`` add them up
    (``finalize_model_grads.py:357-370``); without it every rank applies a
    different partial gradient to a parameter that is supposed to be
    replicated.
    """
    kda = _build_kda(tp1_process_group)
    gains = list(kda.out_norm.parameters())
    assert gains, "out_norm has no parameters to flag"

    # At tp_size == 1 there is nothing to reduce and the flag stays off.
    assert all(not getattr(p, "sequence_parallel", False) for p in gains)

    kda.tp_size = 2
    kda.config.sequence_parallel = True
    kda._mark_out_norm_grads_for_tp_reduction()
    assert all(getattr(p, "sequence_parallel", False) for p in gains)


def test_tensor_parallel_without_sequence_parallel_is_refused(tp1_process_group):
    """The reduction only runs under ``config.sequence_parallel``.

    So TP>1 with it off would silently desynchronise ``out_norm.weight``
    across ranks. MoE + TP>1 is refused at forward time without it anyway
    (``moe_layer.py:484-488``).
    """
    kda = _build_kda(tp1_process_group)
    kda.tp_size = 4
    kda.config.sequence_parallel = False
    with pytest.raises(AssertionError, match="requires sequence_parallel=True"):
        kda._mark_out_norm_grads_for_tp_reduction()
