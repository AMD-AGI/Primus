###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the ``situ`` activation.

Two things need pinning:

* the **math**, against :class:`HFSituAndMul` below — a transcription of
  ``SituAndMul`` (``modeling_kimi_linear.py:64-82``) that shares no code
  with the implementation under test. fp32 agreement is asserted
  bit-for-bit with ``torch.equal``, and the gradient is checked with
  ``torch.autograd.gradcheck`` in float64;
* the **wiring**, because ``situ`` is a fused GLU activation and
  Megatron's ``config.activation_func`` hook only ever sees the gate
  half (``mlp.py:312-319``). ``test_mlp_*`` builds a real Megatron
  :class:`MLP` through the ``activation_func`` module slot and shows it
  reproduces an eager ``down(situ(gate, up))``.

Like the KDA tests these are not hardware-gated: the activation is pure
PyTorch. The Megatron ``MLP`` wiring tests need a process group and are
skipped without an accelerator (Megatron's column-parallel init and RNG
tracker both want one).
"""

from __future__ import annotations

import os

import pytest

# transformer_engine SIGABRTs unless torch is imported first (see node/README.md).
import torch
import torch.nn as nn

from primus.backends.megatron.core.transformer.kimi_k3.situ_activation import (
    SituActivation,
    situ_betas_from_config,
    situ_pre_mul,
    situ_pre_mul_fused,
)

# Kimi K3's released values (config.json / DESIGN.md §3.1).
BETA = 4.0
LINEAR_BETA = 25.0


class HFSituAndMul(nn.Module):
    """Verbatim transcription of ``SituAndMul`` (``modeling_kimi_linear.py:64-82``)."""

    def __init__(self, beta: float = 1.0, linear_beta: float | None = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(x.dtype)


# ---------------------------------------------------------------------------
# Math: parity with the HF reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(7, 16), (2, 3, 64), (5,), (1, 1, 2)])
@pytest.mark.parametrize("linear_beta", [LINEAR_BETA, None])
def test_matches_hf_bit_for_bit_in_fp32(kda_device, shape, linear_beta):
    """fp32 parity is exact: same ops, same order, same dtype."""
    x = torch.randn(*shape[:-1], 2 * shape[-1], device=kda_device, dtype=torch.float32)

    got = situ_pre_mul_fused(x, beta=BETA, linear_beta=linear_beta)
    want = HFSituAndMul(beta=BETA, linear_beta=linear_beta)(x)

    assert got.shape == x.shape[:-1] + (shape[-1],)
    assert torch.equal(got, want), (got - want).abs().max().item()


def test_split_and_fused_forms_agree(kda_device):
    """``situ_pre_mul_fused`` is exactly ``situ_pre_mul`` on the two halves."""
    x = torch.randn(4, 6, 32, device=kda_device, dtype=torch.float32)
    gate, up = x.chunk(2, dim=-1)

    assert torch.equal(
        situ_pre_mul_fused(x, beta=BETA, linear_beta=LINEAR_BETA),
        situ_pre_mul(gate, up, beta=BETA, linear_beta=LINEAR_BETA),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_matches_hf_and_keeps_dtype(kda_device, dtype):
    """Low precision: HF's fp32 interior is reproduced, output dtype preserved."""
    x = torch.randn(3, 8, 24, device=kda_device, dtype=dtype)

    got = situ_pre_mul_fused(x, beta=BETA, linear_beta=LINEAR_BETA)
    want = HFSituAndMul(beta=BETA, linear_beta=LINEAR_BETA)(x)

    assert got.dtype == dtype
    assert torch.equal(got, want)


def test_gradcheck_float64(kda_device):
    """Gradient correctness in float64.

    HF hardcodes ``.to(torch.float32)``, which would silently truncate a
    float64 gradcheck; the implementation promotes instead, so this is a
    genuine check of the analytic gradient.
    """
    x = torch.randn(3, 8, dtype=torch.float64, device=kda_device, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: situ_pre_mul_fused(t, beta=BETA, linear_beta=LINEAR_BETA), (x,))


def test_float64_is_not_truncated_to_fp32(kda_device):
    """The promotion is observable: fp32 interior loses float64 precision."""
    x = torch.randn(4, 8, dtype=torch.float64, device=kda_device)

    promoted = situ_pre_mul_fused(x, beta=BETA, linear_beta=LINEAR_BETA)
    truncated = HFSituAndMul(beta=BETA, linear_beta=LINEAR_BETA)(x)

    assert promoted.dtype == torch.float64
    assert not torch.equal(promoted, truncated)
    torch.testing.assert_close(promoted, truncated, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# Math: the properties the soft clamps exist for
# ---------------------------------------------------------------------------


def test_gate_branch_is_silu_with_a_soft_clamped_linear_factor(kda_device):
    """``situ_a`` is ``silu`` with its linear factor clamped to ±beta.

    ``silu(g) = g * sigmoid(g)``; ``situ_a`` replaces the bare ``g`` with
    ``beta * tanh(g / beta)``. So ``|situ_a| <= beta`` everywhere, and
    ``situ_a -> silu`` as ``|g| -> 0``. The bound is closed rather than
    open in floating point: ``tanh`` saturates to exactly 1.0.
    """
    gate = torch.linspace(-500.0, 500.0, 4001, device=kda_device, dtype=torch.float32)
    ones = torch.ones_like(gate)

    situ_a = situ_pre_mul(gate, ones, beta=BETA, linear_beta=None)

    assert situ_a.abs().max().item() <= BETA
    # The clamped linear factor saturates at +beta for large positive gate
    # (sigmoid -> 1) and at 0 for large negative gate (sigmoid -> 0).
    assert situ_a[-1].item() == pytest.approx(BETA, rel=1e-6)
    assert situ_a[0].item() == pytest.approx(0.0, abs=1e-12)

    small = torch.linspace(-0.5, 0.5, 101, device=kda_device, dtype=torch.float32)
    torch.testing.assert_close(
        situ_pre_mul(small, torch.ones_like(small), beta=BETA, linear_beta=None),
        torch.nn.functional.silu(small),
        rtol=2e-3,
        atol=2e-3,
    )


def test_up_branch_is_soft_clamped(kda_device):
    """``linear_beta * tanh(up / linear_beta)`` bounds the up branch to ±linear_beta."""
    up = torch.linspace(-1000.0, 1000.0, 2001, device=kda_device, dtype=torch.float32)
    # A gate of +inf-ish makes situ_a exactly beta, isolating the up branch.
    gate = torch.full_like(up, 1e4)

    out = situ_pre_mul(gate, up, beta=BETA, linear_beta=LINEAR_BETA) / BETA

    assert out.abs().max().item() <= LINEAR_BETA
    assert out[-1].item() == pytest.approx(LINEAR_BETA, rel=1e-6)
    torch.testing.assert_close(out, LINEAR_BETA * torch.tanh(up / LINEAR_BETA), rtol=1e-6, atol=1e-6)


def test_large_betas_recover_plain_swiglu(kda_device):
    """Both soft clamps vanish as their bounds grow: ``situ -> silu(gate) * up``."""
    x = torch.randn(6, 32, device=kda_device, dtype=torch.float64)
    gate, up = x.chunk(2, dim=-1)

    got = situ_pre_mul(gate, up, beta=1e8, linear_beta=1e8)
    want = torch.nn.functional.silu(gate) * up

    torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-9)


def test_output_is_bounded_by_the_product_of_the_two_bounds(kda_device):
    """The point of the activation: a hard bound for FP8 / FP4 range."""
    x = torch.randn(64, 128, device=kda_device, dtype=torch.float32) * 1e3

    out = situ_pre_mul_fused(x, beta=BETA, linear_beta=LINEAR_BETA)

    assert out.abs().max().item() <= BETA * LINEAR_BETA
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_betas_come_from_the_config_field_names_wp1_declared():
    """``activation_situ_beta`` / ``activation_situ_linear_beta``."""
    from primus.backends.megatron.core.models.kimi_k3 import KimiK3TransformerConfig

    config = KimiK3TransformerConfig(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        kv_channels=16,
        rope_type="rope",
        sequence_parallel=False,
        activation_situ_beta=BETA,
        activation_situ_linear_beta=LINEAR_BETA,
    )

    assert situ_betas_from_config(config) == (BETA, LINEAR_BETA)


class _Bag:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.mark.parametrize(
    "attrs, expected",
    [
        # HF's `beta or 1.0` (modeling_kimi_linear.py:88-91): unset and zero
        # both mean 1.0, which is a real soft clamp and not "clamping off".
        ({}, (1.0, None)),
        ({"activation_situ_beta": None, "activation_situ_linear_beta": None}, (1.0, None)),
        ({"activation_situ_beta": 0.0}, (1.0, None)),
        ({"activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0}, (4.0, 25.0)),
        ({"activation_situ_linear_beta": 25.0}, (1.0, 25.0)),
    ],
)
def test_betas_fallbacks_match_hf(attrs, expected):
    assert situ_betas_from_config(_Bag(**attrs)) == expected


def test_module_reads_the_config_and_holds_no_parameters(kda_device):
    act = SituActivation(config=_Bag(activation_situ_beta=BETA, activation_situ_linear_beta=LINEAR_BETA))

    assert (act.beta, act.linear_beta) == (BETA, LINEAR_BETA)
    assert list(act.parameters()) == []
    assert act.state_dict() == {}

    x = torch.randn(5, 16, device=kda_device, dtype=torch.float32)
    assert torch.equal(act(x), situ_pre_mul_fused(x, beta=BETA, linear_beta=LINEAR_BETA))


def test_module_explicit_betas_override_the_config(kda_device):
    act = SituActivation(
        config=_Bag(activation_situ_beta=BETA, activation_situ_linear_beta=LINEAR_BETA),
        beta=2.0,
        linear_beta=3.0,
    )
    assert (act.beta, act.linear_beta) == (2.0, 3.0)


# ---------------------------------------------------------------------------
# Shape contracts
# ---------------------------------------------------------------------------


def test_odd_last_dim_raises(kda_device):
    with pytest.raises(ValueError, match=r"\[gate \| up\] last dim"):
        situ_pre_mul_fused(torch.randn(4, 7, device=kda_device), beta=BETA)


def test_mismatched_split_shapes_raise(kda_device):
    with pytest.raises(ValueError, match="matching gate / up shapes"):
        situ_pre_mul(torch.randn(4, 8, device=kda_device), torch.randn(4, 9, device=kda_device))


# ---------------------------------------------------------------------------
# Wiring: a real Megatron MLP
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 64
FFN_HIDDEN_SIZE = 128


@pytest.fixture(scope="module")
def tp1_process_group():
    """A 1-rank gloo process group with Megatron model-parallel state.

    Same fixture as ``test_kda_module.py``; ``model_parallel_cuda_manual_seed``
    is what makes the TP RNG tracker usable.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29573")
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


def _mlp_config():
    import torch.nn.functional as F
    from megatron.core.transformer import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        # situ is a GLU activation: fc1 emits [gate | up] at 2 * ffn_hidden_size
        # and fc2 consumes ffn_hidden_size.
        gated_linear_unit=True,
        # Route the activation to the fused module slot (mlp.py:226-229, 256-259).
        use_te_activation_func=True,
        # Kept only to satisfy the {gelu, silu, relu} whitelist that guards
        # use_te_activation_func (transformer_config.py:1638-1644); the module
        # slot wins, so this callable is never invoked.
        activation_func=F.silu,
        bias_activation_fusion=False,
        add_bias_linear=False,
        params_dtype=torch.float32,
        init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
        output_layer_init_method=lambda w: torch.nn.init.normal_(w, std=0.02),
        use_cpu_initialization=True,
        perform_initialization=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


def _build_mlp(config):
    from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.mlp import MLP, MLPSubmodules

    return MLP(
        config=config,
        submodules=MLPSubmodules(
            linear_fc1=ColumnParallelLinear,
            linear_fc2=RowParallelLinear,
            activation_func=SituActivation,
        ),
        ffn_hidden_size=FFN_HIDDEN_SIZE,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron MLP needs an accelerator")
def test_mlp_activation_slot_receives_the_fused_tensor(tp1_process_group):
    """The module slot is what makes situ reachable at all.

    ``config.activation_func`` would only ever see the gate half
    (``mlp.py:312-319``), so the ``up``-branch soft clamp has nowhere to
    live. Assert the slot is wired and that it sees ``2 * ffn_hidden_size``.
    """
    config = _mlp_config()
    config.activation_situ_beta = BETA
    config.activation_situ_linear_beta = LINEAR_BETA
    mlp = _build_mlp(config).cuda()

    assert isinstance(mlp.activation_func, SituActivation)
    assert (mlp.activation_func.beta, mlp.activation_func.linear_beta) == (BETA, LINEAR_BETA)

    seen = []
    mlp.activation_func.register_forward_pre_hook(lambda _m, args: seen.append(args[0].shape))

    x = torch.randn(6, 2, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    out, bias = mlp(x)

    assert bias is None
    assert out.shape == x.shape
    assert seen == [torch.Size([6, 2, 2 * FFN_HIDDEN_SIZE])]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron MLP needs an accelerator")
def test_mlp_matches_an_eager_situ_ffn(tp1_process_group):
    """End to end: the Megatron MLP equals ``down(situ(gate, up))``.

    The eager reference mirrors ``KimiMLP.forward``
    (``modeling_kimi_linear.py:294-301``), which concatenates
    ``gate_proj`` / ``up_proj`` and calls the activation on the pair.
    """
    config = _mlp_config()
    config.activation_situ_beta = BETA
    config.activation_situ_linear_beta = LINEAR_BETA
    mlp = _build_mlp(config).cuda()

    x = torch.randn(6, 2, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    out, _ = mlp(x)

    # fc1 is column-parallel with stride=2: at tp_size == 1 the weight rows are
    # plain [gate | up] (mlp.py:198-204).
    w1 = mlp.linear_fc1.weight
    gate = x @ w1[:FFN_HIDDEN_SIZE].t()
    up = x @ w1[FFN_HIDDEN_SIZE:].t()
    want = HFSituAndMul(beta=BETA, linear_beta=LINEAR_BETA)(torch.cat([gate, up], dim=-1))
    want = want @ mlp.linear_fc2.weight.t()

    torch.testing.assert_close(out, want, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Distributed checkpointing (WP7)
# ---------------------------------------------------------------------------


def test_situ_activation_has_a_sharded_state_dict():
    """Without this, no Kimi K3 checkpoint can be saved at all.

    ``MLP.sharded_state_dict`` walks ``self._modules`` and calls
    ``module.sharded_state_dict(...)`` on every child *unconditionally*
    (``mlp.py:348-363``) -- unlike ``sharded_state_dict_default``, which guards
    the call with ``hasattr`` (``utils.py:240-253``). ``SituActivation`` sits in
    the ``activation_func`` slot of the dense MLP on layer 0 and of the shared
    experts on every MoE layer, so a save raised ``AttributeError`` before this
    method existed. Found by WP7's first checkpoint save.
    """
    act = SituActivation(beta=BETA, linear_beta=LINEAR_BETA)
    assert hasattr(act, "sharded_state_dict")
    assert act.sharded_state_dict(prefix="mlp.activation_func.") == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron MLP needs an accelerator")
def test_mlp_with_situ_can_produce_a_sharded_state_dict(tp1_process_group):
    """The regression test for the save path, one level up from the unit above."""
    from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group

    config = _mlp_config()
    mlp = _build_mlp(config).cuda()

    # ``MLP.sharded_state_dict`` passes ``metadata`` straight through to its
    # children, and ``ColumnParallelLinear.sharded_state_dict`` dereferences
    # ``metadata['dp_cp_group']`` (``layers.py:1049``). In a real save the
    # enclosing ``sharded_state_dict_default`` has already filled it in
    # (``utils.py:238``); calling an MLP directly means doing that here.
    metadata = ensure_metadata_has_dp_cp_group(None)
    sharded = mlp.sharded_state_dict(prefix="mlp.", metadata=metadata)
    assert "mlp.linear_fc1.weight" in sharded
    assert "mlp.linear_fc2.weight" in sharded
    # Parameter-less, so it contributes no keys -- but it must not raise.
    assert not [k for k in sharded if "activation_func" in k]
