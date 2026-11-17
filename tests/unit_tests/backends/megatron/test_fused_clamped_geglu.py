###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

LIMIT, OFFSET = 7.0, 1.0


def _fwd_bwd(fn, x, dy):
    x = x.clone().requires_grad_()
    y = fn(x)
    y.backward(dy)
    return y.detach(), x.grad


@pytest.fixture(autouse=True)
def fresh_checks(monkeypatch):
    from primus.backends.megatron.core.fusions import bitwise_check

    monkeypatch.setattr(bitwise_check, "_results", {})


@pytest.mark.parametrize("shape", [(64, 2, 2 * 1536), (37, 1, 2 * 1000)])
def test_matches_eager_bitwise(shape, require_bitwise):
    from primus.backends.megatron.core.fusions.bitwise_check import same_bits
    from primus.backends.megatron.core.fusions.fused_clamped_geglu import (
        edge_values,
        eligible,
        fused_clamped_quick_geglu,
        reference,
    )

    torch.manual_seed(0)
    x = (4 * torch.randn(shape, device="cuda")).to(torch.bfloat16)
    # the clamp bounds and their neighbours (torch's clamp backward differs between
    # versions there), signed zeros, infinities and NaN, in both halves of other rows
    # than the self-check's
    edges = edge_values(LIMIT, OFFSET).cuda()
    rows, half = x.view(-1, shape[-1]), shape[-1] // 2
    rows[-1, : edges.numel()] = edges
    rows[-1, half : half + edges.numel()] = edges
    rows[-2, : edges.numel()] = edges.flip(0)
    rows[-2, half : half + edges.numel()] = edges
    dy = torch.randn(*shape[:-1], half, device="cuda").to(torch.bfloat16)
    require_bitwise(eligible(x.shape, x.dtype, x.device, LIMIT, OFFSET), "fused_clamped_geglu")

    ref_y, ref_dx = _fwd_bwd(lambda t: reference(t, LIMIT, OFFSET), x, dy)
    y, dx = _fwd_bwd(lambda t: fused_clamped_quick_geglu(t, LIMIT, OFFSET), x, dy)

    assert same_bits(y, ref_y)
    assert same_bits(dx, ref_dx)


def test_edge_values():
    from primus.backends.megatron.core.fusions.fused_clamped_geglu import edge_values

    v = edge_values(LIMIT, OFFSET).float()
    assert v.dtype == torch.float32 and edge_values(LIMIT, OFFSET).dtype == torch.bfloat16
    assert {LIMIT, -LIMIT, -OFFSET, float("inf"), float("-inf")} <= set(v.tolist())
    assert v.isnan().any() and (v == 0).sum() == 2 and v[(v == 0)].signbit().sum() == 1
    # the BF16 neighbours of the bounds
    assert ((v > LIMIT) & (v < 7.1)).any() and ((v < LIMIT) & (v > 6.9)).any()


def test_not_eligible_without_torch_compile(monkeypatch, require_bitwise):
    from primus.backends.megatron.core.fusions.fused_clamped_geglu import eligible

    args = ((4, 2 * 64), torch.bfloat16, torch.device("cuda"), LIMIT, OFFSET)
    require_bitwise(eligible(*args), "fused_clamped_geglu")
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    assert not eligible(*args)


@pytest.mark.parametrize(("limit", "offset"), [(7.3, OFFSET), (LIMIT, 0.1)])
def test_not_eligible_when_bf16_does_not_hold_the_scalars(limit, offset):
    from primus.backends.megatron.core.fusions.fused_clamped_geglu import (
        ineligible_reason,
    )

    reason = ineligible_reason((4, 2 * 64), torch.bfloat16, torch.device("cuda"), limit, offset)
    assert reason is not None and "exact in BF16" in reason


def test_not_eligible_when_the_self_check_fails(monkeypatch):
    from primus.backends.megatron.core.fusions import fused_clamped_geglu as mod

    def off_by_one_ulp(x, limit, offset):
        y = mod.FusedClampedQuickGeGLU.apply(x, limit, offset)
        return (y.view(torch.int16) ^ 1).view(torch.bfloat16)

    monkeypatch.setattr(mod, "fused_clamped_quick_geglu", off_by_one_ulp)
    reason = mod.ineligible_reason((4, 2 * 64), torch.bfloat16, torch.device("cuda"), LIMIT, OFFSET)
    assert reason is not None and "not bit-identical" in reason


def test_the_self_check_probes_the_edge_values(monkeypatch):
    """A kernel wrong only where ``x_glu`` is +inf (clamped to a finite value) must fail the
    check, which random inputs alone would not catch."""
    from primus.backends.megatron.core.fusions import fused_clamped_geglu as mod

    def wrong_at_inf(x, limit, offset):
        y = mod.FusedClampedQuickGeGLU.apply(x, limit, offset)
        hit = x[..., : y.shape[-1]] == float("inf")
        return torch.where(hit, (y.view(torch.int16) ^ 1).view(torch.bfloat16), y)

    monkeypatch.setattr(mod, "fused_clamped_quick_geglu", wrong_at_inf)
    reason = mod.ineligible_reason((4, 2 * 64), torch.bfloat16, torch.device("cuda"), LIMIT, OFFSET)
    assert reason is not None and "not bit-identical" in reason


def test_an_empty_first_call_decides_nothing():
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.fusions.fused_clamped_geglu import eligible

    assert not eligible((0, 2 * 64), torch.bfloat16, torch.device("cuda"), LIMIT, OFFSET)
    assert bitwise_check._results == {}


class _Linear(torch.nn.Module):
    """Megatron's linear interface: ``(output, bias)`` with the bias returned, not added."""

    def __init__(self, n_in, n_out, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(0.05 * torch.randn(n_out, n_in, device="cuda", dtype=torch.bfloat16))
        self.bias = (
            torch.nn.Parameter(torch.randn(n_out, device="cuda", dtype=torch.bfloat16)) if bias else None
        )

    def forward(self, x):
        return x @ self.weight.t(), self.bias


def _mlp(hidden, ffn, bias):
    from megatron.core.fusions.fused_bias_geglu import quick_gelu

    torch.manual_seed(0)
    config = SimpleNamespace(
        gated_linear_unit=True,
        activation_func=quick_gelu,
        activation_func_clamp_value=LIMIT,
        glu_linear_offset=OFFSET,
        use_te_activation_func=False,
        bias_activation_fusion=False,
        fp8=None,
        fp4=None,
    )
    return SimpleNamespace(
        config=config,
        activation_func=quick_gelu,
        linear_fc1=_Linear(hidden, 2 * ffn, bias),
        linear_fc2=_Linear(ffn, hidden, False),
    )


@pytest.fixture
def patched_mlp_forward(monkeypatch):
    """``(patched, original)`` MLP forwards; the patch is undone after the test."""
    from megatron.core.transformer.mlp import MLP

    import primus.backends.megatron.patches.fused_clamped_geglu_patches as patches

    original = MLP.forward
    monkeypatch.setattr(MLP, "forward", original)
    monkeypatch.setattr(patches, "_DECISION_LOGGED", False)
    patches._install()
    return MLP.forward, original


@pytest.mark.parametrize("bias", [False, True])
def test_patched_mlp_matches_megatron_bitwise(patched_mlp_forward, bias, require_bitwise):
    from primus.backends.megatron.core.fusions.bitwise_check import (
        outputs_and_grads,
        same_bits,
    )
    from primus.backends.megatron.core.fusions.fused_clamped_geglu import eligible

    patched, original = patched_mlp_forward
    mlp = _mlp(256, 384, bias)
    x = torch.randn(32, 2, 256, device="cuda").to(torch.bfloat16)
    require_bitwise(eligible((32, 2, 768), x.dtype, x.device, LIMIT, OFFSET), "fused_clamped_geglu")

    def run(forward):
        return outputs_and_grads(lambda t: forward(mlp, t)[0], x)

    got, want = run(patched), run(original)
    assert len(got) == len(want) == 2
    assert all(same_bits(a, b) for a, b in zip(got, want))
    assert mlp._primus_fused_clamped_geglu


def test_patched_mlp_runs_megatron_when_not_eligible(patched_mlp_forward, monkeypatch):
    from primus.backends.megatron.core.fusions import fused_clamped_geglu as mod

    patched, _ = patched_mlp_forward
    monkeypatch.setattr(mod, "fused_clamped_quick_geglu", lambda *a: pytest.fail("fused path taken"))
    x = torch.randn(8, 1, 64, device="cuda").to(torch.bfloat16)

    per_token_scale = _mlp(64, 96, False)
    patched(per_token_scale, x, torch.ones(8, 1, device="cuda"))
    te_activation = _mlp(64, 96, False)
    te_activation.config.use_te_activation_func = True
    # stands in for TE's GLU activation module, which halves the width
    te_activation.activation_func = lambda t: t[..., : t.shape[-1] // 2]
    patched(te_activation, x)
    assert not te_activation._primus_fused_clamped_geglu


class _Fc1Float(torch.nn.Module):
    """An fc1 whose output dtype differs from its input's."""

    def __init__(self, fc1):
        super().__init__()
        self.weight = fc1.weight

    def forward(self, x):
        return (x @ self.weight.t()).float(), None


def test_patched_mlp_rejects_an_unexpected_fc1_output(patched_mlp_forward, require_bitwise):
    from primus.backends.megatron.core.fusions.fused_clamped_geglu import eligible

    patched, _ = patched_mlp_forward
    mlp = _mlp(64, 96, False)
    x = torch.randn(8, 1, 64, device="cuda").to(torch.bfloat16)
    require_bitwise(eligible((8, 1, 192), x.dtype, x.device, LIMIT, OFFSET), "fused_clamped_geglu")
    mlp.linear_fc1 = _Fc1Float(mlp.linear_fc1)
    with pytest.raises(RuntimeError, match="linear_fc1 returned torch.float32"):
        patched(mlp, x)
