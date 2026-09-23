###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for PrimusTurboRMSNorm's handling of zero-centered gamma.

Regression guard. ``PrimusTurboRMSNorm`` forwards ``zero_centered_gamma`` to its
TE base class -- which initialises the weight to *zeros* -- but computes the
norm with a turbo kernel that takes the scale verbatim. Without adding the 1
back in ``forward``, every norm output is ``normalize(x) * 0 == 0``: the model
still trains "successfully", with the loss pinned at the uniform value and
gradients three orders of magnitude too small.

These tests exercise the class the runtime actually builds, not the upstream
``TENorm`` it replaces -- testing the upstream class is what let the bug
through in the first place.
"""

import pytest
import torch

pytest.importorskip("transformer_engine.pytorch")
pytest.importorskip("primus_turbo")

from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboRMSNorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="turbo RMSNorm needs a GPU")

DIM = 64
EPS = 1.0e-6


def _reference(x, weight, *, zero_centered):
    """RMSNorm in fp32, the way the TE kernel and Gemma/MiniMax-M3 define it."""
    x_fp32 = x.float()
    normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + EPS)
    scale = (1.0 + weight.float()) if zero_centered else weight.float()
    return (normed * scale).type_as(x)


def _norm(zero_centered):
    norm = PrimusTurboRMSNorm(DIM, eps=EPS, zero_centered_gamma=zero_centered).cuda()
    assert norm.zero_centered_gamma is zero_centered
    return norm


def test_zero_centered_gamma_inits_weight_at_zero():
    """The init half of the contract: TE centres gamma on zero."""
    assert _norm(True).weight.detach().abs().sum().item() == 0.0
    # The default path keeps the usual ones init, so the two cannot share a forward.
    assert _norm(False).weight.detach().eq(1.0).all().item()


def test_zero_centered_gamma_is_not_the_identity_scale():
    """The bug in one assertion: at init, a zero-centred gamma must still pass
    the normalised activations through, not multiply them by zero."""
    norm = _norm(True)
    out = norm(torch.randn(8, DIM, device="cuda"))

    assert out.abs().sum().item() > 0.0


@pytest.mark.parametrize("zero_centered", [True, False])
def test_matches_the_reference_formula(zero_centered):
    norm = _norm(zero_centered)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(DIM, device="cuda") * 0.1)
    x = torch.randn(8, DIM, device="cuda")

    torch.testing.assert_close(
        norm(x), _reference(x, norm.weight, zero_centered=zero_centered), rtol=1e-5, atol=1e-5
    )


def test_matches_transformer_engine():
    """Parity with the TE RMSNorm the turbo class stands in for -- swapping the
    implementation must not change what the model computes."""
    import transformer_engine.pytorch as te

    turbo = _norm(True)
    reference = te.RMSNorm(DIM, eps=EPS, zero_centered_gamma=True).cuda()
    with torch.no_grad():
        gamma = torch.randn(DIM, device="cuda") * 0.1
        turbo.weight.copy_(gamma)
        reference.weight.copy_(gamma)
    x = torch.randn(8, DIM, device="cuda")

    torch.testing.assert_close(turbo(x), reference(x), rtol=1e-5, atol=1e-5)
