###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""mxfp6_gate_mul_pack_bias: fc2's bias gradient from the GateMul pack's column sums.

The single block's gated MLP + out-projection packs ``gate * dy`` for its backward GEMMs. With the gate on, the same
pack also returns per-tile column sums, and fc2's bias gradient is their sum instead of a separate ``(dy * gate)``
reduction. Checked at the pack the Function calls (``_pack_grad_gate_mul``), for the MXFP6 gradient format and the
A4W4 (FlyDSL packed-scale, stochastic rounding) one:

  * asking for the column sums does not change a single packed byte (the GEMM operands are what they were);
  * the sums equal the fp32 column sums of the product the packer stages, bf16(dy * gate), to fp32 summation order;
  * and they are within bf16 product rounding of the reduction they replace.
"""

import pytest
import torch

from primus.backends.megatron.core.models.diffusion.common import mxfp6_gates
from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import Mxfp6Gates

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("primus_turbo.pytorch")
if not hasattr(torch.ops.primus_turbo, "quantize_mxfp6_gate_mul_impl"):
    pytest.skip("Primus-Turbo without the GateMul packer", allow_module_level=True)

from primus.backends.megatron.core.extensions import (  # noqa: E402
    primus_turbo_mxfp6_local as local,
)
from primus_turbo.pytorch.ops.quantization import set_sr_seed  # noqa: E402

# A single block's linear2 gradient: S tokens x B batch rows of H, the gate one row per batch entry.
S, B, H = 256, 32, 3072

GATES = {
    "mxfp6": dict(),
    "a4w4_flydsl_packed_sr": dict(
        bwd_fp4_dgrad=True, bwd_fp4_wgrad=True, bwd_fp4_sr=True, bwd_fp4_backend="flydsl_packed"
    ),
}


@pytest.fixture(autouse=True)
def _restore_gates():
    yield
    mxfp6_gates.reset()


def _operands(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dy = torch.randn(S * B, H, device="cuda", generator=g).bfloat16()
    gate_full = torch.randn(B, 3 * H, device="cuda", generator=g).bfloat16()
    return dy, gate_full[:, H : 2 * H]  # a chunk view, as AdaLN hands the gate over


def _pack(dy, gate, want_col_sum, b4):
    set_sr_seed(1234)  # the same SR draws for both calls
    out = local._pack_grad_gate_mul(dy, gate, want_col_sum, b4)
    return [t.clone() for t in out]


@pytest.mark.parametrize("name", sorted(GATES))
def test_col_sums_leave_the_pack_unchanged(name):
    g = Mxfp6Gates(**GATES[name])
    g.validate()
    mxfp6_gates.reset(g)
    b4 = local._b4()
    dy, gate = _operands()
    *plain, none = _pack(dy, gate, False, b4)
    *with_sums, partial = _pack(dy, gate, True, b4)
    assert none.numel() == 0 and partial.numel() > 0
    if name == "mxfp6":
        # MXFP6 blobs carry never-written guard padding, so compare what consumes them: a GEMM on the row pack
        # and one on the column pack (as Primus-Turbo's GateMul verification does).
        from primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl import gemm_fp6_impl

        dual = torch.ops.primus_turbo.quantize_mxfp6_dual_impl
        g = torch.Generator(device="cuda").manual_seed(7)
        w = dual(torch.randn(256, H, device="cuda", generator=g).bfloat16(), block_size=32)
        z = dual(torch.randn(256, S * B, device="cuda", generator=g).bfloat16(), block_size=32)
        for p in (plain, with_sums):
            p.append(gemm_fp6_impl(p[0], p[1], w[0], w[1], S * B, 256, H, torch.bfloat16, 4))
            p.append(gemm_fp6_impl(p[2], p[3], z[0], z[1], H, 256, S * B, torch.bfloat16, 4))
        plain, with_sums = plain[4:], with_sums[4:]
    for a, b in zip(plain, with_sums):
        assert torch.equal(a, b), "the column sums must not change the packed operands"


@pytest.mark.parametrize("name", sorted(GATES))
def test_col_sums_are_the_bias_gradient(name):
    g = Mxfp6Gates(**GATES[name])
    g.validate()
    mxfp6_gates.reset(g)
    dy, gate = _operands(1)
    *_, partial = _pack(dy, gate, True, local._b4())
    got = partial.sum(0)
    staged = (dy.view(S, B, H).float() * gate.float()).bfloat16().float()  # what the packer stages
    torch.testing.assert_close(got, staged.sum((0, 1)), rtol=1e-5, atol=1e-3)
    # The reduction it replaces, (dy * gate).sum((0, 1)) on the unrounded product: bf16 product rounding apart.
    replaced = (dy.view(S, B, H).float() * gate.float()).sum((0, 1))
    torch.testing.assert_close(got, replaced, rtol=1e-2, atol=0.5)


def test_gate_is_read_from_config():
    class Cfg:
        mxfp6_gate_mul_pack = True
        mxfp6_gate_mul_pack_bias = True

    resolved = mxfp6_gates.configure(Cfg())
    assert resolved.gate_mul_pack and resolved.gate_mul_pack_bias
