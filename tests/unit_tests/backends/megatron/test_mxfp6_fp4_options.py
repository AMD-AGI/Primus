###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The MXFP4 quantization options (mxfp6_fp4_*) reach the pack formats they are meant for, and no other.

The packers take the options as fmt bits 16-24 (Primus-Turbo ``fp4_options``); these tests decode the formats the
MXFP6 Functions would pass and check: defaults leave every format untouched; each option lands on the operand class
and GEMM it names; FP6 directions never carry one; and the two operands of every A4W4 GEMM always carry the same
Hadamard, whatever the settings.
"""

import itertools

import pytest
import torch

from primus.backends.megatron.core.models.diffusion.common import mxfp6_gates
from primus.backends.megatron.core.models.diffusion.common.mxfp6_gates import Mxfp6Gates

_mx = pytest.importorskip("primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack")
if not hasattr(_mx, "fp4_options"):
    pytest.skip("Primus-Turbo without fp4_options", allow_module_level=True)

from primus.backends.megatron.core.extensions import (  # noqa: E402
    primus_turbo_mxfp6_local as local,
)

ROUND = {0: "rceil", 1: "m0", 2: "m1", 3: "m2"}
HAD = {0: "h32", 1: "none", 2: "h16"}
# A tilescale A4W4 backward at Flux single-block shapes: M tokens, a 3072 -> 12288 fc1.
M, N_OUT, K_IN = 8192, 12288, 3072


def _decode(fmt):
    """(row (round, had) or None, col (round, had) or None, tile2d): None for an FP6 direction."""
    r4, c4 = _mx.mx_fp4_dirs(fmt)
    row = (ROUND[(fmt >> 16) & 3], HAD[(fmt >> 20) & 3]) if r4 else None
    col = (ROUND[(fmt >> 18) & 3], HAD[(fmt >> 22) & 3]) if c4 else None
    if not r4:
        assert (fmt >> 16) & 3 == 0 and (fmt >> 20) & 3 == 0, f"options on an FP6 row direction: {fmt:#x}"
    if not c4:
        assert (fmt >> 18) & 3 == 0 and (fmt >> 22) & 3 == 0, f"options on an FP6 column direction: {fmt:#x}"
    return row, col, bool(fmt >> 24 & 1)


def _gates(**kw):
    base = dict(
        bwd_fp4_dgrad=True, bwd_fp4_wgrad=True, gemm_layout="tilescale", fwd_fp4_single_fc1=True
    )
    base.update(kw)
    g = Mxfp6Gates(**base)
    g.validate()
    mxfp6_gates.reset(g)
    return g


@pytest.fixture(autouse=True)
def _reset_gates():
    yield
    mxfp6_gates.reset()


def _formats(sr=False):
    b4 = local._b4()
    if sr:
        b4 = (b4[0], b4[1], True, b4[3])
    return {
        "grad": local._grad_fmt(b4),
        "act": local._act_fmt((M, K_IN), N_OUT),
        "weight": local._weight_fmt((N_OUT, K_IN), M),
        "fwd_act": local._fwd_fp4_fmt("act", M, N_OUT, K_IN),
        "fwd_weight": local._fwd_fp4_fmt("weight", M, N_OUT, K_IN),
    }


@pytest.mark.parametrize("layout", ["blob", "tilescale"])
def test_defaults_leave_every_format_untouched(layout):
    _gates(gemm_layout=layout, fwd_fp4_single_fc1=layout == "tilescale")
    for name, fmt in _formats().items():
        assert fmt >> 16 == 0, (name, hex(fmt))


def test_all_options_land_per_operand_class():
    """Every option at once: m0 gradients and FP4-forward operands, m2 FP4 copies of MXFP6-forward
    operands; no Hadamard on forward / dgrad, H16 on wgrad; 2-D weights."""
    _gates(
        fp4_scale_rounding_grad="m0",
        fp4_scale_rounding_actw_hp="m2",
        fp4_scale_rounding_actw_fp4fwd="m0",
        fp4_hadamard_fwd="none",
        fp4_hadamard_dgrad="none",
        fp4_hadamard_wgrad="h16",
        fp4_weight_2d=True,
    )
    f = {k: _decode(v) for k, v in _formats().items()}
    assert f["grad"] == (("m0", "none"), ("m0", "h16"), False)
    assert f["act"] == (None, ("m2", "h16"), False)  # FP6 forward rows
    assert f["weight"] == (None, ("m2", "none"), True)
    assert f["fwd_act"] == (("m0", "none"), ("m0", "h16"), False)
    assert f["fwd_weight"] == (("m0", "none"), ("m0", "none"), True)


@pytest.mark.parametrize("fwd,dgrad,wgrad", list(itertools.product(["h32", "h16", "none"], repeat=3)))
def test_every_gemm_pairs_its_hadamard(fwd, dgrad, wgrad):
    """dgrad = gradient rows x weight columns, wgrad = gradient columns x activation columns, forward (MXFP4
    layers) = activation rows x weight rows: each pair must agree for every setting."""
    _gates(fp4_hadamard_fwd=fwd, fp4_hadamard_dgrad=dgrad, fp4_hadamard_wgrad=wgrad)
    f = {k: _decode(v) for k, v in _formats().items()}
    assert f["grad"][0][1] == f["weight"][1][1] == f["fwd_weight"][1][1] == dgrad
    assert f["grad"][1][1] == f["act"][1][1] == f["fwd_act"][1][1] == wgrad
    assert f["fwd_act"][0][1] == f["fwd_weight"][0][1] == fwd


def test_sr_gradient_keeps_its_options():
    _gates(fp4_scale_rounding_grad="m0", fp4_hadamard_wgrad="none")
    g = _formats(sr=True)["grad"]
    assert _mx.mx_fmt_base(g) & _mx.MX_FMT_FLY_SR
    assert _decode(g)[:2] == (("m0", "h32"), ("m0", "none"))


def test_a6w6_blocks_stay_mxfp6():
    """Blocks that keep the A6W6 backward (bwd_fp4 gates off around them) pack fmt 0, with no options."""
    g = _gates(fp4_scale_rounding_grad="m0", fp4_hadamard_dgrad="none")
    g.bwd_fp4_dgrad = g.bwd_fp4_wgrad = False
    assert local._grad_fmt(local._b4()) == 0
    assert local._act_fmt((M, K_IN), N_OUT) == 0 and local._weight_fmt((N_OUT, K_IN), M) == 0


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError, match="scale_rounding_grad"):
        _gates(fp4_scale_rounding_grad="rne")
    with pytest.raises(ValueError, match="hadamard_wgrad"):
        _gates(fp4_hadamard_wgrad="h8")
    with pytest.raises(ValueError, match="weight_2d"):
        _gates(fp4_weight_2d=True, fp4_hadamard_fwd="none")  # dgrad still H32
    with pytest.raises(ValueError, match="weight_2d"):
        _gates(fp4_weight_2d=True, fp4_hadamard_dgrad="none")  # an MXFP4 forward still H32
    _gates(fp4_weight_2d=True, fp4_hadamard_dgrad="none", fwd_fp4_single_fc1=False)


def _mlp_run(x, w1, b1, w2, bf16_fc1, grad):
    xr = x.detach().clone().requires_grad_()
    out, y1 = local.MXFP6MLPFunction.apply(xr, w1, b1, w2, False, True, False, False, False, bf16_fc1)[:2]
    out.backward(grad)
    return out.detach(), y1.detach(), xr.grad, w1.grad.clone(), w2.grad.clone()


def _ref_mlp(x, w1, b1, w2, grad):
    xr, w1r, b1r, w2r = (t.detach().float().clone().requires_grad_() for t in (x, w1, b1, w2))
    out = torch.nn.functional.gelu(xr @ w1r.t() + b1r, approximate="tanh") @ w2r.t()
    out.backward(grad.float())
    return out.detach(), xr.grad, w1r.grad, w2r.grad


def _snr(got, ref):
    return 10 * torch.log10(ref.pow(2).sum() / (got.float() - ref).pow(2).sum()).item()


@pytest.mark.parametrize("layout", ["tilescale", None])
def test_bf16_joint_img_fc1(layout):
    """fc1 forward in bf16: its pre-activation is the bf16 GEMM exactly; the backward still runs (A4W4 or A6W6) on
    column packs of the bf16 operands, and every output tracks an fp32 reference at least as well as the MXFP6
    fc1 does."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if layout:
        _gates(gemm_layout=layout, fwd_fp4_single_fc1=False, fwd_bf16_joint_img_fc1=True)
    else:
        mxfp6_gates.reset(Mxfp6Gates(fwd_bf16_joint_img_fc1=True))
    m, k, f, h = 1024, 3072, 12288, 3072
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(m, k, device="cuda", generator=g).bfloat16()
    w1 = (torch.randn(f, k, device="cuda", generator=g) * k**-0.5).bfloat16().requires_grad_()
    b1 = (torch.randn(f, device="cuda", generator=g) * 0.1).bfloat16().requires_grad_()
    w2 = (torch.randn(h, f, device="cuda", generator=g) * f**-0.5).bfloat16().requires_grad_()
    grad = torch.randn(m, h, device="cuda", generator=g).bfloat16()
    ref = _ref_mlp(x, w1, b1, w2, grad)
    snrs = {}
    for bf16 in (True, False):
        w1.grad = w2.grad = b1.grad = None
        out, y1, gx, gw1, gw2 = _mlp_run(x, w1, b1, w2, bf16, grad)
        if bf16:
            assert torch.equal(y1, torch.mm(x, w1.detach().t()))
        snrs[bf16] = [_snr(a, b) for a, b in zip((out, gx, gw1, gw2), (ref[0], ref[1], ref[2], ref[3]))]
    assert snrs[True][0] > snrs[False][0], snrs  # the forward gains from the bf16 fc1
    for got, base in zip(snrs[True][1:], snrs[False][1:]):
        assert got > base - 1.0, snrs  # the backward is the same quantized backward


def test_sr_actw_marks_only_activation_and_weight_columns():
    """fp4_sr_actw: column-only SR on every activation / weight pack (backward copies), never on a gradient pack (its
    own SR is bwd_fp4_sr), and never on an FP4 forward row."""
    _gates(fp4_sr_actw=True)
    f = _formats()
    for name in ("act", "weight", "fwd_act", "fwd_weight"):
        assert f[name] & _mx.MX_FMT_FP4_COL_SR, name
        assert not _mx.mx_fmt_base(f[name]) & _mx.MX_FMT_FLY_SR, name  # rows stay round-to-nearest
    assert not f["grad"] & _mx.MX_FMT_FP4_COL_SR
    with pytest.raises(ValueError, match="sr_actw"):
        _gates(fp4_sr_actw=True, gemm_layout="blob", fwd_fp4_single_fc1=False)


def test_retired_keys_raise():
    """A retired gate key is an error that names its replacement, never a silent no-op."""
    import types

    for key in mxfp6_gates.RETIRED_KEYS:
        with pytest.raises(ValueError, match=key):
            mxfp6_gates.check_retired(types.SimpleNamespace(**{key: "x"}))
    mxfp6_gates.check_retired(types.SimpleNamespace(mxfp6_gemm_layout="tilescale"))
