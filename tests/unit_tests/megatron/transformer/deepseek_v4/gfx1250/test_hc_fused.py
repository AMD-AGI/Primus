###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused HyperConnection (HC) application: pre-mix + folded RMSNorm + post, forward and backward, bytewise.

Chain of two HC applications (the second consumes the expanded stream of the first, the first post feeding the
second's weight layout), checked on every output and every gradient: x, and the main_grad of W / scale / base / gamma
(the DDP hook is emulated for the reference: first-write copy or BF16 add when accumulating).

* Tile / cast knobs of the fused kernels (``PRIMUS_HC_EXPAND_TILE``, ``PRIMUS_HC_POST_TILE``, ``PRIMUS_HC_NOCAST``)
  do not change a single bit of a single-application run.
* The boundary fusions (``PRIMUS_HC_GRAD_DIRECT``, ``PRIMUS_HC_XRMS``, ``PRIMUS_HC_B3X`` with its config
  ``PRIMUS_HC_B3X_CFG``) equal the unfused chain, for first-write and accumulating main_grad.
* Split-K weight gradient (``PRIMUS_SPLITK_HC_TS`` / ``_NMASK``): outputs and all gradients except dW are
  byte-equal, dW differs only by FP32 summation order; repeated runs are bit-stable.
* ``PRIMUS_HC_FUSED_GLUON=0`` (Triton B3 / post fallback) stays within BF16 rounding of the Gluon path (outputs and
  gradients; a handful of elements differ by one unit in the last place).

``PRIMUS_HC_FUSED=1`` selects the fused HC path; the reference runs here switch the boundary fusions off.
"""

import pytest
import torch

from .helpers import bits_equal, requires_gluon

pytestmark = requires_gluon

K = 4
M, D = 512, 1024
BASE_ENV = {"PRIMUS_HC_FUSED": "1", "PRIMUS_HC_EXPAND_TILE": "1,1024,8", "PRIMUS_HC_POST_TILE": "1,1",
            "PRIMUS_HC_NOCAST": "1"}  # fmt: skip
SWITCHES = ["PRIMUS_HC_GRAD_DIRECT", "PRIMUS_HC_XRMS", "PRIMUS_HC_B3X"]
B3X_CFG = "bc=64,w=8,bt=32,buf=1"


def _hf():
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        hc_fused,
    )

    return hc_fused


def _set(monkeypatch, env):
    for k in (
        "PRIMUS_HC_GRAD_DIRECT",
        "PRIMUS_HC_XRMS",
        "PRIMUS_HC_B3X",
        "PRIMUS_HC_B3X_CFG",
        "PRIMUS_SPLITK_HC_TS",
        "PRIMUS_SPLITK_HC_NMASK",
        "PRIMUS_HC_FUSED_GLUON",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in {**BASE_ENV, **env}.items():
        monkeypatch.setenv(k, v)


def _make_chain(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def P(t):
        p = torch.nn.Parameter(t)
        p.grad_added_to_main_grad = False
        p.main_grad = torch.zeros_like(t)
        return p

    mix = []
    for _ in range(2):
        w = (torch.randn(24, K * D, device="cuda", generator=g) / (K * D) ** 0.5).to(torch.bfloat16)
        scale = (1 + 0.1 * torch.randn(3, device="cuda", generator=g)).to(torch.bfloat16)
        base = (0.1 * torch.randn(24, device="cuda", generator=g)).to(torch.bfloat16)
        gamma = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
        mix.append([w, scale, base, gamma])
    x = torch.randn(1, M, K, D, device="cuda", dtype=torch.bfloat16, generator=g)
    r = [torch.randn(1, M, D, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(2)]
    r_new = torch.randn(1, M, K, D, device="cuda", dtype=torch.float32, generator=g)
    mg0 = [[torch.randn_like(t) for t in m] for m in mix]
    return x, mix, r, r_new, mg0, P


def _run_chain(inp, acc):
    from primus_turbo.pytorch.ops.utils import set_main_grad_first_write_policy

    hf = _hf()
    x0, mix, r, r_new, mg0, P = inp
    set_main_grad_first_write_policy(lambda p: not acc)
    params = []
    for mi, m in enumerate(mix):
        ps = [P(t.clone()) for t in m]
        for p, m0 in zip(ps, mg0[mi]):
            p.main_grad.copy_(m0 if acc else torch.zeros_like(m0))
        params.append(ps)
    x = x0.clone().requires_grad_(True)
    outs = []
    h = x
    for li in range(2):
        w, s, b, gm = params[li]
        normed, post, combt, link = hf.hc_fused_pre(h, w, s, b, 1e-6, 20, norm_weight=gm, norm_eps=1e-6)
        out = normed * r[li]
        nw = params[1][0] if li == 0 else None
        h = hf.hc_fused_post(h, out, post, combt, link, nw)
        outs += [normed.detach(), post.detach(), combt.detach(), h.detach()]
    (h.float() * r_new).sum().backward()
    torch.cuda.synchronize()
    grads = [x.grad]
    for ps in params:
        for p in ps:
            if p.grad is not None and not p.grad_added_to_main_grad:  # emulate the DDP hook
                p.main_grad.add_(p.grad) if acc else p.main_grad.copy_(p.grad)
            grads.append(p.main_grad.clone())
    return outs, grads


GNAMES = ["dx", "dwA", "dsA", "dbA", "dgA", "dwB", "dsB", "dbB", "dgB"]
CONFIGS = {
    "grad_direct": {"PRIMUS_HC_GRAD_DIRECT": "1"},
    "xrms": {"PRIMUS_HC_XRMS": "1"},
    "b3x": {"PRIMUS_HC_B3X": "1", "PRIMUS_HC_B3X_CFG": B3X_CFG},
    "all": {
        "PRIMUS_HC_GRAD_DIRECT": "1",
        "PRIMUS_HC_XRMS": "1",
        "PRIMUS_HC_B3X": "1",
        "PRIMUS_HC_B3X_CFG": B3X_CFG,
    },
}


@pytest.mark.parametrize("acc", [False, True], ids=["first_write", "accumulate"])
@pytest.mark.parametrize("cfg", sorted(CONFIGS))
def test_boundary_fusions_equal_unfused_chain(monkeypatch, cfg, acc):
    inp = _make_chain(0)
    _set(monkeypatch, {})
    ref_o, ref_g = _run_chain(inp, acc)
    _set(monkeypatch, CONFIGS[cfg])
    o, g = _run_chain(inp, acc)
    assert all(bits_equal(a, b) for a, b in zip(o, ref_o)), "forward outputs"
    bad = [n for n, a, b in zip(GNAMES, g, ref_g) if not bits_equal(a, b)]
    assert not bad, bad


@pytest.mark.parametrize("acc", [False, True], ids=["first_write", "accumulate"])
def test_gluon_off_fallback_stays_close_to_gluon_path(monkeypatch, acc):
    inp = _make_chain(1)
    _set(monkeypatch, CONFIGS["all"])
    ref_o, ref_g = _run_chain(inp, acc)
    _set(monkeypatch, {**CONFIGS["all"], "PRIMUS_HC_FUSED_GLUON": "0"})
    o, g = _run_chain(inp, acc)
    names = ["normA", "postA", "combtA", "x1", "normB", "postB", "combtB", "x2"] + GNAMES
    for n, a, b in zip(names, o + g, ref_o + ref_g):
        assert torch.isfinite(a.float()).all(), n
        err = ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()
        assert err < 5e-3, (n, err)


@pytest.mark.parametrize("acc", [False, True], ids=["first_write", "accumulate"])
def test_splitk_weight_grad_differs_only_in_summation_order(monkeypatch, acc):
    inp = _make_chain(0)
    cfg = {**CONFIGS["all"], "PRIMUS_HC_B3X_CFG": B3X_CFG}
    _set(monkeypatch, {**cfg, "PRIMUS_HC_B3X_CFG": "bc=64,w=8,bt=32"})
    ref_o, ref_g = _run_chain(inp, acc)
    _set(monkeypatch, {**cfg, "PRIMUS_SPLITK_HC_TS": "12", "PRIMUS_SPLITK_HC_NMASK": "1"})
    o1, g1 = _run_chain(inp, acc)
    o2, g2 = _run_chain(inp, acc)
    assert all(bits_equal(a, b) for a, b in zip(o1 + g1, o2 + g2)), "not bit-stable run to run"
    assert all(bits_equal(a, b) for a, b in zip(o1, ref_o))
    for n, a, b in zip(GNAMES, g1, ref_g):
        if n.startswith("dw"):
            assert torch.isfinite(a.float()).all()
            assert (a.float() - b.float()).abs().max() <= 1e-2 * max(b.float().abs().max().item(), 1e-6), n
        else:
            assert bits_equal(a, b), n


# ---- single application, kernel tile / cast knobs ----


def _run_single(seed, M=256, D=4096):
    hf = _hf()
    g = torch.Generator(device="cuda").manual_seed(seed)
    x0 = torch.randn(1, M, K, D, device="cuda", dtype=torch.bfloat16, generator=g)
    w0 = (torch.randn(24, K * D, device="cuda", generator=g) / (K * D) ** 0.5).to(torch.bfloat16)
    s0 = (1 + 0.1 * torch.randn(3, device="cuda", generator=g)).to(torch.bfloat16)
    b0 = (0.1 * torch.randn(24, device="cuda", generator=g)).to(torch.bfloat16)
    g0 = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(torch.bfloat16)
    r_out = torch.randn(1, M, D, device="cuda", dtype=torch.bfloat16, generator=g)
    r_new = torch.randn(1, M, K, D, device="cuda", dtype=torch.float32, generator=g)
    x, w, s, b, gm = (t.clone().requires_grad_(True) for t in (x0, w0, s0, b0, g0))
    normed, post, combt, link = hf.hc_fused_pre(x, w, s, b, 1e-6, 20, norm_weight=gm, norm_eps=1e-6)
    new = hf.hc_fused_post(x, normed * r_out, post, combt, link, None)
    (new.float() * r_new).sum().backward()
    torch.cuda.synchronize()
    return [
        normed.detach(),
        post.detach(),
        combt.detach(),
        new.detach(),
        x.grad,
        w.grad,
        s.grad,
        b.grad,
        gm.grad,
    ]


@pytest.mark.parametrize(
    "knob",
    [{"PRIMUS_HC_EXPAND_TILE": "1,2048,8"}, {"PRIMUS_HC_POST_TILE": "1,1"}, {"PRIMUS_HC_NOCAST": "1"}],
    ids=["expand_tile", "post_tile", "nocast"],
)
def test_tile_knobs_do_not_change_bits(monkeypatch, knob):
    for k in ("PRIMUS_HC_EXPAND_TILE", "PRIMUS_HC_POST_TILE", "PRIMUS_HC_NOCAST"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PRIMUS_HC_FUSED", "1")
    ref = _run_single(0)
    for k, v in knob.items():
        monkeypatch.setenv(k, v)
    res = _run_single(0)
    assert all(bits_equal(a, b) for a, b in zip(res, ref))


@pytest.mark.parametrize("acc", [False, True], ids=["first_write", "accumulate"])
def test_hcfuse_post_and_batched_b3x_stay_close_and_stable(monkeypatch, acc):
    """HC launch merges (``PRIMUS_HCFUSE_POST=2``, B3X ``bat=1``, folded-norm backward 2 programs per CU, tail
    block width 512) may change summation order: finite, run-to-run bit-stable, close to the unfused chain."""
    inp = _make_chain(0)
    _set(monkeypatch, {})
    ref_o, ref_g = _run_chain(inp, acc)
    cfg = {
        "PRIMUS_HC_GRAD_DIRECT": "1", "PRIMUS_HC_XRMS": "1", "PRIMUS_HC_B3X": "1",
        "PRIMUS_HC_B3X_CFG": "bc=64,w=8,bt=32,pf2=1,bat=1", "PRIMUS_SPLITK_HC_TAILBW": "512",
        "PRIMUS_HCFUSE_FOLD_CFG": "bwd_ppc=2", "PRIMUS_HCFUSE_POST": "2",
    }  # fmt: skip
    _set(monkeypatch, cfg)
    o1, g1 = _run_chain(inp, acc)
    o2, g2 = _run_chain(inp, acc)
    assert all(bits_equal(a, b) for a, b in zip(o1 + g1, o2 + g2)), "not bit-stable run to run"
    for n, a, b in list(zip(range(len(o1)), o1, ref_o)) + list(zip(GNAMES, g1, ref_g)):
        assert torch.isfinite(a.float()).all(), n
        err = ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()
        assert err < 2e-2, (n, err)
