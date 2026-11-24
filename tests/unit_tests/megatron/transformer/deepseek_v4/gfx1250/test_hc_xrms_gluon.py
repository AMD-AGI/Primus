###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HC boundary expand + next-HC rms-linear partials (``hc_boundary.expand_rms``): Gluon kernel vs Triton kernel.

* Under the Gluon gate, ``expand_rms`` runs ``hc_xrms_gluon`` whenever its preconditions hold; new / P_RAW / P_SS
  are byte-equal to ``_hc_expand_rms_kernel`` (zero rows, -0, huge values, post 0, bf16 and fp32 mixing weights,
  3-D ``out`` as in the training step).
* Shapes outside the Gluon kernel (N <= 16, M % 64 != 0) run the Triton kernel.
* With the Gluon gate closed (``PRIMUS_GLUON=0``) the Gluon module is not imported and the Triton kernel gives the
  same bytes.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from .helpers import bits_equal, requires_gluon

pytestmark = requires_gluon


def _hb():
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        hc_boundary,
    )

    return hc_boundary


def _inputs(seed, M, D, N, mix_dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x3 = torch.randn(M, 4, D, device="cuda", generator=g).to(torch.bfloat16)
    out = torch.randn(1, M, D, device="cuda", generator=g).to(torch.bfloat16)
    post = (torch.rand(M, 4, device="cuda", generator=g) * 2).to(mix_dtype)
    combt = torch.rand(M, 4, 4, device="cuda", generator=g).to(mix_dtype)
    w = (torch.randn(N, 4 * D, device="cuda", generator=g) * 0.01).to(torch.bfloat16)
    if seed == 1:
        x3[3] = 0
        out[0, 3] = 0
        x3[5, 1] = -0.0
        x3[7, 2, 11] = 3e38
        out[0, 9, 5] = -3e38
        post[13] = 0
        x3[17] = x3[17] * 1e-30
    return x3, out, post, combt, w


def _triton_ref(hb, monkeypatch, args):
    with monkeypatch.context() as m:
        m.setattr(hb, "_HAVE_GLUON", False)
        return hb.expand_rms(*args)


def _assert_same(a, b):
    assert a[3] == b[3]
    for p, q in zip(a[:3], b[:3]):
        assert bits_equal(p, q)


@pytest.mark.parametrize("mix_dtype", [torch.bfloat16, torch.float32], ids=["bf16_mix", "fp32_mix"])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("M,D,N", [(4096, 4096, 24), (256, 1024, 24), (128, 2048, 32)])
def test_gluon_expand_rms_matches_triton(monkeypatch, M, D, N, seed, mix_dtype):
    hb = _hb()
    assert hb._HAVE_GLUON
    args = _inputs(seed, M, D, N, mix_dtype)
    assert hb.xrms_gluon_supported(*args)
    calls = []
    real = hb.expand_rms_gluon
    monkeypatch.setattr(hb, "expand_rms_gluon", lambda *a: calls.append(1) or real(*a))
    got = hb.expand_rms(*args)
    torch.cuda.synchronize()
    assert calls, "the Gluon kernel did not run"
    _assert_same(got, _triton_ref(hb, monkeypatch, args))


@pytest.mark.parametrize("M,N", [(256, 16), (200, 24)], ids=["n16", "m_not_64"])
def test_unsupported_shapes_run_triton(monkeypatch, M, N):
    hb = _hb()
    args = _inputs(0, M, 1024, N)
    assert not hb.xrms_gluon_supported(*args)

    def _fail(*a):
        raise AssertionError("Gluon kernel called outside its preconditions")

    monkeypatch.setattr(hb, "expand_rms_gluon", _fail)
    got = hb.expand_rms(*args)
    torch.cuda.synchronize()
    BN = max(16, 1 << (N - 1).bit_length())
    assert got[1].shape == (4 * 1024 // 1024, M, BN)
    _assert_same(got, _triton_ref(hb, monkeypatch, args))


def test_gluon_gate_closed_falls_back_to_triton(tmp_path):
    hb = _hb()
    args = _inputs(1, 256, 1024, 24)
    ref = hb.expand_rms(*args)
    torch.cuda.synchronize()
    inp = tmp_path / "in.pt"
    res = tmp_path / "out.pt"
    torch.save([a.cpu() for a in args], inp)
    code = textwrap.dedent(
        f"""
        import sys, torch
        from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import hc_boundary as hb
        assert not hb._HAVE_GLUON
        assert not any(m.endswith("hc_xrms_gluon") for m in sys.modules)
        args = [a.cuda() for a in torch.load({str(inp)!r})]
        out = hb.expand_rms(*args)
        torch.save([o.cpu() for o in out[:3]] + [out[3]], {str(res)!r})
        """
    )
    env = {**os.environ, "PRIMUS_GLUON": "0"}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    got = torch.load(res)
    assert got[3] == ref[3]
    for p, q in zip(got[:3], ref[:3]):
        assert bits_equal(p, q.cpu())
