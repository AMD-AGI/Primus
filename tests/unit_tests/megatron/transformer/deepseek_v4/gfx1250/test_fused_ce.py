###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused cross entropy (``PrimusCrossEntropyFn``) over a large vocabulary.

* Loss and logsumexp equal an fp64 reference (FP32 partial statistics), dx equals the fp64 gradient within BF16
  rounding, ignored rows give zero loss and zero gradient.
* ``PRIMUS_MB_SMALLS=1`` (partial buffers not pre-filled) gives the same bytes as the default.
* ``PRIMUS_V4_CE_BWD_GT=1`` (with ``PRIMUS_LM_HEAD_WGRAD_LAYOUT=tn``) also produces d(logits)^T in the same kernel:
  the stashed transpose equals ``dx.t()`` and dx equals the plain backward byte for byte.
"""

import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"
T, V, IGN = 256, 129408, -100


class _Ctx:
    def save_for_backward(self, *t):
        self.saved_tensors = t


def _run(monkeypatch, smalls, gt, logits, y, gout):
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        fused_ce as FC,
    )

    monkeypatch.setenv("PRIMUS_MB_SMALLS", "1" if smalls else "0")
    monkeypatch.setenv("PRIMUS_V4_CE_BWD_GT", "1" if gt else "0")
    monkeypatch.setenv("PRIMUS_LM_HEAD_WGRAD_LAYOUT", "tn")
    c = _Ctx()
    loss = FC.PrimusCrossEntropyFn.forward(c, logits, y, IGN)
    dx = FC.PrimusCrossEntropyFn.backward(c, gout)[0]
    dxt = FC._GT_STASH[dx.data_ptr()][0].clone() if gt else None
    torch.cuda.synchronize()
    return loss.clone(), dx.clone(), dxt, c.saved_tensors[2].clone()


def _data():
    torch.manual_seed(0)
    logits = (torch.randn(1, T, V, device=DEV) * 3).to(torch.bfloat16)
    y = torch.randint(0, V, (1, T), device=DEV, dtype=torch.int64)
    y[0, ::97] = IGN
    gout = torch.full((1, T), 1.0 / T, device=DEV, dtype=torch.float32)
    return logits, y, gout


def test_ce_matches_fp64_and_variants_are_bit_equal(monkeypatch):
    logits, y, gout = _data()
    base = _run(monkeypatch, False, False, logits, y, gout)
    small = _run(monkeypatch, True, False, logits, y, gout)
    gt = _run(monkeypatch, True, True, logits, y, gout)
    assert bits_equal(base[0], small[0]) and bits_equal(base[1], small[1]) and bits_equal(base[3], small[3])
    assert bits_equal(base[1], gt[1]) and bits_equal(base[0], gt[0])
    assert bits_equal(gt[2], gt[1].view(T, V).t().contiguous())

    x2 = logits.view(T, V).double()
    yy = y.view(T)
    valid = yy != IGN
    lse64 = torch.logsumexp(x2, 1)
    loss64 = torch.where(valid, lse64 - x2.gather(1, yy.clamp(min=0)[:, None])[:, 0], torch.zeros_like(lse64))
    p = torch.softmax(x2, 1)
    p[torch.arange(T, device=DEV)[valid], yy[valid]] -= 1.0
    dx64 = p * torch.where(valid, 1.0 / T, 0.0)[:, None]

    def rel(a, b):
        return ((a.double() - b).norm() / b.norm()).item()

    assert rel(base[0].view(T), loss64) < 1e-6
    assert rel(base[3], lse64) < 1e-6
    assert rel(base[1].view(T, V), dx64) < 5e-3  # BF16 gradient
    assert (base[0].view(T)[~valid] == 0).all()
    assert (base[1].view(T, V)[~valid] == 0).all()
