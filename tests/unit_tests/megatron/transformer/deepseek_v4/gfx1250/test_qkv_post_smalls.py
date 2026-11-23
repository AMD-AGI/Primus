###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused q / kv post-processing (``qkv_post``): backward with row-strided incoming gradients.

With ``PRIMUS_MB_SMALLS=1`` the backward reads gradients that are views of a wider buffer in place instead of
copying them contiguous first. dx and dgamma must be byte-equal to the contiguous-copy path, for strided and
non-strided inputs (``PRIMUS_V4_QKV_POST_FUSED=1`` selects the fused kernel).
"""

import pytest
import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"
N = 512


def _run(monkeypatch, smalls, seed, pad_q, pad_kv, HQ=1024, HKV=512, RD=64):
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        qkv_post as QP,
    )

    monkeypatch.setenv("PRIMUS_V4_QKV_POST_FUSED", "1")
    monkeypatch.setenv("PRIMUS_MB_SMALLS", "1" if smalls else "0")
    monkeypatch.setenv("PRIMUS_GLUE_RMSNORM", "0")
    torch.manual_seed(seed)
    qa = (torch.randn(N, HQ, device=DEV) * 0.7).to(torch.bfloat16).requires_grad_(True)
    kva = (torch.randn(N, HKV, device=DEV) * 1.3).to(torch.bfloat16).requires_grad_(True)
    gq = torch.nn.Parameter((1 + 0.1 * torch.randn(HQ, device=DEV)).to(torch.bfloat16))
    gk = torch.nn.Parameter((1 + 0.1 * torch.randn(HKV, device=DEV)).to(torch.bfloat16))
    bq = (torch.randn(N, HQ + pad_q, device=DEV) * 0.01).to(torch.bfloat16)
    bk = (torch.randn(N, HKV + pad_kv, device=DEV) * 0.02).to(torch.bfloat16)
    dyq, dyk = bq[:, :HQ], bk[:, :HKV]  # row-strided views when the pad is non-zero
    pos = torch.arange(N, device=DEV) % 4096
    inv = 1.0 / (160000.0 ** (torch.arange(0, RD, 2, device=DEV).float() / RD))
    oq, ok_ = QP.qkv_post(qa, kva, gq, gk, pos, inv, eps_q=1e-6, eps_kv=1e-6, rotary_dim=RD, q_first=True)
    torch.autograd.backward([oq, ok_], [dyq, dyk])
    return [t.detach().clone() for t in (qa.grad, kva.grad, gq.grad, gk.grad)]


@pytest.mark.parametrize("pad_q, pad_kv", [(0, 64), (128, 64), (128, 0), (0, 0)])
@pytest.mark.parametrize("seed", [0, 1])
def test_strided_incoming_grads_read_in_place(monkeypatch, seed, pad_q, pad_kv):
    a = _run(monkeypatch, False, seed, pad_q, pad_kv)
    b = _run(monkeypatch, True, seed, pad_q, pad_kv)
    assert all(bits_equal(x, y) for x, y in zip(a, b))
