###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused q/kv post-processing (``qkv_post``, ``PRIMUS_V4_QKV_POST_FUSED``) against the unfused chain
``rmsnorm_g90(q_a)``, ``rmsnorm_g90(kv) -> RoPE(kv)`` followed by the concat op's FP8 ``_quant_cat`` of
``[dq_a | dkv]``. Outputs, input gradients, the quantised dY bytes (row / transposed data and scales) and the
norm-weight gradients (``.grad``, or ``main_grad`` when ``PRIMUS_GLUE_RMSNORM=1``) are compared byte for byte.
"""

import pytest
import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"
F8 = torch.float8_e4m3fn


def _mkparam(w, glue):
    p = torch.nn.Parameter(w.clone())
    if glue:
        p.main_grad = torch.zeros(w.shape, device=DEV, dtype=torch.bfloat16)
        p.grad_added_to_main_grad = False
    return p


@pytest.mark.parametrize(
    "N,seed,q_first,glue",
    [(128, 0, False, False), (512, 1, False, True), (512, 2, True, True), (512, 3, True, False)],
)
def test_qkv_post_matches_unfused_chain(monkeypatch, N, seed, q_first, glue):
    pytest.importorskip("primus_turbo")
    from primus_turbo.pytorch.ops import utils as tu
    from primus_turbo.pytorch.ops.gemm_fp8_concat import _quant_cat

    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        qkv_post as QP,
    )
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.rmsnorm_g90 import (
        rmsnorm_g90,
    )
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.rope_interleaved_partial import (
        apply_rope_from_positions,
    )

    HQ, HKV, RD = 1024, 512, 64
    monkeypatch.setenv("PRIMUS_GLUE_ROPE_NOCAST", "1")
    monkeypatch.setenv("PRIMUS_GLUE_RMSNORM", "1" if glue else "0")
    tu.set_main_grad_first_write_policy((lambda p: True) if glue else None)
    try:
        torch.manual_seed(seed)
        qa = (torch.randn(N, HQ, device=DEV) * 0.7).to(torch.bfloat16)
        kva = (torch.randn(N, HKV, device=DEV) * 1.3).to(torch.bfloat16)
        gq0 = (1 + 0.1 * torch.randn(HQ, device=DEV)).to(torch.bfloat16)
        gk0 = (1 + 0.1 * torch.randn(HKV, device=DEV)).to(torch.bfloat16)
        dyq = (torch.randn(N, HQ, device=DEV) * 0.01).to(torch.bfloat16)
        dyk = (torch.randn(N, HKV, device=DEV) * 0.02).to(torch.bfloat16)
        pos = torch.arange(N, device=DEV) % 4096
        inv = 1.0 / (160000.0 ** (torch.arange(0, RD, 2, device=DEV).float() / RD))

        # unfused reference chain
        gq, gk = _mkparam(gq0, glue), _mkparam(gk0, glue)
        x1 = qa.clone().requires_grad_(True)
        x2 = kva.clone().requires_grad_(True)
        yq = rmsnorm_g90(x1, gq, 1e-6)
        yk = apply_rope_from_positions(
            rmsnorm_g90(x2, gk, 1e-6).view(N, 1, HKV), pos, inv, rotary_dim=RD
        ).view(N, HKV)
        torch.autograd.backward([yq, yk], [dyq, dyk])
        ref_q = _quant_cat([x1.grad, x2.grad] if q_first else [x2.grad, x1.grad], F8)
        ref_dg = (gq.main_grad, gk.main_grad) if glue else (gq.grad, gk.grad)

        # fused op
        gq2, gk2 = _mkparam(gq0, glue), _mkparam(gk0, glue)
        y1 = qa.clone().requires_grad_(True)
        y2 = kva.clone().requires_grad_(True)
        oq, ok_ = QP.qkv_post(
            y1, y2, gq2, gk2, pos, inv, eps_q=1e-6, eps_kv=1e-6, rotary_dim=RD, q_first=q_first
        )
        torch.autograd.backward([oq, ok_], [dyq, dyk])
        torch.cuda.synchronize()
        got_q = _quant_cat([y1.grad, y2.grad] if q_first else [y2.grad, y1.grad], F8)
        dg = (gq2.main_grad, gk2.main_grad) if glue else (gq2.grad, gk2.grad)
    finally:
        tu.set_main_grad_first_write_policy(None)

    checks = {
        "yq": (yq, oq),
        "ykv": (yk, ok_),
        "dq_a": (x1.grad, y1.grad),
        "dkv": (x2.grad, y2.grad),
        "dgq": (ref_dg[0], dg[0]),
        "dgkv": (ref_dg[1], dg[1]),
    }
    checks.update({n: (a, b) for n, a, b in zip(("q_row", "e_row", "q_t", "e_t"), ref_q, got_q)})
    bad = [n for n, (a, b) in checks.items() if not bits_equal(a.contiguous(), b.contiguous())]
    assert not bad, f"fused qkv_post differs from the unfused chain in: {bad}"
