###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Eager versus CUDA-graph replay for the attention-side fused kernels.

The fused q-norm + RoPE (``q_norm_rope_pad``) and the fused qkv post-processing (``qkv_post``) are run forward and
backward eagerly, then captured in a CUDA graph and replayed three times (the full-iteration graph replays the
same launches every step). Every output and gradient of every replay is byte-equal to the eager result.
"""

import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"
_PKG = "primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common"


def test_attention_kernels_graph_replay_matches_eager(monkeypatch):
    import importlib

    monkeypatch.setenv("PRIMUS_GLUE_ROPE_NOCAST", "1")
    monkeypatch.setenv("PRIMUS_GLUE_RMSNORM", "0")
    Q = importlib.import_module(f"{_PKG}.q_norm_rope")
    QP = importlib.import_module(f"{_PKG}.qkv_post")

    N, H, D, RD, HQ, HKV = 256, 8, 512, 64, 1024, 512
    torch.manual_seed(0)
    xq = torch.randn(1, N, H, D, device=DEV).bfloat16().requires_grad_(True)
    qa = (torch.randn(N, HQ, device=DEV) * 0.7).bfloat16().requires_grad_(True)
    kva = (torch.randn(N, HKV, device=DEV) * 1.3).bfloat16().requires_grad_(True)
    gq = torch.nn.Parameter((1 + 0.1 * torch.randn(HQ, device=DEV)).bfloat16())
    gk = torch.nn.Parameter((1 + 0.1 * torch.randn(HKV, device=DEV)).bfloat16())
    pos = torch.arange(N, device=DEV)
    inv = 1.0 / (160000.0 ** (torch.arange(0, RD, 2, device=DEV).float() / RD))
    d_qnr = torch.randn(1, N, H, D, device=DEV).bfloat16()
    d_q = (torch.randn(N, HQ, device=DEV) * 0.01).bfloat16()
    d_kv = (torch.randn(N, HKV, device=DEV) * 0.02).bfloat16()

    def chain():
        y = Q.q_norm_rope_pad(xq, pos, inv, rotary_dim=RD, eps=1e-6, pad=64)
        oq, okv = QP.qkv_post(qa, kva, gq, gk, pos, inv, eps_q=1e-6, eps_kv=1e-6, rotary_dim=RD)
        grads = torch.autograd.grad([y, oq, okv], [xq, qa, kva, gq, gk], [d_qnr, d_q, d_kv])
        return [y.detach(), oq.detach(), okv.detach(), *grads]

    def snap(res):
        return [r.detach().clone() for r in res]

    # warm up (kernel compilation, allocator) on a side stream, as a graph capture requires
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            chain()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    ref = snap(chain())
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static = chain()
    for rep in range(3):
        g.replay()
        torch.cuda.synchronize()
        for i, (a, b) in enumerate(zip(snap(static), ref)):
            assert bits_equal(a.contiguous(), b.contiguous()), f"replay {rep}: tensor {i} differs from eager"
