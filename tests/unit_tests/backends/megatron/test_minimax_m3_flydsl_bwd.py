###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The flydsl MSA backward against autograd through the eager backend.

The reference is autograd in fp32. The gate is a global SNR of 40 dB, the bar
Primus-Turbo sets for bf16 attention gradients, plus the worst element against
the eager backend's own bf16 error: the kernels round P and dS to bf16 for the
MFMAs, as eager rounds P, and one wrong row barely moves a global SNR.
"""

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("flydsl")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="the flydsl MSA kernels target gfx950",
)

BLOCK, D = 128, 128


def _table(B, Hkv, S, topk, gen):
    n_blocks = -(-S // BLOCK)
    scores = torch.randn(B, Hkv, S, n_blocks, device="cuda", generator=gen)
    qb = torch.arange(S, device="cuda") // BLOCK
    future = torch.arange(n_blocks, device="cuda")[None, :] > qb[:, None]
    scores = scores.masked_fill(future, float("-inf"))
    scores.scatter_(-1, qb.view(1, 1, S, 1).expand(B, Hkv, S, 1), float("inf"))
    k = min(topk, n_blocks)
    vals, idx = scores.topk(k, dim=-1)
    idx = idx.masked_fill(vals == float("-inf"), -1)
    if k < topk:
        idx = torch.nn.functional.pad(idx, (0, topk - k), value=-1)
    return idx.to(torch.int32).contiguous()


def _inputs(S, B, Hkv, topk, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    Hq = 16 * Hkv
    q = torch.randn(S, B, Hq, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(S, B, Hkv, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(S, B, Hkv, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    dout = torch.randn(S, B, Hq, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    return q, k, v, dout, _table(B, Hkv, S, topk, gen)


def _eager_grads(q, k, v, dout, table, dtype):
    from primus.backends.megatron.core.transformer.minimax_m3.eager import (
        build_block_keep,
        eager_block_sparse_attention,
    )

    qs, ks, vs = (t.detach().to(dtype).requires_grad_(True) for t in (q, k, v))
    keep = build_block_keep(table.long(), q.shape[0], BLOCK)
    out, _ = eager_block_sparse_attention(*(t.permute(1, 2, 0, 3) for t in (qs, ks, vs)), keep, D**-0.5)
    out.backward(dout.to(dtype).permute(1, 2, 0, 3))
    return [t.grad.float() for t in (qs, ks, vs)]


def _flydsl_grads(q, k, v, dout, table):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
        msa_token_bwd,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd import (
        msa_token_fwd,
    )

    out, lse = msa_token_fwd(q, k, v, table)
    return msa_token_bwd(dout, q, k, v, out, lse, table)


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def _snr_db(ref, x):
    ref, x = ref.double(), x.double()
    return (10 * torch.log10(ref.norm() ** 2 / ((ref - x).norm() ** 2 + 1e-12))).item()


@pytest.mark.parametrize(
    "S, B, Hkv, topk",
    [
        (300, 1, 1, 16),  # partial last block, fewer blocks than topk (-1 slots)
        (1000, 1, 4, 4),  # topk below the visible block count
        (1000, 2, 2, 16),  # batch > 1
        (2148, 1, 4, 16),
    ],
)
def test_gradients_match_eager(S, B, Hkv, topk):
    q, k, v, dout, table = _inputs(S, B, Hkv, topk)
    grads = _flydsl_grads(q, k, v, dout, table)
    ref = _eager_grads(q, k, v, dout, table, torch.float32)
    bf16 = _eager_grads(q, k, v, dout, table, torch.bfloat16)

    for name, g, r, e in zip(("dq", "dk", "dv"), grads, ref, bf16):
        assert torch.isfinite(g).all(), name
        snr = _snr_db(r, g.float())
        assert snr >= 40.0, f"{name}: SNR {snr:.1f} dB"
        err, bar = _rel(g.float(), r), _rel(e, r)
        assert err <= max(2.0 * bar, 1e-2), f"{name}: worst element {err:.2e} vs eager bf16 {bar:.2e}"


def test_backward_is_deterministic():
    """No atomics: the chunked dK/dV partials are reduced in a fixed order."""
    q, k, v, dout, table = _inputs(2048, 1, 4, 16, seed=1)
    first = _flydsl_grads(q, k, v, dout, table)
    second = _flydsl_grads(q, k, v, dout, table)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_inverted_table_and_chunks():
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
        build_inverted_table,
        plan_chunks,
    )

    B, Hkv, S, topk = 2, 2, 700, 4
    table = _table(B, Hkv, S, topk, torch.Generator(device="cuda").manual_seed(2))
    n_blocks = -(-S // BLOCK)
    ptr, tokens = build_inverted_table(table, n_blocks)

    for b in range(B):
        for g in range(Hkv):
            for blk in range(n_blocks):
                row = (b * Hkv + g) * n_blocks + blk
                got = tokens[ptr[row] : ptr[row + 1]].tolist()
                want = (table[b, g] == blk).any(dim=-1).nonzero().flatten().tolist()
                assert got == want, (b, g, blk)

    chunks, chunk_ptr, n_chunks = plan_chunks(ptr, tokens.numel(), target_chunks=64, min_chunk=16)
    assert chunks.shape[0] == n_chunks
    for row in range(ptr.numel() - 1):
        mine = chunks[chunk_ptr[row] : chunk_ptr[row + 1]]
        assert mine.shape[0] >= 1 and (mine[:, 0] == row).all()
        assert mine[0, 1] == ptr[row] and mine[-1, 2] == ptr[row + 1]
        assert (mine[1:, 1] == mine[:-1, 2]).all(), "chunks must tile the row without gaps"
    tail = chunks[chunk_ptr[-1] :]
    assert (tail[:, 1] == tail[:, 2]).all(), "chunks past the live ones must be empty"


def test_no_host_sync():
    """Neither pass may block the CPU on the GPU: that would stall every MSA layer."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
        msa_token_bwd,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd import (
        msa_token_fwd,
    )

    q, k, v, dout, table = _inputs(1000, 2, 2, 16, seed=3)
    out, lse = msa_token_fwd(q, k, v, table)  # compile outside the check
    msa_token_bwd(dout, q, k, v, out, lse, table)
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        out, lse = msa_token_fwd(q, k, v, table)
        msa_token_bwd(dout, q, k, v, out, lse, table)
    finally:
        torch.cuda.set_sync_debug_mode("default")
