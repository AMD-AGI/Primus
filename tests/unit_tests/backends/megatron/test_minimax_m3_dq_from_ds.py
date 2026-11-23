###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_dq_from_ds`` (relaxed numerics): the MSA backward with dQ taken from the dK/dV kernel's dS
(``build_dq_ds_w32``) against a dense FP32 reference and the reference kernels, with explicit
tolerances; determinism; the dispatcher's first-call comparison and its fallbacks."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("flydsl")
pytest.importorskip("triton")


def _gcn_arch() -> str:
    if not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_properties(0).gcnArchName


pytestmark = pytest.mark.skipif(
    "gfx1250" not in _gcn_arch(), reason="dQ from dS is in the gfx1250 wave32 MSA kernels"
)

BLOCK, D = 128, 128


def _table(B, Hkv, S, topk, kind, seed=0):
    """Causal, left-packed, -1 padded selections, own block in slot 0. ``random``: independent
    per token; ``local``: neighbours pick nearly the same blocks in different orders; ``hot``: a
    few blocks most tokens select; ``hostile``: -1 holes, rolled rows, a repeated block."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n_blocks = -(-S // BLOCK)
    qb = torch.arange(S, device="cuda") // BLOCK
    if kind == "random":
        scores = torch.randn(B, Hkv, S, n_blocks, device="cuda", generator=gen)
    else:
        per_block = torch.randn(B, Hkv, n_blocks, n_blocks, device="cuda", generator=gen)
        scores = per_block[:, :, qb] + 0.3 * torch.randn(B, Hkv, S, n_blocks, device="cuda", generator=gen)
        if kind == "hot":
            scores[..., : max(1, n_blocks // 8)] += 6.0
    future = torch.arange(n_blocks, device="cuda")[None, :] > qb[:, None]
    scores = scores.masked_fill(future, float("-inf"))
    scores.scatter_(-1, qb.view(1, 1, S, 1).expand(B, Hkv, S, 1), float("inf"))
    k = min(topk, n_blocks)
    vals, idx = scores.topk(k, dim=-1)
    idx = idx.masked_fill(vals == float("-inf"), -1)
    if k < topk:
        idx = torch.nn.functional.pad(idx, (0, topk - k), value=-1)
    table = idx.to(torch.int32).contiguous()
    if kind == "hostile":
        holes = torch.rand(B, Hkv, S, topk, device="cuda", generator=gen) < 0.3
        holes[..., 0] = False
        mixed = table.masked_fill(holes, -1)
        mixed[:, :, 1::2] = mixed[:, :, 1::2].roll(1, dims=-1)
        mixed[:, :, 2::4] = mixed[:, :, 2::4].roll(3, dims=-1)
        mixed[:, :, 3::8, -1] = mixed[:, :, 3::8, 0]  # one block twice in a row
        # only for rows whose topk slots are all read (the kernels read the first min(visible, topk)
        # slots): earlier rows stay left-packed, so every token keeps a visible key
        full = (torch.arange(S, device="cuda") // BLOCK + 1 >= topk).view(1, 1, S, 1)
        table = torch.where(full, mixed, table).contiguous()
    return table


def _fp32_reference(q, k, v, dout, table):
    """Dense FP32 attention over the selected blocks: o, the per-slot attention mass (group mean),
    dq, dk, dv. ``q`` ``[S, B, Hq, D]``, ``table`` ``[B, Hkv, S, topk]``."""
    S, B, Hq, _ = q.shape
    Hkv = k.shape[2]
    R = Hq // Hkv
    topk = table.shape[-1]
    n_blocks = -(-S // BLOCK)
    pos = torch.arange(S, device="cuda")
    keep = torch.zeros(B, Hkv, S, n_blocks + 1, dtype=torch.bool, device="cuda")
    keep.scatter_(-1, torch.where(table < 0, n_blocks, table).long(), True)
    keep = keep[..., :n_blocks][..., pos // BLOCK] & (pos.view(1, 1, 1, S) <= pos.view(1, 1, S, 1))
    qf = q.float().requires_grad_(True)
    kf = k.float().requires_grad_(True)
    vf = v.float().requires_grad_(True)
    qh = qf.permute(1, 2, 0, 3).reshape(B, Hkv, R, S, D)
    kh = kf.permute(1, 2, 0, 3)
    vh = vf.permute(1, 2, 0, 3)
    logits = torch.einsum("bgrsd,bgkd->bgrsk", qh, kh) * D**-0.5
    logits = logits.masked_fill(~keep.unsqueeze(2), float("-inf"))
    p = torch.softmax(logits, dim=-1)
    o = torch.einsum("bgrsk,bgkd->bgrsd", p, vh).reshape(B, Hq, S, D).permute(2, 0, 1, 3)
    (o * dout.float()).sum().backward()
    # mass of slot s of token t: its block's keys
    pb = torch.nn.functional.pad(p.detach(), (0, n_blocks * BLOCK - S)).view(B, Hkv, R, S, n_blocks, BLOCK)
    pb = pb.sum(-1).mean(2)  # [B, Hkv, S, n_blocks]
    mass = pb.gather(-1, table.clamp_min(0).long()).masked_fill(table < 0, 0.0)
    return o.detach(), mass, qf.grad, kf.grad, vf.grad


def _rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _inputs(S, B, Hkv, topk, kind, seed=11):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    Hq = 16 * Hkv
    q = torch.randn(S, B, Hq, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(S, B, Hkv, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn(S, B, Hkv, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    dout = torch.randn(S, B, Hq, D, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.05
    table = _table(B, Hkv, S, topk, kind, seed=5)
    if kind == "hostile":
        # a repeated block would be counted twice by the kernels but once by the dense reference
        table[:, :, 3::8, -1] = -1
    return q, k, v, dout, table


def _backward(q, k, v, dout, table, walk=None, **kwargs):
    """(dq, dk, dv) of ``msa_token_bwd`` after the forward on the same table (or its walk)."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
        build_block_plan,
        msa_token_bwd,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd_w32 import (
        msa_token_fwd_wave32,
    )

    plan = build_block_plan(table, BLOCK)
    steps = table if walk is None else walk[0]
    out, lse = msa_token_fwd_wave32(q, k, v, steps, None, BLOCK, False, shared_tiles=True)
    if walk is not None:
        kwargs = dict(kwargs, walk_slots=walk[1].to(torch.int32))
    return msa_token_bwd(dout, q, k, v, out, lse, steps, None, BLOCK, plan=plan, shared_tiles=True, **kwargs)


CASES = [
    # S, B, Hkv, topk, kind, aligned walk
    (1024, 1, 2, 4, "local", False),
    (1024, 2, 1, 8, "hot", False),  # B > 1, hot blocks that many queries select
    (768, 1, 2, 16, "local", False),  # early queries only: fewer blocks than topk, -1 padded rows
    (900, 1, 2, 4, "random", False),  # S not a multiple of the block
    (641, 1, 1, 4, "random", False),  # odd S: one token per workgroup
    (1280, 1, 1, 4, "hostile", False),
    (1024, 1, 2, 4, "local", True),  # with msa_aligned_block_walk: records found through the slots
    (1152, 2, 1, 16, "hot", True),
]


@pytest.mark.parametrize("S, B, Hkv, topk, kind, aligned", CASES)
def test_backward_matches_fp32_and_the_reference_kernels(S, B, Hkv, topk, kind, aligned):
    """Relaxed numerics: dq, dk and dv within an explicit tolerance of the FP32 reference, no
    further from it than 1.25 x the reference kernels (plus 1e-6), within 1e-3 of the reference
    kernels, and bit-identical between two runs."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import block_walk

    q, k, v, dout, table = _inputs(S, B, Hkv, topk, kind)
    walk = block_walk._kernel(table) if aligned else None
    ref = _fp32_reference(q, k, v, dout, table)[2:]
    base = _backward(q, k, v, dout, table, walk)
    got = _backward(q, k, v, dout, table, walk, dq_from_ds="force")
    again = _backward(q, k, v, dout, table, walk, dq_from_ds="force")
    tol = {"dq": 6e-3, "dk": 6e-3, "dv": 5e-3}
    for name, r, b, g, a in zip(["dq", "dk", "dv"], ref, base, got, again):
        assert torch.equal(g, a), f"{name}: two runs differ"
        assert torch.isfinite(g.float()).all()
        err, err_base = _rel(g, r), _rel(b, r)
        assert err <= tol[name], f"{name}: relative error {err:.3e} vs FP32"
        assert err <= 1.25 * err_base + 1e-6, f"{name}: {err:.3e} vs reference kernels {err_base:.3e}"
        assert _rel(g, b) <= 1e-3, f"{name}: {_rel(g, b):.3e} from the reference kernels"


def test_row_dot_matches_torch():
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.row_dot import row_dot

    gen = torch.Generator(device="cuda").manual_seed(3)
    for shape in [(513, 1, 64, D), (64, 3, 16, D), (1, 1, 16, D)]:
        a = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen)
        b = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen)
        want = (a.double() * b.double()).sum(-1)
        got = row_dot(a, b)
        assert got.dtype == torch.float32 and got.shape == want.shape
        assert torch.equal(got, row_dot(a, b))
        assert (got.double() - want).abs().max().item() <= 1e-4 * want.abs().max().item()


def test_dispatcher_decides_once_per_shape_and_falls_back(monkeypatch):
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import msa_token_bwd as bwd

    q, k, v, dout, table = _inputs(512, 1, 2, 4, "local")
    saved = dict(bwd._DS_OK)
    bwd._DS_OK.clear()
    try:
        want = _backward(q, k, v, dout, table)
        assert bwd._DS_OK == {}  # the switch is off: nothing decided
        first = _backward(q, k, v, dout, table, dq_from_ds=True)
        assert list(bwd._DS_OK.values()) == [True]
        second = _backward(q, k, v, dout, table, dq_from_ds=True)
        assert all(torch.equal(a, b) for a, b in zip(first, second))
        assert not torch.equal(first[0], want[0])  # another dS: last-bit differences in dq
        assert all(_rel(a, b) <= 1e-3 for a, b in zip(first, want))

        # a comparison that fails keeps the reference kernels, bit for bit, for the shape
        bwd._DS_OK.clear()
        monkeypatch.setattr(bwd, "_DS_CHECK_TOL", 0.0)
        for _ in range(2):
            got = _backward(q, k, v, dout, table, dq_from_ds=True)
            assert all(torch.equal(a, b) for a, b in zip(got, want))
        assert list(bwd._DS_OK.values()) == [False]

        # a dS buffer over the limit: not eligible, nothing decided, reference kernels
        bwd._DS_OK.clear()
        monkeypatch.setattr(bwd, "_DS_MAX_BYTES", 1024)
        got = _backward(q, k, v, dout, table, dq_from_ds=True)
        assert all(torch.equal(a, b) for a, b in zip(got, want))
        assert bwd._DS_OK == {}
    finally:
        bwd._DS_OK.clear()
        bwd._DS_OK.update(saved)


def test_lockstep_kernel_refuses_a_sequence_it_does_not_divide(monkeypatch):
    """The barrier count is workgroup-uniform only when the group divides S."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl import msa_token_bwd_w32

    assert msa_token_bwd_w32.dq_ds_tokens_per_wg(642, BLOCK) == 1
    assert msa_token_bwd_w32.dq_ds_tokens_per_wg(644, BLOCK) == 4
    q, k, v, dout, table = _inputs(642, 1, 1, 4, "local")
    monkeypatch.setattr(msa_token_bwd_w32, "dq_ds_tokens_per_wg", lambda S, block: 4)  # a broken helper
    with pytest.raises(ValueError, match="msa_dq_from_ds dQ"):
        _backward(q, k, v, dout, table, dq_from_ds="force")


def _fake_attention(**config):
    defaults = dict(msa_dq_from_ds=True, msa_exact_numerics=False)
    defaults.update(config)
    return SimpleNamespace(config=SimpleNamespace(**defaults), _gfx1250=True, block_size=BLOCK)


def test_switches_turn_it_off():
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )
    from primus.backends.megatron.core.transformer.minimax_m3.attention import MinimaxSparseAttention

    assert MSATransformerConfig.__dataclass_fields__["msa_dq_from_ds"].default is False
    assert MinimaxSparseAttention._dq_from_ds(_fake_attention(), None) == {
        "dq_from_ds": True
    }  # msa_ds_chunk_store unset
    slots = torch.zeros(1, 1, 4, 4, dtype=torch.int64)
    kwargs = MinimaxSparseAttention._dq_from_ds(_fake_attention(), (None, slots))
    assert kwargs["dq_from_ds"] is True and kwargs["walk_slots"].dtype == torch.int32
    # msa_exact_numerics reproduces the reference kernels bit for bit
    assert MinimaxSparseAttention._dq_from_ds(_fake_attention(msa_exact_numerics=True), None) == {}
    assert MinimaxSparseAttention._dq_from_ds(_fake_attention(msa_dq_from_ds=False), None) == {}
    other_gpu = _fake_attention()
    other_gpu._gfx1250 = False
    assert MinimaxSparseAttention._dq_from_ds(other_gpu, None) == {}
