###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Packed-sequence (THD) equivalence for DeepSeek-V4 attention.

The defining property of packed training is that packing must be INVISIBLE to the
model: a sequence's outputs must not depend on what happens to be packed next to it.
That is exactly what these tests assert, and it is a much sharper check than "the
packed run trains" -- a leak across a sequence boundary still produces a plausible
loss curve, it just silently conditions each sample on its neighbours.

Method: run one sequence alone (BSHD), then run the SAME sequence embedded in a pack
with different neighbours (THD), and require the rows belonging to that sequence to
match. Two neighbour configurations are used, because a leak that happens to be
symmetric could survive a single one.

These are CPU tests at tiny head_dim on the eager backend, mirroring
``test_deepseek_v4_attention.py`` -- the Triton kernels are specialised for
head_dim=512 and cannot run here.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

from tests.unit_tests.megatron.transformer.deepseek_v4.test_deepseek_v4_attention import (  # noqa: E402
    _make_attention,
    _make_compressed_attention,
    _make_v4_config,
)


class _PackedSeqParams:
    """Minimal stand-in for Megatron's PackedSeqParams (only cu_seqlens_q is read)."""

    def __init__(self, cu_seqlens):
        self.cu_seqlens_q = cu_seqlens
        self.cu_seqlens_kv = cu_seqlens
        self.qkv_format = "thd"


def _attention(compress_ratio: int):
    # Same tiny shapes the sibling attention tests use, so this file exercises the same
    # eager reference path they validate against.
    torch.manual_seed(0)
    cfg = _make_v4_config(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        rotary_dim=8,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=8,
        attn_sink=True,
    )
    # compress_ratio must be set at CONSTRUCTION: the compressor (and, for CSA, the
    # indexer) are built in __init__ from it, so assigning the attribute afterwards
    # leaves those submodules as None and the branch dies on a null call.
    if compress_ratio == 0:
        attn = _make_attention(cfg)
    else:
        attn = _make_compressed_attention(config=cfg, compress_ratio=compress_ratio)
    attn.eval()
    return attn


def _run(attn, hidden, cu=None):
    # Everything follows `hidden`'s device: the fused path hands these straight to Triton,
    # which rejects host pointers with a bare "cannot be accessed from Triton (cpu tensor?)".
    dev = hidden.device
    S = hidden.shape[1]
    pos = torch.arange(S, device=dev).unsqueeze(0)
    psp = None
    if cu is not None:
        psp = _PackedSeqParams(torch.tensor(cu, dtype=torch.int32, device=dev))
        # Under packing, positions restart at every sequence boundary.
        pos = torch.cat([torch.arange(cu[i + 1] - cu[i], device=dev) for i in range(len(cu) - 1)]).unsqueeze(
            0
        )
    with torch.no_grad():
        return attn(hidden, pos, packed_seq_params=psp)


def _seq_starts_reference(cu):
    out = []
    for i in range(len(cu) - 1):
        out.extend([cu[i]] * (cu[i + 1] - cu[i]))
    return torch.tensor(out, dtype=torch.int32)


def test_seq_starts_maps_every_row_to_its_sequence():
    attn = _attention(0)
    cu = [0, 3, 7, 10]
    got = attn._thd_seq_starts(_PackedSeqParams(torch.tensor(cu, dtype=torch.int32)), 1, 10, "cpu")
    assert torch.equal(got, _seq_starts_reference(cu))


def test_seq_starts_is_none_without_packing():
    attn = _attention(0)
    assert attn._thd_seq_starts(None, 1, 10, "cpu") is None


def test_seq_starts_rejects_short_cu_seqlens():
    """A cu_seqlens that does not reach S would leave the tail rows attending across
    boundaries -- silently. It must raise instead."""
    attn = _attention(0)
    psp = _PackedSeqParams(torch.tensor([0, 3, 7], dtype=torch.int32))
    with pytest.raises(RuntimeError, match="cu_seqlens must cover"):
        attn._thd_seq_starts(psp, 1, 10, "cpu")


def test_seq_starts_rejects_batch_gt_1():
    attn = _attention(0)
    psp = _PackedSeqParams(torch.tensor([0, 10], dtype=torch.int32))
    with pytest.raises(RuntimeError, match="micro_batch_size=1"):
        attn._thd_seq_starts(psp, 2, 10, "cpu")


@pytest.mark.parametrize("compress_ratio", [0])
def test_packed_matches_unpacked_dense(compress_ratio):
    """The middle sequence of a pack must produce the same output as when run alone.

    Run twice with DIFFERENT neighbours: if the two packed runs agree with each other
    but not with the standalone run, the bug is a constant offset; if they disagree with
    each other, the neighbours are leaking in.
    """
    attn = _attention(compress_ratio)
    D = attn.config.hidden_size
    torch.manual_seed(1)

    L, LA, LB = 6, 4, 5
    target = torch.randn(1, L, D)
    alone = _run(attn, target)

    for seed in (2, 3):
        torch.manual_seed(seed)
        left = torch.randn(1, LA, D)
        right = torch.randn(1, LB, D)
        packed = torch.cat([left, target, right], dim=1)
        cu = [0, LA, LA + L, LA + L + LB]
        out = _run(attn, packed, cu)
        got = out[:, LA : LA + L]
        torch.testing.assert_close(
            got,
            alone,
            rtol=1e-4,
            atol=1e-4,
            msg=lambda m: f"packed sequence differs from standalone (neighbour seed {seed}):\n{m}",
        )


@pytest.mark.parametrize("compress_ratio", [4, 128])
@pytest.mark.parametrize("lengths", [(8, 12, 8), (7, 13, 5)])
def test_packed_matches_unpacked_compressed(compress_ratio, lengths):
    """Same invisibility property for the compressed branches (CSA cr=4, HCA cr=128).

    These are where packing is hardest: the compressor pools fixed windows over the row
    axis, so a window can straddle a sequence boundary and mix two samples into one
    compressed key. Overlap mode (cr=4) is worse -- it stitches window i with window
    i-1, so even correctly-binned windows leak across the boundary.

    ``lengths`` covers both multiples and non-multiples of the ratio: a sequence whose
    length is not a whole number of windows is the case where a naive `S // ratio`
    reshape silently absorbs the next sequence's first tokens.
    """
    attn = _attention(compress_ratio)
    D = attn.config.hidden_size
    LA, L, LB = lengths
    torch.manual_seed(1)
    target = torch.randn(1, L, D)
    # Baseline is the target packed ALONE, not the contiguous (non-packed) path: a
    # standalone length that is not a multiple of `ratio` cannot go through the
    # contiguous compressor at all (it asserts). Comparing "packed with neighbours" to
    # "packed alone" is the same invisibility property and is well-defined for any
    # length -- which is the case that matters, since real samples are ragged.
    alone = _run(attn, target, [0, L])

    for seed in (2, 3):
        torch.manual_seed(seed)
        packed = torch.cat([torch.randn(1, LA, D), target, torch.randn(1, LB, D)], dim=1)
        cu = [0, LA, LA + L, LA + L + LB]
        out = _run(attn, packed, cu)
        torch.testing.assert_close(
            out[:, LA : LA + L],
            alone,
            rtol=1e-4,
            atol=1e-4,
            msg=lambda m: f"cr={compress_ratio} lengths={lengths} seed={seed}:\n{m}",
        )


def _fused_attention(compress_ratio: int):
    """V4 attention on the FUSED triton_v2 path, at the head_dim the kernels specialise for.

    The eager tests above run at head_dim=16 on CPU, which the Triton kernels cannot do --
    so they say nothing about the fused index construction. This builds the real thing.
    """
    torch.manual_seed(0)
    cfg = _make_v4_config(
        hidden_size=512,
        num_heads=8,  # in the fused indexer's _SUPPORTED_H
        head_dim=512,  # what dsa_fwd/bwd_v4_triton are specialised for
        rotary_dim=64,
        q_lora_rank=256,
        o_groups=2,
        o_lora_rank=128,
        attn_sink=True,
    )
    cfg.use_v4_attention_backend = "triton_v2"
    cfg.use_v4_csa_attention_backend = "triton_v2"
    # The sparse-MLA adapter asserts swa_window > 0 -- it builds the local half of the
    # index matrix from the window, so a zero window has no meaning there. Keep it well
    # under the shortest test sequence so the window is the binding constraint rather
    # than the sequence length.
    cfg.attn_sliding_window = 64
    if compress_ratio == 0:
        attn = _make_attention(cfg)
    else:
        attn = _make_compressed_attention(config=cfg, compress_ratio=compress_ratio)
    attn = attn.to(device="cuda", dtype=torch.bfloat16).eval()
    # DualRoPE is not assigned as a submodule (`self.rope` is never set in __init__), so
    # nn.Module.to() does not reach its inv_freq buffer and Triton then rejects the host
    # pointer. Move it explicitly.
    rope = getattr(attn, "rope", None)
    if rope is not None and hasattr(rope, "to"):
        rope.to(device="cuda")
        for sub in ("compress_rope", "attn_rope"):
            r = getattr(rope, sub, None)
            if r is not None and hasattr(r, "to"):
                r.to(device="cuda")
    return attn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused path needs a GPU")
@pytest.mark.parametrize("compress_ratio", [0, 4, 128])
def test_packed_matches_unpacked_fused(compress_ratio):
    """Same invisibility property, on the fused triton_v2 path.

    This is the configuration the 128k example actually runs, and it exercises a
    completely different code path from the eager tests: the per-row origin has to reach
    the index matrix that the kernel consumes, and cross-sequence columns have to be
    masked with the -1 sentinel rather than an additive -inf.

    Tolerances are bf16-scale. The property is exact in exact arithmetic -- what differs
    between the two runs is only the summation order inside the kernel, because the packed
    run has more (masked-out) key slots.
    """
    attn = _fused_attention(compress_ratio)
    D = attn.config.hidden_size
    LA, L, LB = 256, 384, 128
    torch.manual_seed(1)
    target = torch.randn(1, L, D, device="cuda", dtype=torch.bfloat16)
    alone = _run(attn, target, [0, L])

    for seed in (2, 3):
        torch.manual_seed(seed)
        packed = torch.cat(
            [
                torch.randn(1, LA, D, device="cuda", dtype=torch.bfloat16),
                target,
                torch.randn(1, LB, D, device="cuda", dtype=torch.bfloat16),
            ],
            dim=1,
        )
        out = _run(attn, packed, [0, LA, LA + L, LA + L + LB])
        torch.testing.assert_close(
            out[:, LA : LA + L].float(),
            alone.float(),
            rtol=2e-2,
            atol=2e-2,
            msg=lambda m: f"fused cr={compress_ratio} seed={seed}:\n{m}",
        )


def test_first_sequence_of_pack_matches_unpacked_dense():
    """The first sequence has seq_start == 0, so it is the one case a scalar-origin
    implementation gets right by accident. Included so a fix cannot pass by only
    special-casing row 0."""
    attn = _attention(0)
    D = attn.config.hidden_size
    torch.manual_seed(1)
    L = 6
    target = torch.randn(1, L, D)
    alone = _run(attn, target)
    tail = torch.randn(1, 5, D)
    out = _run(attn, torch.cat([target, tail], dim=1), [0, L, L + 5])
    torch.testing.assert_close(out[:, :L], alone, rtol=1e-4, atol=1e-4)
