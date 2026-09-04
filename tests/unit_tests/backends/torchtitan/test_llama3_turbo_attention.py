###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the LLaMA3 Primus-Turbo Attention mirror (``forward``).

These tests need torchtitan (to build the real ``wq``/``wk``/``wv``/``wo``
linears via the base ``Attention.__init__``) and are auto-skipped otherwise.
The heavy ``inner_attention`` (TurboAttention) is replaced with a lightweight
stub so the tests exercise only the Primus ``forward`` override: projection,
reshape (no ``repeat_kv``/transpose), rotary embedding application, and the
final output projection.
"""

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def llama3_args():
    pytest.importorskip("torchtitan")
    from torchtitan.models.llama3.model.args import TransformerModelArgs

    return TransformerModelArgs(
        dim=32,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=64,
        n_layers=1,
        max_seq_len=16,
    )


class _IdentityInnerAttention(nn.Module):
    """Stand-in for TurboAttention: (bs, seqlen, n_heads, head_dim) in and out."""

    def __init__(self):
        super().__init__()

    def forward(self, xq, xk, xv):
        return xq


class _CaptureInnerAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes = {}

    def forward(self, xq, xk, xv):
        self.shapes["xq_shape"] = tuple(xq.shape)
        self.shapes["xk_shape"] = tuple(xk.shape)
        self.shapes["xv_shape"] = tuple(xv.shape)
        return xq


class TestLlama3TurboAttentionForward:
    def test_forward_output_shape(self, llama3_args):
        from primus.backends.torchtitan.models.llama3.model.model import Attention

        attn = Attention(llama3_args)
        attn.inner_attention = _IdentityInnerAttention()

        bs, seqlen = 2, 8
        x = torch.randn(bs, seqlen, llama3_args.dim)
        freqs_cis = torch.ones(seqlen, llama3_args.dim // llama3_args.n_heads // 2, dtype=torch.complex64)

        out = attn.forward(x, freqs_cis, attention_masks=None)

        assert out.shape == (bs, seqlen, llama3_args.dim)

    def test_inner_attention_receives_unrepeated_untransposed_heads(self, llama3_args):
        # The Primus forward skips `repeat_kv` and the (bs, heads, seq, dim)
        # transpose entirely -- TurboAttention consumes (bs, seq, heads, dim)
        # directly and expands GQA internally. Assert xk/xv keep n_kv_heads
        # (not repeated up to n_heads) and nothing is transposed.
        from primus.backends.torchtitan.models.llama3.model.model import Attention

        attn = Attention(llama3_args)
        stub = _CaptureInnerAttention()
        attn.inner_attention = stub

        bs, seqlen = 2, 8
        head_dim = llama3_args.dim // llama3_args.n_heads
        x = torch.randn(bs, seqlen, llama3_args.dim)
        freqs_cis = torch.ones(seqlen, head_dim // 2, dtype=torch.complex64)

        attn.forward(x, freqs_cis, attention_masks=None)

        assert stub.shapes["xq_shape"] == (bs, seqlen, llama3_args.n_heads, head_dim)
        # GQA: kv heads stay at n_kv_heads, i.e. NOT repeated up to n_heads.
        assert stub.shapes["xk_shape"] == (bs, seqlen, llama3_args.n_kv_heads, head_dim)
        assert stub.shapes["xv_shape"] == (bs, seqlen, llama3_args.n_kv_heads, head_dim)

    def test_forward_passes_positions_to_rotary_emb(self, llama3_args, monkeypatch):
        import primus.backends.torchtitan.models.llama3.model.model as llama3_mirror

        attn = llama3_mirror.Attention(llama3_args)
        attn.inner_attention = _IdentityInnerAttention()

        captured = {}
        real_apply_rotary_emb = llama3_mirror.apply_rotary_emb

        def _spy_apply_rotary_emb(xq, xk, freqs_cis, positions=None):
            captured["positions"] = positions
            return real_apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, positions=positions)

        monkeypatch.setattr(llama3_mirror, "apply_rotary_emb", _spy_apply_rotary_emb)

        bs, seqlen = 1, 4
        head_dim = llama3_args.dim // llama3_args.n_heads
        x = torch.randn(bs, seqlen, llama3_args.dim)
        freqs_cis = torch.ones(seqlen, head_dim // 2, dtype=torch.complex64)
        positions = torch.arange(seqlen).unsqueeze(0)

        attn.forward(x, freqs_cis, attention_masks=None, positions=positions)

        assert captured["positions"] is positions

    def test_forward_defaults_positions_to_none(self, llama3_args):
        from primus.backends.torchtitan.models.llama3.model.model import Attention

        attn = Attention(llama3_args)
        attn.inner_attention = _IdentityInnerAttention()

        bs, seqlen = 1, 4
        head_dim = llama3_args.dim // llama3_args.n_heads
        x = torch.randn(bs, seqlen, llama3_args.dim)
        freqs_cis = torch.ones(seqlen, head_dim // 2, dtype=torch.complex64)

        # Should not raise even though `positions` is omitted.
        out = attn.forward(x, freqs_cis, attention_masks=None)
        assert out.shape == (bs, seqlen, llama3_args.dim)

    def test_forward_uses_wo_projection(self, llama3_args):
        # Sanity check that the mirror still routes through the base
        # `Attention.wo` output projection rather than returning the raw
        # inner_attention output.
        import primus.backends.torchtitan.models.llama3.model.model as llama3_mirror

        attn = llama3_mirror.Attention(llama3_args)
        attn.inner_attention = _IdentityInnerAttention()

        bs, seqlen = 1, 4
        head_dim = llama3_args.dim // llama3_args.n_heads
        x = torch.randn(bs, seqlen, llama3_args.dim)
        freqs_cis = torch.ones(seqlen, head_dim // 2, dtype=torch.complex64)

        out = attn.forward(x, freqs_cis, attention_masks=None)

        xq = attn.wq(x).view(bs, seqlen, -1, head_dim)
        xk = attn.wk(x).view(bs, seqlen, -1, head_dim)
        # Build `expected` via the same `apply_rotary_emb` call `Attention.forward`
        # makes, rather than assuming freqs_cis=1 makes RoPE an exact no-op (an
        # implementation detail of `apply_rotary_emb` this test shouldn't rely on).
        xq, _ = llama3_mirror.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, positions=None)
        expected = attn.wo(xq.contiguous().view(bs, seqlen, -1))
        assert torch.allclose(out, expected, atol=1e-5)
