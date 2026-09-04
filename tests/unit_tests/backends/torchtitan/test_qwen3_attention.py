###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Qwen3 Primus-Turbo Attention.forward override.

Needs torchtitan (auto-skipped otherwise). The override in
``primus.backends.torchtitan.models.qwen3.model.model.Attention`` diverges from
the upstream ``torchtitan`` Qwen3 attention in one key way: with
``inner_attention`` replaced by ``TurboAttention`` (which natively handles GQA
and causal masking on the (bs, seqlen, n_heads, head_dim) layout), the forward
pass must skip the upstream transpose-to-(bs, n_heads, seqlen, head_dim) and
must not thread ``attention_masks``/``scale``/``enable_gqa`` into the inner
call. These tests pin that contract with a stub standing in for
``TurboAttention``.
"""

import pytest

pytest.importorskip("torchtitan")

import torch
import torch.nn as nn

from primus.backends.torchtitan.models.qwen3.model.model import Attention


def _make_args(qk_norm=True):
    from torchtitan.models.qwen3.model.args import Qwen3ModelArgs

    return Qwen3ModelArgs(
        dim=16,
        n_heads=4,
        n_kv_heads=2,
        head_dim=4,
        qk_norm=qk_norm,
        max_seq_len=8,
    )


def _rope_cache(head_dim, seqlen):
    from torchtitan.models.qwen3.model.model import precompute_rope_cache

    return precompute_rope_cache(head_dim, seqlen)


class _CapturingInnerAttention(nn.Module):
    """Stub for Primus-Turbo's TurboAttention: records call shape/signature."""

    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.calls = []

    def forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        xq = args[0]
        return xq


class TestQwen3AttentionForward:
    def test_output_shape(self):
        args = _make_args()
        attn = Attention(args)
        attn.inner_attention = _CapturingInnerAttention(args.head_dim)
        bs, seqlen = 2, 5
        x = torch.randn(bs, seqlen, args.dim)
        rope_cache = _rope_cache(args.head_dim, args.max_seq_len)

        out = attn(x, rope_cache, attention_masks=None)

        assert out.shape == (bs, seqlen, args.dim)

    def test_inner_attention_receives_untransposed_bs_seqlen_layout(self):
        # Upstream torchtitan's Qwen3 Attention.forward transposes xq/xk/xv to
        # (bs, n_heads, seqlen, head_dim) before calling inner_attention. The
        # Primus-Turbo override must skip that transpose, since TurboAttention
        # expects (bs, seqlen, n_heads, head_dim).
        args = _make_args()
        attn = Attention(args)
        stub = _CapturingInnerAttention(args.head_dim)
        attn.inner_attention = stub

        bs, seqlen = 2, 5
        x = torch.randn(bs, seqlen, args.dim)
        rope_cache = _rope_cache(args.head_dim, args.max_seq_len)

        attn(x, rope_cache, attention_masks=None)

        assert len(stub.calls) == 1
        call_args, call_kwargs = stub.calls[0]
        xq, xk, xv = call_args
        assert xq.shape == (bs, seqlen, args.n_heads, args.head_dim)
        assert xk.shape == (bs, seqlen, args.n_kv_heads, args.head_dim)
        assert xv.shape == (bs, seqlen, args.n_kv_heads, args.head_dim)
        # No block_mask/scale/enable_gqa kwargs threaded through: TurboAttention
        # handles GQA and causal masking internally.
        assert call_kwargs == {}

    def test_attention_masks_are_ignored(self):
        # Upstream forward asserts on attention_masks depending on attn_type
        # (e.g. `assert attention_masks is None` for sdpa). The Primus-Turbo
        # override must accept (and ignore) any value here, including a
        # non-None sentinel that would fail the upstream assertion.
        args = _make_args()
        attn = Attention(args)
        attn.inner_attention = _CapturingInnerAttention(args.head_dim)

        bs, seqlen = 1, 3
        x = torch.randn(bs, seqlen, args.dim)
        rope_cache = _rope_cache(args.head_dim, args.max_seq_len)

        out = attn(x, rope_cache, attention_masks="not-a-block-mask")

        assert out.shape == (bs, seqlen, args.dim)

    def test_qk_norm_applied_when_enabled(self):
        args = _make_args(qk_norm=True)
        attn = Attention(args)
        assert attn.q_norm is not None
        assert attn.k_norm is not None

        calls = {"q": 0, "k": 0}
        orig_q_forward = attn.q_norm.forward
        orig_k_forward = attn.k_norm.forward

        def q_wrapper(x):
            calls["q"] += 1
            return orig_q_forward(x)

        def k_wrapper(x):
            calls["k"] += 1
            return orig_k_forward(x)

        attn.q_norm.forward = q_wrapper
        attn.k_norm.forward = k_wrapper

        x = torch.randn(1, 3, args.dim)
        rope_cache = _rope_cache(args.head_dim, args.max_seq_len)
        attn(x, rope_cache, attention_masks=None)

        assert calls == {"q": 1, "k": 1}

    def test_qk_norm_skipped_when_disabled(self):
        args = _make_args(qk_norm=False)
        attn = Attention(args)
        assert attn.q_norm is None
        assert attn.k_norm is None

        x = torch.randn(1, 3, args.dim)
        rope_cache = _rope_cache(args.head_dim, args.max_seq_len)
        # Should not raise despite q_norm/k_norm being None.
        out = attn(x, rope_cache, attention_masks=None)
        assert out.shape == (1, 3, args.dim)

    def test_positions_forwarded_to_rotary_embedding(self):
        # Inspect what reaches inner_attention directly (rather than the final
        # wo-projected output) so this only pins the rotary embedding's use of
        # `positions`, independent of how any particular inner_attention
        # implementation subsequently mixes across sequence positions.
        # Seeded for determinism: both the module's random weight init and
        # the random input feed into the non-equality assertion below.
        torch.manual_seed(0)
        args = _make_args()
        attn = Attention(args)
        stub = _CapturingInnerAttention(args.head_dim)
        attn.inner_attention = stub

        bs, seqlen = 1, 3
        x = torch.randn(bs, seqlen, args.dim)
        rope_cache = _rope_cache(args.head_dim, args.max_seq_len)
        positions = torch.arange(seqlen).unsqueeze(0)

        attn(x, rope_cache, attention_masks=None, positions=positions)
        xq_with_positions, xk_with_positions, _ = stub.calls[0][0]

        stub.calls.clear()
        attn(x, rope_cache, attention_masks=None, positions=None)
        xq_default, xk_default, _ = stub.calls[0][0]

        # positions=[0..seqlen) is the same sequential order as the default
        # None path, so the rotary embedding applied should match.
        assert torch.allclose(xq_with_positions, xq_default)
        assert torch.allclose(xk_with_positions, xk_default)

        stub.calls.clear()
        alt_positions = torch.tensor([[0, 2, 1]])
        attn(x, rope_cache, attention_masks=None, positions=alt_positions)
        xq_alt, xk_alt, _ = stub.calls[0][0]

        # A non-trivial change in position ordering must change the rotary
        # embedding applied to at least one of q/k.
        assert not torch.allclose(xq_with_positions, xq_alt) or not torch.allclose(xk_with_positions, xk_alt)
