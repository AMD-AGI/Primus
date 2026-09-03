###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Primus-Turbo DeepSeek-V3 (MLA) ``Attention.forward``.

``primus.backends.torchtitan.models.deepseek_v3.model.model.Attention`` is
swapped in for upstream's MLA attention when
``primus_turbo.use_turbo_attention`` is enabled (see
``primus/backends/torchtitan/patches/turbo/attention_patches.py``). Its
``forward`` differs from upstream's in exactly one way: it does *not*
transpose q/k/v to ``(bsz, n_heads, seqlen, head_dim)`` before calling
``self.inner_attention`` and does not pass a ``scale`` kwarg, because when
turbo attention is enabled ``inner_attention`` is replaced by
``primus_turbo``'s ``TurboAttention``, which consumes the
``(bsz, seqlen, n_heads, head_dim)`` layout directly and computes its own
scale internally. If this module ever regressed to upstream's
transpose-before/after-call behavior, the kernel would silently receive the
wrong tensor layout.
"""

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def deepseek_v3_args():
    pytest.importorskip("torchtitan")
    from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs

    return DeepSeekV3ModelArgs(
        dim=32,
        n_heads=4,
        q_lora_rank=0,
        kv_lora_rank=16,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        attn_type="sdpa",
        max_seq_len=8,
        original_seq_len=8,
        rope_theta=10000.0,
    )


class _RecordingInnerAttention(torch.nn.Module):
    """Host-side stand-in for ``TurboAttention``: records the exact tensors
    and call signature ``Attention.forward`` uses, then echoes ``v`` back so
    the surrounding reshape/``wo`` projection can be exercised end-to-end."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, q, k, v, *args, **kwargs):
        self.calls.append(
            {
                "q_shape": tuple(q.shape),
                "k_shape": tuple(k.shape),
                "v": v,
                "args": args,
                "kwargs": kwargs,
            }
        )
        return v


class TestDeepSeekV3TurboAttentionForward:
    def test_forward_calls_inner_attention_with_untransposed_layout(self, deepseek_v3_args):
        from torchtitan.models.deepseek_v3.model.model import precompute_freqs_cis

        from primus.backends.torchtitan.models.deepseek_v3.model.model import Attention

        attn = Attention(deepseek_v3_args)
        stub = _RecordingInnerAttention()
        attn.inner_attention = stub

        bsz, seqlen = 2, 8
        x = torch.randn(bsz, seqlen, deepseek_v3_args.dim)
        freqs_cis = precompute_freqs_cis(deepseek_v3_args)

        output = attn(x, freqs_cis, attention_masks=None)

        assert len(stub.calls) == 1
        call = stub.calls[0]

        # Turbo layout: (bsz, seqlen, n_heads, head_dim) — NOT the upstream
        # (bsz, n_heads, seqlen, head_dim) that a pre-transpose would produce.
        # This is the falsifiable crux: reintroducing upstream's
        # ``q.transpose(1, 2)`` (etc.) before the call would flip dims 1/2
        # here and fail the assertion.
        n_heads = deepseek_v3_args.n_heads
        qk_head_dim = deepseek_v3_args.qk_nope_head_dim + deepseek_v3_args.qk_rope_head_dim
        v_head_dim = deepseek_v3_args.v_head_dim
        assert call["q_shape"] == (bsz, seqlen, n_heads, qk_head_dim)
        assert call["k_shape"] == (bsz, seqlen, n_heads, qk_head_dim)
        assert tuple(call["v"].shape) == (bsz, seqlen, n_heads, v_head_dim)

        # No upstream-style ``scale=`` kwarg (or any other kwarg/positional
        # extra) is forwarded — TurboAttention computes its own scale.
        assert call["args"] == ()
        assert call["kwargs"] == {}

        # Output is reshaped straight from the (bsz, seqlen, n_heads, v_head_dim)
        # echo, with no intervening transpose-back, then projected by wo. Since
        # the stub echoes v verbatim, the output must equal wo(v.view(bsz,
        # seqlen, -1)) exactly — a transpose-back before the view would permute
        # which head's slice lands in which output channel and break this.
        expected = attn.wo(call["v"].contiguous().view(bsz, seqlen, -1))
        torch.testing.assert_close(output, expected)
