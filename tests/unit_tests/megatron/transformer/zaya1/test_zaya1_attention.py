###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Causal-mask detection, dropping a causal mask for the flash path, and the
flash path against the dense-mask path."""

import pytest
import torch
from test_zaya1_math import _cfg  # noqa: E402  (pytest puts this dir on sys.path)

from primus.backends.megatron.core.models.zaya1 import zaya1_modules as zm


def _causal(seq, batch=1):
    # Megatron's convention: True blocks a key.
    return torch.ones(seq, seq, dtype=torch.bool).triu(1).expand(batch, 1, seq, seq).clone()


def test_is_causal_mask():
    seq = 5
    causal = _causal(seq, batch=2)
    assert zm.is_causal_mask(None)
    assert zm.is_causal_mask(causal)
    assert zm.is_causal_mask(causal[0, 0])
    assert zm.is_causal_mask(causal.to(torch.uint8))
    assert zm.is_causal_mask(causal.float())
    padded = causal.clone()
    padded[1, :, 1:, 0] = True  # row 1 also blocks its first key
    assert not zm.is_causal_mask(padded)
    assert not zm.is_causal_mask(~causal)
    assert not zm.is_causal_mask(torch.zeros(seq, seq, dtype=torch.bool))
    assert not zm.is_causal_mask(torch.ones(2, seq, seq + 1, dtype=torch.bool).triu(1))
    assert not zm.is_causal_mask(torch.zeros(seq, dtype=torch.bool))


def test_stack_drops_only_a_causal_mask(monkeypatch):
    torch.manual_seed(3)
    stack = zm.ZayaStack(_cfg(num_layers=4))
    stack.eval()
    got_mask = []
    attention = zm.attention

    def recording_attention(query, key, value, attention_mask):
        got_mask.append(attention_mask is not None)
        return attention(query, key, value, attention_mask)

    monkeypatch.setattr(zm, "attention", recording_attention)
    seq, batch = 6, 2
    hidden = torch.randn(seq, batch, 32)
    causal = _causal(seq, batch)
    padded = causal.clone()
    padded[1, :, 1:, 0] = True
    with torch.no_grad():
        monkeypatch.setattr(zm, "ATTN_FLASH", True)
        dropped = stack(hidden, None, causal)
        assert got_mask == [False, False]  # both CCA stages ran without the mask
        got_mask.clear()
        stack(hidden, None, padded)
        assert got_mask == [True, True]  # a mask that blocks more than the future is kept
        got_mask.clear()
        monkeypatch.setattr(zm, "ATTN_FLASH", False)
        dense = stack(hidden, None, causal)
        assert got_mask == [True, True]
    torch.testing.assert_close(dropped, dense, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or zm._flash_attn_func is None, reason="needs a GPU and flash_attn"
)
def test_flash_attention_matches_the_dense_mask_path(monkeypatch):
    """ZAYA1's attention shape: 8 query heads over 2 KV groups, head_dim 128, bf16."""
    torch.manual_seed(4)
    seq, batch = 128, 2
    shapes = ((seq, batch, 8, 128), (seq, batch, 2, 128), (seq, batch, 2, 128))
    inputs = [torch.randn(s, device="cuda", dtype=torch.bfloat16) for s in shapes]
    weights = torch.randn(seq, batch, 8, 128, device="cuda")
    runs = []
    for flash, mask in ((True, None), (False, _causal(seq).cuda())):
        monkeypatch.setattr(zm, "ATTN_FLASH", flash)
        q, k, v = (x.clone().requires_grad_(True) for x in inputs)
        out = zm.attention(q, k, v, mask)
        (out.float() * weights).sum().backward()
        runs.append([out.detach(), q.grad, k.grad, v.grad])
    for name, a, b in zip(("output", "dq", "dk", "dv"), *runs):
        err = (a.float() - b.float()).norm() / b.float().norm()
        assert err < 1e-2, f"{name}: relative error {err:.3e}"
