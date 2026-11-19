###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Causal-mask detection, dropping a causal mask for the flash path, and the
flash path, on flash_attn or Primus Turbo's flash_attn_func, against the
dense-mask path."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
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
    flash_stack = zm.ZayaStack(_cfg(num_layers=4, zaya_attn_flash=True)).eval()
    dense_stack = zm.ZayaStack(_cfg(num_layers=4, zaya_attn_flash=False)).eval()
    dense_stack.load_state_dict(flash_stack.state_dict())
    got_mask = []
    attention = zm.attention

    def recording_attention(query, key, value, attention_mask, flash=False, turbo=False):
        got_mask.append(attention_mask is not None)
        return attention(query, key, value, attention_mask, flash=flash, turbo=turbo)

    monkeypatch.setattr(zm, "attention", recording_attention)
    seq, batch = 6, 2
    hidden = torch.randn(seq, batch, 32)
    causal = _causal(seq, batch)
    padded = causal.clone()
    padded[1, :, 1:, 0] = True
    with torch.no_grad():
        dropped = flash_stack(hidden, None, causal)
        assert got_mask == [False, False]  # both CCA stages ran without the mask
        got_mask.clear()
        flash_stack(hidden, None, padded)
        assert got_mask == [True, True]  # a mask that blocks more than the future is kept
        got_mask.clear()
        dense = dense_stack(hidden, None, causal)
        assert got_mask == [True, True]
    torch.testing.assert_close(dropped, dense, atol=1e-5, rtol=1e-5)


@pytest.fixture
def turbo_attention_calls(monkeypatch):
    """A stand-in primus_turbo whose flash_attn_func keeps Turbo's contract, recording each call's layout.

    Turbo takes ``[b, s, h, d]``-shaped tensors and returns the output in the
    query's memory layout; ZAYA1 must hand it views of ``[s, b, h, d]`` storage.
    """
    calls = []

    def flash_attn_func(q, k, v, causal=False, **kwargs):
        assert not kwargs and causal
        calls.append([t.transpose(0, 1).is_contiguous() for t in (q, k, v)])
        out = F.scaled_dot_product_attention(
            *(t.transpose(1, 2) for t in (q, k, v)), is_causal=True, enable_gqa=q.shape[2] != k.shape[2]
        )
        return out.permute(2, 0, 1, 3).contiguous().transpose(0, 1)

    ops = types.ModuleType("primus_turbo.pytorch.ops")
    ops.flash_attn_func = flash_attn_func
    package = types.ModuleType("primus_turbo")
    package.pytorch = types.ModuleType("primus_turbo.pytorch")
    package.pytorch.ops = ops
    for name, module in (
        ("primus_turbo", package),
        ("primus_turbo.pytorch", package.pytorch),
        ("primus_turbo.pytorch.ops", ops),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_turbo_attention_takes_the_sbhd_storage_as_is(turbo_attention_calls):
    torch.manual_seed(5)
    q, k, v = torch.randn(6, 2, 8, 16), torch.randn(6, 2, 2, 16), torch.randn(6, 2, 2, 16)
    out = zm.attention(q, k, v, None, flash=True, turbo=True)
    assert turbo_attention_calls == [[True, True, True]]
    assert out.shape == q.shape and out.is_contiguous()
    torch.testing.assert_close(out, zm.attention(q, k, v, None, flash=True))
    # Only the causal path without a mask is Turbo's.
    zm.attention(q, k, v, _causal(6, 2), flash=True, turbo=True)
    zm.attention(q, k, v, None, flash=False, turbo=True)
    assert len(turbo_attention_calls) == 1


def test_turbo_attention_needs_both_settings_and_zaya_attn_flash(monkeypatch, turbo_attention_calls):
    def stack(enable, use, attn_flash):
        args = SimpleNamespace(enable_primus_turbo=enable, use_turbo_attention=use)
        monkeypatch.setattr(zm, "_megatron_args", lambda: args)
        torch.manual_seed(3)
        return zm.ZayaStack(_cfg(num_layers=4, zaya_attn_flash=attn_flash)).eval()

    for enable, use, attn_flash in ((True, False, True), (False, True, True), (True, True, False)):
        assert not stack(enable, use, attn_flash).layers[0].self_attn.turbo_attn
    turbo = stack(True, True, True)
    assert turbo.layers[0].self_attn.turbo_attn
    flash = stack(False, False, True)
    hidden = torch.randn(6, 2, 32)
    with torch.no_grad():
        got = turbo(hidden, None, _causal(6, 2))
        want = flash(hidden, None, _causal(6, 2))
    assert turbo_attention_calls == [[True, True, True]] * 2  # both CCA stages
    torch.testing.assert_close(got, want)


def _attention_runs(paths, seq=128, batch=2):
    """Output and q/k/v gradients at ZAYA1's attention shape: 8 query heads over 2 KV groups, head_dim 128, bf16."""
    torch.manual_seed(4)
    shapes = ((seq, batch, 8, 128), (seq, batch, 2, 128), (seq, batch, 2, 128))
    inputs = [torch.randn(s, device="cuda", dtype=torch.bfloat16) for s in shapes]
    weights = torch.randn(seq, batch, 8, 128, device="cuda")
    runs = []
    for flash, turbo, mask in paths:
        q, k, v = (x.clone().requires_grad_(True) for x in inputs)
        out = zm.attention(q, k, v, mask, flash=flash, turbo=turbo)
        (out.float() * weights).sum().backward()
        runs.append([out.detach(), q.grad, k.grad, v.grad])
    return runs


def _assert_close_runs(runs):
    for name, a, b in zip(("output", "dq", "dk", "dv"), *runs):
        err = (a.float() - b.float()).norm() / b.float().norm()
        assert err < 1e-2, f"{name}: relative error {err:.3e}"


@pytest.mark.skipif(
    not torch.cuda.is_available() or zm._flash_attn_func is None, reason="needs a GPU and flash_attn"
)
def test_flash_attention_matches_the_dense_mask_path():
    _assert_close_runs(_attention_runs(((True, False, None), (False, False, _causal(128).cuda()))))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU and primus_turbo")
def test_turbo_attention_matches_the_dense_mask_path():
    pytest.importorskip("primus_turbo.pytorch.ops")
    # Over 256 keys: the CI image's GPU_ARCHS="gfx942;gfx950" makes AITER take its MI300X runner for
    # gfx950, and AITER's gfx950 rule for <= 256 keys picks a backward kernel that gives NaN dk on gfx942.
    seq = 512
    runs = _attention_runs(((True, True, None), (False, False, _causal(seq).cuda())), seq=seq)
    assert runs[0][0].is_contiguous()  # the output came back [s, b, h, d] without a copy
    _assert_close_runs(runs)
