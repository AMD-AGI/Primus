###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU contract + GPU-gated parity tests for FLAFlashAttention.forward (PRPUNDIT-4).

The wrapper owns Megatron [s,b,h,d] -> contiguous flash-attn [b,s,h,d], the
dropout_p=0.0 / causal=True / softmax_scale forwarding, the inverse repack to
[s,b,h*d_v], and the packed-seq / non-causal guards. A recording stub pins those
mechanics without hardware. The ROCm test compares the real flash_attn_func path
against an independent float32 causal SDPA reference.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")

from megatron.core.transformer.enums import AttnMaskType

import primus.backends.megatron.core.transformer.fla_flash_attention as ffa

_SEQ, _BATCH, _HEADS, _DIM, _DIM_V = 5, 3, 2, 4, 6
_SOFTMAX_SCALE = 0.125
_GPU_SEQ, _GPU_BATCH, _GPU_HEADS, _GPU_DIM = 7, 3, 2, 128
# BF16 flash-attn vs an independent FP32 causal SDPA reference.
_GPU_ATOL = 2e-2
_GPU_RTOL = 5e-2


def _non_causal_mask():
    return next(member for member in AttnMaskType if member != AttnMaskType.causal)


@pytest.fixture
def silence_banner(monkeypatch):
    monkeypatch.setattr(ffa, "_BANNER_PRINTED", True)
    monkeypatch.setattr(ffa, "_flash_attn_func", None)


def _install_recording_stub(monkeypatch):
    calls = []

    def stub(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
        calls.append(
            {
                "q": q,
                "k": k,
                "v": v,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "kwargs": kwargs,
            }
        )
        b, s, h, _ = q.shape
        d_v = v.shape[-1]
        b_idx = torch.arange(b, dtype=q.dtype, device=q.device).view(b, 1, 1, 1)
        s_idx = torch.arange(s, dtype=q.dtype, device=q.device).view(1, s, 1, 1)
        h_idx = torch.arange(h, dtype=q.dtype, device=q.device).view(1, 1, h, 1)
        d_idx = torch.arange(d_v, dtype=q.dtype, device=q.device).view(1, 1, 1, d_v)
        out = (b_idx + 1) * 1000 + (s_idx + 1) * 100 + (h_idx + 1) * 10 + (d_idx + 1)
        calls[-1]["out"] = out
        return out

    monkeypatch.setattr(ffa, "_load_flash_attn", lambda: stub)
    return calls


def _make_module(attn_mask_type=AttnMaskType.causal, softmax_scale=_SOFTMAX_SCALE):
    return ffa.FLAFlashAttention(
        config=SimpleNamespace(),
        layer_number=1,
        attn_mask_type=attn_mask_type,
        softmax_scale=softmax_scale,
    )


def _cpu_qkv():
    # Distinct values; s!=b and h!=d_v so a swapped transpose or view cannot hide.
    query = torch.arange(_SEQ * _BATCH * _HEADS * _DIM, dtype=torch.float32).reshape(
        _SEQ, _BATCH, _HEADS, _DIM
    )
    key = query + 0.25
    value = torch.arange(_SEQ * _BATCH * _HEADS * _DIM_V, dtype=torch.float32).reshape(
        _SEQ, _BATCH, _HEADS, _DIM_V
    )
    return query, key, value


def test_forward_records_contiguous_bshd_and_repacks_sbhd(silence_banner, monkeypatch):
    calls = _install_recording_stub(monkeypatch)
    module = _make_module()
    query, key, value = _cpu_qkv()

    out = module.forward(query, key, value)

    assert len(calls) == 1
    rec = calls[0]
    for name, src in (("q", query), ("k", key), ("v", value)):
        got = rec[name]
        expected = src.transpose(0, 1)
        assert got.shape == expected.shape, name
        assert got.is_contiguous(), name
        torch.testing.assert_close(got, expected.contiguous(), atol=0, rtol=0)
    assert rec["dropout_p"] == 0.0
    assert rec["causal"] is True
    assert rec["softmax_scale"] == _SOFTMAX_SCALE

    stub_out = rec["out"]
    assert out.shape == (_SEQ, _BATCH, _HEADS * _DIM_V)
    for seq_i in range(_SEQ):
        for batch_i in range(_BATCH):
            torch.testing.assert_close(
                out[seq_i, batch_i],
                stub_out[batch_i, seq_i].reshape(_HEADS * _DIM_V),
                atol=0,
                rtol=0,
            )


def test_forward_rejects_packed_seq_params_before_stub(silence_banner, monkeypatch):
    calls = _install_recording_stub(monkeypatch)
    module = _make_module()
    query, key, value = _cpu_qkv()

    with pytest.raises(NotImplementedError, match="packed_seq_params"):
        module.forward(query, key, value, packed_seq_params=object())
    assert calls == []


def test_forward_rejects_explicit_non_causal_mask_before_stub(silence_banner, monkeypatch):
    calls = _install_recording_stub(monkeypatch)
    module = _make_module()
    query, key, value = _cpu_qkv()

    with pytest.raises(NotImplementedError, match="causal masking"):
        module.forward(query, key, value, attn_mask_type=_non_causal_mask())
    assert calls == []


def test_forward_rejects_inherited_non_causal_mask_before_stub(silence_banner, monkeypatch):
    calls = _install_recording_stub(monkeypatch)
    module = _make_module(attn_mask_type=_non_causal_mask())
    query, key, value = _cpu_qkv()

    with pytest.raises(NotImplementedError, match="causal masking"):
        module.forward(query, key, value)
    assert calls == []


def test_forward_argument_causal_overrides_inherited_non_causal(silence_banner, monkeypatch):
    calls = _install_recording_stub(monkeypatch)
    module = _make_module(attn_mask_type=_non_causal_mask())
    query, key, value = _cpu_qkv()

    out = module.forward(query, key, value, attn_mask_type=AttnMaskType.causal)
    assert len(calls) == 1
    assert out.shape == (_SEQ, _BATCH, _HEADS * _DIM_V)


def _causal_sdpa_reference(query, key, value, scale):
    """Independent FP32 causal SDPA; inputs [s,b,h,d], output [s,b,h*d_v]."""
    q = query.float().permute(1, 2, 0, 3)
    k = key.float().permute(1, 2, 0, 3)
    v = value.float().permute(1, 2, 0, 3)
    seq = q.shape[-2]
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    causal = torch.ones(seq, seq, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v).permute(2, 0, 1, 3).contiguous()
    seq, batch, heads, dim_v = out.shape
    return out.view(seq, batch, heads * dim_v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_forward_rocm_parity_against_causal_sdpa_reference(silence_banner, monkeypatch):
    pytest.importorskip("flash_attn")
    from tests.utils import install_aiter_deepbind_hook

    install_aiter_deepbind_hook()

    recorded = []
    real = ffa._load_flash_attn()

    def wrapped(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
        recorded.append(
            {
                "q": q,
                "k": k,
                "v": v,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
            }
        )
        return real(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, **kwargs)

    monkeypatch.setattr(ffa, "_load_flash_attn", lambda: wrapped)
    monkeypatch.setattr(ffa, "_flash_attn_func", wrapped)

    generator = torch.Generator(device="cpu").manual_seed(0)
    query = torch.randn(
        _GPU_SEQ, _GPU_BATCH, _GPU_HEADS, _GPU_DIM, dtype=torch.bfloat16, generator=generator
    ).cuda()
    key = torch.randn(
        _GPU_SEQ, _GPU_BATCH, _GPU_HEADS, _GPU_DIM, dtype=torch.bfloat16, generator=generator
    ).cuda()
    value = torch.randn(
        _GPU_SEQ, _GPU_BATCH, _GPU_HEADS, _GPU_DIM, dtype=torch.bfloat16, generator=generator
    ).cuda()

    module = _make_module(softmax_scale=_SOFTMAX_SCALE)
    got = module.forward(query, key, value)
    ref = _causal_sdpa_reference(query, key, value, _SOFTMAX_SCALE).to(dtype=got.dtype)

    assert got.shape == (_GPU_SEQ, _GPU_BATCH, _GPU_HEADS * _GPU_DIM)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), ref.float(), atol=_GPU_ATOL, rtol=_GPU_RTOL)

    assert len(recorded) == 1
    rec = recorded[0]
    for name, src in (("q", query), ("k", key), ("v", value)):
        assert rec[name].is_contiguous()
        assert rec[name].shape == (src.shape[1], src.shape[0], src.shape[2], src.shape[3])
    assert rec["dropout_p"] == 0.0
    assert rec["causal"] is True
    assert rec["softmax_scale"] == _SOFTMAX_SCALE
