###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU tests: MuseSparkSelfAttention's fused gated QKV + Q / K norm + RoPE (``muse_spark_fused_qk_norm_rope``,
Primus-Turbo's ``fused_qkv_rmsnorm_rope``) against Megatron's split / norm / RoPE."""

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from megatron.core.models.common.embeddings.rope_utils import (  # noqa: E402
    _apply_rotary_pos_emb_bshd,
)

from primus.backends.megatron.core.transformer.muse_spark.attention import (  # noqa: E402
    MuseSparkQKNorm,
    MuseSparkSelfAttention,
)

EPS = 1e-5
DEV = torch.device("cuda")


def _snr(got, want):
    want = want.detach().float()
    noise = float((got.detach().float() - want).norm())
    return math.inf if noise == 0 else -20 * math.log10(noise / float(want.norm()))


# ---------------------------------------------------------------- fused Q / K norm + RoPE
class _Linear(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = torch.nn.Parameter(w)

    def forward(self, x):
        return x @ self.w.t(), None


def _attention(G, H, D, hidden, no_rope_freq=None):
    a = object.__new__(MuseSparkSelfAttention)
    torch.nn.Module.__init__(a)
    a.config = SimpleNamespace(
        test_mode=False, num_query_groups=G, no_rope_freq=no_rope_freq, rotary_interleaved=False,
        apply_rope_fusion=False, multi_latent_attention=False,
    )  # fmt: skip
    a.attention_type, a.world_size, a.layer_number = "self", 1, 1
    a.pg_collection = SimpleNamespace(cp=SimpleNamespace(size=lambda: 1, rank=lambda: 0))
    a.num_attention_heads_per_partition, a.num_query_groups_per_partition = G * H, G
    a.hidden_size_per_attention_head = D
    a.linear_qkv = _Linear((torch.randn(G * (2 * H + 2) * D, hidden, device=DEV) / hidden**0.5).bfloat16())
    a.q_layernorm = MuseSparkQKNorm(None, D, eps=EPS)
    a.k_layernorm = MuseSparkQKNorm(None, D, eps=EPS)
    a.qk_norm_rope_fused = True
    a._qk_norm_rope = None
    return a


def _freqs(S, D):
    inv = 1.0 / (500000.0 ** (torch.arange(0, D, 2, device=DEV, dtype=torch.float32) / D))
    f = torch.outer(torch.arange(S, device=DEV, dtype=torch.float32), inv)
    return torch.cat([f, f], dim=-1)[:, None, None, :]


def _rope_rounded_once(t, freqs):
    """Rotate-half RoPE from fp32 cos / sin, rounded once per output (the fused op's contract)."""
    cos, sin = freqs.cos(), freqs.sin()
    lo, hi = t.float().chunk(2, dim=-1)
    rot = torch.cat((-hi, lo), dim=-1)
    return (t.float() * cos + rot * sin).to(t.dtype)


@pytest.mark.parametrize("rope", [True, False])
def test_fused_qkv_matches_megatron_split_norm_rope(rope):
    torch.manual_seed(0)
    S, B, G, H, D, hidden = 256, 2, 4, 8, 64, 512
    a = _attention(G, H, D, hidden)
    freqs = _freqs(S, D)
    h0 = torch.randn(S, B, hidden, device=DEV).bfloat16()
    outs, grads = [], []
    # Megatron's split / norm / unfused RoPE, the same with the RoPE rounded once, then the fused path
    for mode in ("megatron", "rounded_once", "fused"):
        a.linear_qkv.w.grad = None
        h = h0.clone().requires_grad_()
        a._qk_norm_rope = (freqs if rope else None, rope) if mode == "fused" else None
        q, k, v, gate = a.get_query_key_value_tensors(h, output_gate=True)
        a._qk_norm_rope = None
        if rope and mode != "fused":
            rot = _apply_rotary_pos_emb_bshd if mode == "megatron" else _rope_rounded_once
            q, k = rot(q, freqs), rot(k, freqs)
        torch.manual_seed(1)
        loss = sum((t.float() * torch.randn_like(t.float())).sum() for t in (q, k, v, gate))
        loss.backward()
        outs.append((q, k, v, gate))
        grads.append((h.grad, a.linear_qkv.w.grad.clone()))
    (q0, k0, v0, g0), (qr, kr, _, _), (q1, k1, v1, g1) = outs
    assert q1.shape == q0.shape and k1.shape == k0.shape and g1.shape == g0.shape
    assert torch.equal(v1, v0) and torch.equal(g1, g0)
    for got, once, megatron in ((q1, qr, q0), (k1, kr, k0)):
        assert float((got != once).float().mean()) <= 1e-4
        assert _snr(got, megatron) >= (45 if rope else 60)
    for i in (0, 1):
        assert _snr(grads[2][i], grads[0][i]) >= 45


def test_fused_qkv_context_selection():
    G, H, D = 2, 4, 64
    freqs = _freqs(16, D)
    a = _attention(G, H, D, 64, no_rope_freq=[0])
    assert a._qk_norm_rope_context((), {"rotary_pos_emb": freqs}) == (freqs, True)
    assert a._qk_norm_rope_context((), {"rotary_pos_emb": (freqs, freqs)}) == (freqs, True)
    assert a._qk_norm_rope_context((), {"rotary_pos_emb": (freqs, freqs.clone())}) is None
    assert a._qk_norm_rope_context((), {"rotary_pos_emb": freqs, "packed_seq_params": object()}) is None
    assert a._qk_norm_rope_context((None,), {"rotary_pos_emb": freqs}) is None  # positional call
    a.config.no_rope_freq = [1]
    assert a._qk_norm_rope_context((), {"rotary_pos_emb": None}) == (None, False)
    a.qk_norm_rope_fused = False
    assert a._qk_norm_rope_context((), {"rotary_pos_emb": freqs}) is None


def test_turbo_without_the_gate_block_keeps_megatrons_path(monkeypatch):
    """A Primus-Turbo whose fused_qkv_rmsnorm_rope still needs the norm weights falls back, not raises."""
    import primus_turbo.pytorch.kernels.rope.qk_rmsnorm_rope_impl as impl

    def old_shape_error(qkv, q_gamma, k_gamma, freqs, split_sizes):
        return None if q_gamma.shape else "unreachable"  # q_gamma is None: AttributeError, as an older Turbo

    monkeypatch.setattr(impl, "qk_rmsnorm_rope_shape_error", old_shape_error)
    S, B, G, H, D, hidden = 32, 1, 2, 4, 64, 128
    a = _attention(G, H, D, hidden)
    freqs = _freqs(S, D)
    h = torch.randn(S, B, hidden, device=DEV).bfloat16()
    a._qk_norm_rope = (freqs, True)
    q, k, v, gate = a.get_query_key_value_tensors(h, output_gate=True)
    a._qk_norm_rope = None
    q0, k0, _, _ = a.get_query_key_value_tensors(h, output_gate=True)
    torch.testing.assert_close(q, _apply_rotary_pos_emb_bshd(q0, freqs), rtol=0, atol=0)
    torch.testing.assert_close(k, _apply_rotary_pos_emb_bshd(k0, freqs), rtol=0, atol=0)
