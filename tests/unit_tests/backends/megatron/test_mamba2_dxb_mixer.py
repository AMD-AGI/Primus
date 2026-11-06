###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import contextlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")
# The hybrid package imports its layer specs, which need both.
pytest.importorskip("triton")
pytest.importorskip("transformer_engine")

import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer import TransformerConfig

from primus.backends.megatron.core.models.hybrid import mamba2_dxb_mixer
from primus.backends.megatron.core.models.hybrid.mamba2_dxb_mixer import (
    Mamba2DxbMixer,
    Mamba2DxbMixerSubmodules,
    repeat_groups,
)


def _llama_repeat_kv(t, width, repeats):
    """MambaInLlama's path: ``(g w) -> g l w``, Llama ``repeat_kv``, then ``h l w -> l h w``."""
    batch, length, channels = t.shape
    t = t.view(batch, length, channels // width, width).permute(0, 2, 1, 3)
    t = t[:, :, None].expand(batch, t.shape[1], repeats, length, width)
    return t.reshape(batch, -1, length, width).permute(0, 2, 1, 3)


def test_repeat_groups_matches_llama_repeat_kv():
    t = torch.randn(2, 5, 3 * 4)
    assert torch.equal(repeat_groups(t, 4, 2), _llama_repeat_kv(t, 4, 2))
    heads = repeat_groups(torch.arange(8.0).view(1, 1, 8), 4, 2)
    assert torch.equal(heads[0, 0, 3], torch.arange(4.0, 8.0))


class _Linear(nn.Linear):
    """Stand-in for the TE projections: ``build_module`` keywords in, ``(output, bias)`` out."""

    def __init__(self, in_features, out_features, *, init_method, bias, **kwargs):
        super().__init__(in_features, out_features, bias=bias)
        init_method(self.weight)

    def forward(self, x):
        return super().forward(x), None


def _recurrent_forward(mixer, hidden):
    """The grouped Mamba-2 layer, one token and one head at a time."""
    u = hidden.transpose(0, 1)
    batch, length, _ = u.shape
    d_inner, d_xb, nheads, d_state = mixer.d_inner, mixer.d_xb, mixer.nheads, mixer.d_state
    z, xBC, dt = (u @ mixer.in_proj.weight.T).split([d_inner, d_inner + 2 * d_xb, nheads], dim=-1)
    xBC = F.conv1d(
        xBC.transpose(1, 2),
        mixer.conv1d.weight,
        mixer.conv1d.bias,
        padding=mixer.d_conv - 1,
        groups=xBC.shape[-1],
    )
    x, B, C = F.silu(xBC[..., :length]).transpose(1, 2).split([d_xb, d_xb, d_inner], dim=-1)
    dt = F.softplus(dt + mixer.dt_bias)
    A = -torch.exp(mixer.A_log.float())

    y = torch.zeros(batch, length, nheads, d_state, device=u.device)
    for head in range(nheads):
        group = slice((head // mixer.repeats) * d_state, (head // mixer.repeats + 1) * d_state)
        own = slice(head * d_state, (head + 1) * d_state)
        state = torch.zeros(batch, d_state, d_state, device=u.device)
        for t in range(length):
            step = dt[:, t, head, None, None]
            state = torch.exp(step * A[head]) * state + step * x[:, t, group, None] * B[:, t, None, group]
            y[:, t, head] = torch.einsum("bpn,bn->bp", state, C[:, t, own]) + mixer.D[head] * x[:, t, group]

    gated = (y.flatten(2) * F.silu(z)).view(batch, length, mixer.ngroups, -1)
    gated = gated * torch.rsqrt(gated.pow(2).mean(-1, keepdim=True) + 1e-5)
    out = gated.flatten(2) * mixer.norm.weight
    return (out @ mixer.out_proj.weight.T).transpose(0, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mamba-ssm kernels need a GPU")
# in_proj widths 52 and 104; causal_conv1d's channel-last kernel only takes widths divisible by 8.
@pytest.mark.parametrize("groups, d_xb", [(4, 8), (8, 16)])
def test_mixer_matches_grouped_mamba2_recurrence(monkeypatch, groups, d_xb):
    pytest.importorskip("mamba_ssm")
    monkeypatch.setattr(
        mamba2_dxb_mixer, "get_cuda_rng_tracker", lambda: SimpleNamespace(fork=contextlib.nullcontext)
    )
    torch.manual_seed(0)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        mamba_state_dim=4,
        mamba_head_dim=4,
        mamba_num_groups=groups,
    )
    single = SimpleNamespace(size=lambda: 1)
    mixer = Mamba2DxbMixer(
        config,
        Mamba2DxbMixerSubmodules(in_proj=_Linear, out_proj=_Linear),
        d_model=16,
        d_xb=d_xb,
        pg_collection=SimpleNamespace(tp=single, cp=single),
    ).cuda()
    assert (mixer.d_inner, mixer.repeats) == (4 * groups, 2)
    hidden = torch.randn(7, 2, 16, device="cuda")

    with torch.no_grad():
        out, bias = mixer(hidden)
        expected = _recurrent_forward(mixer, hidden)

    assert bias is None
    torch.testing.assert_close(out, expected, rtol=5e-3, atol=5e-3)
