###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytest.importorskip("megatron.core")
# The hybrid package imports its layer specs, which need both.
pytest.importorskip("triton")
pytest.importorskip("transformer_engine")

from megatron.core.extensions.transformer_engine import (  # noqa: E402
    TEColumnParallelLinear,
)

from primus.backends.megatron.core.models.hybrid.grouped_kv_mla import (  # noqa: E402
    GroupedKVUpProjection,
    repeat_kv_heads,
)

KV_RANK, PER_HEAD = 128, 32 + 64


def _llama_repeat_kv(hidden, n_rep):
    """transformers' ``repeat_kv``: ``[b, kv, s, d]`` to ``[b, kv * n_rep, s, d]``."""
    batch, kv_heads, seq, head_dim = hidden.shape
    hidden = hidden[:, :, None, :, :].expand(batch, kv_heads, n_rep, seq, head_dim)
    return hidden.reshape(batch, kv_heads * n_rep, seq, head_dim)


def _by_head(kv, heads, head_dim):
    seq, batch = kv.shape[:2]
    return kv.view(seq, batch, heads, head_dim).permute(1, 2, 0, 3)


def test_repeat_kv_heads_matches_llama_repeat_kv():
    kv_heads, repeats, head_dim = 2, 4, 5
    kv = torch.randn(3, 2, kv_heads * head_dim)
    actual = _by_head(repeat_kv_heads(kv, head_dim, repeats), kv_heads * repeats, head_dim)
    assert torch.equal(actual, _llama_repeat_kv(_by_head(kv, kv_heads, head_dim), repeats))


def _plain_linear_init(self, input_size, output_size, *, config, **kwargs):
    nn.Module.__init__(self)
    self.weight = nn.Parameter(torch.randn(output_size, input_size))


def test_grouped_kv_up_projection_stores_kv_heads_and_repeats_them(monkeypatch):
    monkeypatch.setattr(TEColumnParallelLinear, "__init__", _plain_linear_init)
    monkeypatch.setattr(TEColumnParallelLinear, "forward", lambda self, x: (x @ self.weight.t(), None))
    config = SimpleNamespace(num_attention_heads=32)
    proj = GroupedKVUpProjection(KV_RANK, 32 * PER_HEAD, config=config, kv_heads=8, bias=False)
    assert proj.weight.shape == (8 * PER_HEAD, KV_RANK)

    latent = torch.randn(3, 2, KV_RANK)
    out, bias = proj(latent)
    per_kv_head = (latent @ proj.weight.t()).view(3, 2, 8, 1, PER_HEAD)
    assert bias is None
    assert torch.equal(out, per_kv_head.expand(3, 2, 8, 4, PER_HEAD).reshape(3, 2, 32 * PER_HEAD))


def test_grouped_kv_up_projection_reads_hylo_mla_kv_heads(monkeypatch):
    import megatron.training

    monkeypatch.setattr(TEColumnParallelLinear, "__init__", _plain_linear_init)
    monkeypatch.setattr(megatron.training, "get_args", lambda: SimpleNamespace(hylo_mla_kv_heads=8))
    config = SimpleNamespace(num_attention_heads=32)
    proj = GroupedKVUpProjection(KV_RANK, 32 * PER_HEAD, config=config)
    assert (proj.weight.shape[0], proj.kv_repeats) == (8 * PER_HEAD, 4)
    with pytest.raises(ValueError, match="hylo_mla_kv_heads"):
        GroupedKVUpProjection(KV_RANK, 32 * PER_HEAD, config=config, kv_heads=5)
