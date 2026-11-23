###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``_NopeKeyValue`` (the NoPE MLA key / value builder) against upstream's split / cat / expand chain.

Upstream (``multi_latent_attention.py``) builds ``key = cat(k_no_pe, k_pos_emb expanded over the heads)`` and
``value = kv[..., qk:].contiguous()`` from ``kv`` and the MQA-shared ``k_pos_emb``; the slice-copy version must give
the same key and value and the same gradients of ``kv`` and ``k_pos_emb``, bit for bit.
"""

import pytest

torch = pytest.importorskip("torch")

# kimi_k3/__init__.py imports megatron.core (kimi_delta_attention.py)
pytest.importorskip(
    "megatron.core.transformer.module",
    reason="megatron.core is not importable in this environment",
)

try:
    from primus.backends.megatron.core.transformer.kimi_k3.kimi_k3_mla_attention import (
        _NopeKeyValue,
    )
except RuntimeError as error:
    # the MLA module's import chain needs a GPU (as in test_model_registry.py)
    if torch.cuda.is_available():
        raise
    pytest.skip(f"importing kimi_k3_mla_attention needs a GPU: {error}", allow_module_level=True)

QK, V, PE = 128, 128, 64


def _upstream(kv, k_pos_emb, heads):
    k_no_pe, value = torch.split(kv, [QK, V], dim=-1)
    k_pos_emb = torch.unsqueeze(k_pos_emb, -2).expand(-1, -1, heads, -1)
    key = torch.cat([k_no_pe, k_pos_emb], dim=-1)
    return key.contiguous(), value.contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_matches_upstream_bit_for_bit(dtype):
    s, b, heads = 33, 2, 6
    g = torch.Generator().manual_seed(0)
    kv = torch.randn(s, b, heads, QK + V, generator=g).to(dtype)
    pe = torch.randn(s, b, PE, generator=g).to(dtype)
    d_key = torch.randn(s, b, heads, QK + PE, generator=g).to(dtype)
    d_value = torch.randn(s, b, heads, V, generator=g).to(dtype)

    outs, grads = [], []
    for fn in (lambda a, c: _upstream(a, c, heads), lambda a, c: _NopeKeyValue.apply(a, c, QK)):
        a, c = kv.clone().requires_grad_(), pe.clone().requires_grad_()
        key, value = fn(a, c)
        torch.autograd.backward((key, value), (d_key, d_value))
        outs.append((key.detach(), value.detach()))
        grads.append((a.grad, c.grad))

    for x, y in zip(outs[0] + grads[0], outs[1] + grads[1]):
        assert x.shape == y.shape and x.is_contiguous() == y.is_contiguous()
        assert torch.equal(x, y)


def test_value_is_contiguous_and_key_owns_its_storage():
    kv = torch.randn(4, 1, 2, QK + V)
    key, value = _NopeKeyValue.apply(kv, torch.randn(4, 1, PE), QK)
    assert key.is_contiguous() and value.is_contiguous()
    assert key.untyped_storage().data_ptr() != kv.untyped_storage().data_ptr()
