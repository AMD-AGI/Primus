###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU unit tests for ``PrimusTurboLocalAttention.forward``'s parameter guards.

``forward`` accepts ``attention_bias`` and ``packed_seq_params`` because Megatron's
core-attention protocol says it must, but this class forwards neither: the kernel
call passes ``bias=None``/``alibi_slopes=None`` unconditionally and never builds
cu_seqlens. Both used to be accepted and ignored -- a silent drop, which trains a
different model than the config asks for while the log stays clean. These tests pin
the refusals, and pin that they do NOT fire on the ordinary dense call.

No GPU and no real Turbo build are needed: ``forward`` is invoked unbound against a
lightweight stub, so nothing in ``__init__`` (device queries, process groups,
``get_args``) runs.
"""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.enums import AttnMaskType

from primus.backends.megatron.core.extensions.primus_turbo_local_spec import (
    PrimusTurboLocalAttention,
)


class _Recorder:
    """Stands in for the bound ``attn_func``; records what it was handed."""

    def __init__(self):
        self.calls = []

    def __call__(self, q, k, v, **kwargs):
        self.calls.append({"q": q, "k": k, "v": v, **kwargs})
        return torch.zeros_like(q)


def _stub(attn_func=None):
    """The exact attribute surface ``forward`` reads -- nothing more."""
    return SimpleNamespace(
        attn_func=attn_func if attn_func is not None else _Recorder(),
        softmax_scale=0.125,
        force_contiguous_qkv=False,
        attn_kwargs={},
    )


def _qkv(s=4, b=2, h=2, d=8):
    """sbhd, the layout ``forward`` documents as its input."""
    return [torch.randn(s, b, h, d) for _ in range(3)]


def _call(stub, *, attention_bias=None, packed_seq_params=None, mask_type=AttnMaskType.causal):
    q, k, v = _qkv()
    return PrimusTurboLocalAttention.forward(
        stub,
        q,
        k,
        v,
        attention_mask=None,
        attn_mask_type=mask_type,
        attention_bias=attention_bias,
        packed_seq_params=packed_seq_params,
    )


def test_attention_bias_is_refused_not_dropped():
    with pytest.raises(NotImplementedError, match="attention_bias"):
        _call(_stub(), attention_bias=torch.zeros(1, 2, 4, 4))


def test_the_attention_bias_refusal_names_alibi():
    # ALiBi is the way this actually reaches people: Megatron delivers it as an
    # attention bias, so a user who sets position_embedding_type=alibi lands here
    # without ever typing the word "bias". The message has to say so.
    with pytest.raises(NotImplementedError, match="(?i)alibi"):
        _call(_stub(), attention_bias=torch.zeros(1, 2, 4, 4))


def test_packed_seq_params_is_refused_not_dropped():
    with pytest.raises(NotImplementedError, match="packed sequences"):
        _call(_stub(), packed_seq_params=SimpleNamespace(qkv_format="thd"))


def test_the_packed_refusal_points_at_the_spec_that_implements_it():
    # A refusal that does not say where to go is a dead end; PrimusTurboAttention
    # does implement qkv_format='thd'.
    with pytest.raises(NotImplementedError, match="PrimusTurboAttention"):
        _call(_stub(), packed_seq_params=SimpleNamespace(qkv_format="thd"))


def test_a_packed_seq_params_carrying_nothing_is_still_refused():
    # Emptiness is not permission. An object with no cu_seqlens still means the
    # caller believes it is packing; guessing that it is not is exactly the
    # approximation this guard exists to prevent.
    with pytest.raises(NotImplementedError, match="packed sequences"):
        _call(_stub(), packed_seq_params=SimpleNamespace())


def test_the_ordinary_dense_call_is_untouched():
    rec = _Recorder()
    out = _call(_stub(rec))
    assert len(rec.calls) == 1
    call = rec.calls[0]
    # The guards must not have perturbed the call the kernel actually receives.
    assert call["causal"] is True
    assert call["bias"] is None
    assert call["alibi_slopes"] is None
    assert call["window_size"] == (-1, -1)
    # sbhd in -> sbhd out with heads merged.
    assert out.shape == (4, 2, 2 * 8)


def test_no_mask_still_reaches_the_kernel_as_non_causal():
    rec = _Recorder()
    _call(_stub(rec), mask_type=AttnMaskType.no_mask)
    assert rec.calls[0]["causal"] is False
