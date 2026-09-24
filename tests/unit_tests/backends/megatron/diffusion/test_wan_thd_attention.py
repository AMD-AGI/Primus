# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""The thd (packed) attention path on the Primus-Turbo local spec.

``PrimusTurboLocalAttention`` serves a thd batch from Turbo's varlen kernel and
everything else from the dense one. Three things have to hold for that to be a
layout choice rather than a behaviour change:

  * the two layouts agree numerically, so packing is not a silent regression;
  * only ``qkv_format == "thd"`` reaches the varlen kernel, so every existing
    caller keeps the dense path it had;
  * the Wan blocks pack only when asked, so the default stays dense.

The varlen kernel takes ``max_seqlen_q``/``max_seqlen_kv`` as arguments where TE
derives them from ``cu_seqlens``, so a packing built for TE is not sufficient
here; that is asserted rather than left to fail inside aiter.

NOTE: GPU-only and container-only, for the same reasons as
``test_te_vs_local_spec_attention``: the module under test imports the Megatron
and Primus-Turbo stack.
"""

import argparse

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import set_args

from primus.backends.megatron.core.extensions.primus_turbo_local_spec import (
    PrimusTurboLocalAttention,
)
from primus.backends.megatron.core.models.diffusion.wan.config import WanConfig
from primus.backends.megatron.core.models.diffusion.wan.model import WanTransformer3D

# head_dim 128 is the size aiter serves from its assembly kernels, dense and
# packed alike, and the one Wan runs at; seq > 128 keeps the forward off the
# decode path that prefers CK.
_DIM = 512
_HEADS = 4
_HEAD_DIM = _DIM // _HEADS  # 128
_SEQ = 256
_BATCH = 2


def _build_attention() -> PrimusTurboLocalAttention:
    """The production module, wired the way Megatron's attention.py wires it."""
    set_args(argparse.Namespace(enable_turbo_attention_float8=False))
    config = TransformerConfig(
        num_layers=1,
        hidden_size=_DIM,
        num_attention_heads=_HEADS,
        kv_channels=_HEAD_DIM,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    return build_module(
        PrimusTurboLocalAttention,
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.no_mask,
        attention_type="self",
        softmax_scale=config.softmax_scale,
    )


def _sbhd_inputs(seed=42):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [
        torch.randn(_SEQ, _BATCH, _HEADS, _HEAD_DIM, dtype=torch.bfloat16, generator=generator).cuda()
        for _ in range(3)
    ]


def _packing(seq_len=_SEQ, batch=_BATCH, with_max_seqlen=True):
    cu_seqlens = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda") * seq_len
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=seq_len if with_max_seqlen else None,
        max_seqlen_kv=seq_len if with_max_seqlen else None,
    )


def _to_thd(tensor):
    return tensor.transpose(0, 1).reshape(_BATCH * _SEQ, _HEADS, _HEAD_DIM).contiguous()


def test_packed_and_dense_agree():
    """Packing the batch into the token dim does not change the result.

    Every sequence here is the same length, so the packed batch describes
    exactly the same attention problem as the dense one and the two kernels have
    no licence to disagree.
    """
    attention = _build_attention()
    query, key, value = _sbhd_inputs()

    dense = attention(query, key, value, None, AttnMaskType.no_mask)
    packed = attention(
        _to_thd(query),
        _to_thd(key),
        _to_thd(value),
        None,
        AttnMaskType.no_mask,
        packed_seq_params=_packing(),
    )

    assert dense.shape == (_SEQ, _BATCH, _DIM)
    assert packed.shape == (_BATCH * _SEQ, _DIM)

    packed_as_sbhd = packed.reshape(_BATCH, _SEQ, _DIM).transpose(0, 1)
    torch.testing.assert_close(packed_as_sbhd, dense, rtol=0, atol=0)


def test_only_thd_reaches_the_varlen_kernel():
    """A non-thd packing leaves the dense path exactly as it was.

    ``packed_seq_params`` is threaded through Megatron for several purposes and
    was accepted and ignored here before the thd path existed; anything that is
    not thd has to keep the dense behaviour rather than be reinterpreted as a
    packed batch.
    """
    attention = _build_attention()
    query, key, value = _sbhd_inputs()

    expected = attention(query, key, value, None, AttnMaskType.no_mask)
    sbhd_packing = PackedSeqParams(qkv_format="sbhd")
    actual = attention(query, key, value, None, AttnMaskType.no_mask, packed_seq_params=sbhd_packing)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["max_seqlen_q", "max_seqlen_kv"])
def test_thd_requires_max_seqlen(missing):
    """A TE-shaped packing is rejected with a reason, not left to aiter.

    TE derives the maxima from ``cu_seqlens``; Turbo's varlen kernel takes them
    as arguments, so a packing built for TE is incomplete here.
    """
    attention = _build_attention()
    query, key, value = (_to_thd(t) for t in _sbhd_inputs())

    packing = _packing()
    setattr(packing, missing, None)

    with pytest.raises(ValueError, match=missing):
        attention(query, key, value, None, AttnMaskType.no_mask, packed_seq_params=packing)


def _build_wan_on_meta(**overrides):
    """Structure only: the flag is read in __init__, so no weights are needed."""
    set_args(argparse.Namespace(enable_turbo_attention_float8=False))
    config = WanConfig.wan2_1_t2v_1_3b()
    config.transformer_impl = "local"
    config.use_cpu_initialization = False
    config.perform_initialization = False
    for name, value in overrides.items():
        setattr(config, name, value)
    with torch.device("meta"):
        return WanTransformer3D(config)


@pytest.mark.parametrize("overrides,expected", [({}, False), ({"local_thd_attention": True}, True)])
def test_wan_local_thd_is_opt_in(overrides, expected):
    """The Wan blocks pack only when ``local_thd_attention`` asks them to.

    Packing swaps one aiter kernel family for another, so it is a deliberate
    choice rather than something a run should acquire by upgrading.
    """
    block = _build_wan_on_meta(**overrides).blocks[0]
    assert block.attn1.local_thd is expected
    assert block.attn2.local_thd is expected


def test_wan_local_thd_ignores_the_old_env_var(monkeypatch):
    """The layout is config, so an environment left over from a sweep cannot flip it."""
    monkeypatch.setenv("PRIMUS_WAN_THD_ATTN", "1")

    assert _build_wan_on_meta().blocks[0].attn1.local_thd is False


def test_local_thd_attention_needs_the_local_path():
    config = WanConfig.wan2_1_t2v_1_3b()
    config.local_thd_attention = True

    with pytest.raises(ValueError, match="local_thd_attention"):
        config.validate()
