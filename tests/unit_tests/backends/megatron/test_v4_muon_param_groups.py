###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the DeepSeek-V4 Muon parameter split (report §9.5.1) and the
Newton-Schulz coefficient resolution (hybrid 'deepseekv4' schedule).

CPU-friendly: exercises the pure helpers
``primus...optimizer.moun._param_goes_to_muon`` and
``_resolve_muon_coefficient_type`` directly, so it does not require the
``emerging_optimizers`` package, a GPU, or distributed init.
"""

from __future__ import annotations

import types

import pytest
import torch

from primus.backends.megatron.core.optimizer.moun import (
    _param_goes_to_muon,
    _resolve_muon_coefficient_type,
)


def _p(shape, *, embedding_or_output=False):
    t = torch.zeros(*shape) if shape else torch.zeros(())
    param = torch.nn.Parameter(t)
    if embedding_or_output:
        param.is_embedding_or_output_parameter = True
    return param


# ---------------------------------------------------------------------------
# §9.5.1 parameter split: 2-D matrices -> Muon; embedding/output + 1-D -> AdamW
# ---------------------------------------------------------------------------


def test_2d_weight_matrices_go_to_muon():
    # attention / MoE expert / mHC mixing matrix (HyperMixer.fn) are 2-D.
    assert _param_goes_to_muon(_p((4096, 1024))) is True  # attn proj
    assert _param_goes_to_muon(_p((2048, 4096))) is True  # MoE expert fc
    assert _param_goes_to_muon(_p((24, 16384))) is True   # HyperMixer.fn (K*D->(2+K)K)


def test_embedding_and_output_go_to_adamw():
    # Even though embeddings / output head are 2-D, the flag forces AdamW.
    assert _param_goes_to_muon(_p((129280, 4096), embedding_or_output=True)) is False
    assert _param_goes_to_muon(_p((4096, 129280), embedding_or_output=True)) is False


def test_1d_params_go_to_adamw():
    # RMSNorm weight, mHC static bias (HyperMixer.base / HyperHead.base),
    # mHC gating scale (HyperMixer.scale shape [3]), and biases are 1-D.
    assert _param_goes_to_muon(_p((4096,))) is False   # RMSNorm weight
    assert _param_goes_to_muon(_p((24,))) is False     # HyperMixer.base
    assert _param_goes_to_muon(_p((3,))) is False      # HyperMixer.scale
    assert _param_goes_to_muon(_p((4,))) is False      # HyperHead.base (K)


def test_scalar_params_go_to_adamw():
    # 0-dim (scalar) params must not reach Newton-Schulz (which computes
    # grad.size(-2)); they belong in AdamW.
    assert _param_goes_to_muon(_p(())) is False


# ---------------------------------------------------------------------------
# Newton-Schulz coefficient resolution
# ---------------------------------------------------------------------------


class _FakeV4Config:
    """Stand-in whose class name matches the V4 transformer config the
    resolver auto-detects (it checks ``type(...).__name__``)."""


_FakeV4Config.__name__ = "DeepSeekV4TransformerConfig"


def _chunk(cfg):
    return types.SimpleNamespace(config=cfg)


def test_v4_autoselects_deepseekv4_when_default():
    cfg = types.SimpleNamespace(muon_coefficient_type="quintic")
    chunks = [_chunk(_FakeV4Config())]
    assert _resolve_muon_coefficient_type(cfg, chunks) == "deepseekv4"


def test_non_v4_keeps_quintic_default():
    cfg = types.SimpleNamespace(muon_coefficient_type="quintic")
    chunks = [_chunk(types.SimpleNamespace())]  # not a V4 config
    assert _resolve_muon_coefficient_type(cfg, chunks) == "quintic"


def test_explicit_coefficient_type_wins_over_autoselect():
    cfg = types.SimpleNamespace(muon_coefficient_type="polar_express")
    chunks = [_chunk(_FakeV4Config())]
    assert _resolve_muon_coefficient_type(cfg, chunks) == "polar_express"


def test_missing_field_defaults_to_quintic():
    cfg = types.SimpleNamespace()  # no muon_coefficient_type attr
    chunks = [_chunk(types.SimpleNamespace())]
    assert _resolve_muon_coefficient_type(cfg, chunks) == "quintic"
