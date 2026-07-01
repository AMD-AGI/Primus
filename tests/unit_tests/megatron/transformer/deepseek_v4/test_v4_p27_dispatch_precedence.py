###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Runtime dispatch of :class:`DeepseekV4Attention` on the unified selectors.

Mock-based gate for the ``use_v4_attention_backend`` (dense/HCA) and
``use_v4_csa_attention_backend`` (CSA) string selectors. For each
``(compress_ratio, backend) -> expected kernel`` combination we patch the
candidate kernels and assert exactly one fires. ``__init__`` is bypassed via
``__new__`` (CPU-only) so we exercise the real ``forward`` /
``_attention_backend_forward`` / ``_csa_forward`` / ``_log_kernel_choice`` code.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

# isort: off
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (  # noqa: F401
    DeepSeekV4TransformerConfig,
)
import primus.backends.megatron.core.transformer.deepseek_v4_attention as v4_attn_mod
from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
    DeepseekV4Attention,
)

# isort: on


def _make_bare_attn(
    *,
    compress_ratio,
    attn_backend="triton_v1",
    csa_backend="triton_v1",
    use_core_attention=False,
    layer_number=0,
):
    attn = DeepseekV4Attention.__new__(DeepseekV4Attention)
    attn.compress_ratio = int(compress_ratio)
    attn.layer_number = int(layer_number)
    attn._use_core_attention = bool(use_core_attention)
    attn._attn_backend = attn_backend
    attn._csa_backend = csa_backend
    return attn


# ---------------------------------------------------------------------------
# _log_kernel_choice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cr,use_core,attn_be,csa_be,expected",
    [
        pytest.param(0, True, "triton_v1", "triton_v1", "core_attention", id="cr0_turbo"),
        pytest.param(0, False, "gluon", "triton_v1", "dense attention backend = gluon", id="cr0_gluon"),
        pytest.param(
            0, False, "triton_v1", "triton_v1", "dense attention backend = triton_v1", id="cr0_triton_v1"
        ),
        pytest.param(
            128, False, "triton_v2", "triton_v1", "HCA attention backend = triton_v2", id="cr128_v2"
        ),
        pytest.param(4, False, "triton_v1", "gluon", "CSA attention backend = gluon", id="cr4_gluon"),
        pytest.param(4, False, "triton_v1", "eager", "CSA attention backend = eager", id="cr4_eager"),
    ],
)
def test_log_kernel_choice(cr, use_core, attn_be, csa_be, expected, caplog):
    attn = _make_bare_attn(
        compress_ratio=cr,
        attn_backend=attn_be,
        csa_backend=csa_be,
        use_core_attention=use_core,
        layer_number=17,
    )
    caplog.set_level(logging.INFO, logger=v4_attn_mod.__name__)
    attn._log_kernel_choice()
    assert expected in caplog.text, f"expected {expected!r} in log, got:\n{caplog.text}"


def test_log_kernel_choice_format(caplog):
    attn = _make_bare_attn(compress_ratio=0, layer_number=42)
    caplog.set_level(logging.INFO, logger=v4_attn_mod.__name__)
    attn._log_kernel_choice()
    assert "[V4-attn]" in caplog.text and "Layer 42" in caplog.text
    assert "cr=0" in caplog.text and "kernel = " in caplog.text


# ---------------------------------------------------------------------------
# dense/HCA dispatch: _attention_backend_forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attn_be,fired",
    [
        ("gluon", "v4_attention_gluon"),
        ("triton_v2", "v4_attention_v2"),
    ],
)
def test_dense_backend_forward_dispatch(attn_be, fired):
    attn = _make_bare_attn(compress_ratio=0, attn_backend=attn_be)
    attn.attn_sink = None
    attn.attn_sliding_window = 128
    attn.attn_dropout = 0.0
    attn.training = False
    attn._attention_scale = MagicMock(return_value=0.125)
    sentinel = object()
    with patch.object(v4_attn_mod, fired, return_value=sentinel) as mock_fired:
        out = DeepseekV4Attention._attention_backend_forward(
            attn,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            additive_mask=None,
            hca_local_seqlen=0,
            S=8,
            device="cpu",
            dtype=None,
        )
    assert out is sentinel
    mock_fired.assert_called_once()


def test_dense_backend_forward_triton_v1_calls_launcher_helper():
    attn = _make_bare_attn(compress_ratio=0, attn_backend="triton_v1")
    attn.attn_sliding_window = 128
    attn._attention_forward_via_v4_triton = MagicMock(return_value="TRITONV1")
    out = DeepseekV4Attention._attention_backend_forward(
        attn,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        additive_mask=None,
        hca_local_seqlen=0,
        S=8,
        device="cpu",
        dtype=None,
    )
    assert out == "TRITONV1"
    attn._attention_forward_via_v4_triton.assert_called_once()


def test_via_v4_triton_helper_calls_v4_attention_v1():
    attn = _make_bare_attn(compress_ratio=0, attn_backend="triton_v1")
    attn.attn_sink = None
    attn.attn_sliding_window = 0
    attn.attn_dropout = 0.0
    attn.training = False
    attn._attention_scale = MagicMock(return_value=0.125)
    sentinel = object()
    with patch.object(v4_attn_mod, "v4_attention_v1", return_value=sentinel) as mock_kernel:
        out = DeepseekV4Attention._attention_forward_via_v4_triton(
            attn,
            q=MagicMock(),
            k=MagicMock(),
            v=MagicMock(),
            attn_mask=MagicMock(),
        )
    assert out is sentinel
    mock_kernel.assert_called_once()


# ---------------------------------------------------------------------------
# CSA dispatch: _csa_forward
# ---------------------------------------------------------------------------


def _stub_csa_inputs():
    q_bh = MagicMock()
    q_bh.shape = (1, 2, 3, 4)  # B, H, S, Dh
    return MagicMock(), q_bh, MagicMock(), MagicMock(), MagicMock()


@pytest.mark.parametrize(
    "csa_be,fired,builds_gathered",
    [
        ("gluon", "v4_csa_attention_gluon", False),
        ("triton_v2", "v4_csa_attention_v2", False),
        ("triton_v1", "v4_csa_attention_v1", False),
        ("eager", "eager_v4_csa_attention", True),
    ],
)
def test_csa_forward_dispatch(csa_be, fired, builds_gathered):
    attn = _make_bare_attn(compress_ratio=4, csa_backend=csa_be)
    attn.attn_sink = None
    attn.attn_sliding_window = 128
    attn.attn_dropout = 0.0
    attn.training = False
    attn._attention_scale = MagicMock(return_value=0.125)
    pool_mock = MagicMock()
    pool_mock.shape = (1, 8, 4)  # [B, P, Dh]
    pool_mock.unsqueeze.return_value.expand.return_value = MagicMock()
    attn._build_compressed_pool = MagicMock(return_value=pool_mock)
    topk = MagicMock()
    topk.shape = (1, 3, 4)  # [B, S, K]
    topk.clamp.return_value = MagicMock()
    topk.__ge__ = MagicMock(return_value=MagicMock())
    attn.indexer = MagicMock(return_value=(topk, MagicMock()))

    sentinel = object()
    with patch.object(v4_attn_mod, fired, return_value=sentinel) as mock_fired, patch(
        "torch.gather", return_value=MagicMock()
    ), patch("torch.where", return_value=MagicMock()):
        _, q_bh, k_local_bh, v_local_bh, local_mask = _stub_csa_inputs()
        out = DeepseekV4Attention._csa_forward(
            attn,
            hidden=MagicMock(),
            q_bh=q_bh,
            k_local_bh=k_local_bh,
            v_local_bh=v_local_bh,
            local_mask=local_mask,
        )
    assert out is sentinel
    mock_fired.assert_called_once()
