###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The CSA selector path stays high precision unless explicitly opted in.

The open-source reference wraps the DeepSeek-V4 compressor and the indexer's
weight projection in an fp8-disabled context so they keep running in BF16 even
when the enclosing layer is under an FP8 autocast: the indexer decides *which*
512 compressed entries each query attends to, so quantization error there
changes the selection rather than nudging a value.

These tests pin the two ways that contract can be lost:

* the indexer projections must not be quantized as a side effect of enabling
  FP8 on the *attention* projections -- those are separate decisions;
* the compressor must have no quantized branch at all.

Torch-only and CPU-friendly: the FP8 branches are gated on env knobs read at
call time, so the gating can be exercised without a GPU.
"""

from __future__ import annotations

import inspect
import os

import pytest

torch = pytest.importorskip("torch")

from primus.backends.megatron.core.transformer import (  # noqa: E402
    indexer as indexer_mod,
)
from primus.backends.megatron.core.transformer.compressor import (  # noqa: E402
    Compressor,
)

# ---------------------------------------------------------------------------
# Indexer projections
# ---------------------------------------------------------------------------


def test_indexer_proj_fp8_is_off_by_default(monkeypatch):
    monkeypatch.delenv("PRIMUS_V4_FP8_INDEXER_PROJ", raising=False)
    monkeypatch.delenv("PRIMUS_V4_FP8_ATTN_PROJ", raising=False)
    assert indexer_mod._indexer_fp8_proj_enabled() is False


def test_attention_proj_fp8_does_not_quantize_the_selector(monkeypatch):
    """Enabling FP8 on the attention projections must leave the indexer alone.

    This is the regression: the knob used to be shared, so turning on FP8 for
    the attention projections silently quantized the selector too.
    """
    monkeypatch.delenv("PRIMUS_V4_FP8_INDEXER_PROJ", raising=False)
    monkeypatch.setenv("PRIMUS_V4_FP8_ATTN_PROJ", "1")
    assert indexer_mod._indexer_fp8_proj_enabled() is False


def test_indexer_proj_fp8_reads_only_its_own_knob(monkeypatch):
    """The opt-in still exists for QAT experiments, under its own name.

    Asserts on which environment variables the helper actually consults rather
    than on its source text: the docstring legitimately names the attention flag
    while explaining that it is *not* what gates this.
    """
    consulted = []
    real_get = os.environ.get

    def _spy(key, *args, **kwargs):
        consulted.append(key)
        return real_get(key, *args, **kwargs)

    monkeypatch.setattr(os.environ, "get", _spy)
    monkeypatch.setenv("PRIMUS_V4_FP8_INDEXER_PROJ", "1")
    indexer_mod._indexer_fp8_proj_enabled()

    assert "PRIMUS_V4_FP8_INDEXER_PROJ" in consulted
    assert "PRIMUS_V4_FP8_ATTN_PROJ" not in consulted


def test_indexer_forward_is_unaffected_by_the_attention_fp8_knob(monkeypatch):
    """End to end: same scores with and without PRIMUS_V4_FP8_ATTN_PROJ."""
    torch.manual_seed(0)
    idx = indexer_mod.Indexer(
        hidden_size=32,
        index_head_dim=16,
        index_n_heads=4,
        index_topk=4,
        compress_ratio=4,
    )
    hidden = torch.randn(1, 16, 32, dtype=torch.float32)

    monkeypatch.delenv("PRIMUS_V4_FP8_ATTN_PROJ", raising=False)
    with torch.no_grad():
        idxs_a, scores_a = idx(hidden)

    monkeypatch.setenv("PRIMUS_V4_FP8_ATTN_PROJ", "1")
    with torch.no_grad():
        idxs_b, scores_b = idx(hidden)

    assert torch.equal(idxs_a, idxs_b)
    finite = torch.isfinite(scores_a)
    torch.testing.assert_close(scores_a[finite], scores_b[finite], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [4, 128])
def test_compressor_projections_are_plain_linears(ratio: int):
    """No quantized branch: plain ``nn.Linear`` ignores any FP8 autocast."""
    comp = Compressor(hidden_size=32, head_dim=16, ratio=ratio)
    linears = [m for m in comp.modules() if isinstance(m, torch.nn.Linear)]
    assert linears, "compressor must own its KV / gate projections"
    for lin in linears:
        assert type(lin) is torch.nn.Linear, f"unexpected projection type {type(lin).__name__}"


def test_compressor_has_no_fp8_code_path():
    """Guard against a quantized branch being added without revisiting this."""
    from primus.backends.megatron.core.transformer import compressor as compressor_mod

    source = inspect.getsource(compressor_mod.Compressor.forward)
    for token in ("fp8", "FP8", "fp4", "FP4", "quant"):
        assert token not in source, (
            f"Compressor.forward references {token!r}: the reference keeps the "
            "compressor in high precision, so a low-precision branch needs an "
            "explicit decision (and this test updated)."
        )
