###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for V4 block structural rebase + PP K-stream packing (P15).

Covers (plan-2 P15):
* ``DeepseekV4HybridLayer`` is a :class:`TransformerLayer` subclass (type
  identity).
* ``DeepseekV4HybridLayerSubmodules`` extends
  :class:`TransformerLayerSubmodules` and exposes ``input_layernorm`` /
  ``self_attention`` / ``pre_mlp_layernorm`` / ``mlp`` plus the V4-specific
  ``attn_hc`` / ``ffn_hc`` HC mixer hooks.
* ``DeepseekV4TransformerBlock`` is a :class:`TransformerBlock` subclass.
* ``_lift_streams_in`` / ``_lower_streams_out`` are bit-exact under
  roundtrip across the four PP-stage permutations
  (pre_process * post_process), for both single-stream (``hc_mult=1``)
  and multi-stream (``hc_mult>1``).
* ``DeepseekV4Model.forward`` no longer touches the
  ``decoder._v4_token_ids`` attribute stash (audit).
"""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest
import torch
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_block import (
    DeepseekV4HybridLayer,
    DeepseekV4HybridLayerSubmodules,
    DeepseekV4TransformerBlock,
    DeepseekV4TransformerBlockSubmodules,
    _lift_streams_in,
    _lower_streams_out,
)

# ---------------------------------------------------------------------------
# Subclass identity
# ---------------------------------------------------------------------------


def test_hybrid_layer_subclasses_transformer_layer() -> None:
    """``DeepseekV4HybridLayer`` is a :class:`TransformerLayer` subclass."""
    assert issubclass(DeepseekV4HybridLayer, TransformerLayer)


def test_block_subclasses_transformer_block() -> None:
    """``DeepseekV4TransformerBlock`` is a :class:`TransformerBlock` subclass."""
    assert issubclass(DeepseekV4TransformerBlock, TransformerBlock)


def test_hybrid_layer_submodules_extends_transformer_layer_submodules() -> None:
    """``DeepseekV4HybridLayerSubmodules`` extends the upstream dataclass and
    adds V4-specific HC fields."""
    assert issubclass(DeepseekV4HybridLayerSubmodules, TransformerLayerSubmodules)

    field_names = {f.name for f in fields(DeepseekV4HybridLayerSubmodules)}

    # Inherited canonical fields.
    for name in (
        "input_layernorm",
        "self_attention",
        "pre_mlp_layernorm",
        "mlp",
        "self_attn_bda",
        "cross_attention",
        "cross_attn_bda",
    ):
        assert name in field_names, f"missing inherited field: {name}"

    # V4-specific extension fields.
    assert "attn_hc" in field_names
    assert "ffn_hc" in field_names

    # The V4-specific extension fields default to None when omitted.
    sm = DeepseekV4HybridLayerSubmodules()
    assert sm.attn_hc is None
    assert sm.ffn_hc is None


def test_block_submodules_dataclass_shape() -> None:
    """The block-level submodules dataclass exposes the three V4 fields."""
    field_names = {f.name for f in fields(DeepseekV4TransformerBlockSubmodules)}
    assert field_names == {"layer_specs", "hyper_head", "final_layernorm"}


# ---------------------------------------------------------------------------
# Lift / lower helpers — single-stream (hc_mult == 1)
# ---------------------------------------------------------------------------


def _seq_first(B: int, S: int, D: int) -> torch.Tensor:
    """Make a deterministic [S, B, D] tensor for shape tests."""
    return torch.arange(S * B * D, dtype=torch.float32).view(S, B, D)


def test_lift_lower_single_stream_first_stage_roundtrip() -> None:
    """``hc_mult=1``, first PP stage: [S, B, D] -> [B, S, D] -> [S, B, D]."""
    x = _seq_first(B=2, S=4, D=6)
    lifted = _lift_streams_in(x, pre_process=True, hc_mult=1)
    assert lifted.shape == (2, 4, 6)

    # Final stage lower: [B, S, D] -> [S, B, D] (transpose).
    lowered = _lower_streams_out(lifted, post_process=True, hc_mult=1)
    assert lowered.shape == (4, 2, 6)
    assert torch.equal(lowered, x)


def test_lift_lower_single_stream_intermediate_stage_roundtrip() -> None:
    """``hc_mult=1`` non-final stage: [S, B, D] -> [B, S, D] -> [S, B, D]."""
    x = _seq_first(B=2, S=4, D=6)
    lifted = _lift_streams_in(x, pre_process=False, hc_mult=1)
    lowered = _lower_streams_out(lifted, post_process=False, hc_mult=1)
    assert lowered.shape == x.shape
    assert torch.equal(lowered, x)


# ---------------------------------------------------------------------------
# Lift / lower helpers — multi-stream (hc_mult > 1)
# ---------------------------------------------------------------------------


def test_lift_first_stage_multi_stream_expands_k() -> None:
    """First PP stage: [S, B, D] -> [B, S, K, D] with K replicated."""
    K = 3
    S, B, D = 5, 2, 4
    x = _seq_first(B=B, S=S, D=D)
    lifted = _lift_streams_in(x, pre_process=True, hc_mult=K)
    assert lifted.shape == (B, S, K, D)

    # The K dim was expanded from a singleton, so all K planes are equal.
    for k in range(K):
        assert torch.equal(lifted[:, :, k, :], x.transpose(0, 1).contiguous())


def test_lower_intermediate_stage_packs_k_into_seq() -> None:
    """Non-final PP stage: [B, S, K, D] -> [S*K, B, D]."""
    K = 3
    B, S, D = 2, 5, 4
    x4d = torch.randn(B, S, K, D)
    packed = _lower_streams_out(x4d, post_process=False, hc_mult=K)
    assert packed.shape == (S * K, B, D)


def test_lift_lower_multi_stream_intermediate_roundtrip() -> None:
    """K-stream packing roundtrips bit-exactly across PP boundary."""
    K = 4
    B, S, D = 2, 6, 8
    x4d = torch.randn(B, S, K, D)

    # Lower (this stage's output to next stage's input): [B, S, K, D] -> [S*K, B, D]
    packed = _lower_streams_out(x4d, post_process=False, hc_mult=K)

    # Lift (next stage receives the packed tensor): [S*K, B, D] -> [B, S, K, D]
    lifted = _lift_streams_in(packed, pre_process=False, hc_mult=K)
    assert lifted.shape == x4d.shape
    assert torch.equal(lifted, x4d)


def test_lower_post_process_collapses_to_seq_first() -> None:
    """Final PP stage: HyperHead has already collapsed to [B, S, D];
    ``_lower_streams_out`` just transposes to the [S, B, D] output."""
    K = 4
    B, S, D = 2, 6, 8
    x_collapsed = torch.randn(B, S, D)
    out = _lower_streams_out(x_collapsed, post_process=True, hc_mult=K)
    assert out.shape == (S, B, D)
    assert torch.equal(out, x_collapsed.transpose(0, 1).contiguous())


def test_lift_pp_intermediate_rejects_misaligned_seq() -> None:
    """Mismatched ``S*K`` on the incoming PP tensor raises a clear error."""
    K = 3
    bad = torch.randn(7, 2, 4)  # 7 not divisible by K=3
    with pytest.raises(ValueError, match="not divisible by hc_mult"):
        _lift_streams_in(bad, pre_process=False, hc_mult=K)


def test_lower_intermediate_stage_rejects_collapsed_input() -> None:
    """A non-final stage must hand in the 4D K-stream form, not [B, S, D]."""
    K = 3
    bad = torch.randn(2, 5, 4)
    with pytest.raises(ValueError, match=r"\[B, S, K, D\]"):
        _lower_streams_out(bad, post_process=False, hc_mult=K)


def test_lower_post_process_rejects_uncollapsed_input() -> None:
    """The final stage hands in [B, S, D] (post-HyperHead), not the 4D form."""
    K = 3
    bad = torch.randn(2, 5, K, 4)
    with pytest.raises(ValueError, match=r"\[B, S, D\] after HyperHead"):
        _lower_streams_out(bad, post_process=True, hc_mult=K)


# ---------------------------------------------------------------------------
# Token-ids decoder-stash removal (AST audit)
# ---------------------------------------------------------------------------


_MODEL_PATH = Path(__file__).resolve().parents[5] / (
    "primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_model.py"
)


def test_model_forward_does_not_set_decoder_v4_token_ids_attribute() -> None:
    """Plan-2 P15 retires the ``decoder._v4_token_ids`` attribute stash;
    audit the model source AST to confirm no setter / getter remains."""
    src = _MODEL_PATH.read_text()
    tree = ast.parse(src)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "_v4_token_ids":
            line = ast.get_source_segment(src, node) or "?"
            offenders.append(f"{node.lineno}:{line}")

    assert not offenders, "DeepseekV4Model still references decoder._v4_token_ids:\n  " + "\n  ".join(
        offenders
    )


def test_model_forward_passes_token_ids_kwarg_to_decoder() -> None:
    """Forward must pass ``input_ids`` as ``token_ids=`` to the decoder."""
    src = _MODEL_PATH.read_text()
    assert "token_ids=input_ids" in src, (
        "DeepseekV4Model.forward must pass `token_ids=input_ids` to the decoder; "
        "this is the plan-2 P15 token-ids forward-kwarg threading contract."
    )


def test_block_forward_signature_accepts_position_ids_and_token_ids() -> None:
    """The block forward must accept both as kwargs."""
    import inspect

    sig = inspect.signature(DeepseekV4TransformerBlock.forward)
    params = sig.parameters
    assert "position_ids" in params, "block.forward must accept position_ids kwarg"
    assert "token_ids" in params, "block.forward must accept token_ids kwarg"


def test_layer_forward_signature_accepts_attention_mask_kwarg() -> None:
    """Upstream :class:`MultiTokenPredictionLayer` forwards
    ``attention_mask`` to its inner transformer layer; ours must accept
    (and ignore) it for forward-API compatibility."""
    import inspect

    sig = inspect.signature(DeepseekV4HybridLayer.forward)
    params = sig.parameters
    assert "attention_mask" in params
    assert "position_ids" in params
    assert "token_ids" in params
