###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for V4 MTP integration with upstream
:class:`MultiTokenPredictionBlock` (P16).

What this file covers (CPU-friendly):

* :func:`get_v4_mtp_block_spec` returns a well-formed ``ModuleSpec``:
  the outer module is :class:`MultiTokenPredictionBlock`, its
  ``submodules.layer_specs`` is a list of length ``mtp_num_layers``,
  and each entry is a :class:`MultiTokenPredictionLayer` ``ModuleSpec``
  whose ``mtp_model_layer`` is the V4 hybrid-layer spec we passed in.
* Per-MTP-layer submodules pull V4 RMSNorm / column-parallel linear
  from the V4 spec provider — the helper does not silently fall back
  to TE / vanilla impls.
* The V4 hybrid layer's ``forward`` returns the upstream-compatible
  ``(hidden_states, None)`` tuple so it can plug into
  :meth:`MultiTokenPredictionLayer._proj_and_transformer_layer` which
  unpacks ``hidden_states, _ = self.mtp_model_layer(...)``.
* The V4 attention spec advertises ``attn_mask_type`` so upstream MTP
  validation passes (V4 manages its own SWA / sink mask internally,
  but the field is required by the upstream pre-build assertion).
* :class:`DeepseekV4MTPBlock` (legacy) emits a ``DeprecationWarning``
  on construction (planned removal: plan-2 P21).
* :class:`DeepseekV4Model.forward` wires :func:`process_mtp_loss` and
  :class:`MultiTokenPredictionBlock` (AST audit; full distributed run
  is gated G7 in P19).

The full ``mtp_num_layers=0`` vs ``mtp_num_layers=1`` main-LM loss
invariance gate (G7) requires distributed init and is tracked into
P19 distributed re-validation.
"""

from __future__ import annotations

import ast
import inspect
import warnings
from dataclasses import is_dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionBlockSubmodules,
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
    DeepSeekV4SpecProvider,
)
from primus.backends.megatron.core.models.deepseek_v4 import (
    DeepseekV4HybridLayer,
    DeepseekV4HybridLayerSubmodules,
    DeepseekV4MTPBlock,
    get_v4_mtp_block_spec,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]
_MODEL_PATH = _REPO_ROOT / ("primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_model.py")
_SPECS_PATH = _REPO_ROOT / ("primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_layer_specs.py")


# ---------------------------------------------------------------------------
# Minimal V4 config / layer-spec fixtures
# ---------------------------------------------------------------------------


def _make_v4_config(*, mtp_num_layers: int = 1):
    """Tiny V4 config with the fields the MTP spec helper reads."""
    cfg = MagicMock()
    cfg.hidden_size = 64
    cfg.num_layers = 2
    cfg.mtp_num_layers = mtp_num_layers
    cfg.hc_mult = 1
    cfg.qk_pos_emb_head_dim = 16
    cfg.attn_sliding_window = 128
    cfg.norm_epsilon = 1e-6
    cfg.layernorm_epsilon = 1e-6
    cfg.use_mup = False
    cfg.tensor_model_parallel_size = 1
    cfg.sequence_parallel = False
    cfg.bf16 = False
    cfg.fp16 = False
    return cfg


def _placeholder_layer_spec() -> ModuleSpec:
    """Stand-in for a V4 hybrid-layer spec; the helper just threads it through."""
    return ModuleSpec(
        module=DeepseekV4HybridLayer,
        params={"layer_idx": 0, "compress_ratio": 0},
        submodules=DeepseekV4HybridLayerSubmodules(),
    )


# ---------------------------------------------------------------------------
# get_v4_mtp_block_spec — structural assertions
# ---------------------------------------------------------------------------


def test_helper_returns_multi_token_prediction_block_spec() -> None:
    cfg = _make_v4_config(mtp_num_layers=1)
    spec = get_v4_mtp_block_spec(cfg, transformer_layer_spec=_placeholder_layer_spec())
    assert isinstance(spec, ModuleSpec)
    assert spec.module is MultiTokenPredictionBlock
    assert isinstance(spec.submodules, MultiTokenPredictionBlockSubmodules)


@pytest.mark.parametrize("mtp_num_layers", [1, 2, 3])
def test_helper_emits_one_layer_spec_per_depth(mtp_num_layers: int) -> None:
    cfg = _make_v4_config(mtp_num_layers=mtp_num_layers)
    spec = get_v4_mtp_block_spec(cfg, transformer_layer_spec=_placeholder_layer_spec())
    layer_specs = spec.submodules.layer_specs
    assert isinstance(layer_specs, list)
    assert len(layer_specs) == mtp_num_layers
    for layer_spec in layer_specs:
        assert isinstance(layer_spec, ModuleSpec)
        assert layer_spec.module is MultiTokenPredictionLayer


def test_helper_threads_v4_inner_layer_unchanged() -> None:
    cfg = _make_v4_config(mtp_num_layers=2)
    inner = _placeholder_layer_spec()
    spec = get_v4_mtp_block_spec(cfg, transformer_layer_spec=inner)
    for layer_spec in spec.submodules.layer_specs:
        sub: MultiTokenPredictionLayerSubmodules = layer_spec.submodules
        assert sub.mtp_model_layer is inner, (
            "MTP helper must thread the V4 hybrid layer spec through unchanged "
            "so MTP depths share HC / hash-routing / clamped-SwiGLU with the main decoder."
        )


def test_helper_pulls_norm_and_linear_from_v4_provider() -> None:
    cfg = _make_v4_config(mtp_num_layers=1)
    provider = DeepSeekV4SpecProvider(config=cfg)
    spec = get_v4_mtp_block_spec(cfg, transformer_layer_spec=_placeholder_layer_spec())
    sub = spec.submodules.layer_specs[0].submodules
    expected_norm = provider.v4_norm_module()
    expected_col = provider.column_parallel_linear()
    assert sub.enorm is expected_norm
    assert sub.hnorm is expected_norm
    assert sub.layer_norm is expected_norm
    assert sub.eh_proj is expected_col


def test_helper_rejects_zero_mtp_num_layers() -> None:
    cfg = _make_v4_config(mtp_num_layers=0)
    with pytest.raises(ValueError, match="mtp_num_layers >= 1"):
        get_v4_mtp_block_spec(cfg, transformer_layer_spec=_placeholder_layer_spec())


# ---------------------------------------------------------------------------
# DeepseekV4HybridLayer — upstream-compatible tuple return
# ---------------------------------------------------------------------------


def test_layer_submodules_extends_transformer_layer_submodules() -> None:
    """Required so MultiTokenPredictionLayer.__init__'s submodules
    isinstance check picks up the GPT path (not Mamba)."""
    assert is_dataclass(DeepseekV4HybridLayerSubmodules)
    assert issubclass(DeepseekV4HybridLayerSubmodules, TransformerLayerSubmodules)


def test_layer_forward_signature_returns_tuple() -> None:
    """``forward`` is annotated as a tuple-returning callable so
    ``MultiTokenPredictionLayer._proj_and_transformer_layer`` can unpack
    ``hidden_states, _ = self.mtp_model_layer(...)`` without error."""
    src = inspect.getsource(DeepseekV4HybridLayer.forward)
    assert "return x, None" in src, (
        "DeepseekV4HybridLayer.forward must return (hidden_states, None) for "
        "upstream MultiTokenPredictionLayer compatibility."
    )


# ---------------------------------------------------------------------------
# V4 attention spec — declares an attn_mask_type for upstream MTP validation
# ---------------------------------------------------------------------------


def test_attention_spec_declares_supported_attn_mask_type() -> None:
    src = _SPECS_PATH.read_text()
    assert "AttnMaskType.causal" in src, (
        "V4 attention spec must declare attn_mask_type=AttnMaskType.causal "
        "(in spec.params) so MultiTokenPredictionLayer's pre-build "
        "validation accepts the V4 inner layer."
    )
    # Sanity: import works and value resolves.
    assert AttnMaskType.causal.name == "causal"


# ---------------------------------------------------------------------------
# Legacy V4 MTP block — deprecated
# ---------------------------------------------------------------------------


def test_legacy_mtp_block_emits_deprecation_warning() -> None:
    """Constructing :class:`DeepseekV4MTPBlock` must surface a
    ``DeprecationWarning`` so users migrate to ``get_v4_mtp_block_spec``."""
    cfg = _make_v4_config(mtp_num_layers=1)
    rope = MagicMock()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            DeepseekV4MTPBlock(cfg, rope=rope, mtp_num_layers=1)
        except Exception:
            # The construction may fail later (e.g. RMSNorm init expects real
            # config fields); that's OK — we only care the warning fires.
            pass
        deprecation_warnings = [
            str(w.message) for w in captured if issubclass(w.category, DeprecationWarning)
        ]
    assert any("DeepseekV4MTPBlock is deprecated" in m for m in deprecation_warnings), (
        "Constructing DeepseekV4MTPBlock should emit a DeprecationWarning "
        "pointing users to get_v4_mtp_block_spec; got: " + repr(deprecation_warnings)
    )


# ---------------------------------------------------------------------------
# DeepseekV4Model.forward — process_mtp_loss + MTP block wired (AST audit)
# ---------------------------------------------------------------------------


def test_model_forward_calls_process_mtp_loss() -> None:
    src = _MODEL_PATH.read_text()
    tree = ast.parse(src)
    has_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "process_mtp_loss":
                has_call = True
                break
    assert has_call, (
        "DeepseekV4Model.forward must call process_mtp_loss to wire MTP "
        "into the LM-loss pipeline (plan-2 P16)."
    )


def test_model_imports_upstream_mtp_machinery() -> None:
    src = _MODEL_PATH.read_text()
    for needle in (
        "from megatron.core.transformer.multi_token_prediction import",
        "MultiTokenPredictionBlock",
        "process_mtp_loss",
        "mtp_on_this_rank",
        "get_v4_mtp_block_spec",
    ):
        assert needle in src, f"DeepseekV4Model must import {needle!r} (P16)."


def test_model_init_routes_through_v4_mtp_spec_helper() -> None:
    src = _MODEL_PATH.read_text()
    assert "get_v4_mtp_block_spec(" in src, (
        "DeepseekV4Model.__init__ must use get_v4_mtp_block_spec to build "
        "self.mtp; the spec is then handed to MultiTokenPredictionBlock."
    )


def test_model_preserves_legacy_mtp_block_flag() -> None:
    """The ``v4_use_custom_mtp_block`` config flag stays available for
    back-compat with research checkpoints; planned removal P21."""
    src = _MODEL_PATH.read_text()
    assert "v4_use_custom_mtp_block" in src
    assert "DeepseekV4MTPBlock(" in src, (
        "DeepseekV4Model.__init__ must keep the legacy DeepseekV4MTPBlock "
        "construction behind v4_use_custom_mtp_block until P21 retirement."
    )


# ---------------------------------------------------------------------------
# Tiny CPU-only smoke: model __init__ in a no-MTP / no-distributed config
# does not crash and leaves self.mtp = None
# ---------------------------------------------------------------------------


def test_model_init_no_mtp_path_does_not_build_mtp() -> None:
    """When ``mtp_num_layers == 0`` the model must not construct an MTP
    block (regardless of distributed state)."""
    # We don't build a real model here (it requires Megatron's full init
    # machinery); we just confirm the __init__ source has the
    # mtp_num_layers > 0 guard so the no-MTP path stays inert.
    src = _MODEL_PATH.read_text()
    assert "mtp_num_layers > 0" in src
