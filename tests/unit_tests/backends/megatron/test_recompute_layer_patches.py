###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for ``primus/backends/megatron/patches/recompute_layer_patches.py``.

The patch is a thin wrapper around Megatron's
``TransformerBlock._checkpointed_forward``:

    * when ``config.recompute_layer_ids is None`` the call is forwarded
      to Megatron's original implementation verbatim;
    * when ``config.recompute_layer_ids`` is set, a small dedicated branch
      runs and checkpoints exactly the specified (global) layers.

These tests enforce that contract and -- crucially -- **detect upstream
changes to Megatron's ``_checkpointed_forward``** so that whoever upgrades
the vendored Megatron revision must either confirm the patch is still
compatible or re-adapt it.

Strategy:
    1. Signature test: pin the exact parameters we rely on.
    2. Source-fingerprint test: hash the source of
       ``TransformerBlock._checkpointed_forward`` and compare to
       ``_EXPECTED_MEGATRON_CHECKPOINTED_FORWARD_SHA256``.  When Megatron
       updates the function the test fails with the new hash so the
       maintainer can either bump the constant (after re-validating) or
       refactor the patch.
    3. Behaviour tests with stubs: verify the wrapper delegates /
       checkpoints exactly the layers it should, respects pipeline
       stage offsets, and keeps Megatron's fp8/fp4 "no-grad skip" rule.
    4. Validation tests for ``validate_specified_recompute_layers``.
"""

from __future__ import annotations

import hashlib
import inspect
from types import SimpleNamespace
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Pinned Megatron contract
# ---------------------------------------------------------------------------

# Parameters the wrapper forwards to Megatron in declaration order.  Do NOT
# relax this list without updating ``_make_checkpointed_forward_wrapper``.
_EXPECTED_CHECKPOINTED_FORWARD_PARAMS: List[str] = [
    "self",
    "hidden_states",
    "attention_mask",
    "context",
    "context_mask",
    "rotary_pos_emb",
    "attention_bias",
    "packed_seq_params",
    "use_inner_quantization_context",
    "padding_mask",
    "extract_layer_indices",
    "layer_offset",
]

# Source-level fingerprint of Megatron's ``_checkpointed_forward``.  Bump this
# deliberately after re-reading the new upstream implementation and confirming
# the patch still works.  The test prints the current hash on failure.
_EXPECTED_MEGATRON_CHECKPOINTED_FORWARD_SHA256 = (
    "e6cc0b83986f4e59ddd8e0c6c8305a79e781e505b36861a64e7f1be25830b320"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_real_megatron_transformer_block():
    """Import the vendored Megatron ``TransformerBlock``, skipping when deps missing."""
    try:
        from megatron.core.transformer.transformer_block import TransformerBlock
    except Exception as exc:  # pragma: no cover - dependency-guarded
        pytest.skip(f"Megatron not importable in this env: {exc!r}")
    return TransformerBlock


def _current_checkpointed_forward_sha256(TransformerBlock) -> str:
    src = inspect.getsource(TransformerBlock._checkpointed_forward)
    src = src.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _import_patch_module():
    try:
        from primus.backends.megatron.patches import recompute_layer_patches as mod
    except Exception as exc:  # pragma: no cover - dependency-guarded
        pytest.skip(f"recompute_layer_patches not importable: {exc!r}")
    return mod


# ---------------------------------------------------------------------------
# 1. Signature & source-fingerprint tests
# ---------------------------------------------------------------------------


def test_megatron_checkpointed_forward_signature_is_compatible():
    """The wrapper forwards using these exact kwargs; upstream must match."""
    TransformerBlock = _import_real_megatron_transformer_block()
    actual_params = list(inspect.signature(TransformerBlock._checkpointed_forward).parameters)

    assert actual_params == _EXPECTED_CHECKPOINTED_FORWARD_PARAMS, (
        "Megatron's TransformerBlock._checkpointed_forward signature changed.\n"
        f"  expected: {_EXPECTED_CHECKPOINTED_FORWARD_PARAMS}\n"
        f"  actual  : {actual_params}\n"
        "Please audit primus/backends/megatron/patches/recompute_layer_patches.py "
        "and, if compatible, update _EXPECTED_CHECKPOINTED_FORWARD_PARAMS here."
    )


def test_megatron_checkpointed_forward_source_unchanged():
    """Detect ANY edit to Megatron's ``_checkpointed_forward``.

    When this fails, inspect the new implementation and, once compatibility
    is confirmed, update ``_EXPECTED_MEGATRON_CHECKPOINTED_FORWARD_SHA256``.
    """
    TransformerBlock = _import_real_megatron_transformer_block()
    current = _current_checkpointed_forward_sha256(TransformerBlock)

    assert current == _EXPECTED_MEGATRON_CHECKPOINTED_FORWARD_SHA256, (
        "Megatron's TransformerBlock._checkpointed_forward source changed.\n"
        f"  expected sha256: {_EXPECTED_MEGATRON_CHECKPOINTED_FORWARD_SHA256}\n"
        f"  current  sha256: {current}\n"
        "Please audit primus/backends/megatron/patches/recompute_layer_patches.py "
        "and, once compatibility is confirmed, update the expected fingerprint."
    )


# ---------------------------------------------------------------------------
# 2. Behaviour tests with stubs
# ---------------------------------------------------------------------------


class _StubLayer:
    """Fake ``TransformerLayer`` that records invocation."""

    def __init__(self, layer_number: int):
        self.layer_number = layer_number
        self.called = False

    def __call__(self, **kwargs):
        self.called = True
        return kwargs["hidden_states"], kwargs.get("context")


class _FakeTensor:
    """Minimal ``torch.Tensor`` stand-in."""

    def __init__(self, name: str = "h", requires_grad: bool = True):
        self.name = name
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"_FakeTensor({self.name!r})"


class _FakeBlock:
    """Minimal ``TransformerBlock`` stand-in."""

    def __init__(self, num_layers: int, *, fp8: bool = False, fp4: bool = False):
        self.layers = [_StubLayer(layer_number=i + 1) for i in range(num_layers)]
        self.num_layers_per_pipeline_rank = num_layers
        self.vp_stage = None
        self.pg_collection = SimpleNamespace(tp=None)
        self.config = SimpleNamespace(
            recompute_layer_ids=None,
            recompute_method=None,
            recompute_num_layers=None,
            distribute_saved_activations=False,
            fp8=fp8,
            fp4=fp4,
        )

    def _get_layer(self, layer_number: int):
        return self.layers[layer_number]


def _wrapper_kwargs(hidden_states, **overrides):
    """Default kwargs for calling the wrapper (mirrors Megatron's signature)."""
    kwargs = dict(
        hidden_states=hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        attention_bias=None,
        packed_seq_params=None,
        use_inner_quantization_context=False,
        padding_mask=None,
        extract_layer_indices=None,
        layer_offset=0,
    )
    kwargs.update(overrides)
    return kwargs


def _build_wrapper(monkeypatch):
    """Build the wrapper with a recording ``tensor_parallel.checkpoint`` stub.

    The wrapper accesses ``tensor_parallel.checkpoint`` by module attribute,
    so ``monkeypatch.setattr`` intercepts it.  ``layer_offset`` is passed
    directly via the wrapper kwargs (resolved upstream in Megatron's
    ``forward``), so no further patching is needed.
    """
    mod = _import_patch_module()
    try:
        from megatron.core import tensor_parallel
    except Exception as exc:  # pragma: no cover - dependency-guarded
        pytest.skip(f"Megatron not importable in this env: {exc!r}")

    recorder = {"checkpointed_block_layer_idx": [], "current_block_layer_idx": [-1]}

    def fake_checkpoint(forward_func, _distribute, *fwd_args):
        hidden_states, context = forward_func(*fwd_args)
        recorder["checkpointed_block_layer_idx"].append(recorder["current_block_layer_idx"][0])
        return hidden_states, context

    monkeypatch.setattr(tensor_parallel, "checkpoint", fake_checkpoint)

    original_called = {"count": 0, "last_kwargs": None}

    def fake_original(self, **kwargs):
        original_called["count"] += 1
        original_called["last_kwargs"] = kwargs
        return kwargs["hidden_states"]

    wrapper = mod._make_checkpointed_forward_wrapper(fake_original)
    return wrapper, recorder, original_called


def _install_layer_index_spy(block, recorder):
    """Record which block-local layer index the wrapper is currently processing."""
    real_get_layer = block._get_layer

    def spying_get_layer(idx):
        recorder["current_block_layer_idx"][0] = idx
        return real_get_layer(idx)

    block._get_layer = spying_get_layer


def test_wrapper_delegates_when_recompute_layer_ids_is_none(monkeypatch):
    """None => forward verbatim to Megatron's original (no local processing)."""
    wrapper, recorder, original_called = _build_wrapper(monkeypatch)

    block = _FakeBlock(num_layers=3)
    h = _FakeTensor("h0")

    out = wrapper(block, **_wrapper_kwargs(h))

    assert out is h
    assert original_called["count"] == 1
    assert not any(layer.called for layer in block.layers)
    assert recorder["checkpointed_block_layer_idx"] == []


def test_wrapper_checkpoints_only_requested_layers(monkeypatch):
    """Exactly the layers in ``recompute_layer_ids`` go through ``checkpoint``."""
    wrapper, recorder, original_called = _build_wrapper(monkeypatch)

    block = _FakeBlock(num_layers=4)
    block.config.recompute_layer_ids = [0, 2]
    _install_layer_index_spy(block, recorder)

    wrapper(block, **_wrapper_kwargs(_FakeTensor("h0")))

    assert all(layer.called for layer in block.layers)
    assert sorted(recorder["checkpointed_block_layer_idx"]) == [0, 2]
    assert original_called["count"] == 0


def test_wrapper_respects_pipeline_layer_offset(monkeypatch):
    """``recompute_layer_ids`` are global; wrapper uses upstream ``layer_offset``."""
    wrapper, recorder, _ = _build_wrapper(monkeypatch)

    block = _FakeBlock(num_layers=4)
    block.config.recompute_layer_ids = [2, 3]  # global indices
    _install_layer_index_spy(block, recorder)

    wrapper(block, **_wrapper_kwargs(_FakeTensor("h0"), layer_offset=2))

    assert all(layer.called for layer in block.layers)
    # Global 2,3 -> block-local 0,1 when layer_offset=2.
    assert sorted(recorder["checkpointed_block_layer_idx"]) == [0, 1]


def test_wrapper_skips_checkpoint_for_fp8_without_grad(monkeypatch):
    """Mirrors Megatron: fp8/fp4 + no-grad tensor cannot go through re-entrant autograd."""
    wrapper, recorder, _ = _build_wrapper(monkeypatch)

    block = _FakeBlock(num_layers=3, fp8=True)
    block.config.recompute_layer_ids = [0, 1, 2]
    _install_layer_index_spy(block, recorder)

    wrapper(block, **_wrapper_kwargs(_FakeTensor("h0", requires_grad=False)))

    assert all(layer.called for layer in block.layers)
    # Every layer is "requested" but grad=False forces the skip path.
    assert recorder["checkpointed_block_layer_idx"] == []


def test_wrapper_rejects_extract_layer_indices_when_recompute_layer_ids_set(monkeypatch):
    wrapper, _, _ = _build_wrapper(monkeypatch)

    block = _FakeBlock(num_layers=2)
    block.config.recompute_layer_ids = [0]

    with pytest.raises(NotImplementedError):
        wrapper(block, **_wrapper_kwargs(_FakeTensor("h0"), extract_layer_indices={0}))


def test_wrapper_exposes_idempotency_markers(monkeypatch):
    """``patch_custom_recompute_layer_ids`` uses these attrs to skip double-wrapping."""
    wrapper, _, _ = _build_wrapper(monkeypatch)

    assert getattr(wrapper, "_primus_patched", False) is True
    assert getattr(wrapper, "_primus_original", None) is not None


# ---------------------------------------------------------------------------
# 3. validate_specified_recompute_layers tests
# ---------------------------------------------------------------------------


def _valid_args(**overrides):
    base = SimpleNamespace(
        num_layers=8,
        recompute_granularity="full",
        recompute_method=None,
        distribute_saved_activations=False,
        sequence_parallel=False,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_validate_specified_recompute_layers_noop_when_none():
    mod = _import_patch_module()
    config = SimpleNamespace(recompute_layer_ids=None)
    mod.validate_specified_recompute_layers(config, _valid_args())
    assert config.recompute_layer_ids is None


def test_validate_specified_recompute_layers_accepts_string_and_list():
    mod = _import_patch_module()

    config = SimpleNamespace(recompute_layer_ids="1, 3, 1, 5")
    mod.validate_specified_recompute_layers(config, _valid_args())
    assert config.recompute_layer_ids == [1, 3, 5]

    config = SimpleNamespace(recompute_layer_ids=[2, 4, "4"])
    mod.validate_specified_recompute_layers(config, _valid_args())
    assert config.recompute_layer_ids == [2, 4]


def test_validate_specified_recompute_layers_rejects_bad_config():
    mod = _import_patch_module()

    with pytest.raises(ValueError, match="recompute_granularity"):
        mod.validate_specified_recompute_layers(
            SimpleNamespace(recompute_layer_ids=[0]),
            _valid_args(recompute_granularity="selective"),
        )

    with pytest.raises(ValueError, match="recompute_method"):
        mod.validate_specified_recompute_layers(
            SimpleNamespace(recompute_layer_ids=[0]),
            _valid_args(recompute_method="block"),
        )

    with pytest.raises(ValueError, match="between 0"):
        mod.validate_specified_recompute_layers(
            SimpleNamespace(recompute_layer_ids=[99]),
            _valid_args(num_layers=4),
        )

    with pytest.raises(ValueError, match="between 0"):
        mod.validate_specified_recompute_layers(
            SimpleNamespace(recompute_layer_ids=[-1]),
            _valid_args(num_layers=4),
        )

    with pytest.raises(ValueError, match="must not be empty"):
        mod.validate_specified_recompute_layers(
            SimpleNamespace(recompute_layer_ids=[]),
            _valid_args(),
        )

    with pytest.raises(ValueError, match="distribute_saved_activations"):
        mod.validate_specified_recompute_layers(
            SimpleNamespace(recompute_layer_ids=[0]),
            _valid_args(distribute_saved_activations=True, sequence_parallel=True),
        )
