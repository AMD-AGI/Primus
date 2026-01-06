###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for TorchTitan model override patch.
"""

import types
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from primus.backends.torchtitan.patches.model_override_patches import (
    _flatten_model_overrides,
    patch_torchtitan_model_override,
)
from primus.core.patches import PatchContext


@dataclass
class MockModelArgs:
    """Mock model args dataclass for testing."""

    n_layers: int = 32
    dim: int = 4096
    n_heads: int = 32


class MockTrainSpec:
    """Mock train spec for testing."""

    def __init__(self):
        self.model_args = {
            "debugmodel": MockModelArgs(),
            "1B": MockModelArgs(n_layers=16, dim=2048),
        }


# ----------------------- Helper Function Tests -----------------------


def test_flatten_model_overrides_flat():
    """Test flattening already-flat overrides."""
    overrides = {"model.n_layers": 8, "model.dim": 2048}
    result = _flatten_model_overrides(overrides)
    assert result == {"model.n_layers": 8, "model.dim": 2048}


def test_flatten_model_overrides_nested():
    """Test flattening nested overrides."""
    overrides = {"model": {"n_layers": 8, "dim": 2048}}
    result = _flatten_model_overrides(overrides)
    assert result == {"model.n_layers": 8, "model.dim": 2048}


def test_flatten_model_overrides_mixed():
    """Test flattening mixed flat and nested overrides."""
    overrides = {"model": {"n_layers": 8}, "model.dim": 2048}
    result = _flatten_model_overrides(overrides)
    assert result == {"model.n_layers": 8, "model.dim": 2048}


def test_flatten_model_overrides_empty():
    """Test flattening empty overrides."""
    result = _flatten_model_overrides({})
    assert result == {}


# ----------------------- Patch Function Tests -----------------------


def test_patch_no_overrides():
    """Test patch skips when no model_overrides provided."""
    ctx = PatchContext(
        backend="torchtitan",
        phase="setup",
        extra={
            "module_config": SimpleNamespace(
                name="test_module",
                params=SimpleNamespace(model=SimpleNamespace(name="llama3", flavor="debugmodel")),
            )
        },
    )

    # Should return early without error
    with patch("primus.backends.torchtitan.patches.model_override_patches.log_rank_0") as mock_log:
        patch_torchtitan_model_override(ctx)
        mock_log.assert_called_once()
        assert "No model_overrides provided" in str(mock_log.call_args)


def test_patch_bad_keys():
    """Test patch raises error for invalid override keys."""
    ctx = PatchContext(
        backend="torchtitan",
        phase="setup",
        extra={
            "module_config": SimpleNamespace(
                name="test_module",
                params=SimpleNamespace(
                    model=SimpleNamespace(name="llama3", flavor="debugmodel"),
                    model_overrides={"bad_key": 123},  # Missing 'model.' prefix
                ),
            )
        },
    )

    with pytest.raises(ValueError, match="Invalid override keys detected"):
        patch_torchtitan_model_override(ctx)


def test_patch_missing_model_name():
    """Test patch raises error when model.name is missing."""
    ctx = PatchContext(
        backend="torchtitan",
        phase="setup",
        extra={
            "module_config": SimpleNamespace(
                name="test_module",
                params=SimpleNamespace(
                    model=SimpleNamespace(flavor="debugmodel"),  # Missing name
                    model_overrides={"model.n_layers": 8},
                ),
            )
        },
    )

    with pytest.raises(ValueError, match="model.name is required"):
        patch_torchtitan_model_override(ctx)


def test_patch_missing_flavor():
    """Test patch raises error when model.flavor is missing."""
    ctx = PatchContext(
        backend="torchtitan",
        phase="setup",
        extra={
            "module_config": SimpleNamespace(
                name="test_module",
                params=SimpleNamespace(
                    model=SimpleNamespace(name="llama3"),  # Missing flavor
                    model_overrides={"model.n_layers": 8},
                ),
            )
        },
    )

    with pytest.raises(ValueError, match="model.flavor is required"):
        patch_torchtitan_model_override(ctx)


def test_patch_applies_overrides():
    """Test patch successfully applies model overrides."""
    ctx = PatchContext(
        backend="torchtitan",
        phase="setup",
        extra={
            "module_config": SimpleNamespace(
                name="test_module",
                params=SimpleNamespace(
                    model=SimpleNamespace(name="llama3", flavor="debugmodel"),
                    model_overrides={"model.n_layers": 8, "model.dim": 2048},
                ),
            )
        },
    )

    # Create a mock train_spec module
    mock_spec = MockTrainSpec()
    mock_train_spec_module = types.ModuleType("torchtitan.protocols.train_spec")
    mock_train_spec_module.get_train_spec = lambda name: mock_spec

    with patch.dict("sys.modules", {"torchtitan.protocols.train_spec": mock_train_spec_module}):
        with patch("primus.backends.torchtitan.patches.model_override_patches.log_rank_0"):
            # Apply patch
            patch_torchtitan_model_override(ctx)

            # Verify the patch was applied by calling the patched function
            spec = mock_train_spec_module.get_train_spec("llama3")
            target_args = spec.model_args["debugmodel"]

            assert target_args.n_layers == 8
            assert target_args.dim == 2048
            assert target_args.n_heads == 32  # Unchanged


def test_patch_only_affects_target_model():
    """Test patch only affects the specified model, not others."""
    ctx = PatchContext(
        backend="torchtitan",
        phase="setup",
        extra={
            "module_config": SimpleNamespace(
                name="test_module",
                params=SimpleNamespace(
                    model=SimpleNamespace(name="llama3", flavor="debugmodel"),
                    model_overrides={"model.n_layers": 8},
                ),
            )
        },
    )

    # Create a mock train_spec module with multiple models
    mock_spec_llama = MockTrainSpec()
    mock_spec_other = MockTrainSpec()

    def mock_get_train_spec(name: str):
        if name == "llama3":
            return mock_spec_llama
        else:
            return mock_spec_other

    mock_train_spec_module = types.ModuleType("torchtitan.protocols.train_spec")
    mock_train_spec_module.get_train_spec = mock_get_train_spec

    with patch.dict("sys.modules", {"torchtitan.protocols.train_spec": mock_train_spec_module}):
        with patch("primus.backends.torchtitan.patches.model_override_patches.log_rank_0"):
            # Apply patch
            patch_torchtitan_model_override(ctx)

            # Verify llama3 was patched
            spec_llama = mock_train_spec_module.get_train_spec("llama3")
            assert spec_llama.model_args["debugmodel"].n_layers == 8

            # Verify other model was NOT patched
            spec_other = mock_train_spec_module.get_train_spec("other_model")
            assert spec_other.model_args["debugmodel"].n_layers == 32  # Original value


def test_patch_invalid_flavor():
    """Test patch raises error when flavor doesn't exist."""
    ctx = PatchContext(
        backend="torchtitan",
        phase="setup",
        extra={
            "module_config": SimpleNamespace(
                name="test_module",
                params=SimpleNamespace(
                    model=SimpleNamespace(name="llama3", flavor="nonexistent"),
                    model_overrides={"model.n_layers": 8},
                ),
            )
        },
    )

    mock_spec = MockTrainSpec()
    mock_train_spec_module = types.ModuleType("torchtitan.protocols.train_spec")
    mock_train_spec_module.get_train_spec = lambda name: mock_spec

    with patch.dict("sys.modules", {"torchtitan.protocols.train_spec": mock_train_spec_module}):
        with patch("primus.backends.torchtitan.patches.model_override_patches.log_rank_0"):
            # Apply patch
            patch_torchtitan_model_override(ctx)

            # Call patched function and expect KeyError
            with pytest.raises(KeyError, match="flavor 'nonexistent' not found"):
                mock_train_spec_module.get_train_spec("llama3")


def test_patch_invalid_field():
    """Test patch raises error when field doesn't exist in dataclass."""
    ctx = PatchContext(
        backend="torchtitan",
        phase="setup",
        extra={
            "module_config": SimpleNamespace(
                name="test_module",
                params=SimpleNamespace(
                    model=SimpleNamespace(name="llama3", flavor="debugmodel"),
                    model_overrides={"model.nonexistent_field": 123},
                ),
            )
        },
    )

    mock_spec = MockTrainSpec()
    mock_train_spec_module = types.ModuleType("torchtitan.protocols.train_spec")
    mock_train_spec_module.get_train_spec = lambda name: mock_spec

    with patch.dict("sys.modules", {"torchtitan.protocols.train_spec": mock_train_spec_module}):
        with patch("primus.backends.torchtitan.patches.model_override_patches.log_rank_0"):
            # Apply patch
            patch_torchtitan_model_override(ctx)

            # Call patched function and expect AttributeError
            with pytest.raises(AttributeError, match="has no field 'nonexistent_field'"):
                mock_train_spec_module.get_train_spec("llama3")
