###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for TorchTitanJobConfigBuilder.

Focus areas:
    1. Deep merge mechanism: nested dictionaries merged correctly
    2. Override mechanism: Primus config overrides defaults
    3. Namespace conversion: dict → nested SimpleNamespace
    4. JobConfig conversion: dict → nested dataclass
    5. Interface consistency: same API as MegatronArgBuilder
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Mock torchtitan module before importing argument_builder
sys.modules["torchtitan"] = MagicMock()
sys.modules["torchtitan.config"] = MagicMock()
sys.modules["torchtitan.config.job_config"] = MagicMock()

from primus.backends.torchtitan.argument_builder import (
    TorchTitanJobConfigBuilder,
    _deep_merge,
)


class TestDeepMerge:
    """Test the deep_merge helper function."""

    def test_simple_merge(self):
        """Test simple non-nested merge."""
        base = {"a": 1, "b": 2}
        overrides = {"b": 3, "c": 4}

        result = _deep_merge(base, overrides)

        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Test nested dictionary merge."""
        base = {"model": {"name": "llama3", "size": "8B"}, "training": {"steps": 1000}}
        overrides = {
            "model": {"size": "70B"},  # Override nested value
            "training": {"steps": 5000, "batch_size": 8},  # Override + add
        }

        result = _deep_merge(base, overrides)

        assert result == {
            "model": {"name": "llama3", "size": "70B"},
            "training": {"steps": 5000, "batch_size": 8},
        }

    def test_deep_nested_merge(self):
        """Test deeply nested dictionary merge."""
        base = {"level1": {"level2": {"level3": {"a": 1, "b": 2}}}}
        overrides = {"level1": {"level2": {"level3": {"b": 3, "c": 4}}}}

        result = _deep_merge(base, overrides)

        assert result["level1"]["level2"]["level3"] == {"a": 1, "b": 3, "c": 4}

    def test_override_with_non_dict(self):
        """Test that non-dict values replace dict values."""
        base = {"model": {"name": "llama3", "size": "8B"}}
        overrides = {"model": "simple_string"}

        result = _deep_merge(base, overrides)

        assert result == {"model": "simple_string"}

    def test_empty_overrides(self):
        """Test merge with empty overrides."""
        base = {"a": 1, "b": 2}
        overrides = {}

        result = _deep_merge(base, overrides)

        assert result == {"a": 1, "b": 2}


class TestTorchTitanJobConfigBuilderUpdate:
    """Test the update() method."""

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_update_stores_config(self, mock_load_defaults):
        """Test that update() updates config correctly."""
        mock_load_defaults.return_value = {
            "model": {"name": "default", "flavor": "default"},
            "training": {"steps": 100},
        }

        builder = TorchTitanJobConfigBuilder()

        # Create SimpleNamespace input
        ns = SimpleNamespace(
            model=SimpleNamespace(name="llama3", flavor="debugmodel"), training=SimpleNamespace(steps=1000)
        )

        builder.update(ns)

        # Check that config was updated
        assert "model" in builder.config
        assert "training" in builder.config
        assert builder.config["model"]["name"] == "llama3"
        assert builder.config["training"]["steps"] == 1000

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_multiple_updates_accumulate(self, mock_load_defaults):
        """Test that multiple update calls accumulate."""
        mock_load_defaults.return_value = {}

        builder = TorchTitanJobConfigBuilder()
        builder.update(SimpleNamespace(model=SimpleNamespace(name="llama3")))
        builder.update(SimpleNamespace(training=SimpleNamespace(steps=1000)))

        assert "model" in builder.config
        assert "training" in builder.config

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_later_update_merges_nested(self, mock_load_defaults):
        """Test that later updates merge nested structures."""
        mock_load_defaults.return_value = {}

        builder = TorchTitanJobConfigBuilder()
        builder.update(SimpleNamespace(model=SimpleNamespace(name="llama3", size="8B")))
        builder.update(SimpleNamespace(model=SimpleNamespace(size="70B", flavor="debug")))

        # Should have all three fields
        assert builder.config["model"]["name"] == "llama3"
        assert builder.config["model"]["size"] == "70B"
        assert builder.config["model"]["flavor"] == "debug"

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_update_returns_self(self, mock_load_defaults):
        """Test that update() returns self for chaining."""
        mock_load_defaults.return_value = {}

        builder = TorchTitanJobConfigBuilder()
        result = builder.update(SimpleNamespace(model=SimpleNamespace(name="llama3")))

        assert result is builder

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_chained_updates(self, mock_load_defaults):
        """Test chained update calls."""
        mock_load_defaults.return_value = {}

        builder = TorchTitanJobConfigBuilder()
        builder.update(SimpleNamespace(model=SimpleNamespace(name="llama3"))).update(
            SimpleNamespace(training=SimpleNamespace(steps=1000))
        )

        assert builder.config["model"]["name"] == "llama3"
        assert builder.config["training"]["steps"] == 1000

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_update_with_deeply_nested_namespace(self, mock_load_defaults):
        """Test that update() handles deeply nested SimpleNamespace structures."""
        mock_load_defaults.return_value = {}

        builder = TorchTitanJobConfigBuilder()

        # Create a deeply nested SimpleNamespace
        ns = SimpleNamespace(
            model=SimpleNamespace(name="llama3", size="8B"),
            training=SimpleNamespace(
                steps=1000, batch_size=8, optimizer=SimpleNamespace(lr=0.001, weight_decay=0.01)
            ),
        )

        builder.update(ns)

        # Check all levels are converted correctly
        assert builder.config["model"]["name"] == "llama3"
        assert builder.config["model"]["size"] == "8B"
        assert builder.config["training"]["steps"] == 1000
        assert builder.config["training"]["batch_size"] == 8
        assert builder.config["training"]["optimizer"]["lr"] == 0.001
        assert builder.config["training"]["optimizer"]["weight_decay"] == 0.01


class TestTorchTitanJobConfigBuilderToDict:
    """Test the to_dict() method."""

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_to_dict_merges_defaults_and_overrides(self, mock_load_defaults):
        """Test that to_dict() merges defaults with overrides."""
        mock_load_defaults.return_value = {
            "model": {"name": "default_model", "size": "8B"},
            "training": {"steps": 100},
        }

        builder = TorchTitanJobConfigBuilder()
        builder.update({"model": {"size": "70B"}, "training": {"steps": 1000}})

        result = builder.to_dict()

        # Check merged values
        assert result["model"]["name"] == "default_model"  # From defaults
        assert result["model"]["size"] == "70B"  # Overridden
        assert result["training"]["steps"] == 1000  # Overridden

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_to_dict_with_no_overrides(self, mock_load_defaults):
        """Test to_dict() with no overrides returns defaults."""
        mock_load_defaults.return_value = {"model": {"name": "llama3"}, "training": {"steps": 100}}

        builder = TorchTitanJobConfigBuilder()
        result = builder.to_dict()

        assert result == mock_load_defaults.return_value

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_to_dict_creates_new_dict_each_time(self, mock_load_defaults):
        """Test that to_dict() creates a new dict each time."""
        mock_load_defaults.return_value = {"model": {"name": "llama3"}}

        builder = TorchTitanJobConfigBuilder()
        result1 = builder.to_dict()
        result2 = builder.to_dict()

        # Should be different objects
        assert result1 is not result2

        # But with same content
        assert result1 == result2


class TestTorchTitanJobConfigBuilderToNamespace:
    """Test the to_namespace() method."""

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_to_namespace_returns_simplenamespace(self, mock_load_defaults):
        """Test that to_namespace() returns SimpleNamespace."""
        mock_load_defaults.return_value = {"model": {"name": "llama3"}, "training": {"steps": 1000}}

        builder = TorchTitanJobConfigBuilder()
        result = builder.to_namespace()

        assert isinstance(result, SimpleNamespace)
        assert isinstance(result.model, SimpleNamespace)
        assert isinstance(result.training, SimpleNamespace)

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_to_namespace_preserves_nested_structure(self, mock_load_defaults):
        """Test that nested dicts become nested SimpleNamespace."""
        mock_load_defaults.return_value = {
            "model": {"name": "llama3", "size": "8B"},
            "training": {"steps": 1000, "batch_size": 8},
        }

        builder = TorchTitanJobConfigBuilder()
        result = builder.to_namespace()

        assert result.model.name == "llama3"
        assert result.model.size == "8B"
        assert result.training.steps == 1000
        assert result.training.batch_size == 8

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_to_namespace_with_overrides(self, mock_load_defaults):
        """Test to_namespace() with overrides."""
        mock_load_defaults.return_value = {
            "model": {"name": "default", "size": "8B"},
            "training": {"steps": 100},
        }

        builder = TorchTitanJobConfigBuilder()
        builder.update({"model": {"name": "llama3"}, "training": {"steps": 1000}})

        result = builder.to_namespace()

        assert result.model.name == "llama3"  # Overridden
        assert result.model.size == "8B"  # From defaults
        assert result.training.steps == 1000  # Overridden


class TestTorchTitanJobConfigBuilderToJobConfig:
    """Test the to_job_config() method."""

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    @patch("primus.backends.torchtitan.argument_builder._dict_to_dataclass")
    def test_to_job_config_calls_dict_to_dataclass(self, mock_dict_to_dataclass, mock_load_defaults):
        """Test that to_job_config() calls _dict_to_dataclass."""
        mock_load_defaults.return_value = {"model": {"name": "llama3"}}
        mock_job_config = MagicMock()
        mock_dict_to_dataclass.return_value = mock_job_config

        builder = TorchTitanJobConfigBuilder()
        result = builder.to_job_config()

        # Should have called _dict_to_dataclass
        mock_dict_to_dataclass.assert_called_once()
        assert result == mock_job_config


class TestTorchTitanJobConfigBuilderFinalize:
    """Test the finalize() alias."""

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_finalize_is_alias_for_to_namespace(self, mock_load_defaults):
        """Test that finalize() is an alias for to_namespace()."""
        mock_load_defaults.return_value = {"model": {"name": "llama3"}}

        builder = TorchTitanJobConfigBuilder()
        result1 = builder.finalize()
        result2 = builder.to_namespace()

        # Both should return SimpleNamespace
        assert isinstance(result1, SimpleNamespace)
        assert isinstance(result2, SimpleNamespace)

        # Content should be the same
        assert vars(result1) == vars(result2)


class TestTorchTitanJobConfigBuilderIntegration:
    """Integration tests for complete workflow."""

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_complete_workflow(self, mock_load_defaults):
        """Test complete workflow: defaults → overrides → namespace."""
        mock_load_defaults.return_value = {
            "model": {"name": "default_model", "size": "8B", "flavor": "default"},
            "training": {"steps": 100, "batch_size": 8},
            "parallelism": {"tensor_parallel_degree": 1},
        }

        builder = TorchTitanJobConfigBuilder()

        # Add overrides
        builder.update(
            {
                "model": {"name": "llama3", "size": "70B"},  # Override name and size
                "training": {"steps": 1000},  # Override steps
            }
        )

        result = builder.finalize()

        # Check overrides
        assert result.model.name == "llama3"
        assert result.model.size == "70B"
        assert result.training.steps == 1000

        # Check defaults preserved
        assert result.model.flavor == "default"
        assert result.training.batch_size == 8
        assert result.parallelism.tensor_parallel_degree == 1

        # Check result type
        assert isinstance(result, SimpleNamespace)
        assert isinstance(result.model, SimpleNamespace)

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_builder_reusable(self, mock_load_defaults):
        """Test that builder can be reused with different updates."""
        mock_load_defaults.return_value = {"model": {"name": "default"}, "training": {"steps": 100}}

        builder = TorchTitanJobConfigBuilder()

        # First use
        builder.update({"model": {"name": "llama3"}})
        result1 = builder.finalize()
        assert result1.model.name == "llama3"

        # Second use (accumulates)
        builder.update({"model": {"name": "llama2"}})
        result2 = builder.finalize()
        assert result2.model.name == "llama2"  # Latest value

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_empty_overrides(self, mock_load_defaults):
        """Test builder with no overrides returns defaults."""
        mock_load_defaults.return_value = {"model": {"name": "llama3"}, "training": {"steps": 1000}}

        builder = TorchTitanJobConfigBuilder()
        result = builder.finalize()

        assert result.model.name == "llama3"
        assert result.training.steps == 1000


class TestAPIConsistency:
    """Test API consistency with MegatronArgBuilder."""

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_has_same_methods_as_megatron(self, mock_load_defaults):
        """Test that TorchTitanJobConfigBuilder has similar API to MegatronArgBuilder."""
        mock_load_defaults.return_value = {}

        builder = TorchTitanJobConfigBuilder()

        # Check key methods exist
        assert hasattr(builder, "update")
        assert hasattr(builder, "finalize")
        # Note: TorchTitan uses 'config' instead of 'overrides' (different design)
        assert hasattr(builder, "config")

        # Check update returns self (for chaining)
        assert builder.update(SimpleNamespace()) is builder

    @patch("primus.backends.torchtitan.argument_builder._load_torchtitan_defaults")
    def test_finalize_returns_simplenamespace_like_megatron(self, mock_load_defaults):
        """Test that finalize() returns SimpleNamespace (consistent with Megatron)."""
        mock_load_defaults.return_value = {"model": {"name": "llama3"}}

        builder = TorchTitanJobConfigBuilder()
        result = builder.finalize()

        # Should return SimpleNamespace (same as MegatronArgBuilder.finalize)
        assert isinstance(result, SimpleNamespace)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
