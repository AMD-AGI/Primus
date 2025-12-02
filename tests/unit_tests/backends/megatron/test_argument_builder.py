###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for MegatronArgBuilder.

Focus areas:
    1. Parameter filtering: only Megatron-recognized params accepted
    2. Override mechanism: Primus config overrides defaults
    3. Distributed env injection: automatic injection of world_size, rank, local_rank
    4. None value handling: None values are ignored
    5. Integration: complete workflow from config to final namespace
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from primus.backends.megatron.argument_builder import MegatronArgBuilder


class TestMegatronArgBuilderFiltering:
    """Test parameter filtering: only Megatron params are accepted."""

    @patch("primus.backends.megatron.argument_builder._load_megatron_defaults")
    def test_only_megatron_params_accepted(self, mock_load_defaults):
        """Test that only parameters in Megatron defaults are accepted."""
        # Mock Megatron defaults
        mock_load_defaults.return_value = {
            "num_layers": 12,
            "hidden_size": 768,
            "num_attention_heads": 12,
        }

        builder = MegatronArgBuilder()
        builder.update(
            {
                "num_layers": 32,  # ✓ Megatron param
                "hidden_size": 4096,  # ✓ Megatron param
                "disable_mlflow": True,  # ✗ Primus param (filtered)
                "file_sink_level": "DEBUG",  # ✗ Primus param (filtered)
            }
        )

        # Only Megatron params should be in overrides
        assert "num_layers" in builder.overrides
        assert "hidden_size" in builder.overrides
        assert "disable_mlflow" not in builder.overrides
        assert "file_sink_level" not in builder.overrides

        assert builder.overrides["num_layers"] == 32
        assert builder.overrides["hidden_size"] == 4096

    @patch("primus.backends.megatron.argument_builder._load_megatron_defaults")
    def test_none_values_override_defaults(self, mock_load_defaults):
        """Test that None values are kept and override defaults."""
        mock_load_defaults.return_value = {
            "num_layers": 12,
            "hidden_size": 768,
        }

        builder = MegatronArgBuilder()
        builder.update(
            {
                "num_layers": None,  # Should override default with None
                "hidden_size": 4096,  # Should be accepted
            }
        )

        assert "num_layers" in builder.overrides
        assert builder.overrides["num_layers"] is None
        assert "hidden_size" in builder.overrides
        assert builder.overrides["hidden_size"] == 4096

    @patch("primus.backends.megatron.argument_builder._load_megatron_defaults")
    def test_empty_update(self, mock_load_defaults):
        """Test update with empty dict."""
        mock_load_defaults.return_value = {"num_layers": 12}

        builder = MegatronArgBuilder()
        builder.update({})

        assert len(builder.overrides) == 0

    @patch("primus.backends.megatron.argument_builder._load_megatron_defaults")
    def test_update_accepts_simplenamespace(self, mock_load_defaults):
        """Test that update() accepts SimpleNamespace as input."""
        mock_load_defaults.return_value = {
            "num_layers": 12,
            "hidden_size": 768,
        }

        builder = MegatronArgBuilder()
        ns = SimpleNamespace(num_layers=24, hidden_size=1024, extra_primus_param=True)

        builder.update(ns)

        assert builder.overrides["num_layers"] == 24
        assert builder.overrides["hidden_size"] == 1024
        # Primus-only field should be ignored
        assert "extra_primus_param" not in builder.overrides


class TestMegatronArgBuilderOverrides:
    """Test override mechanism: Primus config overrides defaults."""

    @patch("primus.backends.megatron.argument_builder.get_torchrun_env")
    @patch("primus.backends.megatron.argument_builder._load_megatron_defaults")
    def test_primus_overrides_defaults(self, mock_load_defaults, mock_dist_info):
        """Test that Primus config overrides Megatron defaults."""
        mock_load_defaults.return_value = {
            "num_layers": 12,
            "hidden_size": 768,
            "batch_size": 16,
        }
        mock_dist_info.return_value = {"world_size": 8, "rank": 0, "local_rank": 0}

        builder = MegatronArgBuilder()
        builder.update(
            {
                "num_layers": 32,  # Override default 12
                "hidden_size": 4096,  # Override default 768
                # batch_size not provided, should use default 16
            }
        )

        result = builder.finalize()

        # Check overrides
        assert result.num_layers == 32
        assert result.hidden_size == 4096
        # Check default
        assert result.batch_size == 16

    @patch("primus.backends.megatron.argument_builder.get_torchrun_env")
    @patch("primus.backends.megatron.argument_builder._load_megatron_defaults")
    def test_multiple_updates(self, mock_load_defaults, mock_dist_info):
        """Test that multiple update calls accumulate overrides."""
        mock_load_defaults.return_value = {
            "num_layers": 12,
            "hidden_size": 768,
            "batch_size": 16,
        }
        mock_dist_info.return_value = {"world_size": 8, "rank": 0, "local_rank": 0}

        builder = MegatronArgBuilder()
        builder.update({"num_layers": 32})
        builder.update({"hidden_size": 4096})

        result = builder.finalize()

        assert result.num_layers == 32
        assert result.hidden_size == 4096

    @patch("primus.backends.megatron.argument_builder.get_torchrun_env")
    @patch("primus.backends.megatron.argument_builder._load_megatron_defaults")
    def test_later_update_overwrites(self, mock_load_defaults, mock_dist_info):
        """Test that later updates overwrite earlier ones."""
        mock_load_defaults.return_value = {"num_layers": 12}
        mock_dist_info.return_value = {"world_size": 8, "rank": 0, "local_rank": 0}

        builder = MegatronArgBuilder()
        builder.update({"num_layers": 32})
        builder.update({"num_layers": 64})  # Overwrite previous

        result = builder.finalize()

        assert result.num_layers == 64


class TestMegatronArgBuilderDistributedEnv:
    """Test distributed environment injection."""

    @patch("primus.backends.megatron.argument_builder.get_torchrun_env")
    @patch("primus.backends.megatron.argument_builder._load_megatron_defaults")
    def test_distributed_env_injected(self, mock_load_defaults, mock_dist_info):
        """Test that distributed env is automatically injected."""
        mock_load_defaults.return_value = {"num_layers": 12}
        mock_dist_info.return_value = {"world_size": 8, "rank": 3, "local_rank": 3}

        builder = MegatronArgBuilder()
        result = builder.finalize()

        assert result.world_size == 8
        assert result.rank == 3
        assert result.local_rank == 3

    @patch("primus.backends.megatron.argument_builder.get_torchrun_env")
    @patch("primus.backends.megatron.argument_builder._load_megatron_defaults")
    def test_distributed_env_overrides_config(self, mock_load_defaults, mock_dist_info):
        """Test that distributed env overrides any config values."""
        mock_load_defaults.return_value = {
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
        }
        mock_dist_info.return_value = {"world_size": 8, "rank": 5, "local_rank": 5}

        builder = MegatronArgBuilder()
        builder.update(
            {
                "world_size": 4,  # This will be overridden by dist_env
                "rank": 2,  # This will be overridden by dist_env
            }
        )

        result = builder.finalize()

        # Distributed env should win
        assert result.world_size == 8
        assert result.rank == 5
        assert result.local_rank == 5


class TestMegatronArgBuilderIntegration:
    """Integration tests for complete workflow."""

    @patch("primus.backends.megatron.argument_builder.get_torchrun_env")
    @patch("primus.backends.megatron.argument_builder._load_megatron_defaults")
    def test_complete_workflow(self, mock_load_defaults, mock_dist_info):
        """Test complete workflow: defaults → overrides → dist env → namespace."""
        mock_load_defaults.return_value = {
            "num_layers": 12,
            "hidden_size": 768,
            "batch_size": 16,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
        }
        mock_dist_info.return_value = {"world_size": 8, "rank": 0, "local_rank": 0}

        builder = MegatronArgBuilder()
        builder.update(
            {
                "num_layers": 32,  # Override
                "hidden_size": 4096,  # Override
                "disable_mlflow": True,  # Filtered (not Megatron param)
            }
        )

        result = builder.finalize()

        # Check overrides
        assert result.num_layers == 32
        assert result.hidden_size == 4096

        # Check defaults
        assert result.batch_size == 16

        # Check distributed env
        assert result.world_size == 8
        assert result.rank == 0
        assert result.local_rank == 0

        # Check filtered params not present
        assert not hasattr(result, "disable_mlflow")

        # Check result is SimpleNamespace
        assert isinstance(result, SimpleNamespace)

    @patch("primus.backends.megatron.argument_builder.get_torchrun_env")
    @patch("primus.backends.megatron.argument_builder._load_megatron_defaults")
    def test_finalize_creates_new_namespace_each_time(self, mock_load_defaults, mock_dist_info):
        """Test that finalize() creates a new namespace each time."""
        mock_load_defaults.return_value = {"num_layers": 12}
        mock_dist_info.return_value = {"world_size": 8, "rank": 0, "local_rank": 0}

        builder = MegatronArgBuilder()
        builder.update({"num_layers": 32})

        result1 = builder.finalize()
        result2 = builder.finalize()

        # Should be different objects
        assert result1 is not result2

        # But with same content
        assert result1.num_layers == result2.num_layers

    @patch("primus.backends.megatron.argument_builder.get_torchrun_env")
    @patch("primus.backends.megatron.argument_builder._load_megatron_defaults")
    def test_builder_reusable(self, mock_load_defaults, mock_dist_info):
        """Test that builder can be reused with different updates."""
        mock_load_defaults.return_value = {"num_layers": 12}
        mock_dist_info.return_value = {"world_size": 8, "rank": 0, "local_rank": 0}

        builder = MegatronArgBuilder()

        # First use
        builder.update({"num_layers": 32})
        result1 = builder.finalize()
        assert result1.num_layers == 32

        # Second use (accumulates)
        builder.update({"num_layers": 64})
        result2 = builder.finalize()
        assert result2.num_layers == 64  # Latest value


class TestMegatronArgBuilderChaining:
    """Test method chaining support."""

    @patch("primus.backends.megatron.argument_builder._load_megatron_defaults")
    def test_update_returns_self(self, mock_load_defaults):
        """Test that update() returns self for chaining."""
        mock_load_defaults.return_value = {"num_layers": 12, "hidden_size": 768}

        builder = MegatronArgBuilder()
        result = builder.update({"num_layers": 32})

        assert result is builder

    @patch("primus.backends.megatron.argument_builder.get_torchrun_env")
    @patch("primus.backends.megatron.argument_builder._load_megatron_defaults")
    def test_chained_updates(self, mock_load_defaults, mock_dist_info):
        """Test chained update calls."""
        mock_load_defaults.return_value = {"num_layers": 12, "hidden_size": 768}
        mock_dist_info.return_value = {"world_size": 8, "rank": 0, "local_rank": 0}

        builder = MegatronArgBuilder()
        result = builder.update({"num_layers": 32}).update({"hidden_size": 4096}).finalize()

        assert result.num_layers == 32
        assert result.hidden_size == 4096


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
