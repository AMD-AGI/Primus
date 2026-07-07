###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for BaseTrainer.

Tests initialization, MRO handling, distributed environment setup, and
backend_args handling for the universal base trainer class.
"""

from types import SimpleNamespace

import pytest

from primus.core.trainer.base_trainer import BaseTrainer


class TestBaseTrainer:
    """Tests for BaseTrainer initialization and behavior."""

    def test_init_sets_distributed_environment(self, monkeypatch: pytest.MonkeyPatch):
        """Test that distributed environment attributes are set correctly."""
        mock_dist_env = {
            "rank": 2,
            "world_size": 8,
            "local_rank": 1,
            "master_addr": "192.168.1.1",
            "master_port": 54321,
        }

        monkeypatch.setattr(
            "primus.core.trainer.base_trainer.get_torchrun_env",
            lambda: mock_dist_env,
        )
        monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
        monkeypatch.setattr("sys.stderr.write", lambda *args, **kwargs: None)

        # Create a concrete subclass for testing
        class ConcreteBaseTrainer(BaseTrainer):
            def setup(self):
                pass

            def init(self):
                pass

            def train(self):
                pass

        trainer = ConcreteBaseTrainer(backend_args=SimpleNamespace())

        assert trainer.rank == 2
        assert trainer.world_size == 8
        assert trainer.local_rank == 1
        assert trainer.master_addr == "192.168.1.1"
        assert trainer.master_port == 54321

    def test_init_mro_with_base_module(self, monkeypatch: pytest.MonkeyPatch):
        """Test MRO handling when BaseModule IS in inheritance chain (legacy pattern)."""
        from primus.modules.base_module import BaseModule

        # Create a class that inherits from both BaseTrainer and BaseModule
        class LegacyTrainer(BaseTrainer, BaseModule):
            def setup(self):
                pass

            def init(self):
                pass

            def train(self):
                pass

            def run(self):
                pass  # BaseModule requires run() method

        mock_dist_env = {
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "master_addr": "localhost",
            "master_port": 12345,
        }

        monkeypatch.setattr(
            "primus.core.trainer.base_trainer.get_torchrun_env",
            lambda: mock_dist_env,
        )
        monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
        monkeypatch.setattr("sys.stderr.write", lambda *args, **kwargs: None)

        # Mock BaseModule.__init__ to track if it's called with kwargs and provide required args
        base_module_init_called = []

        def tracked_init(
            self, module_name="test", primus_config=None, module_rank=0, module_world_size=1, **kwargs
        ):
            base_module_init_called.append(("kwargs", kwargs))
            # Don't call original_init, just track the call

        monkeypatch.setattr(BaseModule, "__init__", tracked_init)

        backend_args = SimpleNamespace()
        trainer = LegacyTrainer(
            backend_args=backend_args,
            some_kwarg="value",
            module_name="test",
            primus_config=SimpleNamespace(),
            module_rank=0,
            module_world_size=1,
        )

        # Verify BaseModule.__init__ was called with kwargs
        assert len(base_module_init_called) > 0
        assert "some_kwarg" in base_module_init_called[0][1]
        assert trainer.backend_args is backend_args
