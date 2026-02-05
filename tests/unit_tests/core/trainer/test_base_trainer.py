###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for BaseTrainer."""

from types import SimpleNamespace

from primus.core.trainer.base_trainer import BaseTrainer


class DummyTrainer(BaseTrainer):
    """Minimal concrete implementation of BaseTrainer for testing."""

    def __init__(self, backend_args=None):
        super().__init__(backend_args=backend_args)
        self.train_calls: int = 0
        self.setup_calls: int = 0
        self.init_calls: int = 0

    def init(self, *args, **kwargs):
        """No-op init for testing."""
        self.init_calls += 1
        return None

    def setup(self, *args, **kwargs):
        """No-op setup for testing."""
        self.setup_calls += 1
        return None

    def train(self):
        self.train_calls += 1


class TestBaseTrainer:
    """Verify BaseTrainer interface and lifecycle."""

    def test_trainer_stores_backend_args(self):
        """Backend args should be stored on the trainer instance."""
        backend_args = SimpleNamespace(lr=1e-4, batch_size=32)
        trainer = DummyTrainer(backend_args=backend_args)

        assert trainer.backend_args is backend_args
        assert trainer.backend_args.lr == 1e-4
        assert trainer.backend_args.batch_size == 32

    def test_trainer_allows_none_backend_args(self):
        """Trainer should accept None as backend_args."""
        trainer = DummyTrainer(backend_args=None)
        assert trainer.backend_args is None

    def test_trainer_train_method_executes(self):
        """The train() method should execute correctly."""
        backend_args = SimpleNamespace(lr=1e-4)
        trainer = DummyTrainer(backend_args=backend_args)

        trainer.train()

        assert trainer.train_calls == 1

    def test_trainer_lifecycle_methods_exist(self):
        """Trainer should have setup, init, train, cleanup methods."""
        backend_args = SimpleNamespace()
        trainer = DummyTrainer(backend_args=backend_args)

        # All lifecycle methods should be callable
        trainer.setup()
        trainer.init()
        trainer.train()
        trainer.cleanup()

        assert trainer.setup_calls == 1
        assert trainer.init_calls == 1
        assert trainer.train_calls == 1

    def test_cleanup_accepts_on_error_flag(self):
        """cleanup() should accept on_error parameter."""
        trainer = DummyTrainer(backend_args=None)

        # Should not raise
        trainer.cleanup(on_error=False)
        trainer.cleanup(on_error=True)

    def test_trainer_has_distributed_env_attributes(self):
        """Trainer should expose distributed environment attributes."""
        trainer = DummyTrainer(backend_args=None)

        # These attributes should exist (values depend on env)
        assert hasattr(trainer, "rank")
        assert hasattr(trainer, "world_size")
        assert hasattr(trainer, "local_rank")
        assert hasattr(trainer, "master_addr")
        assert hasattr(trainer, "master_port")
