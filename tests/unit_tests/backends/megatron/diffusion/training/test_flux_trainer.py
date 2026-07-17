# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for FluxPretrainTrainer.

Tests initialization, configuration, method overrides, scheduler setup,
and CFG dropout encoding discovery.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.flux_pretrain_trainer import FluxPretrainTrainer


def _build_flux_trainer(monkeypatch: pytest.MonkeyPatch, backend_args=None):
    """Helper to build FluxPretrainTrainer with stubbed dependencies."""

    # Stub out MegatronBaseTrainer.__init__ to avoid real Megatron imports
    def dummy_init(self, backend_args: any = None):
        self.backend_args = backend_args

    monkeypatch.setattr(
        "primus.backends.megatron.megatron_base_trainer.MegatronBaseTrainer.__init__",
        dummy_init,
    )

    # Silence logging
    monkeypatch.setattr(
        "primus.backends.megatron.flux_pretrain_trainer.log_rank_0",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "primus.backends.megatron.diffusion_trainer.log_rank_0",
        lambda *args, **kwargs: None,
    )

    if backend_args is None:
        backend_args = SimpleNamespace(
            guidance_embed=False,
            guidance_scale=3.5,
            num_train_timesteps=1000,
            scheduler_shift=1.0,
            use_dynamic_shifting=False,
        )

    return FluxPretrainTrainer(backend_args=backend_args)


class TestFluxPretrainTrainer:
    """Tests for FluxPretrainTrainer."""

    def test_flux_trainer_scheduler_creation(self, monkeypatch: pytest.MonkeyPatch):
        """Test that scheduler is created correctly."""
        backend_args = SimpleNamespace(
            guidance_embed=False,
            guidance_scale=3.5,
            num_train_timesteps=1000,
            scheduler_shift=1.0,
            use_dynamic_shifting=False,
        )
        trainer = _build_flux_trainer(monkeypatch, backend_args)

        # Create scheduler
        scheduler = trainer.create_scheduler()

        # Should be FlowMatchEulerDiscreteScheduler
        from primus.backends.megatron.training.diffusion.schedulers import (
            FlowMatchEulerDiscreteScheduler,
        )

        assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
        assert scheduler.num_train_timesteps == 1000
        assert scheduler.shift == 1.0
        assert scheduler.use_dynamic_shifting is False

    def test_flux_trainer_scheduler_with_dynamic_shifting(self, monkeypatch: pytest.MonkeyPatch):
        """Test scheduler creation with dynamic shifting enabled."""
        backend_args = SimpleNamespace(
            guidance_embed=False,
            guidance_scale=3.5,
            num_train_timesteps=1000,
            scheduler_shift=1.0,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
        )
        trainer = _build_flux_trainer(monkeypatch, backend_args)

        scheduler = trainer.create_scheduler()

        assert scheduler.use_dynamic_shifting is True
        assert scheduler.base_shift == 0.5
        assert scheduler.max_shift == 1.15

    def test_flux_trainer_scheduler_lazy_initialization(self, monkeypatch: pytest.MonkeyPatch):
        """Test that scheduler is lazily initialized via property."""
        trainer = _build_flux_trainer(monkeypatch)

        # Scheduler should not be created yet
        assert trainer._scheduler is None

        # Accessing scheduler property should create it
        scheduler1 = trainer.scheduler
        assert scheduler1 is not None
        assert trainer._scheduler is scheduler1

        # Accessing again should return same instance
        scheduler2 = trainer.scheduler
        assert scheduler2 is scheduler1


def _create_encoding_files(directory):
    """Create minimal t5_empty.npy and clip_empty.npy in a directory."""
    os.makedirs(directory, exist_ok=True)
    np.save(os.path.join(directory, "t5_empty.npy"), np.zeros((1, 256, 4096), dtype=np.float32))
    np.save(os.path.join(directory, "clip_empty.npy"), np.zeros((1, 768), dtype=np.float32))


class TestCFGDropoutDiscovery:
    """Tests for CFG dropout empty encoding discovery and error handling."""

    def test_discover_encodings_inside_data_path(self, monkeypatch, tmp_path):
        """Auto-discovers encodings at {data_path}/empty_encodings/."""
        monkeypatch.setattr(
            "primus.backends.megatron.flux_pretrain_trainer.log_rank_0",
            lambda *a, **kw: None,
        )
        enc_dir = tmp_path / "empty_encodings"
        _create_encoding_files(str(enc_dir))

        params = SimpleNamespace(data_path=str(tmp_path))
        result = FluxPretrainTrainer._discover_empty_encodings(params)

        assert result == str(enc_dir)

    def test_discover_encodings_alongside_data_path(self, monkeypatch, tmp_path):
        """Falls back to {data_path}/../empty_encodings/ when not inside."""
        monkeypatch.setattr(
            "primus.backends.megatron.flux_pretrain_trainer.log_rank_0",
            lambda *a, **kw: None,
        )
        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        enc_dir = tmp_path / "empty_encodings"
        _create_encoding_files(str(enc_dir))

        params = SimpleNamespace(data_path=str(data_dir))
        result = FluxPretrainTrainer._discover_empty_encodings(params)

        assert result == str(enc_dir)

    def test_discover_encodings_explicit_path(self, tmp_path):
        """Explicit empty_encodings_path takes priority."""
        explicit_dir = tmp_path / "custom_encodings"
        _create_encoding_files(str(explicit_dir))

        inside_dir = tmp_path / "data" / "empty_encodings"
        _create_encoding_files(str(inside_dir))

        params = SimpleNamespace(
            data_path=str(tmp_path / "data"),
            empty_encodings_path=str(explicit_dir),
        )
        result = FluxPretrainTrainer._discover_empty_encodings(params)

        assert result == str(explicit_dir)

    def test_discover_encodings_returns_none_when_missing(self, tmp_path):
        """Returns None when no encoding files exist anywhere."""
        params = SimpleNamespace(data_path=str(tmp_path))
        result = FluxPretrainTrainer._discover_empty_encodings(params)

        assert result is None

    def test_discover_encodings_returns_none_for_partial_files(self, tmp_path):
        """Returns None when only one of the two required files exists."""
        enc_dir = tmp_path / "empty_encodings"
        enc_dir.mkdir()
        np.save(str(enc_dir / "t5_empty.npy"), np.zeros((1, 256, 4096), dtype=np.float32))

        params = SimpleNamespace(data_path=str(tmp_path))
        result = FluxPretrainTrainer._discover_empty_encodings(params)

        assert result is None

    def test_discover_encodings_handles_list_data_path(self, monkeypatch, tmp_path):
        """Handles data_path passed as a list (takes first element)."""
        monkeypatch.setattr(
            "primus.backends.megatron.flux_pretrain_trainer.log_rank_0",
            lambda *a, **kw: None,
        )
        enc_dir = tmp_path / "empty_encodings"
        _create_encoding_files(str(enc_dir))

        params = SimpleNamespace(data_path=[str(tmp_path), "/nonexistent"])
        result = FluxPretrainTrainer._discover_empty_encodings(params)

        assert result == str(enc_dir)

    def test_cfg_dropout_raises_when_no_encodings_found(self, monkeypatch, tmp_path):
        """Real data without encodings raises FileNotFoundError with instructions."""
        backend_args = SimpleNamespace(
            cfg_dropout_prob=0.1,
            mock_data=False,
            data_path=str(tmp_path),
            tensor_model_parallel_size=1,
        )

        with pytest.raises(FileNotFoundError, match="primus data diffusion-encoded"):
            _build_flux_trainer(monkeypatch, backend_args)
