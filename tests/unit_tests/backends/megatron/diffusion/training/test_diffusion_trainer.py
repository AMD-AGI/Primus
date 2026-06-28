###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for DiffusionPretrainTrainer.

Tests initialization, setup, data provider delegation, forward step,
and scheduler lazy initialization.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from primus.backends.megatron.diffusion_trainer import DiffusionPretrainTrainer


def _build_diffusion_trainer(monkeypatch: pytest.MonkeyPatch, backend_args=None, use_mock_data=True):
    """Helper to build DiffusionPretrainTrainer with stubbed dependencies."""

    # Create a concrete subclass for testing (DiffusionPretrainTrainer is abstract)
    class ConcreteDiffusionTrainer(DiffusionPretrainTrainer):
        def create_model(self, pre_process=True, post_process=True):
            return Mock()

        def create_scheduler(self):
            return Mock()

        def get_task_encoder(self):
            return Mock()

    # Stub out MegatronBaseTrainer.__init__ to avoid real Megatron imports
    def dummy_base_init(self, backend_args=None, *args, **kwargs):
        self.backend_args = backend_args
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.master_addr = "localhost"
        self.master_port = 12345

    monkeypatch.setattr(
        "primus.backends.megatron.megatron_base_trainer.MegatronBaseTrainer.__init__",
        dummy_base_init,
    )

    # Silence logging
    monkeypatch.setattr(
        "primus.backends.megatron.diffusion_trainer.log_rank_0",
        lambda *args, **kwargs: None,
    )

    if backend_args is None:
        backend_args = SimpleNamespace(mock_data=use_mock_data)
    else:
        if not hasattr(backend_args, "mock_data"):
            backend_args.mock_data = use_mock_data

    return ConcreteDiffusionTrainer(backend_args=backend_args)


class TestDiffusionPretrainTrainer:
    """Tests for DiffusionPretrainTrainer."""

    def test_init_with_mock_data_creates_synthetic_provider(self, monkeypatch: pytest.MonkeyPatch):
        """Test that initialization with mock_data=True creates SyntheticDatasetProvider."""
        backend_args = SimpleNamespace(mock_data=True, model_type="flux")
        trainer = _build_diffusion_trainer(monkeypatch, backend_args)

        assert hasattr(trainer, "data_provider")
        assert trainer.data_provider is not None
        # Verify it's SyntheticDatasetProvider (check class name)
        assert "Synthetic" in type(trainer.data_provider).__name__

    def test_setup_calculates_data_parallel_size(self, monkeypatch: pytest.MonkeyPatch):
        """Test that setup() calculates data_parallel_size when not present."""
        backend_args = SimpleNamespace(
            mock_data=True,
            world_size=8,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )
        trainer = _build_diffusion_trainer(monkeypatch, backend_args)

        # Mock parent setup and other dependencies
        setup_calls = []

        def mock_parent_setup(self):
            setup_calls.append("parent_setup")

        monkeypatch.setattr(
            "primus.backends.megatron.megatron_base_trainer.MegatronBaseTrainer.setup",
            mock_parent_setup,
        )

        # Mock set_primus_global_variables
        monkeypatch.setattr(
            "primus.backends.megatron.training.global_vars.set_primus_global_variables",
            lambda args: None,
        )

        trainer.setup()

        # Verify data_parallel_size was calculated
        assert hasattr(trainer.backend_args, "data_parallel_size")
        assert trainer.backend_args.data_parallel_size == 4  # 8 / (2 * 1 * 1)
        assert "parent_setup" in setup_calls

    def test_setup_uses_existing_data_parallel_size(self, monkeypatch: pytest.MonkeyPatch):
        """Test that setup() uses existing data_parallel_size if present."""
        backend_args = SimpleNamespace(
            mock_data=True,
            data_parallel_size=16,
            world_size=8,  # This would normally calculate to 4, but we have existing value
        )
        trainer = _build_diffusion_trainer(monkeypatch, backend_args)

        # Mock parent setup
        monkeypatch.setattr(
            "primus.backends.megatron.megatron_base_trainer.MegatronBaseTrainer.setup",
            lambda self: None,
        )
        monkeypatch.setattr(
            "primus.backends.megatron.training.global_vars.set_primus_global_variables",
            lambda args: None,
        )

        trainer.setup()

        # Verify existing data_parallel_size was not overwritten
        assert trainer.backend_args.data_parallel_size == 16

    def test_setup_sets_model_provider(self, monkeypatch: pytest.MonkeyPatch):
        """Test that setup() sets model_provider with correct signature."""
        trainer = _build_diffusion_trainer(monkeypatch)

        # Mock create_model
        mock_model = Mock()
        trainer.create_model = lambda pre_process=True, post_process=True: mock_model

        # Mock parent setup
        monkeypatch.setattr(
            "primus.backends.megatron.megatron_base_trainer.MegatronBaseTrainer.setup",
            lambda self: None,
        )
        monkeypatch.setattr(
            "primus.backends.megatron.training.global_vars.set_primus_global_variables",
            lambda args: None,
        )

        trainer.setup()

        # Verify model_provider was set
        assert hasattr(trainer, "model_provider")
        assert callable(trainer.model_provider)

        # Verify model_provider signature matches Megatron's interface
        result = trainer.model_provider(
            pre_process=True,
            post_process=True,
            vp_stage=None,
            config=None,
            pg_collection=None,
        )
        assert result is mock_model

    def test_get_datasets_provider_returns_delegating_function(self, monkeypatch: pytest.MonkeyPatch):
        """Test that get_datasets_provider() returns a function that delegates to data_provider."""
        trainer = _build_diffusion_trainer(monkeypatch)

        # Replace data_provider with a mock that has is_distributed as an attribute (not property)
        mock_dataloaders = [Mock(), Mock(), Mock()]  # train, val, test

        class MockDataProvider:
            def __init__(self):
                self.is_distributed = True
                self.create_dataloaders = Mock(return_value=mock_dataloaders)

        mock_data_provider = MockDataProvider()
        trainer.data_provider = mock_data_provider

        # Mock get_args
        mock_args = SimpleNamespace()
        training_mod = types.SimpleNamespace(get_args=lambda: mock_args)
        monkeypatch.setitem(sys.modules, "megatron.training", training_mod)

        provider_func = trainer.get_datasets_provider()

        # Verify it's a function
        assert callable(provider_func)

        # Verify is_distributed flag is set
        assert provider_func.is_distributed is True

        # Call the provider function
        train_val_test_num_samples = [100, 10, 10]
        result = provider_func(train_val_test_num_samples, vp_stage=None)

        # Verify data_provider.create_dataloaders was called
        mock_data_provider.create_dataloaders.assert_called_once_with(
            trainer_config=mock_args,
            train_val_test_num_samples=train_val_test_num_samples,
            vp_stage=None,
        )

        # Verify result matches what data_provider returned
        assert result is mock_dataloaders

    def test_forward_step_calls_flux_forward_step_func(self, monkeypatch: pytest.MonkeyPatch):
        """Test that forward_step() calls flux_forward_step_func and creates loss function."""
        trainer = _build_diffusion_trainer(monkeypatch)

        # Mock scheduler
        mock_scheduler = Mock()
        trainer._scheduler = mock_scheduler

        # Mock flux_forward_step_func
        mock_noise_pred = Mock()
        mock_clean_latents = Mock()
        mock_noise = Mock()
        mock_loss_mask = Mock()
        mock_metrics = {"test_metric": 1.0}

        flux_forward_step_calls = []

        def mock_flux_forward_step(
            data_iterator,
            model,
            scheduler=None,
            use_guidance_embed=False,
            guidance_scale=None,
            timestep_sampler=None,
            cfg_dropout_prob=0.0,
            empty_t5_encodings=None,
            empty_clip_encodings=None,
            **kwargs,
        ):
            flux_forward_step_calls.append(
                (data_iterator, model, scheduler, use_guidance_embed, guidance_scale)
            )
            # 6-tuple per sibling commit 362ee36 (is_validation appended).
            return (
                mock_noise_pred,
                mock_clean_latents,
                mock_noise,
                mock_loss_mask,
                mock_metrics,
                False,
            )

        # Patch at the import location (it's imported inside forward_step)
        monkeypatch.setattr(
            "primus.backends.megatron.training.diffusion.forward_step.flux_forward_step_func",
            mock_flux_forward_step,
        )

        # Mock runtime_state
        trainer.runtime_state = Mock()
        trainer.runtime_state.update_metrics = Mock()

        mock_data_iterator = Mock()
        mock_model = Mock()

        output, loss_func = trainer.forward_step(mock_data_iterator, mock_model)

        # Verify flux_forward_step_func was called
        assert len(flux_forward_step_calls) == 1
        assert flux_forward_step_calls[0][0] is mock_data_iterator
        assert flux_forward_step_calls[0][1] is mock_model
        assert flux_forward_step_calls[0][2] is mock_scheduler

        # Verify values were stored
        assert trainer._last_clean_latents is mock_clean_latents
        assert trainer._last_noise is mock_noise
        assert trainer._last_loss_mask is mock_loss_mask

        # Verify metrics were updated
        trainer.runtime_state.update_metrics.assert_called_once_with(mock_metrics)

        # Verify output is noise_pred
        assert output is mock_noise_pred

        # Verify loss_func is callable
        assert callable(loss_func)

        # Test loss function - mock compute_flow_matching_loss to avoid real computation
        # It's imported inside the loss function, so patch at the import location
        mock_loss = Mock()
        mock_loss.detach.return_value = mock_loss
        monkeypatch.setattr(
            "primus.backends.megatron.training.diffusion.loss_computation.compute_flow_matching_loss",
            lambda *args, **kwargs: mock_loss,
        )

        loss_result = loss_func(mock_noise_pred, non_loss_data=False)
        assert len(loss_result) == 2  # (loss, metrics_dict)
        # Verify the metrics dict has the expected key
        assert "reduced_train_loss" in loss_result[1]

    def test_forward_step_loss_func_with_non_loss_data(self, monkeypatch: pytest.MonkeyPatch):
        """Test that loss function returns output_tensor when non_loss_data=True."""
        trainer = _build_diffusion_trainer(monkeypatch)

        # Mock forward_step components
        mock_scheduler = Mock()
        trainer._scheduler = mock_scheduler

        mock_noise_pred = Mock()
        mock_clean_latents = Mock()
        mock_noise = Mock()
        mock_loss_mask = Mock()
        mock_metrics = {}

        # Patch at the import location (it's imported inside forward_step).
        # 6-tuple per sibling commit 362ee36 (is_validation appended).
        monkeypatch.setattr(
            "primus.backends.megatron.training.diffusion.forward_step.flux_forward_step_func",
            lambda *args, **kwargs: (
                mock_noise_pred,
                mock_clean_latents,
                mock_noise,
                mock_loss_mask,
                mock_metrics,
                False,
            ),
        )

        trainer.runtime_state = Mock()
        trainer.runtime_state.update_metrics = Mock()

        output, loss_func = trainer.forward_step(Mock(), Mock())

        # Test loss function with non_loss_data=True
        result = loss_func(output, non_loss_data=True)

        # Should return output_tensor directly
        assert result is output
