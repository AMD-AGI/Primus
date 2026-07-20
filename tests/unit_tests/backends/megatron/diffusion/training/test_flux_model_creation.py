###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for FluxPretrainTrainer model creation.

Tests high-value Flux behavior in Primus:
- backend_args -> FluxConfig mapping/defaults
- model construction wiring
- torch_compile settings propagation semantics
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.flux_pretrain_trainer import FluxPretrainTrainer


def _build_flux_trainer(monkeypatch: pytest.MonkeyPatch, backend_args=None):
    """Helper to build FluxPretrainTrainer with stubbed dependencies."""

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

    # Stub out DiffusionPretrainTrainer.__init__
    def dummy_diffusion_init(self, *args, **kwargs):
        self.backend_args = kwargs.get("backend_args")
        if self.backend_args is None and args:
            self.backend_args = args[0]
        if self.backend_args is None:
            self.backend_args = SimpleNamespace()
        self._scheduler = None
        self.data_provider = Mock()

    monkeypatch.setattr(
        "primus.backends.megatron.diffusion_trainer.DiffusionPretrainTrainer.__init__",
        dummy_diffusion_init,
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
            mock_data=True,
            guidance_embed=False,
            guidance_scale=3.5,
            num_train_timesteps=1000,
            scheduler_shift=1.0,
            use_dynamic_shifting=False,
        )

    return FluxPretrainTrainer(backend_args=backend_args)


class TestFluxModelCreation:
    """Tests for FluxPretrainTrainer model creation."""

    def test_create_model_calls_build_flux_config(self, monkeypatch: pytest.MonkeyPatch):
        """Test that create_model() calls _build_flux_config_from_yaml()."""
        trainer = _build_flux_trainer(monkeypatch)

        build_config_calls = []
        built_config = None

        def tracked_build_config(self):
            nonlocal built_config
            build_config_calls.append(1)
            from primus.backends.megatron.core.models.diffusion.flux.config import (
                FluxConfig,
            )

            built_config = FluxConfig.flux_535m()
            return built_config

        trainer._build_flux_config_from_yaml = tracked_build_config.__get__(trainer, type(trainer))

        # Mock Flux model
        mock_flux_model = Mock()
        mock_flux_model.parameters.return_value = []
        flux_ctor = Mock(return_value=mock_flux_model)
        monkeypatch.setattr(
            "primus.backends.megatron.core.models.diffusion.flux.model.Flux",
            flux_ctor,
        )

        # Mock get_args
        mock_args = SimpleNamespace(rank=0)
        training_mod = types.SimpleNamespace(get_args=lambda: mock_args)
        monkeypatch.setitem(sys.modules, "megatron.training", training_mod)

        result = trainer.create_model()

        # Verify _build_flux_config_from_yaml was called
        assert len(build_config_calls) == 1
        flux_ctor.assert_called_once()
        assert flux_ctor.call_args.kwargs["backend"] is None
        assert flux_ctor.call_args.kwargs["config"] is built_config
        assert result is mock_flux_model

    def test_build_flux_config_from_yaml_extracts_parameters(self, monkeypatch: pytest.MonkeyPatch):
        """Test that _build_flux_config_from_yaml() extracts parameters from backend_args."""
        backend_args = SimpleNamespace(
            mock_data=True,
            # Model architecture params
            num_joint_layers=24,
            num_single_layers=8,
            hidden_size=2048,
            num_attention_heads=16,
            ffn_hidden_size=8192,
            # Precision
            bf16=True,
            fp16=False,
            params_dtype=torch.bfloat16,
            # Transformer impl
            transformer_impl="local",
        )

        trainer = _build_flux_trainer(monkeypatch, backend_args)

        config = trainer._build_flux_config_from_yaml()

        # Verify config attributes were set
        assert config.num_joint_layers == 24
        assert config.num_single_layers == 8
        assert config.hidden_size == 2048
        assert config.num_attention_heads == 16
        assert config.ffn_hidden_size == 8192
        assert config.bf16 is True
        assert config.fp16 is False
        assert config.params_dtype == torch.bfloat16
        assert config.transformer_impl == "local"

    def test_build_flux_config_from_yaml_torch_compile_settings(self, monkeypatch: pytest.MonkeyPatch):
        """Test that torch_compile settings are extracted from backend_args."""
        backend_args = SimpleNamespace(
            mock_data=True,
            torch_compile=SimpleNamespace(
                enable=True,
                backend="inductor",
                mode="reduce-overhead",
                fullgraph=True,
            ),
        )

        trainer = _build_flux_trainer(monkeypatch, backend_args)

        config = trainer._build_flux_config_from_yaml()

        # Verify torch_compile settings
        assert config.enable_torch_compile is True
        assert config.torch_compile_backend == "inductor"
        assert config.torch_compile_mode == "reduce-overhead"
        assert config.torch_compile_fullgraph is True

    def test_create_model_sets_torch_compile_on_args(self, monkeypatch: pytest.MonkeyPatch):
        """Test that create_model() sets torch_compile attributes on args."""
        backend_args = SimpleNamespace(
            mock_data=True,
            torch_compile=SimpleNamespace(
                enable=True,
                backend="inductor",
                mode="reduce-overhead",
                fullgraph=True,
            ),
        )

        trainer = _build_flux_trainer(monkeypatch, backend_args)

        # Mock Flux model
        mock_flux_model = Mock()
        mock_flux_model.parameters.return_value = []
        monkeypatch.setattr(
            "primus.backends.megatron.core.models.diffusion.flux.model.Flux",
            lambda *args, **kwargs: mock_flux_model,
        )

        # Mock get_args
        mock_args = SimpleNamespace(rank=0)
        training_mod = types.SimpleNamespace(get_args=lambda: mock_args)
        monkeypatch.setitem(sys.modules, "megatron.training", training_mod)

        trainer.create_model()

        # Verify torch_compile attributes were set on args
        assert mock_args.enable_torch_compile is True
        assert mock_args.torch_compile_backend == "inductor"
        assert mock_args.torch_compile_mode == "reduce-overhead"
        assert mock_args.torch_compile_fullgraph is True

    def test_build_flux_config_from_yaml_fp8_settings(self, monkeypatch: pytest.MonkeyPatch):
        """Test that FP8 fields are extracted from backend_args into FluxConfig."""
        backend_args = SimpleNamespace(
            mock_data=True,
            fp8="e4m3",
            fp8_recipe="tensorwise",
            fp8_margin=0,
            fp8_amax_history_len=1,
            fp8_amax_compute_algo="most_recent",
            fp8_wgrad=True,
            fp8_dot_product_attention=False,
            fp8_multi_head_attention=False,
        )

        trainer = _build_flux_trainer(monkeypatch, backend_args)

        config = trainer._build_flux_config_from_yaml()

        assert config.fp8 == "e4m3"
        assert config.fp8_recipe == "tensorwise"
        assert config.fp8_margin == 0
        assert config.fp8_amax_history_len == 1
        assert config.fp8_amax_compute_algo == "most_recent"
        assert config.fp8_wgrad is True
        assert config.fp8_dot_product_attention is False
        assert config.fp8_multi_head_attention is False

    def test_create_model_does_not_overwrite_existing_torch_compile_attrs(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that create_model() doesn't overwrite existing torch_compile attributes."""
        backend_args = SimpleNamespace(
            mock_data=True,
            torch_compile=SimpleNamespace(
                enable=True,
                backend="inductor",
                mode="reduce-overhead",
                fullgraph=True,
            ),
        )

        trainer = _build_flux_trainer(monkeypatch, backend_args)

        # Mock Flux model
        mock_flux_model = Mock()
        mock_flux_model.parameters.return_value = []
        monkeypatch.setattr(
            "primus.backends.megatron.core.models.diffusion.flux.model.Flux",
            lambda *args, **kwargs: mock_flux_model,
        )

        # Mock get_args with existing attributes
        mock_args = SimpleNamespace(
            rank=0,
            enable_torch_compile=False,  # Already set
            torch_compile_backend="aot_eager",  # Already set
        )
        training_mod = types.SimpleNamespace(get_args=lambda: mock_args)
        monkeypatch.setitem(sys.modules, "megatron.training", training_mod)

        trainer.create_model()

        # Verify existing attributes were NOT overwritten
        assert mock_args.enable_torch_compile is False
        assert mock_args.torch_compile_backend == "aot_eager"
        # New attributes should still be set
        assert mock_args.torch_compile_mode == "reduce-overhead"
        assert mock_args.torch_compile_fullgraph is True
