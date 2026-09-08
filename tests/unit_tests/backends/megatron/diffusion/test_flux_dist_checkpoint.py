# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Distributed-checkpoint regression tests for Flux's heterogeneous layers."""

import pytest
import torch
from megatron.core.dist_checkpointing import load, save

from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.model import Flux
from tests.utils import PrimusUT


def _tiny_flux_config() -> FluxConfig:
    """Build the smallest useful joint+single Flux model for checkpoint tests."""
    return FluxConfig(
        num_joint_layers=1,
        num_single_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        ffn_hidden_size=64,
        in_channels=4,
        context_dim=16,
        vec_in_dim=16,
        model_channels=16,
        axes_dim=(2, 2, 4),
        transformer_impl="local",
        params_dtype=torch.float32,
    )


def _runtime_device() -> torch.device:
    """Use CUDA when available, otherwise run on CPU for CI coverage."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestFluxDistCheckpoint(PrimusUT):
    """Verify that joint and single blocks have independent checkpoint schemas."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        """Initialize single-rank model parallelism."""

    def test_adaln_uses_distinct_per_layer_sharded_keys(self):
        """The 6-chunk and 3-chunk adaLN weights must not share a global key."""
        model = Flux(_tiny_flux_config()).to(_runtime_device())
        sharded_state_dict = model.sharded_state_dict()

        double_local_key = "transformer.layers.0.adaln.adaLN_modulation.1.weight"
        single_local_key = "transformer.layers.1.adaln.adaLN_modulation.1.weight"
        double_weight = sharded_state_dict[double_local_key]
        single_weight = sharded_state_dict[single_local_key]

        assert double_weight.key == double_local_key
        assert single_weight.key == single_local_key
        assert double_weight.key != single_weight.key
        assert double_weight.prepend_axis_num == 0
        assert single_weight.prepend_axis_num == 0
        assert double_weight.global_shape == (6 * model.hidden_size, model.hidden_size)
        assert single_weight.global_shape == (3 * model.hidden_size, model.hidden_size)

    def test_torch_dist_save_load_round_trip(self, tmp_path):
        """Save and restore both adaLN shapes through the real torch_dist backend."""
        source = Flux(_tiny_flux_config()).to(_runtime_device())
        double_weight = source.transformer.layers[0].adaln.adaLN_modulation[-1].weight
        single_weight = source.transformer.layers[1].adaln.adaLN_modulation[-1].weight
        with torch.no_grad():
            double_weight.fill_(1.25)
            single_weight.fill_(-2.5)

        checkpoint_dir = tmp_path / "flux_torch_dist"
        checkpoint_dir.mkdir()
        save({"model": source.sharded_state_dict()}, checkpoint_dir)

        target = Flux(_tiny_flux_config()).to(_runtime_device())
        loaded = load({"model": target.sharded_state_dict()}, checkpoint_dir)
        target.load_state_dict(loaded["model"])

        torch.testing.assert_close(
            target.transformer.layers[0].adaln.adaLN_modulation[-1].weight,
            double_weight,
        )
        torch.testing.assert_close(
            target.transformer.layers[1].adaln.adaLN_modulation[-1].weight,
            single_weight,
        )
