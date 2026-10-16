# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
WAN has no validation path, and must say so rather than fake one.

Nothing produces a validation-marked WAN batch and there is no fixed-timestep
validation schedule, so an eval pass through the training forward step would
score random training timesteps and report that as validation loss. The trainer
rejects ``eval_iters > 0``; these pin the forward step's own guard and the mock
data that used to carry Flux's validation timesteps.
"""

import sys
import types

import pytest
import torch

from primus.backends.megatron.data.synthetic import MockWanDataset


@pytest.fixture
def stub_parallel_state():
    """The forward step imports parallel_state inside the function."""
    saved = sys.modules.get("megatron.core.parallel_state")
    sys.modules["megatron.core.parallel_state"] = types.SimpleNamespace(
        get_pipeline_model_parallel_rank=lambda: 0,
        get_pipeline_model_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_world_size=lambda: 1,
    )
    yield
    if saved is None:
        sys.modules.pop("megatron.core.parallel_state", None)
    else:
        sys.modules["megatron.core.parallel_state"] = saved


def test_forward_step_refuses_eval_mode(stub_parallel_state):
    from primus.backends.megatron.training.diffusion.wan_forward_step import (
        wan_forward_step_func,
    )

    model = types.SimpleNamespace(
        training=False,
        config=types.SimpleNamespace(bf16=True, fp16=False, params_dtype=torch.bfloat16),
    )
    with pytest.raises(RuntimeError, match="no validation path"):
        wan_forward_step_func(iter([]), model, scheduler=None)


def test_mock_wan_samples_carry_no_validation_timestep():
    # SyntheticDatasetProvider passes is_validation=True when it builds a
    # validation set; Flux's idx % 8 levels are not WAN timestep indices.
    dataset = MockWanDataset(
        num_samples=2, image_size=16, num_frames=1, text_seq_len=4, text_embed_dim=8, is_validation=True
    )
    assert set(dataset[0]) == {"latents", "encoder_hidden_states"}
