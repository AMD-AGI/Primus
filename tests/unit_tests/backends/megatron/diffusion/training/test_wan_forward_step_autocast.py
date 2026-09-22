# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
The WAN forward step must call the model outside autocast.

Autocast's fp32 policy covers layer_norm and rms_norm, which is the whole of
WanTransformerBlock's norm path, so wrapping the model call in
``torch.amp.autocast("cuda", bfloat16)`` makes every norm return fp32 on a bf16
input. The projection that consumes it narrows straight back to bf16, so the
wider value never reaches a multiply, but it is saved for backward twice -- once
by the norm and once as the projection's input -- and with recompute on it is
computed twice as well.

The batch is cast to compute_dtype before the call and Float16Module holds the
parameters at params_dtype, so there is nothing for autocast to do here. This
pins that down: the wrapper is easy to reintroduce and its cost does not show up
in the loss, which is unchanged either way.
"""

import sys
import types
from typing import Optional

import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from tests.utils import PrimusUT


class _RecordingModel:
    """Stands in for ``Wan``; records the autocast state it is called under."""

    def __init__(self, dtype: torch.dtype):
        self.config = types.SimpleNamespace(bf16=True, fp16=False, params_dtype=dtype)
        self.autocast_enabled: Optional[bool] = None
        self.autocast_dtype: Optional[torch.dtype] = None
        self.input_dtype: Optional[torch.dtype] = None

    def __call__(self, hidden_states, timestep, encoder_hidden_states, boundary_timestep):
        self.autocast_enabled = torch.is_autocast_enabled("cuda")
        self.autocast_dtype = torch.get_autocast_dtype("cuda")
        self.input_dtype = hidden_states.dtype
        return torch.zeros_like(hidden_states)


class _StubScheduler:
    """Minimal scheduler surface the forward step touches."""

    def sample_training_timesteps(self, batch_size, device, timestep_window=None):
        return torch.zeros(batch_size, device=device, dtype=torch.long)

    def add_noise(self, latents, noise, timesteps):
        return latents + noise

    def training_target(self, latents, noise, timesteps):
        return noise - latents

    def conditioning_timestep(self, timesteps, device):
        return timesteps.to(device=device)

    def training_weight(self, model_timesteps):
        return torch.ones(model_timesteps.shape[0], device=model_timesteps.device)


class TestWanForwardStepAutocast(PrimusUT):
    """The model call must not be wrapped in autocast."""

    def setup_method(self, method):
        # The forward step imports parallel_state inside the function, so a stub
        # module is enough and no distributed init is needed.
        self._saved = sys.modules.get("megatron.core.parallel_state")
        sys.modules["megatron.core.parallel_state"] = types.SimpleNamespace(
            get_pipeline_model_parallel_rank=lambda: 0,
            get_pipeline_model_parallel_world_size=lambda: 1,
            get_tensor_model_parallel_world_size=lambda: 1,
        )

    def teardown_method(self, method):
        if self._saved is None:
            sys.modules.pop("megatron.core.parallel_state", None)
        else:
            sys.modules["megatron.core.parallel_state"] = self._saved

    def _run(self):
        from primus.backends.megatron.training.diffusion.wan_forward_step import (
            wan_forward_step_func,
        )

        model = _RecordingModel(torch.bfloat16)
        batch = {
            "latents": torch.randn(1, 4, 2, 4, 4, device="cuda", dtype=torch.float32),
            "encoder_hidden_states": torch.randn(1, 8, 16, device="cuda", dtype=torch.float32),
        }
        wan_forward_step_func(iter([batch]), model, _StubScheduler())
        return model

    def test_model_is_called_outside_autocast(self):
        model = self._run()
        assert model.autocast_enabled is False, (
            "WAN's model call is wrapped in autocast again. Autocast upcasts "
            "layer_norm and rms_norm to fp32, which is the block's whole norm "
            "path; see this module's docstring for what that costs."
        )

    def test_inputs_reach_the_model_at_compute_dtype(self):
        # This is what makes the autocast redundant, so it is the same test seen
        # from the other side: if the batch stopped being cast, removing
        # autocast would be a dtype mismatch rather than a saving.
        model = self._run()
        assert model.input_dtype == torch.bfloat16
