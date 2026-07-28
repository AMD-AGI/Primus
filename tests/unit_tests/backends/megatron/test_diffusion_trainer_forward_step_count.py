# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for DiffusionPretrainTrainer's forward step count.

Validates Fix 1: the `_forward_step_count` instance variable and its
checkpoint-compatible lazy initialization from `args.iteration *
get_num_microbatches()`. Also verifies the counter flows through to
`flux_forward_step_func` as the `step_count` keyword argument.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from primus.backends.megatron.diffusion_trainer import DiffusionPretrainTrainer


class _ConcreteDiffusionTrainer(DiffusionPretrainTrainer):
    """Minimal concrete subclass: implements abstract methods as no-ops.

    These methods are only required for instantiation; the tests never
    invoke them.
    """

    def create_model(self, *args, **kwargs):
        return None

    def create_scheduler(self, *args, **kwargs):
        return None

    def get_task_encoder(self, *args, **kwargs):
        return None


def _make_trainer():
    """Construct a DiffusionPretrainTrainer with __init__ bypassed.

    Sets only the attributes required by `get_forward_step()` and
    `forward_step()`. Other attributes are intentionally left unset so
    that tests fail loudly if production code starts depending on them.
    """
    trainer = _ConcreteDiffusionTrainer.__new__(_ConcreteDiffusionTrainer)
    trainer._forward_step_count = 0
    trainer._forward_step_count_initialized = False
    return trainer


class TestForwardStepCountLazyInit:
    """Tests for lazy initialization of _forward_step_count from checkpoint state."""

    def test_lazy_init_from_checkpoint_iteration(self):
        """First closure call must reconstruct counter as iteration * num_microbatches."""
        trainer = _make_trainer()
        trainer.forward_step = lambda data_iterator, model, return_schedule_plan=False: None

        with patch(
            "megatron.training.get_args",
            return_value=SimpleNamespace(iteration=100),
        ), patch(
            "megatron.core.num_microbatches_calculator.get_num_microbatches",
            return_value=4,
        ):
            closure = trainer.get_forward_step()
            closure(data_iterator=None, model=None)

        assert trainer._forward_step_count == 400
        assert trainer._forward_step_count_initialized is True

    def test_lazy_init_only_runs_once(self):
        """Second call must not re-read args.iteration even if it changes."""
        trainer = _make_trainer()
        trainer.forward_step = lambda data_iterator, model, return_schedule_plan=False: None

        args_state = SimpleNamespace(iteration=10)

        with patch(
            "megatron.training.get_args",
            side_effect=lambda: args_state,
        ), patch(
            "megatron.core.num_microbatches_calculator.get_num_microbatches",
            return_value=2,
        ):
            closure = trainer.get_forward_step()
            closure(data_iterator=None, model=None)
            assert trainer._forward_step_count == 20

            args_state.iteration = 999
            closure(data_iterator=None, model=None)

        # Must remain 20: lazy init flag prevents a second reconstruction.
        assert trainer._forward_step_count == 20
        assert trainer._forward_step_count_initialized is True

    def test_initial_iteration_zero(self):
        """Fresh start (iteration=0) reconstructs counter to 0."""
        trainer = _make_trainer()
        trainer.forward_step = lambda data_iterator, model, return_schedule_plan=False: None

        with patch(
            "megatron.training.get_args",
            return_value=SimpleNamespace(iteration=0),
        ), patch(
            "megatron.core.num_microbatches_calculator.get_num_microbatches",
            return_value=8,
        ):
            closure = trainer.get_forward_step()
            closure(data_iterator=None, model=None)

        assert trainer._forward_step_count == 0
        assert trainer._forward_step_count_initialized is True


class TestCounterFlowsToForwardStepFunc:
    """Integration test: trainer counter must pass through to flux_forward_step_func."""

    def test_counter_flows_to_forward_step_func(self):
        """forward_step must invoke flux_forward_step_func with step_count == counter."""
        trainer = _make_trainer()
        # _scheduler backs the `scheduler` lazy property; setting it pre-empts
        # create_scheduler() which would otherwise raise abstract NotImplementedError.
        trainer._scheduler = None

        # Provide a fake runtime_state so forward_step doesn't fall into the
        # log_rank_0 warning branch (which requires the Primus logger to be
        # initialized — outside this test's scope).
        class _FakeRuntimeState:
            def update_metrics(self, metrics):
                pass

        trainer.runtime_state = _FakeRuntimeState()

        recorded_kwargs = {}

        def recording_func(*args, **kwargs):
            recorded_kwargs.update(kwargs)
            # Return shape matches: (noise_pred, clean_latents, noise, loss_mask, metrics, is_validation)
            t = torch.zeros(1)
            return t, t, t, None, {}, False

        # Training model: `if model.training:` gate in forward_step requires
        # a real attribute (None would AttributeError). Eval-skip behavior is
        # covered in test_forward_step_count_gate.py.
        train_model = Mock()
        train_model.training = True

        with patch(
            "primus.backends.megatron.training.diffusion.forward_step.flux_forward_step_func",
            side_effect=recording_func,
        ):
            # First forward step — counter increments from 0 to 1
            trainer.forward_step(data_iterator=None, model=train_model)
            assert recorded_kwargs["step_count"] == 1
            assert trainer._forward_step_count == 1

            # Second forward step — counter increments to 2
            trainer.forward_step(data_iterator=None, model=train_model)
            assert recorded_kwargs["step_count"] == 2
            assert trainer._forward_step_count == 2
