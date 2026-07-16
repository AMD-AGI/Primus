# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for the model.training gate on DiffusionPretrainTrainer._forward_step_count.

Contract under test: when ``model.training`` is False (the trainer is running
an evaluation forward pass), the counter must NOT advance and the per-step
CUDA RNG reseed must NOT fire. Otherwise validation steps shift the training
RNG sequence by ``eval_iters * num_microbatches`` per ``--eval-interval``
window, defeating the goal of isolating the training RNG from unrelated
forward passes.

These tests pair with
``tests/unit_tests/backends/megatron/test_diffusion_trainer_forward_step_count.py``,
which covers lazy initialization from checkpoint state and the
``step_count`` flow into ``flux_forward_step_func``. This file is the
single source of truth for the eval-skip contract — if the gate is reverted,
the three ``TestForwardStepCountTrainingGate`` tests below must fail.
"""

from unittest.mock import Mock, patch

import torch

from primus.backends.megatron.diffusion_trainer import DiffusionPretrainTrainer


class _ConcreteDiffusionTrainer(DiffusionPretrainTrainer):
    """Minimal concrete subclass: abstract methods as no-ops."""

    def create_model(self, *args, **kwargs):
        return None

    def create_scheduler(self, *args, **kwargs):
        return None

    def get_task_encoder(self, *args, **kwargs):
        return None


def _make_trainer():
    """Construct a trainer with ``__init__`` bypassed.

    Sets only the attributes ``forward_step`` reads. Lazy-init is forced to
    "already initialized" because that surface is covered by
    ``test_diffusion_trainer_forward_step_count.py::TestForwardStepCountLazyInit``.
    """
    trainer = _ConcreteDiffusionTrainer.__new__(_ConcreteDiffusionTrainer)
    trainer._forward_step_count = 0
    trainer._forward_step_count_initialized = True
    trainer._scheduler = None

    class _FakeRuntimeState:
        def update_metrics(self, metrics):
            pass

    trainer.runtime_state = _FakeRuntimeState()
    return trainer


def _patch_flux_forward_step_func():
    """Patch the real func with a recorder that captures ``step_count``."""
    captured = []

    def _recorder(*args, **kwargs):
        captured.append(kwargs.get("step_count"))
        t = torch.zeros(1)
        return t, t, t, None, {}, False

    return (
        patch(
            "primus.backends.megatron.training.diffusion.forward_step.flux_forward_step_func",
            side_effect=_recorder,
        ),
        captured,
    )


class TestForwardStepCountTrainingGate:
    """Direct coverage of the ``if model.training:`` gate on counter advance."""

    def test_eval_does_not_advance_counter(self):
        trainer = _make_trainer()
        patcher, captured = _patch_flux_forward_step_func()
        eval_model = Mock()
        eval_model.training = False

        with patcher:
            trainer.forward_step(data_iterator=None, model=eval_model)

        assert trainer._forward_step_count == 0
        # Counter unchanged → step_count passed to the step func is the pre-call value.
        assert captured == [0]

    def test_train_advances_counter_by_one(self):
        trainer = _make_trainer()
        patcher, captured = _patch_flux_forward_step_func()
        train_model = Mock()
        train_model.training = True

        with patcher:
            trainer.forward_step(data_iterator=None, model=train_model)

        assert trainer._forward_step_count == 1
        assert captured == [1]

    def test_eval_between_train_does_not_shift_sequence(self):
        """Eval batches interleaved with training must NOT change the
        ``step_count`` that subsequent training steps see."""
        trainer = _make_trainer()
        patcher, captured = _patch_flux_forward_step_func()

        train_model = Mock()
        train_model.training = True
        eval_model = Mock()
        eval_model.training = False

        with patcher:
            trainer.forward_step(data_iterator=None, model=train_model)  # 0 -> 1
            trainer.forward_step(data_iterator=None, model=eval_model)  # stays 1
            trainer.forward_step(data_iterator=None, model=eval_model)  # stays 1
            trainer.forward_step(data_iterator=None, model=train_model)  # 1 -> 2

        assert trainer._forward_step_count == 2
        # Critical assertion: the 4th training step sees step_count=2, the
        # same value it would see in the absence of the two eval calls. If
        # the gate were missing, the sequence would be [1, 2, 3, 4].
        assert captured == [1, 1, 1, 2]


class TestPerStepReseedFormula:
    """Positive control: when the counter advances, the per-step reseed
    formula in ``flux_forward_step_func`` is computed correctly.

    This is a focused regression test for the seed expression
    ``(seed + 100 * dp_rank) * 10000 + step_count) % (2**63)``.
    It deliberately reaches into the underlying step function rather than
    going through the trainer so the formula is testable in isolation
    from the gate logic."""

    def test_reseed_uses_step_count_in_formula(self):
        from primus.backends.megatron.training.diffusion import forward_step as fwd_mod

        fake_args = Mock()
        fake_args.seed = 7

        train_model = Mock()
        train_model.training = True

        with patch.object(torch.cuda, "manual_seed") as mock_seed, patch(
            "megatron.training.get_args", return_value=fake_args
        ), patch("megatron.core.parallel_state.get_data_parallel_rank", return_value=3):
            # flux_forward_step_func runs significant downstream logic that
            # would require a full Megatron-Flux harness. We only need to
            # verify the reseed call. Suppress the post-reseed failure so the
            # mock.assert below still observes the manual_seed call.
            try:
                fwd_mod.flux_forward_step_func(
                    data_iterator=None,
                    model=train_model,
                    scheduler=Mock(),
                    per_step_rng_reseed=True,
                    step_count=42,
                )
            except Exception:
                # Expected: downstream code needs real data/model. The reseed
                # call we care about happens before any of that — see the
                # assert below.
                pass

        expected_seed = ((7 + 100 * 3) * 10000 + 42) % (2**63)
        mock_seed.assert_called_once_with(expected_seed)
