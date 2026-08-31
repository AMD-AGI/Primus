# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for the validation RNG stream.

The training forward-step counter is deliberately frozen during evaluation so
that eval passes do not shift the training seed sequence. Reusing that frozen
value to reseed, however, gives every validation microbatch the same seed,
which repeats one draw of the VAE reparameterization epsilon and one draw of
the flow-matching noise across the entire evaluation. These tests pin the
separate, advancing, resume-deterministic eval stream that replaces it.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from primus.backends.megatron.diffusion_trainer import DiffusionPretrainTrainer
from primus.backends.megatron.training.diffusion.forward_step import (
    EVAL_RNG_ITERATION_STRIDE,
    EVAL_RNG_OFFSET,
)


class _ConcreteDiffusionTrainer(DiffusionPretrainTrainer):
    def create_model(self, *args, **kwargs):
        return None

    def create_scheduler(self, *args, **kwargs):
        return None

    def get_task_encoder(self, *args, **kwargs):
        return None


def _make_trainer():
    trainer = _ConcreteDiffusionTrainer.__new__(_ConcreteDiffusionTrainer)
    trainer._forward_step_count = 0
    trainer._forward_step_count_initialized = False
    trainer._eval_rng_iteration = None
    trainer._eval_microbatch_index = 0
    return trainer


def _indices(trainer, iteration, count):
    with patch("megatron.training.get_args", return_value=SimpleNamespace(iteration=iteration)):
        return [trainer._next_eval_step_index() for _ in range(count)]


class TestEvalStepIndex:
    def test_advances_within_one_evaluation(self):
        trainer = _make_trainer()
        assert _indices(trainer, iteration=100, count=4) == [
            100 * EVAL_RNG_ITERATION_STRIDE + i for i in range(4)
        ]

    def test_restarts_each_evaluation(self):
        """Index restarts at zero per evaluation so no counter needs checkpointing."""
        trainer = _make_trainer()
        _indices(trainer, iteration=100, count=3)
        second = _indices(trainer, iteration=200, count=3)

        assert second == [200 * EVAL_RNG_ITERATION_STRIDE + i for i in range(3)]

    def test_consecutive_evaluations_do_not_collide(self):
        trainer = _make_trainer()
        first = set(_indices(trainer, iteration=100, count=64))
        second = set(_indices(trainer, iteration=101, count=64))

        assert not (first & second)

    def test_reproducible_after_resume(self):
        """A fresh trainer at the same iteration must produce the same indices."""
        before = _indices(_make_trainer(), iteration=512, count=8)
        after_resume = _indices(_make_trainer(), iteration=512, count=8)

        assert before == after_resume

    def test_overrun_of_the_stride_is_rejected(self):
        trainer = _make_trainer()
        trainer._eval_rng_iteration = 5
        trainer._eval_microbatch_index = EVAL_RNG_ITERATION_STRIDE

        with patch("megatron.training.get_args", return_value=SimpleNamespace(iteration=5)):
            with pytest.raises(RuntimeError, match="per-iteration stride"):
                trainer._next_eval_step_index()


class TestEvalSeedsAreDisjointFromTraining:
    """The eval stream must not alias the training stream on any rank.

    Mirrors the derivations in forward_step: training seeds are
    (seed + 100 * dp_rank) * 10000 + step_count, eval seeds add EVAL_RNG_OFFSET.
    """

    @staticmethod
    def _train_seed(seed, dp_rank, step_count):
        return ((seed + 100 * dp_rank) * 10000 + step_count) % (2**63)

    @staticmethod
    def _eval_seed(seed, dp_rank, eval_index):
        return (EVAL_RNG_OFFSET + (seed + 100 * dp_rank) * 10000 + eval_index) % (2**63)

    def test_no_overlap_across_plausible_run_shapes(self):
        seed = 42
        train = {self._train_seed(seed, rank, step) for rank in range(64) for step in range(0, 20000, 97)}
        evals = {
            self._eval_seed(seed, rank, iteration * EVAL_RNG_ITERATION_STRIDE + micro)
            for rank in range(64)
            for iteration in range(0, 20000, 512)
            for micro in range(58)
        }

        assert not (train & evals)

    def test_eval_seeds_differ_per_microbatch(self):
        seeds = {self._eval_seed(42, 0, i) for i in range(58)}
        assert len(seeds) == 58

    def test_eval_seeds_differ_per_rank(self):
        seeds = {self._eval_seed(42, rank, 0) for rank in range(8)}
        assert len(seeds) == 8


class TestForwardStepPassesEvalIndex:
    """The trainer must route eval and training passes to different streams."""

    @staticmethod
    def _trainer_with_recorder(per_step_rng_reseed=True):
        trainer = _make_trainer()
        trainer._scheduler = None
        trainer.per_step_rng_reseed = per_step_rng_reseed

        class _FakeRuntimeState:
            def update_metrics(self, metrics):
                pass

        trainer.runtime_state = _FakeRuntimeState()

        recorded = []

        def recording_func(*args, **kwargs):
            recorded.append(dict(kwargs))
            t = torch.zeros(1)
            return t, t, t, None, {}, False

        return trainer, recorded, recording_func

    def test_training_pass_has_no_eval_index(self):
        trainer, recorded, recording_func = self._trainer_with_recorder()
        model = Mock()
        model.training = True

        with patch(
            "primus.backends.megatron.training.diffusion.forward_step.flux_forward_step_func",
            side_effect=recording_func,
        ):
            trainer.forward_step(data_iterator=None, model=model)

        assert recorded[-1]["eval_step_index"] is None
        assert recorded[-1]["step_count"] == 1

    def test_eval_pass_freezes_training_counter_but_advances_eval_index(self):
        trainer, recorded, recording_func = self._trainer_with_recorder()
        trainer._forward_step_count = 7
        model = Mock()
        model.training = False

        with patch(
            "primus.backends.megatron.training.diffusion.forward_step.flux_forward_step_func",
            side_effect=recording_func,
        ), patch("megatron.training.get_args", return_value=SimpleNamespace(iteration=3)):
            trainer.forward_step(data_iterator=None, model=model)
            trainer.forward_step(data_iterator=None, model=model)

        # Training counter stays put, so the training seed sequence is unshifted.
        assert trainer._forward_step_count == 7
        assert [r["step_count"] for r in recorded] == [7, 7]
        # ... but each eval microbatch gets its own seed.
        eval_indices = [r["eval_step_index"] for r in recorded]
        assert eval_indices == [
            3 * EVAL_RNG_ITERATION_STRIDE,
            3 * EVAL_RNG_ITERATION_STRIDE + 1,
        ]
        assert len(set(eval_indices)) == 2

    def test_no_eval_index_when_reseeding_is_off(self):
        """Without reseeding the ambient generator already advances per batch."""
        trainer, recorded, recording_func = self._trainer_with_recorder(per_step_rng_reseed=False)
        model = Mock()
        model.training = False

        with patch(
            "primus.backends.megatron.training.diffusion.forward_step.flux_forward_step_func",
            side_effect=recording_func,
        ):
            trainer.forward_step(data_iterator=None, model=model)

        assert recorded[-1]["eval_step_index"] is None
