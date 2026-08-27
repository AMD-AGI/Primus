# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for the validation reduction and the sample accounting around it.

These cover the code that produces the number the run actually reports. Two
properties matter and neither is visible from the per-microbatch loss function:
the reported loss must be a ratio of globally summed numerator and denominator
rather than an average of per-rank ratios, and the reduced denominator must
survive as a true sample count so an under-read can be detected instead of
being papered over with the count the configuration intended.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from primus.backends.megatron.training.evaluator import (
    VAL_LOSS_KEY,
    _record_consumed_valid_samples,
    reduce_eval_losses,
)

GROUP = object()


class FakeAllReduce:
    """Stand-in for the DP all-reduce that can add other ranks' contributions."""

    def __init__(self, other_ranks=()):
        self.other_ranks = other_ranks
        self.buffers = []
        self.groups = []

    def __call__(self, tensor, op=None, group=None):
        self.buffers.append(tensor.clone())
        self.groups.append(group)
        for contribution in self.other_ranks:
            tensor += torch.tensor(contribution, dtype=tensor.dtype, device=tensor.device)

    @property
    def call_count(self):
        return len(self.buffers)


def _accumulators(pairs):
    """Build the (numerator, denominator) dicts the eval loop accumulates."""
    numerators = {key: torch.tensor(num) for key, (num, _) in pairs.items()}
    denominators = {key: torch.tensor(float(den)) for key, (_, den) in pairs.items()}
    return numerators, denominators


class TestReduction:
    def test_loss_is_the_ratio_of_the_summed_pair(self):
        numerators, denominators = _accumulators({VAL_LOSS_KEY: (150.0, 300.0)})
        with patch("torch.distributed.all_reduce", FakeAllReduce()):
            losses, _ = reduce_eval_losses(numerators, denominators, GROUP)

        assert losses[VAL_LOSS_KEY].item() == pytest.approx(0.5)

    def test_ratio_of_sums_not_the_mean_of_per_rank_ratios(self):
        """The property the old double-reduce lost.

        Rank 0 reports 100/200 = 0.5 and rank 1 reports 900/100 = 9.0. The
        correct sample-weighted answer is 1000/300 = 3.33, not the 4.75 an
        average of the two per-rank ratios would give.
        """
        numerators, denominators = _accumulators({VAL_LOSS_KEY: (100.0, 200.0)})
        other_rank = FakeAllReduce(other_ranks=[[900.0, 100.0]])
        with patch("torch.distributed.all_reduce", other_rank):
            losses, _ = reduce_eval_losses(numerators, denominators, GROUP)

        assert losses[VAL_LOSS_KEY].item() == pytest.approx(1000.0 / 300.0, rel=1e-6)
        assert losses[VAL_LOSS_KEY].item() != pytest.approx((0.5 + 9.0) / 2)

    def test_one_reduction_no_matter_how_many_keys(self):
        numerators, denominators = _accumulators({"loss": (1.0, 2.0), "aux": (3.0, 4.0), "extra": (5.0, 6.0)})
        fake = FakeAllReduce()
        with patch("torch.distributed.all_reduce", fake):
            reduce_eval_losses(numerators, denominators, GROUP)

        assert fake.call_count == 1

    def test_buffer_is_fp64(self):
        """bf16 accumulators must not decide the precision of the reported loss."""
        numerators = {VAL_LOSS_KEY: torch.tensor(150.0, dtype=torch.bfloat16)}
        denominators = {VAL_LOSS_KEY: torch.tensor(300.0, dtype=torch.bfloat16)}
        fake = FakeAllReduce()
        with patch("torch.distributed.all_reduce", fake):
            reduce_eval_losses(numerators, denominators, GROUP)

        assert fake.buffers[0].dtype == torch.float64

    def test_keys_are_packed_as_sorted_pairs(self):
        """Packing order is what lets one flat buffer be unpacked per key."""
        numerators, denominators = _accumulators({"loss": (1.0, 2.0), "aux": (3.0, 4.0)})
        fake = FakeAllReduce()
        with patch("torch.distributed.all_reduce", fake):
            reduce_eval_losses(numerators, denominators, GROUP)

        assert fake.buffers[0].tolist() == [3.0, 4.0, 1.0, 2.0]

    def test_each_key_keeps_its_own_ratio(self):
        numerators, denominators = _accumulators({"loss": (1.0, 2.0), "aux": (9.0, 3.0)})
        with patch("torch.distributed.all_reduce", FakeAllReduce()):
            losses, _ = reduce_eval_losses(numerators, denominators, GROUP)

        assert losses["loss"].item() == pytest.approx(0.5)
        assert losses["aux"].item() == pytest.approx(3.0)

    def test_empty_denominator_yields_zero_rather_than_nan(self):
        numerators, denominators = _accumulators({VAL_LOSS_KEY: (0.0, 0)})
        with patch("torch.distributed.all_reduce", FakeAllReduce()):
            losses, observed = reduce_eval_losses(numerators, denominators, GROUP)

        assert losses[VAL_LOSS_KEY].item() == 0.0
        assert not torch.isnan(losses[VAL_LOSS_KEY])
        assert observed == 0

    def test_result_is_a_zero_dim_tensor(self):
        """Megatron's evaluate_and_print_results and the mlperf logger call .item()."""
        numerators, denominators = _accumulators({VAL_LOSS_KEY: (1.0, 2.0)})
        with patch("torch.distributed.all_reduce", FakeAllReduce()):
            losses, _ = reduce_eval_losses(numerators, denominators, GROUP)

        assert losses[VAL_LOSS_KEY].shape == ()
        assert losses[VAL_LOSS_KEY].dtype == torch.float32

    def test_reduces_over_the_group_it_was_given(self):
        numerators, denominators = _accumulators({VAL_LOSS_KEY: (1.0, 2.0)})
        fake = FakeAllReduce()
        with patch("torch.distributed.all_reduce", fake):
            reduce_eval_losses(numerators, denominators, GROUP)

        assert fake.groups == [GROUP]


class TestObservedSampleCount:
    def test_is_the_globally_summed_denominator(self):
        """29,696 total across 8 ranks of 3,712 -- the MLPerf shape."""
        numerators, denominators = _accumulators({VAL_LOSS_KEY: (2000.0, 3712)})
        seven_more = FakeAllReduce(other_ranks=[[2000.0, 3712.0]] * 7)
        with patch("torch.distributed.all_reduce", seven_more):
            _, observed = reduce_eval_losses(numerators, denominators, GROUP)

        assert observed == 29696

    def test_is_an_integer_not_a_float(self):
        numerators, denominators = _accumulators({VAL_LOSS_KEY: (1.0, 512)})
        with patch("torch.distributed.all_reduce", FakeAllReduce()):
            _, observed = reduce_eval_losses(numerators, denominators, GROUP)

        assert isinstance(observed, int)

    def test_absent_without_the_val_loss_key(self):
        """A microbatch-counting metric shape must not masquerade as a count."""
        numerators, denominators = _accumulators({"lm loss": (1.0, 2.0)})
        with patch("torch.distributed.all_reduce", FakeAllReduce()):
            _, observed = reduce_eval_losses(numerators, denominators, GROUP)

        assert observed is None


class TestConsumedValidSamples:
    """The accounting that turns a short read into a failure instead of a log line."""

    @staticmethod
    def _args(consumed=0):
        return SimpleNamespace(consumed_valid_samples=consumed)

    @staticmethod
    def _record(args, observed, eval_iters=58, eval_batch_size=512, cp_size=1):
        with patch(
            "primus.backends.megatron.training.evaluator.parallel_state." "get_context_parallel_world_size",
            return_value=cp_size,
        ):
            _record_consumed_valid_samples(args, observed, eval_iters, eval_batch_size)

    def test_full_coverage_accumulates_the_observed_count(self):
        args = self._args()
        self._record(args, observed=29696)

        assert args.consumed_valid_samples == 29696

    def test_accumulates_across_evaluations(self):
        args = self._args(consumed=29696)
        self._record(args, observed=29696)

        assert args.consumed_valid_samples == 2 * 29696

    def test_under_read_is_an_error_naming_the_shortfall(self):
        """The 27,776-sample under-read that started all of this."""
        args = self._args()
        with pytest.raises(RuntimeError, match="read 27776 samples"):
            self._record(args, observed=27776)

    def test_under_read_error_points_at_the_worker_count(self):
        args = self._args()
        with pytest.raises(RuntimeError, match="val_num_workers"):
            self._record(args, observed=27776)

    def test_over_read_is_also_an_error(self):
        """Double-counting is as wrong as under-counting, and just as silent."""
        args = self._args()
        with pytest.raises(RuntimeError, match="read 59392 samples"):
            self._record(args, observed=2 * 29696)

    def test_context_parallelism_logs_instead_of_raising(self):
        """CP duplicates the per-sample loss, so the denominator is legitimately inflated."""
        args = self._args()
        with patch("primus.backends.megatron.training.evaluator.log_rank_0") as log:
            self._record(args, observed=2 * 29696, cp_size=2)

        assert args.consumed_valid_samples == 2 * 29696
        assert "context_parallel_size=2" in log.call_args[0][0]

    def test_no_observation_falls_back_to_the_configured_budget(self):
        """Stages with no loss still have to advance the counter."""
        args = self._args()
        self._record(args, observed=None)

        assert args.consumed_valid_samples == 58 * 512

    def test_fallback_does_not_raise_on_a_mismatch_it_cannot_see(self):
        args = self._args()
        self._record(args, observed=None, eval_iters=1, eval_batch_size=7)

        assert args.consumed_valid_samples == 7
