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

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import primus.backends.megatron.training.eval_session as eval_session
from primus.backends.megatron.training.evaluator import (
    VAL_LOSS_KEY,
    _record_consumed_valid_samples,
    _report_eval,
    primus_evaluate,
    reduce_eval_losses,
)

GROUP = object()
EVALUATOR = "primus.backends.megatron.training.evaluator"


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


class TestEvalReporting:
    """Which stream the evaluation's own reporting goes to.

    MLPerf mode reports the loss itself, as an mllog event and as its own line,
    and silences Megatron's reporting to keep the run to one voice; repeating it
    at info level there is noise. What must not follow is a coverage shortfall
    going quiet along with it.
    """

    @staticmethod
    def _args(mlperf_mode=False):
        return SimpleNamespace(consumed_valid_samples=0, mlperf_mode=mlperf_mode)

    @staticmethod
    @contextmanager
    def _streams(cp_size=1):
        """Capture the info and debug streams the evaluator reports through."""
        with patch(
            f"{EVALUATOR}.parallel_state.get_context_parallel_world_size",
            return_value=cp_size,
        ), patch(f"{EVALUATOR}.log_rank_0") as info, patch(f"{EVALUATOR}.debug_rank_0") as debug:
            yield info, debug

    def test_reports_at_info_by_default(self):
        with self._streams() as (info, debug):
            _report_eval(self._args(), "[eval] loss=1.665372")

        assert info.call_args[0][0] == "[eval] loss=1.665372"
        assert debug.call_count == 0

    def test_mlperf_mode_reports_at_debug_instead(self):
        with self._streams() as (info, debug):
            _report_eval(self._args(mlperf_mode=True), "[eval] loss=1.665372")

        assert debug.call_args[0][0] == "[eval] loss=1.665372"
        assert info.call_count == 0

    def test_an_absent_flag_reads_as_off(self):
        """args is built before MLPerf mode is a settled attribute on it."""
        with self._streams() as (info, debug):
            _report_eval(SimpleNamespace(), "[eval] loss=1.665372")

        assert info.call_count == 1
        assert debug.call_count == 0

    def test_mlperf_mode_moves_the_coverage_line_too(self):
        args = self._args(mlperf_mode=True)
        with self._streams() as (info, debug):
            _record_consumed_valid_samples(args, 29696, 58, 512)

        assert "covered 29696 samples" in debug.call_args[0][0]
        assert info.call_count == 0

    def test_mlperf_mode_does_not_quieten_an_under_read(self):
        """The whole point of the coverage check survives the quietening."""
        args = self._args(mlperf_mode=True)
        with self._streams():
            with pytest.raises(RuntimeError, match="read 27776 samples"):
                _record_consumed_valid_samples(args, 27776, 58, 512)

    def test_mlperf_mode_still_reports_a_context_parallel_mismatch(self):
        """A mismatch CP can explain is a diagnostic, not a confirmation."""
        args = self._args(mlperf_mode=True)
        with self._streams(cp_size=2) as (info, debug):
            _record_consumed_valid_samples(args, 2 * 29696, 58, 512)

        assert "context_parallel_size=2" in info.call_args[0][0]
        assert debug.call_count == 0


class TestEvaluationSessions:
    """Every evaluation must announce itself before it reads anything.

    The diffusion trainer derives its validation RNG index from this signal, so
    that two evaluations at the same training step -- the in-loop one that ends
    a run and the one pretrain runs afterwards -- reproduce each other rather
    than drawing fresh noise. That only holds if the evaluator opens a session,
    and the trainer cannot tell that it failed to.
    """

    @pytest.fixture(autouse=True)
    def _isolate_eval_session(self):
        """Driving primus_evaluate advances process-wide state."""
        saved = eval_session._eval_session
        yield
        eval_session._eval_session = saved

    @staticmethod
    def _args():
        return SimpleNamespace(
            vision_pretraining=False,
            vision_pretraining_type=None,
            global_batch_size=512,
            micro_batch_size=64,
            seq_length=256,
            decoder_seq_length=None,
            enable_cuda_graph=False,
            cuda_graph_scope=None,
            eval_iters=2,
            empty_unused_memory_level=0,
            exit_duration_in_mins=None,
            consumed_valid_samples=0,
            mlperf_mode=False,
        )

    @contextmanager
    def _evaluator(self, sessions_seen):
        """primus_evaluate with everything below the session signal stubbed out."""

        def forward_backward_func(**kwargs):
            sessions_seen.append(eval_session.current_eval_session())
            return [{}]

        with patch(f"{EVALUATOR}.get_args", return_value=self._args()), patch(
            f"{EVALUATOR}.get_timers"
        ), patch(f"{EVALUATOR}.get_rerun_state_machine"), patch(f"{EVALUATOR}.ft_integration"), patch(
            f"{EVALUATOR}.get_eval_num_microbatches", return_value=1
        ), patch(
            f"{EVALUATOR}.get_forward_backward_func", return_value=forward_backward_func
        ), patch(
            f"{EVALUATOR}.is_pipeline_stage_containing_loss", return_value=False
        ), patch(
            f"{EVALUATOR}._report_eval"
        ):
            yield

    def _evaluate(self, sessions_seen):
        with self._evaluator(sessions_seen):
            primus_evaluate(
                forward_step_func=lambda *a, **k: None,
                data_iterator=None,
                model=[MagicMock()],
                process_non_loss_data_func=None,
                config=SimpleNamespace(timers=None),
            )

    def test_a_session_is_open_before_the_first_forward_step(self):
        before = eval_session.current_eval_session()
        seen = []

        self._evaluate(seen)

        assert seen, "the evaluation ran no forward steps"
        assert all(session > before for session in seen)

    def test_every_forward_step_of_one_evaluation_sees_the_same_session(self):
        seen = []

        self._evaluate(seen)

        assert len(set(seen)) == 1

    def test_consecutive_evaluations_open_different_sessions(self):
        first, second = [], []

        self._evaluate(first)
        self._evaluate(second)

        assert set(first).isdisjoint(second)
