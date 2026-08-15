# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for MLPerf run termination and resumed-run throughput.

Both behaviours were found on the Issue 220 staged MXFP4-to-BF16 run:

  - the BF16 healing leg reported 4,078 imgs/s at its first evaluation while
    sustaining about 350, because throughput divided the cumulative sample count
    by this process's clock;
  - the MXFP4 control leg exhausted train_iters without reaching the target, exited
    rc=0, and left a log with a run_start and no run_stop, which the MLPerf package
    checker rejects even for the one non-converging run a 10-run set may drop.
"""

import os
import time
import unittest

import pytest

from primus.backends.megatron.patches.mlperf_logging_patches import (
    FluxMLPerfLogger,
    ThroughputTimer,
)

GBS = 512


class RecordingMLLogger:
    """Stands in for mllog's logger and records what would have been emitted."""

    def __init__(self):
        self.events = []

    def event(self, key, value=None, metadata=None):
        self.events.append(("event", key, value, metadata))

    def start(self, key, value=None, metadata=None):
        self.events.append(("start", key, value, metadata))

    def end(self, key, value=None, metadata=None):
        self.events.append(("end", key, value, metadata))


class TestThroughputTimerOnResume(unittest.TestCase):
    def test_from_scratch_counts_every_sample(self):
        timer = ThroughputTimer(GBS)
        timer.update_samples(1)
        self.assertEqual(timer.consumed_samples, GBS)
        timer.update_samples(100)
        self.assertEqual(timer.consumed_samples, 100 * GBS)

    def test_resumed_leg_counts_only_its_own_samples(self):
        timer = ThroughputTimer(GBS)
        # A leg resuming the step-5,120 checkpoint sees its first iteration as 5,121.
        timer.update_samples(5121)
        self.assertEqual(timer.consumed_samples, GBS)
        timer.update_samples(5130)
        self.assertEqual(timer.consumed_samples, 10 * GBS)

    def test_explicit_baseline_beats_guessing_from_the_first_call(self):
        # The patch sets the baseline from args.iteration, which stays exact even
        # if update_samples is not called on every single step.
        timer = ThroughputTimer(GBS)
        timer.baseline_iteration = 5120
        timer.update_samples(5130)
        self.assertEqual(timer.consumed_samples, 10 * GBS)

    def test_throughput_is_this_process_rate_not_the_inherited_one(self):
        timer = ThroughputTimer(GBS)
        timer.baseline_iteration = 5120
        timer.training_start_time = time.time() - 100.0
        timer.update_samples(5130)

        rate = timer.compute_throughput()
        # 10 steps in 100 s is 51.2 imgs/s. The pre-fix arithmetic divided the
        # cumulative 2,626,560 samples by the same 100 s, reporting 26,265.
        self.assertAlmostEqual(rate, 10 * GBS / 100.0, delta=1.0)
        self.assertLess(rate, 1000.0)

    def test_eval_time_is_excluded_from_the_training_rate(self):
        timer = ThroughputTimer(GBS)
        timer.baseline_iteration = 0
        timer.training_start_time = time.time() - 100.0
        timer.update_samples(11)
        timer.eval_cumulative_secs = 50.0

        # Same samples, half the time charged: the training rate doubles while the
        # combined rate stays on the wall clock.
        self.assertGreater(timer.compute_throughput(), timer.compute_combined_throughput())


class TestRunStopTermination(unittest.TestCase):
    def setUp(self):
        pytest.importorskip("mlperf_logging")
        self._rank = os.environ.get("RANK")
        os.environ["RANK"] = "0"
        self.logger = FluxMLPerfLogger(global_batch_size=GBS, micro_batch_size=64)
        self.recorder = RecordingMLLogger()
        self.logger._mllogger = self.recorder

    def tearDown(self):
        if self._rank is None:
            os.environ.pop("RANK", None)
        else:
            os.environ["RANK"] = self._rank

    def _stops(self):
        return [e for e in self.recorder.events if e[1] == self.logger._constants.RUN_STOP]

    def test_unconverged_run_is_closed_as_aborted(self):
        self.logger.log_run_stop(success=False, global_step=20000)

        stops = self._stops()
        self.assertEqual(len(stops), 1)
        _kind, _key, value, metadata = stops[0]
        self.assertEqual(value, "aborted")
        self.assertEqual(metadata["status"], "aborted")
        self.assertEqual(metadata["step"], 20000)
        self.assertEqual(metadata["samples_count"], 20000 * GBS)
        self.assertFalse(self.logger.converged)

    def test_converged_run_is_closed_as_success(self):
        self.logger.log_run_stop(success=True, global_step=15872)

        stops = self._stops()
        self.assertEqual(len(stops), 1)
        self.assertEqual(stops[0][2], "success")
        self.assertTrue(self.logger.converged)

    def test_a_run_stops_exactly_once(self):
        # The convergence path stops the run, then train() returns and the
        # end-of-training path must not append a second one.
        self.logger.log_run_stop(success=True, global_step=15872)
        self.logger.log_run_stop(success=False, global_step=15872)

        stops = self._stops()
        self.assertEqual(len(stops), 1)
        self.assertEqual(stops[0][2], "success")

    def test_repeated_abort_does_not_duplicate(self):
        self.logger.log_run_stop(success=False, global_step=20000)
        self.logger.log_run_stop(success=False, global_step=20000)

        self.assertEqual(len(self._stops()), 1)


if __name__ == "__main__":
    unittest.main()
