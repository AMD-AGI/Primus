# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
"""CPU tests for streaming checkpoint configuration and cadence decisions:

  * `in_window_checkpoint_frequency` — per-window-local batch count
  * `checkpoint_step_frequency`      — monotonic global step ("every 1000 steps")
  * `checkpoint_time_interval_s`     — wall-clock ("hourly")

These run without a GPU / distributed init. Configuration tests enter the
actual Gin-decorated loop and stop after its checkpoint timer is resolved;
pure decision tests cover the three trigger cadences independently.
"""
import os
import unittest
from unittest import mock

import gin

from generative_recommenders.dlrm_v4.train import utils as train_utils
from generative_recommenders.dlrm_v4.train.utils import (
    checkpoint_time_interval_for_saving,
    select_in_window_checkpoint_reason,
)


@gin.configurable
def checkpoint_timer_unexpected_reference() -> int:
    raise AssertionError("Timer path resolution evaluated an unrelated save binding")


def _reason(
    *,
    batch: int = 1,
    step: int = 1,
    elapsed: float = 0.0,
    in_window: int = 0,
    step_freq: int = 0,
    time_s: float = 0.0,
) -> str | None:
    return select_in_window_checkpoint_reason(
        train_batch_idx=batch,
        global_step=step,
        elapsed_since_last_save=elapsed,
        in_window_checkpoint_frequency=in_window,
        checkpoint_step_frequency=step_freq,
        checkpoint_time_interval_s=time_s,
    )


class CheckpointCadenceTest(unittest.TestCase):
    def test_all_disabled_never_fires(self) -> None:
        for batch in (1, 100, 1000):
            for step in (1, 1000, 5000):
                self.assertIsNone(_reason(batch=batch, step=step, elapsed=1e9))

    def test_step_based_every_1000(self) -> None:
        # Fires exactly on multiples of the step frequency.
        self.assertEqual(_reason(step=1000, step_freq=1000), "global_step")
        self.assertEqual(_reason(step=2000, step_freq=1000), "global_step")
        # Does not fire just off a boundary.
        self.assertIsNone(_reason(step=999, step_freq=1000))
        self.assertIsNone(_reason(step=1001, step_freq=1000))

    def test_step_zero_does_not_trigger(self) -> None:
        # global_step==0 must not trivially satisfy `0 % N == 0`.
        self.assertIsNone(_reason(step=0, step_freq=1000))

    def test_time_based_interval(self) -> None:
        # At/over the interval -> fires; under -> no save.
        self.assertEqual(
            _reason(step=3, elapsed=3600.0, time_s=3600.0), "time_interval"
        )
        self.assertEqual(
            _reason(step=3, elapsed=4000.0, time_s=3600.0), "time_interval"
        )
        self.assertIsNone(_reason(step=3, elapsed=3599.9, time_s=3600.0))

    def test_in_window_batch_cadence(self) -> None:
        self.assertEqual(_reason(batch=5, in_window=5), "in_window_batch")
        self.assertEqual(_reason(batch=10, in_window=5), "in_window_batch")
        self.assertIsNone(_reason(batch=4, in_window=5))

    def test_precedence_in_window_over_step_over_time(self) -> None:
        # All three would fire this batch; precedence picks in_window first.
        self.assertEqual(
            _reason(
                batch=5,
                step=1000,
                elapsed=9999.0,
                in_window=5,
                step_freq=1000,
                time_s=3600.0,
            ),
            "in_window_batch",
        )
        # in_window not due this batch -> step wins over time.
        self.assertEqual(
            _reason(
                batch=4,
                step=1000,
                elapsed=9999.0,
                in_window=5,
                step_freq=1000,
                time_s=3600.0,
            ),
            "global_step",
        )
        # Neither batch nor step due -> time wins.
        self.assertEqual(
            _reason(
                batch=4,
                step=999,
                elapsed=9999.0,
                in_window=5,
                step_freq=1000,
                time_s=3600.0,
            ),
            "time_interval",
        )

    def test_step_and_time_combined_independent(self) -> None:
        # Step frequency enabled, time disabled: only step boundaries fire.
        self.assertEqual(_reason(step=1000, step_freq=1000, time_s=0.0), "global_step")
        self.assertIsNone(_reason(step=1000, elapsed=1e9, step_freq=0, time_s=0.0))


class CheckpointTimerConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        gin.clear_config()

    def tearDown(self) -> None:
        gin.clear_config()

    def test_empty_save_default_disables_hourly_polling(self) -> None:
        interval = checkpoint_time_interval_for_saving(3600.0)
        self.assertEqual(interval, 0.0)
        self.assertIsNone(_reason(elapsed=7200.0, time_s=interval))

    def test_direct_empty_binding_overrides_nonempty_environment(self) -> None:
        gin.bind_parameter("save_dmp_checkpoint.path", "")
        with mock.patch.dict(os.environ, {"CKPT_PATH": "/unused/environment/path"}):
            self.assertEqual(checkpoint_time_interval_for_saving(3600.0), 0.0)

    def test_direct_destination_preserves_hourly_cadence(self) -> None:
        gin.bind_parameter("save_dmp_checkpoint.path", "/configured/by/gin")
        with mock.patch.dict(os.environ, {"CKPT_PATH": ""}):
            interval = checkpoint_time_interval_for_saving(3600.0)
        self.assertEqual(interval, 3600.0)
        self.assertIsNone(_reason(elapsed=3599.9, time_s=interval))
        self.assertEqual(_reason(elapsed=3600.0, time_s=interval), "time_interval")

    def test_evaluated_environment_reference_uses_effective_path(self) -> None:
        gin.parse_config(
            """
            save_dmp_checkpoint.path = @checkpoint_timer_test/env_path()
            checkpoint_timer_test/env_path.key = "CKPT_PATH"
            checkpoint_timer_test/env_path.default = ""
            """
        )
        for path, expected in (("", 0.0), ("/environment/checkpoints", 3600.0)):
            with self.subTest(path=path):
                with mock.patch.dict(os.environ, {"CKPT_PATH": path}):
                    self.assertEqual(
                        checkpoint_time_interval_for_saving(3600.0), expected
                    )

    def test_current_scope_matches_the_save_binding(self) -> None:
        gin.bind_parameter("save_dmp_checkpoint.path", "/global/checkpoints")
        gin.bind_parameter("disabled/save_dmp_checkpoint.path", "")
        gin.bind_parameter("disabled/enabled/save_dmp_checkpoint.path", "/scoped")
        self.assertEqual(checkpoint_time_interval_for_saving(3600.0), 3600.0)
        with gin.config_scope("disabled"):
            self.assertEqual(checkpoint_time_interval_for_saving(3600.0), 0.0)
            with gin.config_scope("enabled"):
                self.assertEqual(checkpoint_time_interval_for_saving(3600.0), 3600.0)

    def test_unknown_destination_keeps_polling(self) -> None:
        gin.bind_parameter("save_dmp_checkpoint.path", None)
        self.assertEqual(checkpoint_time_interval_for_saving(3600.0), 3600.0)

    def test_unrelated_save_bindings_are_not_evaluated(self) -> None:
        gin.parse_config(
            """
            save_dmp_checkpoint.path = ""
            save_dmp_checkpoint.keep_last_n = @checkpoint_timer_unexpected_reference()
            """
        )
        self.assertEqual(checkpoint_time_interval_for_saving(3600.0), 0.0)

    def test_resolution_failure_keeps_polling(self) -> None:
        with mock.patch.object(gin, "get_bindings", side_effect=ValueError("unknown")):
            with self.assertLogs(
                "generative_recommenders.dlrm_v4.train.utils", level="WARNING"
            ):
                self.assertEqual(checkpoint_time_interval_for_saving(3600.0), 3600.0)

    def test_disabled_timer_does_not_evaluate_destination(self) -> None:
        with mock.patch.object(gin, "get_bindings") as get_bindings:
            self.assertEqual(checkpoint_time_interval_for_saving(0.0), 0.0)
            self.assertEqual(checkpoint_time_interval_for_saving(-1.0), -1.0)
        get_bindings.assert_not_called()


class _CheckpointClockReached(Exception):
    """Stop the real loop after configuration, before training or evaluation."""


class CheckpointPollingGinTest(unittest.TestCase):
    def setUp(self) -> None:
        gin.clear_config()
        gin.bind_parameter("streaming_train_eval_loop.checkpoint_time_interval_s", 3600.0)

    def tearDown(self) -> None:
        gin.clear_config()

    def _loop_interval(self) -> float:
        # A dataset with no holdout/window API is sufficient for loop setup.
        # Replacing this module's clock (not time.time globally) stops at the
        # checkpoint anchor without affecting logging or evaluating any batch.
        clock = mock.Mock()
        clock.time.side_effect = _CheckpointClockReached
        with (
            mock.patch.object(train_utils, "get_dataset", return_value=(
                lambda **kwargs: object(), {},
            )),
            mock.patch.object(train_utils.torch.distributed, "is_initialized", return_value=False),
            mock.patch.object(train_utils, "time", clock),
        ):
            try:
                train_utils.streaming_train_eval_loop(
                    rank=1,
                    model=None,
                    optimizer=None,
                    metric_logger=None,
                    device=None,
                    num_train_ts=0,
                    hstu_config=None,
                    embedding_table_configs={},
                    use_pipeline=False,
                )
            except _CheckpointClockReached as error:
                # Read the actual loop's resolved value, without copying its
                # expression or substituting a second configurable function.
                traceback = error.__traceback__
                while traceback is not None:
                    frame = traceback.tb_frame
                    if frame.f_code.co_name == "streaming_train_eval_loop":
                        return frame.f_locals["effective_checkpoint_time_interval_s"]
                    traceback = traceback.tb_next
                self.fail("The real streaming loop did not reach its checkpoint clock")
        self.fail("The checkpoint clock did not stop the streaming loop")

    def test_default_skips_polling_for_empty_destination(self) -> None:
        self.assertEqual(self._loop_interval(), 0.0)

    def test_gin_switch_controls_actual_loop_interval(self) -> None:
        for enabled, expected in ((True, 0.0), (False, 3600.0)):
            with self.subTest(enabled=enabled):
                gin.parse_config(
                    "streaming_train_eval_loop.skip_checkpoint_polling_when_disabled = "
                    f"{enabled}"
                )
                with mock.patch.object(
                    train_utils, "checkpoint_time_interval_for_saving",
                    wraps=checkpoint_time_interval_for_saving,
                ) as resolve:
                    interval = self._loop_interval()
                self.assertEqual(interval, expected)
                if enabled:
                    resolve.assert_called_once_with(3600.0)
                    self.assertIsNone(_reason(elapsed=7200.0, time_s=interval))
                else:
                    resolve.assert_not_called()
                    self.assertEqual(
                        _reason(elapsed=7200.0, time_s=interval), "time_interval"
                    )

    def test_off_bypasses_destination_resolution(self) -> None:
        gin.parse_config(
            """
            streaming_train_eval_loop.skip_checkpoint_polling_when_disabled = False
            save_dmp_checkpoint.path = @checkpoint_timer_unexpected_reference()
            """
        )
        with mock.patch.object(gin, "get_bindings", wraps=gin.get_bindings) as bindings:
            self.assertEqual(self._loop_interval(), 3600.0)
        bindings.assert_not_called()

    def test_nonempty_destination_retains_requested_timer_in_both_modes(self) -> None:
        gin.bind_parameter("save_dmp_checkpoint.path", "/configured/checkpoints")
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                gin.bind_parameter(
                    "streaming_train_eval_loop.skip_checkpoint_polling_when_disabled",
                    enabled,
                )
                self.assertEqual(self._loop_interval(), 3600.0)

    def test_scoped_destination_and_switch_reach_actual_loop(self) -> None:
        gin.parse_config(
            """
            save_dmp_checkpoint.path = "/global/checkpoints"
            disabled/save_dmp_checkpoint.path = ""
            disabled/rollback/streaming_train_eval_loop.skip_checkpoint_polling_when_disabled = False
            disabled/saving/save_dmp_checkpoint.path = "/scoped/checkpoints"
            """
        )
        self.assertEqual(self._loop_interval(), 3600.0)
        with gin.config_scope("disabled"):
            self.assertEqual(self._loop_interval(), 0.0)
            with gin.config_scope("rollback"):
                self.assertEqual(self._loop_interval(), 3600.0)
            with gin.config_scope("saving"):
                self.assertEqual(self._loop_interval(), 3600.0)

    def test_unknown_destination_retains_timer_in_actual_loop(self) -> None:
        gin.bind_parameter("save_dmp_checkpoint.path", None)
        self.assertEqual(self._loop_interval(), 3600.0)

    def test_skipping_timer_preserves_batch_and_step_cadences(self) -> None:
        interval = self._loop_interval()
        self.assertEqual(interval, 0.0)
        self.assertEqual(
            _reason(batch=5, in_window=5, time_s=interval), "in_window_batch"
        )
        self.assertEqual(
            _reason(step=1000, step_freq=1000, time_s=interval), "global_step"
        )

    def test_destination_helper_is_independent_of_polling_rollback(self) -> None:
        gin.bind_parameter(
            "streaming_train_eval_loop.skip_checkpoint_polling_when_disabled", False
        )
        # Other callers still need to detect that the save destination is
        # disabled independently of the loop's polling policy.
        self.assertEqual(checkpoint_time_interval_for_saving(1.0), 0.0)
        self.assertEqual(self._loop_interval(), 3600.0)


if __name__ == "__main__":
    unittest.main()
