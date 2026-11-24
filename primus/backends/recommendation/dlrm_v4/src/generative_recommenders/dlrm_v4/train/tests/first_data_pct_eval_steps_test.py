# pyre-strict
"""Unit tests for ``first_data_pct_eval_steps`` -- the training steps from the
start of the train stream to the first data-fraction eval. Item (36)'s eval
holdout read-ahead (cold_prefetch.set_eval_plan) is timed from it; the loop
itself keeps its own trigger (global_step > 0, global_step % interval == 0,
global_step >= skip_eval_until_step), which ``_first_by_loop`` replays here.

No GPU, no distributed init: the helper is pure arithmetic. The production
cases use the cadence the ship13 production runs logged ("[data-pct-eval]
eval_interval_steps=..." and "[skip-eval] ... skip_eval_until_step=..." lines,
first "trigger eval ... global_step=..." line) at 1/2/4 nodes.
"""
import unittest

from generative_recommenders.dlrm_v4.train.utils import first_data_pct_eval_steps


def _first_by_loop(gstep0: int, interval: int, skip: int, limit: int = 100_000) -> int:
    """Steps from gstep0 to the first step whose global step g triggers an eval
    in the loop (g = gstep0 + k after the k-th step); 0 if none within limit."""
    for k in range(1, limit + 1):
        g = gstep0 + k
        if interval > 0 and g > 0 and g % interval == 0 and not g < skip:
            return k
    return 0


class FirstDataPctEvalStepsTest(unittest.TestCase):
    def test_matches_the_loop_trigger(self) -> None:
        for interval in (1, 2, 7, 70, 140, 280):
            for skip in (-5, 0, 1, 3, 69, 70, 71, 2867, 4237, 6992):
                for gstep0 in (0, 1, 5, 69, 70, 71, 2869, 2870, 4339, 6999, 7000):
                    with self.subTest(interval=interval, skip=skip, gstep0=gstep0):
                        self.assertEqual(
                            first_data_pct_eval_steps(gstep0, interval, skip),
                            _first_by_loop(gstep0, interval, skip),
                        )

    def test_cadence_off(self) -> None:
        for interval in (0, -1):
            self.assertEqual(first_data_pct_eval_steps(0, interval, 100), 0)

    def test_production_ship13(self) -> None:
        # (interval, skip_eval_until_step, first eval global step) logged by the
        # ship13 final cold-start runs at 1 / 2 / 4 nodes (GBS 8192/16384/32768)
        for interval, skip, first in ((280, 6992, 7000), (140, 4237, 4340), (70, 2867, 2870)):
            with self.subTest(interval=interval):
                self.assertEqual(first_data_pct_eval_steps(0, interval, skip), first)
                # resumed at global step 3000 (4n): the next boundary past it
                if interval == 70:
                    self.assertEqual(first_data_pct_eval_steps(3000, interval, skip), 10)


if __name__ == "__main__":
    unittest.main()
