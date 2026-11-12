# pyre-strict
"""CPU-only test of the DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT_SHIFT=auto policy."""

import unittest

from generative_recommenders.ops.triton.bwd_fp8 import next_t_shift


class NextTShiftTest(unittest.TestCase):
    def test_clip_steps_up_and_caps_at_7(self) -> None:
        self.assertEqual(next_t_shift(5, 1, 3, 5, 4), (6, 0))
        self.assertEqual(next_t_shift(7, 2, 0, 5, 4), (7, 0))

    def test_relax_after_clean_windows_not_below_floor(self) -> None:
        shift, clean = 6, 0
        for _ in range(3):
            shift, clean = next_t_shift(shift, 0, clean, 5, 4)
            self.assertEqual(shift, 6)
        shift, clean = next_t_shift(shift, 0, clean, 5, 4)
        self.assertEqual((shift, clean), (5, 0))
        for _ in range(10):
            shift, clean = next_t_shift(shift, 0, clean, 5, 4)
        self.assertEqual(shift, 5)

    def test_clip_resets_clean_count(self) -> None:
        shift, clean = next_t_shift(6, 0, 2, 5, 4)
        self.assertEqual((shift, clean), (6, 3))
        shift, clean = next_t_shift(shift, 1, clean, 5, 4)
        self.assertEqual((shift, clean), (7, 0))


if __name__ == "__main__":
    unittest.main()
