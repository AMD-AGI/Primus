#!/usr/bin/env python3

# pyre-strict

"""The FlyDSL backward must not sync the host once per layer.

`.item()` on a device tensor drains the launch queue, and in the backward the
queue is at its deepest -- profiled at 31.7 ms a call against 0.6 ms for the
same call in the forward. Two call sites used to pay it on every layer of every
step; these tests pin the behaviour that replaced them.
"""

import unittest

import torch
from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha


class FlyDSLTargetsMemoTest(unittest.TestCase):
    def setUp(self) -> None:
        mha._targets_max_memo = (None, -1, 0)

    def test_reads_the_max_and_remembers_the_tensor(self) -> None:
        num_targets = torch.tensor([0, 1, 1, 0])
        self.assertEqual(mha._max_num_targets(num_targets), 1)
        self.assertIs(mha._targets_max_memo[0], num_targets)

    def test_second_call_on_the_same_tensor_does_not_re_read(self) -> None:
        num_targets = torch.tensor([0, 1, 1])
        mha._max_num_targets(num_targets)
        # Poison the memo with a value the tensor does not contain: if the
        # second call still returns it, the device read was genuinely skipped.
        mha._targets_max_memo = (num_targets, num_targets._version, 99)
        self.assertEqual(mha._max_num_targets(num_targets), 99)

    def test_a_different_tensor_misses(self) -> None:
        mha._max_num_targets(torch.tensor([0, 1]))
        self.assertEqual(mha._max_num_targets(torch.tensor([0, 2])), 2)

    def test_an_in_place_write_invalidates(self) -> None:
        num_targets = torch.tensor([0, 1, 1])
        self.assertEqual(mha._max_num_targets(num_targets), 1)
        num_targets[0] = 7
        self.assertEqual(mha._max_num_targets(num_targets), 7)

    def test_the_support_check_still_rejects_multi_target_batches(self) -> None:
        """Memoizing must not weaken the guard it feeds.

        num_targets > 1 makes HSTU's target-aware clamp a real operation rather
        than an identity, so the kernel computes a different function and has to
        decline the shape.
        """
        q = torch.zeros(8, 4, 128, dtype=torch.bfloat16)
        reason = mha.unsupported_shape_reason(
            q, q, q, q,
            num_targets=torch.tensor([1, 2]),
            max_attn_len=0,
            contextual_seq_len=0,
            num_softmax_heads=0,
        )
        self.assertIsNotNone(reason)
        self.assertIn("num_targets max=2", reason or "")

    def test_a_single_target_batch_is_still_served(self) -> None:
        q = torch.zeros(8, 4, 128, dtype=torch.bfloat16)
        self.assertIsNone(
            mha.unsupported_shape_reason(
                q, q, q, q,
                num_targets=torch.tensor([0, 1, 1]),
                max_attn_len=0,
                contextual_seq_len=0,
                num_softmax_heads=0,
            )
        )


class FlyDSLBlockCountTest(unittest.TestCase):
    def test_block_count_comes_from_max_seq_len_not_a_device_read(self) -> None:
        """n_blocks must be derived from the caller's N.

        seq_offsets.diff().max() is the same number in production -- audited
        equal on 3,200 calls across 8 ranks -- but reading it costs a sync.
        """
        import inspect

        source = inspect.getsource(mha.flydsl_hstu_attention_bwd)
        self.assertNotIn("lengths.max()", source)
        self.assertIn("n_blocks = int((int(N) + fn.BN - 1) // fn.BN)", source)


if __name__ == "__main__":
    unittest.main()
