###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for _fused_topk_bool_mask — the optimized aux loss routing map
construction that replaces the 4-kernel sequence with a 2-kernel fused path.
"""

import unittest

import torch

from primus.backends.megatron.core.transformer.moe.router import _fused_topk_bool_mask


class TestFusedTopkBoolMask(unittest.TestCase):
    """Test _fused_topk_bool_mask correctness against the original implementation."""

    def _reference_topk_mask(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        """Original 4-kernel implementation for reference."""
        _, top_indices = torch.topk(scores, k=k, dim=1)
        return torch.zeros_like(scores).int().scatter(1, top_indices, 1).bool()

    def test_basic_correctness(self):
        """Fused mask should match reference implementation."""
        scores = torch.randn(32, 64)
        k = 8

        expected = self._reference_topk_mask(scores, k)
        actual = _fused_topk_bool_mask(scores, k)

        self.assertTrue(torch.equal(expected, actual))

    def test_output_dtype_is_bool(self):
        """Output should be bool dtype directly (no int intermediate)."""
        scores = torch.randn(16, 32)
        mask = _fused_topk_bool_mask(scores, k=4)
        self.assertEqual(mask.dtype, torch.bool)

    def test_correct_number_of_true_per_row(self):
        """Each row should have exactly k True values."""
        num_tokens, num_experts, k = 64, 256, 8
        scores = torch.randn(num_tokens, num_experts)
        mask = _fused_topk_bool_mask(scores, k=k)

        true_per_row = mask.sum(dim=1)
        self.assertTrue(torch.all(true_per_row == k))

    def test_topk_positions_match_highest_scores(self):
        """True positions should correspond to the top-k scores."""
        scores = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        mask = _fused_topk_bool_mask(scores, k=3)

        # Top-3 scores are at indices 1 (5.0), 4 (4.0), 2 (3.0)
        expected = torch.tensor([[False, True, True, False, True]])
        self.assertTrue(torch.equal(mask, expected))

    def test_large_tensor(self):
        """Should handle large tensors efficiently (DeepSeek-V3 scale: 4096 tokens × 256 experts)."""
        scores = torch.randn(4096, 256)
        k = 8
        mask = _fused_topk_bool_mask(scores, k=k)

        self.assertEqual(mask.shape, (4096, 256))
        self.assertTrue(torch.all(mask.sum(dim=1) == k))

    def test_k_equals_1(self):
        """Edge case: k=1 should select exactly one expert per token."""
        scores = torch.randn(100, 64)
        mask = _fused_topk_bool_mask(scores, k=1)
        self.assertTrue(torch.all(mask.sum(dim=1) == 1))

    def test_k_equals_num_experts(self):
        """Edge case: k=num_experts should select all experts."""
        num_experts = 8
        scores = torch.randn(10, num_experts)
        mask = _fused_topk_bool_mask(scores, k=num_experts)
        self.assertTrue(torch.all(mask))

    def test_correctness_across_dtypes(self):
        """Should work with float32 and float16 scores."""
        for dtype in [torch.float32, torch.float16]:
            scores = torch.randn(32, 64, dtype=dtype)
            ref = self._reference_topk_mask(scores, k=6)
            actual = _fused_topk_bool_mask(scores, k=6)
            self.assertTrue(torch.equal(ref, actual), f"Mismatch for dtype={dtype}")

    def test_deterministic(self):
        """Same input should always produce same output."""
        scores = torch.randn(50, 128)
        m1 = _fused_topk_bool_mask(scores, k=8)
        m2 = _fused_topk_bool_mask(scores, k=8)
        self.assertTrue(torch.equal(m1, m2))


if __name__ == "__main__":
    unittest.main()
