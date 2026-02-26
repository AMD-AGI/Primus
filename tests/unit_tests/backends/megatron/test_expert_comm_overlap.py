###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for MoE expert ↔ communication overlap chunking logic.

These tests validate the token splitting and merging functions that form
the foundation of the overlap pipeline, without requiring GPU hardware
or Megatron initialization.
"""

import unittest

import torch

from primus.backends.megatron.core.transformer.moe.expert_comm_overlap import (
    _merge_chunk_outputs,
    _split_by_expert_chunks,
)


class TestSplitByExpertChunks(unittest.TestCase):
    """Test token splitting into chunks respecting per-expert boundaries."""

    def test_single_chunk_returns_original(self):
        """When num_chunks=1, output should be identical to input."""
        hidden = torch.randn(10, 64)
        tpe = torch.tensor([3, 4, 3])
        probs = torch.randn(10)

        chunks = _split_by_expert_chunks(hidden, tpe, probs, num_chunks=1)

        self.assertEqual(len(chunks), 1)
        self.assertTrue(torch.equal(chunks[0][0], hidden))
        self.assertTrue(torch.equal(chunks[0][1], tpe))
        self.assertTrue(torch.equal(chunks[0][2], probs))

    def test_two_chunks_preserve_total_tokens(self):
        """Splitting into 2 chunks should preserve total token count."""
        num_experts = 4
        tpe = torch.tensor([10, 8, 6, 12])
        total_tokens = tpe.sum().item()
        hidden = torch.randn(total_tokens, 128)
        probs = torch.randn(total_tokens)

        chunks = _split_by_expert_chunks(hidden, tpe, probs, num_chunks=2)

        self.assertEqual(len(chunks), 2)
        total_in_chunks = sum(c[0].shape[0] for c in chunks)
        self.assertEqual(total_in_chunks, total_tokens)

    def test_tokens_per_expert_sum_preserved(self):
        """Sum of tokens_per_expert across chunks should equal original."""
        tpe = torch.tensor([7, 5, 3, 9])
        total = tpe.sum().item()
        hidden = torch.randn(total, 64)
        probs = torch.randn(total)

        chunks = _split_by_expert_chunks(hidden, tpe, probs, num_chunks=3)

        # Sum tokens_per_expert across all chunks for each expert
        num_experts = tpe.shape[0]
        for e_idx in range(num_experts):
            chunk_sum = sum(c[1][e_idx].item() for c in chunks)
            self.assertEqual(chunk_sum, tpe[e_idx].item())

    def test_three_chunks_with_uneven_distribution(self):
        """Uneven token counts should be distributed correctly."""
        tpe = torch.tensor([1, 2, 0, 5])
        total = tpe.sum().item()
        hidden = torch.randn(total, 32)
        probs = torch.randn(total)

        chunks = _split_by_expert_chunks(hidden, tpe, probs, num_chunks=3)

        self.assertEqual(len(chunks), 3)
        # Expert with 0 tokens should have 0 in all chunks
        for c in chunks:
            self.assertEqual(c[1][2].item(), 0)

    def test_empty_input(self):
        """Empty input should produce single empty chunk."""
        hidden = torch.randn(0, 64)
        tpe = torch.tensor([0, 0, 0])
        probs = torch.randn(0)

        chunks = _split_by_expert_chunks(hidden, tpe, probs, num_chunks=2)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0].shape[0], 0)

    def test_probs_shape_1d(self):
        """1D probs should be correctly sliced."""
        tpe = torch.tensor([4, 6])
        total = tpe.sum().item()
        hidden = torch.randn(total, 16)
        probs = torch.arange(total, dtype=torch.float32)

        chunks = _split_by_expert_chunks(hidden, tpe, probs, num_chunks=2)

        total_probs = sum(c[2].shape[0] for c in chunks)
        self.assertEqual(total_probs, total)


class TestMergeChunkOutputs(unittest.TestCase):
    """Test merging chunked expert outputs back to expert-sorted order."""

    def test_single_chunk_passthrough(self):
        """Single chunk merge should return the tensor unchanged."""
        output = torch.randn(10, 64)
        tpe = [torch.tensor([3, 4, 3])]

        merged = _merge_chunk_outputs([output], tpe, num_experts=3)

        self.assertTrue(torch.equal(merged, output))

    def test_merge_preserves_total_tokens(self):
        """Merged output should have same total tokens as sum of chunks."""
        # 3 experts, 2 chunks
        c0_tpe = torch.tensor([2, 3, 1])
        c1_tpe = torch.tensor([3, 2, 4])
        total = c0_tpe.sum().item() + c1_tpe.sum().item()

        c0_out = torch.randn(c0_tpe.sum().item(), 32)
        c1_out = torch.randn(c1_tpe.sum().item(), 32)

        merged = _merge_chunk_outputs([c0_out, c1_out], [c0_tpe, c1_tpe], num_experts=3)

        self.assertEqual(merged.shape[0], total)
        self.assertEqual(merged.shape[1], 32)

    def test_roundtrip_split_merge(self):
        """Split then merge should produce a tensor with the same content (permuted)."""
        tpe = torch.tensor([4, 6, 5])
        total = tpe.sum().item()
        hidden = torch.randn(total, 16)
        probs = torch.randn(total)

        # Split into chunks
        chunks = _split_by_expert_chunks(hidden, tpe, probs, num_chunks=2)

        # Merge back
        chunk_outputs = [c[0] for c in chunks]
        chunk_tpes = [c[1] for c in chunks]
        merged = _merge_chunk_outputs(chunk_outputs, chunk_tpes, num_experts=3)

        # Total tokens should be preserved
        self.assertEqual(merged.shape[0], total)
        self.assertEqual(merged.shape[1], 16)

    def test_empty_merge(self):
        """Merging empty chunks should work."""
        c0_out = torch.randn(0, 64)
        c0_tpe = torch.tensor([0, 0])

        merged = _merge_chunk_outputs([c0_out], [c0_tpe], num_experts=2)
        self.assertEqual(merged.shape[0], 0)

    def test_merge_values_correct(self):
        """Verify that merged values from two chunks are in correct expert order."""
        # Expert 0: 2 tokens, Expert 1: 3 tokens
        # Chunk 0: expert0=1, expert1=2  → 3 tokens
        # Chunk 1: expert0=1, expert1=1  → 2 tokens
        c0_tpe = torch.tensor([1, 2])
        c1_tpe = torch.tensor([1, 1])

        # Create identifiable values
        # Chunk 0: [e0_t0, e1_t0, e1_t1]
        c0_out = torch.tensor([[1.0], [3.0], [4.0]])
        # Chunk 1: [e0_t1, e1_t2]
        c1_out = torch.tensor([[2.0], [5.0]])

        merged = _merge_chunk_outputs([c0_out, c1_out], [c0_tpe, c1_tpe], num_experts=2)

        # Expected order: expert 0 tokens (1, 2), then expert 1 tokens (3, 4, 5)
        expected = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        self.assertTrue(torch.equal(merged, expected))


if __name__ == "__main__":
    unittest.main()
