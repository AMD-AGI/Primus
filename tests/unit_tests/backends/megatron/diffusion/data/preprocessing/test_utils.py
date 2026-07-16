# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for preprocessing utility functions.

Tests distributed processing utilities and work splitting logic.
"""

from unittest.mock import patch

import pytest

from primus.backends.megatron.data.diffusion.preprocessing.utils import (
    get_distributed_info,
    split_work_for_rank,
)
from tests.utils import PrimusUT


class TestGetDistributedInfo(PrimusUT):
    """Tests for get_distributed_info utility."""

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.is_available")
    def test_not_distributed(self, mock_is_available, mock_is_initialized):
        """Test get_distributed_info when not in distributed mode."""
        mock_is_available.return_value = True
        mock_is_initialized.return_value = False

        rank, world_size = get_distributed_info()

        assert rank == 0
        assert world_size == 1

    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.is_available")
    def test_distributed_mode(
        self, mock_is_available, mock_is_initialized, mock_get_rank, mock_get_world_size
    ):
        """Test get_distributed_info in distributed mode."""
        mock_is_available.return_value = True
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 2
        mock_get_world_size.return_value = 8

        rank, world_size = get_distributed_info()

        assert rank == 2
        assert world_size == 8


class TestSplitWorkForRank(PrimusUT):
    """Tests for split_work_for_rank utility."""

    def test_multiple_ranks_evenly_divisible(self):
        """Test work splitting when items divide evenly."""
        world_size = 4
        total_items = 100

        expected_splits = [
            (0, 25),  # rank 0
            (25, 50),  # rank 1
            (50, 75),  # rank 2
            (75, 100),  # rank 3 (last rank gets remainder)
        ]

        for rank, (expected_start, expected_end) in enumerate(expected_splits):
            start, end = split_work_for_rank(total_items, rank, world_size)
            assert start == expected_start
            assert end == expected_end

    def test_multiple_ranks_with_remainder(self):
        """Test work splitting when items don't divide evenly."""
        world_size = 3
        total_items = 100

        expected_splits = [
            (0, 33),  # rank 0: 33 items
            (33, 66),  # rank 1: 33 items
            (66, 100),  # rank 2: 34 items (gets remainder)
        ]

        for rank, (expected_start, expected_end) in enumerate(expected_splits):
            start, end = split_work_for_rank(total_items, rank, world_size)
            assert start == expected_start
            assert end == expected_end

    def test_last_rank_gets_remainder(self):
        """Test that last rank gets any remaining items."""
        total_items = 107
        world_size = 8
        items_per_rank = total_items // world_size  # 13

        # Last rank should get remainder
        start, end = split_work_for_rank(total_items, rank=7, world_size=world_size)

        assert start == 7 * items_per_rank  # 91
        assert end == total_items  # 107 (gets all remaining)

    def test_covers_all_items(self):
        """Test that all ranks together cover all items."""
        total_items = 137
        world_size = 7

        all_indices = set()
        for rank in range(world_size):
            start, end = split_work_for_rank(total_items, rank, world_size)
            rank_indices = set(range(start, end))
            # No overlap
            assert len(all_indices & rank_indices) == 0
            all_indices.update(rank_indices)

        # All items covered
        assert all_indices == set(range(total_items))

    def test_fewer_items_than_ranks(self):
        """Test work splitting when items < ranks."""
        total_items = 5
        world_size = 10

        # First 5 ranks get 0 items each (5 // 10 = 0)
        # Last rank gets all items
        for rank in range(world_size - 1):
            start, end = split_work_for_rank(total_items, rank, world_size)
            # Each rank gets 0 items except last
            assert start == 0
            assert end == 0

        # Last rank gets all
        start, end = split_work_for_rank(total_items, rank=world_size - 1, world_size=world_size)
        assert end == total_items


class TestWorkSplittingEdgeCases(PrimusUT):
    """Test edge cases for work splitting."""

    def test_empty_work(self):
        """Test with zero items."""
        start, end = split_work_for_rank(total=0, rank=0, world_size=4)

        assert start == 0
        assert end == 0

    def test_one_item(self):
        """Test with single item."""
        world_size = 4

        # Ranks 0-2 get 0 items (1 // 4 = 0)
        for rank in range(world_size - 1):
            start, end = split_work_for_rank(1, rank, world_size)
            assert start == 0
            assert end == 0

        # Last rank gets the 1 item
        start, end = split_work_for_rank(1, rank=world_size - 1, world_size=world_size)
        assert start == 0
        assert end == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
