###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Resettable validation iterator for deterministic MLPerf eval."""


class ResettableDataIterator:
    """Iterator wrapper that restarts from the beginning of the dataloader on reset().

    Unlike cyclic_iter which continuously cycles, this iterator explicitly resets
    to produce the exact same sequence of batches on each reset(). This guarantees
    deterministic validation: every evaluation pass sees identical data in identical
    order, regardless of how many evaluations have occurred.
    """

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._dataloader)
            return next(self._iterator)

    def reset(self):
        """Reset to the beginning of the dataloader for deterministic iteration."""
        self._iterator = iter(self._dataloader)
