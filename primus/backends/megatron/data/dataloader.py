# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Adapted from Megatron-LM multimodal examples
# Reference: examples/multimodal/dataloader_provider.py

"""
Generic Megatron-compatible dataloader wrapper.

This module provides utilities to adapt any iterable (PyTorch DataLoader,
Megatron Energon loader, synthetic data, etc.) to work with Megatron's
training loop.

Key Features:
    - Cyclic iteration (never raises StopIteration)
    - Optional checkpoint support via duck typing
    - Works with any Python iterable
    - No dependencies on specific data sources

Historical Note:
    Originally named "EnergonDataloader" but has no Energon dependencies.
    Renamed to MegatronDataloaderWrapper for clarity.
"""

import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def cyclic_iter(iterator: Iterator) -> Iterator:
    """
    Create a cyclic iterator that restarts when exhausted.

    Megatron's training loop expects infinite iterators that never
    raise StopIteration. This wrapper cycles any iterator infinitely.

    Args:
        iterator: Any Python iterator

    Yields:
        Items from iterator, cycling infinitely

    Example:
        >>> loader = DataLoader(dataset, batch_size=4)
        >>> infinite_loader = cyclic_iter(loader)
        >>> # Never raises StopIteration

    Reference:
        Megatron-LM examples/multimodal/dataloader_provider.py:cyclic_iter
    """
    while True:
        for item in iterator:
            yield item


class MegatronDataloaderWrapper:
    """
    Generic wrapper to make any iterable compatible with Megatron training loop.

    This wrapper is completely generic and has NO dependencies on specific
    data sources. It works with:
        - PyTorch DataLoader (for synthetic/mock data)
        - Megatron Energon loaders (for WebDataset)
        - Any Python iterable

    Features:
        1. Cyclic iteration - never raises StopIteration
        2. Optional state management - duck-typed for compatibility
        3. Megatron training loop compatibility

    The wrapper uses duck typing for checkpoint methods. If your dataloader
    has save_state_rank() or restore_state_rank(), they'll be used.
    Otherwise, state operations are no-ops (perfect for synthetic data).

    Example with PyTorch DataLoader (synthetic data):
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(mock_dataset, batch_size=4)
        >>> megatron_loader = MegatronDataloaderWrapper(loader)
        >>> for batch in megatron_loader:
        ...     # Never exhausts, cycles infinitely
        ...     pass

    Example with Energon (real data):
        >>> from megatron.energon import get_loader
        >>> energon_loader = get_loader(...)
        >>> megatron_loader = MegatronDataloaderWrapper(energon_loader)
        >>> # Checkpoint support via save_state_rank() is available

    Reference:
        Megatron-LM examples/multimodal/dataloader_provider.py:EnergonDataloader
    """

    def __init__(self, dataloader):
        """
        Initialize wrapper for any iterable.

        Args:
            dataloader: Any iterable (PyTorch DataLoader, Energon loader, etc.)
        """
        self._dataloader = dataloader
        self._iter = iter(cyclic_iter(dataloader))
        logger.debug(f"Initialized MegatronDataloaderWrapper for {type(dataloader).__name__}")

    def __next__(self):
        """Get next batch (never raises StopIteration)."""
        return next(self._iter)

    def __iter__(self):
        """Return iterator."""
        return self._iter

    def save_state(self) -> Any:
        """
        Save dataloader state for checkpointing (if supported).

        Uses duck typing - checks for save_state_rank() method.
        Returns None if not supported (e.g., for synthetic/mock data).

        Returns:
            Dataloader state dictionary, or None if not supported
        """
        if hasattr(self._dataloader, "save_state_rank"):
            return self._dataloader.save_state_rank()
        else:
            logger.debug(
                f"{type(self._dataloader).__name__} does not support save_state_rank() "
                f"(OK for synthetic data)"
            )
            return None

    def restore_state(self, state: Any):
        """
        Restore dataloader state from checkpoint (if supported).

        Uses duck typing - checks for restore_state_rank() method.
        No-op if not supported (e.g., for synthetic/mock data).

        Args:
            state: Dataloader state dictionary
        """
        if hasattr(self._dataloader, "restore_state_rank"):
            self._dataloader.restore_state_rank(state)
            # Recreate iterator after restore
            self._iter = iter(cyclic_iter(self._dataloader))
            logger.info("Restored dataloader state from checkpoint")
        else:
            logger.debug(
                f"{type(self._dataloader).__name__} does not support restore_state_rank() "
                f"(OK for synthetic data)"
            )


# Backwards compatibility alias (DEPRECATED)
# NOTE: Deprecated alias kept for backward compatibility.
EnergonDataloader = MegatronDataloaderWrapper


__all__ = [
    "MegatronDataloaderWrapper",
    "cyclic_iter",
    "EnergonDataloader",  # Deprecated, use MegatronDataloaderWrapper
]
