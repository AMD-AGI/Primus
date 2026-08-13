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
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

DATALOADER_STATE_KEY = "dataloader_state_dict"
DATALOADER_STATE_FORMAT_VERSION = 1
DATALOADER_STATE_PAYLOAD_KEY = "state"


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

    def __init__(
        self,
        dataloader,
        *,
        data_parallel_rank: Optional[int] = None,
        data_parallel_world_size: Optional[int] = None,
    ):
        """
        Initialize wrapper for any iterable.

        Args:
            dataloader: Any iterable (PyTorch DataLoader, Energon loader, etc.)
        """
        if (data_parallel_rank is None) != (data_parallel_world_size is None):
            raise ValueError("data_parallel_rank and data_parallel_world_size must be provided together")
        if data_parallel_rank is not None:
            if data_parallel_rank < 0:
                raise ValueError("data_parallel_rank must be non-negative")
            if data_parallel_world_size <= 0:
                raise ValueError("data_parallel_world_size must be positive")
            if data_parallel_rank >= data_parallel_world_size:
                raise ValueError("data_parallel_rank must be smaller than data_parallel_world_size")

        self._dataloader = dataloader
        self._data_parallel_rank = data_parallel_rank
        self._data_parallel_world_size = data_parallel_world_size
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
            state = self._dataloader.save_state_rank()
            if self._data_parallel_rank is None:
                return state
            if state is None:
                raise RuntimeError("Stateful dataloader returned no state for exact checkpoint continuation")
            return {
                "format_version": DATALOADER_STATE_FORMAT_VERSION,
                "data_parallel_rank": self._data_parallel_rank,
                "data_parallel_world_size": self._data_parallel_world_size,
                DATALOADER_STATE_PAYLOAD_KEY: state,
            }
        else:
            logger.debug(
                f"{type(self._dataloader).__name__} does not support save_state_rank() "
                f"(OK for synthetic data)"
            )
            return None

    def restore_state(self, state: Any, *, strict: bool = False) -> bool:
        """
        Restore dataloader state from checkpoint (if supported).

        Uses duck typing - checks for restore_state_rank() method.
        No-op if not supported (e.g., for synthetic/mock data), unless strict=True.

        Args:
            state: Dataloader state dictionary
            strict: Raise when the wrapped dataloader cannot restore state.

        Returns:
            True when state was restored, otherwise False.
        """
        if state is None and strict:
            raise RuntimeError("Dataloader checkpoint contains an empty state")

        if self._data_parallel_rank is not None:
            if not isinstance(state, dict):
                raise RuntimeError("Dataloader checkpoint has no topology-aware state envelope")
            expected_metadata = {
                "format_version": DATALOADER_STATE_FORMAT_VERSION,
                "data_parallel_rank": self._data_parallel_rank,
                "data_parallel_world_size": self._data_parallel_world_size,
            }
            observed_metadata = {key: state.get(key) for key in expected_metadata}
            if observed_metadata != expected_metadata:
                raise RuntimeError(
                    "Dataloader checkpoint topology does not match the current run: "
                    f"saved={observed_metadata}, expected={expected_metadata}"
                )
            if DATALOADER_STATE_PAYLOAD_KEY not in state:
                raise RuntimeError("Dataloader checkpoint state envelope has no payload")
            state = state[DATALOADER_STATE_PAYLOAD_KEY]
            if state is None:
                raise RuntimeError("Dataloader checkpoint state payload is empty")

        if hasattr(self._dataloader, "restore_state_rank"):
            self._dataloader.restore_state_rank(state)
            # Recreate iterator after restore
            self._iter = iter(cyclic_iter(self._dataloader))
            logger.info("Restored dataloader state from checkpoint")
            return True
        else:
            if strict:
                raise RuntimeError(f"{type(self._dataloader).__name__} does not support restore_state_rank()")
            logger.debug(
                f"{type(self._dataloader).__name__} does not support restore_state_rank() "
                f"(OK for synthetic data)"
            )
            return False


def restore_dataloader_state_from_checkpoint(
    dataloader: MegatronDataloaderWrapper,
    checkpoint_root: str,
    iteration: int,
    data_parallel_rank: int,
    *,
    checkpoint_name_fn: Optional[Callable[..., str]] = None,
    load_fn: Optional[Callable[[Path], Any]] = None,
) -> Path:
    """Restore one data-parallel rank's dataloader checkpoint.

    ``maybe_save_dataloader_state`` in Megatron writes one
    ``train_dataloader_dprankNNN.pt`` file per data-parallel rank under the
    model checkpoint iteration directory. This helper resolves that exact
    path and fails closed when the file or payload is invalid.
    """
    if iteration < 0:
        raise ValueError(f"iteration must be non-negative, got {iteration}")
    if data_parallel_rank < 0:
        raise ValueError(f"data_parallel_rank must be non-negative, got {data_parallel_rank}")

    if checkpoint_name_fn is None:
        from megatron.training.checkpointing import get_checkpoint_name

        checkpoint_name_fn = get_checkpoint_name

    checkpoint_path = Path(
        checkpoint_name_fn(
            checkpoint_root,
            iteration,
            tensor_rank=0,
            pipeline_rank=0,
            basename=f"train_dataloader_dprank{data_parallel_rank:03d}.pt",
        )
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Energon dataloader checkpoint is missing for "
            f"iteration={iteration}, data_parallel_rank={data_parallel_rank}: "
            f"{checkpoint_path}"
        )

    if load_fn is None:
        import torch

        # This file is produced by the same trusted Megatron checkpoint run.
        def _torch_load(path: Path) -> Any:
            return torch.load(path, map_location="cpu", weights_only=False)

        load_fn = _torch_load

    payload = load_fn(checkpoint_path)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a dictionary in dataloader checkpoint {checkpoint_path}, "
            f"got {type(payload).__name__}"
        )
    if DATALOADER_STATE_KEY not in payload:
        raise KeyError(f"Dataloader checkpoint {checkpoint_path} has no {DATALOADER_STATE_KEY!r} entry")

    dataloader.restore_state(payload[DATALOADER_STATE_KEY], strict=True)
    return checkpoint_path


# Backwards compatibility alias (DEPRECATED)
# NOTE: Deprecated alias kept for backward compatibility.
EnergonDataloader = MegatronDataloaderWrapper


__all__ = [
    "DATALOADER_STATE_FORMAT_VERSION",
    "DATALOADER_STATE_KEY",
    "DATALOADER_STATE_PAYLOAD_KEY",
    "MegatronDataloaderWrapper",
    "cyclic_iter",
    "EnergonDataloader",  # Deprecated, use MegatronDataloaderWrapper
    "restore_dataloader_state_from_checkpoint",
]
