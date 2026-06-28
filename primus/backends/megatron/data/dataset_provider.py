# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Abstract DatasetProvider for Megatron trainers.

This module provides a strategy pattern for pluggable data pipelines,
allowing different trainers to use different data sources while sharing
the same training infrastructure.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple


class DatasetProvider(ABC):
    """
    Abstract strategy for dataset/dataloader creation.

    Implementations provide different data pipelines:
        - EnergonDatasetProvider: Megatron Energon for multimodal/diffusion
        - SyntheticDatasetProvider: synthetic/mock batches for smoke tests

    The provider pattern allows MegatronTrainer to remain agnostic about
    the underlying data source while maintaining compatibility with
    Megatron's build_train_valid_test_data_iterators().
    """

    @abstractmethod
    def create_dataloaders(
        self, trainer_config: Any, train_val_test_num_samples: List[int], vp_stage: Optional[int] = None
    ) -> Tuple[Any, Any, Any]:
        """
        Create train/valid/test dataloaders.

        Args:
            trainer_config: Megatron args namespace (from megatron.training.get_args())
            train_val_test_num_samples: [train_samples, valid_samples, test_samples]
            vp_stage: Virtual pipeline stage (for VP parallelism)

        Returns:
            Tuple of (train_dataloader, valid_dataloaders, test_dataloader)
            - train_dataloader: Training data iterator
            - valid_dataloaders: List of validation data iterators (or None)
            - test_dataloader: Test data iterator (or None)
        """

    @property
    @abstractmethod
    def is_distributed(self) -> bool:
        """
        Whether dataloaders are distributed across ranks.

        This flag tells Megatron whether to bypass indexed dataset logic:
            False: GPTDataset (uses BlendedMegatronDatasetBuilder, rank-specific slicing)
            True: Energon (handles distribution internally via WorkerConfig)

        Returns:
            bool: True if dataloaders handle distribution internally
        """


__all__ = ["DatasetProvider"]
