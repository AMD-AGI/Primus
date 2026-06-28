# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
EnergonDatasetProvider: Megatron Energon dataset provider.

Provides Energon-based dataloaders for multimodal and diffusion models,
using WebDataset format with distributed loading via WorkerConfig.

Reference:
    Megatron-LM examples/multimodal/dataloader_provider.py
"""

import logging
from typing import Any, Callable, List, Optional, Tuple

from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from megatron.energon import (
    LimitDataset,
    RepeatDataset,
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_datasets,
)

from primus.backends.megatron.data.dataloader import MegatronDataloaderWrapper
from primus.backends.megatron.data.dataset_provider import DatasetProvider
from primus.modules.module_utils import log_rank_0

logger = logging.getLogger(__name__)


class EnergonDatasetProvider(DatasetProvider):
    """
    Dataset provider for multimodal/diffusion models using Megatron Energon.

    Uses:
        - Megatron Energon (WebDataset format)
        - Task encoders (model-specific, injected via factory)
        - WorkerConfig for distributed data loading

    Architecture:
        - Task encoder factory is called during setup() to create encoder
        - WorkerConfig handles rank-based data sharding automatically
        - Dataloaders are only created on specific ranks (first TP, first/last PP)

    Reference:
        Megatron-LM examples/multimodal/dataloader_provider.py
    """

    def __init__(self, task_encoder_factory: Callable[[], Any]):
        """
        Initialize Energon dataset provider.

        Args:
            task_encoder_factory: Function that creates task encoder
                Signature: () -> TaskEncoder
                Example: lambda: EncodedDiffusionTaskEncoder(...)

                The factory pattern is used because task encoders may need
                access to trainer state (e.g., self.module_config) which
                isn't available during __init__.
        """
        self.task_encoder_factory = task_encoder_factory

    def create_dataloaders(
        self, trainer_config: Any, train_val_test_num_samples: List[int], vp_stage: Optional[int] = None
    ) -> Tuple[Any, Any, Any]:
        """
        Build train, validation, and test dataloaders using Energon.

        This closely follows the pattern from Megatron-LM multimodal examples,
        with adaptations for Primus configuration style.
        """
        from megatron.training import get_args

        args = get_args()

        # Check if we should create dataloaders on this rank
        # (Only first TP rank and first/last PP stage)
        if not self._is_dataloader_rank():
            log_rank_0(
                "Skipping dataloader creation on this rank (not first TP rank or not first/last PP stage)"
            )
            return None, None, None

        # Create task encoder using factory
        task_encoder = self.task_encoder_factory()
        log_rank_0(f"Created task encoder: {type(task_encoder).__name__}")

        # Create worker config for distributed loading
        worker_config = self._create_worker_config(args)

        # Get data path
        data_path = self._get_data_path(args)

        # Create training dataset using Energon
        log_rank_0(f"Creating training dataset from: {data_path}")
        train_dataset = get_train_dataset(
            data_path,
            batch_size=args.micro_batch_size,
            task_encoder=task_encoder,
            worker_config=worker_config,
            virtual_epoch_length=getattr(args, "virtual_epoch_length", 1_000_000_000),
            max_samples_per_sequence=getattr(args, "max_samples_per_sequence", 100),
            shuffle_buffer_size=getattr(args, "shuffle_buffer_size", None),
            handler=lambda *args: None,  # Error handler (print errors but continue)
        )

        # Wrap in savable loader for checkpointing support
        prefetch_factor = getattr(args, "prefetch_factor", 2)
        log_rank_0(f"Dataloader prefetch_factor: {prefetch_factor}")
        train_dataloader = get_savable_loader(
            train_dataset, worker_config=worker_config, prefetch_factor=prefetch_factor
        )
        train_dataloader = MegatronDataloaderWrapper(train_dataloader)
        log_rank_0("Created training dataloader")

        # Create validation dataloaders if evaluation is enabled
        valid_dataloaders = None
        if args.eval_iters > 0:
            try:
                log_rank_0("Creating validation dataloaders...")
                val_datasets = get_val_datasets(
                    data_path,
                    batch_size=args.micro_batch_size,
                    task_encoder=task_encoder,
                    worker_config=worker_config,
                    handler=lambda *args: None,
                )

                # Limit validation datasets to eval_iters * num_microbatches
                val_datasets_limited = [
                    LimitDataset(
                        RepeatDataset(val_ds, worker_config=worker_config),
                        length=args.eval_iters * get_num_microbatches(),
                        worker_config=worker_config,
                        reset_after_epoch=True,
                    )
                    for val_ds, _src_ds in val_datasets
                ]

                valid_dataloaders = [
                    MegatronDataloaderWrapper(
                        get_loader(valid_ds, worker_config=worker_config, prefetch_factor=prefetch_factor)
                    )
                    for valid_ds in val_datasets_limited
                ]
                log_rank_0(f"Created {len(valid_dataloaders)} validation dataloaders")
            except Exception as e:
                log_rank_0("=" * 80)
                log_rank_0("WARNING: Could not create validation dataloaders")
                log_rank_0(f"Reason: {e}")
                log_rank_0("")
                log_rank_0("This typically means:")
                log_rank_0("  - The dataset does not have a validation split")
                log_rank_0("  - The dataset path is incorrect")
                log_rank_0("")
                log_rank_0("AUTOMATIC FIX: Disabling evaluation (setting eval_iters = 0)")
                log_rank_0("To enable evaluation, provide a dataset with validation split")
                log_rank_0("=" * 80)
                valid_dataloaders = None

                # Automatically disable evaluation when no validation data exists
                args.eval_iters = 0

        # Test dataloaders not implemented for Energon
        test_dataloader = None

        return train_dataloader, valid_dataloaders, test_dataloader

    @property
    def is_distributed(self) -> bool:
        """
        Energon dataloaders are distributed (handle sharding internally).

        Returns True to tell Megatron to bypass indexed dataset logic.
        """
        return True

    def _is_dataloader_rank(self) -> bool:
        """
        Check if we should have the dataloader on this rank.

        Energon dataloaders should only run on:
            - First tensor parallel rank (data will be broadcast to others)
            - First or last pipeline parallel stage (where embeddings/outputs are)

        Reference:
            Megatron-LM examples/multimodal/dataloader_provider.py:is_dataloader_rank()
        """
        # Run dataloader only on first tensor parallel rank
        is_first_tp_rank = get_tensor_model_parallel_rank() == 0

        # Check pipeline parallel stage
        pp_size = get_pipeline_model_parallel_world_size()
        if pp_size == 1:
            # No pipeline parallelism
            is_valid_pp_stage = True
        else:
            # With pipeline parallelism, run on first and last stage
            pp_rank = get_pipeline_model_parallel_rank()
            is_valid_pp_stage = pp_rank in (0, pp_size - 1)

        return is_first_tp_rank and is_valid_pp_stage

    def _create_worker_config(self, args) -> WorkerConfig:
        """
        Create Energon WorkerConfig for distributed loading.

        WorkerConfig tells Energon how to shard data across workers.
        """
        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        data_parallel_group = parallel_state.get_data_parallel_group()

        return WorkerConfig(
            rank=rank,
            world_size=world_size,
            num_workers=getattr(args, "num_workers", 4),
            data_parallel_group=data_parallel_group,
        )

    def _get_data_path(self, args) -> str:
        """
        Extract data path from args.

        Handles both string and list formats.
        Energon's get_train_dataset() can accept either:
        - Directory path (will look for .nv-meta/dataset.yaml or dataset.yaml)
        - Direct path to dataset.yaml file

        This method returns the path as-is, letting Energon handle the resolution.
        """
        from pathlib import Path

        data_path = args.data_path
        if isinstance(data_path, list):
            data_path = data_path[0]

        if not data_path:
            raise ValueError("data_path not found in configuration")

        data_path = str(data_path)
        path_obj = Path(data_path)

        # Check if path exists
        if not path_obj.exists():
            raise ValueError(
                f"data_path does not exist: {data_path}\n"
                f"Please verify the path is correct and the dataset has been prepared."
            )

        # If it's a directory, check for dataset.yaml indicators
        if path_obj.is_dir():
            dataset_yaml = path_obj / "dataset.yaml"
            nv_meta_yaml = path_obj / ".nv-meta" / "dataset.yaml"

            if not dataset_yaml.exists() and not nv_meta_yaml.exists():
                log_rank_0("=" * 80)
                log_rank_0("WARNING: No dataset.yaml found in dataset directory")
                log_rank_0(f"  Directory: {data_path}")
                log_rank_0(f"  Expected: {dataset_yaml} or {nv_meta_yaml}")
                log_rank_0("")
                log_rank_0("This dataset may not be properly indexed with Energon.")
                log_rank_0("Run: energon prepare <dataset_dir> --num-workers 4")
                log_rank_0("=" * 80)
            elif nv_meta_yaml.exists():
                log_rank_0(f"Found Energon metadata at: {nv_meta_yaml}")
            elif dataset_yaml.exists():
                log_rank_0(f"Found dataset.yaml at: {dataset_yaml}")

        log_rank_0(f"Using data path: {data_path}")
        return data_path


__all__ = ["EnergonDatasetProvider"]
