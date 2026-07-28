# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
SyntheticDatasetProvider: Dataset provider for synthetic/mock data.

Provides dataloaders for synthetic data generation during development,
testing, and benchmarking without requiring real datasets.
"""

import logging
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

from primus.backends.megatron.data.dataloader import MegatronDataloaderWrapper
from primus.backends.megatron.data.dataset_provider import DatasetProvider
from primus.core.utils.module_utils import log_rank_0

logger = logging.getLogger(__name__)


class SyntheticDatasetProvider(DatasetProvider):
    """
    Dataset provider for synthetic/mock data.

    This provider creates synthetic datasets based on configuration.
    The dataset class and parameters are specified in YAML config,
    keeping dataset creation logic separate from trainer logic.

    Usage in YAML:
        modules:
          pre_trainer:
            mock_data: true
            mock_dataset:
              class: "primus.backends.megatron.data.synthetic.PreGeneratedMockFluxDataset"
              params:
                num_samples: 1000
                image_size: 512
    """

    DEFAULT_DATASETS = {
        "flux": "primus.backends.megatron.data.synthetic.PreGeneratedMockFluxDataset",
        "flux_onthefly": "primus.backends.megatron.data.synthetic.MockFluxDataset",
        "flux_schnell": "primus.backends.megatron.data.synthetic.PreGeneratedMockFluxSchnellDataset",
        "flux_schnell_onthefly": "primus.backends.megatron.data.synthetic.MockFluxSchnellDataset",
    }

    def __init__(self, dataset_config: Optional[Dict[str, Any]] = None, model_type: str = "flux"):
        """
        Initialize synthetic dataset provider.

        Args:
            dataset_config: Dictionary with 'class' and 'params' keys:
                {
                    'class': 'fully.qualified.ClassName',  # Optional, uses default for model_type
                    'params': {  # Optional, dataset-specific parameters
                        'num_samples': 1000,
                        'image_size': 512,
                        # ... other dataset params
                    }
                }
            model_type: Model type for default dataset selection ('flux', etc.)
                Used when dataset_config['class'] is not specified.
        """
        self.dataset_config = dataset_config or {}
        self.model_type = model_type

        # Determine dataset class to use
        dataset_class_from_config = self.dataset_config.get("class")

        # If class is None or not specified, use default for model_type
        if dataset_class_from_config is None or dataset_class_from_config == "null":
            self.dataset_class_path = self.DEFAULT_DATASETS.get(model_type)
        else:
            self.dataset_class_path = dataset_class_from_config

        if not self.dataset_class_path:
            raise ValueError(
                f"No default dataset for model_type='{model_type}'. "
                f"Available types: {list(self.DEFAULT_DATASETS.keys())}. "
                f"Or specify dataset_config['class'] explicitly."
            )

        # Get dataset parameters (will be augmented with trainer config later)
        self.dataset_params = self.dataset_config.get("params", {})

        logger.debug(f"SyntheticDatasetProvider initialized: {self.dataset_class_path}")

    def _import_dataset_class(self):
        """
        Dynamically import the dataset class.

        Returns:
            Dataset class

        Raises:
            ImportError: If class cannot be imported
            AttributeError: If class doesn't exist in module
        """
        try:
            module_path, class_name = self.dataset_class_path.rsplit(".", 1)
            module = import_module(module_path)
            dataset_class = getattr(module, class_name)
            return dataset_class
        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import dataset class '{self.dataset_class_path}': {e}\n"
                f"Ensure the class path is correct and the module is installed."
            ) from e

    def create_dataloaders(
        self, trainer_config: Any, train_val_test_num_samples: List[int], vp_stage: Optional[int] = None
    ) -> Tuple[Any, Any, Any]:
        """
        Create synthetic dataloaders.

        Args:
            trainer_config: Megatron args namespace (from megatron.training.get_args())
            train_val_test_num_samples: [train_samples, valid_samples, test_samples]
            vp_stage: Virtual pipeline stage (for VP parallelism)

        Returns:
            Tuple of (train_dataloader, None, None)
            Note: Validation/test loaders not supported for synthetic data
        """
        from megatron.training import get_args
        from torch.utils.data import DataLoader

        args = get_args()

        log_rank_0("=" * 80)
        log_rank_0("Creating SYNTHETIC/MOCK dataloaders")
        log_rank_0(f"Dataset class: {self.dataset_class_path}")
        log_rank_0("=" * 80)

        # Import dataset class
        dataset_class = self._import_dataset_class()

        # Merge config: YAML params + runtime params from trainer
        final_params = {
            **self.dataset_params,  # From YAML
            "seed": getattr(trainer_config, "seed", 42),  # Always use trainer seed
        }

        # Override with trainer config if specified (backwards compatibility)
        if hasattr(trainer_config, "image_size") and "image_size" not in self.dataset_params:
            final_params["image_size"] = trainer_config.image_size

        # Forward vae_latent_mode from trainer config to mock dataset
        if hasattr(trainer_config, "vae_latent_mode") and "vae_latent_mode" not in self.dataset_params:
            final_params["vae_latent_mode"] = trainer_config.vae_latent_mode

        # Set defaults if not specified
        final_params.setdefault("num_samples", 1000)
        final_params.setdefault("image_size", 512)

        log_rank_0(f"Dataset parameters:")
        for key, value in sorted(final_params.items()):
            log_rank_0(f"  {key}: {value}")

        # Create dataset
        dataset = dataset_class(**final_params)

        log_rank_0(f"Created mock dataset: {dataset_class.__name__}")
        log_rank_0(f"  num_samples: {len(dataset)}")
        log_rank_0(f"  batch_size: {args.micro_batch_size}")

        # Create PyTorch DataLoader
        # Note: num_workers=0 for synthetic data (generation is fast)
        train_loader = DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        # Wrap in MegatronDataloaderWrapper for:
        #   1. Cyclic iteration (never exhausts)
        #   2. Megatron training loop compatibility
        #   3. Checkpoint interface (no-op for synthetic data)
        train_loader = MegatronDataloaderWrapper(train_loader)

        log_rank_0("✓ Synthetic dataloader ready (infinite iteration)")
        log_rank_0("=" * 80)

        # Create validation dataloader when eval_iters > 0
        val_loader = None
        eval_iters = getattr(args, "eval_iters", 0)
        if eval_iters > 0:
            val_num_samples = max(256, eval_iters * args.micro_batch_size)
            val_params = {**final_params, "is_validation": True, "num_samples": val_num_samples}
            val_dataset = dataset_class(**val_params)
            val_raw_loader = DataLoader(
                val_dataset,
                batch_size=args.micro_batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=True,
            )
            val_loader = MegatronDataloaderWrapper(val_raw_loader)
            log_rank_0(f"Created validation dataloader: {val_num_samples} samples, is_validation=True")

        return train_loader, val_loader, None

    @property
    def is_distributed(self) -> bool:
        """
        Synthetic data doesn't need distributed handling.

        Returns True to tell Megatron to bypass indexed dataset logic.
        Each rank creates its own synthetic data independently.
        """
        return True


__all__ = ["SyntheticDatasetProvider"]
