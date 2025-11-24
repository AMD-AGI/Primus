###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# """
# Dataset provider for Megatron training.

# Provides model-specific dataset providers compatible with Megatron's training loop.
# """

# from types import SimpleNamespace


# def get_train_valid_test_datasets_provider(args: SimpleNamespace):
#     """
#     Get the dataset provider function based on model configuration.

#     Args:
#         args: Megatron argument namespace

#     Returns:
#         Dataset provider function that returns (train_ds, valid_ds, test_ds)
#     """
#     # Get model type
#     model_type = getattr(args, "model_type", "GPT").upper()

#     if "GPT" in model_type or "DECODER" in model_type:
#         return _get_gpt_datasets_provider(args)
#     elif "BERT" in model_type or "ENCODER" in model_type:
#         return _get_bert_datasets_provider(args)
#     elif "T5" in model_type:
#         return _get_t5_datasets_provider(args)
#     else:
#         # Default to GPT-style datasets
#         return _get_gpt_datasets_provider(args)


# def _get_gpt_datasets_provider(args: SimpleNamespace):
#     """Get dataset provider for GPT-style decoder models."""
#     try:
#         # Try to import from pretrain_gpt
#         from pretrain_gpt import train_valid_test_datasets_provider  # type: ignore

#         return train_valid_test_datasets_provider
#     except ImportError:
#         # Fallback: create a basic dataset provider
#         return _create_default_datasets_provider(args)


# def _get_bert_datasets_provider(args: SimpleNamespace):
#     """Get dataset provider for BERT-style encoder models."""
#     try:
#         from pretrain_bert import train_valid_test_datasets_provider  # type: ignore

#         return train_valid_test_datasets_provider
#     except ImportError:
#         return _create_default_datasets_provider(args)


# def _get_t5_datasets_provider(args: SimpleNamespace):
#     """Get dataset provider for T5-style encoder-decoder models."""
#     try:
#         from pretrain_t5 import train_valid_test_datasets_provider  # type: ignore

#         return train_valid_test_datasets_provider
#     except ImportError:
#         return _create_default_datasets_provider(args)


# def _create_default_datasets_provider(args: SimpleNamespace):
#     """
#     Create a default dataset provider function.

#     This is a fallback that uses Megatron's core dataset utilities.
#     """

#     def default_provider(train_val_test_num_samples):
#         """
#         Default dataset provider that uses data paths from args.

#         Args:
#             train_val_test_num_samples: Tuple of (train, valid, test) sample counts

#         Returns:
#             Tuple of (train_dataset, valid_dataset, test_dataset)
#         """
#         from megatron.training import get_args  # type: ignore
#         from megatron.training.utils import print_rank_0  # type: ignore

#         args = get_args()

#         print_rank_0(f"[Primus:DataProvider] Building datasets with paths: {args.data_path}")

#         # Import dataset building utilities
#         try:
#             # Try GPT dataset builder first
#             from megatron.core.datasets.blended_megatron_dataset_builder import (  # type: ignore
#                 BlendedMegatronDatasetBuilder,
#             )
#             from megatron.core.datasets.gpt_dataset import (
#                 GPTDatasetConfig,  # type: ignore
#             )

#             # Configure dataset
#             config = GPTDatasetConfig(
#                 is_built_on_rank=lambda: True,
#                 random_seed=args.seed,
#                 sequence_length=args.seq_length,
#                 blend=args.data_path,
#                 split=args.split,
#             )

#             # Build datasets
#             train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
#                 GPTDatasetConfig, train_val_test_num_samples, lambda: True, config
#             ).build()

#             return train_ds, valid_ds, test_ds

#         except ImportError:
#             # Fallback to legacy dataset builder
#             from megatron.training import (
#                 build_train_valid_test_datasets,  # type: ignore
#             )

#             return build_train_valid_test_datasets(train_val_test_num_samples)

#     return default_provider
