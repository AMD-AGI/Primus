###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron-native SFT dataset and dataset-builder entrypoints."""

import os
from typing import Optional, Tuple

from torch.utils.data import Dataset

from primus.backends.megatron.sft.formatters import create_formatter
from primus.backends.megatron.sft.preprocessing import (
    load_local_records,
    log_rank_0,
    normalize_sft_sample,
    tokenize_formatted_sft_sample,
)


class SFTDataset(Dataset):
    """Megatron-local SFT dataset with formatter-driven supervision."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        max_seq_length: int,
        split: str = "train",
        formatter: str = "alpaca",
        seed: int = 1234,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.formatter_name = formatter
        self.formatter = create_formatter(formatter)

        is_local_file = (
            dataset_name.endswith(".jsonl")
            or dataset_name.endswith(".json")
            or os.path.isfile(dataset_name)
        )

        if is_local_file:
            log_rank_0(f"Loading dataset from local file: {dataset_name}")
            data = load_local_records(dataset_name)
            try:
                from datasets import Dataset as HFDataset
            except ImportError as exc:
                raise ImportError(
                    "HuggingFace datasets library is required. Install with: pip install datasets"
                ) from exc

            self.dataset = HFDataset.from_list(data)
            log_rank_0(f"Created dataset with {len(self.dataset)} samples")
        else:
            log_rank_0(f"Loading dataset from HuggingFace Hub: {dataset_name}, split: {split}")
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    "HuggingFace datasets library is required. Install with: pip install datasets"
                ) from exc

            self.dataset = load_dataset(dataset_name, split=split, **kwargs)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """Return the normalized/tokenized SFT training sample."""
        sample = normalize_sft_sample(self.dataset[idx])
        formatted_sample = self.formatter.format_sample(sample)
        input_ids, labels, loss_mask = tokenize_formatted_sft_sample(
            formatted_sample=formatted_sample,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
        }


def build_train_valid_test_datasets(
    dataset_name: str,
    tokenizer,
    max_seq_length: int,
    train_val_test_num_samples: list[int],
    formatter: str = "alpaca",
    seed: int = 1234,
    **kwargs,
) -> Tuple[Optional[SFTDataset], Optional[SFTDataset], Optional[SFTDataset]]:
    """Build train/validation/test datasets using Megatron-local SFT helpers."""
    train_samples, valid_samples, test_samples = train_val_test_num_samples

    train_ds = None
    valid_ds = None
    test_ds = None

    if train_samples > 0:
        train_ds = SFTDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            split="train",
            formatter=formatter,
            seed=seed,
            **kwargs,
        )

    if valid_samples > 0:
        try:
            valid_ds = SFTDataset(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                split="validation",
                formatter=formatter,
                seed=seed,
                **kwargs,
            )
        except (ValueError, KeyError) as exc:
            log_rank_0(f"Validation split not available: {exc}")
            valid_ds = None

    if test_samples > 0:
        try:
            test_ds = SFTDataset(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                split="test",
                formatter=formatter,
                seed=seed,
                **kwargs,
            )
        except (ValueError, KeyError) as exc:
            log_rank_0(f"Test split not available: {exc}")
            test_ds = None

    return train_ds, valid_ds, test_ds


__all__ = [
    "SFTDataset",
    "build_train_valid_test_datasets",
]
