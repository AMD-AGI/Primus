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
        bridge_compat_inline_bos: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.formatter_name = formatter
        self.formatter = create_formatter(formatter)
        self.bridge_compat_inline_bos = bool(bridge_compat_inline_bos)

        is_local_file = (
            dataset_name.endswith(".jsonl") or dataset_name.endswith(".json") or os.path.isfile(dataset_name)
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
            bridge_compat_inline_bos=self.bridge_compat_inline_bos,
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
    enable_packed_sequences: bool = False,
    bridge_compat_inline_bos: bool = False,
    **kwargs,
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    """Build train/validation/test datasets using Megatron-local SFT helpers.

    When ``enable_packed_sequences`` is True, returns ``PackedSFTDataset``
    instances that concatenate multiple short samples into single
    ``max_seq_length`` sequences with per-segment ``cu_seqlens``.

    ``bridge_compat_inline_bos`` is an opt-in tokenize-time switch that
    reproduces NeMo Megatron-Bridge's packed-parquet layout (per-segment
    tokenize -> inline BOS in front of each supervised segment -> trailing
    EOS). It is forwarded to ``PackedSFTDataset`` / ``SFTDataset`` so it
    does not accidentally leak into HuggingFace ``load_dataset(**kwargs)``.

    Special-case dispatch -- mlperf packed npy directory
    ----------------------------------------------------
    If ``dataset_name`` points to a directory containing the mlperf-style
    pre-tokenised + pre-packed artefacts (``train.npy`` /
    ``validation.npy`` / ``packed_metadata.jsonl``), we short-circuit the
    HF / jsonl tokenize+pack pipeline and route to
    ``MlperfPackedDataset``. This lets a Native SFT run consume the exact
    byte-identical packs produced by the upstream mlperf
    ``download_dataset.py + convert_dataset.py + create_metadata.py``
    pipeline (used by ``examples/megatron_bridge/configs/MI355X/
    ``llama2_70b_lora_mlperf_posttrain.yaml``).
    """
    from primus.backends.megatron.sft.mlperf_packed_dataset import (
        build_mlperf_packed_datasets,
        is_mlperf_packed_dir,
    )

    if is_mlperf_packed_dir(dataset_name):
        log_rank_0(
            f"[SFT] sft_dataset_name={dataset_name} resolved to an mlperf "
            f"packed directory; bypassing HF/jsonl tokenize+pack and "
            f"routing to MlperfPackedDataset."
        )
        return build_mlperf_packed_datasets(
            data_dir=dataset_name,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            train_val_test_num_samples=train_val_test_num_samples,
        )

    train_samples, valid_samples, test_samples = train_val_test_num_samples

    if enable_packed_sequences:
        from primus.backends.megatron.sft.packing import PackedSFTDataset

        DatasetCls = PackedSFTDataset
    else:
        DatasetCls = SFTDataset

    train_ds = None
    valid_ds = None
    test_ds = None

    if train_samples > 0:
        train_ds = DatasetCls(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            split="train",
            formatter=formatter,
            seed=seed,
            bridge_compat_inline_bos=bridge_compat_inline_bos,
            **kwargs,
        )

    if valid_samples > 0:
        try:
            valid_ds = DatasetCls(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                split="validation",
                formatter=formatter,
                seed=seed,
                bridge_compat_inline_bos=bridge_compat_inline_bos,
                **kwargs,
            )
        except (ValueError, KeyError) as exc:
            log_rank_0(f"Validation split not available: {exc}")
            valid_ds = None

    if test_samples > 0:
        try:
            test_ds = DatasetCls(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                split="test",
                formatter=formatter,
                seed=seed,
                bridge_compat_inline_bos=bridge_compat_inline_bos,
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
