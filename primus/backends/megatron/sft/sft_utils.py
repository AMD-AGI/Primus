###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
SFT (Supervised Fine-Tuning) utilities for native Megatron-LM.

This module provides SFT-specific components adapted from Megatron-LM's
examples/post_training/modelopt/finetune.py for use with Primus.

Components:
    - SFTDataset: Dataset class for SFT data (JSON/JSONL/HuggingFace)
    - train_valid_test_sft_datasets_provider: Dataset provider for pretrain()
    - get_batch: Batch generation with answer-only loss masking
    - forward_step: Forward step for SFT training
"""

from __future__ import annotations

import itertools
import json
import os
from functools import partial
from typing import Any, Dict, List, Optional

import torch

from primus.modules.module_utils import log_rank_0


def get_eos_id():
    """
    Return the EOS token ID for the current tokenizer.

    We insert eos_token between two samples during packing. However, if the
    eos_token is used in message or after turns, we need to replace it with
    some other special tokens that do not appear in message.

    Returns:
        int: EOS token ID
    """
    from megatron.training import get_tokenizer

    tokenizer = get_tokenizer()
    hf_tokenizer = tokenizer._tokenizer

    # Handle special cases for different model families
    if hf_tokenizer.eos_token == "<|eot_id|>":
        return 128001  # Llama 3.x
    if hf_tokenizer.eos_token == "<|eot|>":
        return 200001  # Some custom models
    if hf_tokenizer.eos_token == "<|im_end|>":
        return 151643  # Qwen/ChatML
    if hf_tokenizer.eos_token == "<|return|>":
        return 199999  # Some custom models

    return hf_tokenizer.eos_token_id


class SFTDataset(torch.utils.data.Dataset):
    """
    Dataset class for Supervised Fine-Tuning.

    Supports:
        - JSON/JSONL files with conversation data
        - HuggingFace datasets
        - Chat template application
        - Sample packing for efficient training
        - Answer-only loss masking

    The raw data is processed and packed to an indexed dataset on the fly.
    Users specify the total number of packed samples and the dataloader
    accesses the packed dataset by indices.
    """

    # Mapping of HuggingFace dataset names to load kwargs
    HF_DATASET_KWARGS = {
        "Open-Orca/OpenOrca": {"split": "train"},
        "Open-Orca/SlimOrca": {"split": "train"},
        "nvidia/Daring-Anteater": {"split": "train"},
        "Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered": {"split": "train"},
        "HuggingFaceH4/ultrachat_200k": {"split": "train_sft"},
    }

    # Mapping of HuggingFace dataset names to conversation transformers
    HF_DATASET_CONVERTERS = {
        "Open-Orca/OpenOrca": lambda data: SFTDataset._to_conversation(
            data["question"], data["response"]
        ),
        "Open-Orca/SlimOrca": lambda data: SFTDataset._sharegpt_to_openai(data),
        "nvidia/Daring-Anteater": lambda data: SFTDataset._sharegpt_to_openai(data),
        "Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered": lambda data: SFTDataset._sharegpt_to_openai(
            data
        ),
    }

    def __init__(
        self,
        num_packed_samples: int,
        data_path: Optional[str],
        tokenizer: Any,
        seq_length: int,
        hf_dataset: Optional[str] = None,
        num_shards: int = 1,
        shard_index: int = 0,
    ):
        """
        Initialize SFT dataset.

        Args:
            num_packed_samples: Total number of packed samples (cyclic access)
            data_path: Path to JSON/JSONL file
            tokenizer: HuggingFace tokenizer instance
            seq_length: Maximum sequence length
            hf_dataset: HuggingFace dataset name (alternative to data_path)
            num_shards: Number of data shards for distributed training
            shard_index: Index of current shard
        """
        import transformers

        if not isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
            raise ValueError("SFTDataset only supports transformers.PreTrainedTokenizerBase!")

        self.num_packed_samples = num_packed_samples
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.hf_dataset = hf_dataset
        self.data_transformation = lambda data: data
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.indexed_dataset: List[Dict[str, Any]] = []
        self._raw_sample_index = 0

        # Load raw samples from file or HuggingFace
        self._raw_samples = self._load_raw_samples()

        # Setup data transformation based on source
        self._setup_transformation()

    def _load_raw_samples(self) -> Any:
        """Load raw samples from file or HuggingFace dataset."""
        if self.data_path is not None:
            return self._load_from_file()
        elif self.hf_dataset is not None:
            return self._load_from_hf()
        else:
            raise ValueError("Either hf_dataset or data_path must be provided!")

    def _load_from_file(self) -> List[Dict[str, Any]]:
        """Load samples from JSON/JSONL file."""
        if self.data_path.endswith(".json"):
            with open(self.data_path) as f:
                return json.load(f)
        elif self.data_path.endswith(".jsonl"):
            import jsonlines

            with jsonlines.open(self.data_path, mode="r") as reader:
                return [obj for obj in reader]
        else:
            raise ValueError("data_path must be .json or .jsonl file")

    def _load_from_hf(self) -> Any:
        """Load samples from HuggingFace dataset."""
        import datasets

        hf_kwargs = self.HF_DATASET_KWARGS.get(self.hf_dataset, {"split": "train"})
        raw_samples = datasets.load_dataset(self.hf_dataset, **hf_kwargs)
        raw_samples = raw_samples.shard(num_shards=self.num_shards, index=self.shard_index)

        log_rank_0(
            f"Rank {torch.distributed.get_rank()}/{torch.distributed.get_world_size()} "
            f"creates SFT shard {self.shard_index}/{self.num_shards} "
            f"with {len(raw_samples)} raw samples"
        )

        return raw_samples

    def _setup_transformation(self):
        """Setup data transformation based on source."""
        if self.tokenizer.chat_template is None:
            # Use default prompt template if no chat template
            self.tokenizer.chat_template = "{{ messages['question'] + ' ' + messages['response'] + ' ' }}"
        elif self.hf_dataset is not None:
            # Use dataset-specific converter
            self.data_transformation = self.HF_DATASET_CONVERTERS.get(
                self.hf_dataset, lambda data: data
            )

        if self.tokenizer.chat_template is None:
            raise ValueError("No valid chat template!")

    def __len__(self):
        return self.num_packed_samples

    def __getitem__(self, idx):
        """
        Get packed sample at index.

        The packed data index is different from the raw data index where a
        packed sample of sequence-length may require concatenating multiple
        raw samples. When all raw data are used up, the last packed data is
        thrown away, and we have a packed dataset in memory.
        """
        idx = idx // self.num_shards

        while idx >= len(self.indexed_dataset):
            packed_samples = self._process_and_pack_example()
            if packed_samples is None:
                break
            else:
                self.indexed_dataset.append(packed_samples)

            if len(self.indexed_dataset) % 10000 == 0:
                log_rank_0(
                    f"Rank {torch.distributed.get_rank()}/{torch.distributed.get_world_size()} "
                    f"packed {len(self.indexed_dataset)} SFT samples"
                )

        idx = idx % len(self.indexed_dataset)
        torch_sample = {}
        for key, val in self.indexed_dataset[idx].items():
            if key != "token_count":
                torch_sample[key] = torch.LongTensor(val)
        return torch_sample

    def _process_and_pack_example(self) -> Optional[Dict[str, Any]]:
        """Process multiple raw samples and pack them into fixed sequence length."""
        required_packed_tokens = self.seq_length + 1
        current_packed_samples = []
        current_packed_samples_token_count = 0

        while current_packed_samples_token_count < required_packed_tokens:
            if self._raw_sample_index >= len(self._raw_samples):
                return None

            raw_sample = self._raw_samples[self._raw_sample_index]
            self._raw_sample_index += 1
            processed_sample = self._process_example(raw_sample)

            if processed_sample is not None:
                current_packed_samples.append(processed_sample)
                current_packed_samples_token_count += processed_sample["token_count"]

        packed_samples = {}

        for key in ["input_ids", "loss_mask"]:
            packed_samples[key] = list(
                itertools.chain.from_iterable([obj[key] for obj in current_packed_samples])
            )

        for key in ["token_count"]:
            packed_samples[key] = [obj[key] for obj in current_packed_samples]

        return packed_samples

    def _process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply chat template and compute answer-only loss mask.

        Args:
            example: Raw sample dictionary

        Returns:
            Processed sample with input_ids, loss_mask, and token_count
        """
        if not isinstance(example, Dict):
            raise ValueError(f"Sample must be a Dict but got {type(example)}")

        # Apply data transformation (e.g., ShareGPT to OpenAI format)
        example = self.data_transformation(example)

        # Check if this is OpenAI chat format
        conversations = example.get("conversations", None)
        if conversations is None:
            conversations = example.get("messages", None)

        # Skip if no assistant reply or conversation starts with assistant
        if conversations is not None:
            example = conversations
            if len(conversations) < 2 or example[0]["role"] == "assistant":
                return None

        # Apply chat template
        input_ids = self.tokenizer.apply_chat_template(example)
        current_loss_mask = [1] * len(input_ids)

        # Add EOS token for sample boundary
        input_ids = input_ids + [get_eos_id()]
        current_loss_mask += [0]

        assert len(input_ids) == len(current_loss_mask)

        # Truncate if needed
        if len(input_ids) > self.seq_length:
            input_ids = input_ids[: self.seq_length]
            current_loss_mask = current_loss_mask[: self.seq_length]

        return {
            "input_ids": input_ids,
            "loss_mask": current_loss_mask,
            "token_count": len(input_ids),
        }

    @classmethod
    def _to_conversation(cls, question: str, response: str) -> Dict[str, Any]:
        """Convert question/response pair to OpenAI conversation format."""
        return {
            "conversations": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]
        }

    @classmethod
    def _sharegpt_to_openai(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ShareGPT format to OpenAI conversation format."""
        role_mapping = {
            "user": "user",
            "User": "user",
            "human": "user",
            "assistant": "assistant",
            "Assistant": "assistant",
            "gpt": "assistant",
            "system": "system",
            "System": "system",
        }
        processed_data = {"conversations": []}
        for msg in data["conversations"]:
            role = role_mapping.get(msg["from"], msg["from"])
            content = msg["value"]
            processed_data["conversations"].append({"role": role, "content": content})
        return processed_data


def train_valid_test_sft_datasets_provider(train_val_test_num_samples):
    """
    Build train, validation, and test SFT datasets.

    This function is designed to be passed to Megatron's pretrain() as the
    dataset provider.

    Args:
        train_val_test_num_samples: List of [train_samples, valid_samples, test_samples]

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    import transformers

    from megatron.core import mpu
    from megatron.training import get_args, get_tokenizer
    from megatron.training.utils import print_rank_0

    print_rank_0("> building train, validation, and test SFT datasets ...")

    args = get_args()
    tokenizer = get_tokenizer()

    if not isinstance(tokenizer._tokenizer, transformers.PreTrainedTokenizerBase):
        raise ValueError("SFTDataset only supports transformers.PreTrainedTokenizerBase!")

    if args.micro_batch_size > 1:
        raise ValueError("SFTDataloader only supports micro_batch_size=1.")

    # Common kwargs for all datasets
    kwargs = {
        "tokenizer": tokenizer._tokenizer,
        "seq_length": args.seq_length,
        "hf_dataset": getattr(args, "finetune_hf_dataset", None),
        "num_shards": mpu.get_data_parallel_world_size(),
        "shard_index": mpu.get_data_parallel_rank(),
    }

    # Get data paths
    train_path = args.train_data_path[0] if getattr(args, "train_data_path", None) else None
    valid_path = args.valid_data_path[0] if getattr(args, "valid_data_path", None) else None
    test_path = args.test_data_path[0] if getattr(args, "test_data_path", None) else None

    # Create datasets
    train_ds = SFTDataset(train_val_test_num_samples[0], train_path, **kwargs)
    valid_ds = SFTDataset(train_val_test_num_samples[1], valid_path, **kwargs)
    test_ds = SFTDataset(train_val_test_num_samples[2], test_path, **kwargs)

    print_rank_0("> finished creating SFT datasets ...")

    return train_ds, valid_ds, test_ds


def get_batch(data_iterator):
    """
    Generate a batch for SFT training.

    This function handles:
        - Batch broadcasting across tensor parallel ranks
        - Loss mask generation for answer-only training
        - Position IDs and attention mask generation

    Args:
        data_iterator: Data iterator from dataloader

    Returns:
        Dictionary with tokens, labels, loss_mask, attention_mask, position_ids
    """
    from megatron.core import mpu, tensor_parallel
    from megatron.training import get_args
    from megatron.training.utils import get_batch_on_this_cp_rank, get_ltor_masks_and_position_ids

    # Skip for intermediate pipeline stages
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None

    args = get_args()

    # Broadcast data since only TP rank-0 has the data_iterator
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    keys = ["input_ids", "loss_mask"]
    datatype = torch.int64
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack the data
    tokens_ = data_b["input_ids"]
    tokens = tokens_[:, 0 : 0 + args.seq_length].contiguous()
    labels = tokens_[:, 1 : 1 + args.seq_length].contiguous()
    answer_only_loss_mask = data_b["loss_mask"][:, 1 : 1 + args.seq_length].contiguous()

    # Get masks and position IDs
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        get_eos_id(),
        get_eos_id(),
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        False,
    )

    # Apply answer-only loss mask
    loss_mask = loss_mask * answer_only_loss_mask.to(dtype=loss_mask.dtype)

    labels = labels.contiguous()
    loss_mask = loss_mask.contiguous()

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    # Slice batch for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch


def forward_step(data_iterator, model):
    """
    Forward training step for SFT.

    This function:
        - Gets batch from data_iterator
        - Runs forward pass through model
        - Returns output tensor and loss function

    Args:
        data_iterator: Input data iterator
        model: The GPT Model

    Returns:
        Tuple of (output_tensor, loss_func_partial)
    """
    from megatron.training import get_timers

    timers = get_timers()

    # Get the batch
    timers("batch-generator", log_level=2).start()
    batch = get_batch(data_iterator)
    tokens = batch["tokens"]
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    attention_mask = batch["attention_mask"]
    position_ids = batch["position_ids"]
    timers("batch-generator").stop()

    # Forward pass
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def loss_func(loss_mask, output_tensor):
    """
    Compute loss for SFT training.

    Args:
        loss_mask: Mask for answer-only loss
        output_tensor: Output from model forward pass

    Returns:
        Tuple of (loss, loss_dict)
    """
    from megatron.training.utils import average_losses_across_data_parallel_group

    # Get losses from output tensor
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    # Compute masked loss
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Average across data parallel group
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {"lm loss": averaged_loss[0]}
