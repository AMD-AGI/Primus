###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron-local SFT preprocessing helpers for samples and token masks."""

import json
import os
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch

from primus.backends.megatron.sft.schema import FormattedSFTSample, SFTSample

try:
    from primus.modules.module_utils import log_rank_0 as _primus_log_rank_0
except ImportError:
    _primus_log_rank_0 = None


def log_rank_0(msg):
    """Log through Primus when available, otherwise fall back to stdout."""
    if _primus_log_rank_0 is None:
        print(msg)
        return
    try:
        _primus_log_rank_0(msg)
    except AttributeError:
        # Unit tests may import dataset helpers without initializing the global logger.
        print(msg)


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_num} in {file_path}: {exc.msg}",
                    exc.doc,
                    exc.pos,
                ) from exc

    log_rank_0(f"Loaded {len(data)} samples from {file_path}")
    return data


def load_local_records(file_path: str) -> List[Dict[str, Any]]:
    """Load local JSON/JSONL records for Megatron offline SFT."""
    if file_path.endswith(".jsonl"):
        return load_jsonl_file(file_path)

    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"JSON file must contain a list of objects, got {type(data)}")
        log_rank_0(f"Loaded {len(data)} samples from {file_path}")
        return data

    try:
        return load_jsonl_file(file_path)
    except json.JSONDecodeError:
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("File must contain a list of objects or be JSONL format")
        log_rank_0(f"Loaded {len(data)} samples from {file_path}")
        return data


def normalize_sft_sample(sample: Mapping[str, Any]) -> SFTSample:
    """Normalize raw dataset rows into the Megatron SFT schema."""
    return SFTSample.from_mapping(sample)


def tokenize_text(tokenizer, text: str) -> List[int]:
    """Tokenize text while tolerating the tokenizer variants used in Megatron."""
    try:
        result = tokenizer.tokenize(text)
        if isinstance(result, (list, tuple)):
            if not result:
                return []
            if isinstance(result[0], int):
                return list(result)
            if hasattr(tokenizer, "convert_tokens_to_ids"):
                return tokenizer.convert_tokens_to_ids(result)
        if hasattr(tokenizer, "encode"):
            return tokenizer.encode(text, add_special_tokens=False)
        raise AttributeError("Tokenizer missing required methods")
    except (AttributeError, TypeError, IndexError) as exc:
        if hasattr(tokenizer, "encode"):
            return tokenizer.encode(text, add_special_tokens=False)
        raise TypeError(
            "Tokenizer must provide either encode() or tokenize() "
            f"compatible with Megatron SFT. Got tokenizer type: {type(tokenizer)}. Error: {exc}"
        ) from exc


def tokenize_formatted_sft_sample(
    formatted_sample: FormattedSFTSample,
    tokenizer,
    max_seq_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize a formatted sample and build its SFT loss mask."""
    text = formatted_sample.text
    token_ids = tokenize_text(tokenizer, text)
    if len(token_ids) > max_seq_length:
        token_ids = token_ids[:max_seq_length]

    loss_mask = np.zeros(len(token_ids), dtype=np.int64)
    prefix_text = ""
    prefix_token_count = 0
    for segment in formatted_sample.segments:
        start = prefix_token_count
        prefix_text += segment.text
        prefix_token_count = len(tokenize_text(tokenizer, prefix_text))
        end = prefix_token_count

        if segment.supervise and start < len(token_ids):
            loss_mask[start:min(end, len(token_ids))] = 1
        if start >= len(token_ids):
            break

    input_ids = torch.tensor(token_ids, dtype=torch.int64)
    labels = input_ids.clone()
    loss_mask_tensor = torch.tensor(loss_mask, dtype=torch.int64)
    return input_ids, labels, loss_mask_tensor


__all__ = [
    "load_jsonl_file",
    "load_local_records",
    "log_rank_0",
    "normalize_sft_sample",
    "tokenize_formatted_sft_sample",
    "tokenize_text",
]
