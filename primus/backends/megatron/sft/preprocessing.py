###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron-local SFT preprocessing helpers for samples and token masks."""

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

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


def _resolve_pad_token_id(tokenizer) -> int:
    """Best-effort lookup of a pad token id across Megatron/HF tokenizer variants."""
    for attr in ("pad", "pad_token_id", "eod", "eos_token_id", "eos_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and value >= 0:
            return value
    return 0


def _resolve_bos_token_id(tokenizer) -> Optional[int]:
    """Best-effort lookup of a BOS token id across Megatron/HF tokenizer variants.

    Returns ``None`` if the tokenizer has no BOS concept (e.g. GPT-2 style); the
    caller is expected to treat that as "do not insert any inline BOS".
    """
    for attr in ("bos", "bos_token_id", "bos_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _resolve_eos_token_id(tokenizer) -> Optional[int]:
    """Best-effort lookup of an EOS token id across Megatron/HF tokenizer variants."""
    for attr in ("eos", "eos_token_id", "eos_id", "eod"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def tokenize_formatted_sft_sample(
    formatted_sample: FormattedSFTSample,
    tokenizer,
    max_seq_length: int,
    bridge_compat_inline_bos: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize a formatted sample and build its SFT loss mask.

    Returns input_ids/labels/loss_mask all padded to ``max_seq_length`` so that
    PyTorch's default_collate (used by Megatron's pretrain dataloader) can stack
    samples whose original token lengths differ across a micro-batch.
    Padding tokens are masked out via ``loss_mask=0`` so they do not contribute
    to the SFT loss.

    See ``packing._tokenize_no_pad`` for the meaning of
    ``bridge_compat_inline_bos``; this function honours the same flag so the
    non-packed code path stays consistent with the packed path.
    """
    if not bridge_compat_inline_bos:
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
                loss_mask[start : min(end, len(token_ids))] = 1
            if start >= len(token_ids):
                break
    else:
        eos_id = _resolve_eos_token_id(tokenizer)

        ids_list: List[int] = []
        mask_list: List[int] = []
        # Per-segment tokenize. Megatron's ``TextTokenizer.tokenize`` calls
        # ``HuggingFaceTokenizer.text_to_ids`` which already adds a BOS at the
        # start of each standalone segment (``include_special_tokens=True``
        # default). Do NOT manually prepend ``bos_id`` here -- doubling up
        # would emit ``<s><s>`` and inflate iter-1 loss ~2 nats above Bridge.
        for segment in formatted_sample.segments:
            seg_ids = list(tokenize_text(tokenizer, segment.text))
            mask_val = 1 if segment.supervise else 0
            ids_list.extend(seg_ids)
            mask_list.extend([mask_val] * len(seg_ids))

        if eos_id is not None and formatted_sample.segments:
            tail_supervised = bool(formatted_sample.segments[-1].supervise)
            ids_list.append(eos_id)
            mask_list.append(1 if tail_supervised else 0)

        if len(ids_list) > max_seq_length:
            ids_list = ids_list[:max_seq_length]
            mask_list = mask_list[:max_seq_length]
        token_ids = ids_list
        loss_mask = np.asarray(mask_list, dtype=np.int64)

    seq_len = len(token_ids)
    if seq_len < max_seq_length:
        pad_len = max_seq_length - seq_len
        pad_id = _resolve_pad_token_id(tokenizer)
        token_ids = list(token_ids) + [pad_id] * pad_len
        loss_mask = np.concatenate([loss_mask, np.zeros(pad_len, dtype=np.int64)])

    input_ids = torch.tensor(token_ids, dtype=torch.int64)
    # Next-token prediction: ``labels[i] = input_ids[i+1]``. Megatron's
    # ``compute_language_model_loss(labels, logits)`` does NOT shift labels
    # internally, so the dataset MUST emit shifted labels. The last position
    # has no in-sequence next-token target and is masked out below.
    # Bridge does the equivalent shift in its collate_fn
    # (see ``megatron/bridge/data/datasets/sft.py:1206``).
    labels = input_ids.clone()
    if labels.numel() >= 2:
        labels[:-1] = input_ids[1:]
    loss_mask_tensor = torch.tensor(loss_mask, dtype=torch.int64)
    # The very last position has no next-token target; mask it out so the
    # bogus ``labels[-1]`` slot never contributes to the loss. This is
    # belt-and-suspenders -- under the typical SFT prompt template the last
    # token is already eos/pad with loss_mask=0, but enforcing it here makes
    # the contract explicit.
    if loss_mask_tensor.numel() >= 1:
        loss_mask_tensor[-1] = 0
    return input_ids, labels, loss_mask_tensor


__all__ = [
    "load_jsonl_file",
    "load_local_records",
    "log_rank_0",
    "normalize_sft_sample",
    "tokenize_formatted_sft_sample",
    "tokenize_text",
]
