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
    from primus.core.utils.module_utils import log_rank_0 as _primus_log_rank_0
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


# (tokenizer class, warning kind) pairs already logged. Keyed by class, not object: no tokenizer
# is held (nothing leaks in a long-lived process), and the new tokenizer of each dataset / split
# does not repeat a warning (the messages name the class, not the object).
_LOGGED_ONCE: set = set()
_MISSING = object()


def _log_once(tokenizer, kind: str, msg: str) -> None:
    """Log ``msg`` the first time ``kind`` is seen for this tokenizer class in this process: the
    resolvers run once per sample."""
    key = (type(tokenizer).__module__, type(tokenizer).__qualname__, kind)
    if key not in _LOGGED_ONCE:
        _LOGGED_ONCE.add(key)
        log_rank_0(msg)


def _special_token_name(attr: str) -> str:
    """The special token a token-id attribute reads: ``eos_id`` / ``eod`` -> ``eos_token``."""
    stem = "eos" if attr == "eod" else attr.split("_")[0]
    return f"{stem}_token"


def _token_string_is_none(tokenizer, attr: str) -> bool:
    """True when the tokenizer (or the library tokenizer it wraps) has the special-token string
    behind ``attr`` set to ``None``: the token does not exist."""
    name = _special_token_name(attr)
    for obj in (tokenizer, getattr(tokenizer, "_tokenizer", None)):
        if obj is not None and getattr(obj, name, _MISSING) is None:
            return True
    return False


def _token_id_attr(tokenizer, attr: str):
    """Read a token-id attribute, or ``None`` when the tokenizer does not have that token.

    The id is read first; a valid id is always used. Two failures of the read mean "absent":

    - ``TypeError`` while the special-token string behind ``attr`` (``eos_token`` for ``eos_id``
      and ``eod``) is ``None`` on the tokenizer or the library tokenizer it wraps: Megatron's
      ``HuggingFaceTokenizer.eos_id`` has no ``None`` guard (thinkingmachines/Inkling has no
      EOS). A ``TypeError`` while the token string exists is a bug and propagates.
    - ``NotImplementedError``: Megatron's legacy tokenizer base class, for a special-token
      property its subclass does not support.

    A missing attribute is the ``getattr`` default. Any other exception propagates. A skipped
    attribute is logged once per tokenizer class and attribute (``_LOGGED_ONCE``), not once per sample.
    """
    try:
        return getattr(tokenizer, attr, None)
    except NotImplementedError as exc:
        reason = f"NotImplementedError: {exc}"
    except TypeError:
        if not _token_string_is_none(tokenizer, attr):
            raise
        reason = f"its {_special_token_name(attr)} is None"
    _log_once(
        tokenizer,
        f"absent_{attr}",
        f"[SFT] tokenizer {type(tokenizer).__name__}.{attr} is unavailable ({reason}); treated as absent",
    )
    return None


def _tokenizer_pad_token_id(tokenizer) -> Optional[int]:
    """The tokenizer's own pad id (pad, else EOS / EOD), or None when it has neither."""
    for attr in ("pad", "pad_token_id", "eod", "eos_token_id", "eos_id"):
        value = _token_id_attr(tokenizer, attr)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _resolve_pad_token_id(tokenizer) -> int:
    """Best-effort lookup of a pad token id across Megatron/HF tokenizer variants; 0 (warned once)
    when the tokenizer has no pad or EOS id."""
    value = _tokenizer_pad_token_id(tokenizer)
    if value is not None:
        return value
    _log_once(
        tokenizer,
        "pad_fallback",
        f"[SFT] WARNING: tokenizer {type(tokenizer).__name__} has no pad or EOS token id; padding "
        "uses token id 0. Token 0 is never a loss target: the padding is loss-masked, and so is "
        "the last token of an unpacked sample (its label would be the first pad); attention "
        "never reads padding past a sample's end under the causal mask",
    )
    return 0


def _resolve_bos_token_id(tokenizer) -> Optional[int]:
    """Best-effort lookup of a BOS token id across Megatron/HF tokenizer variants.

    Returns ``None`` if the tokenizer has no BOS concept (e.g. GPT-2 style); the
    caller is expected to treat that as "do not insert any inline BOS".
    """
    for attr in ("bos", "bos_token_id", "bos_id"):
        value = _token_id_attr(tokenizer, attr)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _warn_no_inline_eos(tokenizer) -> None:
    """Warn (once per tokenizer class, ``_LOGGED_ONCE``) that bridge_compat_inline_bos appends no EOS."""
    _log_once(
        tokenizer,
        "no_eos",
        f"[SFT] WARNING: tokenizer {type(tokenizer).__name__} has no EOS token id, so "
        "bridge_compat_inline_bos appends no EOS to the samples and the model is not trained "
        "to stop. End each response with the stop token in the chat template or the data.",
    )


def _resolve_eos_token_id(tokenizer) -> Optional[int]:
    """Best-effort lookup of an EOS token id across Megatron/HF tokenizer variants."""
    for attr in ("eos", "eos_token_id", "eos_id", "eod"):
        value = _token_id_attr(tokenizer, attr)
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

        if eos_id is None and formatted_sample.segments:
            _warn_no_inline_eos(tokenizer)
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
        own_pad = _tokenizer_pad_token_id(tokenizer)
        # one lookup per sample; the resolver runs only for its warn-once fallback to 0
        pad_id = own_pad if own_pad is not None else _resolve_pad_token_id(tokenizer)
        if seq_len > 0 and own_pad is None:
            # Pad fallback (no pad and no EOS id): the last token's label after the shift below
            # is the first pad, token 0, a real vocab entry. Not a target, or the model learns to
            # emit token 0 after each response. With a real pad / EOS id that label is kept as
            # before (an EOS / EOD pad trains the model to stop).
            loss_mask = np.asarray(loss_mask, dtype=np.int64).copy()
            loss_mask[seq_len - 1] = 0
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
