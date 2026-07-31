###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Hugging Face chat-template tokenization for Megatron-native SFT."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from primus.backends.megatron.sft.preprocessing import _resolve_pad_token_id
from primus.backends.megatron.sft.schema import SFTSample

GENERATION_REGEX = re.compile(r"{%\s*if\s+add_generation_prompt\s*%}")

# Tokens reserved for chat template markup, special tokens, and the assistant turn.
_TEMPLATE_AND_ASSISTANT_TOKEN_RESERVE = 512


def gov_report_sample_to_messages(sample: SFTSample) -> List[Dict[str, str]]:
    """Build user/assistant messages for SCROLLS GovReport-style rows.

    Supports:
      * QA rows (MLPerf / SQuAD-style): ``context``, ``question``, ``answers``
      * SCROLLS summarization: ``input`` (report) + ``output`` (summary)
      * Alpaca-style fallbacks already normalized on ``SFTSample``
    """
    context = (sample.input_text or "").strip()
    question = (sample.instruction or "").strip()
    answer = (sample.response or "").strip()

    if context and question:
        user_content = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    elif context:
        user_content = (
            "Summarize the following government report.\n\n" f"{context}"
        )
    elif question:
        user_content = question
    else:
        user_content = "Answer the following question."

    messages: List[Dict[str, str]] = []
    if sample.system_prompt:
        messages.append({"role": "system", "content": sample.system_prompt})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": answer})
    return messages


def _cap_user_message_chars(
    messages: List[Dict[str, str]], max_seq_length: int
) -> List[Dict[str, str]]:
    """Truncate long user turns *before* ``apply_chat_template`` (GovReport-scale docs)."""
    msgs = [dict(m) for m in messages]
    reserve = min(_TEMPLATE_AND_ASSISTANT_TOKEN_RESERVE, max(128, max_seq_length // 16))
    user_token_budget = max(256, max_seq_length - reserve)
    char_cap = user_token_budget * 3
    for msg in msgs:
        if msg.get("role") == "user" and len(msg.get("content", "")) > char_cap:
            msg["content"] = msg["content"][:char_cap]
    return msgs


def _is_transformers_pretrained_tokenizer(tok) -> bool:
    mod = getattr(type(tok), "__module__", "") or ""
    return mod.startswith("transformers.")


def _unwrap_hf_auto_tokenizer(tokenizer):
    """Return the underlying ``transformers`` tokenizer for chat templates."""
    tok = tokenizer
    for _ in range(6):
        if _is_transformers_pretrained_tokenizer(tok):
            return tok
        inner = getattr(tok, "_tokenizer", None)
        if inner is not None and inner is not tok:
            if _is_transformers_pretrained_tokenizer(inner):
                return inner
            tok = inner
            continue
        inner = getattr(tok, "tokenizer", None)
        if inner is not None and inner is not tok:
            if _is_transformers_pretrained_tokenizer(inner):
                return inner
            if hasattr(inner, "apply_chat_template"):
                tok = inner
                continue
            break
        break
    return tok


def _resolve_chat_template(tokenizer, hf_tok) -> str | None:
    for src in (hf_tok, tokenizer, getattr(tokenizer, "_tokenizer", None)):
        if src is None:
            continue
        tpl = getattr(src, "chat_template", None)
        if isinstance(tpl, str) and tpl.strip():
            return tpl
    return None


def _apply_chat_template_via_megatron_text(
    messages: List[Dict[str, str]],
    tokenizer,
    chat_template: str | None,
    *,
    max_seq_length: int | None = None,
) -> Tuple[List[int], List[int]]:
    """Fallback when the Megatron text wrapper owns ``apply_chat_template``."""
    base_kwargs: Dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": False,
    }
    if max_seq_length is not None:
        base_kwargs["truncation"] = True
        base_kwargs["max_length"] = max_seq_length
    if chat_template is None:
        raise ValueError(
            "hf_chat formatter requires a chat_template on the tokenizer "
            f"(got {type(tokenizer)!r})."
        )

    tokenized = tokenizer.apply_chat_template(
        messages,
        chat_template,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **base_kwargs,
    )
    input_ids = tokenized.get("input_ids") or []
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    mask = tokenized.get("assistant_masks") or tokenized.get("assistant_mask")
    if mask is None:
        raise ValueError("Megatron apply_chat_template did not return assistant_masks.")
    if isinstance(mask, torch.Tensor):
        mask = mask.tolist()
    loss_mask = [int(x) for x in mask]
    if max_seq_length is not None:
        input_ids = input_ids[:max_seq_length]
        loss_mask = loss_mask[:max_seq_length]
    return list(input_ids), loss_mask


def _apply_chat_template(
    messages: List[Dict[str, str]],
    tokenizer,
    *,
    max_seq_length: int | None = None,
) -> Tuple[List[int], List[int]]:
    """Tokenize ``messages`` with the model tokenizer's chat template."""
    hf_tok = _unwrap_hf_auto_tokenizer(tokenizer)
    chat_template = _resolve_chat_template(tokenizer, hf_tok)

    if not _is_transformers_pretrained_tokenizer(hf_tok):
        if hasattr(tokenizer, "apply_chat_template") and hasattr(tokenizer, "_tokenizer"):
            return _apply_chat_template_via_megatron_text(
                messages, tokenizer, chat_template, max_seq_length=max_seq_length
            )
        raise ValueError(
            "hf_chat formatter requires a Hugging Face tokenizer with "
            "apply_chat_template (e.g. Qwen3-235B). "
            f"Got {type(tokenizer)!r} (unwrapped {type(hf_tok)!r})."
        )

    template_has_generation = (
        bool(chat_template) and GENERATION_REGEX.search(chat_template) is not None
    )

    base_kwargs: Dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": False,
    }
    if chat_template is not None:
        base_kwargs["chat_template"] = chat_template
    if max_seq_length is not None:
        base_kwargs["truncation"] = True
        base_kwargs["max_length"] = max_seq_length

    try:
        tokenized = hf_tok.apply_chat_template(
            messages,
            return_dict=True,
            return_assistant_tokens_mask=template_has_generation,
            **base_kwargs,
        )
        input_ids = tokenized.get("input_ids") or []
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
    except TypeError:
        tokenized = hf_tok.apply_chat_template(messages, **base_kwargs)
        if isinstance(tokenized, torch.Tensor):
            tokenized = tokenized.tolist()
        input_ids = tokenized
        tokenized = {"input_ids": input_ids}
        template_has_generation = False

    if max_seq_length is not None and len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]

    if template_has_generation and "assistant_masks" in tokenized:
        mask = tokenized["assistant_masks"]
        if isinstance(mask, torch.Tensor):
            mask = mask.tolist()
        loss_mask = [int(x) for x in mask]
        if max_seq_length is not None:
            loss_mask = loss_mask[:max_seq_length]
    else:
        prompt_kwargs = dict(base_kwargs)
        prompt_kwargs["add_generation_prompt"] = True
        prompt_ids = hf_tok.apply_chat_template(messages[:-1], **prompt_kwargs)
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.tolist()
        loss_mask = [0] * len(input_ids)
        prompt_len = min(len(prompt_ids), len(input_ids))
        for i in range(prompt_len, len(input_ids)):
            loss_mask[i] = 1

    if len(loss_mask) != len(input_ids):
        loss_mask = [1] * len(input_ids)

    return list(input_ids), loss_mask


def tokenize_chat_sft_sample_no_pad(
    sample: SFTSample,
    tokenizer,
    max_seq_length: int,
    *,
    task: str = "gov_report",
) -> Dict[str, np.ndarray]:
    """Tokenize one sample with HF chat template (no padding)."""
    if task == "gov_report":
        messages = gov_report_sample_to_messages(sample)
    else:
        raise ValueError(f"Unknown hf_chat task: {task}")

    messages = _cap_user_message_chars(messages, max_seq_length=max_seq_length)
    input_ids, loss_mask = _apply_chat_template(
        messages, tokenizer, max_seq_length=max_seq_length
    )

    if not any(loss_mask):
        return {
            "input_ids": np.asarray([], dtype=np.int64),
            "loss_mask": np.asarray([], dtype=np.int64),
            "length": 0,
        }

    return {
        "input_ids": np.asarray(input_ids, dtype=np.int64),
        "loss_mask": np.asarray(loss_mask, dtype=np.int64),
        "length": len(input_ids),
    }


def tokenize_chat_sft_sample(
    sample: SFTSample,
    tokenizer,
    max_seq_length: int,
    *,
    task: str = "gov_report",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize + pad to ``max_seq_length`` for non-packed ``SFTDataset``."""
    tok = tokenize_chat_sft_sample_no_pad(
        sample, tokenizer, max_seq_length, task=task
    )
    token_ids = tok["input_ids"].tolist()
    loss_mask = tok["loss_mask"]

    seq_len = len(token_ids)
    if seq_len < max_seq_length:
        pad_len = max_seq_length - seq_len
        pad_id = _resolve_pad_token_id(tokenizer)
        token_ids = token_ids + [pad_id] * pad_len
        loss_mask = np.concatenate([loss_mask, np.zeros(pad_len, dtype=np.int64)])

    input_ids = torch.tensor(token_ids, dtype=torch.int64)
    labels = input_ids.clone()
    if labels.numel() >= 2:
        labels[:-1] = input_ids[1:]
    loss_mask_tensor = torch.tensor(loss_mask, dtype=torch.int64)
    if loss_mask_tensor.numel() >= 1:
        loss_mask_tensor[-1] = 0
    return input_ids, labels, loss_mask_tensor


__all__ = [
    "gov_report_sample_to_messages",
    "tokenize_chat_sft_sample",
    "tokenize_chat_sft_sample_no_pad",
    "_cap_user_message_chars",
    "_unwrap_hf_auto_tokenizer",
]
