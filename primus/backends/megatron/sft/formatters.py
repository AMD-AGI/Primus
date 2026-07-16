###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron-local SFT conversation formatters and strategy selection."""

from typing import Dict, Mapping, Optional, Sequence, Tuple

from primus.backends.megatron.sft.schema import (
    FormattedSFTSample,
    SFTMessage,
    SFTSample,
    TextSegment,
    collapse_messages_to_single_turn,
)


def _normalize_messages(messages: Sequence[object]) -> Tuple[SFTMessage, ...]:
    """Normalize untyped message payloads to SFTMessage objects."""
    normalized = []
    for message in messages:
        if isinstance(message, SFTMessage):
            normalized.append(message)
        elif isinstance(message, Mapping):
            normalized.append(SFTMessage.from_mapping(message))
    return tuple(normalized)


class ConversationFormatter:
    """Base formatter that maps normalized SFT samples to formatted text."""

    supports_messages = False

    def format_sample(self, sample: SFTSample) -> FormattedSFTSample:
        """Format a normalized sample into supervision-aware segments."""
        raise NotImplementedError

    def format_conversation(
        self,
        instruction: str,
        response: str,
        input_text: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> tuple[str, int]:
        """Backward-compatible helper used by existing dataset tests/callers."""
        formatted = self.format_sample(
            SFTSample(
                instruction=instruction,
                response=response,
                input_text=input_text,
                system_prompt=system_prompt,
            )
        )
        return formatted.text, formatted.first_supervised_offset

    def get_special_tokens(self) -> Dict[str, str]:
        """Return special tokens used in this format."""
        return {}


class SquadFormatter(ConversationFormatter):
    """SQuAD-style single-line prompt/response format.

    This formatter is byte-identical to what Megatron-Bridge's
    ``process_squad_example`` (combined with the default
    ``prompt_template="{input} {output}"`` of ``GPTSFTDataset``) emits, so
    Native SFT can be A/B-benchmarked against Bridge under the *same*
    training text on the SQuAD dataset.

    Bridge-side reference:
      * ``megatron/bridge/data/hf_processors/squad.py``::
            _input  = f"Context: {context} Question: {question} Answer:"
            _output = answers["text"][0]
      * ``megatron/bridge/data/datasets/sft.py``::
            prompt_template = "{input} {output}"

    Final text (with all three fields ``.strip(" ")``-ed, matching Bridge):
        ``Context: <ctx> Question: <q> Answer: <ans>``

    Notes:
      * The single space between ``Answer:`` and the answer is held at the
        START of the supervised segment, not at the END of the prompt
        segment. This mirrors Bridge's
        ``prompt_template = "{input} {output}"`` where ``_input`` ends with
        ``Answer:`` (no trailing space) and the space is part of the join
        between input and output. With this layout, after per-segment
        tokenize the leading space token of the answer becomes the first
        token of segment 1 (rather than the last token of segment 0), which
        is what makes Native's supervised range start exactly at the same
        token Bridge supervises (the colon-prefixed answer span).
      * ``system_prompt`` and the ``messages`` multi-turn channel are not
        supported -- SQuAD is single-turn QA. If a sample carries either,
        we fall back to using the collapsed instruction/response pair.
    """

    def format_sample(self, sample: SFTSample) -> FormattedSFTSample:
        sample = collapse_messages_to_single_turn(sample)
        ctx = (sample.input_text or "").strip(" ")
        q = (sample.instruction or "").strip(" ")
        ans = (sample.response or "").strip(" ")
        return FormattedSFTSample(
            segments=(
                TextSegment(text=f"Context: {ctx} Question: {q} Answer:"),
                TextSegment(text=f" {ans}", supervise=True),
            )
        )


class AlpacaFormatter(ConversationFormatter):
    """Alpaca-style prompt/response format."""

    def format_sample(self, sample: SFTSample) -> FormattedSFTSample:
        sample = collapse_messages_to_single_turn(sample)

        prompt_template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
        )
        if sample.system_prompt:
            prompt_template = sample.system_prompt + "\n\n"

        instruction_part = f"### Instruction:\n{sample.instruction}\n\n"
        if sample.input_text:
            instruction_part += f"### Input:\n{sample.input_text}\n\n"
        instruction_part += "### Response:\n"

        return FormattedSFTSample(
            segments=(
                TextSegment(text=prompt_template + instruction_part),
                TextSegment(text=sample.response, supervise=True),
            )
        )


class ChatMLFormatter(ConversationFormatter):
    """ChatML format with assistant-only supervision."""

    def format_sample(self, sample: SFTSample) -> FormattedSFTSample:
        sample = collapse_messages_to_single_turn(sample)

        segments = []
        if sample.system_prompt:
            segments.append(TextSegment(text=f"<|im_start|>system\n{sample.system_prompt}<|im_end|>\n"))

        user_content = sample.instruction
        if sample.input_text:
            user_content = f"{sample.instruction}\n\n{sample.input_text}"

        segments.extend(
            (
                TextSegment(text=f"<|im_start|>user\n{user_content}<|im_end|>\n"),
                TextSegment(text="<|im_start|>assistant\n"),
                TextSegment(text=sample.response, supervise=True),
                TextSegment(text="<|im_end|>"),
            )
        )
        return FormattedSFTSample(segments=tuple(segments))

    def get_special_tokens(self) -> Dict[str, str]:
        """Return ChatML special tokens."""
        return {
            "im_start": "<|im_start|>",
            "im_end": "<|im_end|>",
        }


class OpenAIMessagesFormatter(ConversationFormatter):
    """OpenAI messages / multi-turn chat format."""

    supports_messages = True

    def _messages_from_sample(self, sample: SFTSample) -> Tuple[SFTMessage, ...]:
        if sample.messages:
            return sample.messages

        messages = []
        if sample.system_prompt:
            messages.append(SFTMessage(role="system", content=sample.system_prompt))

        user_content = sample.instruction
        if sample.input_text:
            user_content = f"{sample.instruction}\n\n{sample.input_text}"
        messages.append(SFTMessage(role="user", content=user_content))
        messages.append(SFTMessage(role="assistant", content=sample.response))
        return tuple(messages)

    def format_messages(self, messages: Sequence[object]) -> tuple[str, list[tuple[int, int]]]:
        """Backward-compatible helper returning text plus assistant char ranges."""
        formatted = self.format_sample(SFTSample(messages=_normalize_messages(messages)))
        return formatted.text, [(span.start, span.end) for span in formatted.supervised_char_spans]

    def format_conversation(
        self,
        instruction: str = "",
        response: str = "",
        input_text: Optional[str] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[Sequence[object]] = None,
    ) -> tuple[str, int]:
        """Compatibility wrapper that preserves legacy messages behavior."""
        if messages is not None:
            formatted = self.format_sample(SFTSample(messages=_normalize_messages(messages)))
            return formatted.text, 0
        return super().format_conversation(
            instruction=instruction,
            response=response,
            input_text=input_text,
            system_prompt=system_prompt,
        )

    def format_sample(self, sample: SFTSample) -> FormattedSFTSample:
        segments = []
        for message in self._messages_from_sample(sample):
            if not message.role or not message.content:
                continue
            segments.extend(
                (
                    TextSegment(text=f"<|im_start|>{message.role}\n"),
                    TextSegment(text=message.content, supervise=message.role == "assistant"),
                    TextSegment(text="<|im_end|>\n"),
                )
            )
        return FormattedSFTSample(segments=tuple(segments))

    def get_special_tokens(self) -> Dict[str, str]:
        """Return special tokens used in this format."""
        return {
            "im_start": "<|im_start|>",
            "im_end": "<|im_end|>",
        }


def create_formatter(name: str) -> ConversationFormatter:
    """Select the formatter implementation used by Megatron SFT."""
    if name == "alpaca":
        return AlpacaFormatter()
    if name == "chatml":
        return ChatMLFormatter()
    if name in {"openai", "messages"}:
        return OpenAIMessagesFormatter()
    if name in {"squad", "bridge_squad"}:
        return SquadFormatter()
    raise ValueError(f"Unknown formatter: {name}. Supported: alpaca, chatml, openai, messages, squad")


__all__ = [
    "AlpacaFormatter",
    "ChatMLFormatter",
    "ConversationFormatter",
    "OpenAIMessagesFormatter",
    "SquadFormatter",
    "create_formatter",
]
