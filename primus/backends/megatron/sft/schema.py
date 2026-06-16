###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron-native SFT task schema and normalized sample structures."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple


def _coerce_text(value: Any) -> str:
    """Convert a possibly-missing field to a stable text value."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_optional_text(value: Any) -> Optional[str]:
    """Preserve missing optional fields while normalizing present values."""
    if value is None:
        return None
    return _coerce_text(value)


@dataclass(frozen=True)
class CharSpan:
    """Character span inside a formatted training sample."""

    start: int
    end: int


@dataclass(frozen=True)
class TextSegment:
    """A text fragment plus whether it should contribute to loss."""

    text: str
    supervise: bool = False


@dataclass(frozen=True)
class SFTMessage:
    """Normalized multi-turn message."""

    role: str
    content: str

    @classmethod
    def from_mapping(cls, message: Mapping[str, Any]) -> "SFTMessage":
        """Create a normalized message from untyped input data."""
        return cls(
            role=_coerce_text(message.get("role")),
            content=_coerce_text(message.get("content")),
        )


@dataclass(frozen=True)
class SFTSample:
    """Megatron SFT sample representation before formatting/tokenization."""

    instruction: str = ""
    response: str = ""
    input_text: Optional[str] = None
    system_prompt: Optional[str] = None
    messages: Tuple[SFTMessage, ...] = ()

    @property
    def is_multi_turn(self) -> bool:
        """Whether this sample uses a message list instead of a single turn."""
        return bool(self.messages)

    @classmethod
    def from_mapping(cls, sample: Mapping[str, Any]) -> "SFTSample":
        """Normalize a raw record from JSON/HF dataset into SFT semantics.

        Supports the following dataset schemas:
          * Alpaca (tatsu-lab/alpaca): instruction / input / output
          * OpenAI messages: messages = [{role, content}, ...]
          * SQuAD (rajpurkar/squad): question / context / answers={text:[...]}
              question -> instruction
              context  -> input_text
              answers.text[0] -> response
          * Generic: prompt/response, question/answer
        """
        raw_messages = sample.get("messages")
        if isinstance(raw_messages, Sequence) and not isinstance(raw_messages, (str, bytes)):
            messages = tuple(
                SFTMessage.from_mapping(message) for message in raw_messages if isinstance(message, Mapping)
            )
            return cls(messages=messages)

        instruction = sample.get("instruction")
        if instruction is None:
            instruction = sample.get("prompt")
        if instruction is None:
            instruction = sample.get("question", "")

        response = sample.get("response")
        if response is None:
            response = sample.get("output")
        if response is None:
            response = sample.get("answer")
        if response is None:
            # SQuAD-style answers field: {"text": [...], "answer_start": [...]}
            answers = sample.get("answers")
            if isinstance(answers, Mapping):
                answer_texts = answers.get("text")
                if isinstance(answer_texts, Sequence) and not isinstance(answer_texts, (str, bytes)):
                    response = answer_texts[0] if len(answer_texts) > 0 else ""
            elif isinstance(answers, Sequence) and not isinstance(answers, (str, bytes)):
                response = answers[0] if len(answers) > 0 else ""

        # SQuAD has `context` instead of `input`; fall back to it when `input`
        # is absent so the alpaca formatter renders context as the ``### Input``
        # block.
        input_text = sample.get("input")
        if input_text is None:
            input_text = sample.get("context")

        return cls(
            instruction=_coerce_text(instruction),
            response=_coerce_text(response),
            input_text=_coerce_optional_text(input_text),
            system_prompt=_coerce_optional_text(sample.get("system")),
        )


@dataclass(frozen=True)
class FormattedSFTSample:
    """Formatter output with explicit supervision boundaries."""

    segments: Tuple[TextSegment, ...]

    @property
    def text(self) -> str:
        """Full training text passed to the tokenizer/model."""
        return "".join(segment.text for segment in self.segments)

    @property
    def supervised_char_spans(self) -> Tuple[CharSpan, ...]:
        """Character spans that should contribute to SFT loss."""
        spans = []
        cursor = 0
        for segment in self.segments:
            end = cursor + len(segment.text)
            if segment.supervise and segment.text:
                spans.append(CharSpan(start=cursor, end=end))
            cursor = end
        return tuple(spans)

    @property
    def first_supervised_offset(self) -> int:
        """Compatibility helper for legacy single-turn callers."""
        spans = self.supervised_char_spans
        if spans:
            return spans[0].start
        return len(self.text)


def collapse_messages_to_single_turn(sample: SFTSample) -> SFTSample:
    """Best-effort collapse of multi-turn data for single-turn formatters."""
    if not sample.is_multi_turn:
        return sample

    instruction = ""
    response = ""
    system_prompt = None

    for message in sample.messages:
        if not message.role or not message.content:
            continue
        if message.role == "system":
            system_prompt = message.content
        elif message.role == "user":
            instruction = message.content
        elif message.role == "assistant":
            response = message.content

    return SFTSample(
        instruction=instruction,
        response=response,
        system_prompt=system_prompt,
    )


__all__ = [
    "CharSpan",
    "FormattedSFTSample",
    "SFTMessage",
    "SFTSample",
    "TextSegment",
    "collapse_messages_to_single_turn",
]
