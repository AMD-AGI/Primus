###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

import collections
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class VisionCollator:
    processor: Any

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.processor.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.processor.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]
        inputs = collections.defaultdict(list)
        for instance in instances:
            for key, values in instance.items():
                inputs[key].append(values)

        batched_inputs = {}
        if "input_ids" in inputs:
            input_ids = inputs.pop("input_ids")
            input_ids = self.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.processor.tokenizer.pad_token_id,
            )
            batched_inputs["input_ids"] = input_ids
        if "labels" in inputs:
            labels = inputs.pop("labels")
            labels = self.pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100,
            )
            batched_inputs["labels"] = labels

        if "attention_mask" in inputs:
            inputs.pop("attention_mask")

        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id).long()
        batched_inputs["attention_mask"] = attention_mask

        # for the other keys
        for key, values in inputs.items():
            # Handle scalar/boolean values ( use_audio_in_video)
            if isinstance(values[0], bool) or (
                isinstance(values[0], (int, float)) and not isinstance(values[0], torch.Tensor)
            ):
                batched_inputs[key] = values[0]
            else:
                batched_inputs[key] = torch.stack(values, dim=0)
        return batched_inputs

    @property
    def image_token_id(self):
        return self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)


@dataclass
class RawBatchCollator:
    """
    A minimal collator that returns raw samples as a list of dicts.

    This is useful when model-specific padding/encoding happens in a separate
    batch preparation step (e.g. processor.prepare_batch / trainer.prepare_batch),
    keeping the DataLoader and trainer model-agnostic.
    """

    def __call__(self, instances: Sequence[dict]) -> list[dict]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]
        return list(instances)
