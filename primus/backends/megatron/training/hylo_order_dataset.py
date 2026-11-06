###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Replay the exact training batches of a reference HyLo (TRL) Enhanced-ILD run.

The reference run records, per data-parallel rank, the ``input_ids`` of every training step
(``hylo_train_dump.py`` in the parity tools): ``<dir>/batches_rank<r>.bin`` holds int32
``(steps, micro_batch, seq_len)`` and ``<dir>/meta.json`` the shape and world size. This dataset
serves them so that Megatron's sequential sampler (``dataloader_type: single``), which gives rank
``r`` the samples ``[t * G + r * M, t * G + (r + 1) * M)`` of step ``t`` (``G = M * DP``), hands
each rank the batch the reference rank of the same index trained on at the same step.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def unset(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in ("", "null", "none"))


class HyLoOrderDataset(Dataset):
    def __init__(self, dump_dir: str, seq_length: int, micro_batch_size: int, data_parallel_size: int):
        meta = json.load(open(os.path.join(dump_dir, "meta.json")))
        self.mbs, self.seq, self.dp = int(meta["micro_batch"]), int(meta["seq_len"]), int(meta["world_size"])
        if (self.mbs, self.seq, self.dp) != (micro_batch_size, seq_length, data_parallel_size):
            raise ValueError(
                f"{dump_dir} holds micro-batches of {self.mbs} x {self.seq} tokens from {self.dp} ranks; "
                f"this run has micro_batch_size={micro_batch_size}, seq_length={seq_length}, "
                f"data-parallel size {data_parallel_size}"
            )
        self.ranks = []
        for r in range(self.dp):
            path = os.path.join(dump_dir, f"batches_rank{r}.bin")
            flat = np.memmap(path, dtype=np.int32, mode="r")
            if flat.size % (self.mbs * self.seq):
                raise ValueError(f"{path} is not a whole number of {self.mbs} x {self.seq} micro-batches")
            self.ranks.append(flat.reshape(-1, self.mbs, self.seq))
        self.steps = min(len(b) for b in self.ranks)
        self.position_ids = torch.arange(self.seq, dtype=torch.long)
        loss_mask = torch.ones(self.seq, dtype=torch.float32)
        loss_mask[-1] = 0.0
        self.loss_mask = loss_mask

    def __len__(self):
        return self.steps * self.mbs * self.dp

    def __getitem__(self, idx):
        if idx is None:
            idx = 0
        step, within = divmod(int(idx), self.mbs * self.dp)
        rank, j = divmod(within, self.mbs)
        tokens = torch.from_numpy(np.array(self.ranks[rank][step, j], dtype=np.int64))
        labels = torch.roll(tokens, shifts=-1, dims=0)
        labels[-1] = 0
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": self.loss_mask.clone(),
            "position_ids": self.position_ids,
        }
