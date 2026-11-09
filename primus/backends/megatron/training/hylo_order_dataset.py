###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Replay the exact training batches of a reference HyLo (TRL) run.

The reference run records, per data-parallel rank, the ``input_ids`` of every training step
(``tools/hybrid/hylo_train_dump.py``): ``<dir>/batches_rank<r>.bin`` holds int32
``(steps, micro_batch, seq_len)`` and ``<dir>/meta.json`` the shape, world size and step count. This
dataset serves them so that Megatron's sequential sampler (``dataloader_type: single``), which gives
rank ``r`` the samples ``[t * G + r * M, t * G + (r + 1) * M)`` of step ``t`` (``G = M * DP``), hands
each rank the batch the reference rank of the same index trained on at the same step.

Every position is supervised, as in HyLo's packed KL (FLA's ``kl_div_kernel`` over all of the
window's logits). Enhanced-ILD ignores the mask on packed windows.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from primus.backends.megatron.training.hylo_utils import (  # noqa: F401  (re-exported)
    unset,
)


class HyLoOrderDataset(Dataset):
    def __init__(self, dump_dir: str, seq_length: int, micro_batch_size: int, data_parallel_size: int):
        with open(os.path.join(dump_dir, "meta.json")) as f:
            meta = json.load(f)
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
        lengths = sorted({len(b) for b in self.ranks})
        if lengths != [int(meta["steps"])]:
            raise ValueError(
                f"{dump_dir}: meta.json records {meta['steps']} steps, the rank files hold {lengths}"
            )
        self.steps = lengths[0]
        self.position_ids = torch.arange(self.seq, dtype=torch.long)

    def __len__(self):
        return self.steps * self.mbs * self.dp

    def __getitem__(self, idx):
        # Megatron pads a short last batch with None; like GPTDataset, it contributes no loss.
        padding = idx is None
        step, within = divmod(0 if padding else int(idx), self.mbs * self.dp)
        rank, j = divmod(within, self.mbs)
        tokens = torch.from_numpy(np.array(self.ranks[rank][step, j], dtype=np.int64))
        labels = torch.roll(tokens, shifts=-1, dims=0)
        labels[-1] = 0
        loss_mask = torch.zeros(self.seq) if padding else torch.ones(self.seq)
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": self.position_ids,
        }
