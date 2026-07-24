"""
FLA-order GPT dataset for Megatron.

Loads FLA's preprocessed HuggingFace dataset and serves samples in the exact
same order that FLA's training pipeline (HuggingFace Trainer + DistributedSampler)
would present them.  This lets Primus/Megatron reproduce FLA's training trajectory
bit-for-bit (modulo bf16 numerics), proving that any remaining loss gap comes
solely from data ordering.

Usage:
    Set PRIMUS_FLA_DATA=1 and PRIMUS_FLA_CACHE_DIR=<path_to_fla_preprocessed_data>
    before launching training.

    Example:
        PRIMUS_FLA_DATA=1 \\
        PRIMUS_FLA_CACHE_DIR=/home/vanbhati@amd.com/flash-linear-attention/legacy/training/data/HuggingFaceFW/fineweb-edu/sample-10BT/train \\
        EXP=examples/megatron/configs/MI300X/zebra_llama_300M_gdn_pure-pretrain.yaml \\
        GPUS_PER_NODE=8 bash examples/run_pretrain.sh
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class FLAOrderGPTDataset(Dataset):
    """
    Wraps a preprocessed HuggingFace dataset (2048-token sequences) and serves
    samples in the exact order that FLA's DistributedSampler would.

    Megatron's MegatronPretrainingSampler assigns *contiguous blocks* of
    micro_batch_size to each rank:
        rank r at step t gets global indices [t*G + r*M, ..., t*G + r*M + M-1]
        where G = micro_batch_size * data_parallel_size, M = micro_batch_size.

    FLA's DistributedSampler assigns *strided* indices from a shuffled permutation:
        rank r at step t gets shuffled_dataset[perm[r + (t*M + j)*D]] for j in 0..M-1
        where D = data_parallel_size, perm = randperm(N, seed=42).

    This dataset pre-builds an index_map so that:
        __getitem__(megatron_gidx) returns the same data FLA would serve
        for the same (rank, step, position-in-batch) tuple.
    """

    def __init__(
        self,
        cache_dir: str,
        seq_length: int,
        micro_batch_size: int,
        data_parallel_size: int,
        seed: int = 42,
        pad_token_id: int = 0,
        eod_token: int = 128000,
        eod_mask_loss: bool = False,
    ):
        from datasets import load_from_disk

        raw_dataset = load_from_disk(cache_dir)
        self.dataset = raw_dataset.shuffle(seed=seed)
        self.N = len(self.dataset)
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id
        self.eod_token = eod_token
        self.eod_mask_loss = eod_mask_loss

        mbs = micro_batch_size
        dp = data_parallel_size
        gb = mbs * dp

        # Replicate PyTorch DistributedSampler's permutation
        g = torch.Generator()
        g.manual_seed(seed)  # seed + epoch(0)
        total_size = ((self.N + dp - 1) // dp) * dp
        perm = torch.randperm(self.N, generator=g).tolist()
        padding = total_size - self.N
        perm = perm + perm[:padding]

        # Build megatron_gidx → fla_dataset_idx mapping
        self.index_map = np.zeros(total_size, dtype=np.int64)
        for gidx in range(total_size):
            t = gidx // gb
            r = (gidx % gb) // mbs
            j = gidx % mbs
            fla_perm_idx = r + (t * mbs + j) * dp
            if fla_perm_idx < len(perm):
                self.index_map[gidx] = perm[fla_perm_idx] % self.N
            else:
                self.index_map[gidx] = 0

        self._cached_loss_mask = None
        self._cached_position_ids = None

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if idx is None:
            idx = 0

        fla_idx = int(self.index_map[idx % len(self.index_map)])
        input_ids = self.dataset[fla_idx]["input_ids"]

        tokens = torch.tensor(input_ids[: self.seq_length], dtype=torch.long)
        labels = torch.roll(tokens, shifts=-1, dims=0)
        labels[-1] = self.pad_token_id

        if self._cached_loss_mask is None:
            loss_mask = torch.ones(self.seq_length, dtype=torch.float32)
            loss_mask[-1] = 0.0
            if self.eod_mask_loss:
                eod_positions = tokens == self.eod_token
                loss_mask[eod_positions] = 0.0
            self._cached_loss_mask = loss_mask
            self._cached_position_ids = torch.arange(
                self.seq_length, dtype=torch.long
            )

        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": self._cached_loss_mask.clone(),
            "position_ids": self._cached_position_ids,
        }
