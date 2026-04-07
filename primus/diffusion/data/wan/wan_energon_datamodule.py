# Copyright (c) 2025, Advanced Micro Devices, Inc.
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""WAN diffusion dataset config: synthetic video-latent batches (path unset) or Energon (path set)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional, Tuple

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from torch.utils.data import Dataset

from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider

# Per-patch channel count: out_channels * patch_temporal * patch_spatial**2 (WAN defaults)
_PATCH_DIM = 16 * 1 * 2 * 2


def _wan_collate(batch: List[dict]) -> dict:
    if len(batch) != 1:
        raise NotImplementedError(
            "Synthetic WAN dataset currently supports micro_batch_size=1 only "
            "(packed_seq_params are not merged for larger batches)."
        )
    return batch[0]


class _SyntheticWanDataset(Dataset):
    """Minimal map-style dataset for WAN forward smoke tests (micro_batch_size=1)."""

    collate_fn: Callable[[List[dict]], dict] = staticmethod(_wan_collate)

    def __init__(
        self,
        length: int,
        *,
        seq_tokens: int = 32,
        text_len: int = 32,
        text_dim: int = 4096,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._length = max(1, int(length))
        self.seq_tokens = int(seq_tokens)
        self.text_len = int(text_len)
        self.text_dim = int(text_dim)
        self._seed = int(seed)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        g = torch.Generator().manual_seed(self._seed + index)
        s = self.seq_tokens
        b = 1
        d = _PATCH_DIM
        # Layout before wan_data_step.transpose(0,1): [S, B, D] (see wan_data_step + FlowMatchingPipeline)
        video_latents = torch.randn((s, b, d), generator=g, dtype=torch.float32)
        context_embeddings = torch.randn((self.text_len, b, self.text_dim), generator=g, dtype=torch.float32)
        loss_mask = torch.ones((s, b, d), dtype=torch.float32)
        grid_sizes = torch.tensor([[1, 4, 4]], dtype=torch.long)

        # micro_batch_size=1 → one packed segment for video (length s) and one for text (length text_len)
        cu_q = torch.tensor([0, s], dtype=torch.int32)
        cu_kv = torch.tensor([0, self.text_len], dtype=torch.int32)
        pself = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_q,
            cu_seqlens_q_padded=cu_q,
            cu_seqlens_kv_padded=cu_q,
            max_seqlen_q=s,
            max_seqlen_kv=s,
        )
        pcross = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            cu_seqlens_q_padded=cu_q,
            cu_seqlens_kv_padded=cu_kv,
            max_seqlen_q=s,
            max_seqlen_kv=self.text_len,
        )
        packed = {"self_attention": pself, "cross_attention": pcross}

        return {
            "video_latents": video_latents,
            "context_embeddings": context_embeddings,
            "grid_sizes": grid_sizes,
            "loss_mask": loss_mask,
            "packed_seq_params": packed,
        }


@dataclass(kw_only=True)
class WanDatasetConfig(DatasetProvider):
    """DatasetProvider for WAN pretrain (synthetic or future Energon path)."""

    path: Optional[str] = None
    seq_length: int = 1024
    packing_buffer_size: Optional[int] = None
    micro_batch_size: int = 1
    global_batch_size: int = 1
    num_workers: int = 8

    dataloader_type: Optional[Literal["single", "cyclic", "batch", "external"]] = "cyclic"

    def build_datasets(
        self, context: DatasetBuildContext
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        use_path = self.path is not None and (not isinstance(self.path, str) or bool(self.path.strip()))

        if use_path:
            raise NotImplementedError(
                "WAN training with dataset.path (Energon) is not implemented in Primus yet. "
                "Unset path for synthetic data, or extend primus.diffusion.data.wan.wan_energon_datamodule."
            )

        if self.micro_batch_size != 1:
            raise NotImplementedError(
                "Synthetic WAN dataset requires micro_batch_size=1 (packed sequence metadata is per-sample)."
            )

        self.dataloader_type = "cyclic"

        def _make_ds(n: int) -> Optional[_SyntheticWanDataset]:
            if n <= 0:
                return None
            return _SyntheticWanDataset(length=max(1, n))

        return (
            _make_ds(context.train_samples),
            _make_ds(context.valid_samples),
            _make_ds(context.test_samples),
        )
