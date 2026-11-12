###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Random multi-view batches for trainer smoke tests."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticMultiView(Dataset):
    def __init__(self, length: int, views: int, image_size: int):
        self.length = length
        self.views = views
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        images = torch.rand(
            self.views,
            3,
            self.image_size,
            self.image_size,
            generator=generator,
        )
        return {"images": images}


def build_synthetic_dataset(dataset_config: dict):
    dataset = SyntheticMultiView(
        length=int(dataset_config.get("length", 8)),
        views=int(dataset_config.get("views", 2)),
        image_size=int(dataset_config.get("image_size", 32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(dataset_config.get("batch_size", 1)),
        num_workers=int(dataset_config.get("num_workers", 0)),
        shuffle=False,
    )
    return loader, None
