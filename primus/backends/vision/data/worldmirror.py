###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Hypersim loader for WorldMirror, using the in-tree sampler."""

from __future__ import annotations

import os

_TRANSFORM = (
    "ImageAugmentation(apply_aug=True, color_jitter={'brightness': 0.5, 'contrast': 0.5, "
    "'saturation': 0.5, 'hue': 0.1, 'p': 0.9}, gray_scale=True, gau_blur=False)"
)


def _train_expression(dataset_config: dict) -> str:
    configured = dataset_config.get("train_datasets")
    if configured:
        if isinstance(configured, (list, tuple)):
            return " + ".join(str(item) for item in configured)
        return str(configured)
    root = str(dataset_config.get("hypersim_dir") or "").strip()
    if not root:
        raise ValueError("Set HYPERSIM_DIR or data.hypersim_dir for WorldMirror.")
    views = int(dataset_config.get("max_images_per_gpu", 24))
    return (
        "400 @ HyperSim_Multi(split='train', "
        f"num_views={views}, aug_crop=0.77, transform={_TRANSFORM}, ROOT={root!r})"
    )


def build_worldmirror_dataset(dataset_config: dict):
    from torch.utils.data import DataLoader

    from primus.backends.vision.worldmirror.training.data.sampler.dynamic_sampler import (
        DynamicBatchSampler,
        DynamicDistributedSampler,
    )
    from primus.backends.vision.worldmirror.training.data.train.hypersim import HyperSim_Multi
    from primus.backends.vision.worldmirror.training.utils.image import ImageAugmentation

    expression = _train_expression(dataset_config)
    dataset = eval(  # noqa: S307 - same dataset expression WorldMirror resolves from its config
        expression,
        {
            "__builtins__": __builtins__,
            "HyperSim_Multi": HyperSim_Multi,
            "ImageAugmentation": ImageAugmentation,
        },
    )
    # Pass the replica count explicitly so a single process does not need torch.distributed.
    sampler = DynamicDistributedSampler(
        dataset,
        num_replicas=int(os.environ.get("WORLD_SIZE", "1")),
        rank=int(os.environ.get("RANK", "0")),
        shuffle=True,
    )
    max_images = int(dataset_config.get("max_images_per_gpu", 24))
    batch_sampler = DynamicBatchSampler(
        sampler,
        int(dataset_config.get("min_num_of_views", 2)),
        max_images,
        max_img_per_gpu=max_images,
        aspect_ratio_range=list(dataset_config.get("aspect_ratios") or [0.5, 2.0]),
        num_pixels_range=list(dataset_config.get("num_pixels_range") or [100000, 250000]),
    )
    workers = int(dataset_config.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)
    return loader, None
