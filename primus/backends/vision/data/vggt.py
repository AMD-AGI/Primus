###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Co3D loader for VGGT, using the in-tree VGGT sampler."""

from __future__ import annotations

from primus.backends.vision.vggt_model.data.dynamic_dataloader import DynamicTorchDataset


def _common_config(dataset_config: dict, training: bool):
    from omegaconf import OmegaConf

    image_nums = list(dataset_config.get("image_num_range") or ([2, 24] if training else [2, 12]))
    aspects = list(dataset_config.get("aspect_ratios") or ([0.33, 1.0] if training else [1.0, 1.0]))
    return OmegaConf.create(
        {
            "fix_img_num": -1,
            "fix_aspect_ratio": 1.0,
            "load_track": False,
            "track_num": 1024,
            "training": training,
            "inside_random": True,
            "img_size": int(dataset_config.get("img_size", 518)),
            "patch_size": int(dataset_config.get("patch_size", 14)),
            "rescale": True,
            "rescale_aug": training,
            "landscape_check": False,
            "debug": False,
            "get_nearby": True,
            "load_depth": True,
            "img_nums": image_nums,
            "max_img_per_gpu": int(dataset_config.get("max_images_per_gpu", 24)),
            "allow_duplicate_img": True,
            "repeat_batch": False,
            "augs": {
                "cojitter": training,
                "cojitter_ratio": 0.3 if training else 0.5,
                "scales": [0.8, 1.2] if training else None,
                "aspects": aspects,
                "color_jitter": {
                    "brightness": 0.5,
                    "contrast": 0.5,
                    "saturation": 0.5,
                    "hue": 0.1,
                    "p": 0.9,
                }
                if training
                else None,
                "gray_scale": training,
                "gau_blur": False,
            },
        }
    )


def _loader(dataset_config: dict, split: str):
    from omegaconf import OmegaConf

    annotation_dir = str(dataset_config.get("annotation_dir") or "").strip()
    dataset_path = str(dataset_config.get("dataset_path") or "").strip()
    if not annotation_dir or not dataset_path:
        raise ValueError(
            "VGGT Co3D training needs data.dataset_path (CO3D_DIR) and data.annotation_dir "
            "(CO3D_ANNOTATION_DIR, with files named {category}_{split}.jgz)."
        )
    dataset = OmegaConf.create(
        {
            "_target_": "primus.backends.vision.vggt_model.data.composed_dataset.ComposedDataset",
            "dataset_configs": [
                {
                    "_target_": "primus.backends.vision.vggt_model.data.datasets.co3d.Co3dDataset",
                    "split": split,
                    "CO3D_DIR": dataset_path,
                    "CO3D_ANNOTATION_DIR": annotation_dir,
                }
            ],
        }
    )
    return DynamicTorchDataset(
        dataset=dataset,
        common_config=_common_config(dataset_config, training=split == "train"),
        num_workers=int(dataset_config.get("num_workers", 4)),
        shuffle=split == "train",
        pin_memory=True,
        max_img_per_gpu=int(dataset_config.get("max_images_per_gpu", 24)),
    )


def build_vggt_dataset(dataset_config: dict):
    return _loader(dataset_config, "train"), None
