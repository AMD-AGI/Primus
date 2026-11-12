###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Factory registry for vision model, dataset, and trainer builders."""

from __future__ import annotations

from typing import Any, Callable


def _build_vggt_model(model_config: dict):
    from primus.backends.vision.models.vggt import build_vggt_model

    return build_vggt_model(model_config)


def _build_worldmirror_model(model_config: dict):
    from primus.backends.vision.models.worldmirror import build_worldmirror_model

    return build_worldmirror_model(model_config)


def _build_synthetic_model(model_config: dict):
    from primus.backends.vision.models.synthetic import build_synthetic_model

    return build_synthetic_model(model_config)


def _build_vggt_dataset(dataset_config: dict):
    from primus.backends.vision.data.vggt import build_vggt_dataset

    return build_vggt_dataset(dataset_config)


def _build_worldmirror_dataset(dataset_config: dict):
    from primus.backends.vision.data.worldmirror import build_worldmirror_dataset

    return build_worldmirror_dataset(dataset_config)


def _build_synthetic_dataset(dataset_config: dict):
    from primus.backends.vision.data.synthetic import build_synthetic_dataset

    return build_synthetic_dataset(dataset_config)


def _build_fsdp2_trainer(
    *,
    model,
    dataset,
    processor,
    trainer_args: dict,
    eval_dataset=None,
    eval_processor=None,
):
    from primus.backends.vision.trainers.fsdp2 import VisionFSDP2Trainer

    return VisionFSDP2Trainer(
        model=model,
        dataset=dataset,
        processor=processor,
        trainer_args=trainer_args,
        eval_dataset=eval_dataset,
        eval_processor=eval_processor,
    )


MODEL_BUILDERS: dict[str, Callable[[dict], Any]] = {
    "vggt": _build_vggt_model,
    "worldmirror": _build_worldmirror_model,
    "synthetic": _build_synthetic_model,
}
DATASET_BUILDERS: dict[str, Callable[[dict], tuple]] = {
    "vggt": _build_vggt_dataset,
    "worldmirror": _build_worldmirror_dataset,
    "synthetic": _build_synthetic_dataset,
}
TRAINER_BUILDERS: dict[str, Callable[..., Any]] = {
    "fsdp2": _build_fsdp2_trainer,
}


def get_model_builder(name: str) -> Callable[[dict], Any]:
    try:
        return MODEL_BUILDERS[name]
    except KeyError:
        raise KeyError(f"Unknown vision model name: {name}") from None


def get_dataset_builder(name: str) -> Callable[[dict], tuple]:
    try:
        return DATASET_BUILDERS[name]
    except KeyError:
        raise KeyError(f"Unknown vision dataset name: {name}") from None


def get_trainer_builder(name: str) -> Callable[..., Any]:
    try:
        return TRAINER_BUILDERS[name]
    except KeyError:
        raise KeyError(f"Unknown vision trainer name: {name}") from None
