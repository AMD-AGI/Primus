###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from primus.backends.diffusion.data.flux_precomputed import (
    FluxPrecomputedDataset,
    FluxPrecomputedProcessor,
    FluxRawImageTextDataset,
    FluxRawImageTextProcessor,
)
from primus.backends.diffusion.utils.log import logger


def _build_flux_dataset_from_config(dataset_config: dict, *, role: str):
    processor_config = dataset_config.get("processor_config", {}) or {}
    dataset_type = str(dataset_config.get("dataset_type", "precomputed")).lower()
    if dataset_type == "raw":
        processor = FluxRawImageTextProcessor(processor_config)
        processor.build()
        dataset = FluxRawImageTextDataset(
            dataset_path=dataset_config.get("dataset_path"),
            dataset_format=dataset_config.get("dataset_format", "webdataset"),
            dataset_name=dataset_config.get("dataset"),
            data_folder=dataset_config.get("data_folder"),
        )
        logger.info(f"Built FLUX {role} raw image-text dataset with {len(dataset)} samples")
    elif dataset_type == "precomputed":
        processor = FluxPrecomputedProcessor(processor_config)
        processor.build()
        dataset = FluxPrecomputedDataset(dataset_config["dataset_path"])
        logger.info(f"Built FLUX {role} precomputed dataset with {len(dataset)} samples")
    else:
        raise ValueError("FLUX dataset_type must be either 'precomputed' or 'raw'")
    return dataset, processor


def build_flux_dataset(dataset_config: dict):
    dataset, processor = _build_flux_dataset_from_config(dataset_config, role="train")
    eval_dataset_path = dataset_config.get("eval_dataset_path")
    if not eval_dataset_path:
        return dataset, processor

    eval_config = dict(dataset_config)
    eval_config["dataset_path"] = eval_dataset_path
    eval_processor_config = dict(eval_config.get("processor_config", {}) or {})
    eval_processor_config["prompt_dropout_prob"] = 0.0
    eval_config["processor_config"] = eval_processor_config
    eval_dataset, eval_processor = _build_flux_dataset_from_config(eval_config, role="eval")
    return dataset, processor, eval_dataset, eval_processor
