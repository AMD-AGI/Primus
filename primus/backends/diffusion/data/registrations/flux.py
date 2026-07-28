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


def build_flux_dataset(dataset_config: dict):
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
        logger.info(f"Built FLUX raw image-text dataset with {len(dataset)} samples")
    elif dataset_type == "precomputed":
        processor = FluxPrecomputedProcessor(processor_config)
        processor.build()
        dataset = FluxPrecomputedDataset(dataset_config["dataset_path"])
        logger.info(f"Built FLUX precomputed dataset with {len(dataset)} samples")
    else:
        raise ValueError("FLUX dataset_type must be either 'precomputed' or 'raw'")
    return dataset, processor
