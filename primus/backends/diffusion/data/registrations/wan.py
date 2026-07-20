###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Register Wan dataset builder."""

from primus.backends.diffusion.data import (
    DatasetConfig,
    WanVideoDataProcessor,
    WanVideoDataset,
)
from primus.backends.diffusion.utils.log import logger


def build_wan_dataset(dataset_config: dict):
    processor_config = dataset_config["processor_config"]
    processor = WanVideoDataProcessor(processor_config)
    processor.build()
    logger.info("Built data processor")

    dataset = WanVideoDataset(
        processor=processor,
        config=DatasetConfig(**dataset_config),
    )
    logger.info(f"Built dataset with {len(dataset)} samples")
    return dataset, processor
