###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Register the precomputed HY-WorldPlay SFT dataset."""

from primus.backends.diffusion.data.worldplay import (
    WorldPlayLatentDataset,
    WorldPlayLatentProcessor,
)
from primus.backends.diffusion.utils.log import logger


def build_worldplay_dataset(dataset_config: dict):
    dataset = WorldPlayLatentDataset(dataset_config)
    processor = WorldPlayLatentProcessor()
    logger.info(f"Built WorldPlay latent dataset with {len(dataset)} samples")
    return dataset, processor
