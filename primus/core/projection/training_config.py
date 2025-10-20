###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Configuration for training the profiler models.
    """

    # batch size for training
    micro_batch_size: int = 1
