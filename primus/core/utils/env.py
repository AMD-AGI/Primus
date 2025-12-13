###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

from primus.core.utils import constant_vars as const


def get_torchrun_env():
    """Get torchrun-provided environment variables with defaults."""
    return {
        "rank": int(os.getenv("RANK", const.LOCAL_NODE_RANK)),
        "world_size": int(os.getenv("WORLD_SIZE", const.LOCAL_WORLD_SIZE)),
        "master_addr": os.getenv("MASTER_ADDR", const.LOCAL_MASTER_ADDR),
        "master_port": int(os.getenv("MASTER_PORT", const.LOCAL_MASTER_PORT)),
        "local_rank": int(os.getenv("LOCAL_RANK", const.LOCAL_NODE_RANK)),
        "local_world_size": int(os.getenv("LOCAL_WORLD_SIZE", const.LOCAL_WORLD_SIZE)),
    }
