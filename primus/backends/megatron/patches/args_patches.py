###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Arguments Patches

Historically, all Megatron argument patches lived in this single module.
They have now been split into one-file-per-patch under the
``primus.backends.megatron.patches.args`` package:

    - checkpoint_path_patches
    - tensorboard_path_patches
    - wandb_config_patches
    - logging_level_patches
    - data_path_split_patches
    - mock_data_patches
    - sequence_parallel_tp1_patches
    - iterations_to_skip_default_patches
    - moe_layer_freq_patches

This module re-exports the original patch functions for backward
compatibility so existing imports continue to work.
"""

from primus.backends.megatron.patches.args.checkpoint_path_patches import (  # noqa: F401
    patch_checkpoint_path,
)
from primus.backends.megatron.patches.args.data_path_split_patches import (  # noqa: F401
    patch_data_path_split,
)
from primus.backends.megatron.patches.args.iterations_to_skip_default_patches import (  # noqa: F401
    patch_iterations_to_skip_default,
)
from primus.backends.megatron.patches.args.logging_level_patches import (  # noqa: F401
    patch_logging_level,
)
from primus.backends.megatron.patches.args.mock_data_patches import (  # noqa: F401
    patch_mock_data,
)
from primus.backends.megatron.patches.args.moe_layer_freq_patches import (  # noqa: F401
    patch_moe_layer_freq,
)
from primus.backends.megatron.patches.args.sequence_parallel_tp1_patches import (  # noqa: F401
    patch_sequence_parallel_tp1,
)
from primus.backends.megatron.patches.args.tensorboard_path_patches import (  # noqa: F401
    patch_tensorboard_path,
)
from primus.backends.megatron.patches.args.wandb_config_patches import (  # noqa: F401
    patch_wandb_config,
)

__all__ = [
    "patch_checkpoint_path",
    "patch_tensorboard_path",
    "patch_wandb_config",
    "patch_logging_level",
    "patch_data_path_split",
    "patch_mock_data",
    "patch_sequence_parallel_tp1",
    "patch_iterations_to_skip_default",
    "patch_moe_layer_freq",
]
