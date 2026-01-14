###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron argument patches.

This package contains one file per argument-related patch. Each module defines
exactly one ``@register_patch`` function; the implementations are identical to
the original versions in ``args_patches.py``.

The top-level ``primus.backends.megatron.patches.__init__`` uses an auto-import
mechanism that discovers all ``*_patches`` modules (including those under this
package), so simply importing this package is not required for patch
registration but is provided for convenience.
"""
from . import (  # noqa: F401
    checkpoint_path_patches,
    data_path_split_patches,
    iterations_to_skip_default_patches,
    logging_level_patches,
    mock_data_patches,
    moe_layer_freq_patches,
    sequence_parallel_tp1_patches,
    tensorboard_path_patches,
    wandb_config_patches,
)

__all__ = [
    "checkpoint_path_patches",
    "tensorboard_path_patches",
    "wandb_config_patches",
    "logging_level_patches",
    "data_path_split_patches",
    "mock_data_patches",
    "sequence_parallel_tp1_patches",
    "iterations_to_skip_default_patches",
    "moe_layer_freq_patches",
]
