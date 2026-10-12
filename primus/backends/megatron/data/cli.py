###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Data-command provider hook for the Megatron backend."""

from primus.backends.megatron.data.diffusion.preprocessing.commands import (
    register_data_subcommands,
)

__all__ = ["register_data_subcommands"]
