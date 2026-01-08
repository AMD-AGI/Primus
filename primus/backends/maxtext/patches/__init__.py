###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText Patches

Import all patch modules to trigger registration.
"""

# Import all patches to trigger registration
from primus.backends.maxtext.patches import logger_patches  # noqa: F401

# TODO: Add more patches as needed:
# from primus.backends.maxtext.patches import wandb_patches  # noqa: F401
# from primus.backends.maxtext.patches import checkpoint_patches  # noqa: F401
# from primus.backends.maxtext.patches import layer_patches  # noqa: F401
