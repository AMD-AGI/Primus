###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Imported lazily: the JAX images have no torch, and compiled_step (MaxText) must
# import without pulling in the torch-based projector.


def launch_projection_from_cli(*args, **kwargs):
    from .projection import launch_projection_from_cli as launch

    return launch(*args, **kwargs)


__all__ = [
    "launch_projection_from_cli",
]
