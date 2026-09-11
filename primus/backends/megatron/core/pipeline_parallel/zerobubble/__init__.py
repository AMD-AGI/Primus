###############################################################################
# Some parts of this code are copied and modified from
# Sea AI Lab's zero-bubble-pipeline-parallelism project
# (https://github.com/sail-sg/zero-bubble-pipeline-parallelism).
#
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

__all__ = ["get_zero_bubble_forward_backward_func"]


def get_zero_bubble_forward_backward_func(*args, **kwargs):
    """Load the PuLP-backed zero-bubble runtime only when it is requested."""
    from .runtime import (
        get_zero_bubble_forward_backward_func as _get_zero_bubble_forward_backward_func,
    )

    return _get_zero_bubble_forward_backward_func(*args, **kwargs)
