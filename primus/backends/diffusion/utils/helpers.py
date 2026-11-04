###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################


def to_2tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value, value)
