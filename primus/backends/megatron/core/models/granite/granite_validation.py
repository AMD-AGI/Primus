###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Validation shared by the Granite converter and the Granite model config.

Kept free of Megatron imports so the SFT checkpoint hook can use it on its own."""

import math


def positive_finite(name, value, where=""):
    """``value`` as a float if it is a positive finite int/float. Bools, strings, None,
    zero, negative and non-finite values raise ``ValueError`` (``where`` prefixes the
    message, e.g. ``"Granite config.json: "``)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}{name} must be a positive finite number, got {value!r}")
    value = float(value)
    if not (value > 0.0 and math.isfinite(value)):
        raise ValueError(f"{where}{name} must be a positive finite number, got {value}")
    return value
