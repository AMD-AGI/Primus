###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Small value checks shared by model configs, converters and fusions.

Free of torch and Megatron imports, so converters and hooks can use it on their own."""

import math
import numbers


def positive_finite(name, value, where=""):
    """``value`` as a float if it is a positive finite real number (``int``, ``float``, a
    numpy scalar, any ``numbers.Real``). Bools, strings (also a quoted number such as
    ``'1'``: config values are checked as written), None, and zero, negative, non-finite or
    out-of-float-range values raise ``ValueError``. ``where`` prefixes the message, e.g.
    ``"Granite config.json: "``."""
    if not isinstance(value, bool) and isinstance(value, numbers.Real):
        try:
            number = float(value)
        except (OverflowError, ValueError):
            number = math.nan
        if number > 0.0 and math.isfinite(number):
            return number
    try:
        shown = repr(value)
    except Exception:  # e.g. an int above Python's integer-string limit
        shown = f"a {type(value).__name__} that cannot be shown"
    raise ValueError(f"{where}{name} must be a positive finite number, got {shown}")
