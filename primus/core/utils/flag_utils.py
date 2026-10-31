###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Parse boolean recipe flags read through ``get_param`` in patch conditions."""

import numbers
import sys
from typing import Any

TRUE_STRINGS = ("true", "1", "yes", "on")
FALSE_STRINGS = ("false", "0", "no", "off")


def _is_numpy_bool(value) -> bool:
    numpy = sys.modules.get("numpy")  # a numpy value means numpy is already imported
    return numpy is not None and isinstance(value, numpy.bool_)


def as_bool(value: Any, name: str) -> bool:
    """``value`` as a bool, strictly.

    Accepted: a bool or numpy bool; None (False: the flag is not set); an integral 0 or 1
    (``int``, numpy integers, any ``numbers.Integral``); and the
    strings in ``TRUE_STRINGS`` / ``FALSE_STRINGS`` (case and surrounding spaces are
    ignored). A string comes from env-substituted YAML such as ``${VAR:false}``, a quoted
    scalar or a CLI override, so ``"false"`` is False, not True as with ``bool("false")``.

    Everything else raises ValueError: other ints (a typo such as 2 is not taken as
    true), floats, lists, other strings, and the empty string. An empty string usually
    means an env substitution such as ``${VAR:}`` with no value; it raises rather than
    silently turning a flag off whose default is true (``get_param`` replaces only None
    with the default).

    ``per_head_muon._as_bool`` is a separate, looser parser kept as it is: it also
    accepts "none"/"null"/"" and any non-string (through ``bool()``), and existing
    Per-Head Muon recipes rely on that."""
    if value is None or isinstance(value, bool) or _is_numpy_bool(value):
        return bool(value)
    if isinstance(value, numbers.Integral) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in TRUE_STRINGS:
            return True
        if v in FALSE_STRINGS:
            return False
    accepted = "/".join(f"{t}/{f}" for t, f in zip(TRUE_STRINGS, FALSE_STRINGS))
    raise ValueError(f"{name} must be a boolean ({accepted}), got {value!r}")
