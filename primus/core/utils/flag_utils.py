###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Strict yes/no parsing of config values (a YAML value from an environment substitution other
than true/false stays a string, which plain truthiness would read as True).

``bool_flag`` moved here unchanged from ``primus.backends.megatron.core.models.muse_spark.model_types``
(which re-exports it with ``TRUE_TEXT`` / ``FALSE_TEXT``), so other models share one parser."""

import numbers

__all__ = ["FALSE_TEXT", "TRUE_TEXT", "bool_flag"]

TRUE_TEXT = ("1", "true", "yes", "on")
FALSE_TEXT = ("0", "false", "no", "off")


def bool_flag(value, key, default=False) -> bool:
    """A boolean config key (``key``) as a bool.

    * None or a blank string (unset, or an unset env-substituted value): ``default``, parsed by
      these same rules (a string default "false" is False; a None or blank default is False);
    * a bool, or a numpy bool (``np.bool_``): as is;
    * the ints 0 / 1: False / True;
    * a string: true for 1 / true / yes / on, false for 0 / false / no / off (any case, surrounding
      whitespace stripped).

    Anything else (e.g. a typo such as ``"ture"``, ``2``, ``1.0``) raises ValueError naming ``key``:
    a typo must not silently turn a feature on or off."""
    if value is None or (isinstance(value, str) and not value.strip()):
        # the default is parsed by the same rules (a string default "false" is False)
        return bool_flag(default, key, False)
    if isinstance(value, bool):
        return value
    if getattr(getattr(value, "dtype", None), "kind", None) == "b" and getattr(value, "ndim", None) == 0:
        return bool(value)  # np.bool_ (not a numbers.Integral)
    if isinstance(value, numbers.Integral) and int(value) in (0, 1):
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in TRUE_TEXT:
            return True
        if text in FALSE_TEXT:
            return False
    raise ValueError(f"{key}={value!r} is not a boolean: use true / false (also yes / no, on / off, 1 / 0)")
