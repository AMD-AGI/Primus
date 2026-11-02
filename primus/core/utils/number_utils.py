###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Number parsing for config values (standard library only: the conversion hook uses it
without torch / Megatron)."""

import numbers
import re
from decimal import Decimal
from fractions import Fraction

# digits with single ``_`` separators between them (as in Python literals), an optional
# fraction and exponent: "512", "60_000", "60e9", "512.0", "+1E3"
# (ASCII digits only: ``\d`` would also take e.g. full-width digits)
_NUMBER = re.compile(r"[+]?[0-9]+(?:_[0-9]+)*(?:\.[0-9]+(?:_[0-9]+)*)?(?:[eE][+-]?[0-9]+)?", re.ASCII)
# The largest accepted value: int64, what torch / TE / C code take. The check runs on the
# Decimal's exponent first, before any int is built, so "1e10000000" is rejected at once.
MAX_VALUE = 2**63 - 1
# digits of MAX_VALUE (19): a Decimal whose exponent is above this is above MAX_VALUE
_MAX_DIGITS = len(str(MAX_VALUE))
_MAX_TEXT_LEN = 64


def _decimal_to_int(d: Decimal, bad: ValueError) -> int:
    """An integral, finite ``d`` as an int (exponent checked first: no huge int is built)."""
    try:  # a huge +/- exponent raises InvalidOperation in to_integral; NaN / inf raise too
        if not d.is_finite() or d.adjusted() > _MAX_DIGITS or d != d.to_integral_value():
            raise bad
        return int(d)
    except ArithmeticError:
        raise bad from None


def parse_positive_int(value, default: int, error: str) -> int:
    """``value`` as a positive int; ``default`` when it is None or a blank string (the
    caller's default is checked the same way: it must itself be a positive int).

    Accepted: an int (also a numpy integer), an integral real number (``512.0``, also a
    numpy float, a ``Fraction``), an integral ``decimal.Decimal`` (``Decimal("512")``), and a string of a number (ASCII digits) with surrounding whitespace and
    ``_`` digit separators between digits (``" 512 "``, ``"60_000"``, ``"60e9"``, ``"512.0"``). A
    string is checked exactly (``"512.0000000000000001"`` is a fraction, not 512).
    Anything else (a bool or numpy bool, bytes, a fraction, NaN / inf, <= 0, a value above
    MAX_VALUE (int64) such as ``"1e400"``, non-ASCII digits, a malformed
    string such as ``"_512_"`` or another type) raises ``ValueError(error)``: a typo must
    not silently run with another value.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        if isinstance(default, bool) or not isinstance(default, int) or not 1 <= default <= MAX_VALUE:
            raise ValueError(f"parse_positive_int: default {default!r} is not a positive int")
        return default
    bad = ValueError(error)
    if isinstance(value, bool):  # an int subclass; numpy bools are not numbers: else branch
        raise bad
    if isinstance(value, numbers.Integral):
        n = int(value)
    elif isinstance(value, numbers.Real):
        try:  # exact (no float rounding of a large Fraction); inf / NaN raise
            q = Fraction(value) if isinstance(value, (float, numbers.Rational)) else Fraction(float(value))
        except (ArithmeticError, ValueError, TypeError):  # inf, NaN, too large for float
            raise bad from None
        if q.denominator != 1:
            raise bad
        n = q.numerator
    elif isinstance(value, Decimal):  # not a numbers.Real; exact like the string branch
        n = _decimal_to_int(value, bad)
    elif isinstance(value, str):
        text = value.strip()
        if len(text) > _MAX_TEXT_LEN or not _NUMBER.fullmatch(text):
            raise bad
        try:  # a huge exponent can raise InvalidOperation already in Decimal()
            d = Decimal(text.replace("_", ""))
        except ArithmeticError:
            raise bad from None
        n = _decimal_to_int(d, bad)  # exact
    else:
        raise bad
    if not 1 <= n <= MAX_VALUE:
        raise bad
    return n
