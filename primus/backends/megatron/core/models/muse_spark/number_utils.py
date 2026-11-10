###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Number parsing for config values (standard library only: the conversion hook uses it
without torch / Megatron)."""

import math
import numbers
import re
from decimal import Decimal
from fractions import Fraction

# an optional sign, digits with single ``_`` separators between them (as in Python
# literals), an optional fraction and exponent; unlike Python, the exponent digits take no
# ``_`` ("1e1_0" is refused, Python reads 1e10): "512", "60_000", "60e9", "512.0", "+1E3", "-1e3"; not
# "-+5", "_512", "1e" or "0x10" (ASCII digits only)
_NUMBER = re.compile(r"[+-]?[0-9]+(?:_[0-9]+)*(?:\.[0-9]+(?:_[0-9]+)*)?(?:[eE][+-]?[0-9]+)?", re.ASCII)
# The largest magnitude accepted: INT64_MAX, what torch / TE / C code take (the range is
# symmetric, [-MAX_VALUE, MAX_VALUE], so INT64_MIN itself is refused). The check runs on the
# Decimal's exponent first, before any int is built, so "1e10000000" is rejected at once.
MAX_VALUE = 2**63 - 1
# Decimal.adjusted() (the exponent of the leading digit, digits - 1) of MAX_VALUE: 18. A
# larger exponent is above MAX_VALUE; an equal one is checked by the final range test.
_MAX_ADJUSTED_EXPONENT = len(str(MAX_VALUE)) - 1
# A config value is short: parse_int rejects a longer string (whitespace included) as a
# typo or junk before stripping it; parse_positive_int maps an all-blank string to its
# default first.
_MAX_TEXT_LEN = 64
# the whitespace a string may have around the number (ASCII only, like the digits)
_ASCII_SPACE = " \t\n\r\f\v"


def _decimal_to_int(d: Decimal, bad: ValueError) -> int:
    """An integral, finite ``d`` as an int (exponent checked first: no huge int is built)."""
    try:  # a huge +/- exponent raises InvalidOperation in to_integral; NaN / inf raise too
        if not d.is_finite():
            raise bad
        if d.is_zero():  # Decimal("0e30").adjusted() is 30: a zero is 0 whatever its exponent
            return 0
        if d.adjusted() > _MAX_ADJUSTED_EXPONENT or d != d.to_integral_value():
            raise bad
        return int(d)
    except ArithmeticError:
        raise bad from None


def parse_int(value, error: str) -> int:
    """``value`` as an int of any sign, in ``[-MAX_VALUE, MAX_VALUE]``.

    Accepted: an int (also a numpy integer, signed or unsigned), an integral real number
    (``512.0``, ``Fraction(512, 1)``, also a numpy float), an integral
    ``decimal.Decimal``, and a string of a number (ASCII digits, an optional sign) with
    surrounding whitespace and ``_`` digit separators between digits (``" 512 "``,
    ``"60_000"``, ``"60e9"``, ``"512.0"``, ``"-1e3"``). A string is checked exactly
    (``"512.0000000000000001"`` is not 512). A float is taken at its exact value: a float
    that already lost precision (``9007199254740993.0`` is stored as ...992) is taken as
    stored, so pass large values as an int or a string.
    Rejected with ``ValueError(error)``: None, a blank string, a bool or numpy bool, bytes,
    a non-integral value (``1.5``, ``Fraction(3, 2)``, ``"0.5"``), NaN / inf, a value
    outside ``[-MAX_VALUE, MAX_VALUE]`` (int64) such as ``"1e400"``, non-ASCII digits, a
    malformed string (``"_512_"``, ``"-+5"``), a string longer than ``_MAX_TEXT_LEN``
    characters (surrounding whitespace counts), non-ASCII whitespace (e.g. NBSP), or another type: a
    typo must not silently run with another value.
    """
    bad = ValueError(error)
    if isinstance(value, bool):  # an int subclass; numpy bools are not numbers: else branch
        raise bad
    if isinstance(value, numbers.Integral):
        n = int(value)  # int() first: no arithmetic on a numpy unsigned value (it wraps)
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
        if len(value) > _MAX_TEXT_LEN:
            raise bad
        text = value.strip(_ASCII_SPACE)
        if not _NUMBER.fullmatch(text):
            raise bad
        try:  # a huge exponent can raise InvalidOperation already in Decimal()
            d = Decimal(text.replace("_", ""))
        except ArithmeticError:
            raise bad from None
        n = _decimal_to_int(d, bad)  # exact
    else:
        raise bad
    if not -MAX_VALUE <= n <= MAX_VALUE:
        raise bad
    return n


def parse_positive_int(value, default: int, error: str) -> int:
    """``value`` as a positive int, or ``default`` when ``value`` is None or ASCII-only
    whitespace (non-ASCII whitespace such as NBSP raises, as in :func:`parse_int`).
    ``default`` must itself be an int (not a bool) in ``[1, MAX_VALUE]``; it is not parsed,
    and another default raises ``ValueError`` on every call, also when ``value`` is set.

    Accepts what :func:`parse_int` accepts (an int, an integral ``512.0`` /
    ``Fraction(512, 1)`` / ``Decimal("512")``, ``"60_000"``, ``"60e9"``, ...) when the
    value is in ``[1, MAX_VALUE]``. Anything else (what :func:`parse_int` rejects, such as
    a bool, a non-integral value like ``1.5`` or ``Fraction(3, 2)``, NaN / inf, ``"1e400"``,
    ``"_512_"``; and any value <= 0) raises ``ValueError(error)``.
    """
    # the default is checked on every call, set value or not: a bad default is a code bug
    if isinstance(default, bool) or not isinstance(default, int) or not 1 <= default <= MAX_VALUE:
        raise ValueError(f"parse_positive_int: default {default!r} is not a positive int")
    if value is None or (isinstance(value, str) and not value.strip(_ASCII_SPACE)):
        return default
    return require_positive_int(value, error)


def require_positive_int(value, error: str) -> int:
    """``value`` as an int in ``[1, MAX_VALUE]`` (what :func:`parse_int` accepts), with no
    default: None or a blank string raises ``ValueError(error)`` like any other bad value.
    The one positivity rule :func:`parse_positive_int` uses after its default."""
    n = parse_int(value, error)
    if n < 1:
        raise ValueError(error)
    return n


def parse_softcap(cap, tag: str = "Muse Spark:"):
    """``final_logit_softcapping``: None / 0 (no cap) or a finite number > 0, as a float.

    ``cap * tanh(x / cap)`` with an inf cap is NaN for every logit and a negative cap flips
    the sign of every logit, so those (and NaN, bools including ``numpy.bool_``, and
    non-numbers) raise ValueError. None stays None; ``"0"`` / ``-0.0`` give 0.0 and ``"20"``
    20.0."""
    if cap is None:
        return None
    is_bool = isinstance(cap, bool) or getattr(getattr(cap, "dtype", None), "kind", None) == "b"
    try:
        value = float(cap)
    except (TypeError, ValueError, OverflowError):  # 10**400: too large for a float
        value = float("nan")
    if is_bool or not math.isfinite(value) or value < 0:
        raise ValueError(f"{tag} final_logit_softcapping={cap!r} must be null, 0 or a finite number > 0")
    return value + 0.0  # -0.0 -> 0.0 (no cap)
