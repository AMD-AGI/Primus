###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""primus.core.utils.number_utils.parse_positive_int (standard library only)."""

import importlib.util
from fractions import Fraction
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[4] / "primus/core/utils/number_utils.py"
_spec = importlib.util.spec_from_file_location("_number_utils", _PATH)
number_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(number_utils)


def parse(value):
    return number_utils.parse_positive_int(value, 7, "bad value")


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, 7),
        ("", 7),
        ("  ", 7),
        (512, 512),
        (512.0, 512),
        ("512", 512),
        (" 512 ", 512),
        ("+512", 512),
        ("60_000", 60000),
        ("60e9", 60 * 10**9),
        ("6.0E10", 60 * 10**9),
        ("512.0", 512),
        ("816_000_000_000", 816 * 10**9),
        (Fraction(10, 1), 10),
        (Fraction(10**18 + 1, 1), 10**18 + 1),  # exact: not rounded through float
    ],
)
def test_accepted(value, expected):
    assert parse(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        0,
        -1,
        "0",
        "-5",
        512.5,
        float("nan"),
        float("inf"),
        "nan",
        "inf",
        "512.0000000000000001",  # a fraction: exact, not rounded to 512 by float
        "_512",
        "512_",
        "_5_1_2_",
        "5__12",
        "1e3_0",
        "0x200",
        "512 MiB",
        b"512",
        [512],
        Fraction(1, 2),
        "\uff15\uff11\uff12",  # full-width 512
        "\u0665\u0661\u0662",  # Arabic-Indic 512
        "1e400",
        "1e19",
        "10_000_000_000_000_000_000",
        10**19,
        1e19,
        "1" + "0" * 100,
        "9223372036854775808",  # int64 max + 1
        2**63,
        "1e99999999999999999999",  # exponent beyond Decimal's context: no InvalidOperation
        "1e-99999999999999999999",
        "1.5e-99999999999999999999",
        Fraction(10**400, 1),  # too large for a float: no OverflowError
        Fraction(10**18 + 1, 2),
    ],
)
def test_rejected(value):
    with pytest.raises(ValueError, match="bad value"):
        parse(value)


def test_numpy_scalars():
    np = pytest.importorskip("numpy")
    assert parse(np.int64(512)) == 512
    assert parse(np.float32(512.0)) == 512
    with pytest.raises(ValueError):
        parse(np.bool_(True))


def test_a_huge_exponent_is_rejected_at_once():
    import time

    t0 = time.monotonic()
    for text in ("1e10000000", "1e999999999999", "9" * 60 + "e9", "1e99999999999999999999"):
        with pytest.raises(ValueError, match="bad value"):
            parse(text)
    assert time.monotonic() - t0 < 1.0


def test_the_largest_accepted_value_is_int64_max():
    assert number_utils.MAX_VALUE == 2**63 - 1
    assert parse("9_223_372_036_854_775_807") == 2**63 - 1
    assert parse(2**63 - 1) == 2**63 - 1
    assert parse("816e9") == 816 * 10**9


def test_decimal_is_parsed_like_its_string():
    from decimal import Decimal

    assert number_utils.parse_positive_int(Decimal("5"), 7, "ERR") == 5
    assert number_utils.parse_positive_int(Decimal("5.000"), 7, "ERR") == 5
    for bad in (
        Decimal("5.5"),
        Decimal("0"),
        Decimal("-1"),
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("1e400"),
    ):
        with pytest.raises(ValueError, match="ERR"):
            number_utils.parse_positive_int(bad, 7, "ERR")


@pytest.mark.parametrize("default", [0, -1, 1.5, True, "5", 2**63])
def test_a_bad_default_is_not_returned(default):
    with pytest.raises(ValueError, match="default"):
        number_utils.parse_positive_int(None, default, "ERR")
