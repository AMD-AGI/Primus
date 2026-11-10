###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""primus.backends.megatron.core.models.muse_spark.number_utils.parse_positive_int (standard library only)."""

import importlib.util
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[5] / "primus/backends/megatron/core/models/muse_spark/number_utils.py"
)
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
    got = parse(value)
    assert got == expected and type(got) is int


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        0,
        -1,
        "0",
        "-0",
        "-5",
        "-1e3",
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
    assert type(parse(np.float32(512.0))) is int and type(parse(np.int64(512))) is int
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
    # also when a value is set: the default is checked on every call
    with pytest.raises(ValueError, match="default"):
        number_utils.parse_positive_int("7", default, "ERR")


@pytest.mark.parametrize("value", ["0" * 70 + "512", "0_" * 35 + "512"])
def test_a_long_text_is_rejected_by_the_length_limit_alone(value):
    # valid and in range apart from its length: only _MAX_TEXT_LEN rejects it
    assert len(value) > number_utils._MAX_TEXT_LEN
    assert number_utils._NUMBER.fullmatch(value)
    with pytest.raises(ValueError, match="bad value"):
        parse(value)


@pytest.mark.parametrize("value", [" " * 1000 + "512", " " * 100 + "5", "\u00a0512", "512\u3000"])
def test_whitespace_counts_toward_the_limit_and_must_be_ascii(value):
    # the length limit applies to the raw value, and only ASCII spaces are stripped
    with pytest.raises(ValueError, match="bad value"):
        number_utils.parse_int(value, "bad value")
    assert number_utils.parse_int(" \t512\n", "bad value") == 512


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0),
        (-5, -5),
        ("-1e3", -1000),
        ("-0", 0),
        ("0.0", 0),
        (-512.0, -512),
        ("+7", 7),
        (Fraction(-4, 1), -4),
        # a zero is 0 whatever its exponent (Decimal("0e30").adjusted() is 30)
        ("0e30", 0),
        ("0E+19", 0),
        ("-0e400", 0),
        ("0.000e999999", 0),
        (Decimal("0E+20"), 0),
    ],
)
def test_parse_int_takes_any_sign(value, expected):
    got = number_utils.parse_int(value, "bad value")
    assert got == expected and type(got) is int


@pytest.mark.parametrize(
    "value", [None, "", True, "-+5", "--1", "-0.5", -0.5, Fraction(-3, 2), "-1e400", -(2**63), "nan"]
)
def test_parse_int_rejects(value):
    with pytest.raises(ValueError, match="bad value"):
        number_utils.parse_int(value, "bad value")


@pytest.mark.parametrize("value", ["\u00a0", " \u00a0 ", "\u2003"])
def test_non_ascii_whitespace_is_not_blank(value):
    # a non-ASCII space (e.g. NBSP) is an error, not the default
    with pytest.raises(ValueError, match="^bad value$"):
        parse(value)
