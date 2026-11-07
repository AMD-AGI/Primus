###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for primus.core.utils.validation (pure Python)."""

import math
from fractions import Fraction

import pytest

from primus.core.utils.validation import positive_finite


@pytest.mark.parametrize("value", [0.22, 1, 12.0, 1e-300, Fraction(1, 4)])
def test_positive_finite_accepts_positive_real_numbers(value):
    out = positive_finite("m", value)
    assert type(out) is float and out == float(value)


def test_positive_finite_accepts_numpy_scalars():
    np = pytest.importorskip("numpy")
    for value in (np.float32(0.22), np.float64(0.22), np.int64(3)):
        out = positive_finite("m", value)
        assert type(out) is float and out == pytest.approx(float(value))


@pytest.mark.parametrize(
    "value", [True, False, "0.22", "1", None, [0.22], 0, 0.0, -1, -0.5, math.inf, -math.inf, math.nan]
)
def test_positive_finite_rejects_with_value_error(value):
    with pytest.raises(ValueError, match="cfg: m must be a positive finite number"):
        positive_finite("m", value, where="cfg: ")


def test_positive_finite_huge_int_is_value_error_not_overflow():
    # json.loads gives arbitrary-precision ints; callers catch ValueError.
    for value in (10**400, -(10**400)):
        with pytest.raises(ValueError, match="positive finite") as e:
            positive_finite("m", value)
        assert e.value.__cause__ is None


@pytest.mark.parametrize(
    "value, shown", [(0, "got 0"), (-1, "got -1"), ("1", "got '1'"), (math.nan, "got nan")]
)
def test_positive_finite_message_shows_the_value_as_given(value, shown):
    with pytest.raises(ValueError) as e:
        positive_finite("m", value)
    assert shown in str(e.value)


def test_positive_finite_string_message_hints_at_yaml_float_syntax():
    # PyYAML reads 1e-2 (no dot) as the string '1e-2'.
    with pytest.raises(ValueError, match=r"got '1e-2' \(a string: .*1\.0e-2\)"):
        positive_finite("m", "1e-2")


def test_positive_finite_real_whose_float_raises_type_error_is_value_error():
    class BadReal(float):
        def __float__(self):
            raise TypeError("no float")

    with pytest.raises(ValueError, match="positive finite"):
        positive_finite("m", BadReal(1.0))


def test_positive_finite_rejects_numpy_nan_and_zero():
    np = pytest.importorskip("numpy")
    for value in (np.float64("nan"), np.float32(0.0), np.int64(-2)):
        with pytest.raises(ValueError, match="positive finite"):
            positive_finite("m", value)


def test_positive_finite_message_never_raises():
    # A value whose repr() raises (here a stand-in): the message still names where and the key.
    class Unshowable(float):
        def __repr__(self):
            raise ValueError("no repr")

    with pytest.raises(ValueError, match=r"^cfg: m must be a positive finite number, got a Unshowable"):
        positive_finite("m", Unshowable(-1.0), where="cfg: ")
