###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for primus.core.utils.checker (pure assertion helpers)."""

import pytest

from primus.core.utils.checker import (
    check_equal,
    check_false,
    check_not_equal,
    check_true,
)


def test_check_equal_passes_and_fails():
    check_equal(3, 3)  # no raise
    with pytest.raises(RuntimeError):
        check_equal(3, 4)


def test_check_equal_custom_message():
    with pytest.raises(RuntimeError, match="boom"):
        check_equal(1, 2, msg="boom")


def test_check_not_equal_passes_and_fails():
    check_not_equal(1, 2)
    with pytest.raises(RuntimeError):
        check_not_equal(5, 5)


def test_check_true_passes_and_fails():
    check_true(True)
    check_true(1)
    with pytest.raises(RuntimeError):
        check_true(0)


def test_check_false_passes_and_fails():
    check_false(False)
    check_false("")
    with pytest.raises(RuntimeError):
        check_false(True)
