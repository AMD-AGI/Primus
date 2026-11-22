###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest

from primus.core.utils.flag_utils import bool_flag


@pytest.mark.parametrize(
    "value,want",
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("yes", True),
        (" ON ", True),
        ("1", True),
        ("no", False),
        ("off", False),
        ("0", False),
        ("False", False),
    ],
)
def test_bool_flag_values(value, want):
    assert bool_flag(value, "k") is want


@pytest.mark.parametrize("value", ["ture", 2, 1.0, "maybe"])
def test_bool_flag_refuses_other_values(value):
    with pytest.raises(ValueError, match="my_key"):
        bool_flag(value, "my_key")


@pytest.mark.parametrize(
    "value,default,want",
    [(None, False, False), ("", True, True), ("  ", "false", False), (None, None, False), (None, "", False)],
)
def test_bool_flag_unset_is_the_default(value, default, want):
    """Unset (None / blank) takes the default, parsed by the same rules; a None or blank default
    is False (the behaviour bool_flag had in muse_spark, unchanged by the move)."""
    assert bool_flag(value, "k", default) is want


@pytest.mark.parametrize("default", ["maybe", "ture", 2])
def test_a_bad_default_raises_naming_the_key(default):
    with pytest.raises(ValueError, match=r"^k="):
        bool_flag(None, "k", default)
