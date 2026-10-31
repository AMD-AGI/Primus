###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for primus/core/utils/flag_utils.py (``as_bool``)."""

import pytest


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, True),
        (False, False),
        (None, False),
        (1, True),
        (0, False),
        ("true", True),
        (" On ", True),
        ("yes", True),
        ("1", True),
        ("false", False),
        ("off", False),
        ("no", False),
        ("0", False),
    ],
)
def test_as_bool_accepts(value, expected):
    from primus.core.utils.flag_utils import as_bool

    assert as_bool(value, "flag") is expected


@pytest.mark.parametrize("name", ["bool_", "int64", "uint8"])
def test_as_bool_accepts_numpy_scalars(name):
    np = pytest.importorskip("numpy")
    from primus.core.utils.flag_utils import as_bool

    for raw in (True, False) if name == "bool_" else (1, 0):
        assert as_bool(getattr(np, name)(raw), "flag") is bool(raw)


def test_as_bool_rejects_numpy_non_booleans():
    np = pytest.importorskip("numpy")
    from primus.core.utils.flag_utils import as_bool

    for value in (np.int64(2), np.float64(1.0)):
        with pytest.raises(ValueError):
            as_bool(value, "flag")


@pytest.mark.parametrize("value", [2, -1, 1.0, 0.0, "maybe", "2", "", "  ", [], {}])
def test_as_bool_rejects(value):
    from primus.core.utils.flag_utils import as_bool

    with pytest.raises(ValueError, match="flag must be a boolean \\(true/false/1/0/yes/no/on/off\\)"):
        as_bool(value, "flag")
