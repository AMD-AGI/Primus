###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest

from primus.core.utils.flag_utils import bool_flag


@pytest.mark.parametrize(
    "value,default,want",
    [(None, None, False), (None, "", False), ("  ", None, False)],
)
def test_bool_flag_unset_is_the_default(value, default, want):
    """Cases the Muse Spark tests (test_muse_spark_cpu.py) do not cover: a None or blank default
    is False. Values, typos and string defaults are tested there, through the re-export."""
    assert bool_flag(value, "k", default) is want


@pytest.mark.parametrize("default", ["maybe", "ture", 2])
def test_a_bad_default_raises_naming_the_key_when_unset(default):
    """Unchanged by the move from muse_spark: the default is parsed only when the value is unset."""
    with pytest.raises(ValueError, match=r"^k="):
        bool_flag(None, "k", default)
    assert bool_flag(True, "k", default) is True


def test_a_numpy_bool_is_converted():
    np = pytest.importorskip("numpy")
    got = bool_flag(np.bool_(True), "k")
    assert got is True and type(got) is bool
