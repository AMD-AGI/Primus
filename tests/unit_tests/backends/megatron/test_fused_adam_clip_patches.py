###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest

from primus.backends.megatron.patches.te_patches.fused_adam_clip_patches import (
    _clip_coeff,
)


@pytest.mark.parametrize(
    ("total_norm", "max_norm", "expected"),
    [
        (0.25, 1.0, 1.0),
        (1.0, 1.0, 1.0 / 1.000001),
        (2.0, 1.0, 1.0 / 2.000001),
        (8.0, 2.0, 2.0 / 8.000001),
    ],
)
def test_clip_coeff(total_norm, max_norm, expected):
    assert _clip_coeff(total_norm, max_norm) == pytest.approx(expected)
