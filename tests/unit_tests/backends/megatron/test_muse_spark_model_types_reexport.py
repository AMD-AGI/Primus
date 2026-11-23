###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest


def test_muse_spark_model_types_re_exports_the_moved_names():
    """model_types (where base defined them) still exports bool_flag, TRUE_TEXT and FALSE_TEXT."""
    pytest.importorskip("megatron.core")
    from primus.backends.megatron.core.models.muse_spark import model_types
    from primus.core.utils import flag_utils

    for name in ("bool_flag", "TRUE_TEXT", "FALSE_TEXT"):
        assert name in model_types.__all__
        assert getattr(model_types, name) is getattr(flag_utils, name)
