###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch


@pytest.fixture(autouse=True)
def _fresh_compile_caches():
    """Compile the ZAYA1 helpers afresh in every test.

    Dynamo keeps a function's compiled graphs for the whole process. Without a
    reset, the shapes of earlier tests use up a helper's recompile limit, and
    its wrapper falls back to eager for every later test.
    """
    torch._dynamo.reset()
    yield
