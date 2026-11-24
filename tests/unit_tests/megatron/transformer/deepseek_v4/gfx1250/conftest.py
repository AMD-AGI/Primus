###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""gfx1250 (MI455X) DeepSeek-V4 kernel tests.

The parent conftest skips every test in ``deepseek_v4/`` unless the GPU is an MI355X. The tests in this directory
target gfx1250 instead, so this conftest replaces that gate by a gfx1250 one (the individual tests add the Gluon /
Turbo requirements they need through ``helpers``).
"""

import pytest

from .helpers import is_gfx1250


@pytest.fixture(autouse=True)
def _require_mi355x():
    """Override of the parent directory's MI355X gate: these tests need gfx1250 instead."""
    if not is_gfx1250():
        pytest.skip("requires a gfx1250 GPU")
