###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest

from primus.backends.nemo_automodel import options


@pytest.fixture(autouse=True)
def _default_options():
    """Every test starts from the defaults: settings are process-global."""
    options.load({})
    yield
    options.load({})


@pytest.fixture
def set_option():
    """``set_option("primus_turbo.fp8_linear", True)``; ``None`` unsets."""
    sections = {}

    def _set(path, value):
        section, key = path.split(".", 1)
        sections.setdefault(section, {})[key] = value
        options.load(sections)

    return _set
