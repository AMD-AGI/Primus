###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

import pytest

from primus.backends.diffusion.schedulers.flow_match import FlowMatchScheduler


def test_exponential_shift_requires_mu_or_dynamic_shift_len():
    with pytest.raises(ValueError, match="exponential_shift=True"):
        FlowMatchScheduler(exponential_shift=True)
