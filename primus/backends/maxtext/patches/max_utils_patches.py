###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText max_utils Patches

Replaces MaxText's ``print_system_information`` and ``save_device_information``
with Primus implementations that provide richer device logging and local-filesystem
support for device-info JSON dumps.
"""

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0


@register_patch(
    patch_id="maxtext.max_utils",
    backend="maxtext",
    phase="setup",
    description="Replace MaxText max_utils helpers with Primus implementations",
    condition=lambda ctx: True,  # Always enabled
)
def patch_max_utils(ctx: PatchContext) -> None:
    """
    Monkey-patch ``MaxText.max_utils.print_system_information`` and
    ``MaxText.max_utils.save_device_information`` with Primus versions.
    """
    log_rank_0("[Patch:maxtext.max_utils] Patching MaxText max_utils...")

    import MaxText.max_utils as orig_max_utils

    from primus.backends.maxtext.max_utils import (
        print_system_information,
        save_device_information,
    )

    orig_max_utils.print_system_information = print_system_information
    orig_max_utils.save_device_information = save_device_information

    warning_rank_0("[Patch:maxtext.max_utils] max_utils patched successfully.")
