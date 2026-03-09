###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText Checkpoint Patches

Replaces MaxText's ``create_orbax_checkpoint_manager`` with a Primus
implementation that adds local-filesystem support and additional logging.
"""

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0


@register_patch(
    patch_id="maxtext.checkpoint",
    backend="maxtext",
    phase="setup",
    description="Replace MaxText create_orbax_checkpoint_manager with Primus implementation",
    condition=lambda ctx: True,  # Always enabled
)
def patch_checkpointing(ctx: PatchContext) -> None:
    """
    Monkey-patch ``MaxText.checkpointing.create_orbax_checkpoint_manager``
    with the Primus version that supports local storage.
    """
    log_rank_0("[Patch:maxtext.checkpoint] Patching MaxText checkpointing...")

    import MaxText.checkpointing as orig_checkpointing

    from primus.backends.maxtext.checkpointing import (
        create_orbax_checkpoint_manager,
    )

    orig_checkpointing.create_orbax_checkpoint_manager = create_orbax_checkpoint_manager

    warning_rank_0("[Patch:maxtext.checkpoint] checkpointing patched successfully.")
