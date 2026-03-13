###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText Config Types Patch

Replaces ``MaxText.configs.types.MaxTextConfig`` with ``PrimusMaxTextConfig``
so that ``pyconfig.initialize`` (which validates through Pydantic) accepts
Primus-specific fields (wandb, turbo, heartbeat timeout, etc.) without
filtering them out or raising ``extra="forbid"`` errors.

Only active for new MaxText (>= 0.1.1) which uses the Pydantic config system.
Old MaxText (2025.x.x) is dict-based and does not need this patch.
"""

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0


@register_patch(
    patch_id="maxtext.config_types",
    backend="maxtext",
    phase="setup",
    description="Replace MaxTextConfig with PrimusMaxTextConfig (Dec version)",
    condition=lambda ctx: True,
    backend_versions=["0.1.1"],
    priority=5,
)
def patch_config_types(ctx: PatchContext) -> None:
    """
    Replace ``MaxText.configs.types.MaxTextConfig`` with ``PrimusMaxTextConfig``.

    This must run before ``pyconfig.initialize`` is called, since
    ``_prepare_for_pydantic`` filters keys based on ``MaxTextConfig.model_fields``.
    """
    log_rank_0("[Patch:maxtext.config_types] Patching MaxText config types...")

    import MaxText.configs.types as orig_config_types

    from primus.backends.maxtext.configs.types import PrimusMaxTextConfig

    if PrimusMaxTextConfig is None:
        warning_rank_0(
            "[Patch:maxtext.config_types] PrimusMaxTextConfig is None — "
            "skipping (Pydantic config system not available)"
        )
        return

    orig_config_types.MaxTextConfig = PrimusMaxTextConfig

    warning_rank_0("[Patch:maxtext.config_types] MaxTextConfig → PrimusMaxTextConfig patched successfully.")
