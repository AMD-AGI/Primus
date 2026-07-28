###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Primus-Turbo MoE grouped_mm Patch

This patch mirrors ``TorchTitanPretrainTrainer.patch_torchtitan_moe`` using
the generic Primus patch system so that MoE grouped_mm integration with
Primus-Turbo can be managed declaratively.

Behavior:
    - Enabled when ``primus_turbo.use_turbo_grouped_mm`` is True in the
      TorchTitan job config.
    - Uses ``primus_turbo.use_moe_fp8`` to control FP8 MoE behavior.
    - Monkey patches ``torchtitan.models.moe.moe._run_experts_grouped_mm`` to
      delegate to Primus' grouped_mm implementation with ``use_fp8`` bound.
"""

from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.module_utils import log_rank_0

# Upstream apply_compile skips torch.compile when the grouped_mm __qualname__
# already contains this string. We reuse it to opt out of compile.
_ALREADY_PATCHED_SENTINEL = "_run_experts_grouped_mm_dynamic"


@register_patch(
    "torchtitan.primus_turbo.moe_grouped_mm",
    backend="torchtitan",
    phase="setup",
    description="Use Primus-Turbo grouped_mm for TorchTitan MoE experts",
    condition=lambda ctx: (
        get_param(ctx, "primus_turbo.enable_primus_turbo", False)
        and get_param(ctx, "primus_turbo.use_turbo_grouped_mm", False)
    ),
)
def patch_torchtitan_moe(ctx: PatchContext) -> None:
    """
    Patch TorchTitan MoE to use Primus-Turbo grouped_mm implementation.
    """
    import torchtitan.models.moe.moe

    from primus.backends.torchtitan.models.moe.moe import _run_experts_grouped_mm

    # Get MoE FP8 configuration and bind it onto the replacement function.
    use_moe_fp8 = get_param(ctx, "primus_turbo.use_moe_fp8", False)
    log_rank_0(
        "[Patch:torchtitan.primus_turbo.moe_grouped_mm] " f"Set MoE FP8 mode: {use_moe_fp8}",
    )

    # The turbo grouped_mm kernels are torch.compiler.disable'd, so letting
    # upstream compile our wrapper under fullgraph traces into them and hard-errors
    # (dynamo gb0098). Decorating the wrapper with torch.compiler.disable does not
    # help (torch.compile unwraps it via innermost_fn). Instead we name the wrapper
    # with the sentinel so apply_compile treats it as already patched and skips
    # compile, keeping the turbo kernels in eager as designed.
    def _run_experts_grouped_mm_dynamic(*args, **kwargs):
        kwargs.setdefault("use_fp8", use_moe_fp8)
        return _run_experts_grouped_mm(*args, **kwargs)

    _run_experts_grouped_mm_dynamic.__qualname__ = _ALREADY_PATCHED_SENTINEL
    _run_experts_grouped_mm_dynamic.__name__ = _ALREADY_PATCHED_SENTINEL

    torchtitan.models.moe.moe._run_experts_grouped_mm = _run_experts_grouped_mm_dynamic

    log_rank_0(
        "[Patch:torchtitan.primus_turbo.moe_grouped_mm] "
        "Successfully patched torchtitan moe with Primus-Turbo grouped_mm."
    )
