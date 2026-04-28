###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tier 1A — fused (residual + RMSNorm) patches.

Wraps the runtime install hook
:mod:`primus.backends.megatron.core.extensions.fused_residual_rmsnorm` in a
``@register_patch`` so it runs at the standard ``before_train`` phase via
``run_patches(...)`` instead of being explicitly installed from the trainer
entry point.

Two gates (mirroring the install hook):
  * ``PRIMUS_FUSED_RESIDUAL_NORM=1``    — V1, in-layer ADD#1+norm fusion.
  * ``PRIMUS_FUSED_RESIDUAL_NORM_V2=1`` — V2, cross-layer ADD#2 carry
    (implies V1).

Either env var enables the patch. The install hook itself further requires
``PrimusTurboRMSNorm`` (i.e. ``use_turbo_rms_norm=true``) and bails
gracefully otherwise, so it's safe to register unconditionally.
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _is_fused_residual_norm_enabled(ctx: PatchContext) -> bool:
    return _env_truthy("PRIMUS_FUSED_RESIDUAL_NORM") or _env_truthy("PRIMUS_FUSED_RESIDUAL_NORM_V2")


@register_patch(
    "megatron.turbo.fused_residual_norm",
    backend="megatron",
    phase="before_train",
    description=(
        "Fuse residual+add into PrimusTurboRMSNorm via Triton "
        "triton_rmsnorm_residual; gated by PRIMUS_FUSED_RESIDUAL_NORM(_V2)."
    ),
    condition=_is_fused_residual_norm_enabled,
    # Run after megatron.turbo.rms_norm so PrimusTurboRMSNorm is in place
    # before we extend its forward signature.
    priority=60,
)
def patch_fused_residual_norm(ctx: PatchContext):
    """Install the fused residual+RMSNorm runtime monkeypatch."""
    from primus.backends.megatron.core.extensions import fused_residual_rmsnorm

    log_rank_0(
        "[Patch:megatron.turbo.fused_residual_norm] Installing fused "
        "residual+RMSNorm (V2={v2}, V1={v1})".format(
            v1=_env_truthy("PRIMUS_FUSED_RESIDUAL_NORM"),
            v2=_env_truthy("PRIMUS_FUSED_RESIDUAL_NORM_V2"),
        )
    )
    ok = fused_residual_rmsnorm.install()
    if ok:
        log_rank_0("[Patch:megatron.turbo.fused_residual_norm]   install() returned True")
    else:
        log_rank_0(
            "[Patch:megatron.turbo.fused_residual_norm]   install() returned False "
            "(precondition not met; e.g. use_turbo_rms_norm=false)"
        )
