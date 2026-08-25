###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxDiffusion Logger Patch

MaxDiffusion's ``max_logging.log`` is a bare ``print``. Primus rebinds
``builtins.print`` to a DEBUG-level logger call (see
``primus.core.runtime.logging``), so without this patch every MaxDiffusion
message -- including the per-step ``completed step: N, ... loss: ...`` lines --
is emitted at DEBUG and therefore dropped by the console sink, which runs at
``stderr_sink_level`` (INFO by default). The lines survive only in
``debug.log``, which makes a healthy run look like it produced no output.

This is the MaxDiffusion counterpart of
``primus.backends.maxtext.patches.logger_patches``.
"""

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import error_rank_0, log_rank_0


@register_patch(
    patch_id="maxdiffusion.logger",
    backend="maxdiffusion",
    phase="setup",
    description="Redirect MaxDiffusion logger to Primus unified logger",
    condition=lambda ctx: True,  # Always enabled
)
def patch_maxdiffusion_logger(ctx: PatchContext) -> None:
    """Route ``maxdiffusion.max_logging.log`` through Primus at INFO level."""
    del ctx

    try:
        from maxdiffusion import max_logging
    except ImportError as exc:  # noqa: BLE001 - never abort a run over logging
        error_rank_0(f"[Patch:maxdiffusion.logger] Failed to import maxdiffusion.max_logging: {exc!r}")
        return

    if not hasattr(max_logging, "log"):
        error_rank_0("[Patch:maxdiffusion.logger] maxdiffusion.max_logging has no 'log' function.")
        return

    # Rebind before logging anything: the whole point of this patch is to make
    # backend output visible, so it must not be skipped by a failure in the very
    # logging path it is repairing.
    #
    # Every MaxDiffusion call site is module-qualified (`max_logging.log(...)`),
    # so rebinding the module attribute covers all of them. log_rank_0 resolves
    # the sink at call time and reports the calling frame, so step lines are
    # attributed to the trainer rather than to max_logging itself.
    max_logging.log = log_rank_0

    log_rank_0("[Patch:maxdiffusion.logger] MaxDiffusion logger patched successfully.")
