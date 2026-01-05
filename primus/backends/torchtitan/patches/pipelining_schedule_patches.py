###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan pipelining schedules compatibility patch.

This patch mirrors ``TorchTitanPretrainTrainer.patch_torch_pipelining_schedules``
using the generic Primus patch system so that missing
``ScheduleDualPipeV`` symbols in ``torch.distributed.pipelining.schedules``
can be handled in a backend-agnostic way.
"""

from primus.core.patches import PatchContext, register_patch


@register_patch(
    "torchtitan.torch.pipelining_schedules_dualpipev",
    backend="torchtitan",
    phase="setup",
    description=(
        "Ensure torch.distributed.pipelining.schedules.ScheduleDualPipeV exists, "
        "installing a fallback alias if necessary"
    ),
)
def patch_torch_pipelining_schedules(ctx: PatchContext) -> None:  # noqa: ARG001
    """
    Ensure torch.distributed.pipelining.schedules.ScheduleDualPipeV exists.

    If this class is missing in the current PyTorch build (common in ROCm 7.0 / 2.9),
    we create a fallback alias that inherits from Schedule1F1B or ScheduleGPipe.
    This prevents ImportError in TorchTitan pipeline modules.
    """
    from primus.core.utils.logger import _logger as primus_logger

    try:
        import torch.distributed.pipelining.schedules as sched
    except Exception as e:  # pragma: no cover - defensive
        primus_logger.warning(f"[PrimusPatch][Pipe] failed to import schedules: {e}")
        return

    # Check if DualPipeV is already provided
    if hasattr(sched, "ScheduleDualPipeV"):
        primus_logger.info("[PrimusPatch][Pipe] ScheduleDualPipeV available, no patch needed.")
        return  # No patch needed

    # Pick a safe fallback
    fallback = getattr(sched, "Schedule1F1B", None) or getattr(sched, "ScheduleGPipe", None)

    if fallback is None:
        primus_logger.warning(
            "[PrimusPatch][Pipe] No pipeline schedule available; " "pipeline parallel may be unsupported."
        )
        return

    # Define the fallback class with identical signature
    class ScheduleDualPipeV(fallback):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):  # noqa: ANN001, ANN002
            primus_logger.warning(
                f"[PrimusPatch][Pipe] ScheduleDualPipeV not found, using fallback {fallback.__name__}. "
                "This is a temporary compatibility patch; functionality may differ "
                "from the official DualPipeV."
            )
            super().__init__(*args, **kwargs)

    # Inject into torch namespace
    setattr(sched, "ScheduleDualPipeV", ScheduleDualPipeV)
    primus_logger.warning(f"[PrimusPatch][Pipe] Installed fallback: ScheduleDualPipeV -> {fallback.__name__}")
