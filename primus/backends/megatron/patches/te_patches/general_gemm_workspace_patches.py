###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Transformer Engine general_gemm workspace patches.

Patch Megatron's TE wrapper to use Primus' workspace helper for
``te_general_gemm``. This keeps the compatibility fix inside Primus while
avoiding direct edits under ``third_party/Megatron-LM``.
"""

import inspect

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


def _te_general_gemm_workspace_mode():
    """
    Detect how the current TE ``general_gemm`` expects workspace handling.

    Returns:
      - ``"bind_primus"`` when ``workspace`` is part of the callable signature.
      - ``"disable"`` when ``workspace`` is not accepted.
      - ``"unknown"`` when the signature cannot be inspected.
    """
    try:
        from megatron.core.extensions import transformer_engine as te_ext
    except ImportError:
        return None

    general_gemm = getattr(te_ext, "general_gemm", None)
    if general_gemm is None:
        return None

    try:
        signature = inspect.signature(general_gemm)
    except (TypeError, ValueError):
        return "unknown"

    return "bind_primus" if "workspace" in signature.parameters else "disable"


def _needs_te_general_gemm_workspace_patch(_ctx: PatchContext) -> bool:
    """Check whether Megatron exposes the workspace helper used by te_general_gemm."""
    try:
        from megatron.core.extensions import transformer_engine as te_ext
    except ImportError:
        return False

    return hasattr(te_ext, "_get_workspace") and getattr(te_ext, "te_general_gemm", None) is not None


@register_patch(
    "megatron.te.general_gemm_workspace_helper",
    backend="megatron",
    phase="before_train",
    description="Patch Megatron te_general_gemm workspace helper for TE compatibility",
    condition=_needs_te_general_gemm_workspace_patch,
)
def patch_te_general_gemm_workspace_helper(ctx: PatchContext):
    """
    Patch workspace handling in Megatron's ``te_general_gemm`` wrapper.

    Some TE revisions still require a ``workspace`` argument, while newer
    ROCm-side variants may allocate it internally and reject the kwarg. Bind
    Primus' helper only when the current TE signature still exposes the
    parameter; otherwise disable workspace injection entirely.
    """
    del ctx

    from megatron.core.extensions import transformer_engine as te_ext

    mode = _te_general_gemm_workspace_mode()
    if mode == "disable":
        te_ext._get_workspace = None
        log_rank_0(
            "[Patch:megatron.te.general_gemm_workspace_helper] "
            "TE general_gemm does not expose a workspace parameter; disabled workspace injection"
        )
        return

    # Importing the Primus workspace helper transitively pulls in TE PyTorch
    # extensions (and on ROCm images, ``aiter``/``csrc``). If that chain is
    # broken in the runtime image we still want a graceful fallback rather
    # than a hard patch failure that leaves Megatron's stock helper untouched
    # in some half-applied state.
    try:
        from primus.backends.transformer_engine.pytorch.module.base import (
            get_workspace as workspace_helper,
        )
        helper_source = "Primus"
    except (ImportError, ModuleNotFoundError) as exc:
        # The TE-native helper is the upstream fallback Megatron itself uses
        # when its own optional import succeeds. Reusing it keeps
        # ``te_general_gemm`` callable even when the Primus extension chain
        # is unavailable; setting ``_get_workspace`` to None would instead
        # skip the required ``workspace`` kwarg and crash inside TE.
        try:
            from transformer_engine.pytorch.module.base import (
                get_workspace as workspace_helper,
            )
            helper_source = f"TE (Primus helper unavailable: {exc})"
        except (ImportError, ModuleNotFoundError) as te_exc:
            te_ext._get_workspace = None
            log_rank_0(
                "[Patch:megatron.te.general_gemm_workspace_helper] "
                f"No workspace helper available (Primus: {exc}; TE: {te_exc}); "
                "disabled workspace injection"
            )
            return

    te_ext._get_workspace = workspace_helper

    if mode == "unknown":
        log_rank_0(
            "[Patch:megatron.te.general_gemm_workspace_helper] "
            f"Could not inspect TE general_gemm signature; rebound workspace helper to {helper_source} get_workspace"
        )
    else:
        log_rank_0(
            "[Patch:megatron.te.general_gemm_workspace_helper] "
            f"Rebound workspace helper for megatron.core.extensions.transformer_engine.te_general_gemm using {helper_source} get_workspace"
        )
