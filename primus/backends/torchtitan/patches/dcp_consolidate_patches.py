###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan DCP safetensors consolidation compatibility patch.

This patch mirrors ``TorchTitanPretrainTrainer.patch_torch_dcp_consolidate``
using the generic Primus patch system so that missing
``consolidate_safetensors_files_on_every_rank`` symbols in
``torch.distributed.checkpoint._consolidate_hf_safetensors`` can be handled
in a backend-agnostic way.
"""

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "torchtitan.torch.dcp_consolidate_fallback",
    backend="torchtitan",
    phase="setup",
    description=(
        "Provide a fallback consolidate_safetensors_files_on_every_rank "
        "to avoid ImportError in older torch builds"
    ),
)
def patch_torch_dcp_consolidate(ctx: PatchContext) -> None:  # noqa: ARG001
    """
    Monkey patch for torch.distributed.checkpoint._consolidate_hf_safetensors
    when current torch build does not export consolidate_safetensors_files_on_every_rank.

    This avoids ImportError in TorchTitan when last_save_in_hf=True.
    """
    import sys
    import types
    import warnings

    mod_name = "torch.distributed.checkpoint._consolidate_hf_safetensors"
    func_name = "consolidate_safetensors_files_on_every_rank"

    log_rank_0(
        "[Patch:torchtitan.torch.dcp_consolidate_fallback] "
        f"Checking for {func_name} in {mod_name}...",
    )

    try:
        mod = __import__(mod_name, fromlist=["*"])
        if hasattr(mod, func_name):
            log_rank_0(
                "[Patch:torchtitan.torch.dcp_consolidate_fallback] "
                f"{func_name} available in {mod_name}, no patch needed.",
            )
            return  # OK, torch build supports it
        else:
            log_rank_0(
                "[Patch:torchtitan.torch.dcp_consolidate_fallback] "
                f"{func_name} NOT found in {mod_name}, will install fallback.",
            )
    except Exception as e:
        log_rank_0(
            "[Patch:torchtitan.torch.dcp_consolidate_fallback] "
            f"Failed to import {mod_name}: {e}, will install fallback.",
        )
        # Fall through to install dummy module/function

    # Patch missing module/function
    dummy_mod = types.ModuleType(mod_name)

    def _warn_and_skip(*args, **kwargs):  # noqa: ANN001, ANN002
        warnings.warn(
            "[PrimusPatch][DCP] Current PyTorch build does not support "
            f"{mod_name}.{func_name}; safetensors export will be skipped.",
            UserWarning,
        )
        return None

    setattr(dummy_mod, func_name, _warn_and_skip)
    sys.modules[mod_name] = dummy_mod

    log_rank_0(
        "[Patch:torchtitan.torch.dcp_consolidate_fallback][WARN] "
        f"Installed fallback for missing {mod_name}.{func_name}, "
        "HuggingFace safetensors export will be disabled.",
    )
