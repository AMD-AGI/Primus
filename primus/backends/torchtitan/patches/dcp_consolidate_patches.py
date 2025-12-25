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

    from primus.core.utils.logger import _logger as primus_logger

    mod_name = "torch.distributed.checkpoint._consolidate_hf_safetensors"
    func_name = "consolidate_safetensors_files_on_every_rank"

    try:
        mod = __import__(mod_name, fromlist=["*"])
        if hasattr(mod, func_name):
            primus_logger.info("[PrimusPatch][DCP] consolidate available, no patch needed.")
            return  # OK, torch build supports it
    except Exception:
        # Fall through to install dummy module/function
        pass

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

    primus_logger.warning(
        f"[PrimusPatch][DCP] Installed fallback for missing {mod_name}.{func_name}, "
        "HuggingFace safetensors export will be disabled."
    )
