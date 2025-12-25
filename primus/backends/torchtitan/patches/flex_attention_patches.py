###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Flex Attention AuxOutput Compatibility Patch

This patch mirrors ``TorchTitanPretrainTrainer.patch_torch_flex_attention_auxoutput``
using the generic Primus patch system so that missing ``AuxOutput`` symbols in
``torch.nn.attention.flex_attention`` can be handled in a backend-agnostic way.

Behavior:
    - Attempts to import ``torch.nn.attention.flex_attention`` as ``flex_mod``.
    - If ``flex_mod.AuxOutput`` already exists, does nothing.
    - Otherwise, injects a lightweight ``AuxOutput`` dataclass stub to keep
      TorchTitan imports working on older PyTorch builds.
"""

from primus.core.patches import PatchContext, register_patch


@register_patch(
    "torchtitan.torch.flex_attention_auxoutput",
    backend="torchtitan",
    phase="setup",
    description="Ensure torch.nn.attention.flex_attention has an AuxOutput symbol",
)
def patch_torch_flex_attention_auxoutput(ctx: PatchContext) -> None:  # noqa: ARG001
    """
    Ensure torch.nn.attention.flex_attention has an AuxOutput symbol.
    """
    from primus.core.utils.logger import _logger as primus_logger

    try:
        import torch.nn.attention.flex_attention as flex_mod
    except Exception as e:  # pragma: no cover - defensive
        primus_logger.warning(f"[PrimusPatch][FlexAttn] flex_attention import failed: {e}")
        return

    # If AuxOutput already exists, nothing to do.
    if hasattr(flex_mod, "AuxOutput"):
        primus_logger.info("[PrimusPatch][FlexAttn] AuxOutput available, no patch needed.")
        return

    primus_logger.warning(
        "[PrimusPatch][FlexAttn] AuxOutput not found. "
        "This torch build predates the new debug/profiling return type. "
        "Injecting a lightweight stub so Titan model imports can succeed."
    )

    from dataclasses import dataclass

    import torch

    @dataclass
    class _AuxOutput:
        attn_probs: torch.Tensor = torch.empty(0)
        block_mask: torch.Tensor | None = None
        stats: dict | None = None
        extra: dict | None = None

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    setattr(flex_mod, "AuxOutput", _AuxOutput)
    primus_logger.info(
        "[PrimusPatch][FlexAttn] Injected fallback AuxOutput stub " "(Titan does not rely on this)."
    )
