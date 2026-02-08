###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron runtime hook patches.

These patches are applied early (priority=0) to ensure Megatron runtime
behaves correctly on ROCm / AMD GPUs.
"""

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.runtime.skip_compile_dependencies",
    backend="megatron",
    phase="before_train",
    priority=0,
    description="Skip Megatron CUDA fused kernel compilation (ROCm compatibility)",
)
def patch_skip_compile_dependencies(ctx: PatchContext) -> None:
    """
    Patch Megatron-LM runtime to skip CUDA fused kernel compilation.

    On ROCm, Megatron's CUDA-specific compilation path is not applicable.
    This patch overrides `megatron.training.initialize._compile_dependencies`
    with a no-op implementation.
    """
    try:
        import megatron.training.initialize as megatron_initialize  # type: ignore

        if not hasattr(megatron_initialize, "_compile_dependencies"):
            log_rank_0(
                "[Patch:megatron.runtime.skip_compile_dependencies] "
                "WARNING: megatron.training.initialize has no _compile_dependencies; skipping"
            )
            return

        if getattr(megatron_initialize._compile_dependencies, "_primus_patched", False):
            return

        log_rank_0(
            "[Patch:megatron.runtime.skip_compile_dependencies] "
            "Patching _compile_dependencies to skip CUDA kernel compilation"
        )

        def _skip_compile_dependencies() -> None:
            log_rank_0(
                "    Skipped _compile_dependencies() because CUDA kernels are not compatible with ROCm"
            )

        _skip_compile_dependencies._primus_patched = True  # type: ignore[attr-defined]
        megatron_initialize._compile_dependencies = _skip_compile_dependencies

        log_rank_0(
            "[Patch:megatron.runtime.skip_compile_dependencies] "
            "Patched _compile_dependencies to skip CUDA kernel compilation"
        )
    except (ImportError, AttributeError) as exc:
        log_rank_0(
            "[Patch:megatron.runtime.skip_compile_dependencies] "
            f"WARNING: Failed to patch Megatron-LM runtime hooks: {exc}"
        )
