###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Author: Hu, Botao
#
# See LICENSE for license information.
###############################################################################

"""
Megatron checkpoint-CONVERSION patches (``phase="convert"``).

These run in-process inside the native (bridge-free) HF->Megatron converter
entrypoint, BEFORE it builds the mcore model, so they take effect for the
single-process conversion without touching the Megatron-LM submodule.

Currently one patch: neutralise the legacy CUDA-only ``fused_kernels`` build on
ROCm. This replaces the previous in-submodule edit to
``megatron/legacy/fused_kernels/__init__.py``. The other three former submodule
edits (``schema_core.py`` QK-layernorm keys, ``saver_base.py`` TE-metadata /
per-expert placement, ``loader_llama_mistral.py`` flag + fused-kernels guard) are
NOT needed by the native converters: those live on the ``tools/checkpoint``
loader<->saver queue path, which the Primus converters bypass entirely by
building the mcore model directly and mapping weights onto it.
"""

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _is_rocm() -> bool:
    """True on a HIP/ROCm torch build (or when nvcc/CUDA_HOME is absent)."""
    try:
        import torch

        if getattr(torch.version, "hip", None):
            return True
        from torch.utils import cpp_extension

        return cpp_extension.CUDA_HOME is None
    except Exception:
        return False


@register_patch(
    "megatron.checkpoint_convert.fused_kernels_rocm_noop",
    backend="megatron",
    phase="convert",
    priority=10,
    description=(
        "Skip the legacy CUDA-only fused_kernels build on ROCm during checkpoint "
        "conversion (weights are only reshaped on CPU; the kernels are never run)."
    ),
    condition=lambda ctx: _is_rocm(),
)
def patch_fused_kernels_noop_on_rocm(ctx: PatchContext):
    """Wrap ``megatron.legacy.fused_kernels.load`` so it is a no-op on ROCm.

    The legacy ``load()`` shells out to ``nvcc`` (via ``CUDA_HOME``) to JIT-build
    CUDA-only kernels and crashes when ``CUDA_HOME``/``nvcc`` is absent (the norm
    on a ROCm image). Checkpoint conversion never executes these kernels, so on a
    HIP/ROCm torch build we short-circuit the build; on CUDA the original build
    still runs.
    """
    import megatron.legacy.fused_kernels as fk

    if getattr(fk.load, "_primus_rocm_wrapped", False):
        log_rank_0(
            "[Patch:megatron.checkpoint_convert.fused_kernels_rocm_noop]   "
            "already wrapped; skipping"
        )
        return

    _orig_load = fk.load

    def _rocm_safe_load(args):
        try:
            import torch

            from torch.utils import cpp_extension

            if getattr(torch.version, "hip", None) or cpp_extension.CUDA_HOME is None:
                log_rank_0(
                    "[Patch:megatron.checkpoint_convert.fused_kernels_rocm_noop]   "
                    "ROCm/HIP detected -> skipping legacy fused_kernels build"
                )
                return
        except Exception:
            pass
        return _orig_load(args)

    _rocm_safe_load._primus_rocm_wrapped = True
    fk.load = _rocm_safe_load
    log_rank_0(
        "[Patch:megatron.checkpoint_convert.fused_kernels_rocm_noop]   "
        "Patched megatron.legacy.fused_kernels.load -> ROCm no-op wrapper"
    )
