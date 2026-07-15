###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron build_model / get_model patches.

This module contains patches that modify Megatron's model construction
behavior to better integrate with Primus.

Current patches:
    - Disable the second DDP construction inside ``torch.cuda.stream()``
      in ``megatron.training.training.get_model`` by temporarily replacing
      ``torch.cuda.stream`` with a no-op context manager while calling
      the original ``get_model``.
    - Skip Float16Module wrapping when using FSDP2 FP32 param optimizer (model
      parameters must stay in FP32 for FSDP2's MixedPrecisionPolicy).
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


@register_patch(
    "megatron.training.training.get_model.disable_second_ddp",
    backend="megatron",
    phase="before_train",
    description=(
        "Monkey patch megatron.training.training.get_model to disable the "
        "second DDP construction inside torch.cuda.stream()."
    ),
    condition=lambda ctx: not getattr(get_args(ctx), "disable_build_model_patches", False),
)
def patch_megatron_get_model_disable_second_ddp(ctx: PatchContext) -> None:
    """
    Patch ``megatron.training.training.get_model`` to avoid a second DDP wrap,
    mirroring the original ad-hoc implementation in ``primus/pretrain.py``.
    """
    import megatron.training.training as training  # type: ignore
    import torch

    original_get_model = training.get_model

    # Avoid double patching if we've already wrapped get_model (lightweight guard).
    if getattr(original_get_model, "_primus_disable_second_ddp", False):
        return

    def _patched_get_model(*args, **kwargs):
        _orig_stream_ctx = torch.cuda.stream

        class _DummyCtx:
            def __enter__(self):
                return None

            def __exit__(self, *exc_info):
                return False

        def _noop_stream(*_a, **_k):
            return _DummyCtx()

        torch.cuda.stream = _noop_stream
        try:
            return original_get_model(*args, **kwargs)
        finally:
            torch.cuda.stream = _orig_stream_ctx

    setattr(_patched_get_model, "_primus_disable_second_ddp", True)
    training.get_model = _patched_get_model
    log_rank_0("[Patch:megatron.get_model] Disabled second DDP via torch.cuda.stream no-op wrapper")


@register_patch(
    "megatron.training.training.skip_float16_module",
    backend="megatron",
    phase="before_train",
    description=(
        "Skip Float16Module wrapping when using FSDP2 FP32 param optimizer. "
        "Model parameters must stay FP32 for FSDP2's MixedPrecisionPolicy to handle casting."
    ),
    priority=40,
    condition=lambda ctx: (
        getattr(get_args(ctx), "use_fsdp2_fp32_param_optimizer", False)
        and getattr(get_args(ctx), "use_torch_fsdp2", False)
        and getattr(get_args(ctx), "bf16", False)
    ),
)
def patch_skip_float16_module(ctx: PatchContext) -> None:
    """Replace Float16Module with a passthrough wrapper that preserves
    the attributes Megatron's training loop expects (module, config,
    vp_size, vp_stage, pg_collection) but does NOT convert parameters
    to BF16/FP16.

    This is necessary because FSDP2's MixedPrecisionPolicy handles
    the FP32->BF16 casting during forward/backward, and Float16Module
    would convert parameters to BF16 before FSDP2 wrapping, defeating
    the purpose of FP32 parameter storage.
    """
    from megatron.core.transformer.module import MegatronModule

    class _FSDP2PassthroughModule(MegatronModule):
        """Drop-in replacement for Float16Module that keeps params in FP32."""

        def __init__(self, config, module):
            super().__init__(config)
            self.config = config
            self.add_module("module", module)
            self.vp_size = config.virtual_pipeline_model_parallel_size
            self.vp_stage = getattr(module, "vp_stage", None)
            self.pg_collection = getattr(module, "pg_collection", None)

        def set_input_tensor(self, input_tensor):
            return self.module.set_input_tensor(input_tensor)

        def forward(self, *inputs, **kwargs):
            return self.module(*inputs, **kwargs)

    import megatron.core.transformer.module as module_mod
    import megatron.training.training as training_mod

    module_mod.Float16Module = _FSDP2PassthroughModule
    training_mod.Float16Module = _FSDP2PassthroughModule
    log_rank_0(
        "[Patch:skip_float16_module] Replaced Float16Module with FP32 passthrough "
        "(FSDP2 MixedPrecisionPolicy handles BF16 casting)"
    )
