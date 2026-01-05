###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron build_model / get_model patches.

This module contains patches that modify Megatron's model construction
behavior to better integrate with Primus.

Current patch:
    - Disable the second DDP construction inside ``torch.cuda.stream()``
      in ``megatron.training.training.get_model`` by temporarily replacing
      ``torch.cuda.stream`` with a no-op context manager while calling
      the original ``get_model``.
"""

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.training.training.get_model.disable_second_ddp",
    backend="megatron",
    phase="before_train",
    description=(
        "Monkey patch megatron.training.training.get_model to disable the "
        "second DDP construction inside torch.cuda.stream()."
    ),
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
