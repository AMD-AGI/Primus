###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark under torch FSDP2 (``use_torch_fsdp2``).

Muse Spark keeps the router weight and the residual gates fp32 in a bf16 model.
Primus's FSDP2 wrapper shards them as their own fp32 groups; this patch lets
Megatron's fp16 optimizer take their gradients (FSDP2 sets ``.grad``, not ``main_grad``).
"""

from primus.backends.megatron.core.models.muse_spark.model_types import (
    is_muse_spark_model_type,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


@register_patch(
    "megatron.muse_spark.fsdp2_fp32_param_grads",
    backend="megatron",
    phase="before_train",
    description=(
        "FSDP2: fp32 params kept inside a bf16 model (no main_grad) hand their sharded .grad "
        "to Float16OptimizerWithFloat16Params."
    ),
    condition=lambda ctx: is_muse_spark_model_type(ctx) and getattr(get_args(ctx), "use_torch_fsdp2", False),
)
def patch_muse_spark_fsdp2_fp32_param_grads(ctx: PatchContext):
    """``Float16OptimizerWithFloat16Params`` reads ``main_grad`` from its fp32 params.

    Megatron's DDP gives every param a ``main_grad``; torch FSDP2 does not, it reduce-
    scatters into the sharded param's ``.grad``. A model that keeps some params fp32 in a
    bf16 run (e.g. an fp32 router weight, sharded as its own FSDP2 group) then fails with
    "'DTensor' object has no attribute 'main_grad'". Their ``.grad`` already holds the
    reduced gradient, so upstream's ``grad = main_grad`` step is skipped for exactly
    those params: FSDP2-sharded ``DTensor`` params without ``main_grad``. Any other
    param without ``main_grad`` is passed to upstream unchanged and fails there.
    """
    from megatron.core.optimizer import optimizer as opt_module

    cls = opt_module.Float16OptimizerWithFloat16Params
    if getattr(cls._copy_model_grads_to_main_grads, "_primus_fsdp2_fp32", False):
        return
    from torch.distributed.tensor import DTensor

    orig = cls._copy_model_grads_to_main_grads

    def _is_fsdp2_grad_param(p):
        return isinstance(p, DTensor) and not hasattr(p, "main_grad")

    def _copy_model_grads_to_main_grads(self):
        # Upstream ends with ``param.grad = param.main_grad`` for every fp32 param.
        # FSDP2 params have no main_grad: their reduce-scattered grad is already in
        # ``.grad``. Hide only those params from upstream's loop, for this call only.
        saved = self.fp32_from_fp32_groups
        self.fp32_from_fp32_groups = [[p for p in group if not _is_fsdp2_grad_param(p)] for group in saved]
        try:
            return orig(self)
        finally:
            self.fp32_from_fp32_groups = saved

    _copy_model_grads_to_main_grads._primus_fsdp2_fp32 = True
    cls._copy_model_grads_to_main_grads = _copy_model_grads_to_main_grads
    log_rank_0(
        "[Patch:megatron.muse_spark.fsdp2_fp32_param_grads] FSDP2 (DTensor) fp32 params without main_grad keep their .grad"
    )
