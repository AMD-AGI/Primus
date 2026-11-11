###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark under torch FSDP2 (``use_torch_fsdp2``).

Muse Spark keeps the router weight and the residual gates fp32 in a bf16 model.
``megatron.muse_spark.fsdp2_dtype_islands`` shards them as their own fp32 groups
(``muse_spark/fsdp2_islands.py``, through the ``fully_shard`` Primus's FSDP2 wrapper calls);
``megatron.muse_spark.fsdp2_fp32_param_grads`` lets Megatron's fp16 optimizer take their
gradients (FSDP2 sets ``.grad``, not ``main_grad``). Both run for Muse Spark with
``use_torch_fsdp2`` only.

Both replace a process-wide function. Each replacement acts only on Muse Spark objects when
it is called, so a later non-Muse model in the same process takes upstream's path: the
``fully_shard`` wrapper adds groups only for modules that set ``fsdp2_dtype_island`` (only
Muse Spark's classes do; otherwise it calls the original ``fully_shard`` alone), and the
optimizer replacement calls upstream's method unchanged unless the optimizer holds a Muse
Spark parameter (``holds_muse_spark_params``, as the optimizer-checkpoint patches do).
"""

from primus.backends.megatron.core.models.muse_spark.model_types import (
    bool_flag,
    is_muse_spark_model_type,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

# per optimizer: (its fp32_from_fp32_groups, the groups without FSDP2 no-main_grad params, or
# None when it holds no Muse Spark param)
_CACHE_ATTR = "_primus_muse_spark_fsdp2_groups"


def _is_muse_spark_fsdp2(ctx: PatchContext) -> bool:
    # bool_flag: an env-substituted "false" / "0" is off (bool("false") would be True)
    return is_muse_spark_model_type(ctx) and bool_flag(
        getattr(get_args(ctx), "use_torch_fsdp2", False), "use_torch_fsdp2"
    )


@register_patch(
    "megatron.muse_spark.fsdp2_dtype_islands",
    backend="megatron",
    phase="before_train",
    description="FSDP2: shard Muse Spark's fp32 router / residual gates as their own fp32 groups.",
    condition=_is_muse_spark_fsdp2,
)
def patch_muse_spark_fsdp2_dtype_islands(ctx: PatchContext):
    """Wrap the ``fully_shard`` of Primus's FSDP2 wrapper module (a module global it
    calls per wrapped submodule and on the root) with ``with_dtype_islands``."""
    del ctx
    from primus.backends.megatron.core.distributed import (
        torch_fully_sharded_data_parallel as fsdp_mod,
    )
    from primus.backends.megatron.core.models.muse_spark.fsdp2_islands import (
        with_dtype_islands,
    )

    if getattr(fsdp_mod.fully_shard, "_primus_dtype_islands", False):
        return
    fsdp_mod.fully_shard = with_dtype_islands(fsdp_mod.fully_shard)
    log_rank_0(
        "[Patch:megatron.muse_spark.fsdp2_dtype_islands] FSDP2 shards the fp32 islands as their own groups"
    )


@register_patch(
    "megatron.muse_spark.fsdp2_fp32_param_grads",
    backend="megatron",
    phase="before_train",
    description=(
        "FSDP2: fp32 params kept inside a bf16 model (no main_grad) hand their sharded .grad "
        "to Float16OptimizerWithFloat16Params."
    ),
    condition=_is_muse_spark_fsdp2,
)
def patch_muse_spark_fsdp2_fp32_param_grads(ctx: PatchContext):
    """``Float16OptimizerWithFloat16Params`` reads ``main_grad`` from its fp32 params.

    Megatron's DDP gives every param a ``main_grad``; torch FSDP2 does not, it reduce-
    scatters into the sharded param's ``.grad``. A model that keeps some params fp32 in a
    bf16 run (e.g. an fp32 router weight, sharded as its own FSDP2 group) then fails with
    "'DTensor' object has no attribute 'main_grad'". Their ``.grad`` already holds the
    reduced gradient, so upstream's ``grad = main_grad`` step is skipped for exactly
    those params: FSDP2-sharded ``DTensor`` params without ``main_grad``. Any other
    param without ``main_grad`` is passed to upstream unchanged and fails there. An
    optimizer that holds no Muse Spark parameter (``holds_muse_spark_params``) takes
    upstream's method unchanged, checked at each call.
    """
    from megatron.core.optimizer import optimizer as opt_module

    cls = opt_module.Float16OptimizerWithFloat16Params
    if getattr(cls._copy_model_grads_to_main_grads, "_primus_fsdp2_fp32", False):
        return
    from torch.distributed.tensor import DTensor

    from primus.backends.megatron.patches import (
        muse_spark_optimizer_ckpt_patches as ckpt_patches,
    )

    orig = cls._copy_model_grads_to_main_grads

    def _is_fsdp2_grad_param(p):
        return isinstance(p, DTensor) and not hasattr(p, "main_grad")

    def _copy_model_grads_to_main_grads(self):
        # Upstream ends with ``param.grad = param.main_grad`` for every fp32 param.
        # FSDP2 params have no main_grad: their reduce-scattered grad is already in
        # ``.grad``. Hide only those params from upstream's loop, for this call only.
        # Another model's optimizer takes upstream's path unchanged (and fails loudly there).
        # The param groups are fixed after setup: the check and the filtered groups are
        # computed once per optimizer (per groups object), not per step.
        saved = self.fp32_from_fp32_groups
        cached = getattr(self, _CACHE_ATTR, None)
        if cached is None or cached[0] is not saved:
            filtered = None
            if ckpt_patches.holds_muse_spark_params(self):
                filtered = [[p for p in group if not _is_fsdp2_grad_param(p)] for group in saved]
            cached = (saved, filtered)
            setattr(self, _CACHE_ATTR, cached)
        if cached[1] is None:
            return orig(self)
        self.fp32_from_fp32_groups = cached[1]
        try:
            return orig(self)
        finally:
            self.fp32_from_fp32_groups = saved

    _copy_model_grads_to_main_grads._primus_fsdp2_fp32 = True
    cls._copy_model_grads_to_main_grads = _copy_model_grads_to_main_grads
    log_rank_0(
        "[Patch:megatron.muse_spark.fsdp2_fp32_param_grads] FSDP2 (DTensor) fp32 params without main_grad keep their .grad"
    )
