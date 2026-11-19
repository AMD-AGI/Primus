###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Reduce the deferred output-layer wgrads with the rest of their bucket.

With ``defer_embedding_wgrad_compute`` the output layer hands DDP a placeholder
weight gradient each microbatch and the real GEMMs run in
``finish_embedding_wgrad_compute`` after the pipeline schedule. With
``overlap_grad_reduce`` the reduce-scatter for that weight's bucket has already
been issued by then (by ``grad_sync_func`` or by the DDP grad-ready hooks), so
the drained gradient is never reduced (#1330). Upstream Megatron-LM has the
same ordering.

This patch drains the buffered wgrads when the bucket group holding the output
weight starts its grad sync, and makes the schedule's later drain a no-op once
the buffers are empty.
"""

import functools
from typing import Any, Callable, List, Tuple

from primus.backends.megatron.patches._rebind import rebind_everywhere
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_KEY = "megatron.pipeline_parallel.defer_embedding_wgrad_grad_sync"
_PATCHED_ATTR = "_primus_defer_embedding_wgrad_patched"

# (config, embedding_module) of the last-stage model in the running schedule call.
_pending: List[Tuple[Any, Any]] = []


def _deferred_weight(embedding_module):
    # The weight finish_embedding_wgrad_compute drains into.
    if embedding_module.share_embeddings_and_output_weights:
        return embedding_module.output_layer.weight
    return embedding_module.shared_embedding_or_output_weight()


def _tp_group(embedding_module):
    group = getattr(embedding_module.output_layer, "tp_group", None)
    if group is None:
        from megatron.core import parallel_state

        group = parallel_state.get_tensor_model_parallel_group()
    return group


def wrap_clear_embedding_activation_buffer(orig: Callable) -> Callable:
    @functools.wraps(orig)
    def clear_embedding_activation_buffer(config, model, is_last_stage):
        embedding_module = orig(config, model, is_last_stage)
        _pending[:] = [] if embedding_module is None else [(config, embedding_module)]
        return embedding_module

    return clear_embedding_activation_buffer


def wrap_finish_embedding_wgrad_compute(orig: Callable) -> Callable:
    warned = False

    @functools.wraps(orig)
    def finish_embedding_wgrad_compute(config, embedding_module, is_last_stage, tp_group):
        nonlocal warned
        if embedding_module is not None:
            # Already drained by the bucket's grad sync; draining empty buffers would raise.
            if not embedding_module.grad_output_buffer:
                return None
            if not warned:
                warning_rank_0(
                    f"[Patch:{_PATCH_KEY}] Deferred output-layer wgrads were not drained before their "
                    "bucket's grad sync, so they are not reduced; a start_grad_sync override may be "
                    "bypassing this patch."
                )
                warned = True
        return orig(config, embedding_module, is_last_stage, tp_group)

    return finish_embedding_wgrad_compute


def wrap_start_grad_sync(orig: Callable, drain: Callable) -> Callable:
    @functools.wraps(orig)
    def start_grad_sync(self, *args, **kwargs):
        for config, embedding_module in _pending:
            if embedding_module.grad_output_buffer and _deferred_weight(embedding_module) in self.params:
                drain(config, embedding_module, True, _tp_group(embedding_module))
        return orig(self, *args, **kwargs)

    return start_grad_sync


def _is_enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return (
        getattr(args, "defer_embedding_wgrad_compute", False)
        and getattr(args, "overlap_grad_reduce", False)
        and getattr(args, "pipeline_model_parallel_size", 1) > 1
        and not getattr(args, "use_megatron_fsdp", False)
        and not getattr(args, "use_torch_fsdp2", False)
    )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Drain deferred output-layer wgrads before their DDP bucket's reduce-scatter "
        "when defer_embedding_wgrad_compute and overlap_grad_reduce are both on (#1330)."
    ),
    condition=_is_enabled,
    # Outermost start_grad_sync wrapper: drain before rccl_sdma_grad_reduce_scatter (60)
    # or grad_buffer_ownership (70) inspects or reduces the bucket.
    priority=80,
)
def patch_defer_embedding_wgrad_grad_sync(ctx: PatchContext) -> None:
    import megatron.core.pipeline_parallel.schedules as schedules
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

    if getattr(_ParamAndGradBucketGroup.start_grad_sync, _PATCHED_ATTR, False):
        return

    drain = schedules.finish_embedding_wgrad_compute
    clear = wrap_clear_embedding_activation_buffer(schedules.clear_embedding_activation_buffer)
    finish = wrap_finish_embedding_wgrad_compute(drain)
    start_grad_sync = wrap_start_grad_sync(_ParamAndGradBucketGroup.start_grad_sync, drain)
    setattr(start_grad_sync, _PATCHED_ATTR, True)

    rebound = rebind_everywhere(schedules, "clear_embedding_activation_buffer", clear)
    rebound += rebind_everywhere(schedules, "finish_embedding_wgrad_compute", finish)
    _ParamAndGradBucketGroup.start_grad_sync = start_grad_sync
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] Draining deferred output-layer wgrads before their bucket's "
        f"grad sync (rebound in {sorted(set(rebound))})"
    )
