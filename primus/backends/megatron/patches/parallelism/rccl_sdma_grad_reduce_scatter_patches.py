###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Route Megatron gradient ReduceScatter directly through RCCL CE.

When ``MEGATRON_GRAD_REDUCE_BACKEND=rccl_sdma`` is explicitly selected, this
patch allocates ``grad_data`` from the same symmetric-memory pool as the direct
parameter buffer and routes distributed-optimizer ReduceScatter through the
dedicated zero-CTA process group.

The allocation wrapper is deliberately ordered outside the parameter wrapper
so each one intercepts only its own ``torch.zeros`` call. The collective
wrapper reproduces Megatron's native ReduceScatter branch with one process-group
substitution; changing only the inner collective would mismatch Megatron's
coalescing-manager group.

Gradient AllReduce, multiple distributed-optimizer instances, FP32
ReduceScatter accumulation, and unsupported buffer layouts fall back to
Megatron's original implementation.
"""

from __future__ import annotations

import functools
import inspect
import os
import threading
from unittest import mock

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

from .rccl_sdma_param_all_gather_patches import (
    rccl_sdma_param_gather_enabled,
    validate_global_cta_policy,
)

# Captured at import time, before any patch (this one or the param-gather one)
# ever mocks `torch.zeros`. Used only in the non-eager pool-allocation fallback
# below, specifically so it never resolves to another wrap's interceptor.
_REAL_TORCH_ZEROS = torch.zeros

# Runs after the param-gather patch's default priority=50, guaranteeing this
# file's __init__ wrap composes as the OUTER layer (see module docstring).
_PRIORITY = 60

BACKEND_ENV = "MEGATRON_GRAD_REDUCE_BACKEND"
RCCL_SDMA_BACKEND = "rccl_sdma"


def rccl_sdma_grad_reduce_enabled(_ctx: PatchContext | None = None) -> bool:
    """Return whether the dedicated RCCL CE gradient path was selected."""
    return os.getenv(BACKEND_ENV, "").strip().lower() == RCCL_SDMA_BACKEND


def validate_grad_reduce_backend_config() -> None:
    """Fail early when the gradient selector cannot reach RCCL CE."""
    if not rccl_sdma_param_gather_enabled():
        raise RuntimeError(
            "MEGATRON_GRAD_REDUCE_BACKEND=rccl_sdma requires "
            "MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma because gradient ReduceScatter "
            "reuses the dedicated zero-CTA process group and symmetric-memory pool."
        )
    if os.getenv("RCCL_CE_REDUCESCATTER", "").strip() != "1":
        raise RuntimeError(
            "MEGATRON_GRAD_REDUCE_BACKEND=rccl_sdma requires "
            "RCCL_CE_REDUCESCATTER=1 to be present before RCCL initializes. "
            "Use the Primus launcher hook or export both variables before launch."
        )


def make_grad_and_param_buffer_init(original):
    """Extend an already-direct-param-data ``__init__`` so ``grad_data`` is ALSO
    a direct symmetric buffer, allocated from the same pool as ``param_data``.
    """
    from primus.backends.megatron.core.distributed.rccl_sdma_param_gather import (
        mark_direct_param_buffer,
        prepare_direct_param_buffer_pool,
        rendezvous_direct_param_buffer,
    )

    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        ddp_config = bound.arguments["ddp_config"]
        original_group = bound.arguments["data_parallel_group"]
        nccl_ub = bound.arguments["nccl_ub"]

        if not ddp_config.use_distributed_optimizer or nccl_ub:
            # Preserve the existing parameter-wrapper behavior for unsupported
            # direct-gather layouts. With the rccl_sdma backend explicitly
            # selected, that inner wrapper rejects these configurations before
            # Megatron allocation rather than silently changing backends.
            return original(self, *args, **kwargs)

        device = torch.device("cuda", torch.cuda.current_device())
        group, pool = prepare_direct_param_buffer_pool(original_group, device)

        allocation_thread = threading.get_ident()
        grad_data_allocated = False

        def allocate_grad_data(*zeros_args, **zeros_kwargs):
            nonlocal grad_data_allocated
            if threading.get_ident() == allocation_thread and not grad_data_allocated:
                grad_data_allocated = True
                try:
                    with torch.cuda.use_mem_pool(pool):
                        return _REAL_TORCH_ZEROS(*zeros_args, **zeros_kwargs)
                except RuntimeError as exc:
                    raise RuntimeError(
                        "RCCL-SDMA could not allocate the direct symmetric gradient "
                        "buffer from the shared pool right after its param_data "
                        "sibling succeeded."
                    ) from exc
            # Not our target call (e.g. the shared MXFP8 buffer path only calls
            # zeros once and never reaches here, or a future Megatron version
            # adds another call): defer to the true allocator, never to
            # whatever another wrap's interceptor currently is.
            return _REAL_TORCH_ZEROS(*zeros_args, **zeros_kwargs)

        with mock.patch.object(torch, "zeros", allocate_grad_data):
            result = original(self, *args, **kwargs)

        if not grad_data_allocated or self.grad_data is None:
            # The MXFP8 shared-buffer branch (`self.grad_data is self.shared_buffer`,
            # aliasing param_data) never calls torch.zeros a second time. Leave it
            # alone -- reuse_grad_buf_for_mxfp8_param_ag is not this campaign's target
            # and the buffer is already `param_data`, which the param patch handles.
            return result

        symmetric_memory = rendezvous_direct_param_buffer(self.grad_data, group)
        self._primus_rccl_sdma_grad_symmetric_memory = symmetric_memory
        mark_direct_param_buffer(self.grad_data)
        for bucket in self.buckets:
            if bucket.grad_data is not None:
                mark_direct_param_buffer(bucket.grad_data)
        return result

    return wrapped


def make_start_grad_sync(original):
    """Build the RCCL CE replacement for Megatron's gradient ReduceScatter."""
    from megatron.core.distributed.param_and_grad_buffer import shard_buffer
    from torch.distributed.distributed_c10d import _coalescing_manager

    from primus.backends.megatron.core.distributed.rccl_sdma_param_gather import (
        get_sdma_process_group,
        is_direct_param_buffer,
    )

    @functools.wraps(original)
    def start_grad_sync(self, force_all_reduce: bool = False):
        eligible = (
            self.ddp_config.use_distributed_optimizer
            and not force_all_reduce
            and self.ddp_config.num_distributed_optimizer_instances == 1
            and not getattr(self.ddp_config, "reduce_scatter_with_fp32_accumulation", False)
            and len(self.buckets) > 0
            and all(is_direct_param_buffer(bucket.grad_data) for bucket in self.buckets)
        )
        if not eligible:
            return original(self, force_all_reduce=force_all_reduce)

        # From here on this mirrors Megatron's own reduce-scatter-only branch of
        # `start_grad_sync` exactly (gradient scaling, reduce_op selection, the
        # first-batch/no-multiple-outstanding-calls invariants, NaN/large-grad
        # checks), swapping only which process group performs the collective.
        if self.is_first_batch and self.grad_reduce_handle is not None:
            return None
        assert (
            self.grad_reduce_handle is None
        ), "Should not have multiple communication calls outstanding at once"

        # CUDA graph replay is asynchronous with respect to the outer autograd
        # hooks. Match Megatron's native path by waiting before reading,
        # scaling, or reducing any replay-produced gradient.
        ready_events = []
        seen_event_ids = set()
        for bucket in self.buckets:
            for param in getattr(bucket, "params_list", ()):
                event = getattr(param, "_cudagraph_wgrad_ready_event", None)
                if event is not None and id(event) not in seen_event_ids:
                    ready_events.append(event)
                    seen_event_ids.add(id(event))
        if ready_events:
            current_stream = torch.cuda.current_stream()
            for event in ready_events:
                current_stream.wait_event(event)

        # Higher-precision local accumulation keeps param.main_grad outside
        # bucket.grad_data. Stage it into the communication buffer exactly as
        # Megatron does before checks, scaling, and reduce-scatter.
        with torch.no_grad():
            for bucket in self.buckets:
                for param in getattr(bucket, "params_with_extra_main_grads", ()):
                    grad_buffer_view = getattr(param, "main_grad_copy_in_grad_buffer", None)
                    if grad_buffer_view is not None:
                        grad_buffer_view.copy_(param.main_grad)

        if self.ddp_config.check_for_nan_in_grad or self.ddp_config.check_for_large_grads:
            self.check_grads(
                check_for_nan_or_inf=self.ddp_config.check_for_nan_in_grad,
                check_for_large=self.ddp_config.check_for_large_grads,
            )

        with torch.no_grad():
            for bucket in self.buckets:
                if bucket.gradient_scaling_factor != 1.0:
                    bucket.grad_data *= bucket.gradient_scaling_factor

        reduce_op = torch.distributed.ReduceOp.SUM
        if self.ddp_config.average_in_collective:
            reduce_op = torch.distributed.ReduceOp.AVG

        group = get_sdma_process_group(self.intra_distributed_optimizer_instance_group)

        # Always async at the NCCL level -- RCCL's synchronous null-stream
        # fallback crashes on imported ROCr VMM pointers (the same reason
        # `make_start_param_sync` always launches through the async stream).
        # Megatron's own `overlap_grad_reduce` semantics are reproduced below by
        # deciding, AFTER the coalesced launch, whether to store the handle for
        # later or wait+synchronize on it right here.
        with _coalescing_manager(group, async_ops=True) as cm:
            for idx, bucket in enumerate(self.buckets):
                if self.cached_grad_buffer_shard_list[idx] is None:
                    self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                        bucket.grad_data, self.intra_distributed_optimizer_instance_size
                    )
                local_data_view = self.cached_grad_buffer_shard_list[idx][
                    self.intra_distributed_optimizer_instance_rank
                ]
                torch.distributed.reduce_scatter_tensor(
                    local_data_view,
                    bucket.grad_data,
                    op=reduce_op,
                    group=group,
                    async_op=True,
                )

        self.grad_reduce_dispatched = True
        if self.ddp_config.overlap_grad_reduce:
            self.grad_reduce_handle = cm
        else:
            cm.wait()
            torch.cuda.current_stream(self.buckets[0].grad_data.device).synchronize()
            self.grad_reduce_handle = None
        return None

    return start_grad_sync


@register_patch(
    "megatron.distributed.rccl_sdma_grad_reduce_scatter",
    backend="megatron",
    phase="before_train",
    description=(
        "Route the single-instance distributed-optimizer gradient ReduceScatter "
        "through the same dedicated zero-CTA RCCL copy-engine process group "
        "already used for the parameter AllGather."
    ),
    condition=rccl_sdma_grad_reduce_enabled,
    priority=_PRIORITY,
)
def patch_rccl_sdma_grad_reduce_scatter(ctx: PatchContext) -> None:
    validate_grad_reduce_backend_config()

    # The param-gather patch validates NCCL_CTA_POLICY once; re-validating here
    # is cheap and keeps this file correct if it is ever registered/enabled on
    # its own condition in the future.
    validate_global_cta_policy()

    try:
        import megatron.core.distributed.param_and_grad_buffer as pgb
    except ImportError as exc:
        warning_rank_0(
            "[Patch:megatron.distributed.rccl_sdma_grad_reduce_scatter] "
            f"Megatron distributed modules are unavailable; skipping: {exc}"
        )
        return

    param_and_grad_buffer = getattr(pgb, "_ParamAndGradBuffer", None)
    if param_and_grad_buffer is None:
        raise RuntimeError("RCCL-SDMA direct grad reduce-scatter requires _ParamAndGradBuffer")

    # Defensive ordering guarantee, independent of the `priority=` sort above:
    # this wrap MUST compose as the outer layer around the param-gather patch's
    # __init__ wrap (see module docstring). If the param patch has not run yet
    # for any reason, apply it now -- it is idempotent and marker-gated.
    if not getattr(param_and_grad_buffer, "_primus_rccl_sdma_direct_allocation_patched", False):
        from .rccl_sdma_param_all_gather_patches import patch_rccl_sdma_param_all_gather

        patch_rccl_sdma_param_all_gather(ctx)
        if not getattr(param_and_grad_buffer, "_primus_rccl_sdma_direct_allocation_patched", False):
            raise RuntimeError(
                "RCCL-SDMA direct grad reduce-scatter requires the param-gather "
                "patch's direct allocation wrap to be applied first, and applying "
                "it on demand did not set its own marker"
            )

    grad_alloc_marker = "_primus_rccl_sdma_grad_direct_allocation_patched"
    if not getattr(param_and_grad_buffer, grad_alloc_marker, False):
        param_and_grad_buffer.__init__ = make_grad_and_param_buffer_init(param_and_grad_buffer.__init__)
        setattr(param_and_grad_buffer, grad_alloc_marker, True)

    bucket_group = getattr(pgb, "_ParamAndGradBucketGroup", None)
    if bucket_group is None:
        warning_rank_0(
            "[Patch:megatron.distributed.rccl_sdma_grad_reduce_scatter] "
            "_ParamAndGradBucketGroup is unavailable; skipping"
        )
        return

    grad_sync_marker = "_primus_rccl_sdma_grad_reduce_scatter_patched"
    if not getattr(bucket_group, grad_sync_marker, False):
        bucket_group.start_grad_sync = make_start_grad_sync(bucket_group.start_grad_sync)
        setattr(bucket_group, grad_sync_marker, True)

    log_rank_0(
        "[Patch:megatron.distributed.rccl_sdma_grad_reduce_scatter] installed "
        f"(force={os.getenv('RCCL_FORCE_CE_REDUCESCATTER', '0')}, "
        f"per_chunk={os.getenv('RCCL_CE_REDUCE_PER_CHUNK', '0')}, "
        f"max_blocks={os.getenv('RCCL_CE_REDUCE_MAX_BLOCKS', '46')})"
    )
