###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in MORI async SDMA parameter all-gather for Megatron ZeRO-1.

Required environment when enabled:

- ``ENABLE_MORI_ALLGATHER=1``
- ``MORI_ENABLE_SDMA=1`` (must be set before ranks start)
- ``MORI_SHMEM_HEAP_SIZE`` (bytes; must hold every ZeRO-1 param buffer plus
  the MORI input transit allocation)

Optional:

- ``MEGATRON_MORI_MAX_INPUT_BYTES`` (default 288 MiB; must cover the padded
  shard of the largest DDP bucket)
- ``MEGATRON_MORI_STRICT=1`` (fail instead of falling back to RCCL)
- ``MEGATRON_MORI_DEBUG=1``
- ``MEGATRON_MORI_RCCL_FALLBACK_BUCKETS`` (default 0)

Mutually exclusive with ``ENABLE_SDMA_ALLGATHER``. The path is single-node
world-DP only; hierarchical / multi-node groups stay on RCCL.
"""

import os
import warnings

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0


def _mori_allgather_enabled(_ctx: PatchContext) -> bool:
    return os.getenv("ENABLE_MORI_ALLGATHER", "0") == "1"


def _get_rccl_fallback_bucket_count() -> int:
    value = os.getenv("MEGATRON_MORI_RCCL_FALLBACK_BUCKETS", "0")
    try:
        count = int(value)
    except ValueError:
        warnings.warn(f"Invalid MEGATRON_MORI_RCCL_FALLBACK_BUCKETS={value!r}; using 0.")
        return 0
    if count < 0:
        warnings.warn(f"MEGATRON_MORI_RCCL_FALLBACK_BUCKETS must be non-negative; got {count}, using 0.")
        return 0
    return count


def _make_wrapped_buffer_init(orig_init):
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        from primus.backends.megatron.core.distributed.mori_param_gather import (
            remap_param_buffer_to_mori,
        )

        remap_param_buffer_to_mori(self)

    return __init__


def _make_start_param_sync(orig_start_param_sync):
    from megatron.core.distributed.param_and_grad_buffer import shard_buffer

    from primus.backends.megatron.core.distributed.mori_param_gather import (
        _WaitableHandle,
        start_mori_all_gathers,
    )

    def start_param_sync(self, force_sync: bool = False):
        if not self.ddp_config.use_distributed_optimizer:
            return orig_start_param_sync(self, force_sync=force_sync)

        if force_sync:
            if self.param_gather_handle is not None:
                self.param_gather_handle.wait()
                self.param_gather_handle = None
                return
        else:
            assert self.param_gather_handle is None

        async_op = self.ddp_config.overlap_param_gather and not force_sync
        param_gather_order = getattr(self, "param_gather_order", None)
        fallback_count = _get_rccl_fallback_bucket_count()
        use_rccl = param_gather_order is not None and param_gather_order < fallback_count

        operations = []
        for index, bucket in enumerate(self.buckets):
            if self.cached_param_buffer_shard_list[index] is None:
                self.cached_param_buffer_shard_list[index] = shard_buffer(
                    bucket.param_data, self.intra_distributed_optimizer_instance_size
                )
            local_data_view = self.cached_param_buffer_shard_list[index][
                self.intra_distributed_optimizer_instance_rank
            ]
            operations.append((bucket.param_data, local_data_view))

        if use_rccl:
            import torch

            handles = [
                torch.distributed.all_gather_into_tensor(
                    output,
                    input_tensor,
                    group=self.intra_distributed_optimizer_instance_group,
                    async_op=async_op,
                )
                for output, input_tensor in operations
            ]
            handle = _WaitableHandle(wait_fn=lambda: [work.wait() for work in handles if work is not None])
        else:
            handle = start_mori_all_gathers(
                operations,
                group=self.intra_distributed_optimizer_instance_group,
                async_op=async_op,
            )

        self.param_gather_handle = handle if async_op else None
        self.param_gather_dispatched = True

    return start_param_sync


def _make_wrapped_ddp_init(orig_init):
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        for groups_attr in ("bucket_groups", "expert_parallel_bucket_groups"):
            groups = getattr(self, groups_attr, None) or []
            for order, bucket_group in enumerate(reversed(groups)):
                bucket_group.param_gather_order = order

    return __init__


@register_patch(
    "megatron.distributed.mori_param_all_gather",
    backend="megatron",
    phase="before_train",
    description=(
        "Allocate ZeRO-1 parameters from MORI uncached memory and route parameter "
        "all-gather through async SDMA; gated by ENABLE_MORI_ALLGATHER=1."
    ),
    condition=_mori_allgather_enabled,
)
def patch_mori_param_all_gather(ctx: PatchContext):
    del ctx
    if os.getenv("ENABLE_SDMA_ALLGATHER", "0") == "1":
        raise RuntimeError("ENABLE_MORI_ALLGATHER and ENABLE_SDMA_ALLGATHER are mutually exclusive")

    try:
        import megatron.core.distributed.param_and_grad_buffer as pgb
        from megatron.core.distributed.distributed_data_parallel import (
            DistributedDataParallel,
        )
    except ImportError as exc:
        warning_rank_0(
            "[Patch:megatron.distributed.mori_param_all_gather] Megatron distributed "
            f"modules not importable; skipping: {exc}"
        )
        return

    bucket_group_cls = getattr(pgb, "_ParamAndGradBucketGroup", None)
    buffer_cls = getattr(pgb, "_ParamAndGradBuffer", None)
    if bucket_group_cls is None or buffer_cls is None:
        warning_rank_0(
            "[Patch:megatron.distributed.mori_param_all_gather] required Megatron "
            "parameter-buffer classes not found; skipping."
        )
        return

    if not getattr(buffer_cls, "_primus_mori_param_buffer_patched", False):
        buffer_cls.__init__ = _make_wrapped_buffer_init(buffer_cls.__init__)
        buffer_cls._primus_mori_param_buffer_patched = True

    if not getattr(bucket_group_cls, "_primus_mori_param_gather_patched", False):
        bucket_group_cls.start_param_sync = _make_start_param_sync(bucket_group_cls.start_param_sync)
        bucket_group_cls._primus_mori_param_gather_patched = True

    if not getattr(DistributedDataParallel, "_primus_mori_param_gather_patched", False):
        DistributedDataParallel.__init__ = _make_wrapped_ddp_init(DistributedDataParallel.__init__)
        DistributedDataParallel._primus_mori_param_gather_patched = True

    log_rank_0(
        "[Patch:megatron.distributed.mori_param_all_gather] Installed MORI "
        "uncached parameter-buffer allocation and async SDMA all-gather."
    )


@register_patch(
    "megatron.distributed.mori_param_all_gather_finalize",
    backend="megatron",
    phase="after_train",
    description="Finalize MORI SHMEM after all parameter all-gathers complete.",
    condition=_mori_allgather_enabled,
)
def finalize_mori_param_all_gather(ctx: PatchContext):
    del ctx
    from primus.backends.megatron.core.distributed.mori_param_gather import (
        finalize_mori_runtime,
    )

    finalize_mori_runtime()
    log_rank_0("[Patch:megatron.distributed.mori_param_all_gather_finalize] " "Finalized MORI SHMEM.")
