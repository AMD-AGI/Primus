###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
SDMA (copy-engine) distributed-optimizer param all-gather patch.

Migrated from the source patch ``megatron_sdma_allgather.patch`` (mpo branch).

This replaces the in-place edits the source patch made to
``third_party/Megatron-LM`` with three runtime monkey-patches, all gated by
``ENABLE_SDMA_ALLGATHER=1``:

  1. ``_ParamAndGradBucketGroup.start_param_sync`` -- the distributed-optimizer
     path is re-implemented to dispatch one all-gather per bucket through
     :func:`all_gather_into_tensor_sdma` (copy-engine) instead of the RCCL
     ``_coalescing_manager`` group. The first two bucket groups (by gather
     order) stay on the regular RCCL fallback. The layer-wise optimizer path is
     delegated unchanged to the original method.
  2. ``DistributedDataParallel.__init__`` -- after construction, each bucket
     group is annotated with ``param_gather_order`` (reverse dispatch order,
     mirroring the source patch).
  3. MoE experts ``forward`` -- ``tokens_per_expert`` is moved onto the
     dispatched-input device before the experts run (the source patch's
     ``moe_layer.py`` one-liner).

When the SDMA primitives (Primus-Turbo symmetric memory + ``hip``) are
unavailable, :func:`all_gather_into_tensor_sdma` falls back to
``torch.distributed.all_gather_into_tensor``, so enabling the flag is safe even
on images without the kernels.
"""

import os

import torch

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0


def _sdma_allgather_enabled(_ctx: PatchContext) -> bool:
    return os.environ.get("ENABLE_SDMA_ALLGATHER", "0") == "1"


def _make_start_param_sync(orig_start_param_sync):
    """Build a replacement ``start_param_sync`` for ``_ParamAndGradBucketGroup``."""
    from megatron.core.distributed.param_and_grad_buffer import shard_buffer

    from primus.backends.megatron.core.distributed.sdma_param_gather import (
        _all_gather_into_tensor_waitable_fallback,
        _WaitableHandle,
        all_gather_into_tensor_sdma,
    )

    def start_param_sync(self, force_sync: bool = False):
        # Layer-wise optimizer path is unchanged; only the distributed-optimizer
        # path routes through SDMA.
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

        # Keep the first two bucket groups (by gather order) on the regular
        # RCCL all-gather; route the rest through SDMA. param_gather_order is
        # assigned in the DDP __init__ wrapper below.
        param_gather_order = getattr(self, "param_gather_order", None)
        enable_sdma = os.getenv("ENABLE_SDMA_ALLGATHER") == "1"
        all_gather_func = (
            _all_gather_into_tensor_waitable_fallback
            if (param_gather_order is not None and param_gather_order < 2) or not enable_sdma
            else all_gather_into_tensor_sdma
        )

        param_gather_handles = []
        for idx, bucket in enumerate(self.buckets):
            if self.cached_param_buffer_shard_list[idx] is None:
                self.cached_param_buffer_shard_list[idx] = shard_buffer(
                    bucket.param_data, self.intra_distributed_optimizer_instance_size
                )
            local_data_view = self.cached_param_buffer_shard_list[idx][
                self.intra_distributed_optimizer_instance_rank
            ]
            handle = all_gather_func(
                bucket.param_data,
                local_data_view,
                group=self.intra_distributed_optimizer_instance_group,
                async_op=async_op,
            )
            if async_op:
                param_gather_handles.append(handle)

        if async_op:

            def _wait_all_param_gathers():
                for handle in param_gather_handles:
                    handle.wait()

            self.param_gather_handle = _WaitableHandle(wait_fn=_wait_all_param_gathers)
        else:
            self.param_gather_handle = None
        self.param_gather_dispatched = True

    return start_param_sync


def _make_wrapped_ddp_init(orig_init):
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # Mirror the source patch: number bucket groups in reverse dispatch
        # order so start_param_sync can keep the first two on RCCL.
        for groups_attr in ("bucket_groups", "expert_parallel_bucket_groups"):
            groups = getattr(self, groups_attr, None) or []
            for order, bucket_group in enumerate(reversed(groups)):
                bucket_group.param_gather_order = order

    return __init__


def _make_wrapped_experts_forward(orig_forward):
    def forward(self, permuted_local_hidden_states, tokens_per_expert, *args, **kwargs):
        # Source patch moved tokens_per_expert onto the dispatched-input device
        # before the experts run.
        if torch.is_tensor(tokens_per_expert):
            tokens_per_expert = tokens_per_expert.to(permuted_local_hidden_states.device)
        return orig_forward(self, permuted_local_hidden_states, tokens_per_expert, *args, **kwargs)

    return forward


@register_patch(
    "megatron.distributed.sdma_param_all_gather",
    backend="megatron",
    phase="before_train",
    description=(
        "Route the distributed-optimizer param all-gather through Primus-Turbo "
        "SDMA (copy-engine) memcpys; gated by ENABLE_SDMA_ALLGATHER=1."
    ),
    condition=_sdma_allgather_enabled,
)
def patch_sdma_param_all_gather(ctx: PatchContext):
    del ctx

    try:
        import megatron.core.distributed.param_and_grad_buffer as pgb
        from megatron.core.distributed.distributed_data_parallel import (
            DistributedDataParallel,
        )
    except ImportError as exc:
        warning_rank_0(
            f"[Patch:megatron.distributed.sdma_param_all_gather] Megatron distributed "
            f"modules not importable; skipping: {exc}"
        )
        return

    bucket_group_cls = getattr(pgb, "_ParamAndGradBucketGroup", None)
    if bucket_group_cls is None:
        warning_rank_0(
            "[Patch:megatron.distributed.sdma_param_all_gather] "
            "_ParamAndGradBucketGroup not found; skipping."
        )
        return

    if not getattr(bucket_group_cls, "_primus_sdma_param_gather_patched", False):
        bucket_group_cls.start_param_sync = _make_start_param_sync(bucket_group_cls.start_param_sync)
        bucket_group_cls._primus_sdma_param_gather_patched = True

    if not getattr(DistributedDataParallel, "_primus_sdma_param_gather_patched", False):
        DistributedDataParallel.__init__ = _make_wrapped_ddp_init(DistributedDataParallel.__init__)
        DistributedDataParallel._primus_sdma_param_gather_patched = True

    # MoE experts: move tokens_per_expert onto the dispatched-input device.
    try:
        from megatron.core.transformer.moe import experts as experts_mod

        for cls_name in ("GroupedMLP", "TEGroupedMLP", "SequentialMLP"):
            cls = getattr(experts_mod, cls_name, None)
            if cls is None or getattr(cls, "_primus_sdma_tokens_to_device", False):
                continue
            cls.forward = _make_wrapped_experts_forward(cls.forward)
            cls._primus_sdma_tokens_to_device = True
    except ImportError as exc:
        warning_rank_0(
            f"[Patch:megatron.distributed.sdma_param_all_gather] MoE experts module "
            f"not importable; skipping tokens_per_expert device fixup: {exc}"
        )

    log_rank_0(
        "[Patch:megatron.distributed.sdma_param_all_gather] Installed SDMA param "
        "all-gather (start_param_sync + param_gather_order + experts tokens_per_expert)."
    )
