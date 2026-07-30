###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Distributed initialization patches (opt-in, FSDP2 only).

Patches Megatron's _initialize_distributed to pass device_id to
torch.distributed.init_process_group. Without device_id, RCCL guesses
the GPU-to-rank mapping, which was reported to cause deadlocks on MI355X
after the first FSDP2 iteration.

device_id triggers eager RCCL communicator creation for the world PG and all
~26 Megatron sub-groups. That is expensive, so the patch is opt-in via
enable_init_process_group_device_id (default false):

- +16.0 GiB/GPU peak VRAM and +9.9 s startup, measured on llama3-70B BF16
  FSDP2, 1 node x 8 MI355X, TP/PP/EP=1, ROCm 26.5. The memory is invisible to
  the torch caching allocator (max reserved is byte-identical across arms), so
  it is communicator-side, not activations or parameters.
- No throughput effect there (-0.07 %, within run-to-run noise over two
  interleaved repeats per arm).
- Under ODC the eager communicators also serialize XGMI copy streams; see the
  condition below.

The deadlock this guards against did not reproduce on that recipe with the
patch disabled, on either ROCm 26.5 or 26.3, so it must not be assumed
necessary everywhere. It has not been tested multi-node or with TP/PP/EP > 1,
which is why the patch is kept rather than removed. See AIMA-227.
"""

import os

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


@register_patch(
    "megatron.distributed.init_process_group_device_id",
    backend="megatron",
    phase="before_train",
    description=(
        "Inject device_id into torch.distributed.init_process_group to "
        "prevent RCCL device mapping deadlocks on MI355X "
        "(opt-in, FSDP2 only)."
    ),
    priority=10,
    # Opt-in: the eager communicator creation costs +16.0 GiB/GPU and +9.9 s of
    # startup for no throughput gain on llama3-70B FSDP2 (see module docstring),
    # so callers must ask for it on the configurations where it buys liveness.
    #
    # ODC (enable_odc=true) drives gradient exchange over rocSHMEM P2P, not RCCL, and
    # relies on those P2P copy streams overlapping with compute in the backward pass.
    # The device_id injection here eagerly creates the world + ~26 sub-group RCCL
    # communicators, whose resident streams/DMA queues serialize ODC's XGMI copy
    # streams onto the critical path (profiled: cross-stream overlap 120ms -> 2.4ms,
    # ~+128ms/step on single-node 1.5B). nccl_pad is unaffected (it uses these RCCL
    # comms as its native reduce-scatter). So skip the eager-RCCL device_id patch
    # under ODC. Safe on MI300X (the MI355X deadlock this guards does not trigger
    # here; commits before this patch existed ran ODC correctly).
    condition=lambda ctx: (
        getattr(get_args(ctx), "enable_init_process_group_device_id", False)
        and getattr(get_args(ctx), "use_torch_fsdp2", False)
        and not getattr(get_args(ctx), "enable_odc", False)
    ),
)
def patch_init_process_group_device_id(ctx: PatchContext):
    """
    Wrap _initialize_distributed so that torch.distributed.init_process_group
    receives an explicit device_id.

    Megatron computes device_id = torch.device(f'cuda:{args.local_rank}') but
    never passes it to init_process_group.  On MI355X with RCCL, the missing
    device_id causes PyTorch to guess the GPU-to-rank mapping, leading to
    deadlocks on the second FSDP2 iteration.
    """
    import megatron.training.initialize as init_module
    import torch.distributed as dist

    _orig_init_distributed = init_module._initialize_distributed

    def _patched_initialize_distributed(get_embedding_ranks, get_position_embedding_ranks, store):
        _orig_init_pg = dist.init_process_group

        def _init_pg_with_device_id(*args, **kwargs):
            if "device_id" not in kwargs and torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
                log_rank_0(
                    f"[Patch:device_id] Injected device_id=cuda:{local_rank} " f"into init_process_group"
                )
            return _orig_init_pg(*args, **kwargs)

        dist.init_process_group = _init_pg_with_device_id
        try:
            _orig_init_distributed(get_embedding_ranks, get_position_embedding_ranks, store)
        finally:
            dist.init_process_group = _orig_init_pg

    init_module._initialize_distributed = _patched_initialize_distributed
    log_rank_0(
        "[Patch:device_id] Patched _initialize_distributed to inject " "device_id into init_process_group"
    )
