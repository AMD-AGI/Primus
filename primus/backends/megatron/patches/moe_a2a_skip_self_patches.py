###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MoE all-to-all without the self round.

``megatron.core.tensor_parallel.mappings._AllToAll`` (the MoE token dispatch and
combine, and their backward) calls ``all_to_all_single``, which hands RCCL one
send/recv pair per rank including the rank itself. On gfx1250 single-node RCCL
gives every peer 64 p2p channels and reserves two work batches per channel and
round, so a 4-rank exchange does not fit one kernel's argument space: RCCL splits
it into two kernels that run one after the other (self + 2 peers, then the last
peer), each about as long as the whole exchange could take.

With ``PRIMUS_MOE_A2A_SKIP_SELF=1`` the rank's own chunk is copied locally and
RCCL gets a list all-to-all whose self entry is empty, so the 3 remote peers fit
one kernel. Outputs are bit-identical.
"""

import os

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enabled(ctx: PatchContext) -> bool:
    return os.environ.get("PRIMUS_MOE_A2A_SKIP_SELF", "0").strip().lower() in ("1", "true", "yes", "on")


def _all_to_all_skip_self(group, input, output_split_sizes, input_split_sizes):
    world = group.size()
    rank = group.rank()
    rest = list(input.size()[1:])
    if output_split_sizes is None:
        n = input.size(0) // world
        input_split_sizes = output_split_sizes = [n] * world
    else:
        output_split_sizes = [int(x) for x in output_split_sizes]
        input_split_sizes = [int(x) for x in input_split_sizes]
    output = input.new_empty(
        size=[sum(output_split_sizes)] + rest, dtype=input.dtype, device=torch.cuda.current_device()
    )
    ins = list(input.split(input_split_sizes))
    outs = list(output.split(output_split_sizes))
    outs[rank].copy_(ins[rank])
    empty = input.new_empty([0] + rest)
    ins[rank] = empty
    outs[rank] = empty
    torch.distributed.all_to_all(outs, ins, group=group)
    return output


@register_patch(
    "megatron.moe_a2a_skip_self",
    backend="megatron",
    phase="before_train",
    description="MoE all-to-all: copy the self chunk locally so RCCL runs the 3 peers in one kernel",
    condition=_enabled,
)
def patch_moe_a2a_skip_self(ctx: PatchContext):
    from megatron.core.tensor_parallel import mappings

    cls = mappings._AllToAll
    if getattr(cls.forward, "_primus_skip_self", False):
        return

    @staticmethod
    def forward(ctx_, group, input, output_split_sizes, input_split_sizes):
        ctx_.group = group
        ctx_.output_split_sizes = output_split_sizes
        ctx_.input_split_sizes = input_split_sizes
        if group.size() == 1:
            return input
        return _all_to_all_skip_self(group, input.contiguous(), output_split_sizes, input_split_sizes)

    forward.__func__._primus_skip_self = True
    cls.forward = forward
    log_rank_0("[Patch:megatron.moe_a2a_skip_self] MoE all-to-all copies the self chunk locally")
