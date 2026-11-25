###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The megatron.moe.replica_ddp patch on CPU, with stand-ins for Megatron's DDP and bucket group: a replica owner's
weights are marked ready for the gradient reduction by end_step instead of by their backward hook, and the weight
push waits for their all-gather as DDP's forward pre-hook would."""

from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.patches.moe_patches import moe_replica_patches as patches


class FakeBucketGroup:
    def __init__(self):
        self.is_last_microbatch = True
        self.ready = []
        self.param_syncs = []

    def register_grad_ready(self, param, force_all_reduce=False):
        self.ready.append((param, force_all_reduce))

    def finish_param_sync(self, skip_next_bucket_dispatch=False):
        self.param_syncs.append(skip_next_bucket_dispatch)


class FakeDDP:
    def __init__(self, param, overlap_grad_reduce=True, overlap_param_gather=True):
        self.group = FakeBucketGroup()
        self.param_to_bucket_group = {param: self.group}
        self.ddp_config = SimpleNamespace(
            overlap_grad_reduce=overlap_grad_reduce,
            overlap_param_gather=overlap_param_gather,
            align_param_gather=False,
        )
        self.force_all_reduce = False
        self.use_forward_hook = overlap_param_gather
        self.remove_forward_pre_hook_handles = {"layer": "handle"} if overlap_param_gather else {}
        self.overlap_param_gather_with_optimizer_step = False


def owner(ddp_kwargs=None):
    param = torch.nn.Parameter(torch.zeros(2))
    ddp = FakeDDP(param, **(ddp_kwargs or {}))
    seen = []
    hook = patches.defer_owner_grad_ready(ddp, param, lambda *args: seen.append(ddp.group.is_last_microbatch))
    return param, ddp, hook, seen


def test_the_last_microbatch_marks_owners_ready_only_at_the_end_of_the_step():
    param, ddp, hook, seen = owner()
    ddp.group.is_last_microbatch = False
    hook()  # an earlier micro-batch (inside no_sync)
    with pytest.raises(RuntimeError, match="did not run in the last micro-batch"):
        param.moe_replica_grad_ready()
    ddp.group.is_last_microbatch = True
    hook()
    ddp.force_all_reduce = True  # read when the owner is marked ready, not when its hook ran
    # DDP's own hook ran (main_grad add) as if it were not the last micro-batch, so it registered nothing
    assert seen == [False, False] and ddp.group.is_last_microbatch and ddp.group.ready == []
    param.moe_replica_grad_ready()  # end_step, after the replica gradients are in main_grad
    assert ddp.group.ready == [(param, True)]
    with pytest.raises(RuntimeError, match="did not run in the last micro-batch"):
        param.moe_replica_grad_ready()  # once per step


def test_a_failing_ddp_hook_leaves_the_last_microbatch_flag_as_it_was():
    param = torch.nn.Parameter(torch.zeros(2))
    ddp = FakeDDP(param)

    def failing_hook(*args):
        raise RuntimeError("ddp hook failed")

    hook = patches.defer_owner_grad_ready(ddp, param, failing_hook)
    with pytest.raises(RuntimeError, match="ddp hook failed"):
        hook()
    assert ddp.group.is_last_microbatch


def test_without_overlap_grad_reduce_the_hook_is_unchanged():
    param, ddp, hook, seen = owner({"overlap_grad_reduce": False, "overlap_param_gather": False})
    hook()
    param.moe_replica_grad_ready()
    param.moe_replica_wait_param_gather()
    assert seen == [True] and ddp.group.ready == [] and ddp.group.param_syncs == []


def test_the_push_waits_for_the_owners_all_gather_like_the_forward_pre_hook():
    param, ddp, _, _ = owner()
    param.moe_replica_wait_param_gather()
    ddp.ddp_config.align_param_gather = True
    param.moe_replica_wait_param_gather()
    assert ddp.group.param_syncs == [False, True]
    # forward pre-hooks off (first iteration, evaluation, checkpointing): DDP gathers synchronously, no wait
    ddp.remove_forward_pre_hook_handles = {}
    param.moe_replica_wait_param_gather()
    assert ddp.group.param_syncs == [False, True]


def test_the_patch_defers_only_replica_owners(monkeypatch):
    from megatron.core.distributed import distributed_data_parallel

    class FakeDistributedDataParallel(FakeDDP):
        def _make_backward_post_hook(self, param):
            return "ddp hook"

    owner_weights, other = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    owner_weights.moe_replica_owner = True
    monkeypatch.setattr(distributed_data_parallel, "DistributedDataParallel", FakeDistributedDataParallel)
    monkeypatch.setattr(patches, "log_rank_0", lambda msg: None)
    patches.patch_moe_replica_ddp(None)
    ddp = FakeDistributedDataParallel(owner_weights)
    assert ddp._make_backward_post_hook(other) == "ddp hook" and not hasattr(other, "moe_replica_grad_ready")
    assert callable(ddp._make_backward_post_hook(owner_weights)) and hasattr(
        owner_weights, "moe_replica_grad_ready"
    )


def test_overlap_without_the_ddp_patch_is_an_error(monkeypatch):
    # the patch runner logs a failed patch and goes on; replication must not then run with overlapped reduction
    from megatron.core.distributed import distributed_data_parallel
    from megatron.core.transformer.moe import moe_layer

    from primus.backends.megatron.core.transformer.moe.replica import scheduler as sch

    class FakeMoELayer:
        pass

    class FakeDistributedDataParallel:
        def _make_backward_post_hook(self, param):
            return "ddp hook"

    monkeypatch.setattr(moe_layer, "MoELayer", FakeMoELayer)
    monkeypatch.setattr(distributed_data_parallel, "DistributedDataParallel", FakeDistributedDataParallel)
    monkeypatch.setattr(sch, "attach", lambda layer: None)
    monkeypatch.setattr(patches, "get_args", lambda ctx: SimpleNamespace(overlap_grad_reduce=True))
    monkeypatch.setattr(patches, "log_rank_0", lambda msg: None)
    patches.patch_moe_replica_attach(None)
    with pytest.raises(RuntimeError, match="replica_ddp patch, which was not applied"):
        FakeMoELayer()
    patches.patch_moe_replica_ddp(None)
    FakeMoELayer()
