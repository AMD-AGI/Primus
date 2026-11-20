###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ReplicaScheduler bookkeeping on one CPU process (gloo, EP size 1): configuration, evaluation vs. recompute
forwards, and the end-of-step guard."""

import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from primus.backends.megatron.core.transformer.moe.replica import scheduler as sch


@pytest.fixture(scope="module")
def group(tmp_path_factory):
    owned = not dist.is_initialized()
    if owned:
        store = dist.FileStore(str(tmp_path_factory.mktemp("moe_replica") / "store"), 1)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    yield dist.new_group([0], backend="gloo")
    if owned:
        dist.destroy_process_group()


def make(group, placement="per_step", planner="ultraep", num_slots=2):
    cfg = sch.ReplicaConfig(
        planner=planner, placement=placement, num_slots=num_slots, min_rows=1, min_gain=0.0, async_comm=False
    )
    s = sch.ReplicaScheduler(cfg, group, 8)
    module = torch.nn.Linear(1, 1)
    s.experts = weakref.ref(module)
    return s, module


def topk(seed):
    return torch.randint(0, 8, (16, 2), generator=torch.Generator().manual_seed(seed))


def replica_args(**overrides):
    args = dict(
        moe_replica_planner="ultraep",
        moe_replica_placement="per_step",
        moe_replica_slots=4,
        moe_replica_min_rows=1024,
        moe_replica_min_gain=0.1,
        moe_replica_cost=0,
        moe_replica_block=1,
        moe_replica_replan_above=0.0,
        moe_replica_async_comm=True,
        moe_replica_log_interval=0,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_config_from_args():
    assert sch.ReplicaConfig.from_args(replica_args(moe_replica_planner=None)) is None
    cfg = sch.ReplicaConfig.from_args(replica_args(moe_replica_planner="moonep", moe_replica_slots=8))
    assert cfg == sch.ReplicaConfig(planner="moonep", num_slots=8)


def test_moonep_needs_a_slot_per_local_expert(group):
    with pytest.raises(ValueError, match="moe_replica_slots >= 8"):
        make(group, planner="moonep", num_slots=2)


def test_eval_forward_keeps_routes_on_owners(group):
    s, module = make(group)
    module.eval()
    with torch.no_grad():
        phys = s.remap(topk(0))
    assert not s.use_slots and not s.replay
    assert torch.equal(phys, topk(0) // s.L * (s.L + s.B) + topk(0) % s.L)
    assert s.mb_in_step == 0 and s.plan is None


@pytest.mark.parametrize("placement", ["per_step", "per_microbatch"])
def test_recompute_replays_the_forward(group, placement):
    s, module = make(group, placement)
    module.train()
    outs = []
    for mb in range(3):
        with torch.no_grad():
            fwd = s.remap(topk(mb))
        state = (s.use_slots, s.plan, s.cur_mb)
        assert len(s.replay) == 1
        s.use_slots, s.cur_mb = "anything", 99  # the recompute must restore both
        rec = s.remap(topk(mb))
        assert rec is fwd and not s.replay
        assert (s.use_slots, s.plan, s.cur_mb) == state
        outs.append(state[2])
    assert outs == [0, 1, 2] and s.mb_in_step == 2
    s.end_step()
    assert s.mb_in_step == 0


def test_unrecomputed_forward_is_an_error(group):
    s, module = make(group)
    module.train()
    with torch.no_grad():
        s.remap(topk(0))
    with pytest.raises(RuntimeError, match="never recomputed"):
        s.end_step()


class RecordingTransport(sch.SlotTransport):
    """Records the transfers ReplicaSlots asks for (as plain lists) instead of moving anything."""

    def __init__(self, num_ranks=2, rank=0, num_local_experts=2):
        super().__init__(num_ranks, rank, num_local_experts)
        self.calls = []

    def alloc(self, shape, dtype, device):
        return torch.zeros(shape, dtype=dtype)

    def push_weights(self, own, storage, held, table):
        self.calls.append(("push", held.tolist(), table.tolist()))
        return sch.Pending()

    def return_grads(self, storage, held, owner_grad):
        self.calls.append(("return", held.tolist()))
        return sch.Pending()


def slots_with_table(table):
    transport = RecordingTransport()
    sched = SimpleNamespace(
        R=2,
        B=2,
        rank=0,
        transport=transport,
        plan=SimpleNamespace(slot_expert=torch.tensor(table)),
        version=1,
        slots=[],
        cfg=SimpleNamespace(async_comm=False),
        is_last_microbatch=lambda: False,
    )
    slots = sch.ReplicaSlots(sched, torch.nn.Parameter(torch.zeros(2, 3, 5)))
    return slots, sched, transport


def test_slots_push_only_changed_experts_and_return_before_reassigning():
    slots, sched, transport = slots_with_table([[2, -1], [0, 1]])
    assert slots.w.shape == (2, 3, 5) and slots.main_grad.dtype == torch.float32
    slots.sync()
    assert transport.calls == [("push", [[-1, -1], [-1, -1]], [[2, -1], [0, 1]])]
    slots.sync()  # same table: nothing moves
    assert len(transport.calls) == 1
    # a new expert in an empty slot: push only
    sched.plan.slot_expert, sched.version = torch.tensor([[2, 3], [0, 1]]), 2
    slots.sync()
    assert transport.calls[1:] == [("push", [[2, -1], [0, 1]], [[2, 3], [0, 1]])]
    # a slot that held expert 1 gets expert 3: its gradients go back first, then the push
    sched.plan.slot_expert, sched.version = torch.tensor([[2, 3], [0, 3]]), 3
    slots.sync()
    assert transport.calls[2:] == [
        ("return", [[2, 3], [0, 1]]),
        ("push", [[2, 3], [0, 1]], [[2, 3], [0, 3]]),
    ]


def test_slots_return_before_a_slot_is_emptied():
    # per_microbatch: the next plan has fewer replicas on rank 1, so its slot 1 (expert 1) is emptied; the gradients
    # it collected must go back first, or they are lost and later added to whatever expert the slot gets next
    slots, sched, transport = slots_with_table([[2, -1], [0, 1]])
    slots.sync()
    sched.plan.slot_expert, sched.version = torch.tensor([[2, -1], [0, -1]]), 2
    slots.sync()
    assert transport.calls[1:] == [
        ("return", [[2, -1], [0, 1]]),
        ("push", [[2, -1], [0, 1]], [[2, -1], [0, -1]]),
    ]


def test_slots_return_everything_and_start_over_after_invalidate():
    slots, sched, transport = slots_with_table([[2, -1], [0, 1]])
    slots.sync()
    slots.return_grads()
    slots.invalidate()
    sched.version = 2  # the next step's plan, with the same table
    slots.sync()
    assert transport.calls[1:] == [
        ("return", [[2, -1], [0, 1]]),
        ("push", [[-1, -1], [-1, -1]], [[2, -1], [0, 1]]),
    ]


class _FakeLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sched = self.allocated = None

    def attach_replica(self, sched):
        self.sched = sched

    def allocate_replica_slots(self):
        self.allocated = self.sched
        return self.sched


def test_attach_allocate_and_end_step_follow_the_model(group):
    s, _ = make(group)
    s.experts = None
    layer = torch.nn.Module()
    layer.token_dispatcher = SimpleNamespace(replica=s)
    layer.experts = torch.nn.Module()
    layer.experts.linear_fc1, layer.experts.linear_fc2 = _FakeLinear(), _FakeLinear()
    sch.attach(layer)
    assert layer.experts.linear_fc1.sched is s and layer.experts.linear_fc2.sched is s
    assert s.experts() is layer.experts and s.moe_layer() is layer
    assert sch.allocate_slots([layer]) == [s] and layer.experts.linear_fc2.allocated is s
    ended = []
    s.end_step = lambda: ended.append(s)
    sch.end_step_all([layer, torch.nn.Module()])
    assert ended == [s]


def test_attach_needs_turbo_grouped_linears(group):
    s, _ = make(group)
    layer = torch.nn.Module()
    layer.token_dispatcher = SimpleNamespace(replica=s)
    layer.experts = torch.nn.Module()
    layer.experts.linear_fc1, layer.experts.linear_fc2 = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    with pytest.raises(RuntimeError, match="use_turbo_grouped_gemm"):
        sch.attach(layer)
    no_replica = torch.nn.Module()
    no_replica.token_dispatcher = SimpleNamespace()
    sch.attach(no_replica)  # replication off: nothing to do


def test_unattached_scheduler_is_an_error(group):
    s, _ = make(group)
    s.experts = None
    with pytest.raises(RuntimeError, match="not attached"):
        s.remap(topk(0))


def test_step_without_end_step_is_an_error(group, monkeypatch):
    monkeypatch.setattr(sch, "_num_microbatches", lambda: 2)
    s, module = make(group)
    module.train()
    s.remap(topk(0))
    s.remap(topk(1))
    with pytest.raises(RuntimeError, match="finalize_model_grads"):
        s.remap(topk(2))
    s.end_step()
    s.remap(topk(3))
    assert s.mb_in_step == 0
