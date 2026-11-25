###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ReplicaScheduler bookkeeping on one CPU process (gloo, EP size 1): configuration, evaluation vs. recompute
forwards, the end-of-step guard, and the routes and grouped GEMM operand handed to the dispatcher and the
experts."""

import weakref
from collections import deque
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from primus.backends.megatron.core.transformer.moe.replica import scheduler as sch

_MEGATRON_NUM_MICROBATCHES = sch._num_microbatches


@pytest.fixture(autouse=True)
def num_microbatches(monkeypatch):
    # Megatron's micro-batch calculator is not set up here; tests end their steps by hand or set fewer
    monkeypatch.setattr(sch, "_num_microbatches", lambda: 8)


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
        accumulate_allreduce_grads_in_fp32=True,
        params_dtype=torch.bfloat16,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_config_from_args():
    assert sch.ReplicaConfig.from_args(replica_args(moe_replica_planner=None)) is None
    cfg = sch.ReplicaConfig.from_args(replica_args(moe_replica_planner="moonep", moe_replica_slots=8))
    assert cfg == sch.ReplicaConfig(planner="moonep", num_slots=8, grad_dtype=torch.float32)
    # the main_grad dtype of Megatron's DDP: params_dtype unless gradients accumulate in fp32
    bf16 = sch.ReplicaConfig.from_args(replica_args(accumulate_allreduce_grads_in_fp32=False))
    assert bf16.grad_dtype == torch.bfloat16


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


class WaitRecordingTransport(RecordingTransport):
    """Also records when a gradient return is waited for."""

    def return_grads(self, storage, held, owner_grad):
        super().return_grads(storage, held, owner_grad)
        calls = self.calls

        class _Pending(sch.Pending):
            def wait(self):
                calls.append(("wait",))

        return _Pending()


def slots_with_table(table, grad_dtype=torch.float32, transport_cls=RecordingTransport):
    transport = transport_cls()
    sched = SimpleNamespace(
        R=2,
        B=2,
        rank=0,
        transport=transport,
        plan=SimpleNamespace(slot_expert=torch.tensor(table)),
        version=1,
        slots=[],
        cfg=SimpleNamespace(async_comm=False, grad_dtype=grad_dtype),
        is_last_microbatch=lambda: False,
    )
    slots = sch.ReplicaSlots(sched, torch.nn.Parameter(torch.zeros(2, 3, 5)), "cpu")
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


def test_end_step_waits_for_an_early_return_once_and_returns_the_rest():
    # async_comm: the slot leaf's hook posts the last micro-batch's return without waiting; end_step must wait for
    # it once without sending it again, and return the slots whose return is not in flight
    early, _, early_transport = slots_with_table([[2, -1], [0, 1]], transport_cls=WaitRecordingTransport)
    late, _, late_transport = slots_with_table([[2, -1], [0, 1]], transport_cls=WaitRecordingTransport)
    for slots in (early, late):
        slots.sync()
    early.return_grads(wait=False)
    sched = SimpleNamespace(slots=[early, late], replay=deque(), new_step=False, mb_in_step=3)
    sch.ReplicaScheduler.end_step(sched)
    for transport in (early_transport, late_transport):
        assert [c[0] for c in transport.calls] == ["push", "return", "wait"], transport.calls
    assert all(bool((s.held < 0).all()) and s.returning is None for s in (early, late))
    assert sched.new_step and sched.mb_in_step == 0


def test_the_push_waits_for_the_owner_and_end_step_marks_every_owner_ready_last():
    # DDP callbacks of the megatron.moe.replica_ddp patch: wait for the owner's all-gather before pushing, mark it
    # ready for the gradient reduction after its replica gradients are back, also when its slots were never used
    used, _, used_transport = slots_with_table([[2, -1], [0, 1]], transport_cls=WaitRecordingTransport)
    unused, _, unused_transport = slots_with_table([[2, -1], [0, 1]], transport_cls=WaitRecordingTransport)
    for slots, transport in ((used, used_transport), (unused, unused_transport)):
        slots.param.moe_replica_wait_param_gather = lambda t=transport: t.calls.append(("wait_param_gather",))
        slots.param.moe_replica_grad_ready = lambda t=transport: t.calls.append(("grad_ready",))
    used.sync()
    sched = SimpleNamespace(slots=[used, unused], replay=deque(), new_step=False, mb_in_step=3)
    sch.ReplicaScheduler.end_step(sched)
    assert [c[0] for c in used_transport.calls] == [
        "wait_param_gather",
        "push",
        "return",
        "wait",
        "grad_ready",
    ]
    assert [c[0] for c in unused_transport.calls] == ["return", "wait", "grad_ready"]


def test_slots_follow_the_configured_grad_dtype_and_check_the_owner():
    slots, _, _ = slots_with_table([[2, -1], [0, 1]], grad_dtype=torch.bfloat16)
    assert slots.main_grad.dtype == torch.bfloat16 and slots.w.dtype == slots.param.dtype
    slots._check_owner()  # no main_grad yet (no DDP): nothing to compare
    slots.param.main_grad = torch.zeros(2, 3, 5, dtype=torch.bfloat16)
    slots._check_owner()
    slots.param.main_grad = torch.zeros(2, 3, 5, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="accumulate_allreduce_grads_in_fp32"):
        slots._check_owner()


class _FakeLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sched = None

    def attach_replica(self, sched):
        self.sched = sched


class _FakeMoELayer(torch.nn.Module):
    def forward(self, x, scale=1.0):
        return x * scale


def test_attach_hands_over_the_scheduler_and_ends_the_step_after_the_last_microbatch(group, monkeypatch):
    monkeypatch.setattr(sch, "_num_microbatches", lambda: 2)
    s, _ = make(group)
    s.experts = None
    layer = _FakeMoELayer()
    layer.token_dispatcher = SimpleNamespace(replica=s)
    layer.experts = torch.nn.Module()
    layer.experts.linear_fc1, layer.experts.linear_fc2 = _FakeLinear(), _FakeLinear()
    sch.attach(layer)
    assert layer.experts.linear_fc1.sched is s and layer.experts.linear_fc2.sched is s
    assert s.experts() is layer.experts and s.moe_layer() is layer
    ended = []
    s.end_step = lambda: ended.append(s.cur_mb)
    x = torch.ones(3, requires_grad=True)
    for mb in range(2):
        s.cur_mb = mb  # set by the micro-batch's remap
        layer(x, 2.0).sum().backward()  # the arguments after the input pass through
        assert ended == ([1] if mb == 1 else [])
    assert torch.equal(x.grad, torch.full((3,), 4.0))
    with torch.no_grad():
        layer(x)  # evaluation: no backward, no end of step
    assert ended == [1]
    # a training input without a gradient would never end the step
    with pytest.raises(RuntimeError, match="needs a gradient"):
        layer(torch.ones(3))


def test_the_step_needs_megatrons_microbatch_count(group, monkeypatch):
    import megatron.core.num_microbatches_calculator as calculator

    monkeypatch.setattr(sch, "_num_microbatches", _MEGATRON_NUM_MICROBATCHES)
    monkeypatch.setattr(calculator, "_GLOBAL_NUM_MICROBATCHES_CALCULATOR", None)
    s, _ = make(group)
    # without the count no step would ever end, and the slot gradients would never reach the owners
    with pytest.raises(AttributeError):
        s.is_last_microbatch()
    monkeypatch.setattr(calculator, "_GLOBAL_NUM_MICROBATCHES_CALCULATOR", SimpleNamespace(get=lambda: 3))
    s.cur_mb = 2
    assert s.is_last_microbatch()


def test_attach_patch_rejects_moe_layer_subclasses(monkeypatch):
    from megatron.core.transformer.moe import moe_layer

    from primus.backends.megatron.patches.moe_patches import moe_replica_patches

    class FakeMoELayer:
        def __init__(self):
            self.built = True

    class CustomMoELayer(FakeMoELayer):
        pass

    attached = []
    monkeypatch.setattr(moe_layer, "MoELayer", FakeMoELayer)
    monkeypatch.setattr(sch, "attach", attached.append)
    monkeypatch.setattr(moe_replica_patches, "log_rank_0", lambda msg: None)
    monkeypatch.setattr(moe_replica_patches, "get_args", lambda ctx: SimpleNamespace())
    moe_replica_patches.patch_moe_replica_attach(None)
    layer = FakeMoELayer()
    assert layer.built and attached == [layer]
    with pytest.raises(NotImplementedError, match="not its subclass CustomMoELayer"):
        CustomMoELayer()
    assert attached == [layer]


def test_attach_needs_the_turbo_dispatcher_and_grouped_linears(group):
    s, _ = make(group)
    layer = torch.nn.Module()
    layer.token_dispatcher = SimpleNamespace(replica=s)
    layer.experts = torch.nn.Module()
    layer.experts.linear_fc1, layer.experts.linear_fc2 = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    with pytest.raises(RuntimeError, match="use_turbo_grouped_gemm"):
        sch.attach(layer)
    # replication on, but the dispatcher is not the Turbo DeepEP one: no silent run without replicas
    no_replica = torch.nn.Module()
    no_replica.token_dispatcher = SimpleNamespace()
    with pytest.raises(RuntimeError, match="has no replica scheduler"):
        sch.attach(no_replica)


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
    with pytest.raises(RuntimeError, match="the last step did not end"):
        s.remap(topk(2))
    s.end_step()
    s.remap(topk(3))
    assert s.mb_in_step == 0


def test_to_physical_routes(group, monkeypatch):
    s, module = make(group)
    module.train()
    first = topk(3)[:, 0]
    # two distinct experts per token, as a router picks them
    ids = torch.stack([first, (first + 1 + topk(4)[:, 0] % 7) % 8], dim=1)
    probs = torch.rand(16, 8, generator=torch.Generator().manual_seed(5))
    routing_map = torch.zeros(16, 8, dtype=torch.bool).scatter(1, ids, True)

    # the real remap: one rank keeps every route on its owner
    rmap, phys_probs, phys = s.to_physical_routes(None, probs, ids, 2)
    assert s.plan.num_replicas() == 0 and rmap is None and phys_probs.shape == (16, s.num_physical)

    # a stand-in plan whose physical ids differ from the logical ones
    logicals = []

    def remap(t):
        logicals.append(t.clone())
        return (t + 3) % s.num_physical

    monkeypatch.setattr(s, "remap", remap)
    for routes, given in ((routing_map, None), (None, ids)):
        rmap, phys_probs, phys = s.to_physical_routes(routes, probs, given, 2)
        logical = logicals[-1]
        assert torch.equal(logical.sort(dim=1).values, ids.sort(dim=1).values)  # each token's own experts
        assert rmap is None and torch.equal(phys, (logical + 3) % s.num_physical)
        want = torch.zeros(16, s.num_physical).scatter(1, phys, probs.gather(1, logical))
        assert torch.equal(phys_probs, want)
    # a logical top-k given as token_indices is remapped as it is
    assert torch.equal(logicals[1], ids)


@pytest.mark.parametrize("has_replicas", [False, True])
def test_weight_segments(has_replicas):
    calls = []
    slots = SimpleNamespace(
        sched=SimpleNamespace(L=4, B=2, has_replicas=has_replicas),
        _check_owner=lambda: calls.append("check"),
        sync=lambda: calls.append("sync"),
        leaf=lambda fused: calls.append(("leaf", fused)) or "slot leaf",
    )
    weights, m_splits = object(), torch.tensor([3, 1, 0, 2, 5, 0])
    got_weights, got_splits = sch.ReplicaSlots.weight_segments(slots, weights, m_splits, fused=True)
    if has_replicas:
        assert got_weights == [weights, "slot leaf"] and torch.equal(got_splits, m_splits)
        assert calls == ["check", "sync", ("leaf", True)]
    else:
        # the own weights and groups only: the same operand as without replication
        assert got_weights is weights and torch.equal(got_splits, m_splits[:4]) and calls == []
    with pytest.raises(RuntimeError, match="5 groups for 4 local experts and 2 replica slots"):
        sch.ReplicaSlots.weight_segments(slots, weights, m_splits[:5], fused=False)
