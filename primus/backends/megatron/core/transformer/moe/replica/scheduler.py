###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Dynamic redundant experts inside a Megatron MoE layer (BF16, TP 1, PP 1).

``ReplicaScheduler`` (one per MoE layer, owned by the token dispatcher) all-gathers the routed counts of every
micro-batch, plans (a full plan on the first micro-batch of a step with ``per_step``, every micro-batch with
``per_microbatch``) and remaps the logical expert ids to physical ones, so a dispatcher built for ``R * (L + B)``
experts sends every route to the rank that computes it. With ``per_step`` the later micro-batches of a step split
every expert's rows among its owner and replicas in the first plan's proportions, on the device.

``ReplicaSlots`` (one per expert grouped linear) keeps the slot table its ``B`` slots hold and decides what moves:
before the slots are used in a forward, the weights of every slot whose expert changed (after returning the
gradients of the slots being reassigned); slot gradients add up across micro-batches and go back into the owner's
``main_grad``, in increasing source-rank order, before the slots are reassigned and at the end of the step: in the
backward of the step's last micro-batch, once the gradient of the MoE layer's input is computed (``end_step``), so
before Megatron's ``finalize_model_grads`` reduces the expert gradients. The slot tensors live in a ``SlotStorage``
(``storage.py``) and a ``SlotTransport`` (``transport.py``, NCCL point to point here) moves them. The router, the
aux loss and the expert bias only ever see logical experts.

Activation recompute: a training forward that runs without grad (the first pass of Megatron's checkpoint) plans and
pushes as usual and queues its remap; the recompute in the backward replays the queued remap instead of planning
again, so both passes route every token to the same rank and slot. The scheduler tells training from evaluation by
the ``training`` flag of the layer's experts.

Wiring (the ``megatron.moe.replica_attach`` patch in
``primus/backends/megatron/patches/moe_patches/moe_replica_patches.py``): ``attach`` runs once the MoE layer is
built. It hands the dispatcher's scheduler to the layer's two expert linears, which allocate their slots then, in the
dtype ``ReplicaConfig.grad_dtype`` predicts for the owners' ``main_grad``, and puts the end of the step on the
gradient of the layer's input. Under DDP, the ``megatron.moe.replica_ddp`` patch puts two callbacks on the owners'
weights: ``end_step`` marks them ready for the gradient reduction after the replica gradients are in (needed with
``overlap_grad_reduce``), and the weight push first waits for their all-gather (needed with
``overlap_param_gather``).
"""

import weakref
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist

from primus.core.utils.module_utils import log_rank_0

from .planner import (
    ReplicaPlan,
    alloc_fractions,
    make_plan,
    resplit_alloc,
    resplit_if_balanced,
)
from .remap import remap_alloc, remap_topk, slot_lookup
from .storage import LayerwiseStorage
from .transport import NcclTransport, Pending, SlotTransport


@dataclass
class ReplicaConfig:
    """The ``moe_replica_*`` arguments (see ``primus/configs/models/megatron/language_model.yaml``)."""

    planner: str
    placement: str = "per_step"
    num_slots: int = 4
    min_rows: int = 1024
    min_gain: float = 0.1
    replica_cost: int = 0
    block: int = 1
    replan_above: float = 0.0
    async_comm: bool = True
    log_interval: int = 0
    grad_dtype: torch.dtype = torch.float32  # dtype of the owners' main_grad, so of the slots' main_grad

    @classmethod
    def from_args(cls, args) -> Optional["ReplicaConfig"]:
        """None when ``moe_replica_planner`` is unset (replication off)."""
        if not args.moe_replica_planner:
            return None
        return cls(
            planner=args.moe_replica_planner,
            placement=args.moe_replica_placement,
            num_slots=args.moe_replica_slots,
            min_rows=args.moe_replica_min_rows,
            min_gain=args.moe_replica_min_gain,
            replica_cost=args.moe_replica_cost,
            block=args.moe_replica_block,
            replan_above=args.moe_replica_replan_above,
            async_comm=args.moe_replica_async_comm,
            log_interval=args.moe_replica_log_interval,
            # the main_grad dtype of Megatron's DDP (grad_reduce_in_fp32 is accumulate_allreduce_grads_in_fp32)
            grad_dtype=torch.float32 if args.accumulate_allreduce_grads_in_fp32 else args.params_dtype,
        )


class _EndStepAfterLayerBackward(torch.autograd.Function):
    """Identity on the MoE layer's input. Its backward runs after the whole layer's backward, so after both expert
    linears put their slot gradients in the slots' ``main_grad``; after the step's last micro-batch it ends the
    step."""

    @staticmethod
    def forward(ctx, x, sched):
        ctx.sched = sched
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        if ctx.sched.is_last_microbatch():
            ctx.sched.end_step()
        return grad, None


def attach(moe_layer: torch.nn.Module) -> None:
    """Once a MoE layer is built with replication on: hand its token dispatcher's scheduler to the layer's two
    expert linears, which then also compute the replica slots (and allocate them), and end the step in the backward
    of the layer's input."""
    dispatcher = getattr(moe_layer, "token_dispatcher", None)
    sched = getattr(dispatcher, "replica", None)
    if sched is None:
        raise RuntimeError(
            f"moe_replica_planner is set but the MoE layer's token dispatcher ({type(dispatcher).__name__}) has "
            "no replica scheduler: replication needs the Primus-Turbo DeepEP dispatcher (use_turbo_deepep), "
            "check that the Primus-Turbo patches were applied"
        )
    experts = moe_layer.experts
    linears = [getattr(experts, name, None) for name in ("linear_fc1", "linear_fc2")]
    if not all(hasattr(linear, "attach_replica") for linear in linears):
        raise RuntimeError(
            "moe_replica_planner needs the Primus-Turbo grouped linear experts (use_turbo_grouped_gemm), got "
            f"{[type(linear).__name__ for linear in linears]}"
        )
    sched.experts = weakref.ref(experts)
    sched.moe_layer = weakref.ref(moe_layer)
    for linear in linears:
        linear.attach_replica(sched)

    def end_step_after_backward(module, args):
        x = args[0]
        if sched.training and torch.is_grad_enabled() and not x.requires_grad:
            raise RuntimeError(
                "moe replica: the MoE layer's input needs a gradient in training; the step ends in its backward"
            )
        return (_EndStepAfterLayerBackward.apply(x, sched), *args[1:])

    moe_layer.register_forward_pre_hook(end_step_after_backward)


class ReplicaScheduler:
    def __init__(
        self,
        cfg: ReplicaConfig,
        ep_group,
        num_experts: int,
        transport: Optional[SlotTransport] = None,
    ):
        self.cfg = cfg
        self.group = ep_group
        self.R = ep_group.size()
        self.rank = ep_group.rank()
        self.E = num_experts
        self.L = num_experts // self.R
        self.B = cfg.num_slots
        if cfg.planner == "moonep" and self.B < self.L:
            raise ValueError(
                f"moe_replica_planner moonep needs moe_replica_slots >= {self.L} (local experts per rank), "
                f"got {self.B}"
            )
        self.num_physical = self.R * (self.L + self.B)
        self.transport = transport if transport is not None else NcclTransport(ep_group, self.L)
        self.experts = None  # weakref to the layer's experts (their training flag), set by attach()
        self.moe_layer = None  # weakref to the MoE layer (its layer number, for logs), set by attach()
        self.plan: Optional[ReplicaPlan] = None
        self.lut = None
        self.cum_frac = None  # per_step: the step's plan as per-expert shares, on the device
        self.version = 0  # bumped whenever the slot table changes
        self.new_step = True
        self.use_slots = False  # the current micro-batch may route to replica slots (the same on every rank)
        self.slots: List["ReplicaSlots"] = []
        self.num_plans = 0
        self.num_kept = 0  # per_microbatch micro-batches that kept their slots (replan_above)
        self.mb_in_step = 0  # index of the current micro-batch in the step (forward order)
        self.cur_mb = 0  # micro-batch of the forward (or recompute) whose experts run next
        self.replay = deque()  # remaps of no-grad training forwards, in order, for their recompute

    @property
    def training(self) -> bool:
        m = self.experts() if self.experts is not None else None
        return m.training if m is not None else torch.is_grad_enabled()

    @property
    def layer_number(self):
        m = self.moe_layer() if self.moe_layer is not None else None
        return getattr(m, "layer_number", None)

    @property
    def has_replicas(self) -> bool:
        """The current micro-batch may route to replica slots (the same answer on every rank)."""
        return self.use_slots

    def is_last_microbatch(self) -> bool:
        return self.cur_mb == _num_microbatches() - 1

    def _owner_ids(self, topk: torch.Tensor) -> torch.Tensor:
        flat = topk.to(torch.int64)
        return (flat // self.L) * (self.L + self.B) + flat % self.L

    def remap(self, topk: torch.Tensor) -> torch.Tensor:
        """Physical ids for this rank's logical ``topk`` ``[S, K]`` (no -1). Collective over the EP group."""
        if self.experts is None:
            raise RuntimeError(
                "moe replica: the MoE layer's experts were not attached (the megatron.moe.replica_attach patch "
                "attaches them when the layer is built)"
            )
        grad = torch.is_grad_enabled()
        if not grad and not self.training:
            # evaluation: no backward will return slot gradients, so compute every route on its owner
            self.use_slots = False
            return self._owner_ids(topk)
        if grad and self.replay:
            phys, use_slots, plan, version, mb = self.replay.popleft()
            if version != self.version:
                raise RuntimeError("moe replica: the slots changed between a forward and its recompute")
            assert phys.shape == topk.shape, "moe replica: recompute routed a different number of tokens"
            self.use_slots, self.plan, self.cur_mb = use_slots, plan, mb
            return phys
        phys = self._remap(topk)
        if not grad:
            self.replay.append((phys, self.use_slots, self.plan, self.version, self.cur_mb))
        return phys

    def to_physical_routes(self, routing_map, probs: torch.Tensor, token_indices, topk: int):
        """Logical routes -> the ``R * (L + B)`` physical experts a dispatcher built for them takes. Returns
        ``(None, probs, token_indices)``: the probabilities per physical expert ``[S, R * (L + B)]`` and each
        token's physical experts ``[S, topk]``. The logical top-k is ``token_indices`` when given, otherwise
        read from ``routing_map`` ``[S, E]``. Collective over the EP group (``remap``)."""
        if token_indices is None:
            _, sorted_ids = routing_map.to(torch.int8).sort(dim=-1, descending=True, stable=True)
            token_indices = sorted_ids[:, :topk]
        phys = self.remap(token_indices)
        # a token's physical experts are distinct, so the scatter keeps every probability
        phys_probs = probs.new_zeros(probs.shape[0], self.num_physical).scatter(
            1, phys, probs.gather(1, token_indices)
        )
        return None, phys_probs, phys

    def _remap(self, topk: torch.Tensor) -> torch.Tensor:
        cfg, L, B = self.cfg, self.L, self.B
        if not self.new_step:
            self.mb_in_step += 1
        n = _num_microbatches()
        if self.mb_in_step >= n:
            raise RuntimeError(
                f"moe replica: micro-batch {self.mb_in_step} of a {n}-micro-batch step; the last step did not end "
                "(the backward of its last micro-batch never reached the MoE layer's input, where attach() ends "
                "the step)"
            )
        self.cur_mb = self.mb_in_step
        later = cfg.placement == "per_step" and not self.new_step and self.plan is not None
        if later and self.plan.num_replicas() == 0:
            # the step's plan built no replica, so a re-split could only keep every route on its owner
            self.use_slots = False
            return self._owner_ids(topk)
        local = torch.bincount(topk.reshape(-1), minlength=self.E)
        counts_dev = torch.empty(self.R, self.E, device=topk.device, dtype=local.dtype)
        dist.all_gather_into_tensor(counts_dev.view(-1), local, group=self.group)
        if later:
            # the step's slots and shares, applied to this micro-batch's counts without a host round trip
            self.use_slots = True
            alloc = resplit_alloc(self.cum_frac, counts_dev.sum(dim=0))
            return remap_alloc(topk, counts_dev, self.rank, alloc, self.lut, B)
        counts = counts_dev.cpu()
        plan = None
        if (
            cfg.placement == "per_microbatch"
            and cfg.replan_above > 0
            and not self.new_step
            and self.plan is not None
        ):
            plan = resplit_if_balanced(counts, self.plan.slot_expert, B, cfg.block, cfg.replan_above)
            self.num_kept += plan is not None
        if plan is None:
            plan = make_plan(
                cfg.planner,
                counts,
                B,
                min_rows=cfg.min_rows,
                block=cfg.block,
                min_gain=cfg.min_gain,
                replica_cost=cfg.replica_cost,
            )
        self.new_step = False
        if self.plan is None or not torch.equal(plan.slot_expert, self.plan.slot_expert):
            self.version += 1
            self.lut = slot_lookup(plan)
        self.plan = plan
        self.use_slots = plan.num_replicas() > 0
        if cfg.placement == "per_step":
            self.cum_frac = alloc_fractions(plan.alloc, L).to(topk.device)
        self.num_plans += 1
        if cfg.log_interval and self.rank == 0 and (self.num_plans - 1) % cfg.log_interval == 0:
            mean = counts.sum().item() / self.R
            raw = counts.sum(dim=0).view(self.R, L).sum(dim=1)
            log_rank_0(
                f"[moe_replica] layer {self.layer_number} plan {self.num_plans}: rank load max/mean "
                f"{raw.max().item() / mean:.3f} -> {plan.rank_loads().max().item() / mean:.3f}, "
                f"replicas {plan.num_replicas()}, kept {self.num_kept}"
            )
        if cfg.async_comm:
            for s in self.slots:
                s.start_sync()
        return remap_topk(topk, counts_dev, self.rank, plan, self.lut)

    def end_step(self):
        """Return all pending slot gradients; the owners' weights change next, so every slot is pushed again. Runs
        in the backward of the step's last micro-batch (``attach``)."""
        # whether a return is in flight is the same on every rank (posted by all of them), so reduce() stays collective
        returned = [s.returning is not None for s in self.slots]
        for s in self.slots:
            s.finish_return()
        for s, done in zip(self.slots, returned):
            if not done:
                s.return_grads()
            s.invalidate()
        for s in self.slots:
            # the owners' main_grad is complete only now: their DDP bucket may start its reduction (set by the
            # megatron.moe.replica_ddp patch; once per step, whether or not the step had replicas)
            grad_ready = getattr(s.param, "moe_replica_grad_ready", None)
            if grad_ready is not None:
                grad_ready()
        self.new_step = True
        self.mb_in_step = 0
        if self.replay:
            raise RuntimeError(f"moe replica: {len(self.replay)} forwards without grad were never recomputed")


class ReplicaSlots:
    """Replica slots of one expert grouped linear whose own weights are ``param`` ``[L, out, in]``: the slot table
    they hold and the rules for what to move when; the tensors are in ``storage``, the transfers go through the
    scheduler's transport."""

    def __init__(self, sched: ReplicaScheduler, param: torch.nn.Parameter, device):
        """``device``: where ``param`` will be in training (it may still be on the host when the layer is
        built)."""
        self.sched = sched
        self.param = param
        self.storage = LayerwiseStorage(
            sched.transport, sched.B, tuple(param.shape[1:]), param.dtype, sched.cfg.grad_dtype, device
        )
        self.held = self._empty_table()
        self.version = -1
        self.pushing: Optional[Pending] = None  # weight push in flight (start_sync)
        self.returning: Optional[Pending] = None  # gradient return in flight
        sched.slots.append(self)

    @property
    def w(self) -> torch.Tensor:
        return self.storage.w

    @property
    def main_grad(self) -> torch.Tensor:
        return self.storage.main_grad

    def _empty_table(self) -> torch.Tensor:
        return torch.full((self.sched.R, self.sched.B), -1, dtype=torch.int64)

    def invalidate(self):
        self.held = self._empty_table()
        self.version = -1

    def sync(self):
        """Make the slots match the scheduler's current slot table (collective over the EP group) before they are
        used."""
        self.start_sync()
        if self.pushing is not None:
            self.pushing.wait()
            self.pushing = None

    def start_sync(self):
        """Post the weight push for the current slot table without waiting for it."""
        s = self.sched
        if self.version == s.version:
            return
        self.finish_return()
        table = s.plan.slot_expert
        held = self.held if self.storage.claim(self) else self._empty_table()
        # a slot that held an expert and now holds another one or none: its gradients go back first. Decided from
        # the slot tables only, which every rank agrees on: the return is collective
        if bool(((held >= 0) & (table != held)).any()):
            self.return_grads()
        # with overlap_param_gather the owners' weights are all-gathered only by the expert linears' forward
        # pre-hook, after this push (set by the megatron.moe.replica_ddp patch)
        wait_param_gather = getattr(self.param, "moe_replica_wait_param_gather", None)
        if wait_param_gather is not None:
            wait_param_gather()
        self.pushing = s.transport.push_weights(self.param.detach(), self.storage, held, table)
        self.held = table.clone()
        self.version = s.version

    def leaf(self, fused: bool) -> torch.Tensor:
        """The slot weights as a fresh autograd leaf whose gradient lands in ``main_grad``: written by the grouped
        GEMM itself when ``fused`` (its fused wgrad accumulation), added here otherwise."""
        s = self.sched
        used = max((c + 1 for c, e in enumerate(self.held[s.rank].tolist()) if e >= 0), default=0)
        w = self.w.detach().requires_grad_()
        w.main_grad = self.main_grad
        w.grad_added_to_main_grad = False
        # every rank builds a leaf for the same micro-batches (has_replicas is global), so all of them post the
        # early return at the same point of the backward
        early = s.cfg.async_comm and s.is_last_microbatch()

        def hook(leaf):
            if not fused and used:
                self.main_grad[:used] += leaf.grad[:used].to(self.main_grad.dtype)
            # main_grad has it now (fused: the GEMM wrote it and returned a dummy); free the [B, out, in] copy
            leaf.grad = None
            if early:
                self.return_grads(wait=False)

        w.register_post_accumulate_grad_hook(hook)
        return w

    def weight_segments(self, weights, m_splits: torch.Tensor, fused: bool):
        """The grouped GEMM operand of the current micro-batch, ``m_splits`` having ``L + B`` groups (the own
        experts, then the slots): without replicas the own ``weights`` and their ``L`` groups; with replicas
        the slots are synced and the operand is ``[weights, slot leaf]`` over all ``L + B`` groups (``fused``:
        see ``leaf``)."""
        L, B = self.sched.L, self.sched.B
        if m_splits.numel() != L + B:
            raise RuntimeError(
                f"moe replica: {m_splits.numel()} groups for {L} local experts and {B} replica slots"
            )
        if not self.sched.has_replicas:
            # every row is on this rank's own experts: the slot groups are empty
            return weights, m_splits[:L]
        self._check_owner()
        self.sync()
        return [weights, self.leaf(fused)], m_splits

    def _check_owner(self):
        """The slots were allocated before the owners' weights reached their device and got their ``main_grad``:
        make sure they match, so no gradient is added in the wrong precision."""
        main = getattr(self.param, "main_grad", None)
        want = (self.param.dtype, main.dtype if main is not None else self.main_grad.dtype, self.param.device)
        got = (self.w.dtype, self.main_grad.dtype, self.w.device)
        if got != want:
            raise RuntimeError(
                f"moe replica: the slots hold (weight dtype, main_grad dtype, device) {got}, the owner experts "
                f"{want}; the slots' main_grad dtype follows accumulate_allreduce_grads_in_fp32 and params_dtype "
                "as Megatron's DDP does"
            )

    def _owner_grad(self) -> torch.Tensor:
        main = getattr(self.param, "main_grad", None)
        if main is not None:
            return main
        if self.param.grad is not None:
            return self.param.grad
        raise RuntimeError("owner expert weights have neither main_grad nor grad")

    def return_grads(self, wait: bool = True):
        """Every replica sends its slot gradients to the owner, which adds them in source-rank order. Without
        ``wait`` only post the transfers; ``finish_return`` waits and adds."""
        self.finish_return()
        self.returning = self.sched.transport.return_grads(self.storage, self.held, self._owner_grad)
        if wait:
            self.finish_return()

    def finish_return(self):
        if self.returning is not None:
            returning, self.returning = self.returning, None
            returning.wait()


def _num_microbatches() -> int:
    from megatron.core.num_microbatches_calculator import get_num_microbatches

    return get_num_microbatches()
