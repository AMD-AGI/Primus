###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Replica slots on 2 GPUs (EP 2): one expert layer computed with replicas of the hot experts, against an fp32
reference without them.

Routes go to their ranks through ``all_to_all_single`` in place of DeepEP, the experts run as one Primus-Turbo
grouped GEMM over ``[own experts, replica slots]``, and the replica gradients go back through
``ReplicaScheduler.end_step``. Needs a Primus-Turbo whose ``grouped_gemm`` takes weight segments.
"""


import weakref

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from primus.backends.megatron.core.transformer.moe.replica import scheduler as sch

WORLD = 2
E, H, N = 8, 256, 384  # experts, hidden (GEMM K), expert output (GEMM N)
S, K = 1024, 2  # tokens per rank, top-k
B = 2  # replica slots per rank
# max |a - b| / max |b| over the tensor, as in the replica correctness checks on MI355X
TOL_OUT, TOL_DGRAD, TOL_WGRAD = 4.7e-3, 6.7e-3, 3.6e-3


def _turbo_takes_segments() -> bool:
    try:
        from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (  # noqa: F401
            grouped_gemm_segments_impl,
        )
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD or not _turbo_takes_segments(),
    reason=f"needs {WORLD} GPUs and a Primus-Turbo grouped_gemm that takes weight segments",
)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def _exchange(t: torch.Tensor, send_counts: torch.Tensor, recv_counts: torch.Tensor) -> torch.Tensor:
    out = t.new_empty((int(recv_counts.sum()), *t.shape[1:]))
    dist.all_to_all_single(out, t.contiguous(), recv_counts.tolist(), send_counts.tolist())
    return out


def _worker(rank: int, init_file: str, fused: bool, async_comm: bool):
    from primus_turbo.pytorch.ops import grouped_gemm

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    try:
        dev = torch.device("cuda", rank)
        L = E // WORLD
        gen = torch.Generator().manual_seed(0)
        w_all = (torch.randn(E, N, H, generator=gen) * 0.05).to(dev, torch.bfloat16)
        own = torch.nn.Parameter(w_all[rank * L : (rank + 1) * L].clone())
        own.main_grad = torch.zeros(own.shape, device=dev, dtype=torch.float32)
        own.grad_added_to_main_grad = False
        if not fused:
            # what Megatron's DDP does with a weight gradient the GEMM returns
            def add_to_main_grad(p):
                p.main_grad += p.grad.float()
                p.grad = None

            own.register_post_accumulate_grad_hook(add_to_main_grad)

        gen = torch.Generator().manual_seed(1 + rank)
        x = torch.randn(S, H, generator=gen).to(dev, torch.bfloat16)
        # experts 0 and 1 (both on rank 0) are hot
        logits = torch.randn(S, E, generator=gen) + torch.tensor([3.0, 2.0] + [0.0] * (E - 2))
        topk = torch.topk(logits, K, dim=-1).indices.to(dev)
        dy = torch.randn(S * K, N, generator=gen).to(dev, torch.bfloat16)

        cfg = sch.ReplicaConfig(
            planner="ultraep", num_slots=B, min_rows=1, min_gain=0.0, async_comm=async_comm
        )
        sched = sch.ReplicaScheduler(cfg, dist.group.WORLD, E)
        experts = torch.nn.Module()  # stands in for the MoE layer's experts (their training flag)
        sched.experts = weakref.ref(experts)
        sch._num_microbatches = lambda: 1  # one micro-batch per step: its backward is the step's last
        phys = sched.remap(topk).reshape(-1)
        assert sched.has_replicas and sched.plan.num_replicas() > 0, "the skewed routing must build replicas"
        slots = sch.ReplicaSlots(sched, own)
        slots.sync()
        # held the way PrimusTurboGroupedLinear holds them: nothing new in a checkpoint
        holder = torch.nn.Module()
        holder.weights = own
        holder._replica_slots = slots
        assert list(holder.state_dict()) == ["weights"]

        # dispatch: every route to the rank and physical expert that computes it
        dest, local = phys // (L + B), phys % (L + B)
        order = torch.argsort(dest * (L + B) + local, stable=True)
        send_counts = torch.bincount(dest, minlength=WORLD)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        send_counts, recv_counts = send_counts.cpu(), recv_counts.cpu()
        logical = topk.reshape(-1)
        rx = _exchange(x.repeat_interleave(K, dim=0)[order], send_counts, recv_counts)
        rdy = _exchange(dy[order], send_counts, recv_counts)
        rlocal = _exchange(local[order], send_counts, recv_counts)
        rlogical = _exchange(logical[order], send_counts, recv_counts)
        by_expert = torch.argsort(rlocal, stable=True)
        rx, rdy, rlocal, rlogical = rx[by_expert], rdy[by_expert], rlocal[by_expert], rlogical[by_expert]
        m_splits = torch.bincount(rlocal, minlength=L + B)

        rx = rx.detach().requires_grad_()
        out = grouped_gemm(
            rx,
            [own, slots.leaf(fused=fused)],
            m_splits,
            trans_b=True,
            fuse_bgrad_accum_pattern="megatron" if fused else None,
        )
        out.backward(rdy)
        sched.end_step()

        xs, gs = rx.detach().float(), rdy.float()
        y_ref = torch.empty(xs.shape[0], N, device=dev)
        dx_ref = torch.empty_like(xs)
        # every rank's contribution to every expert's weight gradient, summed over the ranks
        dw_ref = torch.zeros(E, N, H, device=dev)
        for e in range(E):
            m = rlogical == e
            w = w_all[e].float()
            y_ref[m] = xs[m] @ w.t()
            dx_ref[m] = gs[m] @ w
            dw_ref[e] = gs[m].t() @ xs[m]
        dist.all_reduce(dw_ref)
        dw = own.main_grad

        assert _rel(out, y_ref) <= TOL_OUT, _rel(out, y_ref)
        assert _rel(rx.grad, dx_ref) <= TOL_DGRAD, _rel(rx.grad, dx_ref)
        assert _rel(dw, dw_ref[rank * L : (rank + 1) * L]) <= TOL_WGRAD, _rel(
            dw, dw_ref[rank * L : (rank + 1) * L]
        )
        assert slots.returning is None and not bool(
            slots.main_grad.any()
        ), "slot gradients left after end_step"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("async_comm", [False, True])
def test_replicas_match_reference(tmp_path, fused, async_comm):
    mp.spawn(_worker, args=(str(tmp_path / "pg_init"), fused, async_comm), nprocs=WORLD, join=True)
