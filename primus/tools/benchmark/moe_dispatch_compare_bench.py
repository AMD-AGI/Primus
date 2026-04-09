###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
import time
from typing import Dict, List

import torch
import torch.distributed as dist

from .moe_dispatch_compare_bench_args import add_moe_dispatch_compare_parser


def _pct(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


def _build_tp_groups_for_tp1(world_size: int, rank: int) -> dist.ProcessGroup:
    """Build per-rank TP groups (size=1), return this rank's group."""
    tp_group = None
    for r in range(world_size):
        g = dist.new_group(ranks=[r], backend="nccl")
        if r == rank:
            tp_group = g
    if tp_group is None:
        raise RuntimeError("Failed to create rank-local TP group")
    return tp_group


def _make_inputs(
    tokens: int,
    hidden: int,
    topk: int,
    num_experts: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    router_logits = torch.randn(tokens, num_experts, device=device, dtype=torch.float32)
    topk_vals, topk_idx = torch.topk(router_logits, k=topk, dim=-1)
    probs_topk = torch.softmax(topk_vals, dim=-1).to(torch.float32)
    probs_full = torch.zeros(tokens, num_experts, device=device, dtype=torch.float32)
    probs_full.scatter_(1, topk_idx, probs_topk)
    routing_map = torch.zeros(tokens, num_experts, device=device, dtype=torch.bool)
    routing_map.scatter_(1, topk_idx, True)
    return {
        "x": x,
        "topk_idx": topk_idx.to(torch.int32).contiguous(),
        "probs_topk": probs_topk.contiguous(),
        "probs_full": probs_full.contiguous(),
        "routing_map": routing_map.contiguous(),
    }


def _gather_rank_max(local_times: List[float]) -> List[float]:
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_times)
    if dist.get_rank() != 0:
        return []
    iters = min(len(t) for t in gathered if t is not None)
    return [max(gathered[r][i] for r in range(len(gathered))) for i in range(iters)]


def _run_deepep(
    data: Dict[str, torch.Tensor],
    warmup: int,
    iters: int,
) -> Dict[str, List[float]]:
    import primus_turbo.pytorch as pt

    rank = dist.get_rank()
    world = dist.get_world_size()
    ep_group = dist.group.WORLD
    tp_group = _build_tp_groups_for_tp1(world, rank)
    tp_ep_group = ep_group
    num_experts = data["routing_map"].shape[1]
    topk = data["topk_idx"].shape[1]
    num_tokens = data["x"].shape[0]
    num_worst_tokens = num_tokens * world

    dispatcher = pt.modules.DeepEPTokenDispatcher(
        num_experts=num_experts,
        router_topk=topk,
        ep_group=ep_group,
        tp_group=tp_group,
        tp_ep_group=tp_ep_group,
        expert_capacity_factor=None,
        permute_fusion=True,
        deepep_use_comm_stream=False,
        deepep_num_use_cu=20,
        deepep_num_worst_tokens=num_worst_tokens,
        deepep_use_cuda_num_tokens_per_expert=True,
        deepep_async_finish=True,
        deepep_allocate_on_comm_stream=True,
    )

    def _step():
        hidden_states, probs = dispatcher._pre_dispatch(
            data["x"], data["probs_full"], data["routing_map"], token_indices=None
        )
        dispatched_tokens, dispatched_probs = dispatcher._exec_dispatch(hidden_states, probs)
        permuted_input, _, _ = dispatcher._post_dispatch(dispatched_tokens, dispatched_probs)
        combined_tokens = dispatcher._exec_combine(dispatcher._pre_combine(permuted_input))
        _ = dispatcher._post_combine(combined_tokens)

    dist.barrier()
    torch.cuda.synchronize()
    for _ in range(warmup):
        _step()
    dist.barrier()
    torch.cuda.synchronize()

    dispatch_ms, combine_ms, total_ms = [], [], []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        hidden_states, probs = dispatcher._pre_dispatch(
            data["x"], data["probs_full"], data["routing_map"], token_indices=None
        )
        dispatched_tokens, dispatched_probs = dispatcher._exec_dispatch(hidden_states, probs)
        permuted_input, _, _ = dispatcher._post_dispatch(dispatched_tokens, dispatched_probs)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        combined_tokens = dispatcher._exec_combine(dispatcher._pre_combine(permuted_input))
        _ = dispatcher._post_combine(combined_tokens)
        torch.cuda.synchronize()
        t2 = time.perf_counter_ns()
        dispatch_ms.append((t1 - t0) / 1e6)
        combine_ms.append((t2 - t1) / 1e6)
        total_ms.append((t2 - t0) / 1e6)
    dist.barrier()
    return {"dispatch": dispatch_ms, "combine": combine_ms, "total": total_ms}


def _run_comet_ll(
    data: Dict[str, torch.Tensor],
    warmup: int,
    iters: int,
) -> Dict[str, List[float]]:
    from triton_dist.layers.nvidia import EPLowLatencyAllToAllLayer
    from triton_dist.test.nvidia.ep_a2a_utils import dequant_fp8_bf16
    from triton_dist.utils import init_nvshmem_by_torch_process_group, is_shmem_initialized

    rank = dist.get_rank()
    world = dist.get_world_size()
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
    if not is_shmem_initialized():
        init_nvshmem_by_torch_process_group(dist.group.WORLD)

    layer = EPLowLatencyAllToAllLayer(
        max_m=data["x"].shape[0],
        hidden=data["x"].shape[1],
        topk=data["topk_idx"].shape[1],
        online_quant_fp8=True,
        rank=rank,
        num_experts=data["routing_map"].shape[1],
        local_world_size=local_world,
        world_size=world,
        dtype=torch.bfloat16,
        enable_profiling=False,
    )

    def _step():
        recv_token, recv_scale, _, dispatch_meta = layer.dispatch(
            send_tokens=data["x"], send_scales=None, topk_indices=data["topk_idx"]
        )
        combine_input = dequant_fp8_bf16(recv_token, recv_scale)
        _ = layer.combine(combine_input, data["topk_idx"], data["probs_topk"], dispatch_meta)

    dist.barrier()
    torch.cuda.synchronize()
    for _ in range(warmup):
        _step()
    dist.barrier()
    torch.cuda.synchronize()

    dispatch_ms, combine_ms, total_ms = [], [], []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        recv_token, recv_scale, _, dispatch_meta = layer.dispatch(
            send_tokens=data["x"], send_scales=None, topk_indices=data["topk_idx"]
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        combine_input = dequant_fp8_bf16(recv_token, recv_scale)
        _ = layer.combine(combine_input, data["topk_idx"], data["probs_topk"], dispatch_meta)
        torch.cuda.synchronize()
        t2 = time.perf_counter_ns()
        dispatch_ms.append((t1 - t0) / 1e6)
        combine_ms.append((t2 - t1) / 1e6)
        total_ms.append((t2 - t0) / 1e6)
    dist.barrier()
    layer.finalize()
    return {"dispatch": dispatch_ms, "combine": combine_ms, "total": total_ms}


def _summarize(name: str, stats: Dict[str, List[float]]) -> Dict[str, float]:
    out = {"case": name}
    for part in ("dispatch", "combine", "total"):
        out[f"{part}_p50"] = _pct(stats[part], 50.0)
        out[f"{part}_p95"] = _pct(stats[part], 95.0)
    return out


def run_moe_dispatch_compare_benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for moe-dispatch-compare benchmark.")

    if not dist.is_initialized():
        # Primus benchmark entry skips PG init when WORLD_SIZE=1.
        # Initialize a single-rank group to keep codepath consistent.
        dist.init_process_group(
            backend="nccl",
            rank=0,
            world_size=1,
            init_method="tcp://127.0.0.1:29501",
            device_id=torch.device("cuda:0"),
        )

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if args.num_experts % dist.get_world_size() != 0:
        raise ValueError("num-experts must be divisible by world size")

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    data = _make_inputs(
        tokens=args.tokens,
        hidden=args.hidden,
        topk=args.topk,
        num_experts=args.num_experts,
        device=device,
    )

    cases = [x.strip().lower() for x in args.cases.split(",") if x.strip()]
    supported = {"deepep", "comet-ll"}
    invalid = [c for c in cases if c not in supported]
    if invalid:
        raise ValueError(f"Unsupported cases: {invalid}. Supported: {sorted(supported)}")

    if rank == 0:
        print(
            f"[MoE-Dispatch-Bench] world={dist.get_world_size()} tokens={args.tokens} "
            f"hidden={args.hidden} experts={args.num_experts} topk={args.topk} "
            f"warmup={args.warmup} iters={args.iters} cases={cases}"
        )

    rows = []
    for case_name in cases:
        if case_name == "deepep":
            local = _run_deepep(data, args.warmup, args.iters)
        else:
            local = _run_comet_ll(data, args.warmup, args.iters)
        rankmax = {
            "dispatch": _gather_rank_max(local["dispatch"]),
            "combine": _gather_rank_max(local["combine"]),
            "total": _gather_rank_max(local["total"]),
        }
        if rank == 0:
            rows.append(_summarize(case_name, rankmax))

    if rank == 0:
        print("\n=== MoE Dispatcher Compare (rank-max latency, ms) ===")
        print("case      dispatch_p50  dispatch_p95  combine_p50  combine_p95  total_p50  total_p95")
        for r in rows:
            print(
                f"{r['case']:<9} "
                f"{r['dispatch_p50']:>12.3f} {r['dispatch_p95']:>12.3f} "
                f"{r['combine_p50']:>12.3f} {r['combine_p95']:>12.3f} "
                f"{r['total_p50']:>10.3f} {r['total_p95']:>10.3f}"
            )


def build_moe_dispatch_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoE dispatcher compare benchmark (DeepEP vs Comet-LL)")
    add_moe_dispatch_compare_parser(parser)
    return parser


if __name__ == "__main__":
    parser = build_moe_dispatch_compare_parser()
    cli_args = parser.parse_args()
    run_moe_dispatch_compare_benchmark(cli_args)
