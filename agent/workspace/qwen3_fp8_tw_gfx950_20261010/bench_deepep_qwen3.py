"""Item 3 microbench: Turbo DeepEP intranode at Qwen3-30B-A3B EP8 (32768 tokens/rank, hidden 2048,
128 experts, top-8). Default dispatch/combine configs (what training uses); sweeps num_sms and
compares BF16 vs FP8 (per-128 scales) dispatch, under random top-k and Primus "even" routing.
Each phase is timed separately with CUDA events (median over ITERS, max over ranks); the
non-cached dispatch includes its CPU wait on the notify kernel, the cached one does not.
"""

import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../benchmark/ops/training"))
from deep_ep.utils import init_dist  # noqa: E402

from primus_turbo.pytorch import deep_ep  # noqa: E402

T, H, E, K = 32768, 2048, 128, 8
SMS = [int(s) for s in os.environ.get("SMS", "64,80,96").split(",")]
ITERS = 20


def topk_for(routing, rank):
    if routing == "even":  # PrimusTopKRouter._force_even_routing
        return (torch.arange(T * K, device="cuda").view(T, K) % E).to(torch.int64)
    g = torch.Generator(device="cuda").manual_seed(rank)
    scores = torch.randn((T, E), device="cuda", generator=g).abs() + 1
    return torch.topk(scores, K, dim=-1, sorted=False)[1]


def timed(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return out, s.elapsed_time(e) * 1e3


def worker(local_rank, num_local_ranks):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    buf = deep_ep.Buffer(group, int(2e9), 0, low_latency_mode=False, explicitly_destroy=True)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    x_fp8 = x.to(torch.float8_e4m3fn)
    x_scales = torch.ones(T, H // 128, device="cuda", dtype=torch.float32)
    rows = []
    for routing in ("random", "even"):
        topk_idx = topk_for(routing, rank)
        topk_w = torch.rand(T, K, device="cuda", dtype=torch.float32)
        for sms in SMS:
            deep_ep.Buffer.set_num_sms(sms)
            for prec in ("bf16", "fp8"):
                inp = x if prec == "bf16" else (x_fp8, x_scales)
                ts = {"layout": [], "dispatch": [], "dispatch_cached": [], "combine": []}
                for it in range(ITERS + 3):
                    dist.barrier()
                    (ntr, _, nte, itr, _), t_l = timed(lambda: buf.get_dispatch_layout(topk_idx, E))
                    dist.barrier()
                    (recv, _, rw, _, handle, _), t_d = timed(
                        lambda: buf.dispatch(
                            inp,
                            num_tokens_per_rank=ntr,
                            is_token_in_rank=itr,
                            num_tokens_per_expert=nte,
                            topk_idx=topk_idx,
                            topk_weights=topk_w,
                        )
                    )
                    dist.barrier()
                    _, t_dc = timed(lambda: buf.dispatch(inp, handle=handle))
                    r = recv[0] if isinstance(recv, tuple) else recv
                    t_c = 0.0
                    if prec == "bf16":
                        dist.barrier()
                        _, t_c = timed(lambda: buf.combine(r, handle, topk_weights=rw))
                    if it >= 3:
                        for k, v in zip(ts, (t_l, t_d, t_dc, t_c)):
                            ts[k].append(v)
                med = torch.tensor([statistics.median(v) for v in ts.values()], device="cuda")
                dist.all_reduce(med, op=dist.ReduceOp.MAX)
                rows.append((routing, sms, prec, r.shape[0], med.tolist()))
    if rank == 0:
        print("routing sms prec recv_tokens | layout dispatch dispatch_cached combine (us, max over ranks)")
        for routing, sms, prec, nrecv, m in rows:
            print(
                f"{routing:6s} sms={sms:3d} {prec:4s} recv={nrecv:6d} | "
                f"layout {m[0]:6.0f}  dispatch {m[1]:6.0f}  cached {m[2]:6.0f}  combine {m[3]:6.0f}"
            )
    buf.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.spawn(worker, args=(8,), nprocs=8)
