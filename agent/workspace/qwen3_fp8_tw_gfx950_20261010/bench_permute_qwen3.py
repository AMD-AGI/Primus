"""moe_permute / moe_unpermute at the Qwen3-30B-A3B EP8 dispatcher call (hidden 2048, 16 local
experts, top-8, pad_multiple 0, topk probs): HIP (TURBO) vs TRITON backends, even and random routing.
"""

import os

import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import BackendType

H = int(os.environ.get("H", "2048"))
E_LOCAL = int(os.environ.get("E_LOCAL", "16"))
TOPK, RANKS = 8, 8


def timed(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return sorted(ts)[len(ts) // 2]


def dispatched_indices(routing):
    """Local expert ids per received token (-1 = expert on another rank), as DeepEP returns them."""
    if routing == "even":
        # Each token goes to one rank with 8 consecutive experts.
        n = 32768
        base = (torch.arange(n, device="cuda") % (E_LOCAL // TOPK)) * TOPK
        return (base[:, None] + torch.arange(TOPK, device="cuda")[None, :]).int()
    g = torch.Generator(device="cuda").manual_seed(0)
    scores = torch.rand(32768 * RANKS, E_LOCAL * RANKS, device="cuda", generator=g)
    experts = scores.topk(TOPK, dim=-1).indices
    local = (experts // E_LOCAL) == 0
    keep = local.any(dim=-1)
    idx = torch.where(local, experts % E_LOCAL, torch.full_like(experts, -1))[keep]
    return idx.int()


for routing in ("even", "random"):
    idx = dispatched_indices(routing)
    n = idx.shape[0]
    rows = int((idx >= 0).sum())
    x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(n, TOPK, device="cuda", dtype=torch.float32)
    for backend in (BackendType.TURBO, BackendType.TRITON):

        def permute(x_=x, probs_=probs):
            return turbo.ops.moe_permute(
                x_, topk_indices=idx, num_local_experts=E_LOCAL, num_topk=TOPK, pad_multiple=0,
                num_permuted_tokens=rows, probs=probs_, probs_layout="topk", backend=backend,
            )

        out, row_id_map, *_rest, pprobs = permute()

        def unpermute(y=out):
            return turbo.ops.moe_unpermute(
                y, row_id_map, restore_shape=x.shape, num_local_experts=E_LOCAL, pad_multiple=0,
                backend=backend,
            )

        xg = x.clone().requires_grad_()
        pg = probs.clone().requires_grad_()
        yg = out.detach().clone().requires_grad_()
        gy = torch.randn_like(out)
        gp = torch.randn_like(pprobs)
        gx = torch.randn_like(x)

        def permute_fb():
            o, _, *_r, pp = permute(xg, pg)
            torch.autograd.backward((o, pp), (gy, gp))

        def unpermute_fb():
            unpermute(yg)[0].backward(gx)

        t_p, t_u = timed(permute), timed(unpermute)
        t_pfb, t_ufb = timed(permute_fb), timed(unpermute_fb)
        # permute fwd: read n*H, write rows*H; unpermute fwd: read rows*H, write n*H (bf16)
        gb = (n + rows) * H * 2 / 1e9
        print(
            f"{routing:6s} {backend.name:6s} tokens {n:6d} rows {rows:6d} | permute {t_p:6.1f} us "
            f"({gb / t_p * 1e3:4.2f} TB/s) bwd {t_pfb - t_p:6.1f} | unpermute {t_u:6.1f} us "
            f"({gb / t_u * 1e3:4.2f} TB/s) bwd {t_ufb - t_u:6.1f}"
        )
