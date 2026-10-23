"""Why does Turbo's aiter fwd take 2x TE's time with the same fmha_fwd_hd128_bf16_causal kernel?"""

import torch
from aiter.ops.mha import _flash_attn_forward

S, B, HQ, HKV, D = 4096, 8, 32, 4, 128


def timed(fn, iters=20):
    for _ in range(5):
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


torch.manual_seed(0)
sbhd = [torch.randn(S, B, h, D, device="cuda", dtype=torch.bfloat16) for h in (HQ, HKV, HKV)]
layouts = {
    "sbhd-view": [t.permute(1, 0, 2, 3) for t in sbhd],
    "bshd-contig": [t.permute(1, 0, 2, 3).contiguous() for t in sbhd],
}
q, k, v = layouts["sbhd-view"]
out = torch.empty(S, B, HQ, D, device="cuda", dtype=torch.bfloat16).permute(1, 0, 2, 3)
k32, v32 = (t.repeat_interleave(HQ // HKV, dim=2) for t in (k, v))
for causal in (True, False):
    for name, (kk, vv) in (("gqa 32/4", (k, v)), ("mha 32/32", (k32, v32))):

        def run():
            _flash_attn_forward(
                q, kk, vv, 0.0, D**-0.5, causal, -1, -1, 0, None, None, None, None, None,
                True, False, out=out,
            )

        print(f"causal={causal} {name}: {timed(run):8.1f} us")
