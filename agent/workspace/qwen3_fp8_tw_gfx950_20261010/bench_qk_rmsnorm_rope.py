"""Item 5 microbench: Qwen3-30B-A3B q/k RMSNorm + RoPE at seq 4096, MBS 8, NG=4 KV groups,
NPG=8 q heads per group, head_dim 128 (packed qkv [S, B, NG, (NPG+2)*D]).

unfused: Megatron's current path -- split the packed qkv, TE RMSNorm on q and k (with the
         reshape copies the split views force), TE fused RoPE on q and k, v made contiguous.
fused:   primus_turbo fused_qkv_rmsnorm_rope (one read of qkv, one write of q/k/v).
Reports fwd and fwd+bwd wall time per call (CUDA events) and fused HBM bandwidth.
"""

import os

import torch
import transformer_engine.pytorch as tep
from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

from primus_turbo.pytorch.ops.rope import fused_qkv_rmsnorm_rope

S, B, NG, NPG, D = (int(v) for v in os.environ.get("SHAPE", "4096,8,4,8,128").split(","))
EPS = 1e-6
dev = "cuda"


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


torch.manual_seed(0)
split = [NPG * D, D, D]
qkv = torch.randn(S, B, NG, sum(split), device=dev, dtype=torch.bfloat16, requires_grad=True)
half = torch.randn(S, D // 2, device=dev)
freqs = torch.cat((half, half), -1).reshape(S, 1, 1, D).contiguous()
q_norm = tep.RMSNorm(D, eps=EPS, params_dtype=torch.bfloat16).to(dev)
k_norm = tep.RMSNorm(D, eps=EPS, params_dtype=torch.bfloat16).to(dev)
with torch.no_grad():
    q_norm.weight.normal_()
    k_norm.weight.normal_()
q_gamma = q_norm.weight.detach().clone().requires_grad_()
k_gamma = k_norm.weight.detach().clone().requires_grad_()


def unfused():
    q, k, v = torch.split(qkv, split, dim=3)
    q = q.reshape(S, B, NG * NPG, D)
    k = k.reshape(S, B, NG, D)
    v = v.reshape(S, B, NG, D)
    q = apply_rotary_pos_emb(q_norm(q), freqs, tensor_format="sbhd", fused=True)
    k = apply_rotary_pos_emb(k_norm(k), freqs, tensor_format="sbhd", fused=True)
    return q, k, v.contiguous()


def fused():
    return fused_qkv_rmsnorm_rope(qkv, q_gamma, k_gamma, freqs, split, EPS)


grads = None


def fwd_bwd(fn):
    def run():
        outs = fn()
        torch.autograd.backward(outs, grads)
        qkv.grad = None

    return run


with torch.no_grad():
    ref = unfused()
    out = fused()
grads = tuple(torch.randn_like(t) for t in ref)
diff = [(a.float() - b.float()).abs().max().item() for a, b in zip(out, ref)]

nbytes_fwd = qkv.numel() * 2 * 2  # read packed qkv, write q/k/v of the same size
nbytes_bwd = qkv.numel() * 2 * 3  # read dq/dk/dv + packed qkv, write dqkv
res = {}
for name, fn in (("unfused", unfused), ("fused", fused)):
    with torch.no_grad():
        t_f = bench(fn)
    t_fb = bench(fwd_bwd(fn))
    res[name] = (t_f, t_fb)
print(f"shape S={S} B={B} NG={NG} NPG={NPG} D={D}; fused vs unfused max|diff| q/k/v = {diff}")
for name, (t_f, t_fb) in res.items():
    print(f"{name:8s} fwd {t_f:7.1f} us   fwd+bwd {t_fb:7.1f} us   bwd {t_fb - t_f:7.1f} us")
t_f, t_fb = res["fused"]
print(
    f"fused fwd {nbytes_fwd / t_f / 1e6:.2f} TB/s, bwd {nbytes_bwd / (t_fb - t_f) / 1e6:.2f} TB/s; "
    f"speedup fwd {res['unfused'][0] / t_f:.2f}x fwd+bwd {res['unfused'][1] / t_fb:.2f}x"
)
