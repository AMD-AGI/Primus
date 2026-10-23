"""Qwen3-30B-A3B attention (sbhd, GQA 32/4, hd128, causal, bf16): TE CK vs Turbo backends.

Env: SHAPE=S,B,HQ,HKV,D (default 4096,8,32,4,128), BACKENDS=te,flydsl,aiter
"""

import os

os.environ.setdefault("NVTE_FLASH_ATTN", "0")
os.environ.setdefault("NVTE_FUSED_ATTN", "1")
os.environ.setdefault("NVTE_CK_USES_BWD_V3", "1")

import torch  # noqa: E402  (torch before TE: LLVM option clash otherwise)

S, B, HQ, HKV, D = (int(v) for v in os.environ.get("SHAPE", "4096,8,32,4,128").split(","))
BACKENDS = os.environ.get("BACKENDS", "te,flydsl,aiter").split(",")
ITERS = int(os.environ.get("ITERS", "20"))
dev = "cuda"


def timed(fn, iters=ITERS, warmup=5):
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


def make_te():
    import transformer_engine.pytorch as te

    attn = te.DotProductAttention(
        HQ, D, num_gqa_groups=HKV, attention_dropout=0.0, attn_mask_type="causal", qkv_format="sbhd"
    )

    def run(q, k, v):
        return attn(q, k, v).view(S, B, HQ, D)

    return run


def make_turbo(backend_name):
    from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
    from primus_turbo.pytorch.ops import flash_attn_func

    backend = getattr(BackendType, backend_name.upper())

    def run(q, k, v):
        GlobalBackendManager.set_attn_backend(backend)
        o = flash_attn_func(
            q.permute(1, 0, 2, 3), k.permute(1, 0, 2, 3), v.permute(1, 0, 2, 3), causal=True
        )
        GlobalBackendManager.set_attn_backend(None)
        return o.permute(1, 0, 2, 3)

    return run


torch.manual_seed(0)
q0 = torch.randn(S, B, HQ, D, device=dev, dtype=torch.bfloat16)
k0 = torch.randn(S, B, HKV, D, device=dev, dtype=torch.bfloat16)
v0 = torch.randn(S, B, HKV, D, device=dev, dtype=torch.bfloat16)
do = torch.randn(S, B, HQ, D, device=dev, dtype=torch.bfloat16)

fwd_flops = 4 * B * HQ * S * S * D / 2
results, outs = {}, {}
for name in BACKENDS:
    run = make_te() if name == "te" else make_turbo(name)
    q, k, v = (t.clone().requires_grad_() for t in (q0, k0, v0))
    try:
        o = run(q, k, v)
        o.backward(do)
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: unsupported ({type(exc).__name__}: {str(exc)[:200]})")
        continue
    outs[name] = (o.detach().float(), q.grad.float(), k.grad.float(), v.grad.float())

    def fwd():
        with torch.no_grad():
            run(q0, k0, v0)

    def fwdbwd():
        q.grad = k.grad = v.grad = None
        run(q, k, v).backward(do)

    t_f, t_fb = timed(fwd), timed(fwdbwd)
    results[name] = (t_f, t_fb)
    if os.environ.get("PROFILE") == "1":
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                fwdbwd()
            torch.cuda.synchronize()
        print(f"--- {name} kernels (5 x fwd+bwd)")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=70))

print(f"shape S={S} B={B} HQ={HQ} HKV={HKV} D={D} causal bf16")
for name, (t_f, t_fb) in results.items():
    t_b = t_fb - t_f
    print(
        f"{name:8s} fwd {t_f:8.1f} us ({fwd_flops / t_f / 1e6:6.0f} TF/s)  bwd {t_b:8.1f} us "
        f"({2.5 * fwd_flops / t_b / 1e6:6.0f} TF/s)  fwd+bwd {t_fb:8.1f} us"
    )
if "te" in outs:
    ref = outs["te"]
    for name, got in outs.items():
        if name == "te":
            continue
        diffs = " ".join(
            f"{tag} max {(a - b).abs().max().item():.3g}"
            for tag, a, b in zip(("o", "dq", "dk", "dv"), ref, got)
        )
        print(f"{name} vs te: {diffs}")
