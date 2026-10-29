#!/usr/bin/env python3
"""gfx1250 variable-K wgrad: correctness of the beta=1 FlyDSL epilogue, and its time against
(FlyDSL beta=0 + main_grad.add_) and Triton beta=1, on the DSv3 EP4 expert shapes.

  PYTHONPATH=<turbo worktree> python3 bench_vk_accum.py
"""
import torch

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
    grouped_gemm_bf16_variable_k_flydsl_kernel as fly_vk,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
    grouped_gemm_variable_k_triton_kernel as tri_vk,
)

torch.manual_seed(0)
dev = "cuda"
G = 32


def lens_for(m_total, mode):
    if mode == "even":
        return torch.full((G,), m_total // G, dtype=torch.int64)
    w = torch.rand(G) + 0.2
    w[3] = 0  # an empty expert
    lens = (w / w.sum() * m_total).long()
    lens[0] += m_total - lens.sum()
    return lens


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


def ref_vk(a, b, offs):
    out = torch.empty(G, a.shape[1], b.shape[1], device=dev, dtype=torch.float32)
    for g in range(G):
        lo, hi = int(offs[g]), int(offs[g + 1])
        out[g] = a[lo:hi].float().t() @ b[lo:hi].float()
    return out


def case(name, m_total, out_m, out_n, mode):
    lens = lens_for(m_total, mode)
    offs = torch.zeros(G + 1, dtype=torch.int64)
    offs[1:] = lens.cumsum(0)
    lens_d, offs_d = lens.to(dev), offs.to(dev)
    a = torch.randn(m_total, out_m, device=dev, dtype=torch.bfloat16) * 0.1
    b = torch.randn(m_total, out_n, device=dev, dtype=torch.bfloat16) * 0.1
    main = torch.randn(G, out_m, out_n, device=dev, dtype=torch.bfloat16)

    ref = main.float() + ref_vk(a, b, offs)
    got = main.clone()
    fly_vk(a, b, offs_d, masked_k=lens_d, out=got, accumulate=True)
    torch.cuda.synchronize()
    base = main.clone()
    base.add_(fly_vk(a, b, offs_d, masked_k=lens_d))
    scale = ref.abs().max().item()
    err_acc = (got.float() - ref).abs().max().item() / scale
    err_base = (base.float() - ref).abs().max().item() / scale
    ok = err_acc <= max(2 * err_base, 1e-2)

    work = main.clone()
    t_sep = bench(lambda: work.add_(fly_vk(a, b, offs_d, masked_k=lens_d)))
    t_fly0 = bench(lambda: fly_vk(a, b, offs_d, masked_k=lens_d))
    t_acc = bench(lambda: fly_vk(a, b, offs_d, masked_k=lens_d, out=work, accumulate=True))
    try:
        t_tri = bench(lambda: tri_vk(a, b, offs_d, beta=1.0, out=work))
    except Exception as exc:  # noqa: BLE001
        t_tri = float("nan")
        print(f"  triton beta=1 failed: {type(exc).__name__}: {exc}")
    print(
        f"{name:5s} {mode:7s} M={m_total} out=[{G},{out_m},{out_n}]  "
        f"rel_err acc={err_acc:.2e} base={err_base:.2e} {'OK' if ok else 'MISMATCH'} | "
        f"fly+add {t_sep:7.1f} us  (fly {t_fly0:7.1f})  fly-accum {t_acc:7.1f} us  triton-b1 {t_tri:7.1f} us"
    )
    return ok


all_ok = True
for mode in ("even", "uneven"):
    # fc1 wgrad: dout [M, 2*ffn] x act [M, hidden] -> [G, 4096, 7168]
    all_ok &= case("fc1", 65536, 4096, 7168, mode)
    # fc2 wgrad: dout [M, hidden] x act [M, ffn] -> [G, 7168, 2048]
    all_ok &= case("fc2", 65536, 7168, 2048, mode)
# a short last block on both output axes, so the backed-off window is exercised
all_ok &= case("tail", 8192, 4096 - 96, 2048 + 136, "uneven")
print("ALL OK" if all_ok else "FAILURES")
