"""hipBLASLt FP8 tensorwise: NT-with-explicit-transposes (current gfx950 bwd) vs NN / TN.

Shapes: Qwen3-30B-A3B dense projections per microbatch (M = 32768 tokens).
"""

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl

TW = ScalingGranularity.TENSORWISE.value
HB = BackendType.HIPBLASLT.value
E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2


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


def tr(x):
    return torch.ops.primus_turbo_cpp_extension.transpose_2d(x, 0, 1)


def g(a, ta, b, tb, tc=False):
    one = torch.ones((), device="cuda")
    return gemm_fp8_impl(a, one, ta, b, one, tb, torch.bfloat16, tc, granularity=TW, default_backend=HB)


M = 32768
for name, N, K in [("qkv", 5120, 2048), ("proj", 2048, 4096)]:
    x = (torch.randn(M, K, device="cuda") * 0.5).to(E4)
    w = (torch.randn(N, K, device="cuda") * 0.5).to(E4)
    go = (torch.randn(M, N, device="cuda") * 0.5).to(E5)
    fl = 2 * M * N * K

    fwd = bench(lambda: g(x, False, w, True))
    # dgrad: current = transpose(w) + NT ; alt = NN
    w_t = tr(w)
    d_nt = bench(lambda: g(go, False, w_t, True))
    d_tr = bench(lambda: tr(w))
    d_nn = bench(lambda: g(go, False, w, False))
    # wgrad: current = transpose(x) [fwd] + transpose(go) + NT(a_t, go_t, trans_c) ; alt = TN(go, x)
    x_t, go_t = tr(x), tr(go)
    w_nt = bench(lambda: g(x_t, False, go_t, True, True))
    w_tr = bench(lambda: tr(x)) + bench(lambda: tr(go))
    w_tn = bench(lambda: g(go, True, x, False))
    ref_d = g(go, False, w_t, True).float()
    alt_d = g(go, False, w, False).float()
    ref_w = g(x_t, False, go_t, True, True).float()
    alt_w = g(go, True, x, False).float()
    ed = ((ref_d - alt_d).norm() / ref_d.norm()).item()
    ew = ((ref_w - alt_w).norm() / ref_w.norm()).item()
    print(f"[{name}] M={M} N={N} K={K}")
    print(f"  fwd NT            : {fwd:7.1f} us {fl / fwd / 1e6:6.0f} TFLOP/s")
    print(f"  dgrad NT+transpose: {d_nt:7.1f} + {d_tr:5.1f} us   NN: {d_nn:7.1f} us ({fl / d_nn / 1e6:4.0f} TF/s)  rel diff {ed:.1e}")
    print(f"  wgrad NT+transpose: {w_nt:7.1f} + {w_tr:5.1f} us   TN: {w_tn:7.1f} us ({fl / w_tn / 1e6:4.0f} TF/s)  rel diff {ew:.1e}")
