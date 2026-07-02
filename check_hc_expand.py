#!/usr/bin/env python3
"""Correctness + microbench for the fused HC expand Triton kernel vs eager."""
import importlib.util
import os
import time

import torch

os.environ.setdefault("PRIMUS_HC_EXPAND_TRITON", "1")
# Load the kernel module directly by path to avoid the heavy primus/__init__ chain.
_MOD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "primus/backends/megatron/core/transformer/v4_attention_kernels/_triton/hc_expand.py",
)
_spec = importlib.util.spec_from_file_location("hc_expand", _MOD)
_hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hc)
hc_expand_triton = _hc.hc_expand_triton


def eager_expand(x, out, post, comb):
    write = post.unsqueeze(-1) * out.unsqueeze(-2)
    mix = torch.matmul(comb, x)
    return write + mix


def run(leading, K, D, dtype):
    dev = "cuda"
    torch.manual_seed(0)
    mk = lambda *s: torch.randn(*s, device=dev, dtype=dtype)  # noqa: E731
    x = mk(*leading, K, D)
    out = mk(*leading, D)
    post = mk(*leading, K)
    comb = torch.softmax(mk(*leading, K, K).float(), dim=-1).to(dtype)

    # fp32 reference (ground truth), plus eager and triton at the test dtype.
    x32, o32, p32, c32 = (t.float().clone().requires_grad_(True) for t in (x, out, post, comb))
    xa, oa, pa, ca = (t.clone().requires_grad_(True) for t in (x, out, post, comb))
    xb, ob, pb, cb = (t.clone().requires_grad_(True) for t in (x, out, post, comb))

    ref = eager_expand(x32, o32, p32, c32)
    eag = eager_expand(xa, oa, pa, ca)
    tri = hc_expand_triton(xb, ob, pb, cb)

    g32 = torch.randn_like(ref)
    g = g32.to(dtype)
    ref.backward(g32)
    eag.backward(g)
    tri.backward(g)

    def maxerr(a, b):
        return (a.float() - b.float()).abs().max().item()

    tag = f"{tuple(leading)} K={K} D={D} {str(dtype).split('.')[-1]}"
    ok = True
    for name, r, e, t in [
        ("fwd", ref, eag, tri),
        ("dx", x32.grad, xa.grad, xb.grad),
        ("dout", o32.grad, oa.grad, ob.grad),
        ("dpost", p32.grad, pa.grad, pb.grad),
        ("dcomb", c32.grad, ca.grad, cb.grad),
    ]:
        e_err = maxerr(r, e)  # eager-vs-fp32 (the dtype's intrinsic noise floor)
        t_err = maxerr(r, t)  # triton-vs-fp32
        # Pass if triton is no worse than eager, with an fp32 reduction-order
        # slack relative to the tensor magnitude (length-D sums reorder).
        slack = 1e-4 + 5e-5 * r.abs().max().item()
        passed = t_err <= max(1.5 * e_err, slack)
        ok = ok and passed
        print(f"  {name:6s} triton_err={t_err:.3e} eager_err={e_err:.3e} {'OK' if passed else 'FAIL'}")
    print(f"[{tag}] {'PASS' if ok else 'FAIL'}")
    return ok


def bench(leading, K, D, dtype, iters=50):
    dev = "cuda"
    x = torch.randn(*leading, K, D, device=dev, dtype=dtype, requires_grad=True)
    out = torch.randn(*leading, D, device=dev, dtype=dtype, requires_grad=True)
    post = torch.randn(*leading, K, device=dev, dtype=dtype, requires_grad=True)
    comb = (
        torch.softmax(torch.randn(*leading, K, K, device=dev).float(), dim=-1).to(dtype).requires_grad_(True)
    )
    g = torch.randn(*leading, K, D, device=dev, dtype=dtype)

    def step(fn):
        for t in (x, out, post, comb):
            t.grad = None
        y = fn(x, out, post, comb)
        y.backward(g)

    for fn, name in [(eager_expand, "eager"), (hc_expand_triton, "triton")]:
        step(fn)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            step(fn)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
        print(f"  {name:7s} fwd+bwd {ms:8.3f} ms/iter")


if __name__ == "__main__":
    print("=== correctness ===")
    all_ok = True
    all_ok &= run((4096,), 4, 7168, torch.bfloat16)
    all_ok &= run((4096,), 4, 7168, torch.float32)
    all_ok &= run((2, 17), 4, 320, torch.bfloat16)
    all_ok &= run((3,), 2, 128, torch.float32)
    all_ok &= run((5,), 8, 64, torch.bfloat16)
    print("\n=== bench (production shape B*S=4096, K=4, D=7168, bf16) ===")
    bench((4096,), 4, 7168, torch.bfloat16)
    print("\nALL", "PASS" if all_ok else "FAIL")
