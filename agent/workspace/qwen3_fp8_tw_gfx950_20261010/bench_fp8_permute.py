"""Validate + time tensorwise-FP8 permute (fwd) and FP8 unpermute grad (bwd) at Qwen3 shapes.

Per-rank shape (EP8, MBS8, seq 4096): 32768 dispatched tokens, H=2048, 16 local experts,
topk 8 with all 8 experts on one rank (even routing), pad 16.
"""

import sys

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor
from primus_turbo.pytorch.ops.grouped_mlp_fp8 import grouped_mlp_fp8
from primus_turbo.pytorch.ops.moe.moe_permute import moe_permute, moe_unpermute

T, H, E, TOPK, PAD, I = 32768, 2048, 16, 8, 16, 768
TURBO = BackendType.TURBO
dev = "cuda"
torch.manual_seed(0)


def make_inputs(even=True):
    if even:
        # all 8 experts of a token on this rank, round robin
        base = (torch.arange(T, device=dev) * TOPK) % E
        idx = (base[:, None] + torch.arange(TOPK, device=dev)[None]) % E
    else:
        idx = torch.argsort(torch.rand(T, E, device=dev), dim=1)[:, :TOPK]
    idx = idx.to(torch.int64).contiguous()
    tokens = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    probs = torch.rand(T, TOPK, device=dev, dtype=torch.float32)
    return tokens, idx, probs


def permute(tokens, idx, probs, qdtype=None):
    return moe_permute(
        tokens,
        topk_indices=idx,
        num_local_experts=E,
        num_topk=TOPK,
        pad_multiple=PAD,
        num_permuted_tokens=-1,
        probs=probs,
        probs_layout="topk",
        backend=TURBO,
        quantize_dtype=qdtype,
    )


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


def check_permute(even):
    tokens, idx, probs = make_inputs(even)
    # an unrouted outlier token must not reach the scale
    idx[5] = -1
    tokens[5] = 3.0e4
    ref, ref_map, ref_tpe, _, _, ref_probs = permute(tokens, idx, probs)
    ref_q = QuantizedTensor.quantize(
        ref, float8_e4m3, ScalingGranularity.TENSORWISE, axis=-1, group_lens=ref_tpe, pad_align_last=128
    )
    out, row_map, tpe, _, _, out_probs = permute(tokens, idx, probs, float8_e4m3)
    assert isinstance(out, QuantizedTensor) and out._is_grouped_tensor
    ok = (
        torch.equal(out.qdata.view(torch.uint8), ref_q.qdata.view(torch.uint8))
        and torch.equal(out.scale_inv, ref_q.scale_inv)
        and torch.equal(out.group_offs, ref_q.group_offs)
        and torch.equal(tpe, ref_tpe)
        and torch.equal(out_probs, ref_probs)
        and out.shape == ref.shape
    )
    print(f"permute fwd bit-exact (even={even}): {ok}  scale_inv={out.scale_inv.item():.6e}")
    return ok


class _Capture(torch.autograd.Function):
    grads = []

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        _Capture.grads.append(g)
        return torch.zeros(g.shape, dtype=torch.bfloat16, device=dev)


def check_unpermute_grad(even):
    tokens, idx, probs = make_inputs(even)
    idx[7] = -1
    permuted, row_map, tpe, _, _, _ = permute(tokens, idx, probs)
    gout = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    gout[7] = 3.0e4

    x = permuted.detach().requires_grad_()
    out, _ = moe_unpermute(
        x, row_map, restore_shape=tokens.shape, num_local_experts=E, pad_multiple=PAD, backend=TURBO
    )
    (ref_grad,) = torch.autograd.grad(out, x, gout)
    ref_q = QuantizedTensor.quantize(ref_grad, float8_e5m2, ScalingGranularity.TENSORWISE, axis=-1, pad_align_last=128)

    _Capture.grads.clear()
    x2 = permuted.detach().requires_grad_()
    out2, _ = moe_unpermute(
        _Capture.apply(x2),
        row_map,
        restore_shape=tokens.shape,
        num_local_experts=E,
        pad_multiple=PAD,
        backend=TURBO,
        grad_quantize_dtype=float8_e5m2,
    )
    out2.backward(gout)
    q = _Capture.grads[0]
    ok = (
        isinstance(q, QuantizedTensor)
        and torch.equal(q.qdata.view(torch.uint8), ref_q.qdata.view(torch.uint8))
        and torch.equal(q.scale_inv, ref_q.scale_inv)
        and q.shape == ref_grad.shape
    )
    print(f"unpermute grad bit-exact (even={even}): {ok}")
    return ok


def check_mlp_e2e(fmt):
    """permute -> grouped_mlp_fp8 -> unpermute: FP8 handoff vs bf16 handoff, all bitwise."""
    from primus_turbo.pytorch.ops.utils import _get_fp8_dtype

    tokens, idx, probs = make_inputs(True)
    w1 = (torch.randn(E, 2 * I, H, device=dev) * 0.02).bfloat16()
    w2 = (torch.randn(E, H, I, device=dev) * 0.02).bfloat16()
    gout = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    cfg = Float8QuantConfig(format=fmt)
    x_dtype, g_dtype = _get_fp8_dtype(fmt, True), _get_fp8_dtype(fmt, False)
    print(f"  mlp e2e format={fmt} x={x_dtype} grad={g_dtype}")

    def run(fp8):
        t = tokens.clone().requires_grad_()
        p = probs.clone().requires_grad_()
        a, b = w1.clone().requires_grad_(), w2.clone().requires_grad_()
        x, row_map, tpe, _, _, pp = permute(t, idx, p, x_dtype if fp8 else None)
        y = grouped_mlp_fp8(x, a, b, tpe, probs=pp, trans_w1=True, trans_w2=True, config=cfg, activation="silu")
        o, _ = moe_unpermute(
            y,
            row_map,
            restore_shape=t.shape,
            num_local_experts=E,
            pad_multiple=PAD,
            backend=TURBO,
            grad_quantize_dtype=g_dtype if fp8 else None,
        )
        o.backward(gout)
        return [o.detach(), t.grad, p.grad, a.grad, b.grad]

    ref, new = run(False), run(True)
    names = ["out", "grad_tokens", "grad_probs", "grad_w1", "grad_w2"]
    ok = True
    for n, r, g in zip(names, ref, new):
        same = torch.equal(r, g)
        ok &= same
        print(f"  mlp e2e {n}: bit-exact={same}")
    return ok


def timing():
    tokens, idx, probs = make_inputs(True)
    permuted, row_map, tpe, _, _, _ = permute(tokens, idx, probs)
    ndisp = torch.full((1,), T, dtype=torch.int32, device=dev)

    def ref_fwd():
        p, _, tpe_, _, _, _ = permute(tokens, idx, probs)
        QuantizedTensor.quantize(
            p, float8_e4m3, ScalingGranularity.TENSORWISE, axis=-1, group_lens=tpe_, pad_align_last=128
        )

    def new_fwd():
        permute(tokens, idx, probs, float8_e4m3)

    def perm_only():
        permute(tokens, idx, probs)

    from primus_turbo.pytorch.kernels.moe.moe_permute_impl import (
        moe_permute_routed_amax_impl,
    )

    def amax_only():
        moe_permute_routed_amax_impl(tokens, row_map, ndisp, E)

    print(f"fwd  bf16 permute only          : {bench(perm_only):8.1f} us")
    print(f"fwd  bf16 permute + quantize    : {bench(ref_fwd):8.1f} us")
    print(f"fwd  fp8 permute (new)          : {bench(new_fwd):8.1f} us")
    print(f"     routed amax only           : {bench(amax_only):8.1f} us")

    gout = torch.randn(T, H, device=dev, dtype=torch.bfloat16)

    def bwd(qd):
        x = permuted.detach().requires_grad_()
        y = _Capture.apply(x) if qd is not None else x
        out, _ = moe_unpermute(
            y, row_map, restore_shape=tokens.shape, num_local_experts=E, pad_multiple=PAD, backend=TURBO,
            grad_quantize_dtype=qd,
        )
        if qd is None:
            (g,) = torch.autograd.grad(out, x, gout)
            QuantizedTensor.quantize(g, float8_e5m2, ScalingGranularity.TENSORWISE, axis=-1, pad_align_last=128)
        else:
            out.backward(gout)
            _Capture.grads.clear()

    from primus_turbo.pytorch.kernels.moe.moe_permute_impl import moe_permute_impl
    from primus_turbo.pytorch.ops.moe.moe_permute import _permute_quantized_tensorwise

    num_permuted = permuted.shape[0]

    def ref_bwd():
        g, _, _ = moe_permute_impl(
            TURBO, gout, row_map, None, num_permuted, E, H, PAD, None, None, 0, False, 0
        )
        QuantizedTensor.quantize(g, float8_e5m2, ScalingGranularity.TENSORWISE, axis=-1, pad_align_last=128)

    def new_bwd():
        _permute_quantized_tensorwise(gout, row_map, None, num_permuted, E, PAD, None, 0, float8_e5m2, None)

    print(f"bwd  bf16 permute(grad) + quantize: {bench(ref_bwd):8.1f} us")
    print(f"bwd  fp8 permute(grad) (new)      : {bench(new_bwd):8.1f} us")

    from torch.profiler import ProfilerActivity, profile

    for name, fn in (("ref_fwd", ref_fwd), ("new_fwd", new_fwd), ("ref_bwd", ref_bwd), ("new_bwd", new_bwd)):
        fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
        print(f"--- {name} per-call kernel times")
        for ev in sorted(prof.key_averages(), key=lambda e: -e.device_time_total):
            if ev.device_time_total > 0:
                print(f"   {ev.device_time_total / 10:8.1f} us  {ev.key[:90]}")


if __name__ == "__main__":
    ok = check_permute(True) & check_permute(False)
    ok &= check_unpermute_grad(True) & check_unpermute_grad(False)
    from primus_turbo.pytorch.core.low_precision import Format

    ok &= check_mlp_e2e(Format.E4M3)
    ok &= check_mlp_e2e(Format.HYBRID)
    print("ALL BIT-EXACT:", ok)
    if "--time" in sys.argv:
        timing()
