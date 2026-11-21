"""GPU check + microbench of the FP8-block Adam tile kernel (fused_adam_clip_fp8blk). NOT run yet.

  1. Adam bit-identity: tile kernel vs the unchanged chunk kernel fused_adam_clip (same inputs):
     fp32 master, moments, BF16 param equal bitwise (device FMA contraction included).
  2. Quant bytes: W / scale_inv equal Turbo quant_fp8_blockwise_for_weight_impl (UE8M0) of the
     chunk kernel's BF16 output; W^T == W^T copy; e8m0 / e8m0_t == exponent bytes.
  3. Timing (--bench): baseline = chunk Adam + Turbo weight quant (+ W^T copy, g90 dgrad) vs the
     tile kernel, on the DSv4-proxy FP8 weight set. Take kernel times from a rocprofv3 trace.

Run (dry first):
  ~/dsv41/bin/yqgpu -g 0 -- docker exec <ctr> timeout 60 python test_fp8blk_adam_gpu.py --dry
"""
import argparse
import os
import sys

os.environ.setdefault("PRIMUS_TURBO_FP8_BLOCK_UE8M0", "1")
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMUS = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = f"{PRIMUS}/primus/backends/megatron/patches/te_patches/fused_adam_clip_kernel.cu"
ext = load(name="primus_te_fused_adam_clip_v7_test", sources=[SRC], extra_cflags=["-O3"],
           extra_cuda_cflags=["-O3"], with_cuda=True, verbose=False)
from primus_turbo.pytorch.kernels.quantization.quantization_impl import quant_fp8_blockwise_for_weight_impl

DEV = "cuda"
CHUNK = 65536
LR, B1, B2, EPS, WD, STEP, MAXN = 3e-4, 0.9, 0.95, 1e-8, 0.1, 7, 1.0
# DSv4-Flash proxy FP8 block weights (per layer: wq_a, wkv, wq_b, wo_b, shared fc1/fc2, experts fc1/fc2)
FULL = [(1024, 4096), (576, 4096), (8192, 1024), (4096, 8192), (4096, 4096), (4096, 2048),
        (32, 4096, 4096), (32, 4096, 2048)]
DRY = [(256, 384), (300, 200), (2, 384, 136), (576, 512)]


def make(shapes, seed=0):
    torch.manual_seed(seed)
    ts = []
    for sh in shapes:
        p = torch.randn(sh, device=DEV) * 0.02
        g = (torch.randn(sh, device=DEV) * 1e-3).bfloat16()
        m = (torch.randn(sh, device=DEV) * 1e-4).bfloat16()
        v = (torch.rand(sh, device=DEV) * 1e-6).bfloat16()
        ts.append([g, p, m, v, torch.empty(sh, device=DEV, dtype=torch.bfloat16)])
    return ts


def clone(ts):
    return [[t.clone() for t in row] for row in ts]


def chunk_meta(ts):
    addr, sizes, b2t, coff, total = [], [], [], [], 0
    for i, (g, p, m, v, mp) in enumerate(ts):
        addr += [g.data_ptr(), p.data_ptr(), m.data_ptr(), v.data_ptr(), mp.data_ptr()]
        n = p.numel(); sizes.append(n); coff.append(total)
        c = (n + CHUNK - 1) // CHUNK; b2t += [i] * c; total += c
    T = lambda x, dt: torch.tensor(x, dtype=dt, device=DEV)
    return T(addr, torch.int64), T(sizes, torch.int64), T(b2t, torch.int32), T(coff, torch.int32), total


def tile_meta(ts, outs):
    addr, dims, t2t, toff, total = [], [], [], [], 0
    for i, ((g, p, m, v, mp), o) in enumerate(zip(ts, outs)):
        addr += [g.data_ptr(), p.data_ptr(), m.data_ptr(), v.data_ptr(), mp.data_ptr()] + [x.data_ptr() for x in o]
        n, k = p.shape[-2:]; b = p.numel() // (n * k); tn, tk = -(-n // 128), -(-k // 128)
        dims += [n, k, tn, tk]; toff.append(total); t2t += [i] * (b * tn * tk); total += b * tn * tk
    T = lambda x, dt: torch.tensor(x, dtype=dt, device=DEV)
    return T(addr, torch.int64), T(dims, torch.int64), T(t2t, torch.int32), T(toff, torch.int64), total


def alloc_out(p):
    lead, (n, k) = tuple(p.shape[:-2]), p.shape[-2:]
    tn, tk = -(-n // 128), -(-k // 128)
    f8 = torch.float8_e4m3fn
    return [torch.empty(lead + (n, k), dtype=f8, device=DEV), torch.empty(lead + (k, n), dtype=f8, device=DEV),
            torch.empty(lead + (tn, tk), dtype=torch.float32, device=DEV),
            torch.empty(lead + (tn, tk), dtype=torch.uint8, device=DEV),
            torch.empty(lead + (tk, tn), dtype=torch.uint8, device=DEV)]


def run_chunk(ts, meta, norm):
    a, s, b2t, coff, total = meta
    ext.fused_adam_clip(a, s, b2t, coff, total, CHUNK, LR, B1, B2, EPS, STEP, 1, 1, WD, 1, 1, norm, MAXN)


def run_tile(meta, norm):
    a, d, t2t, toff, total = meta
    ext.fused_adam_clip_fp8blk(a, d, t2t, toff, total, LR, B1, B2, EPS, STEP, 1, 1, WD, 1, 1, norm, MAXN)


def check(shapes):
    norm = torch.tensor([2.5], device=DEV)  # clip coeff 0.4
    A = make(shapes); B = clone(A)
    run_chunk(A, chunk_meta(A), norm)
    outs = [alloc_out(row[1]) for row in B]
    run_tile(tile_meta(B, outs), norm)
    torch.cuda.synchronize()
    ok = True
    for sh, ra, rb, o in zip(shapes, A, B, outs):
        res = {n: torch.equal(x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32),
                              y.view(torch.int16 if y.dtype == torch.bfloat16 else torch.int32))
               for n, x, y in [("master", ra[1], rb[1]), ("m", ra[2], rb[2]), ("v", ra[3], rb[3]), ("bf16", ra[4], rb[4])]}
        q, s = quant_fp8_blockwise_for_weight_impl(ra[4].contiguous(), torch.float8_e4m3fn, block_size=128)
        e = ((s.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
        res["W"] = torch.equal(o[0].view(torch.uint8), q.view(torch.uint8))
        res["WT"] = torch.equal(o[1].view(torch.uint8), q.view(torch.uint8).transpose(-1, -2).contiguous())
        res["scale"] = torch.equal(o[2].view(torch.int32), s.view(torch.int32))
        res["e8m0"] = torch.equal(o[3], e) and torch.equal(o[4], e.transpose(-1, -2).contiguous())
        good = all(res.values()); ok &= good
        print(("PASS" if good else "FAIL"), sh, "" if good else {k: v for k, v in res.items() if not v}, flush=True)
    return ok


def bench(shapes, iters):
    norm = torch.tensor([2.5], device=DEV)
    A = make(shapes); cm = chunk_meta(A)
    outs = [alloc_out(row[1]) for row in A]; tm = tile_meta(A, outs)
    for _ in range(3):
        run_chunk(A, cm, norm); run_tile(tm, norm)
    torch.cuda.synchronize()
    ev = lambda: torch.cuda.Event(enable_timing=True)
    for name, fn in [
        ("baseline: chunk adam", lambda: run_chunk(A, cm, norm)),
        ("baseline: + weight quant", lambda: [quant_fp8_blockwise_for_weight_impl(r[4], torch.float8_e4m3fn, 128) for r in A]),
        ("baseline: + W^T copy (g90 dgrad)", lambda: [o[0].transpose(-1, -2).contiguous() for o in outs]),
        ("fused: tile adam+quant+W^T", lambda: run_tile(tm, norm)),
    ]:
        s, e = ev(), ev(); s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        print(f"{name:36s} {s.elapsed_time(e) / iters * 1e3:9.1f} us/step (event timing; use the rocprofv3 trace)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    ok = check(DRY if a.dry else DRY + FULL)
    print("ALL PASS" if ok else "SOME FAILED", flush=True)
    if a.bench and not a.dry:
        bench(FULL, a.iters)
    sys.exit(0 if ok else 1)
