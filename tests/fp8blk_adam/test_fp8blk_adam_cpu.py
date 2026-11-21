"""CPU check of the FP8-block Adam tile path (no GPU).

The kernel's per-thread phases are compiled for the host (fp8blk_host_test.cpp) and run on random
weights. Checks, all bitwise:
  * Adam: fp32 master, moments and BF16 param equal a numpy float32 reference of the
    fused_adam_clip formula (the host build has no FMA; device FMA identity is checked on the GPU
    by test_fp8blk_adam_gpu.py against the unchanged fused_adam_clip kernel).
  * Quant: W and scale_inv equal Turbo's quant_fp8_blockwise_for_weight_kernel (POW2 / UE8M0) run
    under the Triton interpreter on the updated BF16 weight, and a torch emulation of it;
    W^T == W.transpose(-1, -2); e8m0 == exponent byte of scale_inv; e8m0_t == e8m0^T.
"""
import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
import importlib.util

import numpy as np
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMUS = os.environ.get("PRIMUS_ROOT", os.path.abspath(os.path.join(HERE, "..", "..")))
TURBO = os.environ.get("TURBO_ROOT", "/dsv41/work/opt3adamq/Primus-Turbo")
KSRC = f"{PRIMUS}/primus/backends/megatron/patches/te_patches/fused_adam_clip_kernel.cu"
ext = load(
    name="fp8blk_host_test",
    sources=[f"{HERE}/fp8blk_host_test.cpp"],
    extra_cflags=["-O2"],
    verbose=False,
)

spec = importlib.util.spec_from_file_location(
    "qb", f"{TURBO}/primus_turbo/triton/quantization/quant_blockwise.py"
)
qb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qb)

F8 = torch.float8_e4m3fn


def cdiv(a, b):
    return (a + b - 1) // b


def ref_quant_triton(wb):
    """Turbo's weight quant kernel (POW2) on bf16 [B,N,K] -> (fp8 [B,N,K], scale_inv [B,tn,tk])."""
    B, N, K = wb.shape
    q = torch.empty((B, N, K), dtype=F8)
    s = torch.empty((B, cdiv(N, 128), cdiv(K, 128)), dtype=torch.float32)
    qb.quant_fp8_blockwise_for_weight_kernel[(B, cdiv(N, 128), cdiv(K, 128))](
        wb, q, s, N, K, 128, 448.0, POW2_SCALE=True
    )
    return q, s


def ref_quant_torch(wb):
    B, N, K = wb.shape
    tn, tk = cdiv(N, 128), cdiv(K, 128)
    x = torch.zeros((B, tn * 128, tk * 128), dtype=torch.float32)
    x[:, :N, :K] = wb.float()
    t = x.view(B, tn, 128, tk, 128)
    amax = t.abs().amax(dim=(2, 4)).clamp_min(1e-4)
    sc = torch.tensor(448.0, dtype=torch.float32) / amax
    sc = (sc.view(torch.int32) & 0x7F800000).view(torch.float32)
    q = (t * sc[:, :, None, :, None]).clamp(-448.0, 448.0).to(F8).view(B, tn * 128, tk * 128)
    return q[:, :N, :K].contiguous(), (1.0 / sc)


def np_adam(g, p, m, v, lr, b1, b2, eps, step_size, b2ci, decay, gs, adamw):
    f = np.float32
    g = g * f(gs)
    if not adamw:
        g = g + f(decay) * p
    m = f(b1) * m + (f(1.0) - f(b1)) * g
    v = f(b2) * v + ((f(1.0) - f(b2)) * g) * g
    denom = np.sqrt(v * f(b2ci)) + f(eps)
    if adamw:
        p = (p - f(step_size) * (m / denom)) - (f(lr) * f(decay)) * p
    else:
        p = p - f(step_size) * (m / denom)
    return p, m, v


def bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


def run_case(shape, gdt, mdt, mode=1, lr=3e-4, step=7, decay=0.1, gs=0.73, seed=0, weights=None):
    torch.manual_seed(seed)
    B, N, K = (1, *shape) if len(shape) == 2 else shape
    if weights is not None:
        p = weights.clone().float().reshape(B, N, K)
    else:
        p = torch.randn(B, N, K) * 0.02
    g = (torch.randn(B, N, K) * 1e-3).to(gdt)
    m = (torch.randn(B, N, K) * 1e-4).to(mdt)
    v = (torch.rand(B, N, K) * 1e-6).to(mdt)
    p0, g0, m0, v0 = p.clone(), g.clone(), m.clone(), v.clone()
    model = torch.empty(B, N, K, dtype=torch.bfloat16)
    tn, tk = cdiv(N, 128), cdiv(K, 128)
    w = torch.empty(B, N, K, dtype=torch.uint8)
    wt = torch.empty(B, K, N, dtype=torch.uint8)
    si = torch.empty(B, tn, tk)
    e = torch.empty(B, tn, tk, dtype=torch.uint8)
    et = torch.empty(B, tk, tn, dtype=torch.uint8)
    b1, b2, eps = 0.9, 0.95, 1e-8
    step_size, b2ci = ext.run(g, p, m, v, model, w, wt, si, e, et, B, N, K, lr, b1, b2, eps, step, mode, decay, gs)

    rp, rm, rv = np_adam(
        g0.float().numpy(), p0.numpy(), m0.float().numpy(), v0.float().numpy(),
        lr, b1, b2, eps, step_size, b2ci, decay, gs, mode == 1,
    )
    ok = {}
    ok["master"] = np.array_equal(p.numpy().view(np.int32), rp.view(np.int32))
    ok["m"] = torch.equal(bits(m), bits(torch.from_numpy(rm).to(mdt)))
    ok["v"] = torch.equal(bits(v), bits(torch.from_numpy(rv).to(mdt)))
    ok["bf16"] = torch.equal(bits(model), bits(torch.from_numpy(rp).bfloat16()))

    qt, st = ref_quant_triton(model)
    qr, sr = ref_quant_torch(model)
    # FP8 bytes: torch RNE emulation of the Turbo kernel (the device converts with
    # cvt.scalef32.pk8.fp8.f32, RNE). The Triton 3.8 interpreter's own fp32->e4m3 cast drops the
    # mantissa carry (31.6 -> 16, see scripts/diag_fp8_interp.py), so its bytes are only reported.
    ok["W==torch"] = torch.equal(w, qr.view(torch.uint8))
    ok["scale==triton"] = torch.equal(si.view(torch.int32), st.view(torch.int32))
    ok["scale==torch"] = torch.equal(si.view(torch.int32), sr.view(torch.int32))
    ok["WT"] = torch.equal(wt, qr.view(torch.uint8).transpose(1, 2).contiguous())
    n_interp_diff = int((w != qt.view(torch.uint8)).sum())
    e_ref = ((st.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    ok["e8m0"] = torch.equal(e, e_ref)
    ok["e8m0_t"] = torch.equal(et, e_ref.transpose(1, 2).contiguous())
    nsub = int(((qt.view(torch.uint8) & 0x78) == 0).sum())
    good = all(ok.values())
    bad = "" if good else str({k: v for k, v in ok.items() if not v})
    print(
        f"{'PASS' if good else 'FAIL'} shape={tuple(shape)} grad={gdt} mom={mdt} mode={mode} lr={lr} "
        f"tiles={B * tn * tk} e8m0={sorted(set(e.flatten().tolist()))[:6]} fp8_subnormal_or_zero={nsub} "
        f"(interp-cast diffs {n_interp_diff}) {bad}",
        flush=True,
    )
    return good


def edge_weights(N, K):
    """With lr=0 the weight is kept: tiles whose 448/amax is exactly a power of two
    (amax = 7*2^j), an all-zero tile, a tile with amax < 1e-4, and fp8 subnormals."""
    w = torch.randn(N, K) * 0.02
    w[:128, :128] = w[:128, :128].clamp(-0.4375, 0.4375)
    w[5, 7] = 0.4375  # 448 / 0.4375 = 1024
    w[:128, 128:256] = 0.0
    w[128:256, :128] = torch.randn(128, 128) * 1e-6
    w[128:256, 128:256] = torch.randn(128, 128).clamp(-3, 3) * 3.0
    w[130, 140] = 14.0  # 448 / 14 = 32
    w[128:256, 129] = 1e-5  # fp8 subnormals at scale 32
    return w.bfloat16().float()


if __name__ == "__main__":
    allok = True
    for shape in [(256, 384), (300, 200), (3, 256, 128), (130, 264), (576, 512), (2, 384, 136)]:
        for gdt, mdt in [
            (torch.bfloat16, torch.bfloat16),
            (torch.float32, torch.float32),
            (torch.bfloat16, torch.float32),
            (torch.float32, torch.bfloat16),
        ]:
            allok &= run_case(shape, gdt, mdt)
    allok &= run_case((256, 384), torch.bfloat16, torch.bfloat16, mode=0, seed=3)
    for seed in range(3):
        allok &= run_case((512, 256), torch.bfloat16, torch.bfloat16, seed=10 + seed, step=1 + seed)
    for N, K in [(256, 256), (264, 392)]:
        allok &= run_case((N, K), torch.bfloat16, torch.bfloat16, lr=0.0, decay=0.0, weights=edge_weights(N, K))
    print("ALL PASS" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)
