"""Compare fused QKV RMSNorm+RoPE: baseline Turbo (D=64 only) vs #556 (D=64+128).

Shapes (S,B,NG,NPG,D):
  gpt_oss_20B : 4096,8,8,8,64   — the existing fused path (PRIMUS_FUSED_QK_RMSNORM_ROPE)
  qwen3_30B   : 4096,8,4,8,128  — newly enabled by #556
  minimax_m3  : 4096,8,4,16,128 — D=128, NPG=16 (64 heads / 4 groups)
  lfm2_8B     : 4096,8,8,4,128  — D=128, NPG=4 (32 heads / 8 groups)
  glm5        : 4096,4,1,64,128 — D=128; no GQA in yaml (treat as NG=1,NPG=64); B=4 to fit

Usage (in container):
  PYTHONPATH=<tree> FLYDSL_RUNTIME_CACHE_DIR=/tmp/fly_<tag> \
    python bench_qk_rmsnorm_rope_ab.py [tag] [shape_name ...]
"""

from __future__ import annotations

import os
import sys

import torch

from primus_turbo.pytorch.ops.rope import fused_qkv_rmsnorm_rope

SHAPES = {
    "gpt_oss_20B": (4096, 8, 8, 8, 64),
    "qwen3_30B": (4096, 8, 4, 8, 128),
    "minimax_m3": (4096, 8, 4, 16, 128),
    "lfm2_8B": (4096, 8, 8, 4, 128),
    # B*NG must be a multiple of rows_per_wave (8 at D=128); B=8 keeps that.
    "glm5": (4096, 8, 1, 64, 128),
}

EPS = 1e-6
tag = sys.argv[1] if len(sys.argv) > 1 else "run"
names = sys.argv[2:] or list(SHAPES)


def bench(fn, iters=40, warmup=8):
    for _ in range(warmup):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


def run_shape(name: str):
    S, B, NG, NPG, D = SHAPES[name]
    try:
        from primus_turbo.flydsl.rope.qk_rmsnorm_rope_kernel import QK_RMSNORM_ROPE_HEAD_DIMS

        supported = QK_RMSNORM_ROPE_HEAD_DIMS
    except ImportError:
        from primus_turbo.flydsl.rope.qk_rmsnorm_rope_kernel import QK_RMSNORM_ROPE_HEAD_DIM

        supported = (QK_RMSNORM_ROPE_HEAD_DIM,)
    if D not in supported:
        print(f"{tag:8s} {name:12s} D={D} unsupported (HEAD_DIMS={supported})")
        return

    torch.manual_seed(0)
    split = [NPG * D, D, D]
    qkv = torch.randn(S, B, NG, sum(split), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    half = torch.randn(S, D // 2, device="cuda")
    freqs = torch.cat((half, half), -1).reshape(S, 1, 1, D).contiguous()
    q_gamma = torch.randn(D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k_gamma = torch.randn(D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def fused():
        return fused_qkv_rmsnorm_rope(qkv, q_gamma, k_gamma, freqs, split, EPS)

    with torch.no_grad():
        outs = fused()
    grads = tuple(torch.randn_like(t) for t in outs)

    def fwd_bwd():
        o = fused()
        torch.autograd.backward(o, grads)
        qkv.grad = None
        q_gamma.grad = None
        k_gamma.grad = None

    with torch.no_grad():
        t_f = bench(fused)
    t_fb = bench(fwd_bwd)
    print(
        f"{tag:8s} {name:12s} S={S} B={B} NG={NG} NPG={NPG} D={D}  "
        f"fwd {t_f:7.1f} us   fwd+bwd {t_fb:7.1f} us   bwd {t_fb - t_f:7.1f} us"
    )


print(f"# tag={tag} pythonpath={os.environ.get('PYTHONPATH','').split(':')[0]}")
for n in names:
    run_shape(n)
