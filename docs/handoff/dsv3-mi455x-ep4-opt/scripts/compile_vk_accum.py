# Compile-only (no GPU) build of the gfx1250 FlyDSL variable-K grouped GEMM for the DSv3 expert
# wgrad shapes, with accumulate=0 and accumulate=1 (the beta=1 epilogue).
# Run with COMPILE_ONLY=1 FLYDSL_GPU_ARCH=gfx1250 ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
# FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<dir>/<case> FLYDSL_RUNTIME_CACHE_DIR=<fresh dir>; argv[1] = case.
import os
import sys

import torch

assert os.environ.get("COMPILE_ONLY") == "1"
import primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 as k  # noqa: E402

print("tree:", os.path.dirname(k.__file__), flush=True)
# DSv3 proxy EP4: 32 local experts, 65536 received rows; fc1 wgrad [32, 4096, 7168], fc2 [32, 7168, 2048]
SHAPES = {"fc1": (4096, 7168), "fc2": (7168, 2048)}
G, M_TOTAL = 32, 65536
name, acc = sys.argv[1].rsplit("_", 1)
acc = int(acc)
OUT_M, OUT_N = SHAPES[name]
avg_m = M_TOTAL // G
tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers = k._pick_variable_k_config(OUT_M, OUT_N, avg_m)
cfg = (tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)
split_bar = k._vk_auto_split_bar(cfg, OUT_M, OUT_N, avg_m)
print("cfg", cfg, "split_bar", split_bar, "accumulate", acc, flush=True)


def t(*shape, dtype=torch.bfloat16):
    return torch.zeros(*shape, dtype=dtype)


out, a, b, etab = t(G, OUT_M, OUT_N), t(M_TOTAL, OUT_M), t(M_TOTAL, OUT_N), t(2 * G, dtype=torch.int32)
args = (
    k._as_ptr(out), k._as_ptr(a), k._as_ptr(b), k._as_ptr(etab), None,
    OUT_M, OUT_N, G, (OUT_M + tile_m - 1) // tile_m, (OUT_N + tile_n - 1) // tile_n, OUT_M * OUT_N,
    tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, 0, 16, 1, 32, 0, 1, split_bar, 0, 0, 0, 0,
    k._vk_lds_pad(tile_m, tile_n, 2), k._vk_wave_grid(m_warp, n_warp), 0, 0, acc,
)
k._vk_launch(args)
print("compiled", sys.argv[1], flush=True)
