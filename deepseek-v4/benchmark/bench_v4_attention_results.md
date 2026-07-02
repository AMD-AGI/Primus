# DeepSeek-V4 Attention — Backend Performance

Full forward + backward benchmark of every V4 attention backend, produced by
[`bench_v4_attention.py`](./bench_v4_attention.py).

## Setup

- **GPU**: AMD Instinct MI355X (gfx950), single GPU
- **Config**: `seq_len=4096`, `mbs=1`, bf16, sink **on**, `swa_window=128`
- **Models**: V4-Flash (`H=64`, index_topk=512), V4-Pro (`H=128`, index_topk=1024)
- **Layer kinds**: `cr=0` dense/SWA, `cr=4` CSA (local SWA + sparse top-k pool),
  `cr=128` HCA (local SWA + full compressed pool)
- **Timing**: `--warmup 5 --iters 20`, median latency
- **Cell format**: `latency ms | TFLOP/s`. TFLOP/s is over the *useful* work
  (`2·T·H·TOPK·(D_V+D_V)`, head_dim=512; bwd = 2.5× fwd) with the **same formula
  for every backend**, so rows are directly comparable (the fused backends'
  zero-rope-pad overhead shows up in `ms`, not in counted FLOPs).

## Backends

| Backend | Description |
|---------|-------------|
| `triton` | Production separate-K/V Triton (dense/SWA/HCA launchers; cr=4 uses the split CSA pool fwd + segreduce bwd) |
| `flydsl` | Legacy FlyDSL gathered CSA (**cr=4 only**, scalarized GEMV, no MFMA) — historical reference |
| `gluon` | Fused single-latent (K==V) sparse-MLA, hand-tuned gfx950 gluon (async double-buffered pipeline) |
| `triton_v2` | Fused single-latent sparse-MLA in plain Triton (`tl.dot` / MFMA) — **optimized in this PR** |
| `flydsl_v1` | Fused single-latent sparse-MLA in native FlyDSL MFMA (**forward kernel WIP → FAIL**) |

> FlyDSL requires the FlyDSL source at `/workspace/FlyDSL-amd` (clone
> `https://github.com/ROCm/FlyDSL` tag `v0.2.2`) and the `flydsl` pip package.

## Forward (`latency ms | TFLOP/s`)

| variant | cr (layer) | triton | flydsl | gluon | triton_v2 | flydsl_v1 |
|---------|-----------|--------|--------|-------|-----------|-----------|
| flash | 0 (SWA)   | 0.47 \| 147.6 | — | 0.35 \| 197.4 | **0.31 \| 221.0** | FAIL |
| flash | 4 (CSA)   | 1.40 \| 244.9 | 15.51 \| 22.2 | 1.01 \| 341.8 | **0.93 \| 369.4** | FAIL |
| flash | 128 (HCA) | 0.68 \| 126.4 | — | 0.42 \| 205.9 | **0.39 \| 219.8** | FAIL |
| pro | 0 (SWA)     | 0.84 \| 163.0 | — | 0.63 \| 219.5 | **0.58 \| 237.6** | FAIL |
| pro | 4 (CSA)     | 4.01 \| 308.2 | 54.20 \| 22.8 | 3.05 \| 405.1 | **2.98 \| 414.5** | FAIL |
| pro | 128 (HCA)   | 1.28 \| 133.8 | — | 0.77 \| 221.8 | **0.73 \| 236.8** | FAIL |

## Backward (`latency ms | TFLOP/s`)

| variant | cr (layer) | triton | flydsl | gluon | triton_v2 | flydsl_v1 |
|---------|-----------|--------|--------|-------|-----------|-----------|
| flash | 0 (SWA)   | 2.04 \| 84.4 | — | 1.21 \| 142.1 | **1.16 \| 148.6** | FAIL |
| flash | 4 (CSA)   | 5.18 \| 165.8 | 2909.43 \| 0.3 | 5.28 \| 162.7 | **5.13 \| 167.6** | FAIL |
| flash | 128 (HCA) | 3.04 \| 70.6 | — | 1.78 \| 120.9 | **1.72 \| 124.9** | FAIL |
| pro | 0 (SWA)     | 3.94 \| 87.3 | — | 1.89 \| 182.1 | **1.79 \| 192.4** | FAIL |
| pro | 4 (CSA)     | 15.02 \| 205.8 | 9413.50 \| 0.3 | 14.55 \| 212.6 | **10.87 \| 284.5** | FAIL |
| pro | 128 (HCA)   | 5.92 \| 72.5 | — | 2.71 \| 158.4 | **2.48 \| 173.4** | FAIL |

## Summary

- **`triton_v2` is the fastest backend on all 6 shapes for both forward and
  backward.** vs the hand-tuned `gluon`:
  - Forward: **~1.02–1.13×** (geomean ~1.08×)
  - Backward: **~1.02–1.34×** (geomean ~1.10×); `pro cr=4` bwd 10.87 ms vs 14.55 ms.
- The production separate-K/V `triton` backend is significantly slower than the
  fused backends, especially on backward dense/HCA.
- `flydsl` (legacy) is cr=4-only and orders of magnitude slower (scalarized,
  no MFMA); kept only as a historical reference.
- `flydsl_v1` forward is an upstream WIP (`NotImplementedError`); fwd+bwd FAIL.

## Reproduce

```bash
# one-time: FlyDSL source for the legacy cr=4 backend
git clone --depth 1 --branch v0.2.2 https://github.com/ROCm/FlyDSL /workspace/FlyDSL-amd

PYTHONPATH=<repo> python deepseek-v4/benchmark/bench_v4_attention.py \
    --variant both --cr all --warmup 5 --iters 20
```
