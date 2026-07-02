# DeepSeek-V4 Attention — Backend Performance

Full forward + backward benchmark of every V4 attention backend, produced by
[`bench_v4_attention.py`](./bench_v4_attention.py).

## Setup

- **GPU**: AMD Instinct MI355X (gfx950), single GPU (`smci355-ccs-aus-n04-25`)
- **Config**: `seq_len=4096`, `mbs=1`, bf16, sink **on**, `swa_window=128`
- **Models**: V4-Flash (`H=64`, index_topk=512), V4-Pro (`H=128`, index_topk=1024)
- **Layer kinds**: `cr=0` dense/SWA, `cr=4` CSA (local SWA + sparse top-k pool),
  `cr=128` HCA (local SWA + full compressed pool)
- **Timing**: `--warmup 5 --iters 20`, median latency
- **Cell format**: `latency ms | TFLOP/s`. TFLOP/s is over the *useful* work
  (`2·T·H·TOPK·(D_V+D_V)`, head_dim=512; bwd = 2.5× fwd) with the **same formula
  for every backend**, so rows are directly comparable.

## Backends

| Backend | Description |
|---------|-------------|
| `triton` | Production separate-K/V Triton (dense/SWA/HCA launchers; cr=4 split CSA pool fwd + segreduce bwd) |
| `gluon` | Fused single-latent (K==V) sparse-MLA, hand-tuned gfx950 gluon (async double-buffered pipeline) |
| `triton_v2` | Fused single-latent sparse-MLA in plain Triton (`tl.dot` / MFMA) — fastest backend |
| `flydsl_v1` | Fused single-latent sparse-MLA in **native FlyDSL MFMA** (fwd + bwd). fwd: per-token top-k gather + MFMA QK/PV + online softmax + sink; bwd: native FlyDSL dQ (MFMA) + shared Triton dKV intermediate/scatter-gather |

> `flydsl_v1` depends only on the installed `flydsl` pip package — no
> `/workspace/FlyDSL-amd` source tree required. The legacy `_flydsl_v0_deprecated`
> gathered-CSA backend (scalarized GEMV, correctness issues, `/workspace`
> dependency) is no longer benchmarked.

## Forward (`latency ms | TFLOP/s`)

| variant | cr (layer) | triton | gluon | triton_v2 | flydsl_v1 |
|---------|-----------|--------|-------|-----------|-----------|
| flash | 0 (SWA)   | 0.49 \| 139.0 | 0.33 \| 208.5 | **0.31 \| 224.5** | 0.45 \| 154.2 |
| flash | 4 (CSA)   | 1.45 \| 237.6 | 0.94 \| 365.0 | **0.91 \| 377.4** | 1.35 \| 254.9 |
| flash | 128 (HCA) | 0.68 \| 125.5 | 0.41 \| 209.9 | **0.38 \| 228.0** | 0.56 \| 153.8 |
| pro | 0 (SWA)     | 0.91 \| 151.7 | 0.62 \| 222.6 | **0.59 \| 234.9** | 0.97 \| 141.9 |
| pro | 4 (CSA)     | 4.02 \| 307.4 | 3.04 \| 407.5 | **2.96 \| 417.2** | 4.01 \| 308.6 |
| pro | 128 (HCA)   | 1.31 \| 131.4 | 0.78 \| 221.3 | **0.74 \| 232.3** | 1.12 \| 154.0 |

## Backward (`latency ms | TFLOP/s`)

| variant | cr (layer) | triton | gluon | triton_v2 | flydsl_v1 |
|---------|-----------|--------|-------|-----------|-----------|
| flash | 0 (SWA)   | 2.00 \| 86.0  | 1.20 \| 143.2 | **1.14 \| 150.1** | 2.12 \| 81.0 |
| flash | 4 (CSA)   | 5.15 \| 166.6 | 5.26 \| 163.3 | **5.04 \| 170.4** | 7.59 \| 113.1 |
| flash | 128 (HCA) | 2.90 \| 74.0  | 1.76 \| 122.1 | **1.73 \| 124.4** | 2.75 \| 78.1 |
| pro | 0 (SWA)     | 3.88 \| 88.6  | 1.90 \| 180.4 | **1.81 \| 190.2** | 5.42 \| 63.4 |
| pro | 4 (CSA)     | 15.01 \| 206.0| 15.01 \| 206.0| **10.84 \| 285.2**| 29.48 \| 104.9 |
| pro | 128 (HCA)   | 5.65 \| 76.0  | 2.69 \| 159.4 | **2.50 \| 171.7** | 6.68 \| 64.3 |

## flydsl_v1 vs triton_v2 (speedup = triton_v2_ms / flydsl_v1_ms)

| variant | cr | FWD speedup | BWD speedup |
|---------|----|-------------|-------------|
| flash | 0   | 0.69× | 0.54× |
| flash | 4   | 0.67× | 0.66× |
| flash | 128 | 0.68× | 0.63× |
| pro | 0     | 0.61× | 0.33× |
| pro | 4     | 0.74× | 0.37× |
| pro | 128   | 0.66× | 0.37× |
| **geomean** | | **0.674×** | **0.475×** |

## Summary

- `flydsl_v1` is a **correct** native-FlyDSL MFMA backend for both forward and
  backward (fwd SNR ~47.7 dB, bwd dq ~46.7 dB / dkv ~73 dB vs triton_v2).
- It does **not** beat `triton_v2` (fwd ~0.67× geomean, bwd ~0.48× geomean). Both
  kernels are register-bound (the 512-wide fp32 accumulator, plus Q/dO operands in
  bwd, pin occupancy at 1); `triton_v2`'s edge is Triton's mature IR-level auto
  software-pipeliner (`num_stages`), which hand-written FlyDSL cannot match here
  (register-prefetch spills; `buffer_load_lds` is slow for a scattered top-k
  gather). The bwd is worse for `pro`/large-topk because it runs the whole top-k as
  one chunk (no dQ read-modify-write) with a large dKV intermediate.
- `triton_v2` remains the fastest backend on all 6 shapes for both directions.

## Reproduce

```bash
PYTHONPATH=<repo> python deepseek-v4/benchmark/bench_v4_attention.py \
    --variant both --cr all --warmup 5 --iters 20
```
