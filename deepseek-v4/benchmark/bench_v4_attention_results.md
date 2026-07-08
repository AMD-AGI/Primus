# DeepSeek-V4 Attention — Backend Performance

Full forward + backward benchmark of every V4 attention backend, produced by
[`bench_v4_attention.py`](./bench_v4_attention.py).

## Setup

- **GPU**: AMD Instinct MI355X (gfx950), single GPU (`smci355-ccs-aus-n03-33`)
- **Container**: `dev_primus_wenx`
- **Torch**: `2.10.0a0+git449b176`
- **Config**: `seq_len=4096`, `mbs=1`, bf16, sink **on**, `swa_window=128`
- **Models**: V4-Flash (`H=64`, index_topk=512), V4-Pro (`H=128`, index_topk=1024)
- **Layer kinds**: `cr=0` dense/SWA, `cr=4` CSA, `cr=128` HCA
- **Timing**: `--warmup 10 --iters 30`, median latency
- **Cell format**: `latency ms | TFLOP/s`

Raw log:

- `agent/workspace/gluon_v3_sparse_mla_gfx950_20260708/bench_all_backends_rerun_20260708.log`

## Backends

| Backend | Description |
|---------|-------------|
| `triton` | Production separate-K/V Triton dense/SWA/HCA + split CSA pool kernels |
| `gluon` | 1st-gen fused single-latent Gluon sparse-MLA (`_gluon_dsa`) |
| `triton_v2` | Fused single-latent sparse-MLA in plain Triton |
| `gluon_v2` | 2nd-gen Gluon sparse-MLA baseline |
| `gluon_v3` | Optimized Gluon sparse-MLA: Round 9 CSA formula-pack + aiter Gluon LSE fwd route, plus accepted bwd chunking |
| `flydsl_v1` | Native FlyDSL MFMA sparse-MLA backend |
| `turbo_flydsl` | Extracted Primus-Turbo `sparse_mla_v2` backend |
| `aiter_gluon` | aiter Gluon sparse-MLA prefill reference, fwd-only |

`aiter_gluon` has no backward implementation, so bwd is shown as `—`.

## Forward

`latency ms | TFLOP/s`; **bold** is fastest latency in the row.

| variant | cr | triton | gluon | triton_v2 | gluon_v2 | gluon_v3 | flydsl_v1 | turbo_flydsl | aiter_gluon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flash | 0 | 0.47 \| 146.6 | 0.33 \| 206.9 | 0.33 \| 210.3 | **0.30 \| 230.0** | **0.30 \| 230.3** | 0.46 \| 150.5 | 0.31 \| 222.9 | 0.49 \| 139.2 |
| flash | 4 | 1.49 \| 231.0 | 0.94 \| 366.5 | 0.88 \| 390.7 | 0.75 \| 456.3 | **0.71 \| 487.0** | 1.37 \| 251.2 | 0.78 \| 439.2 | 0.88 \| 389.5 |
| flash | 128 | 0.77 \| 112.2 | 0.41 \| 209.5 | 0.41 \| 211.6 | **0.35 \| 248.0** | **0.35 \| 246.9** | 0.58 \| 147.2 | 0.36 \| 237.7 | 0.53 \| 161.6 |
| pro | 0 | 0.86 \| 160.5 | 0.58 \| 235.5 | 0.60 \| 228.9 | **0.54 \| 253.4** | **0.54 \| 255.4** | 1.06 \| 130.3 | 0.55 \| 250.1 | 0.84 \| 163.9 |
| pro | 4 | 4.48 \| 276.1 | 2.90 \| 425.8 | 2.79 \| 444.0 | 2.37 \| 522.7 | **2.01 \| 615.1** | 4.80 \| 257.9 | 2.09 \| 592.1 | 2.32 \| 532.3 |
| pro | 128 | 1.48 \| 116.4 | 0.74 \| 233.5 | 0.72 \| 239.5 | **0.62 \| 276.4** | **0.62 \| 278.5** | 1.20 \| 143.2 | 0.64 \| 266.9 | 0.91 \| 188.9 |

## Backward

`latency ms | TFLOP/s`; **bold** is fastest latency in the row.

| variant | cr | triton | gluon | triton_v2 | gluon_v2 | gluon_v3 | flydsl_v1 | turbo_flydsl | aiter_gluon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flash | 0 | 2.14 \| 80.2 | 1.22 \| 140.5 | 1.18 \| 146.1 | 1.16 \| 148.2 | **1.15 \| 148.8** | 2.22 \| 77.3 | 1.40 \| 122.7 | — |
| flash | 4 | 5.30 \| 162.0 | 5.26 \| 163.2 | 6.01 \| 142.8 | 4.89 \| 175.8 | **4.04 \| 212.7** | 6.28 \| 136.9 | 4.05 \| 212.3 | — |
| flash | 128 | 2.92 \| 73.5 | 1.66 \| 129.6 | 1.67 \| 128.9 | **1.57 \| 137.1** | **1.57 \| 137.0** | 2.81 \| 76.5 | 1.85 \| 116.1 | — |
| pro | 0 | 4.10 \| 83.8 | 1.87 \| 183.3 | 1.86 \| 184.3 | **1.75 \| 196.8** | 1.77 \| 194.5 | 5.76 \| 59.7 | 2.16 \| 158.9 | — |
| pro | 4 | 15.17 \| 203.9 | 13.46 \| 229.8 | 10.75 \| 287.7 | **8.54 \| 362.3** | 8.55 \| 361.8 | 30.53 \| 101.3 | 9.41 \| 328.6 | — |
| pro | 128 | 5.61 \| 76.6 | 2.43 \| 177.1 | 2.45 \| 175.3 | **2.27 \| 189.3** | **2.27 \| 189.3** | 6.96 \| 61.7 | 2.92 \| 147.3 | — |

## gluon_v3 vs turbo_flydsl

`gluon_v3` is faster than `turbo_flydsl` on every measured fwd/bwd cell in this run.

| variant | cr | FWD speedup | BWD speedup |
|---|---:|---:|---:|
| flash | 0 | 1.03× | 1.22× |
| flash | 4 | 1.10× | 1.00× |
| flash | 128 | 1.03× | 1.18× |
| pro | 0 | 1.02× | 1.22× |
| pro | 4 | 1.04× | 1.10× |
| pro | 128 | 1.03× | 1.29× |

## Summary

- `gluon_v3` is the strongest full fwd+bwd backend in this run: it beats
  `turbo_flydsl` on all six layer shapes in both directions.
- The largest `gluon_v3` forward wins are on the CSA paths:
  `flash cr=4` (`0.71 ms` vs `turbo_flydsl 0.78 ms`) and `pro cr=4`
  (`2.01 ms` vs `turbo_flydsl 2.09 ms`).
- `gluon_v3` backward keeps the accepted `gluon_v2`/Round-2 chunking behavior and
  remains faster than `turbo_flydsl` across all shapes.
- `gluon_v2` remains very competitive and is marginally fastest on a few rounded
  non-CSA cells, but it does not include the Round-9 CSA fwd optimizations.

## Reproduce

```bash
PYTHONPATH=<repo> python deepseek-v4/benchmark/bench_v4_attention.py \
    --variant both --cr all --warmup 10 --iters 30
```
