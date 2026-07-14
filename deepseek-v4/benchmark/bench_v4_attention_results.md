# DeepSeek-V4 Attention — Backend Performance

Full forward + backward benchmark of every V4 attention backend, produced by
[`bench_v4_attention.py`](./bench_v4_attention.py).

## Setup

- **GPU**: AMD Instinct MI355X (gfx950), single GPU (`smci355-ccs-aus-n04-25`)
- **Container**: `dev_primus_wenx`
- **Torch**: `2.10.0+git94c6e04`, Triton `3.7.0`, FlyDSL `0.2.2`
- **Primus-Turbo**: `dev/kyle/flydsl_attn_deepseekv4` @ `350ec3f` (native-FlyDSL sparse-MLA v2)
- **Config**: `seq_len=4096`, `mbs=1`, bf16, sink **on**, `swa_window=128`
- **Models**: V4-Flash (`H=64`, index_topk=512), V4-Pro (`H=128`, index_topk=1024)
- **Layer kinds**: `cr=0` dense/SWA, `cr=4` CSA, `cr=128` HCA
- **Timing**: `--warmup 10 --iters 30`, median latency
- **Cell format**: `latency ms | TFLOP/s`

Raw log: `agent/workspace/_bench_all_final.log`

## Backends

| Backend | Description |
|---------|-------------|
| `triton` | Production separate-K/V Triton dense/SWA/HCA + split CSA pool kernels |
| `gluon` | 1st-gen fused single-latent Gluon sparse-MLA (`_gluon_dsa`) |
| `triton_v2` | Fused single-latent sparse-MLA in plain Triton |
| `gluon_v2` | 2nd-gen Gluon sparse-MLA baseline |
| `gluon_v3` | Optimized Gluon sparse-MLA: Round-9 CSA formula-pack + aiter Gluon LSE fwd route (benchmark-only; not wired for training) |
| `flydsl_v1` | In-tree native FlyDSL MFMA sparse-MLA backend |
| `turbo_flydsl` | Extracted Primus-Turbo `sparse_mla_v2` (June `optimize/...` branch) |
| `_turbo_flydsl` | **Integrated** Primus-Turbo native-FlyDSL sparse-MLA via the turbo API (`primus_turbo.flydsl.attention`); the `turbo` model backend |
| `aiter_gluon` | aiter Gluon sparse-MLA prefill reference, fwd-only |

`aiter_gluon` has no backward implementation, so bwd is shown as `—`.

## Forward

`latency ms | TFLOP/s`; **bold** is fastest latency in the row.

| variant | cr | triton | gluon | triton_v2 | gluon_v2 | gluon_v3 | flydsl_v1 | turbo_flydsl | _turbo_flydsl | aiter_gluon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flash | 0 | 0.46 \| 151.0 | 0.31 \| 222.1 | 0.30 \| 230.0 | 0.28 \| 247.8 | 0.28 \| 248.3 | 0.45 \| 153.3 | 0.30 \| 228.7 | **0.20 \| 335.5** | 0.47 \| 145.2 |
| flash | 4 | 1.47 \| 234.3 | 0.89 \| 386.1 | 0.87 \| 397.1 | 0.73 \| 469.0 | 0.66 \| 523.6 | 1.37 \| 250.9 | 0.72 \| 476.6 | **0.53 \| 651.7** | 0.83 \| 413.4 |
| flash | 128 | 0.75 \| 114.7 | 0.38 \| 226.3 | 0.38 \| 223.9 | 0.33 \| 263.9 | 0.33 \| 263.2 | 0.58 \| 147.3 | 0.35 \| 245.5 | **0.22 \| 384.2** | 0.52 \| 164.7 |
| pro | 0 | 0.86 \| 159.5 | 0.57 \| 239.8 | 0.58 \| 236.2 | 0.51 \| 269.1 | 0.51 \| 269.0 | 1.05 \| 131.0 | 0.55 \| 251.9 | **0.38 \| 357.9** | 0.79 \| 173.1 |
| pro | 4 | 4.44 \| 278.3 | 2.91 \| 425.5 | 2.78 \| 444.3 | 2.36 \| 525.0 | 1.92 \| 645.1 | 4.82 \| 256.6 | 2.09 \| 591.0 | **1.41 \| 878.1** | 2.16 \| 573.4 |
| pro | 128 | 1.46 \| 117.7 | 0.72 \| 239.4 | 0.72 \| 238.6 | 0.61 \| 281.2 | 0.61 \| 280.9 | 1.20 \| 143.5 | 0.63 \| 270.7 | **0.43 \| 395.5** | 0.88 \| 195.2 |

## Backward

`latency ms | TFLOP/s`; **bold** is fastest latency in the row.

| variant | cr | triton | gluon | triton_v2 | gluon_v2 | gluon_v3 | flydsl_v1 | turbo_flydsl | _turbo_flydsl | aiter_gluon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flash | 0 | 2.09 \| 82.2 | 1.20 \| 143.4 | 1.16 \| 148.4 | 1.13 \| 152.6 | 1.13 \| 152.0 | 2.20 \| 78.3 | 1.38 \| 124.6 | **0.67 \| 257.8** | — |
| flash | 4 | 5.18 \| 165.8 | 5.18 \| 166.0 | 5.93 \| 144.9 | 4.81 \| 178.5 | 3.99 \| 215.1 | 6.16 \| 139.5 | 3.94 \| 218.1 | **2.55 \| 336.9** | — |
| flash | 128 | 2.86 \| 75.0 | 1.63 \| 131.4 | 1.67 \| 128.8 | 1.54 \| 139.3 | 1.54 \| 139.2 | 2.77 \| 77.5 | 1.84 \| 117.0 | **0.78 \| 274.9** | — |
| pro | 0 | 4.02 \| 85.4 | 1.82 \| 189.0 | 1.81 \| 190.1 | 1.71 \| 201.4 | 1.70 \| 202.2 | 5.69 \| 60.3 | 2.09 \| 164.5 | **1.29 \| 267.2** | — |
| pro | 4 | 15.10 \| 204.8 | 13.48 \| 229.3 | 10.74 \| 287.8 | 8.52 \| 362.9 | 8.52 \| 362.9 | 30.45 \| 101.5 | 9.41 \| 328.7 | **6.32 \| 489.3** | — |
| pro | 128 | 5.55 \| 77.3 | 2.42 \| 177.4 | 2.47 \| 174.2 | 2.27 \| 189.4 | 2.27 \| 189.4 | 6.94 \| 61.9 | 2.91 \| 147.6 | **1.49 \| 288.2** | — |

## `_turbo_flydsl` (integrated turbo) vs `gluon_v3` (best in-tree)

`_turbo_flydsl` is the fastest backend on **every** fwd and bwd cell. Speedup over
`gluon_v3` (TFLOP/s ratio):

| variant | cr | FWD speedup | BWD speedup |
|---|---:|---:|---:|
| flash | 0 | 1.35× | 1.70× |
| flash | 4 | 1.24× | 1.57× |
| flash | 128 | 1.46× | 1.98× |
| pro | 0 | 1.33× | 1.32× |
| pro | 4 | 1.36× | 1.35× |
| pro | 128 | 1.41× | 1.52× |
| **mean** | | **~1.36×** | **~1.57×** |

## Summary

- **`_turbo_flydsl`** (the integrated Primus-Turbo native-FlyDSL backend, selectable
  in the model via `use_v4_attention_backend = turbo`) is the fastest backend across
  all six shapes in both directions — **~1.36× fwd** and **~1.57× bwd** over the best
  in-tree backend (`gluon_v3`), and larger margins over `triton_v2`/`gluon_v2`.
- The biggest wins are the CSA `cr=4` forward (pro cr=4: `1.41 ms` / 878 TF vs
  gluon_v3 `1.92 ms` / 645 TF) and the backward across the board (flash cr=128 bwd
  `0.78 ms` / 275 TF ≈ **2×** gluon_v3's `1.54 ms` / 139 TF). The fully-native FlyDSL
  backward is the headline improvement.
- `_turbo_flydsl` also clearly beats the older extracted `turbo_flydsl` (June branch),
  chiefly on the backward (e.g. pro cr=4 bwd `6.32 ms` vs `9.41 ms`).
- `gluon_v3` remains the strongest **in-tree** backend but is benchmark-only (not wired
  for training); `gluon_v2` is the wired gluon training backend.

## Reproduce

```bash
PYTHONPATH=<repo> python deepseek-v4/benchmark/bench_v4_attention.py \
    --variant both --cr all --warmup 10 --iters 30
```
