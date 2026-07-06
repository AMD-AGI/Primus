# DeepSeek-V4 Attention — Backend Performance

Full forward + backward benchmark of every V4 attention backend, produced by
[`bench_v4_attention.py`](./bench_v4_attention.py).

## Setup

- **GPU**: AMD Instinct MI355X (gfx950), single GPU (`smci355-ccs-aus-n03-33`)
- **Config**: `seq_len=4096`, `mbs=1`, bf16, sink **on**, `swa_window=128`
- **Models**: V4-Flash (`H=64`, index_topk=512), V4-Pro (`H=128`, index_topk=1024)
- **Layer kinds**: `cr=0` dense/SWA, `cr=4` CSA (local SWA + sparse top-k pool),
  `cr=128` HCA (local SWA + full compressed pool)
- **Timing**: `--warmup 10 --iters 30`, median latency
- **Cell format**: `latency ms | TFLOP/s`. TFLOP/s is over the *useful* work
  (`2·T·H·TOPK·(D_V+D_V)`, head_dim=512; bwd = 2.5× fwd) with the **same formula
  for every backend**, so rows are directly comparable.

## Backends

| Backend | Description |
|---------|-------------|
| `triton` | Production separate-K/V Triton (dense/SWA/HCA launchers; cr=4 split CSA pool fwd + segreduce bwd) |
| `gluon` | Fused single-latent (K==V) sparse-MLA, 1st-gen hand-tuned gfx950 gluon (`_gluon_dsa`; async double-buffered pipeline) |
| `triton_v2` | Fused single-latent sparse-MLA in plain Triton (`tl.dot` / MFMA) |
| `gluon_v2` | **Fused single-latent sparse-MLA, 2nd-gen gluon (`_gluon_v2`) — fastest backend.** Gluon fwd (rope-skip + exp2 softmax + MFMA K=32) + Gluon bwd (rope-skip + MFMA K=32 + single-chunk dQ RMW) |
| `flydsl_v1` | Fused single-latent sparse-MLA in native FlyDSL MFMA (fwd + bwd) |
| `aiter_gluon` | aiter PR#3833 gluon sparse-MLA **prefill (forward-only)** reference (`mla_gluon`, has_pe=False); bwd is not provided |

> `aiter_gluon` is a forward-only reference, so its backward cell is `— (fwd-only)`
> (the harness reports a `NoneType` bwd, expected). `flydsl_v1` depends only on the
> installed `flydsl` pip package.

## Forward (`latency ms | TFLOP/s`, higher TFLOP/s = better; **bold** = fastest)

| variant | cr (layer) | triton | gluon | triton_v2 | gluon_v2 | flydsl_v1 | aiter_gluon |
|---------|-----------|--------|-------|-----------|----------|-----------|-------------|
| flash | 0 (SWA)   | 0.47 \| 147.1 | 0.33 \| 208.7 | 0.32 \| 217.6 | **0.30 \| 232.7** | 0.45 \| 152.8 | 0.50 \| 137.2 |
| flash | 4 (CSA)   | 1.48 \| 232.5 | 0.92 \| 372.6 | 0.88 \| 391.4 | **0.74 \| 465.0** | 1.39 \| 246.9 | 0.89 \| 385.2 |
| flash | 128 (HCA) | 0.76 \| 113.8 | 0.41 \| 210.6 | 0.39 \| 219.0 | **0.35 \| 248.8** | 0.59 \| 146.6 | 0.53 \| 161.1 |
| pro | 0 (SWA)     | 0.86 \| 160.7 | 0.58 \| 237.0 | 0.58 \| 238.2 | **0.53 \| 258.0** | 1.06 \| 129.5 | 0.83 \| 166.4 |
| pro | 4 (CSA)     | 4.47 \| 276.5 | 2.92 \| 423.6 | 2.78 \| 444.7 | **2.36 \| 523.3** | 4.82 \| 256.4 | 2.36 \| 523.5 |
| pro | 128 (HCA)   | 1.48 \| 116.5 | 0.72 \| 237.5 | 0.71 \| 241.3 | **0.61 \| 279.9** | 1.20 \| 142.9 | 0.93 \| 185.7 |

`gluon_v2` is fastest on all 6 fwd shapes; at pro cr=4 it ties the aiter_gluon
prefill reference (523.3 vs 523.5) and beats it on the other 5.

## Backward (`latency ms | TFLOP/s`, higher TFLOP/s = better; **bold** = fastest)

| variant | cr (layer) | triton | gluon | triton_v2 | gluon_v2 | flydsl_v1 | aiter_gluon |
|---------|-----------|--------|-------|-----------|----------|-----------|-------------|
| flash | 0 (SWA)   | 2.14 \| 80.3  | 1.24 \| 138.6 | 1.19 \| 144.5 | **1.16 \| 148.3** | 2.23 \| 76.9 | — (fwd-only) |
| flash | 4 (CSA)   | 5.27 \| 163.1 | 5.29 \| 162.2 | 6.06 \| 141.8 | **4.89 \| 175.6** | 6.23 \| 137.9 | — (fwd-only) |
| flash | 128 (HCA) | 2.92 \| 73.7  | 1.67 \| 128.6 | 1.70 \| 126.6 | **1.57 \| 136.9** | 2.81 \| 76.5 | — (fwd-only) |
| pro | 0 (SWA)     | 4.09 \| 83.9  | 1.86 \| 184.5 | 1.86 \| 184.4 | **1.74 \| 197.2** | 5.76 \| 59.7 | — (fwd-only) |
| pro | 4 (CSA)     | 15.20 \| 203.5| 13.49 \| 229.2| 10.77 \| 287.1| **8.58 \| 360.2** | 30.51 \| 101.4 | — (fwd-only) |
| pro | 128 (HCA)   | 5.61 \| 76.6  | 2.45 \| 175.2 | 2.46 \| 174.4 | **2.27 \| 189.0** | 6.98 \| 61.5 | — (fwd-only) |

`gluon_v2` is fastest on all 6 bwd shapes.

## gluon_v2 vs triton_v2 (speedup = triton_v2_ms / gluon_v2_ms; >1 = gluon_v2 faster)

| variant | cr | FWD speedup | BWD speedup |
|---------|----|-------------|-------------|
| flash | 0   | 1.07× | 1.03× |
| flash | 4   | 1.19× | 1.24× |
| flash | 128 | 1.11× | 1.08× |
| pro | 0     | 1.09× | 1.07× |
| pro | 4     | 1.18× | 1.26× |
| pro | 128   | 1.16× | 1.08× |
| **geomean** | | **1.13×** | **1.12×** |

## Summary

- **`gluon_v2` is the fastest backend on all 6 shapes for both directions** — the
  new default. vs `triton_v2` (the previous best): forward **~1.13× geomean**
  (up to 1.19× at flash cr=4), backward **~1.12× geomean** (up to 1.26× at pro
  cr=4). Forward geomean 317 TFLOP/s (vs triton_v2 280); backward geomean 190
  TFLOP/s (vs triton_v2 170).
- `gluon_v2` wins came from porting aiter-derived techniques into the fused
  single-latent gluon kernels: **rope-skip** (V4 zero-rope-pad ⇒ skip the
  provably-zero rope MMAs + free the rope accumulator VGPR), **exp2 softmax**
  (forward), and **MFMA K=32** for the D_V=512-reduction matmuls; the backward
  additionally uses a **single-chunk dQ read-modify-write** for high head counts.
- `aiter_gluon` (forward-only prefill reference) matches `gluon_v2` at pro cr=4
  (523.5 vs 523.3) but trails it on the other 5 forward shapes; it has no backward.
- `flydsl_v1` remains a correct but slower native-FlyDSL backend (register-bound;
  its large-topk backward runs the whole top-k as one chunk with a big dKV
  intermediate).

## Reproduce

```bash
PYTHONPATH=<repo> python deepseek-v4/benchmark/bench_v4_attention.py \
    --variant both --cr all --warmup 10 --iters 30
```
