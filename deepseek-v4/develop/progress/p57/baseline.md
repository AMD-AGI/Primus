# Plan-8 P57 — baseline lock

Captured 2026-05-15 on `mi355-gpu-8` / `dev_primus_wenx_693` with
`PRIMUS_V4_ATTN_BWD_USE_SPLIT=1 PRIMUS_V4_CSA_BWD_SEGREDUCE=1`
(the P32 RoPE-fix final defaults).

V4-Flash widths: `B=1, H=64, Sq=4096, D=512, swa_window=128,
sink=True, bf16`.

## cr=0 (dense + SWA + sink) — `bench_v4_attention_ep8.py --mode dense`

| metric | value |
|---|---:|
| FWD median | 0.766 ms |
| BWD median | **7.659 ms** |
| Peak memory | 7.00 GiB |

## cr=128 (HCA split-mask) — `bench_v4_attention_ep8.py --mode hca`

| metric | value |
|---|---:|
| FWD median | 0.906 ms |
| BWD median | **11.890 ms** |
| Peak memory | 7.03 GiB |
| Pool size P | 32 |
| `hca_local_seqlen` | 4096 |
| Sk | 4128 |

## cr=4 (CSA: local SWA + sparse top-K + sink) — `bench_csa_attention_ep8.py`

| metric | value |
|---|---:|
| FWD median | **3.182 ms** |
| BWD median | **16.286 ms** |
| Peak memory | 11.63 GiB |
| K_topk | 512 |
| P (pool size) | 1024 |

## P57 targets

| Kernel | Baseline | Target | Speedup |
|---|---:|---:|---:|
| cr=0 BWD | 7.66 ms | **≤ 3.0 ms** | **2.55×** |
| cr=4 FWD | 3.18 ms | **≤ 1.5 ms** | **2.12×** |
| cr=4 BWD | 16.29 ms | **≤ 5.0 ms** | **3.26×** |
| cr=128 BWD | 11.89 ms | **≤ 3.0 ms** | **3.96×** |
