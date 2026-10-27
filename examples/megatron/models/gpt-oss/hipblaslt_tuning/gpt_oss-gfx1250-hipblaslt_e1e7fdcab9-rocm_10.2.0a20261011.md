# GEMM performance with gpt_oss-gfx1250-hipblaslt_e1e7fdcab9-rocm_10.2.0a20261011.txt

GPT-OSS 20B and 120B have the same hidden size (2880), attention (64 heads, 8 KV groups, head
dim 64), expert and shared expert sizes (2880) and vocabulary; they differ only in layer count
(24 / 36) and number of experts (32 / 128). At MBS 8 and sequence 4096 their attention, shared
expert and output layer GEMMs are identical. Router and expert GEMMs depend on the expert count
and on EP. The file has 37 rows and serves both models.

Each GEMM below was timed with hipBLASLt's default kernel selection and with the tuning file.
Measured 2026-10-23 on MI455X, image `amdprimus/amdprimus:tas-20261019` (hipBLASLt 1.5.0
`e1e7fdcab9`), with the `hipblaslt-bench` built as in `../README.md` ("Building hipblaslt-bench").

- Each shape is the dumped training call, replayed through the C API heuristic query
  (`--api_method c --algo_method heuristic --requested_solution 1`), which is how PyTorch and
  Transformer Engine pick the kernel: once without and once with `HIPBLASLT_TUNING_OVERRIDE_FILE`.
  One GPU per run, 512 MB rotating buffer.
- Times are at the steady-state clock. Under a continuous GEMM loop the clock drops from about
  2.2 GHz to about 1.3 GHz within a second; 2880 x 32768 x 5760 (NN) takes 447 us averaged over
  50 iterations, 519 us over 1000 and 532 us over 5000. Each run therefore warms up for 1.5 s and
  times 1 s. Default and tuned alternate twice per shape, and the median is kept. Short runs of 50
  to 100 iterations give 10-15% lower times.
- Kernel is the solution index and macro tile. "same" means the row selects the default kernel;
  "no row" means the file has no row for the shape (see below).

## Per GEMM

Expert GEMMs are labeled by tokens per expert: 20B EP2 is 8192 (20B, EP2, MBS 8, the 2-GPU
runs), 20B EP4 is 16384 (20B, EP4, MBS 8) and 120B EP4 is 4096 (120B, EP4, MBS 8, without Turbo
grouped GEMM).

| GEMM | layout, m x n x k | default kernel | tuned kernel | default us | tuned us | TFLOP/s | speedup |
|---|---|---|---|---|---|---|---|
| output layer fprop | TN 201216 x 32768 x 2880 | 1409 MT256x256x128 | 1437 MT256x256x128 | 26017.3 | 17531.8 | 1460 -> 2166 | 1.48x |
| output layer dgrad | NN 2880 x 32768 x 201216 | 912 MT240x256x128 | 1428 MT256x256x128 | 18279.9 | 16586.7 | 2078 -> 2290 | 1.10x |
| output layer wgrad | NT 2880 x 201216 x 32768 | 484 MT240x256x128 | same | 17795.2 | 17790.9 | 2134 |  |
| QKV fprop | TN 5120 x 32768 x 2880 | 1409 MT256x256x128 | 1437 MT256x256x128 | 586.3 | 424.7 | 1648 -> 2276 | 1.38x |
| QKV dgrad | NN 2880 x 32768 x 5120 | 912 MT240x256x128 | same | 494.8 | 494.9 | 1953 |  |
| QKV wgrad | NT 2880 x 5120 x 32768 | 484 MT240x256x128 | same | 457.3 | 457.3 | 2113 |  |
| attention proj fprop | TN 2880 x 32768 x 4096 | 1358 MT240x256x128 | same | 390.9 | 390.8 | 1978 |  |
| attention proj dgrad | NN 4096 x 32768 x 2880 | 952 MT256x256x128 | same | 454.5 | 454.3 | 1701 |  |
| attention proj wgrad | NT 4096 x 2880 x 32768 | 520 MT256x192x128 | same | 376.1 | 375.6 | 2056 |  |
| shared expert fc1 fprop | TN 5760 x 32768 x 2880 | 1358 MT240x256x128 | same | 669.5 | 671.1 | 1624 |  |
| shared expert fc1 dgrad | NN 2880 x 32768 x 5760 | 912 MT240x256x128 | same | 540.7 | 541.3 | 2011 |  |
| shared expert fc1 wgrad | NT 2880 x 5760 x 32768 | 393 MT176x192x128 | 445 MT192x256x128 | 733.4 | 596.3 | 1482 -> 1823 | 1.23x |
| shared expert fc2 fprop | TN 2880 x 32768 x 2880 | 1358 MT240x256x128 | 1357 MT240x256x64 | 327.3 | 312.1 | 1661 -> 1742 | 1.05x |
| shared expert fc2 dgrad | NN 2880 x 32768 x 2880 | 912 MT240x256x128 | same | 353.4 | 353.9 | 1538 |  |
| shared expert fc2 wgrad | NT 2880 x 2880 x 32768 | 433 MT192x176x128 | 356 MT144x256x128 | 368.7 | 321.7 | 1475 -> 1689 | 1.15x |
| router fprop (fp32 out), 20B | TN 32 x 32768 x 2880 | 1452 MT256x128x64 | 1453 MT16x16x64 | 163.6 | 45.7 | 37 -> 132 | 3.58x |
| router dgrad (fp32), 20B | NN 2880 x 32768 x 32 | 2877 MT32x32x16 | same | 284.5 | 284.3 | 21 |  |
| router wgrad (fp32), 20B | NT 2880 x 32 x 32768 | 2876 MT32x32x16 | same | 1040.7 | 1040.4 | 6 |  |
| router fprop (fp32 out), 120B | TN 128 x 32768 x 2880 | 1452 MT256x128x64 | 1453 MT16x16x64 | 172.1 | 137.5 | 140 -> 176 | 1.25x |
| router dgrad (fp32), 120B | NN 2880 x 32768 x 128 | 2877 MT32x32x16 | same | 427.2 | 427.4 | 57 |  |
| router wgrad (fp32), 120B | NT 2880 x 128 x 32768 | 2876 MT32x32x16 | same | 1057.7 | 1057.0 | 23 |  |
| expert fc1 fprop, 20B EP2 | TN 5760 x 8192 x 2880 | 1358 MT240x256x128 | 1357 MT240x256x64 | 160.6 | 155.0 | 1692 -> 1754 | 1.04x |
| expert fc1 dgrad, 20B EP2 | NN 2880 x 8192 x 5760 | 872 MT192x256x128 | same | 145.4 | 145.4 | 1869 |  |
| expert fc1 wgrad, 20B EP2 | NT 2880 x 5760 x 8192 | 393 MT176x192x128 | 436 MT192x192x128 | 207.4 | 177.3 | 1310 -> 1533 | 1.17x |
| expert fc2 fprop, 20B EP2 | TN 2880 x 8192 x 2880 | 1307 MT192x256x128 | no row | 91.2 | 91.3 | 1490 |  |
| expert fc2 dgrad, 20B EP2 | NN 2880 x 8192 x 2880 | 872 MT192x256x128 | same | 87.2 | 87.3 | 1559 |  |
| expert fc2 wgrad, 20B EP2 | NT 2880 x 2880 x 8192 | 433 MT192x176x128 | 356 MT144x256x128 | 98.7 | 88.2 | 1377 -> 1541 | 1.12x |
| expert fc1 fprop, 20B EP4 | TN 5760 x 16384 x 2880 | 1358 MT240x256x128 | 1357 MT240x256x64 | 335.0 | 312.6 | 1623 -> 1739 | 1.07x |
| expert fc1 dgrad, 20B EP4 | NN 2880 x 16384 x 5760 | 912 MT240x256x128 | same | 273.1 | 273.7 | 1990 |  |
| expert fc1 wgrad, 20B EP4 | NT 2880 x 5760 x 16384 | 393 MT176x192x128 | 356 MT144x256x128 | 376.9 | 333.7 | 1442 -> 1629 | 1.13x |
| expert fc2 fprop, 20B EP4 | TN 2880 x 16384 x 2880 | 1358 MT240x256x128 | 1357 MT240x256x64 | 163.3 | 160.0 | 1664 -> 1698 | 1.02x |
| expert fc2 dgrad, 20B EP4 | NN 2880 x 16384 x 2880 | 912 MT240x256x128 | same | 172.5 | 172.4 | 1576 |  |
| expert fc2 wgrad, 20B EP4 | NT 2880 x 2880 x 16384 | 433 MT192x176x128 | 356 MT144x256x128 | 188.5 | 166.4 | 1442 -> 1633 | 1.13x |
| expert fc1 fprop, 120B EP4 | TN 5760 x 4096 x 2880 | 1307 MT192x256x128 | 1306 MT192x256x64 | 91.4 | 90.9 | 1486 -> 1495 | 1.01x |
| expert fc1 dgrad, 120B EP4 | NN 2880 x 4096 x 5760 | 872 MT192x256x128 | same | 75.7 | 75.8 | 1794 |  |
| expert fc1 wgrad, 120B EP4 | NT 2880 x 5760 x 4096 | 393 MT176x192x128 | 356 MT144x256x128 | 127.9 | 94.4 | 1063 -> 1440 | 1.35x |
| expert fc2 fprop, 120B EP4 | TN 2880 x 4096 x 2880 | 1307 MT192x256x128 | no row | 45.8 | 45.9 | 1483 |  |
| expert fc2 dgrad, 120B EP4 | NN 2880 x 4096 x 2880 | 872 MT192x256x128 | same | 45.7 | 45.7 | 1487 |  |
| expert fc2 wgrad, 120B EP4 | NT 2880 x 2880 x 4096 | 433 MT192x176x128 | 356 MT144x256x128 | 54.9 | 51.8 | 1238 -> 1313 | 1.06x |

- The shared expert and expert fc1 / fc2 wgrad (NT) GEMMs gain 6-35%. hipBLASLt picks a 176x192
  or 192x176 tile for them by default; the tuned kernels use 144x256, 192x256 or 192x192.
- Output layer fprop and QKV fprop keep the 256x256x128 tile, but kernel 1437 (32x4 waves per
  workgroup) replaces 1409 (64x2) and is 1.4-1.5x faster.
- The router fprop has 32 (20B) or 128 (120B) output rows, and the default kernel uses a 256x128
  tile for it; a 16x16 tile is 3.6x and 1.25x faster.
- Apart from the output layer, every dgrad (NN) GEMM keeps the default kernel.
- The fp32 router wgrad runs at 6 TFLOP/s (20B) and 23 TFLOP/s (120B), about 1.05 ms per call,
  and the 120B router dgrad takes 0.43 ms. Tuning found no faster hipBLASLt kernel for them.
- `offline_tune_gemm.py` picked kernel 1306 (192x256x64) for the expert fc2 fprop at 8192 and 4096
  tokens per expert. In steady-state A/Bs (4 alternating rounds, about 1 s each) the default
  kernel 1307 (192x256x128) was faster in every round, by 0.4% and 2.8%, so the file has no row for
  these two shapes and hipBLASLt keeps 1307.

## Per iteration

Per GPU and iteration, weighted by how often each GEMM runs, in ms. Every configuration has 2
micro batches per GPU and iteration; each attention, shared expert and router GEMM runs once per
layer and micro batch, each expert GEMM once per local expert, layer and micro batch, and each
output layer GEMM once per micro batch.

- 20B, 2 GPUs, 12 layers: EP2 DP2, MBS 8 / GBS 32, 16 local experts.
- 20B, 4 GPUs: 24 layers, EP4 DP4, MBS 8 / GBS 64, 8 local experts.

| GEMMs | 20B, 2 GPUs, 12 layers default | 20B, 2 GPUs, 12 layers tuned | 20B, 4 GPUs default | 20B, 4 GPUs tuned |
|---|---|---|---|---|
| output layer | 124.2 | 103.8 (-20.4) | 124.2 | 103.8 (-20.4) |
| attention | 66.2 | 62.3 (-3.9) | 132.5 | 124.7 (-7.8) |
| shared expert | 71.8 | 67.1 (-4.7) | 143.7 | 134.2 (-9.4) |
| router | 35.7 | 32.9 (-2.8) | 71.5 | 65.8 (-5.7) |
| experts | 303.6 | 285.9 (-17.7) | 579.6 | 544.8 (-34.8) |
| **total** | 601.5 | 552.0 (-49.5) | 1051.4 | 973.3 (-78.1) |

- 120B, 2 GPUs, 4 layers: EP2 DP2, MBS 8 / GBS 32, with the 120B settings from `../README.md`
  ("120B tuning"). Those use Turbo grouped GEMM, so the experts do not use hipBLASLt.
- 120B proxy, 4 GPUs: 8 layers, EP4 DP4, MBS 8 / GBS 64, the same settings; the last two columns
  without Turbo grouped GEMM (32 local experts).

| GEMMs | 120B, 2 GPUs, 4 layers default | 120B, 2 GPUs, 4 layers tuned | 120B proxy, 4 GPUs default | 120B proxy, 4 GPUs tuned | 120B proxy, 4 GPUs, no grouped GEMM default | 120B proxy, 4 GPUs, no grouped GEMM tuned |
|---|---|---|---|---|---|---|
| output layer | 124.2 | 103.8 (-20.4) | 124.2 | 103.8 (-20.4) | 124.2 | 103.8 (-20.4) |
| attention | 22.1 | 20.8 (-1.3) | 44.2 | 41.6 (-2.6) | 44.2 | 41.6 (-2.6) |
| shared expert | 23.9 | 22.4 (-1.6) | 47.9 | 44.7 (-3.1) | 47.9 | 44.7 (-3.1) |
| router | 13.3 | 13.0 | 26.5 | 26.0 (-0.6) | 26.5 | 26.0 (-0.6) |
| experts |  |  |  |  | 226.0 | 207.0 (-19.0) |
| **total** | 183.5 | 159.9 (-23.5) | 242.7 | 216.1 (-26.7) | 468.7 | 423.1 (-45.6) |

Against the measured iteration times (`../README.md`, "GEMM tuning"):

- 20B, 2 GPUs: the GEMMs are half of the iteration. The file saves 49.5 ms of GEMM time; the
  measured iteration got 55.8 ms faster (1198.1 to 1142.3 ms).
- 120B, 2 GPUs, 4 layers: the file saves 23.5 ms of GEMM time; the measured iteration got 32.3 ms
  faster (540.6 to 508.3 ms).
- 20B, 4 GPUs: the GEMMs are 1051 ms of the 2200 ms iteration. The file saves 78.1 ms of GEMM
  time; the measured iteration got 93.4 ms faster (2200.2 to 2106.8 ms, 4.2%).
- 120B proxy, 4 GPUs, with the README's 120B settings: the file saves 26.7 ms of GEMM time; the
  measured iteration got 31.9 ms faster (849.6 to 817.7 ms, 3.8%).

## In training

20B on 4 GPUs with the 20B yaml (EP4 DP4, MBS 8 / GBS 64), PyTorch profiler on rank 0 over
iterations 7-8, `--profile True --profile_step_start 6 --profile_step_end 8`, 2026-10-23. GPU
kernel time per iteration in ms, summed over streams; each backward GEMM is attributed to its
module through the autograd sequence number of its forward call. TFLOP/s divides the GEMM FLOPs by
that kernel time.

| component | default | tuned | TFLOP/s |
|---|---|---|---|
| output layer fwd | 49.3 | 28.8 (-20.5) | 1542 -> 2635 |
| output layer bwd | 67.4 | 61.3 (-6.1) | 2253 -> 2480 |
| QKV fwd | 27.1 | 19.0 (-8.1) | 1713 -> 2436 |
| QKV bwd | 39.8 | 39.9 | 2333 |
| attention proj fwd | 16.9 | 17.0 | 2201 |
| attention proj bwd | 36.3 | 36.2 | 2047 |
| shared expert fc1 fwd / bwd | 28.7 / 54.9 | 29.1 / 51.5 (-3.4) | 1817 / 1900 -> 2025 |
| shared expert fc2 fwd / bwd | 13.9 / 31.5 | 13.5 / 28.6 (-3.3) | 1872 -> 1929 / 1659 -> 1824 |
| experts fwd | 358.8 | 320.6 (-38.2) | 873 -> 977 |
| experts bwd | 695.6 | 636.8 (-58.8) | 900 -> 983 |
| router GEMMs (fp32) | 73.4 | 67.6 (-5.8) | |
| attention core fwd / bwd | 108.1 / 460.6 | 109.5 / 459.4 | |
| RCCL | 210.2 | 209.4 | |
| optimizer, RMSNorm, loss | 53.8 | 53.0 | |
| other kernels | 358.3 | 358.0 | |
| **kernel total** | 2684.3 | 2539.5 | |
| iteration (wall) | 2196.1 | 2104.4 | |

- TE GroupedLinear issues 16 forward and 32 backward GEMMs per layer and micro batch (8 local
  experts) over several streams, so the expert rows add up overlapping kernels.
- The other linear GEMMs reach 1.5-2.6 PF/s in training, faster than the steady-state times above.
- With the default kernels, the GPU is busy for 2153 ms of the 2196 ms iteration. "Other kernels"
  includes the FP32 gradient accumulation into `main_grad` (55.5 ms), permute / unpermute and
  elementwise work.
