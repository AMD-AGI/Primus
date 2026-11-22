# DeepSeek-V4 Flash proxy on 1x MI455X: opt3 configuration

Branch `yyq/opt3-int` (Primus + Primus-Turbo). Single GPU, EP=1, 32 experts (one EP8 rank of the 256-expert
proxy), MBS=GBS=1, seq 4096. Precision follows the DeepSeek-V4 paper: FP8 E4M3 with 128x128 weight and 1x128
activation blocks and UE8M0 scales for experts and dense linears; BF16 for embedding, lm_head, router, norms and
the attention core.

## Run

Inside the Primus container (Primus at `/workspace/Primus`, Primus-Turbo at `/workspace/Primus-Turbo`):

```bash
export TRITON_MAIN=<triton checkout>/python   # Triton main with Gluon cdna5 (tested: 3.9.0 83c9228329)
bash examples/megatron/models/deepseek-v4/mi455x_opt3/run_opt3_1gpu.sh graph 20
```

`graph` = Megatron full-iteration CUDA graph (`--cuda_graph_impl full_iteration`); `eager` runs without it.

Requirements:
- Triton main for the Gluon kernels (dsv4up dense GEMM). The image Triton 3.8.0 is only used without
  `PRIMUS_TURBO_FP8_BLOCK_KERNEL=dsv4up`; with it, keep `TRITON_HIP_USE_EXPERT_SCHEDULING=0` (async-LDS-copy miscompile).
- `fla_stub/` shadows the image `fla` package, which fails to import under Triton main. DeepSeek-V4 does not use it.
- `tunableop_dsv4_gfx1250_0.csv`: hipBLASLt TunableOp results for lm_head fwd/dgrad and the grouped o_a bmm
  fwd/dgrad on gfx1250 (PyTorch inserts the device ordinal, hence `_0`).
- Primus-Turbo built for gfx1250 (`primus_turbo/pytorch/_C*.so`, `primus_turbo/lib`).

## Results (1x MI455X, SPX/NPS1, 20 iters, median of the last 10, 2026-11-21)

| configuration | ms/iter | loss @ iter 20 |
|---|---|---|
| baseline: g90host branch, paper recipe, eager | 116 | 9.8688 |
| baseline, full-iteration graph | 97.0 | 9.8688 |
| yyq/opt3-int, all opt3 switches off, graph | 92.7 | 9.8712 |
| all opt3 switches on (+ new fused Adam), graph | 77.9 | 9.8687 |
| + MegaMoE fp8_block experts, graph | 62.1 | 9.8715 |
| + Adam writes the MegaMoE expert caches, graph | 61.9 | 9.8715 |
| + fuseq switches, graph | 60.7 | 9.8715 |
| + MegaMoE flygg k256, graph | 60.1 | 9.8715 |
| + BF16 GEMM layouts and TunableOp for all BF16 shapes, graph | 59.5 | 9.8710 |
| + lagged step logging, DP=1 all-gather skip, graph | 58.2 | 9.8710 |
| + MoE wgrad sum-of-squares fold, dsv4up for all passes, graph | 57.5 | 9.8695 |
| + Adam skips the MegaMoE expert BF16 store, graph | 57.3 (A/B -0.6 ms, see note) | 9.8695 |
| + glue switches, graph | 56.1 (A/B -1.1 ms) | 9.8695 |
| + hc config C, graph | 54.8 (A/B -1.3 ms) | 9.8695 |
| + MoE vkp wgrad sched2, graph | 54.65 (A/B 54.9 -> 54.65) | 9.8695 |
| + wq_a\|\|wkv block-FP8 concat, graph | 54.4 (A/B 54.8 -> 54.4) | 9.8718 |
| + mxgemm mxgk grouped GEMM (GG=mxk), graph | 54.3 (A/B 54.7 -> 54.3) | 9.8718 |
| + fuse-moe ROUTE / MOE_BWD / MOE_ADD, graph | 53.07 (A/B 54.27 -> 53.07) | 9.8710 |
| + adamattn wave-dense tile Adam v10, graph | 51.8 (A/B 53.17 -> 51.83) | 9.8710 |
| + fuse-hc GRAD_DIRECT / B3X / XRMS, graph | 51.0 (A/B 51.70 -> 51.03) | 9.8710 |
| + mxgemm persistent mxgk (GG=mxp) + fuse-moe SWB_TDM, graph | 50.8 (A/B 51.13 -> 50.83) | 9.8710 |
| + fuse-attn F1/F3/F4/F5, graph | 49.9 (A/B 50.90 -> 49.87) | 9.8710 |
| + densegemm SMALLTILE / EPI2 / CE_BWD_GT, graph | 49.5 (A/B 49.87 -> 49.53) | 9.8710 |
| + overlap side streams, graph | 49.1 (A/B 49.47 -> 49.10) | 9.8710 |
| + membound QDUAL/PACKQ/GNORM/ROPE_TBL/SMALLS/QNR, graph | 48.4 (A/B 49.10 -> 48.40) | 9.8712 |
| + wgrad vkq (MoE wgrad scale planes via async VMEM), graph | 48.1 (A/B 48.50 -> 48.13) | 9.8712 |
| + small batch: membound R2 (QDUAL c4096, PACKQ_SKIP f4096, ROUTER_FWD 128,4), splitk HC_TAILBW 512, attnfwd SMLA_FWD_PIPE, graph | 48.1 (A/B 48.23 -> 48.10) | 9.8712 |
| + attnbwd SMLA_BWD_V2 + Adam HWCVT, graph | 47.5 (A/B 48.00 -> 47.47) | 9.8712 |
| + bf16g2 OA_DGRAD + IDXQ, graph | 47.1 (A/B 47.40 -> 47.10) | 9.8712 |
| + fuse2 WPOOL_DAPE/COMP_MG/LOSS/PREFETCH=2/HEAD_GADD/ATT_GADD + static loss scale, graph | 47.1 (A/B 47.20 -> 47.07) | 9.8712 |
| + persist ROUTER_BWD + MOE_COMBINE_DW, graph | 46.8 (A/B 47.10 -> 46.83) | 9.8712 |
| **+ csabwd CSABWD_PP (this config), graph** | **46.8** (A/B 46.90 -> 46.77) | 9.8712 |

Contribution of each switch (graph mode, turning one off from the all-on 83.2 ms config, before the MegaMoE switch):
HC fused -4.9, dsv4up -2.8, fused router -2.3, TunableOp -1.6, fused CE -1.0, HC fold -0.8, grad-norm reuse -0.5,
wgrad fusion -0.4, wq_a||wkv concat -0.4 ms. FP8-block Adam (`PRIMUS_FP8BLK_ADAM_QUANT`) -4.0 ms with the tile kernel
at 1273 us for 879M params (83% of HBM). MegaMoE fp8_block then -15.7 ms (MoE GEMMs 22.8 -> 8.4 ms/step).
`PRIMUS_FP8BLK_ADAM_MEGAMOE` then -0.3 ms (bit-identical; 6 A/B pairs, track adamqmoe): the tile Adam kernel also
writes the MegaMoE expert q/s/qT/sT caches, so the `_w_128x128` re-quant (1.9 ms) goes away, but the tile kernel
takes +1.8 ms on the expert weights (it still writes the same FP8 bytes and runs at ~14.5 vs 15.9 TB/s).
fuseq switches then -1.4 ms (bit-identical; 2 A/B pairs in the int tree, 62.05 -> 60.65): `PRIMUS_TURBO_FUSEQ` (1x128
dual quant writes scales in the consumer GEMM layout, half the `_pack_scales` launches gone), `PRIMUS_V4_ROPE_BWD_INPLACE`
(inverse RoPE bwd on the 64 rope columns only), `PRIMUS_V4_OA_MAIN_GRAD` and `PRIMUS_EMB_MAIN_GRAD` (dW straight
into main_grad, no DDP copy), `PRIMUS_V4_ROUTER_DENSE_OUT` (no bool zero fills). Kernels/step 2178 -> 2048.
`PRIMUS_TURBO_MOE_FP8BLK_GG=k256` then -0.65 ms (bit-identical; 2 A/B pairs in the int tree, 60.75 -> 60.1): the
MegaMoE fp8_block fwd/dgrad grouped GEMM (flygg) uses 256x256x256 tiles, a 2-stage ring, a M/256 grid bound and one
B-scale row per 16-row fragment (kernel -8..-13%, flygg 4.87 -> 4.30 ms/step). It assumes 128x128-block weight scales
and 256-aligned expert offsets; `tile` restores the old kernel.
bf16gemm then -0.75 ms (2 A/B pairs in the int tree, 60.2 -> 59.45; loss@20 9.871018, grad norm 4.152, inside the
9.868-9.872 spread, so NOT bit-identical): lm_head wgrad via torch mm into main_grad on Triton-transposed operands (TN,
tuned), o_a dgrad as a TN bmm, and the `tunableop_dsv4_gfx1250_bf16_0.csv` TunableOp file. The bit-identical variant
`PRIMUS_LM_HEAD_WGRAD_LAYOUT=nt` gave no gain on top of fuseq (60.2 vs 60.2). The TunableOp entries are tied to the
image's hipBLASLt; without the o_a TN entry the o_a dgrad change is slower.
bubble then -1.25 ms (bit-identical; 2 A/B pairs in the int tree, 59.4 -> 58.15): `PRIMUS_BUBBLE_LAG_LOG` snapshots
the log values to host without blocking and prints step k-1 while step k runs, removing the ~2 ms per-step GPU drain
(7 syncs + host logging/zero_grad/graph launch); `PRIMUS_BUBBLE_SKIP_DP1_AG` skips the 32 no-op param all-gathers.
Each iteration's log line prints one step late. Most of the earlier "7.7 ms idle" was profiler overhead (~1.4 us per
launch); real idle was ~3.3 ms, of which ~1.4 ms is the graph-replay per-kernel floor (~1.2 us/kernel).
wgsmall then -0.6 ms (2 A/B pairs in the int tree, 58.05 -> 57.45; loss@20 9.869526, grad norm 4.151, in spread):
`PRIMUS_TURBO_MOE_WGRAD_SUMSQ` (the MegaMoE vkp wgrad epilogue writes per-weight sums of squares, so the grad norm
does not re-read those grads) and an empty `PRIMUS_TURBO_DSV4UP_PREFER_G90` (all supported dense passes on dsv4up).
`PRIMUS_TURBO_DSV4UP_SPLITK=4096x1024x32768:4` (wq_b dgrad split-K) is available but off: -0.35 ms, loss@20 9.8675,
just below the spread.
One-step kernel profile of this config (LAG_LOG off for a clean step window): 2030 kernels, 55.0 ms kernel time;
`~/dsv41/work/int/e2e/c3/tprof_final2.txt` on the dev node.
adamfast then -0.6 ms (bit-identical; 2 A/B pairs in the int tree, 57.9 -> 57.3 with the node ~0.5 ms slower than
at the wgsmall merge; track A/B 3 pairs 58.12 -> 57.27): `PRIMUS_FP8BLK_ADAM_SKIP_BF16` makes the tile Adam skip
the BF16 store of the 16 MegaMoE expert weights (22.06 -> 20.06 B/param); the BF16 weight is rebuilt from the fp32
master (RNE copy, same bytes) before every BF16 reader: Turbo cache refresh/VERIFY (hook), save_checkpoint,
calc_params_l2_norm, check_param_hashes_across_dp_replicas. Model checkpoint at iteration 10 byte-identical to the
baseline. No-op at EP>1, DP>1, with gg_dsv4. Code outside those wrapped functions that reads expert w1/w2 .data
between steps must call `materialize_skipped_bf16_weights()` first.
Optimizer HBM floor at 18.0 TB/s with this: 9.68 ms (tile 8.15 + chunk 1.54) vs 12.59 ms measured.
glue then -1.1 ms (2 A/B pairs in the int tree, 57.2 -> 56.1; track: kernels/step 2030 -> 1878):
`PRIMUS_GLUE_RMSNORM` (LocalRMSNorm bwd with per-program partials + fixed-order reduce instead of fp32 atomics,
dgamma straight into main_grad; changes the compressor kv_norm dgamma bits, now deterministic; logged loss/grad norm
unchanged), `PRIMUS_GLUE_ROPE_NOCAST` (RoPE reads int64 positions / bf16 inv_freq), `PRIMUS_GLUE_COMPRESS_POOL`
(compressor pool kernels read the fused wkv_gate output; the old path still runs for packed sequences / CP),
`PRIMUS_GLUE_EMB_TOUCHED` (clear only the embedding main_grad rows written last step; raises on a second write
per step), `PRIMUS_GLUE_SMALLS` (fused router builds no zero grads). `PRIMUS_GLUE_WQB_DYQ` is correct but slower: off.
hc then -1.3 ms (bit-identical; 2 A/B pairs in the int tree, 56.05 -> 54.75): Triton main routes a 3-D-tile middle-axis
`tl.sum` through LDS + barriers (~3x slower than Triton 3.8), so `_hc_expand_fwd` took 68.8 us in-step;
`PRIMUS_HC_EXPAND_TILE=1,2048,8` and `PRIMUS_HC_POST_TILE=1,1` avoid it (24.3 / 35.1 us, byte-equal);
`PRIMUS_HC_NOCAST` (no torch casts of HC params), `PRIMUS_HC_HEAD_FUSED=1` / `PRIMUS_HC_LIFT_FUSED=1` (HyperHead
collapse and the embedding lift in Triton, torch's summation order), `PRIMUS_HC_B3_VAR=1,0` (B3 W^T in LDS).
The tile choices are bit-identical only under this Triton main build (83c9228329); re-check after a Triton bump.
`HEAD_FUSED=2` (one-pass dpre) puts loss@20 at 9.8672, outside the spread: keep 1.
moegemm2 then -0.25 ms (bit-identical; 2 A/B pairs in the int tree after the 23:03 reboot, 54.9 -> 54.65):
`PRIMUS_TURBO_MOE_FP8BLK_VKP_SCHED2` schedules the MegaMoE vkp wgrad scale loads without per-word LDS waits and decodes
tile parameters with scalar loads (fc1 wgrad 289-294 -> 268-274 us). `PRIMUS_TURBO_MOE_FP8BLK_GG=k256p` (persistent
flygg) is available but off (no clear e2e signal). Merged but off: densefp8's `PRIMUS_V4_QKV_A_CONCAT_FP8BLK`
(wq_a||wkv as one block-FP8 op; ON since the 23:25 A/B: 3 pairs 54.8 -> 54.4 ms, loss@20 9.871798 / grad norm 4.159, in spread; dX summed in fp32 over both K parts, closer to fp64), `PRIMUS_TURBO_DSV4UP_DUAL` and
`PRIMUS_TURBO_DSV4UP_EPI2` (slower in-step / untested). glue `PRIMUS_GLUE_WQB_DYQ` off (slower).
Node note: the SMU stuck state recurred at 22:53 (timings ~8% slow); fixed by reboot + `sudo modprobe amdgpu`.
2026-11-22 batch (int A/B, 3 pairs each, SMU 0):
- mxgemm `PRIMUS_TURBO_MOE_FP8BLK_GG=mxk` (default): mxgk keeps k256's TDM traffic and pipelines fragments across
  k-tiles; MegaMoE fwd/dgrad grouped GEMMs -7..9% per pass, bit-identical, -0.4 ms. `k256` selects the previous kernel.
- fuse-moe `PRIMUS_FUSE_ROUTE` / `PRIMUS_FUSE_MOE_BWD` / `PRIMUS_FUSE_MOE_ADD` (default 1): compact MegaMoE routing from
  the fused router GEMM, backward glue in two launches, routed + shared add inside the combine; 1777 -> 1409
  kernels/step, -1.2 ms. Forward bit-identical; the seq-balance-loss part of the learned routers' d logits differs by
  ~1e-15 (fp32 order), loss@20 9.870956. EP>1 and other router configs keep the dense chain.
- adamattn `PRIMUS_FP8BLK_ADAM_WAVEDENSE` (default 1): v10 tile Adam with wave-dense row accesses, compile-time BF16
  store, LDS-only barriers; in-step 10.84 -> 9.39 ms (87% of its HBM floor), bit-identical, -1.33 ms. 0 rebuilds v9.
- ep: `fused_mega_moe_fp8blk_ep` (Turbo, new files only) runs the fp8_block MegaMoE at EP>1; not wired into Primus yet.
  Single-GPU emulation at R=1/4/8: y/dx/d topk_w bit-identical to EP=1. Use GG=k256p for EP>1.
2026-11-22 batch 2 (int A/B, 3 pairs each, SMU 0, all bit-identical; loss@20 9.870956, grad norm 4.157):
- fuse-hc `PRIMUS_HC_GRAD_DIRECT` / `PRIMUS_HC_B3X` / `PRIMUS_HC_XRMS` (default 1): HC backward writes main_grad in one
  launch; expand(L) also computes HC(L+1)'s rms-linear partials; B3 handles all 4 streams in one program. -0.67 ms.
  Bit-identity depends on the Triton main build's codegen: rerun fusehc's test_fusehc.py / calib_expand.py after a bump.
- mxgemm `PRIMUS_TURBO_MOE_FP8BLK_GG=mxp` (default): persistent mxgk; fuse-moe `PRIMUS_FUSE_MOE_SWB_TDM` (default 1):
  TDM-streamed Gluon SwiGLU backward + dual quant. Together -0.3 ms.
- fuse-attn (default 1): `PRIMUS_V4_OROPE_INPLACE` (in-place inverse RoPE of the attention output),
  `PRIMUS_TURBO_CONCAT_SPLIT_EPI` (wq_a||wkv as one dsv4up GEMM with a split-output epilogue), `PRIMUS_V4_INDEXER_TOPK_TRITON`
  (CSA indexer top-k as one Triton sort, P in 1024..4096), `PRIMUS_V4_QKV_POST_FUSED` (q_a/kv norms + kv RoPE as one op).
  -1.03 ms. Optional bit-identical tile knob `PRIMUS_V4_QNR_FWD_BH=8 PRIMUS_V4_QNR_FWD_NW=2` (not enabled).
- densegemm (default on, bit-identical, -0.33 ms): `PRIMUS_TURBO_DSV4UP_SMALLTILE` sends the listed dense GEMMs
  ("MxNxK[a]:tile", a = beta-1 accumulate call) to the dsv4s smaller-output-tile kernel (wq_b dgrad 64 -> 128 tiles,
  98.6 -> 74.8 us/layer); `PRIMUS_TURBO_DSV4UP_EPI2=1` double-buffered epilogue; `PRIMUS_V4_CE_BWD_GT=1` fused-CE backward
  also writes dlogits^T for the lm_head TN wgrad (removes the 119 us transpose). Set SMALLTILE to empty to disable.
  Dense block-FP8 GEMMs on full grids are at ~85-90% of the measured L2->LDS delivery limit (~24.5 TB/s, ~59 B/clk/CU,
  shader clock ~1.59 GHz under WMMA; mx8w/densegemm REPORTs).
- overlap (default on, bit-identical, -0.37 ms): `PRIMUS_OVERLAP_SHARED/WGRAD/MOE_WGRAD/INDEXER/COMPRESSOR` run the shared
  expert, CSA indexer and HCA compressor forward branches and the main_grad-only wgrads on side streams inside the graph,
  joined after every backward_step. Off when world size > 1 or when DDP grad checks are on. Verified by bit fingerprints
  of params, main_grads and optimizer state.
- membound (default on, -0.7 ms; loss@20 9.871213 in spread because GNORM changes the grad-norm fp32 sum order; the rest is
  byte-identical): QDUAL picks small-code dual-quant variants per call site (the 29 KB variant paid a 20-45 us cold-code
  penalty after GEMMs), PACKQ packs weight scales inside the quant launch, GNORM does the grad-norm sumsq in 2 launches,
  ROPE_TBL memoises RoPE tables, SMALLS drops small fills/copies, QNR_FWD_BH=8/NW=2 retiles q_norm_rope_fwd2.
2026-11-22 batch 3 (int A/B, 3 pairs each, all bit-identical vs the previous config):
- wgrad `PRIMUS_TURBO_MOE_FP8BLK_WGRAD=vkq` (default): vkp with the scale-plane windows loaded by async VMEM instead of the
  in-order TDM queue (-0.37 ms). The overlap-deferred wgrad path honours vkq (fix 871b510d).
- membound round 2, splitk `PRIMUS_SPLITK_HC_TAILBW=512`, attnfwd `PRIMUS_SMLA_FWD_PIPE=1` (-0.13 ms together).
- attnbwd `PRIMUS_SMLA_BWD_V2=1` (CSA sparse-MLA backward: no exposed L2 prefetch, TDM-staged Q/dO/O, fused count scan;
  855 -> 753 us/layer) and adamfetch `PRIMUS_FP8BLK_ADAM_HWCVT=1` (hardware FP32->BF16 rounding, exhaustively equal to c10
  RNE; tile Adam 9.36 -> 8.99 ms): -0.53 ms together.
- bf16g2 `PRIMUS_BF16G2_OA_DGRAD=1` (o_a dgrad as a Gluon NN GEMM, bitdiff 0 vs hipBLASLt, no w transposes) and
  `PRIMUS_BF16G2_IDXQ=1` (indexer query/key RoPE + Hadamard + scale chains fused): -0.30 ms together.
- fuse2 (cross-kernel glue): `PRIMUS_FUSE2_WPOOL_DAPE/COMP_MG/LOSS/HEAD_GADD/ATT_GADD=1`, `PRIMUS_FUSE2_PREFETCH=2`,
  `PRIMUS_STATIC_BF16_LOSS_SCALE_CPU=1`: -0.13 ms on int (it measured -0.82 on the older base; overlap, ROPE_TBL and IDXQ now
  cover part of it). `PRIMUS_FUSE2_IDX_ROT` stays 0 (no-op under IDXQ; fallback when IDXQ is off). PREFETCH consumes one extra
  batch at the end of training; LOSS covers the plain GPT loss_func only (other paths use Megatron's code).
- persist `PRIMUS_PERSIST_ROUTER_BWD=1` (MoE router backward 7 launches -> 2: route_bwd + one megakernel with a
  wait-free last-arriver split reduce into main_grad) and `PRIMUS_PERSIST_MOE_COMBINE_DW=1` (combine + route_dw in one
  launch): -0.27 ms, byte-identical (961/961 checksum fingerprints).
- csabwd `PRIMUS_CSABWD_PP=1` (CSA sparse-MLA backward w1 as a warp-specialized ping-pong kernel: score waves and
  C^T waves one tile apart, 2 waves/SIMD, 464 VGPRs, no spills; CSA call ~754 -> ~670 us): -0.13 ms on int, dq/dkv/dsink
  sha identical to V2. Needs SMLA_BWD_V2=1 and H=64 (falls back otherwise).
`PRIMUS_TURBO_MOE_FP8BLK_VERIFY=1` checks the optimizer-written bytes against `quant_w_128x128` every step
(debug only, ~+15 ms/step).

## Known issues

- `PRIMUS_TURBO_WQB_SPLITK=4`: correct but slower than the dsv4up split-K; keep off. Its earlier "NaN" run was the
  transient below, not split-K.
- `PRIMUS_TURBO_MOE_GG_DSV4=1`: 2% faster than flygg at kernel level, not worth enabling.
- MegaMoE `fp8_block` runs the EP=1 stages only; EP>1 dispatch/combine has no FP8-block path yet.
- `PRIMUS_FP8BLK_ADAM_MEGAMOE=1`: after the first step the MegaMoE FP8 expert caches are written only by the fused
  Adam. Anything else that changes the expert weights mid-run (checkpoint load after step 1, an optimizer fallback)
  would leave them stale. It is a no-op at EP>1, for DP-sharded weights and with `PRIMUS_TURBO_MOE_GG_DSV4=1`.
- Rare nondeterminism (WATCH, 4 occurrences on 2026-11-21 in unrelated configs/containers: int_aqm_off1, fq_ab3_all4,
  wgs_c_sumsq_sk4, and the 12:02 loo_plus_splitk run): the iteration-1 forward loss is already off, grad norm NaN on
  every iteration, ~75 ms/step; identical reruns are clean. Treat any run whose iteration-1 loss is not 11.91726 as
  invalid and rerun it. First seen as 1 run in ~30 (int_aqm_off1) had a different iteration-1 loss
  (11.91723 vs 11.91726) and NaN grad norm on every iteration, 75.5 ms/step; an identical rerun was normal. An
  earlier g90-path run once had grad norm inf at iteration 12. Cause unknown; rerun if iteration-1 loss is off.
- The log line "fused kernel hardcodes unclamped SwiGLU" is stale for fp8_block: the kernel applies the default
  SiLU-SwiGLU with a +-10 clamp on gate and up.
