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
| **+ MegaMoE flygg k256 (this config), graph** | **60.1** | 9.8715 |

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
`PRIMUS_TURBO_MOE_FP8BLK_VERIFY=1` checks the optimizer-written bytes against `quant_w_128x128` every step
(debug only, ~+15 ms/step).

## Known issues

- `PRIMUS_TURBO_WQB_SPLITK=4`: grad norm NaN from iteration 1 (g90 split-K wiring). Keep off.
- `PRIMUS_TURBO_MOE_GG_DSV4=1`: 2% faster than flygg at kernel level, not worth enabling.
- MegaMoE `fp8_block` runs the EP=1 stages only; EP>1 dispatch/combine has no FP8-block path yet.
- `PRIMUS_FP8BLK_ADAM_MEGAMOE=1`: after the first step the MegaMoE FP8 expert caches are written only by the fused
  Adam. Anything else that changes the expert weights mid-run (checkpoint load after step 1, an optimizer fallback)
  would leave them stale. It is a no-op at EP>1, for DP-sharded weights and with `PRIMUS_TURBO_MOE_GG_DSV4=1`.
- Rare nondeterminism (WATCH): 1 run in ~30 on 2026-11-21 (int_aqm_off1) had a different iteration-1 loss
  (11.91723 vs 11.91726) and NaN grad norm on every iteration, 75.5 ms/step; an identical rerun was normal. An
  earlier g90-path run once had grad norm inf at iteration 12. Cause unknown; rerun if iteration-1 loss is off.
- The log line "fused kernel hardcodes unclamped SwiGLU" is stale for fp8_block: the kernel applies the default
  SiLU-SwiGLU with a +-10 clamp on gate and up.
