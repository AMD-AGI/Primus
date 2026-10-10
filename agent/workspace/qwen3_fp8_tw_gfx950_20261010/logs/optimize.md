# Qwen3-30B-A3B FP8 tensorwise pretrain on 8x MI355X — Turbo kernel campaign

Baseline (prompt): 8894 ms/iter, 670.6 TFLOP/s/GPU, Turbo e2d9f1d7, Primus perf/qwen3-30b-a3b-tuning,
examples/megatron/configs/MI355X/qwen3_30B_A3B-FP8-pretrain.yaml. e2e: even routing, MBS 8, GBS 512, 20 iters.
All PR branches are based on Turbo 1103b2df (post-#455).

## Item 1 — tensorwise FP8 quant chain
- Branch perf/quantization/tensorwise-fp8-qwen3 (1194eaa0): flat no-pad kernel for rows < 4 KB.
- 262144x768 quant: 156.5 -> 95.3 us (-39%); 262144x1536 -6%; larger K unchanged (row kernel). Byte-identical.
- Tests: test_quantize_fp8_tensorwise_no_pad_rows; tensorwise GEMM / grouped GEMM FP8 shards ~1400 passed.
- Not done: dense fused quant+transpose, amax fused into permute.

## Item 2 — FlyDSL grouped GEMM tensorwise
- bench_grouped_gemm_qwen3.py (even / uniform / skew / hot). No kernel change committed.

## Item 3 — DeepEP
- Branch perf/deep_ep/dispatch-layout-per-token (11b6d998): get_dispatch_layout 340 -> 44-46 us, 384 calls/iter.
- num_cu sweep (even): 160 CUs best, dispatch 920 -> 612 us, combine 536 -> 484 us. Primus flag only.
- FP8 dispatch: not adopted.

## Item 4 — attention bwd
- TE CK is best overall. FlyDSL bwd main kernel +26% slower. TE bwd per call: main 2680 us, dk_dv_reduce 216,
  odo 152, fill 107, dq_shuffle 94. Turbo AITER fwd 2x slower (causal kernel without tile skipping).
- Recommendation: keep use_turbo_attention false.

## Item 5 — qk RMSNorm + RoPE head_dim 128
- Turbo perf/flydsl/qk-rmsnorm-rope-hd128 (a895fdea): fwd+bwd 1243.6 -> 369.9 us (3.36x).
- Primus perf/qwen3-qk-rmsnorm-rope-hd128 (fb63615f): patch accepts head_dim 128 and TE RMSNorm.
- gfx950 hazard: dwordx4 buffer_store with SGPR soffset + VALU write of data VGPRs is not padded by LLVM.

## Item 6 — permute / unpermute
- #455 TRITON default slower than HIP at every measured shape (Qwen3 even permute fwd 390 vs 255 us).
- TURBO kernel bounds rows by the routed-token count: routed tokens must come first. DeepEP guarantees it,
  arbitrary callers do not, so the global default stays TRITON.
- Branch perf/moe/permute-default-hip (893d22f4): DeepEPTokenDispatcher(permute_backend=TURBO).

## e2e attribution runs
| run | Turbo | flags | ms/iter |
|---|---|---|---|
| r0_base | installed e2d9f1d7 | fused_act_with_probs | 7835-7867 (iters 19-20) |
| rb_base | 1103b2df | fused act | 7925.4 |
| b1_quant | + quant | fused act | 7870.9 |
| b2_layout | + layout | fused act | 7774.2 |
| r1_turbo | turbo-opt (+ permute) | fused act | 7643.5 |
| r2_cu160 | turbo-opt | + num_cu 160 | 7353.4 |
| r3_qknorm | turbo-opt | + fused qk-norm/RoPE | 6987.1 (37,518 tok/s/GPU) |

## Round 2 — toward 40,000 tok/s/GPU (<= 6553.6 ms/iter)
- m1 MegaMoE mxfp8 / m2 bf16: OOM at MBS 8 (264 / 261 GB allocated when the worst-case dispatch pool
  allocation fails); MegaMoE pools are sized for the worst case and kept per layer for backward.
- s2 sync-free stage 2: 7287.8 ms (+300 ms vs R3): num_worst_tokens makes every post-dispatch step scan
  262144 receive rows instead of 32768 tokens. Keep stage 1.
- c192 (turbo_deepep_num_cu 192): NaN forward loss on rank 2 at iteration 5. DeepEP bug at high channel counts.
- R3 profile (rank 0, one step): kernels 6941 ms of 6987 ms; attention 1583, grouped GEMM 1268, hipBLASLt 1003,
  DeepEP 876, quant chain 739 (amax 253), permute/unpermute 346, SwiGLU 262.
- c256: 7028 ms (slower than 160 CUs).
- f1 turbo_fused_grouped_gemm True: 6820.2 ms (38,437 tok/s/GPU). f2 (+ turbo_grouped_gemm_without_padding)
  exits: needs moe_router_padding_for_quantization False, which sync-free stage 1 forces True.
- g1 turbo_fp8_permute True: 6442.4 ms, 925.8 TFLOP/s, 40,690 tok/s/GPU, 223.3 GB, loss@20 11.33404. Target met.
  Per call (32768 tokens, H 2048, 16 local experts, top-8, pad 16): fwd kernels 645 -> 167 us, bwd 687 -> 174 us.
  Turbo perf/moe/fp8-permute-tensorwise (82cf4373, on top of the permute branch + the padding fix).
- Pre-existing TURBO permute bug found by the new worst-token + pad dispatcher test: padding rows live after all
  max_num_dispatched_tokens rows of row_id_map, but the kernel read them right after the routed count, so with an
  unrouted tail (num_worst_tokens > 0) padding rows kept torch.empty garbage (bf16 path too). Fixed in
  fix/moe/permute-padding-rows (941b9525); regression test fails on main, passes with the fix.
- TURBO unpermute leaves rows past the routed count unwritten (harmless: DeepEP combine reads only received rows).
- Primus perf/megatron/turbo-fp8-permute: turbo_fp8_permute flag, grad dtype recomputed in combine_preprocess
  (no dispatcher state, safe under interleaved micro-batches). Re-validated as g2.

## Round 3 — PRs and one Primus PR
- Turbo PRs: #553 merged (padding fix); #554 permute default, #555 FP8 permute (stacked on #554), #556 rope
  head_dim 128, #557 dispatch layout, #558 quant, and fix/fp8-attn-strided-layout opened. All pass pre-commit.
- Primus: one branch perf/qwen3-30b-a3b-fp8-turbo on main 0a68e5cd (bef41e8a). The force-even router fix was
  already on main (#1243) and is dropped. Added unit tests for validate_turbo_fp8_permute and _fp8_permute_dtypes.
  The last commit turns every switch on in the Qwen3 FP8 config (env PRIMUS_FUSED_QK_RMSNORM_ROPE=1 included).
- e2e on that branch: pr1 (switches on the CLI) 6518.7 ms; cfg2 / cfg3 (config only) 6410.6 / 6459.6 ms
  (40,892 / 40,582 tok/s/GPU).
- cfg1 (config only) died at iteration 3: NaN forward loss on rank 4. Arguments identical to pr1. At 160 CUs,
  1 of 13 20-iteration runs hit it; c192 hit it too. Not isolated; next step is a DeepEP intranode
  dispatch/combine stress loop with exact checks at num_sms 80 / 160 / 192.
- All runs: e2e/results.md.
