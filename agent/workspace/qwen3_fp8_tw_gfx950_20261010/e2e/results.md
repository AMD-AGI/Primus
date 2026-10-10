# All end-to-end runs

Qwen3-30B-A3B FP8 tensorwise pretrain on 8x MI355X (EP8, MBS 8, GBS 512, seq 4096, even routing).
`parse_e2e.py` output: mean and standard deviation of iterations 11-20, tokens/s/GPU = 262144 / (s/iter),
max-rank ROCm memory. The raw logs were in the container's `/tmp/q3_e2e_<run>.log` on the old machine.
They are not kept because they contain internal host names.

`cu` is `turbo_deepep_num_cu`. "Fused act" is `use_turbo_fused_act_with_probs`.

| run | what | cu | ms/iter | sd | TFLOP/s/GPU | tokens/s/GPU | max mem GB | loss@20 |
|---|---|---|---|---|---|---|---|---|
| r0_base | installed Turbo e2d9f1d7, fused act | 80 | 7915.6 | 47.8 | 753.5 | 33,117 | 222.7 | 11.33435 |
| rb_base | Turbo 1103b2df, fused act | 80 | 7925.4 | 8.5 | 752.5 | 33,077 | 222.3 | 11.33427 |
| b1_quant | + flat tensorwise quant kernel | 80 | 7870.9 | 20.9 | 757.7 | 33,305 | 222.3 | 11.33415 |
| b2_layout | + per-token `get_dispatch_layout` | 80 | 7774.2 | 63.2 | 767.2 | 33,720 | 222.3 | 11.33431 |
| r1_turbo | + HIP permute in the dispatcher (turbo-opt) | 80 | 7643.5 | 12.3 | 780.3 | 34,296 | 222.2 | 11.33423 |
| r2_cu160 | + `turbo_deepep_num_cu 160` | 160 | 7353.4 | 14.5 | 811.1 | 35,649 | 222.6 | 11.33456 |
| r3_qknorm | + fused qk RMSNorm + RoPE, head_dim 128 | 160 | 6987.1 | 22.0 | 853.6 | 37,518 | 224.0 | 11.33373 |
| s2_syncfree2 | r3 + sync-free MoE stage 2 | 160 | 7287.8 | 43.6 | 818.4 | 35,970 | 227.6 | 11.33385 |
| c192 | r3 with 192 CUs | 192 | NaN forward loss, rank 2, iteration 5 | | | | | |
| c256 | r3 with 256 CUs | 256 | 7028.3 | 41.8 | 848.6 | 37,299 | 224.5 | 11.33399 |
| p3_profile | r3 under the profiler (timings not representative) | 160 | 28309.8 | 21272.2 | 483.9 | 9,260 | 224.0 | 11.44972 |
| m1_mega_mxfp8 | MegaMoE mxfp8 | 160 | OOM at MBS 8 | | | | | |
| m2_mega_bf16 | MegaMoE bf16 | 160 | OOM at MBS 8 | | | | | |
| f1_fused_mlp | r3 + `turbo_fused_grouped_gemm` | 160 | 6820.2 | 34.9 | 874.5 | 38,437 | 224.5 | 11.33412 |
| f1b_fused_mlp | repeat of f1 | 160 | 6815.1 | 28.0 | 875.2 | 38,465 | 224.5 | 11.33421 |
| f2_fused_mlp_nopad | f1 + `turbo_grouped_gemm_without_padding` | 160 | argument validation exit (needs padding off; sync-free stage 1 forces it on) | | | | | |
| g1_fp8perm | f1 + `turbo_fp8_permute` (pre-padding-fix build, stateful Primus) | 160 | 6442.4 | 17.2 | 925.8 | 40,690 | 223.3 | 11.33404 |
| g1b_fp8perm_stateful | g1's Primus code, rebuilt turbo-opt (padding fix) | 160 | 6515.0 | 95.3 | 915.6 | 40,237 | 223.3 | 11.33411 |
| g2_fp8perm_stateless | stateless Primus dispatcher | 160 | 6549.1 | 136.4 | 911.1 | 40,028 | 223.3 | 11.33445 |
| g3_fp8perm_stateless | repeat of g2 | 160 | 6523.8 | 70.9 | 914.3 | 40,183 | 223.3 | 11.33420 |
| pr1 | Primus `perf/qwen3-30b-a3b-fp8-turbo`, switches on the command line | 160 | 6518.7 | 86.1 | 915.1 | 40,214 | 223.3 | 11.33415 |
| cfg1 | Primus `perf/qwen3-30b-a3b-fp8-turbo`, config only | 160 | NaN forward loss, rank 4, iteration 3 | | | | | |
| cfg2 | repeat of cfg1 | 160 | 6410.6 | 20.5 | 930.4 | 40,892 | 223.3 | 11.33433 |
| cfg3 | repeat of cfg1 | 160 | 6459.6 | 45.4 | 923.3 | 40,582 | 223.3 | 11.33445 |

The cfg1 and pr1 runs parsed the same 847 arguments, and their wgrad autotune winners match.
