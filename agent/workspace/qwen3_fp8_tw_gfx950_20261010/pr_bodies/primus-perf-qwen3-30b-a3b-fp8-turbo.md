# Description

Qwen3-30B-A3B FP8 pretraining on 8× MI355X goes from a memory fault at iteration 3 to **40,283 tokens/s/GPU** (6507.6 ms/iter), 47% above the BF16 config. This PR switches the FP8 example config to Primus-Turbo tensorwise FP8 and turns on every Turbo fusion that got there. Two of those fusions are new in this PR:
- **head_dim 128 for the fused QKV split + Q/K RMSNorm + RoPE patch.** The patch was GPT-OSS-only before.
- **`turbo_fp8_permute`**: the DeepEP permute hands the fused grouped MLP FP8 tokens, and takes its gradient back in FP8.

The PR also fixes `enable_turbo_attention_float8`, which failed on its first call.

**Primus-Turbo requirements.**
- `turbo_fp8_permute` needs a Primus-Turbo whose `moe_permute` takes `quantize_dtype` ([AMD-AGI/Primus-Turbo#555](https://github.com/AMD-AGI/Primus-Turbo/pull/555)). With an older Turbo, the first dispatch fails.
- The head_dim 128 fusion needs [AMD-AGI/Primus-Turbo#556](https://github.com/AMD-AGI/Primus-Turbo/pull/556). Without it, the patch keeps the unfused path.
- The other Turbo rows in the table below come from [#554](https://github.com/AMD-AGI/Primus-Turbo/pull/554), [#557](https://github.com/AMD-AGI/Primus-Turbo/pull/557) and [#558](https://github.com/AMD-AGI/Primus-Turbo/pull/558). They need no Primus change.

**End-to-end.** Qwen3-30B-A3B FP8 pretrain on 8× MI355X (EP8, MBS 8, GBS 512, seq 4096, even routing), mean of iterations 11–20 of 20-iteration runs. Each row adds one change on top of the row above:

| run | from | ms/iter | TFLOP/s/GPU | tokens/s/GPU | mem | loss@20 |
|---|---|---|---|---|---|---|
| BF16 config (legacy grouped GEMM) | | 9587 | 622.1 | 27,344 | | |
| FP8 config before this PR | | memory fault at iteration 3 | | | | |
| FP8 tensorwise config (Turbo e2d9f1d7) | this PR | 8894.0 | 670.6 | 29,474 | 277 GB | 11.33393 |
| + `use_turbo_fused_act_with_probs` | this PR | 7915.6 | 753.5 | 33,117 | 223 GB | 11.33435 |
| Turbo 1103b2df | Turbo main | 7925.4 | 752.5 | 33,077 | 222 GB | 11.33427 |
| + flat tensorwise FP8 quant kernel | Turbo #558 | 7870.9 | 757.7 | 33,305 | 222 GB | 11.33415 |
| + per-token `get_dispatch_layout` | Turbo #557 | 7774.2 | 767.2 | 33,720 | 222 GB | 11.33431 |
| + HIP permute in `DeepEPTokenDispatcher` | Turbo #554 | 7643.5 | 780.3 | 34,296 | 222 GB | 11.33423 |
| + `turbo_deepep_num_cu` 80 → 160 | this PR | 7353.4 | 811.1 | 35,649 | 223 GB | 11.33456 |
| + fused QKV split + Q/K RMSNorm + RoPE, head_dim 128 | this PR + Turbo #556 | 6987.1 | 853.6 | 37,518 | 224 GB | 11.33373 |
| + `turbo_fused_grouped_gemm` (mean of 2 runs) | this PR | 6817.7 | 874.9 | 38,451 | 225 GB | 11.33412 |
| + `turbo_fp8_permute` (mean of 4 runs) | this PR + Turbo #555 | 6507.6 | 916.7 | 40,283 | 223 GB | 11.33404 |

From the FP8 tensorwise config to the last row: −26.8% ms/iter, +36.7% tokens/s/GPU. Loss differences are within run-to-run noise (about 4e-4 at iteration 20). This branch, rebased on main 0a68e5cd, with the switches on:

| run | ms/iter | tokens/s/GPU | loss@20 |
|---|---|---|---|
| switches on the command line | 6518.7 | 40,214 | 11.33415 |
| this config alone | 6410.6 | 40,892 | 11.33433 |
| this config alone | 6459.6 | 40,582 | 11.33445 |

**Known issue: intermittent NaN.**
- One more config-only run of this branch hit a NaN forward loss on rank 4 at iteration 3. Its parsed arguments match a passing run exactly.
- Across this work, 1 of 13 20-iteration runs at `turbo_deepep_num_cu 160` hit this, as did the only run at 192 (rank 2, iteration 5).
- None of the 5 runs at 80 CUs did. Those runs also predate the fused paths, though, so the cause is not isolated yet.

## Changes

- `fix(turbo): make enable_turbo_attention_float8 runnable again`
  - `primus/backends/megatron/core/extensions/primus_turbo.py`:
    - Pass `sink=` only when a sink tensor exists; `flash_attn_fp8_func` has no `sink` parameter.
    - Make q/k/v bshd-contiguous on the FP8 path. `flash_attn_fp8_func` permutes sbhd storage as if it were the logical layout (fixed in Turbo by `fix/fp8-attn-strided-layout`).
- `perf(qwen3): tune Qwen3-30B-A3B FP8 MI355X config to Turbo tensorwise`
  - `examples/megatron/configs/MI355X/qwen3_30B_A3B-FP8-pretrain.yaml`: `fp8_recipe: tensorwise`, `use_turbo_gemm` / `use_turbo_grouped_gemm`, and the TE grouped-MLP path instead of the legacy grouped GEMM.
- `perf(megatron): enable the fused packed-QKV RMSNorm + RoPE patch for head_dim 128`
  - `primus/backends/megatron/patches/turbo/qk_rmsnorm_rope_patches.py`:
    - Accept any head_dim in Turbo's `QK_RMSNORM_ROPE_HEAD_DIMS`, falling back to `(64,)` on older Turbo.
    - Accept TE RMSNorm as well as `PrimusTurboRMSNorm`. The fused path only reads weight / eps; `zero_centered_gamma` is still rejected.
  - Fused QKV split + Q/K RMSNorm + RoPE at the Qwen3 attention shape: fwd+bwd 1243.6 → 369.9 µs (3.36×).
- `perf(megatron): hand the Turbo fused grouped MLP FP8 tokens from the DeepEP permute`
  - New opt-in flag `turbo_fp8_permute` (default false in `primus_turbo.yaml`).
  - Under Turbo FP8 tensorwise current scaling, `PrimusTurboDeepEPTokenDispatcher` passes `quantize_dtype` to `_post_dispatch` and `grad_quantize_dtype` to `_pre_combine`. The dtypes follow the FP8 format (HYBRID: e4m3 input, e5m2 gradient), and results are bit-identical to the default path. Any other recipe, or a layer with Turbo FP8 off, keeps the bf16 path.
  - Argument validation requires `enable_primus_turbo`, `use_turbo_deepep` and `turbo_fused_grouped_gemm`, and rejects selective recompute of `moe_act`.
  - Dispatcher kernel time per call at the Qwen3 EP8 shape: forward 645 → 167 µs, backward 687 → 174 µs.
- `perf(qwen3): turn on the Turbo fusions in the Qwen3-30B-A3B FP8 MI355X config`
  - Top-level `env: PRIMUS_FUSED_QK_RMSNORM_ROPE: "1"`.
  - `turbo_fused_grouped_gemm`, `use_turbo_fused_act_with_probs` and `turbo_fp8_permute` set to true.
  - `turbo_deepep_num_cu` 80 → 160. 192 gave a NaN loss on one rank at iteration 5; 256 is slower (7028 ms).

The force-"even" routing fix that this work also needed is already on main (#1243), so it is not part of this PR.

## Tests

- New in `tests/unit_tests/backends/megatron/test_rocm_arg_validation.py`: `validate_turbo_fp8_permute` (each required flag, selective recompute of `moe_act`, disabled no-op).
- New `tests/unit_tests/backends/megatron/test_turbo_fp8_permute_dtypes.py`: `_fp8_permute_dtypes` returns (e4m3, e5m2) for HYBRID and (e4m3, e4m3) for E4M3 tensorwise, and (None, None) with the flag off, Turbo FP8 off or blockwise scaling.
- `test_qk_rmsnorm_rope_patches.py`, `test_rocm_arg_validation.py`, `test_validate_args_patches.py`, `test_router_force_even_routing.py`, `test_turbo_fp8_permute_dtypes.py`: all pass on MI355X.
- End-to-end runs above.
