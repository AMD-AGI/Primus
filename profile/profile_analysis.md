# Profile Analysis

| Item | Value |
|------|-------|
| Trace | `primus-megatron-exp[dsv3-pretrain-mbs_2-gbs_512-PP_4-EP_8-VPP_1-turbodeepep_True-legacygg_True-profile_True-recompute_num_layers_0]-rank[0].1773739477052422662.pt.trace.json` |
| Step | ProfilerStep#6 (7628.61 ms) |
| MBS | 2 |
| GBS | 512 |
| PP | 4 |
| EP | 8 |
| VPP | 1 |
| TurboDeepEP | True |
| LegacyGG | True |
| Profile | True |
| Recompute layers | 0 |

---

# Report A: Phase Summary

| Phase | Time (ms) | % |
|-------|-----------|---|
| Forward | 3154.06 | 36.3% |
| Backward | 3604.53 | 41.5% |
| Other | 1930.78 | 22.2% |
| **Total** | **8689.36** | **100%** |

- Forward microbatches: 32
- Backward microbatches: 32
- Total GPU kernel time: 8689.36 ms
- Step wall time (ProfilerStep): 7628.61 ms
- GPU utilization (kernel / wall): 113.9%

---

# Report B: Forward Layer Analysis

```
TransformerLayer
├── _forward_attention
│   ├── #1   RMSNorm(Attn)                                97 us
│   └── MLASelfAttention
│       ├── #2   MLASelfAttention(QKV proj)                  802 us   8 kernels
│       ├── #3   MLASelfAttention(RoPE)                      297 us   2 kernels
│       ├── #4   MLASelfAttention(FlashAttn)                1278 us   ★
│       └── #5   MLASelfAttention(O proj)                    727 us   2 kernels
└── _forward_mlp
    ├── #6   RMSNorm(MoE)                                 97 us
    └── MoELayer
        ├── #7   MoELayer(TopKRouter)                       1016 us   67 kernels
        ├── #8   MoELayer(dispatch)                          785 us   3 kernels
        │           [gap 756us — cross-node all-to-all]
        ├── #9   MoELayer(token_permute)                     173 us
        ├── GroupedMLP
        │      ├── #10  MoELayer(GroupedMLP FFN1×32)               6302 us   ★  1.97× overlap  32 kernels
        │      ├── #11  MoELayer(GroupedMLP SwiGLU)                 121 us
        │      └── #12  MoELayer(GroupedMLP FFN2×32)               2692 us   ★  1.97× overlap  32 kernels
        ├── #13  MoELayer(token_unpermute)                   183 us
        └── #14  MoELayer(combine)                           770 us   2 kernels
```

## Per-Operator Statistics (execution order)

| # | Operator | Kernels | Wall (us) | Kernel (us) | % | Overlap | Top Kernel |
|---|----------|---------|-----------|-------------|---|---------|------------|
| 1 | RMSNorm(Attn) | 1 | 97 | 97 | 0.6% | - | `void transformer_engine::normalization::rmsnorm_fwd_gen` |
| 2 | **MLASelfAttention(QKV proj)** | 8 | **802** | **802** | **5.2%** | - | `Cijk_Alik_Bljk_F8BS_BH_Bias_HA_S_SAB_SAV_UserArgs_MT256` |
| 3 | MLASelfAttention(RoPE) | 2 | 297 | 297 | 1.9% | - | `rotary_fwd_kv_kernel.kd` |
| 4 | **MLASelfAttention(FlashAttn)** | 1 | **1278** | **1278** | **8.3%** | - | `_ZN5aiter32fmha_fwd_hd192_hd128_bf16_causalE.kd` |
| 5 | MLASelfAttention(O proj) | 2 | 727 | 727 | 4.7% | - | `Cijk_Alik_Bljk_F8BS_BH_Bias_HA_S_SAB_SAV_UserArgs_MT256` |
| 6 | RMSNorm(MoE) | 1 | 97 | 97 | 0.6% | - | `void transformer_engine::normalization::rmsnorm_fwd_gen` |
| 7 | **MoELayer(TopKRouter)** | 67 | **1200** | **1016** | **6.6%** | - | `void at::native::sbtopk::gatherTopK<float, unsigned int` |
| 8 | **MoELayer(dispatch)** | 3 | **810** | **785** | **5.1%** | - | `void primus_turbo::deep_ep::intranode::dispatch<8, 1024` |
| 9 | MoELayer(token_permute) | 1 | 173 | 173 | 1.1% | - | `_permute_kernel.kd` |
| 10 | **MoELayer(GroupedMLP FFN1×32)** | 32 | **3203** | **6302** | **41.1%** | **1.97×** | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x` |
| 11 | MoELayer(GroupedMLP SwiGLU) | 1 | 121 | 121 | 0.8% | - | `triton_poi_fused__to_copy_mul_silu_split_0.kd` |
| 12 | **MoELayer(GroupedMLP FFN2×32)** | 32 | **1366** | **2692** | **17.5%** | **1.97×** | `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x` |
| 13 | MoELayer(token_unpermute) | 1 | 183 | 183 | 1.2% | - | `_unpermute_kernel.kd` |
| 14 | **MoELayer(combine)** | 2 | **770** | **770** | **5.0%** | - | `void primus_turbo::deep_ep::intranode::combine<hip_bflo` |
| | **Total** | **154** | **12139** | **15338** | **100%** | | |

## Overlap Analysis

Expert GEMM shows multi-stream kernel overlap:

- **MoELayer(GroupedMLP FFN1×32)**: kernel sum = 6.30ms, wall = 3.20ms → **1.97× overlap**
- **MoELayer(GroupedMLP FFN2×32)**: kernel sum = 2.69ms, wall = 1.37ms → **1.97× overlap**

## Idle Gaps

| Between | Gap (ms) | Cause |
|---------|----------|-------|
| MoELayer(dispatch) → MoELayer(token_permute) | 0.76 | Kernel launch / scheduling gap |

## Key Takeaways

1. **MoELayer(GroupedMLP FFN1×32) 占 41.1%** (6.30 ms)
2. **MoELayer(GroupedMLP FFN2×32) 占 17.5%** (2.69 ms)
3. **MLASelfAttention(FlashAttn) 占 8.3%** (1.28 ms)
   - 已用 Flash Attention，优化空间有限

4. **Expert GEMM BF16 合计占 58.6%** — 全层最大瓶颈，FP8 升级预估整层提速 30-40%

5. **FP8 路径仅用于 attention 投影**: 10.0% — 相比 Expert GEMM，FP8 覆盖率低


---

# Report C: Backward Layer Analysis

```
TransformerLayer BWD
├── _forward_mlp BWD
│   └── MoELayer
│       ├── #1   MoELayer(dispatch)                          674 us   2 kernels
│       ├── #2   MoELayer(token_permute)                     183 us
│       ├── GroupedMLP
│       │      ├── #3   MoELayer(GroupedMLP dFFN2×64)              6103 us   ★  1.97× overlap  64 kernels
│       │      ├── #4   MoELayer(GroupedMLP SwiGLU BWD)             590 us   2 kernels
│       │      └── #5   MoELayer(GroupedMLP dFFN1×64)              9551 us   ★  1.96× overlap  64 kernels
│       ├── #6   MoELayer(grad_acc)                         2647 us   4 kernels
│       ├── #7   MoELayer(token_unpermute)                   231 us
│       │           [gap 790us — scheduling gap]
│       ├── #8   MoELayer(combine)                          1154 us   2 kernels
│       ├── #9   MoELayer(Router wgrad)                      252 us
│       └── #10  RMSNorm-BWD(MoE)                            987 us   2 kernels
└── _forward_attention BWD
    ├── MLASelfAttention
    │   ├── #11  MLASelfAttention(O wgrad)                  1204 us   2 kernels
    │   ├── #12  MLASelfAttention(FlashAttn BWD)            5321 us   ★  3 kernels
    │   ├── #13  MLASelfAttention(RoPE BWD)                  312 us   2 kernels
    │   └── #14  MLASelfAttention(QKV wgrad)                1756 us   12 kernels
    │   │        [gap 1981us — scheduling gap]
    └── #15  RMSNorm-BWD(Attn)                           967 us   2 kernels
```

## Per-Operator Statistics (execution order)

| # | Operator | Kernels | Wall (us) | Kernel (us) | % | Overlap | Top Kernel |
|---|----------|---------|-----------|-------------|---|---------|------------|
| 1 | MoELayer(dispatch) | 2 | 674 | 674 | 2.1% | - | `void primus_turbo::deep_ep::intranode::dispatch<8, 1024` |
| 2 | MoELayer(token_permute) | 1 | 183 | 183 | 0.6% | - | `_permute_kernel.kd` |
| 3 | **MoELayer(GroupedMLP dFFN2×64)** | 64 | **3098** | **6103** | **19.1%** | **1.97×** | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x` |
| 4 | MoELayer(GroupedMLP SwiGLU BWD) | 2 | 590 | 590 | 1.8% | - | `triton_poi_fused__to_copy_cat_mul_silu_silu_backward_sp` |
| 5 | **MoELayer(GroupedMLP dFFN1×64)** | 64 | **4875** | **9551** | **29.9%** | **1.96×** | `Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x` |
| 6 | **MoELayer(grad_acc)** | 4 | **2647** | **2647** | **8.3%** | - | `void at::native::vectorized_templated_elementwise_kerne` |
| 7 | MoELayer(token_unpermute) | 1 | 231 | 231 | 0.7% | - | `_unpermute_kernel.kd` |
| 8 | MoELayer(combine) | 2 | 1154 | 1154 | 3.6% | - | `void primus_turbo::deep_ep::intranode::combine<hip_bflo` |
| 9 | MoELayer(Router wgrad) | 1 | 252 | 252 | 0.8% | - | `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT256x128x16_` |
| 10 | RMSNorm-BWD(MoE) | 2 | 998 | 987 | 3.1% | - | `void transformer_engine::normalization::rmsnorm_bwd_gen` |
| 11 | MLASelfAttention(O wgrad) | 2 | 1204 | 1204 | 3.8% | - | `Cijk_Alik_Bljk_F8B8BS_BH_Bias_HA_S_SAB_SAV_UserArgs_MT2` |
| 12 | **MLASelfAttention(FlashAttn BWD)** | 3 | **5321** | **5321** | **16.7%** | - | `_ZN5aiter41fmha_bwd_hd192_hd128_bf16_causal_a16_psskE.k` |
| 13 | MLASelfAttention(RoPE BWD) | 2 | 312 | 312 | 1.0% | - | `rotary_bwd_kv_kernel.kd` |
| 14 | **MLASelfAttention(QKV wgrad)** | 12 | **1756** | **1756** | **5.5%** | - | `Cijk_Alik_Bljk_F8B8BS_BH_Bias_HA_S_SAB_SAV_UserArgs_MT2` |
| 15 | RMSNorm-BWD(Attn) | 2 | 978 | 967 | 3.0% | - | `void transformer_engine::normalization::rmsnorm_bwd_gen` |
| | **Total** | **164** | **43207** | **31934** | **100%** | | |

## Overlap Analysis

Expert GEMM shows multi-stream kernel overlap:

- **MoELayer(GroupedMLP dFFN2×64)**: kernel sum = 6.10ms, wall = 3.10ms → **1.97× overlap**
- **MoELayer(GroupedMLP dFFN1×64)**: kernel sum = 9.55ms, wall = 4.87ms → **1.96× overlap**

## Idle Gaps

| Between | Gap (ms) | Cause |
|---------|----------|-------|
| MoELayer(token_unpermute) → MoELayer(combine) | 0.79 | Kernel launch / scheduling gap |
| MoELayer(combine) → MoELayer(Router wgrad) | 0.25 | EP combine return latency |
| MoELayer(Router wgrad) → RMSNorm-BWD(MoE) | 0.44 | Kernel launch / scheduling gap |
| RMSNorm-BWD(MoE) → MLASelfAttention(O wgrad) | 0.17 | Kernel launch / scheduling gap |
| MLASelfAttention(O wgrad) → MLASelfAttention(FlashAttn BWD) | 0.36 | Kernel launch / scheduling gap |
| MLASelfAttention(RoPE BWD) → MLASelfAttention(QKV wgrad) | 0.23 | Kernel launch / scheduling gap |
| MLASelfAttention(QKV wgrad) → RMSNorm-BWD(Attn) | 1.98 | Kernel launch / scheduling gap |

## Key Takeaways

1. **MoELayer(GroupedMLP dFFN1×64) 占 29.9%** (9.55 ms)
2. **MoELayer(GroupedMLP dFFN2×64) 占 19.1%** (6.10 ms)
3. **MLASelfAttention(FlashAttn BWD) 占 16.7%** (5.32 ms)
   - 已用 Flash Attention，优化空间有限

4. **Expert GEMM BF16 合计占 49.0%** — 全层最大瓶颈，FP8 升级预估整层提速 30-40%

5. **FP8 路径仅用于 attention 投影**: 9.3% — 相比 Expert GEMM，FP8 覆盖率低

## FWD vs BWD Comparison

| Metric | FWD | BWD | BWD/FWD |
|--------|-----|-----|--------|
| Kernel time | 15.34 ms | 31.93 ms | 2.08× |
| Expert GEMM | 8.99 ms | 15.65 ms | 1.74× |
| Attention | 1.28 ms | 5.32 ms | 4.16× |
| DeepEP | 1.55 ms | 1.83 ms | 1.18× |



---

# Bottleneck Analysis & Optimization Recommendations

| # | Check | Value | Status | Recommendation |
|---|-------|-------|--------|----------------|
| 1 | GEMM-BF16 dominance | 38.2% | 🟡 MEDIUM | `moe_use_legacy_grouped_gemm=False` → FP8 |
| 2 | NCCL in Other phase | 19.8% | 🔴 HIGH | Enable `overlap_p2p_comm_warmup_flush` |
| 3 | DeepEP overhead | 6.3% | 🟡 CHECK | `turbo_sync_free_moe_stage >= 2` |
| 4 | Elementwise ops | 7.2% | 🔴 HIGH | Check `bias_swiglu_fusion`, `moe_permute_fusion` |
| 5 | Expert GEMM overlap | 1.97× | 🟢 OK | 2.0× multi-stream parallel |


## Full Category Time Budget

| Category | Total (ms) | % of Step |
|----------|-----------|-----------|
| GEMM-BF16 | 3317.95 | 38.2% |
| NCCL | 1817.10 | 20.9% |
| Attn-BWD | 681.51 | 7.8% |
| Elementwise | 628.64 | 7.2% |
| GEMM-FP8 | 577.43 | 6.6% |
| DeepEP | 549.29 | 6.3% |
| RMSNorm-BWD | 283.75 | 3.3% |
| Attn-FWD | 162.23 | 1.9% |
| Triton-Fused | 122.87 | 1.4% |
| FP8-CastTranspose | 113.12 | 1.3% |
| MoE-TopK | 69.47 | 0.8% |
| Optimizer | 56.26 | 0.6% |
| Other | 54.11 | 0.6% |
| MoE-Unpermute | 53.67 | 0.6% |
| MoE-Permute | 44.93 | 0.5% |
| RoPE-BWD | 39.48 | 0.5% |
| RoPE-FWD | 37.11 | 0.4% |
| RMSNorm-FWD | 34.35 | 0.4% |
| MemSet | 25.75 | 0.3% |
| MemCopy | 20.35 | 0.2% |
| **Total** | **8689.36** | **100%** |
