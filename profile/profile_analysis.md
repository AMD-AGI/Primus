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
RMSNorm(Attn) → MLASelfAttention(QKV proj) → MLASelfAttention(RoPE)
→ MLASelfAttention(FlashAttn) → MLASelfAttention(O proj) → RMSNorm(MoE)
→ MoELayer(TopKRouter) → MoELayer(dispatch) → [gap] → MoELayer(token_permute)
→ MoELayer(GroupedMLP FFN1×32) → MoELayer(GroupedMLP SwiGLU)
→ MoELayer(GroupedMLP FFN2×32) → MoELayer(token_unpermute) → MoELayer(combine)
```

## Per-Operator Statistics (execution order)

| # | Operator | Kernels | Wall (ms) | Kernel (ms) | % | Overlap |
|---|----------|---------|-----------|-------------|---|---------|
| 1 | RMSNorm(Attn) | 1 | 0.097 | 0.097 | 0.6% | - |
| 2 | **MLASelfAttention(QKV proj)** | 8 | **0.802** | **0.802** | **5.2%** | - |
| 3 | MLASelfAttention(RoPE) | 2 | 0.297 | 0.297 | 1.9% | - |
| 4 | **MLASelfAttention(FlashAttn)** | 1 | **1.278** | **1.278** | **8.3%** | - |
| 5 | MLASelfAttention(O proj) | 2 | 0.727 | 0.727 | 4.7% | - |
| 6 | RMSNorm(MoE) | 1 | 0.097 | 0.097 | 0.6% | - |
| 7 | **MoELayer(TopKRouter)** | 67 | **1.200** | **1.016** | **6.6%** | - |
| 8 | **MoELayer(dispatch)** | 3 | **0.810** | **0.785** | **5.1%** | - |
| 9 | MoELayer(token_permute) | 1 | 0.173 | 0.173 | 1.1% | - |
| 10 | **MoELayer(GroupedMLP FFN1×32)** | 32 | **3.203** | **6.302** | **41.1%** | **1.97×** |
| 11 | MoELayer(GroupedMLP SwiGLU) | 1 | 0.121 | 0.121 | 0.8% | - |
| 12 | **MoELayer(GroupedMLP FFN2×32)** | 32 | **1.366** | **2.692** | **17.5%** | **1.97×** |
| 13 | MoELayer(token_unpermute) | 1 | 0.183 | 0.183 | 1.2% | - |
| 14 | **MoELayer(combine)** | 2 | **0.770** | **0.770** | **5.0%** | - |
| | **Total** | **154** | **12.14** | **15.34** | **100%** | |

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
MoELayer(dispatch) → MoELayer(token_permute) → MoELayer(GroupedMLP dFFN2×64)
→ MoELayer(GroupedMLP SwiGLU BWD) → MoELayer(GroupedMLP dFFN1×64) → MoELayer(grad_acc)
→ MoELayer(token_unpermute) → [gap] → MoELayer(combine) → MoELayer(Router wgrad)
→ RMSNorm-BWD(MoE) → MLASelfAttention(O wgrad) → MLASelfAttention(FlashAttn BWD)
→ MLASelfAttention(RoPE BWD) → MLASelfAttention(QKV wgrad) → [gap] → RMSNorm-BWD(Attn)
```

## Per-Operator Statistics (execution order)

| # | Operator | Kernels | Wall (ms) | Kernel (ms) | % | Overlap |
|---|----------|---------|-----------|-------------|---|---------|
| 1 | MoELayer(dispatch) | 2 | 0.674 | 0.674 | 2.1% | - |
| 2 | MoELayer(token_permute) | 1 | 0.183 | 0.183 | 0.6% | - |
| 3 | **MoELayer(GroupedMLP dFFN2×64)** | 64 | **3.098** | **6.103** | **19.1%** | **1.97×** |
| 4 | MoELayer(GroupedMLP SwiGLU BWD) | 2 | 0.590 | 0.590 | 1.8% | - |
| 5 | **MoELayer(GroupedMLP dFFN1×64)** | 64 | **4.875** | **9.551** | **29.9%** | **1.96×** |
| 6 | **MoELayer(grad_acc)** | 4 | **2.647** | **2.647** | **8.3%** | - |
| 7 | MoELayer(token_unpermute) | 1 | 0.231 | 0.231 | 0.7% | - |
| 8 | MoELayer(combine) | 2 | 1.154 | 1.154 | 3.6% | - |
| 9 | MoELayer(Router wgrad) | 1 | 0.252 | 0.252 | 0.8% | - |
| 10 | RMSNorm-BWD(MoE) | 2 | 0.998 | 0.987 | 3.1% | - |
| 11 | MLASelfAttention(O wgrad) | 2 | 1.204 | 1.204 | 3.8% | - |
| 12 | **MLASelfAttention(FlashAttn BWD)** | 3 | **5.321** | **5.321** | **16.7%** | - |
| 13 | MLASelfAttention(RoPE BWD) | 2 | 0.312 | 0.312 | 1.0% | - |
| 14 | **MLASelfAttention(QKV wgrad)** | 12 | **1.756** | **1.756** | **5.5%** | - |
| 15 | RMSNorm-BWD(Attn) | 2 | 0.978 | 0.967 | 3.0% | - |
| | **Total** | **164** | **43.21** | **31.93** | **100%** | |

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
