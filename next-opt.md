# FLUX-Slim 下一阶段性能优化与 Profiling 计划

## 结论摘要

当前 `dev/zirui/flux-slim` 已经明显超过早期约 75% scaling efficiency 的版本。

| 版本 | 4-node 稳态吞吐 | Scaling efficiency |
|---|---:|---:|
| 早期 HSDP | 约 1379 samples/s | 约 75% |
| Native + natural backward，BF16 AllReduce | 约 1622 samples/s | 约 84% |
| 当前 Native + block-1024 FP8 AllReduce | 约 1664 samples/s | 约 86% |

当前正在运行的最新版 E2E：

```text
/shared_nfs/zirui/runs/flux_slim_native_fp8_allreduce_4n_gbs1024_e2e_20260822T101321Z
```

截至 2026-08-22 11:01 UTC，运行约到 step 4040：

```text
step time:       0.61–0.62 s
throughput/GPU:  51.9–52.1 samples/s
global steady:   1661–1667 samples/s
cumulative:      约 1614 samples/s（包含周期 validation）
```

任务尚未结束，因此最终 convergence step、TTQ 和 E2E throughput 仍待确认。

## 1. 当前 scaling efficiency

最新版单节点 natural-backward 稳态结果为：

```text
1 node:
step time ≈ 2.113 s
global throughput ≈ 484.5 samples/s
```

当前 4-node FP8 AllReduce 稳态结果为：

```text
4 nodes:
step time ≈ 0.615 s
global throughput ≈ 1664 samples/s
```

因此：

```text
steady scaling efficiency
= 1664 / (4 × 484.5)
≈ 85.9%
```

采用已经完成的 E2E 结果：

```text
1-node E2E throughput:          466.937 samples/s
4-node native BF16 throughput: 1575.280 samples/s

E2E scaling efficiency
= 1575.280 / (4 × 466.937)
≈ 84.3%
```

所以当前合理结论是：

> 最新 FLUX-Slim 的 4-node scaling efficiency 约为 **84%–86%**。早期的 75% 已不是当前版本水平，提升主要来自 HSDP、natural backward、native launcher 和 FP8 gradient AllReduce。

## 2. 距离目标还有多少

目标是：

```text
global throughput = 1976 samples/s
E2E = 70 min
```

在 GBS=1024 下：

```text
目标 step time = 1024 / 1976 = 0.5182 s
当前 step time ≈ 0.615 s
```

即需要：

```text
step time: 0.615 → 0.518 s
吞吐提升: 1976 / 1664 - 1 ≈ 18.75%
每 step 再减少约 97 ms
```

但当前单节点 step time 的四倍理想扩展为：

```text
2.113 / 4 = 0.528 s
```

目标 `0.518s` 甚至比当前单节点性能的完美线性扩展还快约 2%。因此，1976 samples/s 不能只靠继续优化多节点通信达到，必须同时做到：

1. 基本消除剩余多节点开销；
2. 单节点计算路径再提升至少约 2%，实际最好提升 5% 以上。

此外，`1976 samples/s` 与严格的 `70 min E2E` 并不完全一致。若在 step 8192 收敛：

```text
总 samples = 8192 × 1024 = 8,388,608
纯训练时间 = 8,388,608 / 1976 = 4245 s = 70.75 min
```

这还没有包含 validation。当前每 256 step 验证一次，单次约 5 秒，到 step 8192 累计约 160 秒。若严格要求 `run_start → run_stop <= 70 min`，并保留当前 validation 开销，则训练 step time 需要约 `0.493s`，对应 steady throughput 约 `2077 samples/s`。

因此后续实验必须明确区分：

- steady training throughput；
- MLPerf cumulative throughput；
- `run_start → run_stop` TTQ。

## 3. 当前瓶颈

### 3.1 剩余分布式开销约 87 ms/step

由单节点推算的理想四节点 step time 是：

```text
2.113 / 4 = 0.528 s
```

当前约为 `0.615s`，差值约 `87ms/step`。这部分包括：

- 跨节点 HSDP gradient-shard AllReduce；
- collective 尾部未被 backward 覆盖的部分；
- rank 间同步和 straggler；
- FP8 scale MAX collective；
- host-staged RDMA；
- FSDP hook 和 kernel launch 间隙。

FP8 AllReduce 已把 BF16 路径从约 `0.630s` 降到约 `0.615s`，但收益只有约 1.3%–2.8%，说明通信压缩已进入收益递减区间。

### 3.2 计算仍是最大总工作量

最近可用的 natural-backward 4-node trace 显示：

| 类别 | GPU work/step | 唯一 exposed 时间 |
|---|---:|---:|
| FP8 GEMM | 252.9 ms | 65.5 ms |
| RCCL | 487.0 ms | 21.6 ms |
| Other GPU kernels | 99.3 ms | 30.1 ms |
| Norm | 93.6 ms | 13.1 ms |
| Pointwise/layout | 72.9 ms | 31.5 ms |
| Attention | 34.9 ms | 4.6 ms |
| Optimizer/grad norm | 27.0 ms | 15.7 ms |
| FP8 cast/scale | 24.4 ms | 2.9 ms |

RCCL 的约 487ms work 大部分和 backward 重叠，不能直接与 step time 相加。约 21.6ms exposed 只统计 GPU 区间中 RCCL 独占的时间，可能低估依赖等待、CPU gap 和 stream synchronization。Strong-scaling 差值说明完整 distributed overhead 更接近约 87ms。下一版 profile 需要重点解释这两个数字之间的差异。

### 3.3 Validation 已是明显的 E2E 成本

目前每 256 steps 验证一次，单次约 `4.9–5.2s`。到 step 8192 共约 32 次，累计约 `155–165s`，占当前约 88 分钟 E2E 的约 3%。它不影响 steady throughput，但直接影响 70 分钟 TTQ 目标。

### 3.4 CUDA Graph 不是当前的直接解法

此前实验已经确认：

- Inductor `reduce-overhead` / `max-autotune` CUDA Graph 在 per-block 路径持续重录；
- FSDP collective capture 可触发 `hipIpcGetMemHandle` 失败、SIGSEGV 或 NaN；
- 普通 trainer-level graph 与 FSDP2 parameter materialization 的地址生命周期不兼容；
- MBS32 测得 host/launch gap 约 6.3%，即使全部消除，也不足以单独覆盖当前 18.75% 的目标差距。

因此 CUDA Graph 应作为后续 FSDP-aware 设计项，而不是下一轮 P0。

## 4. 重新规划 FLUX-Slim profile

当前 FP8 E2E 运行期间不启动 profiler，避免争用 GPU 或污染 E2E 结果。任务结束并确认四个节点上的旧容器、torchrun 进程和 GPU workload 全部退出后，再执行以下计划。

### Phase A：建立严格 matched baseline

每个候选先跑无 profiler 的 100–150 step，至少重复 3 次：

| Case | Gradient AllReduce | 用途 |
|---|---|---|
| A0 | BF16 | qualified native baseline |
| A1 | tensor-scale E4M3 | 当前最快 FP8 AR 参考 |
| A2 | block-1024 E4M3 | 当前数值候选 |
| A3 | GDR + BF16 | 隔离 GDR 收益 |
| A4 | GDR + 最佳 FP8 | 检查组合收益和边际递减 |

统一实验条件：

```text
native torchrun
4 nodes / 32 GPUs
MBS32 / GA1 / GBS1024
per-block compile
默认 compile mode
checkpoint ratio 0
固定代码 commit、镜像 digest 和 node order
MLPERF_ENABLE=false
SAVE_STRATEGY=none
```

记录：

- step time median、p90、p99；
- 32 ranks 的 step skew；
- global throughput；
- GPU peak memory；
- RCCL transport、NIC 和 GDR 状态。

### Phase B：采集 exact-current training trace

使用最新版 native + natural backward + 最佳 AllReduce 配置，而不是继续依赖旧 trace：

```text
wait:         30 steps
warmup:        2 steps
active:      3–5 steps
with_stack: false
profile rank: 0
```

当前 native 配置尚未暴露 `profile*` 字段，launcher 也未完整传递这些环境变量。应以最小改动将现有 trainer 已支持的 profiling 参数接入 `flux.1_schnell_t2i-native.yaml` 和 launcher，不新增 profiler abstraction。

Trace 分析必须输出：

1. ProfilerStep wall time；
2. GPU-active union；
3. CPU/launch idle gap；
4. RCCL work、overlap 和最后暴露 tail；
5. FP8 scale MAX 与 FP8 SUM 的独立耗时；
6. FP8 GEMM 按 shape、forward/dgrad/wgrad 分类；
7. pointwise/layout/norm top kernels；
8. optimizer、grad norm、zero-grad；
9. rank0 与其他节点代表 rank 的差异。

先 profile rank0；只有日志显示明显 rank skew 时，才分别采集 rank 8/16/24，避免一次产生四份大 trace。

### Phase C：单独 profile validation

训练 trace 无法解释累计约 160 秒的 validation 开销。需要单独 capture 一次完整 validation，测量：

- 29,696 个 eval samples 的实际 batch 数；
- DataLoader 和 host-to-device copy；
- validation forward；
- rank reduction/barrier；
- validation 是否存在可优化的非数值路径开销。

注意：MLPerf FLUX validation 必须保持 submission 语义。NVIDIA v6.0 NeMo submission 的 validation dataloader 使用同一个 `micro_batch_size`，并在预处理数据中为 29,696 个样本绑定 timestep；不能把扩大 eval batch 当作合规优化方向。

### Profile 验收规则

- profiler 前先完成无 profiler baseline，避免把 Kineto 开销当作模型回退；
- 每个性能候选至少 3 次短测，报告 median 和波动；
- 只有稳定提升至少 1% 且 numerical gate 通过，才进入 900-step；
- 只有 900-step validation 与 baseline 对齐，才进入完整 E2E；
- 完整 E2E 至少使用 matched seed，最终 qualification 使用 3 seeds。

## 5. 下一步优化优先级

### P0：GDRDMA matched 验证

当前 ABI-4 provider 和 8 张 Ionic NIC 已正常工作，但主要仍是 host staging。GDR 是唯一可能一次回收几十毫秒、同时不改变收敛语义的通信方向。

必须比较：

```text
host-staged BF16
GDR BF16
host-staged FP8
GDR FP8
```

验收条件：

- 日志明确显示 GDRDMA，不能只设置环境变量；
- 34/81MiB 代表尺寸和真实 8-group benchmark 均提升；
- 真实 training trace 的关键路径缩短；
- 100–150 step 吞吐稳定提升。

合理收益预期约 5%–10%，取决于当前约 87ms distributed overhead 中有多少来自 host staging。若节点持续报 `ibv_reg_mr_iova2 invalid argument`，不应继续盲调环境变量，应更换真正支持 GDR 的节点。

### P1：预编译/缓存后的 `max-autotune-no-cudagraphs`

历史实验中该模式有约 3%–6% 稳态收益，但 32 ranks 同时 autotune 曾产生严重 progress skew。正确路径是：

1. 在单 GPU/单节点覆盖全部固定 shape 完成 autotune；
2. 固定 PyTorch、Triton、FlyDSL 和镜像 digest；
3. 保存并验证 Inductor cache；
4. 让 32 ranks 复用只读或预热后的 cache；
5. 再做 4-node 100-step gate。

候选仅为：

```bash
TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
```

不要使用会启用 Inductor CUDAGraph 的 `reduce-overhead` 或 `max-autotune`。

### P1：针对 exposed support kernels 做融合

旧 trace 中非 GEMM exposed 时间较大：

```text
pointwise/layout  31.5 ms
other GPU         30.1 ms
norm              13.1 ms
optimizer         15.7 ms
```

优先分析和融合：

- RMSNorm backward + residual/add；
- `cat/split/chunk_cat`；
- GELU backward 与 cast/scale；
- backward contraction 周边 layout；
- grad norm + clipping；
- optimizer 前后的 BF16/FP32 copy。

已有 wholeloop17 结果仅预测 raw FP8 GEMM 节省约 15ms/step，实际 E2E 收益会更小。继续替换单个已接近最优的 GEMM 不足以达到目标，需要更完整的 backward/support region fusion。

### P2：Tensor-scale FP8 AllReduce 多 seed qualification

已有结果表明：

- tensor-scale E4M3 比 block-1024 更快；
- block-1024 未恢复 BF16 convergence step；
- 两者在既有 seed 10007 实验中均多跑一个 validation interval。

当前 native block-1024 E2E 结束后，应先确认它在 step 8192 还是 8448 收敛。若仍为 8448，不建议将 block-1024 设为默认。

随后用 2–3 个 matched seeds 比较 BF16 和 tensor-scale E4M3：

- 若 samples-to-quality 没有稳定 regression，选择更快的 tensor-scale；
- 若稳定多一个 interval，保持 BF16 默认，除非与 GDR 组合后 E2E 仍有净收益。

FP8 AllReduce 已被证明无法单独贡献 1.1x。

### P2：optimizer/grad-norm 路径

Profile 中 optimizer/grad norm 约有 15.7ms exposed。检查：

- DTensor norm 的 `full_tensor()` 是否产生额外同步；
- grad norm reduction 能否与最后一个 gradient bucket 合并；
- clipping 与 fused AdamW 之间的 BF16/FP32 copy；
- `zero_grad(set_to_none=True)` 对 storage 生命周期和 launch 的影响。

该方向预期为低个位数收益，但对 0.518s 目标仍有价值。

### P3：FSDP-aware CUDA Graph

只有满足以下条件后才重新投入：

- graph-stable all-gather/parameter buffers；
- persistent gradient buffers；
- FSDP/RCCL capture-aware lifecycle；
- RNG 和 FP8 state 正确前进；
- 新 profile 证明 host gap 仍值得优化。

不再尝试用普通 `torch.cuda.CUDAGraph` 直接包住现有 FSDP2 loop，也不再尝试 Inductor CUDAGraph Trees。

## 6. 推荐执行路线与最终判断

### 执行路线

```text
1. 等当前 native block-1024 FP8 AllReduce E2E 完成
   └─ 记录 convergence step、TTQ、最终 cumulative throughput

2. 补 exact-current 4-node profile
   ├─ 无 profiler三次 baseline
   ├─ rank0 training trace
   ├─ 跨节点 rank skew
   └─ validation trace

3. 在 GDR-capable 节点上做 matched benchmark

4. 建立共享 Inductor autotune cache
   └─ 重新评估 max-autotune-no-cudagraphs

5. 根据 trace 优化 support kernels / backward region
   └─ 每个候选先过 100-step，再过 900-step

6. 组合最佳项
   └─ GDR + 最佳 compile/cache + support fusion

7. 通过后做 3-seed MLPerf E2E qualification
```

### 最终判断

当前最主要的问题已经不是“RCCL 是否工作”，而是：

```text
约 87 ms/step 的剩余 distributed/synchronization overhead
+ exposed pointwise/layout/norm/optimizer 工作
+ 每轮约 160 秒 validation 开销
```

从当前约 1664 samples/s 达到 1976 samples/s，需要额外约 **18.75%**。单一方向无法覆盖：

- FP8 AllReduce：约 1%–3%；
- CUDA Graph 理论上限：约 6%，且当前路径不可用；
- selective GEMM：预计低个位数；
- GDR：可能约 5%–10%。

因此最现实的组合是：

> **GDR + cached max-autotune + targeted backward/support fusion。**

其中，GDR 验证和 exact-current profile 是判断 1976 samples/s 是否可达的两个关键步骤。若 GDR 无法显著缩短关键路径，则必须依赖新的 compute fusion；仅继续调整 RCCL 参数、FP8 block size 或打开 CUDA Graph，无法达到目标。


## 7. 计划执行结果（2026-08-22）

当前 native block-1024 FP8 AllReduce E2E 已完成：

```text
convergence step: 8448
step-8192 loss:   0.586151
final loss:       0.585667
TTQ:              5365.226 s / 89.420 min
E2E throughput:   1612.374 samples/s
compliance:       SUCCESS
```

相比 native BF16 baseline，吞吐提升 2.35%，但多运行一个 256-step validation interval，最终 TTQ 慢 0.75%。

重新规划的 profile 已落地。三次 matched 120-step 短测中位数为：

| AllReduce | Global throughput | 相对 BF16 |
|---|---:|---:|
| BF16 | 1623.39 samples/s | baseline |
| Tensor-scale E4M3 | 1663.89 samples/s | +2.50% |
| Block-1024 E4M3 | 1661.63 samples/s | +2.36% |

Exact-current block-1024 training trace 测得 steady profiler step 中位数约 `616.85ms`、GPU active 约 `97.2%`、host/launch idle 约 `16–17ms/step`。主要 uniquely-exposed 分类为 FP8 GEMM `73.21ms`、other GPU `42.15ms`、pointwise/layout `35.18ms`、optimizer/grad norm `16.81ms`、norm `12.45ms` 和 RCCL `12.14ms`。相对旧 BF16 trace，RCCL work 从约 `487.0ms` 降到 `420.7ms`，uniquely exposed RCCL 从 `21.6ms` 降到 `12.1ms`。

Validation profile 测得合规 batch-32 单次约 `5.068s`，长 profiler step 的 GPU active 为 `95.84%`。参考 NVIDIA v6.0 NeMo submission，validation 使用与训练一致的 `micro_batch_size`，并要求完整消费 29,696 个绑定 timestep 的验证样本；因此不把扩大 eval batch 作为优化方向。之前额外筛查的 eval batch 58/116 分别为 `5.283s/5.515s`，没有收益；64/96/128 因 partial-shape/recompile 增至 `22–30s`，且改变 tensorwise-FP8 数值行为。最终保留 batch 32，不保留 eval-batch 配置改动。

同节点 forced-GDR BF16 probe 在首个 backward HSDP AllReduce 失败，报 `ncclSystemError` 和 service-thread Connect failure；A4 不再执行。GDR 性能验证需要支持该路径的其他节点。

完整产物：

```text
/shared_nfs/zirui/runs/flux_slim_profile_20260822/summary.md
/shared_nfs/zirui/runs/flux_slim_profile_20260822/B_training_trace_fp8_block1024/
/shared_nfs/zirui/runs/flux_slim_profile_20260822/C_validation_trace_eval32/
```

据此更新优先级：

1. P0：针对 FP8 GEMM、natural TN wgrad 和 pointwise/layout/norm 的完整 backward/support fusion；
2. P0/P1：预生成并共享 `max-autotune-no-cudagraphs` cache，避免 32-rank 同时 autotune；
3. P1：在真正支持 GDR 的节点做 matched benchmark；
4. P2：tensor-scale E4M3 多 seed samples-to-quality qualification；
5. Reject：扩大 eval batch（不符合 MLPerf submission 语义，也无性能收益）；
6. Low priority：CUDA Graph。当前只有约 16–17ms host gap，且 FSDP graph 生命周期仍不正确。
