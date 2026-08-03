# Primus 与 NeMo FLUX 训练性能差距分析

## 1. 结论摘要

2026-07-29 完成 BF16 reference 对齐，2026-07-30 落地 selective checkpoint、no-reshard 和逐 block `torch.compile`，2026-07-31 完成三次独立 seed 的完整收敛验证并冻结 native PyTorch BF16 版本。统一为 1 node × 8 MI355X、GBS/MBS=`512/64`；NeMo 解析配置为 `fp8: null`。当前结果如下：

- Primus BF16 reference：`3.5045 s/step`，约 `146.1 samples/s`。
- NeMo BF16：`2.4249 s/step`，约 `211.1 samples/s`。
- Primus selective-checkpoint + no-reshard：`2.5271 s/step`，约 `202.7 samples/s`。
- **Primus 再开启 block compile：`1.6445 s/step`，约 `314.4 samples/s`，峰值显存 `206.61 GiB`。**
- compile 后 step time 比上一版低 `34.9%`、吞吐高 `55.1%`；当前 BF16 稳态吞吐为 NeMo 的 `148.9%`，原有性能 GAP 已反转。该结论仍是短测吞吐，不等同于 time-to-quality 对齐。
- 当前默认配置三次收敛全部通过：seed `10007/10008/10009` 分别在 step `13824/14336/13824` 达到 validation loss `<=0.586`；中位收敛 step 为 `13824`，中位 time-to-quality 为 `33742.86 s`（`9.37 h`）。

2026-08-01 又完成 NeMo FP8 `delayed_short` 与 Primus native dynamic FP8 的
三节点配对短测。统一为 500 steps、seed `10007`、GBS/MBS=`512/64`，排除前
100 steps，并关闭 validation、periodic checkpoint 和 W&B：

- NeMo `delayed_short` 中位数：`0.9743 s/step`、`525.5 global samples/s`。
- Primus 当前 P0（checkpoint ratio `0.25`）：`1.3620 s/step`、`375.9 global samples/s`，吞吐低 `28.48%`。
- Primus P1（checkpoint ratio `0`）：`1.3148 s/step`、`389.5 global samples/s`，吞吐仍低 `25.89%`。
- 关闭 Primus activation checkpoint 仅提升 `3.62%`，同时峰值显存从 `174.36 GiB` 增至 `214.05 GiB`；它只解释约 `9%` 的原始绝对 throughput gap。
- rank-0、5-step Torch profiler 显示 Primus 的主要 GPU 侧增量是 attention `+77.8 ms/step`、norm/pointwise `+56.6 ms/step` 和 BF16 GEMM/optimizer `+31.8 ms/step`；FP8 GEMM + cast/scale 反而比 NeMo 少 `38.3 ms/step` kernel work。
- 将 loss 和 gradient norm 的 `.item()` 延后到 optimizer 之后的日志点，P0 吞吐提升约 `2.5%` 至 `385.2 samples/s`，P1 提升约 `3.2%` 至 `404.4 samples/s`；峰值显存不变。
- `max-autotune-no-cudagraphs` 与 BF16 gradient reduction 组合在三节点达到 `408.0-410.3 samples/s`，中位 `409.1 samples/s`，比 deferred-sync control 高 `6.1%`，但仍比 NeMo 低 `22.1%`。
- 完整 seed `10007` convergence 中，优化后的 ratio `0.25` 在 step `14336` 达标，time-to-quality `26,841.915 s`；no-AC 在 step `14848` 达标，time-to-quality `26,396.741 s`。两者都生成完整 final checkpoint。
- 显式 `flash_attn_aiter` 将两节点稳态吞吐提高到 `444.6-457.2 samples/s`，平均 `450.9 samples/s`；三 seed median TTQ 为 `24,275.36 s`，比 SDPA candidate 快 `9.6%`。
- 六组真实 FLUX shape 的 FP8 raw-GEMM 筛选没有找到 `_scaled_mm` 替代：HipBLASLt、CK、Triton、FlyDSL 的 18-case 总延迟分别慢 `56.0%/1060.7%/35.2%/5.5%`；短测中的局部 FlyDSL 收益在 100-iteration 复测中消失。

因此 BF16 GAP 已反转，但 FP8 GAP 仍然存在且可稳定复现。compile/reduction
优化将 production steady step 压到 `1.2508 s`；AITER 进一步把稳态吞吐提高
到平均 `450.9 samples/s`，并在相同 median convergence step 下把三 seed
median TTQ 降低 `9.6%`。下一步应先重新 profile 这一最终 AITER 组合，再根据
新 trace 处理 norm/pointwise 或 host/GPU scheduling；通用 FP8 GEMM 替换已
停止，不再继续改变 FSDP topology。

这组对齐数据已经排除 FP8 GEMM 是当前 GAP 的必要解释。阶段 1 的主要差异是：

- **Primus reference**：全 block activation checkpoint、逐 block FSDP2 reshard、SDPA、普通 PyTorch BF16 Linear、FP32 gradient reduction。
- **NeMo BF16**：无 activation checkpoint、replicated DDP/no-shard、distributed optimizer + overlap、TE BF16 Linear、ROCm fused attention/RoPE、BF16 gradient reduction。

已确认的关键消融与组合：

1. 关闭 activation checkpoint：`2.7143 s/step`，step time 降 `22.5%`，但峰值显存升至 `259.6 GiB`；单项关闭了约 `73%` 的绝对 BF16 step-time GAP。
2. 关闭 FSDP reshard：`3.0614 s/step`，step time 降 `12.6%`，峰值显存仅 `60.0 GiB`；这是当前收益/显存风险最好的首个默认候选。
3. SDPA 改 AITER：`3.3614 s/step`，step time 仅降 `4.1%`；attention 值得保留，但不是第一瓶颈。
4. checkpoint ratio `0.25` + no-reshard：`2.5271 s/step`，step time 比 reference 降 `27.9%`；比 ratio `0.125` 少用 `32.53 GiB`，性能基本相同。
5. 在上述组合开启 block compile：80-step 稳态 `1.6445 s/step`，比 NeMo BF16 低 `32.2%`；峰值显存还降低 `11.13 GiB`。

`local_runs/run_flux_mlperf.sh` 已默认使用 checkpoint ratio `0.25`、`reshard_after_forward=false`、compile on。通用 YAML 和后端默认仍保持 compile off，避免影响其他 recipe。compile 首 step 约 `44.8 s`，但长跑可快速摊薄；当前 BF16 默认已完成 convergence 和重复性验收。

2026-07-31 又完成真实 FLUX shape 的 BF16 kernel 筛选。compiled PyTorch Linear 在全部六个主干 GEMM shape 上均快于 Primus-Turbo；TE Linear 不能进入当前 `fullgraph=True` block；Primus-Turbo RMSNorm 和 AITER RoPE 也都慢于 Inductor 基线。因此本轮没有增加 Linear/RMSNorm/RoPE backend 分支，FP8/MXFP8 保留为阶段 2。

冻结版本保持原生 `nn.Linear`、`nn.RMSNorm`、FSDP2 和 `torch.compile`，不引入 Transformer Engine module、FP8 state 或 checkpoint 格式。`local_runs/run_flux_mlperf.sh` 固定 MLPerf BF16 recipe 为 checkpoint ratio `0.25`、compile on、no-reshard；通用 YAML 默认不变。后续 FP8/MXFP8 也沿用 native PyTorch 模型结构，只在必要的 GEMM 算子层接入 ROCm kernel。

历史组合配置 `AITER + 无 checkpoint + compile` 曾达到 `1.44~1.65 s/step`，但其 `284 GiB` 峰值显存不适合作为默认方案。当前方案保留 25% checkpoint，稳态落在同一性能区间，同时把峰值控制在 `206.61 GiB`。

---

## 2. 对比范围与依据

### 2026-07-29 BF16 对齐短测

共同条件：1 node × 8 MI355X、global/micro batch=`512/64`、seed=`1234`、learning rate=`2e-4`、预处理 CC12M/COCO 数据。两边均为 BF16 训练；NeMo 启动参数虽设置 `FP8_RECIPE=default`，Hydra 最终配置明确为 `fp8: null`、`precision: bf16-mixed`。

- Primus：`crsuse2-m2m-005`，基于 `local_runs/run_flux_mlperf.sh`，120-step 短测。
- NeMo：`crsuse2-m2m-008`，基于 `/zirui/code/mlperf-training-6-0/flux1/nemo/run_with_docker_debug.sh`，镜像 `rocm/amd-mlperf:flux1_training_6.0`，配置 `MI355X_01x08x16_dev`，120-step 短测。
- 消融：AITER 在 `005`，no-checkpoint 在 `008`，no-reshard 在 `100`。

日志归档：

- NeMo：`/shared_nfs/zirui/runs/flux_bf16_gap_20260729/nemo_008/train_bf16_retry.log`
- Primus reference：`/shared_nfs/zirui/runs/flux_bf16_gap_20260729/primus_005/summary.txt`
- Primus AITER：`/shared_nfs/zirui/runs/flux_bf16_gap_20260729/aiter_only_005/summary.txt`
- Primus no-checkpoint：`/shared_nfs/zirui/runs/flux_bf16_gap_20260729/no_checkpoint_008/summary.txt`
- Primus no-reshard：`/shared_nfs/zirui/runs/flux_bf16_gap_20260729/no_reshard_100/summary.txt`

NeMo launcher 在该节点因 `/dev/fd` 不可用未走完整 wrapper，最终用相同镜像、挂载、环境变量和 `run_and_time.sh` 直接 Docker 启动；训练配置本身不变。当前数据用于稳态性能归因，不是 MLPerf submission result。

### 2026-07-30 Primus 优化短测

镜像统一为 `rocm/primus:v26.2`，其余训练参数与 BF16 reference 一致，运行 80 steps。日志根目录：

- `/shared_nfs/zirui/runs/flux_bf16_opt_20260730/`
- ratio `0.5`：`ratio050_005/summary.txt`
- ratio `0.25`：`ratio025_008/summary.txt`
- ratio `0.125`：`ratio0125_100/summary.txt`
- compile 失败记录：`ratio025_compile_008/`、`ratio025_compile2_008/`、`ratio050_compile_005/`

### 2026-07-30 compile 根因与修复验证

后续在 `rocm/primus:v26.3`、`crsuse2-m2m-008` 上补充了同容器对照：

- TorchTitan `flux_schnell_mlperf_preprocessed` 的真实 FLUX-Schnell + FSDP2 + block compile 成功完成 forward、backward 和 optimizer step，排除了 FSDP2 与 `torch.compile` 的通用不兼容。
- Primus `DoubleStreamBlock` 单 GPU compiled backward 成功；`SingleStreamBlock` 可最小复现 SDPA flash backward stride 断言。
- 将 Primus Q/K/V 和 RoPE 内部布局从 `B,H,L,D` 对齐为 TorchTitan 的 `B,L,H,D` 后，单 block 和完整 8-GPU FSDP2 均通过。
- 80-step 结果：`/shared_nfs/zirui/runs/flux_bf16_compile_inplace_20260730/compile_layout_bs64_ratio025_noreshard_80step_008/summary.txt`
- TorchTitan smoke：`/shared_nfs/zirui/runs/torchtitan_flux_compile_20260730/mlperf_preprocessed_2step/`

80-step 测试跳过前 5 step，统计 75 step；仅 step 9 出现一次 `3.77 s` 抖动，其余稳态主要约 `1.60~1.65 s`。首次编译 step 为 `44.76 s`，未计入稳态均值。

### 2026-07-31 三次完整收敛验证

使用 `rocm/primus:v26.3`，在三个独立节点从 `samples_count=0` 启动；共同配置为 BF16、MBS/GBS=`64/512`、checkpoint ratio `0.25`、no-reshard、block compile、每 512 step validation、每 100 step 保存 checkpoint。

| Seed | 收敛 step | Validation loss | Time-to-quality | 结果 |
|---:|---:|---:|---:|---|
| 10007 | 13824 | 0.585988 | 33742.86 s（9.37 h） | success |
| 10008 | 14336 | 0.586000 | 34365.53 s（9.55 h） | success |
| 10009 | 13824 | 0.585705 | 33144.15 s（9.21 h） | success |

通过率为 `3/3`；中位收敛 step 为 `13824`，step 极差为 `512`；中位 time-to-quality 为 `33742.86 s`，三次极差为 `1221.38 s`（20.36 分钟）。三次峰值显存均为 `206.61 GiB`，最终均生成 `checkpoint-final`，没有 OOM、Traceback 或进程异常。

结果目录：`/shared_nfs/zirui/runs/flux_bf16_compile_convergence_fresh_20260730/`。为避免权限依赖，运行时设置了 `MLPERF_CLEAR_CACHES=false`，因此这些结果用于收敛和工程重复性验证，不作为合规 submission result。当前还缺少同条件 NeMo BF16 完整 time-to-quality 数据，不能据此声称两者 TTT 已对齐。

### 2026-07-31 BF16 kernel 筛选

在 `crsuse2-m2m-006`、`rocm/primus:v26.2`（PyTorch 2.10、ROCm 7.2）上，使用 recipe 的真实 M/K/N、BF16、forward+backward 和 `torch.compile(fullgraph=True)` 比较 PyTorch、Transformer Engine、Primus-Turbo 与 AITER。代码依据为：

- NeMo：`nemo/src/custom_flux.py` 和 MI355X 配置中的 TE Linear、fused RoPE、QK RMSNorm；
- Primus-Turbo：commit `56c789e5` 的 autograd GEMM 和 Triton RMSNorm；
- AITER：commit `d6de77692` 的 tuned BF16 GEMM、trainable RoPE 和 fused QK norm/RoPE。

各候选均先过 eager backward 和 fullgraph 门禁，再记录 GPU event 中位时间。Linear 使用 6 次、RMSNorm 使用 10 次、RoPE 使用 20 次稳定迭代；首次编译和 AITER JIT 时间不计入。详细结果见 5.3、5.5 和 5.6。结论是所有候选均未达到 block `3%` 或端到端 `1.5%` 的接收门槛，因此没有修改模型实现，也无需为本轮重跑 convergence。

### 2026-08-01 FP8 三节点配对短测

三组实验分别在 `crsuse2-m2m-118/119/234` 顺序运行 NeMo N0、Primus P0
和 Primus P1。共同条件为 1 node x 8 MI355X、seed `10007`、MBS/GBS
`64/512`、500 steps、compile enabled、无 validation、无 periodic checkpoint、
无 W&B；统计窗口为 steps 101-500。

| Node | NeMo N0 samples/s | Primus P0 samples/s | P0 gap | Primus P1 no-AC samples/s | P1 gap |
|---|---:|---:|---:|---:|---:|
| `118` | `525.505` | `375.855` | `28.48%` | `388.607` | `26.05%` |
| `119` | `527.308` | `377.931` | `28.33%` | `392.126` | `25.64%` |
| `234` | `523.667` | `375.530` | `28.29%` | `389.465` | `25.63%` |
| **Median** | **`525.505`** | **`375.855`** | **`28.48%`** | **`389.465`** | **`25.89%`** |

NeMo 解析配置确认 `fp8_amax_history_len=4`、MBS/GBS `64/512`、stack
compile、`enable_checkpointing=false` 和 `recompute_granularity=None`。Primus
P0/P1 均为 `float8_recipe=tensorwise`、block compile、no-reshard；唯一消融是
checkpoint ratio `0.25 -> 0`。P1 峰值显存为 `214.05 GiB`，没有 OOM、NaN
或非有限 gradient，但尚未做 convergence qualification。

产物目录：

`/shared_nfs/zirui/runs/flux_fp8_nemo_throughput_20260801T043638Z/`

### 2026-08-01 FP8 profiler 与 deferred-sync 消融

NeMo 和 Primus 分别采集 rank 0 的 5 个 active Torch profiler steps。原始
`kernel` events 按类别聚合，避免 key-averages 同时包含 parent operator 和
child kernel 导致重复计时：

| Kernel group | Primus P1 | NeMo N0 | Delta |
|---|---:|---:|---:|
| FP8 GEMM + cast/scale | `604.0 ms/step` | `642.2 ms/step` | `-38.3 ms` |
| Attention | `152.9 ms/step` | `75.1 ms/step` | `+77.8 ms` |
| Norm + pointwise | `349.0 ms/step` | `292.4 ms/step` | `+56.6 ms` |
| BF16/other GEMM + optimizer | `57.1 ms/step` | `25.3 ms/step` | `+31.8 ms` |
| Communication | `265.0 ms/step` | `273.5 ms/step` | `-8.5 ms` |

kernel work 可以跨 stream overlap，不能直接相加为 step wall time；但该结果
排除了“主要慢在 TorchAO dynamic cast/scale”的假设。Primus attention 使用
SDPA，NeMo 使用 AITER/TE fused attention，是最大的单项 GPU 侧差异。

CPU trace 进一步发现 Primus 每 step 在 backward 前对 loss `.item()`，并在
optimizer 前对 gradient norm `.item()`，分别等待约 `250-308 ms` 和
`697-707 ms`。实现已改为保留 detached tensor，只在 logging step、optimizer
之后转为 Python scalar。该变化不修改 loss、gradient clipping 或 optimizer
数学。

| Run | Step time | Global throughput | Peak memory |
|---|---:|---:|---:|
| P0 deferred sync, ratio `0.25` | `1.3293 s` | `385.2 samples/s` | `174.36 GiB` |
| P1 deferred sync, ratio `0` | `1.2658 s` | `404.4 samples/s` | `214.05 GiB` |

P0 比原三节点中位数提升约 `2.5%`；P1 在 matched node `009` 上比修改前提升
约 `3.2%`。早期 AITER 消融使用了错误 backend 名称；后续显式
`flash_attn_aiter` 已完成 fullgraph、500-step 和三 seed convergence 验收。

产物：

- `/shared_nfs/zirui/runs/flux_fp8_torchprof_20260801T064822Z/`
- `/shared_nfs/zirui/runs/flux_fp8_deferred_sync_20260801T070916Z/`
- `/shared_nfs/zirui/runs/flux_fp8_deferred_sync_p0_20260801T073524Z/`
- `/shared_nfs/zirui/runs/flux_fp8_deferred_sync_aiter_20260801T073300Z/`

### 2026-08-01 FP8 第二轮优化与 convergence

第二轮使用 6 个 MI355X 节点并行筛选，所有有效对照保持 seed `10007`、
MBS/GBS `64/512`、checkpoint ratio `0.25`、block FSDP 和 no-reshard：

| Experiment | Step time | Global samples/s | Result |
|---|---:|---:|---|
| Deferred-sync control | `1.3280 s` | `385.7` | Baseline |
| SDPA CK preference | `1.3203 s` | `387.9` | `+0.6%`, not material |
| No clipping | `1.3090 s` | `391.1` | Numerical risk, reject |
| `max-autotune-no-cudagraphs` | `1.2937 s` | `395.7` | Keep |
| BF16 reduce | `1.2905 s` | `396.6` | Keep as candidate |
| K2 + BF16 reduce, 3-node median | `1.2508 s` | `409.1` | Best production candidate |
| K2 + BF16 reduce + no-AC | `1.1883 s` | `430.8` | Fastest, `214.01 GiB` |
| HSDP `dp_replicate=2` | `1.4195 s` | `360.5` | Reject |
| Root-only FSDP | `1.3003 s` | `393.9` | Reject |
| `reduce-overhead` | `6.0790 s` | `84.2` | Reject |

`crsuse2-m2m-234` 对所有非默认 compile-mode 配置稳定回退到 `7-8 s/step`，
同配置在其他节点正常；该节点的 compile-mode 数据不进入候选统计。

完整 convergence 结果：

| Metric | Optimized ratio `0.25` | Optimized no-AC |
|---|---:|---:|
| Convergence step | `14336` | `14848` |
| Samples | `7,340,032` | `7,602,176` |
| Validation loss | `0.585999` | `0.585429` |
| Time-to-quality | `26,841.915 s` | `26,396.741 s` |
| E2E throughput | `273.5 samples/s` | `288.0 samples/s` |
| Peak memory | `174.33 GiB` | `214.03 GiB` |

旧 qualified FP8 seed `10007` 为 step `13824`、`26,535.193 s`。新的 ratio
`0.25` 虽稳态更快，但晚一个 validation interval，TTQ 反而慢 `1.2%`；no-AC
晚两个 interval，TTQ 只快 `0.5%`。因此 no-AC 不设默认，BF16 reduce 与新
compile mode 仍需更多 seed 才能替代 convergence baseline。

获胜候选的新 profiler 显示 Primus/NeMo profiled step 为 `1.2896/1.0209 s`。
剩余 kernel-work 重点为 attention `+108.4 ms/step`、norm + pointwise
`+77.0 ms/step`、FP8 GEMM + cast/scale `+86.2 ms/step` 和 BF16 GEMM +
optimizer `+46.5 ms/step`。这些值包含 stream overlap，只用于目标排序。

产物：

- `/shared_nfs/zirui/runs/flux_fp8_opt_wave2_20260801T082156Z/`
- `/shared_nfs/zirui/runs/flux_fp8_opt_wave3_20260801T092129Z/`
- `/shared_nfs/zirui/runs/flux_fp8_opt_wave4_20260801T104350Z/`
- `/shared_nfs/zirui/runs/flux_fp8_winner_convergence_20260801T143043Z/`
- `/shared_nfs/zirui/runs/flux_fp8_winner_torchprof_20260801T220505Z/`

### 2026-08-02 AITER attention 三 seed qualification

显式 `flash_attn_aiter` 与 ROCm FlashAttention 2 都通过 compiled FSDP2
forward/backward 和 500-step 稳定性测试。AITER 性能更高：

| Backend | Step time | Global samples/s | Peak memory |
|---|---:|---:|---:|
| SDPA winner | `1.2508 s` | `409.1` | `174.33 GiB` |
| FlashAttention 2 | `1.1870 s` | `431.5` | `177.19 GiB` |
| AITER node `009` | `1.1198 s` | `457.2` | `185.64 GiB` |
| AITER node `235` | `1.1527 s` | `444.6` | `185.64 GiB` |

AITER 两节点平均为 `450.9 samples/s`，比 SDPA winner 高约 `10.2%`，相对
NeMo `525.5 samples/s` 的 gap 缩小到约 `14.2%`。

| Seed | Convergence step | Samples | Validation loss | TTQ |
|---:|---:|---:|---:|---:|
| `10007` | `14336` | `7,340,032` | `0.585916` | `24,275.36 s` |
| `10008` | `13312` | `6,815,744` | `0.585952` | `22,730.13 s` |
| `10009` | `14336` | `7,340,032` | `0.585817` | `24,293.85 s` |
| **Median** | **`14336`** | **`7,340,032`** | | **`24,275.36 s`** |

三次均为 `run_stop=success` 并生成完整 final checkpoint。SDPA compile/reduce
candidate 的三 seed median TTQ 为 `26,841.92 s`；AITER 在 median convergence
step 不变时将 TTQ 降低 `9.6%`，validation-inclusive median throughput 从约
`276.5` 提升到 `302.1 samples/s`。

MLPerf launcher 现在只在 `FLUX_FLOAT8_RECIPE=tensorwise` 且调用方未显式设置
attention backend 时默认选择 `flash_attn_aiter`；BF16 和其他 recipe 继续使用
SDPA。

产物：

- `/shared_nfs/zirui/runs/flux_fp8_attention_perf_20260802T111746Z/`
- `/shared_nfs/zirui/runs/flux_fp8_attention_perf_repeat_20260802T115356Z/`
- `/shared_nfs/zirui/runs/flux_fp8_aiter_convergence_20260802T115356Z/`
- `/shared_nfs/zirui/runs/flux_fp8_aiter_convergence_seeds_20260802T184439Z/`

### 2026-08-03 FP8 raw-GEMM 筛选

在 `zirui3/mlperf-rocm:v0.1-flydsl-v0.2.3` 中为 gfx950 build
Primus-Turbo `56c789e5`，并在全新容器中只通过只读 mount 加载 build 产物。
对六组 `(tokens,input_features,output_features)` shape 的 forward、dgrad 和
wgrad 使用已量化 E4M3/E5M2 operand 与 tensor-wise inverse scale，直接比较
TorchAO/TorchTitan `_scaled_mm` 和 Primus-Turbo backend：

| Backend | 18-case raw latency sum | 相对 `_scaled_mm` | 结论 |
|---|---:|---:|---|
| TorchAO/TorchTitan `_scaled_mm` | `11.886 ms` | control | 保留 |
| HipBLASLt | `18.539 ms` | `+56.0%` | 拒绝 |
| CK | `137.972 ms` | `+1060.7%` | 拒绝 |
| Triton | `16.072 ms` | `+35.2%` | 拒绝 |
| FlyDSL `0.2.3` | `12.539 ms` | `+5.5%` | 拒绝 |

TorchTitan 在这里是 TorchAO control-plane 参考，不是独立 GEMM kernel；相关
路径同样落到 `_scaled_mm`。FlyDSL 数值输出有限并与 control 的 reported
relative L2 一致，但两个 fused single-stream shape 的三种 pass 均没有稳定
收益。10-iteration 初筛中的 QKV forward/dgrad 和 MLP-up forward 局部收益在
`10 warmup + 100 iterations` 复测中分别变成 `0.6%/0.3%/1.7%` 回退。

因此不增加 private TorchAO hook、Primus-Turbo raw bridge 或 shape dispatch
表，也不进入 block/full-training qualification。只有新 kernel 在真实 pass/layout
组合稳定超过 `3%` 时才重开该方向。

产物：

- `/shared_nfs/zirui/runs/primus_turbo_flydsl_build_20260803/`
- `/shared_nfs/zirui/runs/flux_fp8_gemm_flydsl_corrected_full_20260803T053501Z/`
- `/shared_nfs/zirui/runs/flux_fp8_gemm_flydsl_candidates_20260803T054443Z/`

### Primus

- 启动脚本：`local_runs/run_flux_mlperf.sh`
- 配置：`examples/diffusion/configs/MI355X/flux.1_schnell_t2i-pretrain.yaml`
- 此前 Primus 长跑日志（用于代码分析和历史结果交叉检查）：
  - `local_runs/flux_mlperf_mi355x_n1_rcp512_seed1234_full_fixed_20260727_0640.log`
  - `local_runs/flux_mlperf_mi355x_n1_rcp512_seed1234_full_fixed_20260727_0640_mllog.txt`
- FLUX 模型：
  - `primus/backends/diffusion/models/flux/model.py`
  - `primus/backends/diffusion/models/flux/layers.py`
  - `primus/backends/diffusion/models/flux/math.py`
- 训练与 FSDP：
  - `primus/backends/diffusion/trainers/base.py`
  - `primus/backends/diffusion/trainers/fsdp2.py`

### NeMo

- 启动脚本：`/zirui/code/mlperf-training-6-0/flux1/nemo/run_with_docker_debug.sh`
- 实际训练入口：`nemo/run_and_time.sh` → `nemo/src/train.py`
- 此前 NeMo FP8 reference 的解析后配置：
  - `nemo/outputs/2026-07-27/06-08-13/.hydra/config.yaml`
  - `nemo/outputs/2026-07-27/06-08-13/train.log`
- MI355X 配置：`nemo/config_MI355X_01x08x16.sh`
- 自定义 FLUX：`nemo/src/custom_flux.py`

7 月 27 日日志属于此前 FP8 参考；本次 BF16 对齐使用 `_dev` 配置，不能混用两次运行的 precision 和 step time。报告只把代码、解析后配置和日志能够确认的内容写成事实；缺少 profiler 的部分标为推断或待验证。

---

## 3. BF16 对齐结果

| 配置 | 稳态 step time | 吞吐 | 相对 Primus reference | 峰值显存 |
|---|---:|---:|---:|---:|
| NeMo BF16 | `2.4249 s` | `211.1 samples/s` | step time `-30.8%` | 未记录 |
| Primus reference | `3.5045 s` | `146.1 samples/s` | 基线 | `37.11 GiB` |
| Primus AITER only | `3.3614 s` | `152.3 samples/s` | `-4.1%` | `37.11 GiB` |
| Primus no-checkpoint | `2.7143 s` | `188.6 samples/s` | `-22.5%` | `259.62 GiB` |
| Primus no-reshard | `3.0614 s` | `167.2 samples/s` | `-12.6%` | `60.01 GiB` |
| Primus ratio `0.5` + no-reshard | `2.7300 s` | `187.6 samples/s` | `-22.1%` | `161.78 GiB` |
| **Primus ratio `0.25` + no-reshard** | **`2.5271 s`** | **`202.7 samples/s`** | **`-27.9%`** | **`217.74 GiB`** |
| Primus ratio `0.125` + no-reshard | `2.5286 s` | `202.4 samples/s` | `-27.8%` | `250.27 GiB` |
| **Primus ratio `0.25` + no-reshard + compile** | **`1.6445 s`** | **`314.4 samples/s`** | **`-53.1%`** | **`206.61 GiB`** |

NeMo 取 11 个稳态 10-step 窗口，范围为 `2.4198~2.4284 s/step`，均值 `2.424864 s/step`。Primus summary 同样跳过首个日志窗口后求均值；reference 用 11 个窗口，其余成功短测各用 7 个窗口。

未开启 compile 时，优化配置 step time 比 NeMo 高 `4.2%`，吞吐为 NeMo 的 `96.0%`。开启 compile 后，step time 比 NeMo 低 `32.2%`，吞吐为 NeMo 的 `148.9%`。ratio `0.125` 没有比 `0.25` 更快，却多占 `32.53 GiB`，因此仍选择 `0.25`；compile 同时减少了 pointwise/layout 开销，并将峰值显存降低 `11.13 GiB`。

三次完整长跑已经覆盖独立 seed、validation 和 time-to-quality，均达到目标 loss；当前默认配置的 BF16 收敛与工程重复性已验证。与 NeMo 的结论仍限于 BF16 稳态吞吐，因为本轮没有同条件 NeMo BF16 完整 time-to-quality 数据。

### “BF16 应该是 FP8 的 50%”不是端到端性能定律

“50%”只接近某些大 GEMM 在理想利用率下的峰值算力比。FLUX 端到端训练还包括：

- attention、RMSNorm、LayerNorm、RoPE 和大量 pointwise 算子；
- FP8 cast、amax、scale、transpose 等额外开销；
- activation recompute；
- all-gather、reduce-scatter、all-reduce；
- optimizer、gradient norm 和 clipping；
- 数据读取及 validation。

因此 BF16/FP8 峰值比既不能保证端到端恰好为 50%，也不能解释当前所有差距。比较时应同时看：

- steady-state step time；
- samples/s；
- validation 后的 time-to-quality；
- 最终是否达到相同 RCP/target loss。

---

## 4. 两个实现的共同点

当前两个实现的核心模型规模基本一致：

- hidden size：3072
- attention heads：24
- head dimension：128
- double-stream blocks：19
- single-stream blocks：38
- MLP expansion：4 倍，即 12288
- image latent channels：64
- T5 context dimension：4096
- global batch size：512
- micro/local batch size：每 GPU 64
- 8 GPU、TP=PP=CP=1
- AdamW：`beta1=0.9`、`beta2=0.95`、`eps=1e-8`、weight decay 0.1
- validation interval：262144 samples

因此当前差距不是由模型层数、hidden size 或 global batch size 不一致导致的。

---

## 5. 主要实现差异

### 5.1 Activation checkpoint：Primus 已支持 selective，NeMo 当前无 recompute

Primus 原始配置：

- `flux.1_schnell_t2i-pretrain.yaml:57`：`gradient_checkpointing=true`
- `model.py:169-181`：19 个 double block 和 38 个 single block 全部 checkpoint。

这意味着 backward 时需要重新执行几乎整个 transformer forward，增加大量 GEMM、attention、norm 和 pointwise 计算。

NeMo 实际 MCore 配置：

- `train.log:374-378`：`recompute_granularity=None`、`recompute_method=None`、`recompute_num_layers=None`

即当前 NeMo 没有执行对应的全 block activation recompute。

**判断：高影响、已确认。**

本次单变量关闭 checkpoint 后达到 `2.7143 s/step`，比 reference 快 `22.5%`，关闭了约 `73%` 的 Primus-NeMo 绝对 step-time GAP；峰值显存同时从 `37.11 GiB` 升到 `259.62 GiB`。因此 recompute 是首要瓶颈，但默认方案应做 selective checkpoint，而不是全关。

已新增 `gradient_checkpointing_ratio`，默认语义保持 `1.0=全部 block`。MLPerf 本地启动脚本当前选择 `0.25`：57 个 FLUX block 中 checkpoint 14 个。与 no-reshard 组合后达到 `2.5271 s/step`、`217.74 GiB`；ratio `0.125` 性能没有继续提升，因此不值得额外消耗显存。

历史 Primus 快速运行关闭了 checkpoint，并同时启用了 AITER attention 和 block compile：

- `local_runs/flux_mlperf_mi355x_n1_repro_512eval_20260724_1139.log:44-59`
- 平均 `1.4368 s/step`，见对应 summary `:22-24`

该运行峰值显存约 284 GiB，与本次单变量结果一致地说明显存余量很小，不能不经验证直接设成默认值。

### 5.2 并行策略：Primus FSDP2 逐层 reshard，NeMo replicated DDP

Primus：

- `fsdp2.py:103-109`：FP32 参数存储、BF16 forward、FP32 gradient reduction。
- `fsdp2.py:144-215`：默认 `reshard_after_forward=True`，57 个 transformer block 分别 FSDP wrap。
- 当前日志 `:472-475` 确认 wrap 57 个 block。

在 forward 后立刻 reshard 会在 backward 前再次 all-gather 参数。逐 block wrap 还会产生大量粒度较细的 collective 和同步点。

NeMo：

- 当前 Hydra 配置 `strategy.ddp.data_parallel_sharding_strategy=no_shard`
- `use_distributed_optimizer=true`
- `overlap_param_gather=true`
- `overlap_grad_reduce=true`
- bucket size 为 256M

依据：`nemo/outputs/.../train.log:111-123`。

**判断：高影响、已确认。**

NeMo 用显存换通信，并通过 Megatron distributed optimizer 和 bucket overlap 隐藏通信；Primus 当前更偏向节省显存，而不是单节点 8 GPU 的最大吞吐。

本次仅关闭 Primus forward 后 reshard，达到 `3.0614 s/step`，比 reference 快 `12.6%`；峰值显存从 `37.11 GiB` 增到 `60.01 GiB`。这是当前收益/显存比最好的首个默认候选。

该项已落地到 `local_runs/run_flux_mlperf.sh`：`FSDP2_RESHARD_AFTER_FORWARD` 默认从 `true` 改为 `false`。后端和 YAML 的通用默认仍保持 `true`，其他 recipe 不受影响。

后续仅在当前组合仍无法通过 convergence 或需要进一步降低显存时测试：

1. 仅对若干 block group no-reshard；
2. 单节点 replicated DDP；
3. 保留 FSDP，但增大 collective 粒度并验证 prefetch/overlap。

不能仅为了速度直接把 FP32 reduce 改成 BF16；这会改变数值路径，必须重新验证收敛。

### 5.3 Precision 与 Linear：两边均为 BF16，NeMo 使用 TE Linear

Primus 的 QKV、projection 和 MLP 都是普通 PyTorch Linear：

- `layers.py:89-110`：QKV 和 attention projection
- `layers.py:136-159`：double block 两路 MLP
- `layers.py:197-230`：single block 的合并 QKV/MLP

当前没有 Transformer Engine FP8 linear wrapper，也没有 Primus-Turbo FP8 linear。

本次 NeMo BF16 日志明确为 `fp8: null`、`precision: bf16-mixed`；主要 Linear 仍通过 `TEColumnParallelLinear` / `TERowParallelLinear` 执行。因此 TE 的模块实现、kernel dispatch 与布局仍可能优于 Primus `nn.Linear`，但当前 GAP 不能归因于 FP8 GEMM。

**判断：BF16 候选已完成真实 shape 筛选，继续保留 compiled `nn.Linear`；阶段 2 再引入 FP8。**

本轮比较覆盖以下热点 shape：

- `3072 → 12288`：MLP up
- `12288 → 3072`：MLP down
- `3072 → 9216`：QKV
- single block 的合并 QKV + MLP projection

recipe 的图像和文本序列均为 256，MBS=64，因此 double block 的 GEMM `M=16384`，single block 的 GEMM `M=32768`。在 MI355X、PyTorch 2.10/ROCm 7.2 上比较 compiled forward+backward 的中位时间：

| shape `(M,K,N)` | compiled PyTorch | Primus-Turbo | TE eager | 结果 |
|---|---:|---:|---:|---:|
| `(16384,3072,9216)` | `2.295 ms` | `2.406 ms` | `8.671 ms` | Turbo 慢 `4.8%` |
| `(16384,3072,3072)` | `0.961 ms` | `0.977 ms` | `3.047 ms` | Turbo 慢 `1.6%` |
| `(16384,3072,12288)` | `3.128 ms` | `3.324 ms` | `13.303 ms` | Turbo 慢 `6.3%` |
| `(16384,12288,3072)` | `3.079 ms` | `3.182 ms` | `12.619 ms` | Turbo 慢 `3.3%` |
| `(32768,3072,21504)` | `9.481 ms` | `10.619 ms` | `51.071 ms` | Turbo 慢 `12.0%` |
| `(32768,15360,3072)` | `6.983 ms` | `7.132 ms` | `36.642 ms` | Turbo 慢 `2.1%` |

额外兼容性结论：

- 容器中的 `transformer_engine.pytorch.Linear.forward` 带 `torch.compiler.disable`，在当前 block 的 `fullgraph=True` 下直接失败；可运行的 TE eager 路径又比 compiled PyTorch 慢 `3.2~5.4x`，为 TE 拆开 compile region 会放弃已经验证的主要收益。
- Primus-Turbo GEMM 默认也是 HIPBLASLt；显式传递输出 dtype 后可 fullgraph，但没有 shape 收益。
- AITER tuned BF16 GEMM 的公开入口没有可靠的训练 autograd contract；未命中 tuned config 时还会回退 PyTorch，不值得增加自定义 backward 和权重布局。

因此不新增 Linear backend 开关或 wrapper，也不把 Transformer Engine 带入后续精度优化。阶段 2 保持 native PyTorch module、state dict 和 FSDP2，只比较 PyTorch/ROCm、AITER 或 Primus-Turbo 的薄 GEMM 算子路径，并把 scale、cast、transpose 和 backward 开销计入端到端结果。

### 5.4 Attention：Primus 当前 SDPA，NeMo 配置 TE CK/AOTriton fused attention

Primus 当前启动脚本和 YAML 默认：

- `ATTENTION_BACKEND=sdpa`
- `attention/attention.py` 中 SDPA 接收 `B,L,H,D`，内部转成 SDPA 所需的 `B,H,L,D`，输出端再恢复布局。

Primus 已经实现 AITER FlashAttention：

- `attention/aiter.py:36-71`
- `attention/attention.py:299-339`

但当前 MLPerf 运行没有使用它。

NeMo MI355X 配置：

- `NVTE_FUSED_ATTN=1`
- `NVTE_FUSED_ATTN_CK=1`
- `NVTE_FUSED_ATTN_AOTRITON=1`
- `NVTE_CK_USES_FWD_V3=1`
- `NVTE_CK_USES_BWD_V3=1`

实际模型使用 `TEDotProductAttention`，并含 TE 的 Flash/Fused/Unfused dispatcher，见 `train.log:174-190,228-240`。本次运行的容器 stdout 还观察到针对 `gfx950`、BF16、head-dim 128 的 AITER FMHA-v3 forward/backward kernel 被加载，说明实际 attention 不是普通 unfused 路径。不过这部分 console 信息没有保存在仓库的 `train.log` 中；正式性能归档仍应打开 TE debug 或 profiler，记录每个 shape 最终命中的 AITER/CK/AOTriton dispatch。

本次 AITER-only 达到 `3.3614 s/step`，只比 SDPA reference 快 `4.1%`，显存不变。这说明 attention 优化有效，但不是当前 BF16 第一瓶颈。

**判断：中优先级、已完成单变量验证。**

历史 Primus 的 2.1~2.4 倍组合收益同时改变了 attention、checkpoint 和 compile，不能把全部收益写成 AITER attention 的收益。AITER 可作为后续组合测试的默认候选；只有 profiler 显示 attention 仍占主要比例时，再增加 TE CK/AOTriton 分支。

### 5.5 RoPE：Primus block 内已 compile，NeMo 配置 fused RoPE

Primus：

- `math.py:29-36`：每次 forward 通过 `arange`、`einsum`、`cos/sin`、`stack` 构造 RoPE。
- `math.py:39-44`：Q/K 显式转 FP32，应用 RoPE 后再转回原 dtype；该部分位于 compiled transformer block 内。
- `layers.py:21-31`：三个 axis 分别计算后 concat。

位置频率在 block 外构造一次并由所有 block 复用；Q/K 应用则交给 Inductor 编译，不能再按纯 eager 小算子链估算开销。

NeMo：

- `train.log:371`：`apply_rope_fusion=True`
- `train.log:347`：`rotary_interleaved=True`

**判断：AITER 训练 RoPE 实测不满足性能和数值门槛，保留 Inductor 实现。**

将 FLUX multi-axis 角度转换为 AITER GPT-J/interleaved 输入后，对两路 Q/K forward+backward 做了同 shape 测试：

| 序列长度 | compiled PyTorch | AITER RoPE | 数值差异 |
|---|---:|---:|---:|
| 256 | `0.444 ms` | `0.778 ms` | forward/gradient max abs `0.03125` |
| 512 | `0.851 ms` | `1.453 ms` | forward/gradient max abs `0.03125` |

AITER 慢约 `70%~75%`，且现有 autograd wrapper 的 backward 返回签名不完整；fused QK norm + RoPE 路径又偏向 inference/cache/quant，并明确缺少部分 RMS 支持。本轮不增加自定义 autograd 或 FLUX 专用 kernel。只有未来 profiler 证明 RoPE 成为主要热点且上游提供完整训练接口时再评估。

### 5.6 QK RMSNorm：Primus generic `nn.RMSNorm`，NeMo ROCm TE 有 head=128 tuned kernel

Primus：

- `layers.py:66-86`：Q 和 K 分别调用 generic `nn.RMSNorm`。

FLUX head dimension 固定为 128。NeMo ROCm TE 镜像应用了针对 hidden size 128 的 RMSNorm forward/backward tuning patch。

**判断：Primus-Turbo 实测明显慢于 Inductor RMSNorm，保留当前实现。**

对 Q/K 两路 head=128 的完整 forward+backward 测试结果：序列 256 时 compiled PyTorch 为 `0.336 ms`、Primus-Turbo 为 `3.921 ms`；序列 512 时分别为 `0.538 ms` 和 `7.804 ms`。Turbo 慢约 `11.7~14.5` 倍。NeMo compile 路径本身也可把部分 TE RMSNorm 换回本地 PyTorch实现，因此不为对齐模块名称而替换。

此外，当前 `nn.RMSNorm(dim)` 的 `eps=None` 会按 BF16 dtype 选择 epsilon，而 Primus-Turbo 默认 `eps=1e-6`；即使显式对齐 epsilon，性能仍不达标。block 中的 affine-free LayerNorm 与 RMSNorm 语义不同，不做替换。

### 5.7 Block 内 pointwise fusion 与 compile

Primus block 中存在大量独立 eager 操作：

- LayerNorm
- `(1 + scale) * x + shift`
- SiLU/GELU
- gate multiply
- residual add
- split/cat/rearrange

位置主要在 `layers.py:131-193,223-251`。

Primus 已实现逐 block `torch.compile(fullgraph=True)`：

- `fsdp2.py:220-230`

NeMo 当前这次实际运行没有证据显示 `COMPILE_DIT` 已开启；MCore 还报告 `enable_cuda_graph=False`。因此 compile/CUDA graph 不是当前 NeMo 已确认的主要优势，不能把配置文件中“支持 compile”写成“本次运行已使用 compile”。

2026-07-30 的后续隔离确认问题不在 FSDP2：同一容器中的 TorchTitan FLUX compile 能完成 optimizer step，Primus 的 `DoubleStreamBlock` 也能独立完成 compiled backward。失败被缩小到 `SingleStreamBlock`，其 traceback 指向 `_scaled_dot_product_flash_attention_backward` 的真实 stride 与 meta stride 不一致。

根因是 Primus 内部先构造 `B,H,L,D`，再以非连续 view 转成 `B,L,H,D`；TorchTitan 始终使用 sequence-major `B,L,H,D`。修复将 `EmbedND`、QKV rearrange 和 concat 维度统一到 TorchTitan 布局，没有增加 `.contiguous()` 拷贝；同时 compile 改为原地 `block.compile(fullgraph=True)`，保持 FSDP wrap 看到原 module identity。

**判断：已修复并落地。** 8-GPU、MBS 64、ratio `0.25`、no-reshard 的 80-step 结果为 `1.6445 s/step`、`314.4 samples/s`、峰值 `206.61 GiB`。`local_runs/run_flux_mlperf.sh` 默认开启 compile；通用 YAML 和后端默认保持关闭。相比逐个写独立 pointwise kernel，当前方案只依赖 PyTorch compile，同时覆盖 modulation、norm、GELU、residual 和 layout 操作，改动更小。

### 5.8 Optimizer 与 gradient clipping

Primus 已优先使用：

- `torch.optim.AdamW(fused=True)`，见 `base.py:552-565`
- 当前日志确认 fused AdamW，见 `..._0640.log:482`

因此这里不是明显的“缺少 fused Adam kernel”问题。

主要差别是：

- Primus optimizer state 随 FSDP shard，且 gradient norm 对 DTensor 可能执行 `full_tensor()`，见 `base.py:353-375`。
- NeMo 使用 Megatron distributed optimizer，并把 param gather/grad reduce 与计算 overlap。

**判断：中低优先级。**

先 profile optimizer step、grad norm 和通信，再决定是否替换 kernel。不要在没有数据时优先重写 AdamW。

### 5.9 数据管线与主机设置

Primus：

- 4 workers
- pinned memory
- persistent workers
- prefetch factor 2
- precomputed T5/CLIP/VAE 数据

见 `base.py:239-274`。

NeMo：

- Energon
- 8 workers
- 同样读取预处理的 BF16 NumPy payload

见 `custom_data_module.py:23-59` 和 `task_encoder.py:24-56`。

NeMo 启动脚本还执行：

- CPU performance governor
- 关闭深度 idle state
- 关闭 NUMA balancing
- THP always
- 关闭 NMI watchdog/ASLR

见 `runtime_tunables.sh:3-11`。

当前没有日志证据显示 Primus 稳态 step 被 DataLoader 卡住。数据管线和主机设置属于低优先级优化，应先通过 profiler 或 dataloader wait time 确认。

### 5.10 FlyDSL、SwiGLU 与 CUDA Graph 不是本次已确认收益

- `run_with_docker_debug.sh` 支持挂载 FlyDSL kernel 和传递相关环境变量，但当前 Hydra 配置及日志没有证据证明本次训练实际启用了 FlyDSL kernel。
- NeMo 配置中虽存在 `USE_TE_SWIGLU=1`，但实际 FLUX 使用 GELU，且 `gated_linear_unit=False`、`use_te_swiglu=False`。SwiGLU kernel 不适用于当前模型。
- 配置中有 `cuda_graph_impl=local`，但实际 MCore 配置为 `enable_cuda_graph=False`。

这些功能不应被计入当前 NeMo 性能优势。

---

## 6. 可替换为 ROCm 高效 kernel 的位置

| 优先级 | Primus 当前实现 | 建议 ROCm 实现 | 预期作用 | 风险/验证 |
|---|---|---|---|---|
| 已落地 | 全 block checkpoint | ratio `0.25` selective checkpoint | 与 no-reshard 组合后 step time `-27.9%` | full convergence `3/3` 通过 |
| 已落地 | 逐 block forward 后 reshard | no-reshard | 减少参数重聚合 | 当前峰值 `217.74 GiB` |
| 已落地 | eager block | 原地 `Module.compile` + `B,L,H,D` 布局 | step time `-34.9%`，吞吐 `+55.1%` | 首 step 编译约 `44.8 s` |
| 已评估，不替换 | compiled BF16 `nn.Linear` | TE / Primus-Turbo / AITER | Turbo 六个 shape 均更慢；TE 不支持当前 fullgraph | 保留 `nn.Linear` |
| 已评估，不替换 | compiled RoPE + `nn.RMSNorm` | AITER RoPE / Turbo RMSNorm | AITER RoPE 慢 `70%~75%`；Turbo RMSNorm 慢 `11.7~14.5x` | 保留 Inductor 路径 |
| 已落地 | PyTorch SDPA | AITER FlashAttention | FP8 稳态吞吐平均 `+10.2%`，三 seed median TTQ `-9.6%` | 峰值增加 `11.31 GiB`，已通过 convergence |
| 已评估，不替换 | TorchAO `_scaled_mm` | Turbo HipBLASLt/CK/Triton/FlyDSL | 18-case 总延迟均回退 | 不增加 raw bridge |
| P1 | compiled norm/pointwise | 仅针对新 profiler 热点做融合 | 处理剩余 GPU 小算子与 launch 开销 | 先采集 AITER winner trace |
| P2 | fused AdamW / DataLoader / scheduling | 仅在 profiler 证明瓶颈后优化 | 避免无数据的重写 | 当前无主要瓶颈证据 |

BF16 gradient reduction 和 compile mode 已进入 AITER 三 seed candidate；不再
把它们作为未验证的单独默认变更。MXFP8/其他 scaling recipe 属于独立精度项目。

---

## 7. 建议的实施顺序

### 已完成矩阵

| 实验 | 唯一变化 | 结果 | 结论 |
|---|---|---:|---|
| A0 | Primus reference | `3.5045 s` | 基线 |
| A1 | AITER attention | `3.3614 s` | `-4.1%`，保留但不优先深挖 |
| A2 | checkpoint off | `2.7143 s` | `-22.5%`，收益最大但显存过高 |
| A3 | reshard off | `3.0614 s` | `-12.6%`，首个默认候选 |
| A4 | ratio `0.5` + no-reshard | `2.7300 s` | `161.78 GiB`，偏保守 |
| A5 | ratio `0.25` + no-reshard | `2.5271 s` | `217.74 GiB`，compile 前默认 |
| A6 | ratio `0.125` + no-reshard | `2.5286 s` | 无性能收益，显存增至 `250.27 GiB` |
| A7 | A5 + compile（修复前） | 首个 backward 失败 | 定位到 single block SDPA stride mismatch |
| A8 | A5 + compile（布局修复后） | `1.6445 s` | 当前默认；full convergence `3/3` 通过，峰值 `206.61 GiB` |

### 下一步最小矩阵

1. A8 的 3 次重复和 full convergence 已完成，后续把这组结果作为 BF16 回归基线。
2. BF16 Linear、QK RMSNorm 和 RoPE 的当前 ROCm 候选均未通过 shape 级门槛，不新增 backend 分支，也不需要额外 convergence 运行。
3. 若继续阶段 1，只 profile RCCL、recompute、checkpoint I/O 和 optimizer；AITER attention 直接带入候选组合，不扩展多套 attention 后端。
4. 最后单独量化 `grad_reduce_in_fp32=false`，改动后必须重新跑 convergence。
5. 保留 compile eager fallback；只有 PyTorch/ROCm/TE/AITER 版本变化后才重跑本节 kernel 筛选。
6. FP8 rank-0 profiler、scalar-sync、AITER attention 和 raw-GEMM 筛选已完成；下一轮先 profile 最终 AITER winner，再量化 norm/pointwise 与 FSDP/optimizer scheduling。

阶段 1 的近期目标是先把 BF16 GAP 稳定压缩，并保持可接受显存与 convergence。单节点 replicated DDP 仅在 no-reshard/selective checkpoint 仍明显落后时再测，避免同时引入新的训练策略。

阶段 2 在不引入 TE module 的前提下，依次验证 FP8/MXFP8 MLP、QKV 和 output projection，并记录 scale/cast、backward、显存和 loss 曲线。

---

## 8. 验收标准

性能优化不能只看瞬时 step time，建议同时满足：

1. global batch 和样本顺序符合 MLPerf 要求；
2. validation loss 计算与 reference 一致；
3. 固定 seed 下 loss 曲线没有明显漂移；
4. 最终达到 `target loss <= 0.586`；
5. 统计包含 validation 的 time-to-quality；
6. 分别报告 BF16 与 FP8，不把不同 precision 的 RCP 混为一个结果；
7. 至少重复 3 次，报告中位数和波动；
8. profiler 证明优化命中了预期 kernel，而不是只依赖环境变量；
9. 对齐或显式区分 gradient reduction dtype：NeMo 为 BF16，Primus qualified default 为 FP32；BF16 candidate 必须单独做 convergence qualification。

当前 A8 默认已满足三次独立 seed 和 target loss 验收；尚未完成的是同条件 NeMo BF16 time-to-quality 对照、profiler 热点确认，以及清缓存等正式 submission 环境要求。

---

## 9. 最终判断

BF16 优化配置已将 Primus 吞吐从 NeMo 的 `69.2%` 提升到 `148.9%`，稳态性能 GAP 已反转：

- ratio `0.25` selective checkpoint + no-reshard + compile 达到 `1.6445 s/step`、`314.4 samples/s`；
- 峰值显存 `206.61 GiB`，低于 compile 前的 `217.74 GiB`、no-checkpoint 的 `259.62 GiB` 和 ratio `0.125` 的 `250.27 GiB`；
- AITER attention 单项仅提升 `4.1%`，说明 attention 不是第一瓶颈；
- compile 失败根因是 single block SDPA backward 的布局 stride mismatch，不是 FSDP2 不兼容；对齐 TorchTitan 的 `B,L,H,D` 布局后已通过单 block、完整 FSDP2 和 80-step 回归；
- 三次独立 seed 均达到 `validation_loss <= 0.586`，收敛 step 为 `13824/14336/13824`，中位 time-to-quality 为 `9.37 h`，当前 BF16 默认的收敛与重复性风险已关闭；
- NeMo 仍有 TE BF16 Linear、fused RoPE、distributed optimizer overlap 和 BF16 gradient reduction 等实现差异；其中 Linear/RMSNorm/RoPE 已证实不能通过当前 ROCm 候选直接缩小 GAP，不能仅为模块名称对齐而修改。
- FP8 结论与 BF16 不同：Primus P0 native dynamic FP8 中位吞吐为 `375.9 samples/s`，仅为 NeMo `delayed_short` 的 `71.5%`；关闭 activation checkpoint 后也只有 `389.5 samples/s`，为 NeMo 的 `74.1%`。
- deferred sync、compile mode 和 BF16 reduction 将 production steady throughput 提高到 `409.1 samples/s`，但仍比 NeMo 低 `22.1%`。
- full convergence 晚一个 validation interval，说明单 seed time-to-quality 尚未从 steady throughput 优化中获益。
- pre-AITER profiler 曾将目标排序为 attention、norm/pointwise 和 fused native FP8 GEMM；后续 AITER 已落地，而 raw-GEMM 筛选无胜者，HSDP/root-only FSDP 也已实测回退。
- AITER attention 已完成三 seed qualification，将稳态吞吐提高到平均 `450.9 samples/s`，三 seed median TTQ 改善 `9.6%`，当前相对 NeMo steady gap 约 `14.2%`。

最简实施路径是：

1. **保留当前 compile 默认及三次收敛结果作为 BF16 回归基线。**
2. **停止落地当前 TE/Turbo/AITER BF16 Linear、RMSNorm 和 RoPE 候选；后续只 profile checkpoint I/O、RCCL、optimizer 和 reduction dtype。**
3. **补充同条件 NeMo BF16 time-to-quality 对照；随后进入 native PyTorch + ROCm kernel 的 FP8/MXFP8 阶段，不引入 TE。**
4. **保留 deferred scalar sync、block FSDP、checkpoint ratio `0.25`；不把 no-AC 设为默认。**
5. **FP8 MLPerf 默认使用已通过三 seed qualification 的 AITER attention；BF16 继续使用 SDPA。**
6. **先 profile 最终 AITER winner，再按实测结果处理 norm/pointwise 或 host/GPU scheduling；不落地当前 FP8 raw-GEMM backend，也不扩展 FSDP topology。**

Primus 当前 BF16 默认已经完成三次收敛与重复性验证，且 80-step 稳态性能超过本次 NeMo BF16 reference。由于缺少同条件 NeMo BF16 完整长跑，当前仍不能声称两者 time-to-quality 已对齐。FP8 AITER production candidate 已完成三 seed convergence，稳态吞吐提高到平均 `450.9 samples/s`，相对 NeMo gap 约 `14.2%`。现有通用 FP8 GEMM backend 已全部筛除；下一步必须以 AITER winner 的新 profiler 归因为准。
