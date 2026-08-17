# FLUX MXFP8 ROCm 优化分析与落地计划

## 1. 目标与结论

目标是在 MI355X/gfx950 上保持 MLPerf FLUX validation loss `<=0.586`，同时降低 time-to-quality。性能优化必须计入动态量化、scale/layout、forward、dgrad 和 wgrad，不能用预量化 GEMM 的结果代替训练收益。

当前结论：

1. TorchAO MXFP8 已完成正确性闭环，但当前 kernel 没有性能价值。
2. ALTO 证明 MXFP8 可以很快，但当前数值轨迹和 MLPerf 大 batch 性能未通过。
3. 当前 tensor-wise FP8 已将 QKV forward/dgrad 降为 FP8，只有 38 个 double-stream QKV wgrad 保持高精度；单独降低它的端到端收益预计只有 `1-3%`。
4. 当前最值得验证的是 Primus-Turbo 的 gfx950 FlyDSL dense MXFP8 GEMM、dual-cast quant 和 autograd 路径。先做真实 FLUX shape 的完整算子 A/B；只有过性能、数值和 fullgraph gate 后才接入 228-layer FLUX。
5. MXFP4 后置。更细的 scale 不能抵消 E2M1 的 4-bit 信息损失，不能假设 MXFP4 精度必然优于 FP8。

本实验使用独立 worktree：

- Primus 基线：`6719c34392e31a231253f90fbb373cabc13fb9bf`
- worktree：`/shared_nfs/zirui/code/primus-flux-mxfp8-opt`
- branch：`dev/zirui/flux-mxfp8-opt`
- Primus-Turbo 检查基线：`56c789e58f72aaf733b7715f1536be1ed33b69a1`

## 2. 已有实验结论

### 2.1 Qualified tensor-wise FP8 基线

当前稳定策略转换 228 个 transformer-block Linear：

- 190 个非 QKV Linear：FP8 forward/dgrad/wgrad；
- 38 个 image/text QKV：FP8 forward/dgrad，高精度 wgrad；
- 参数、optimizer、FSDP all-gather 和非 Linear 算子保持 BF16/FP32。

seed 10007 的 matched BF16 和 selective FP8 都在 step `13824`、samples `7,077,888` 达到 validation loss `<=0.586`。FP8 TTQ 为 `26,535.193 s`，比 BF16 的 `33,742.858 s` 快 `21.4%`。

依据：`/shared_nfs/zirui/code/primus-flux-fp8-gemm-opt/flux-fp8.md`。

### 2.2 TorchAO MXFP8

Primus commit `fdbe4d00` 与 TorchAO ROCm patch `f9f48318` 已验证：

- gfx950 row-major E8M0 scale layout；
- HIP backward dim1 cast 使用 FlyDSL；
- 真实 FLUX shapes eager/fullgraph FWD、dgrad、wgrad；
- 228-layer、190/38 wgrad policy；
- 2 GPU FSDP2、block compile、AC ratio `0.25` 和 DTCP save/resume。

性能不通过：

| 项目 | BF16 | TorchAO MXFP8 |
|---|---:|---:|
| 2 GPU mean step | `1.0927 s` | `1.0991 s` |
| 首次 compile/warmup | `28.33 s` | `112.69 s` |
| compiled Linear FWD+BWD，M=256 | `0.332-0.346 ms` | `1.194-1.246 ms` |

MXFP8 Linear 慢约 `3.46-3.76x`；M 增至 2048 后仍慢约 `2.98x`。

8 GPU 长跑 validation loss 从 step 1024 的 `0.659656` 降至 step 10752 的 `0.588444`，尚未达到 `0.586`。这证明路径稳定且接近目标，但不构成 convergence pass。

依据：

- `/shared_nfs/zirui/runs/mxfp8_perf/summary.md`
- `/shared_nfs/zirui/runs/primus-torchao-mxfp8-aligned-20260804T034527Z/`

### 2.3 ALTO MXFP8

ALTO 使用相同的 228-layer 和 190/38 policy。2 GPU、local batch 1、compile、AC ratio `0.25` 的短测 mean step 为 `0.7655 s`，比 matched BF16 快约 `30%`，但 100-step 数值偏离明显：

| 指标 | TorchAO vs BF16 | ALTO vs BF16 |
|---|---:|---:|
| Loss MAE | `0.0869` | `0.2509` |
| Loss correlation | `0.962` | `0.659` |
| Grad-norm MAE | `0.885` | `2.744` |
| Grad-norm correlation | `0.849` | `0.301` |

8 GPU 长跑在 step 6144 的 validation loss 为 `0.599558`，同 step TorchAO 为 `0.595649`。ALTO MBS=64 稳态约 `9.6 s/step`，也没有延续小 batch 性能优势。

依据：

- `/shared_nfs/zirui/runs/mxfp8_alto/summary.md`
- `/shared_nfs/zirui/runs/primus-alto-hy-mxfp8-loss-vs-samples-20260803T160700Z/`

### 2.4 MXFP4

现有 ALTO 配置出现过 `MXFP4 -> BF16` 和历史 `MXFP4 -> MXFP8`，没有已验证的 `MXFP8 -> MXFP4` 完整 MLPerf convergence。当前不能把 MXFP4 放入主 recipe。

## 3. QKV 性能上限

当前剩余的高精度 QKV 计算是 38 个 double-stream QKV wgrad。按真实 shape 和调用次数：

- double QKV 全 F/D/W 约占 block Linear FLOPs 的 `9.6%`；
- 仅 QKV wgrad 约占 block Linear FLOPs 的 `2.57%`；
- profiler 中包含 QKV wgrad在内的全部“BF16/other GEMM + optimizer/norm”约 `48.5 ms/step`，相对约 `1.1 s` step 的绝对上界为 `4.3%`。

即使把整个 `48.5 ms` 都错误地归给 QKV wgrad，理想端到端上限也只有：

| 方案 | 理想上限 | 现实预期 |
|---|---:|---:|
| FP8 | 约 `2.2%` | `1-2%` |
| MXFP8 | 约 `2.2%` | `0-2%`，cast/scale 可能抵消收益 |
| MXFP4 | 约 `3.3%` | `1-3%`，精度风险最大 |

已有 FP8 消融中，image/text QKV wgrad 会导致早期 NaN。因此先保持 QKV high-precision wgrad；整体 MXFP8 收敛后，再将 QKV MXFP8 wgrad作为单独实验。

## 4. ROCm kernel 现状

| 实现 | 当前能力 | 主要缺口 | 定位 |
|---|---|---|---|
| TorchAO | FLUX fullgraph/FSDP2/DTCP 已通过 | 真实 shape 慢于 BF16 | correctness baseline |
| ALTO | native gfx950，短测快 | 数值偏离，大 batch 性能差 | 对照和算法参考 |
| Primus-Turbo | FlyDSL dense MXFP8、dual-cast quant、E4M3/E5M2、F/D/W autograd | 高层 `gemm_fp8()` 有 `torch._dynamo.disable`，现有 module compile 不是 fullgraph | 首选 kernel 候选 |
| AITER | MX quant、dot_scaled、MoE/attention 路径 | 无已验收 FLUX dense training contract | raw kernel 对照 |
| MSLK | gfx950 MX8xMX8、MX8xMX4、grouped GEMM | 无 FLUX autograd/FSDP/fullgraph 接入 | raw kernel 对照 |

Primus-Turbo 关键文件：

- `primus_turbo/flydsl/gemm/mxfp8_gemm_kernel.py`
- `primus_turbo/pytorch/kernels/gemm/gemm_fp8_impl.py`
- `primus_turbo/pytorch/ops/gemm_fp8.py`
- `primus_turbo/pytorch/modules/linear_fp8.py`

## 5. 落地顺序与 gates

### P0：真实 shape 算子筛选（当前阶段）

使用 `local_runs/bench_flux_mxfp8.py`，在同一 MI355X、同一软件栈中比较 BF16、TorchAO 和 Primus-Turbo。测量必须包含动态 quant/scale/layout 和完整 FWD+BWD。

真实 shape：

| family | `(M,K,N)` | 训练优先级 |
|---|---|---|
| double QKV | `(16384,3072,9216)` | QKV 只用于 coverage，生产 wgrad 仍为 HP |
| double projection | `(16384,3072,3072)` | P2 |
| double MLP up | `(16384,3072,12288)` | P1 |
| double MLP down | `(16384,12288,3072)` | P1 |
| single fused QKV+MLP up | `(32768,3072,21504)` | P0 |
| single attention+MLP down | `(32768,15360,3072)` | P0 |

Gate：

1. output/input-grad/weight-grad finite；
2. 相对 BF16 的误差不差于 TorchAO MXFP8；
3. 100 次稳定测量，报告 median/p10/p90；
4. 按真实调用频率加权，affected Linear FWD+BWD 至少快 `10%`；
5. 不能只报告预量化 GEMM。

未过 gate 时不增加 Primus provider，不改 trainer/checkpoint。

#### P0 首轮结果

在 `crsuse2-m2m-119` 的一张 MI355X 上完成首轮 eager A/B。环境为 PyTorch `2.12.0+rocm10.1.0a20260811`、ROCm `7.16.26315`，Primus-Turbo `56c789e5` 按 gfx950 在容器内重编译。每项包括动态 quant、scale/layout、FWD、dgrad 和 wgrad；5 次 warmup 后测 20 次。结果位于 `/shared_nfs/zirui/runs/mxfp8_turbo_opt/`。

| shape | BF16 ms | Turbo hybrid ms | speedup | Turbo E4M3 ms | speedup |
|---|---:|---:|---:|---:|---:|
| double QKV | `2.011` | `1.579` | `1.27x` | `1.586` | `1.27x` |
| double projection | `0.741` | `0.702` | `1.06x` | `0.706` | `1.05x` |
| double MLP up | `2.685` | `2.098` | `1.28x` | `2.126` | `1.26x` |
| double MLP down | `2.568` | `1.974` | `1.30x` | `1.963` | `1.31x` |
| single fused up | `9.075` | `6.752` | `1.34x` | `6.965` | `1.30x` |
| single down | `6.298` | `4.547` | `1.39x` | `4.626` | `1.36x` |

六个 case 等权合计，hybrid 比 BF16 快 `1.32x`，E4M3 比 BF16 快 `1.30x`。profiler 确认执行 `kernel_mxfp8_nt_1.kd` 和 MXFP8 dual-quant kernels，不是 BF16 fallback。最大收益来自占 FLUX GEMM 工作量最高的两个 single-stream shape，值得进入 P1。

数值仍需处理：forward relative-L2 约 `0.027`；hybrid E5M2 backward 的 dgrad/wgrad 约 `0.059-0.060`，全 E4M3 降至约 `0.038-0.040`，但仍高于此前 TorchAO probe 的约 `0.027`。首轮输入是随机张量，下一轮需用真实 FLUX tensor比较 E4M3、hybrid 和 TorchAO，并用 100-step 训练决定格式，不能仅凭随机 relative-L2 选择。

`fullgraph=True` 已按预期失败，直接命中 `gemm_fp8()` 的 `torch.compiler.disable`。因此 P0 性能 gate 初步通过，P1 compile gate 未通过；下一步是最小 custom-op/autograd contract，而不是现在接入 228-layer provider。

### P1：compile-friendly raw op

P0 通过后，在 Primus-Turbo 提供最小 raw custom-op/autograd contract：

- 输入为 BF16 activation/weight/grad output；
- 内部使用 dual-cast quant 和 FlyDSL MXFP8 GEMM；
- fake/meta 与 autograd 显式注册；
- `torch.compile(fullgraph=True)` 无 graph break；
- 不新增第二套 Linear/state-dict/checkpoint 格式。

Gate：真实 shape单层 fullgraph FWD+BWD、非零 bias、连续/非连续 grad output，且 compile 后性能仍优于 BF16/TorchAO。

#### P1 结果

在独立 Primus-Turbo worktree `/shared_nfs/zirui/code/Primus-Turbo-flux-mxfp8-compile`、分支 `experiment/flux-mxfp8-compile`、commit `ebe2614e` 增加了最小 `torch.library.custom_op`：

- fake/meta 和 autograd 显式注册；
- forward 保存 columnwise MXFP8 operand供 backward复用；
- 支持 E4M3/hybrid；
- 支持 QKV high-precision wgrad；
- `Float8Linear` 的 MX_BLOCKWISE 路径绕开 graph-disabled 通用 wrapper；
- 非 MXFP8 路径不变。

E4M3/hybrid、MXFP8/HP wgrad、非连续 grad output 的 `fullgraph=True` 测试为 `4 passed`。真实 shape compiled FWD+BWD 相对 compiled BF16：double QKV `1.23x`、double projection `0.86x`、double MLP up `1.34x`、double MLP down `1.26x`、single up/down `1.32x/1.33x`；六 shape 等权合计约 `1.29x`。double projection 应保留按 shape fallback 候选。

### P2：228-layer FLUX 实验 provider

只在 P0/P1 通过后增加 `provider=primus_turbo, recipe=mxfp8`：

- 复用当前精确的 228 FQN；
- 保持 190 MXFP8 wgrad + 38 QKV HP wgrad；
- 参数、optimizer、FSDP all-gather/reduce 和 checkpoint 保持 BF16/FP32；
- 先 1 GPU，再 2 GPU FSDP2/DTCP，再 8 GPU MBS=64；
- block compile 和 MLPerf AC 配置必须与 control 相同。

端到端 gate：相同节点重复测试至少快 `3%`，否则不进入 convergence。

#### P2 首轮结果

Primus 实验 provider 已转换 228 层并保持 190 MXFP8 wgrad + 38 QKV HP wgrad。单元测试通过；2 GPU、block fullgraph、FSDP2、AC ratio `0.25`、AITER attention 完成 12 个 finite step，step 1 loss/grad norm 与 BF16 都为 `1.6148/3.3907`。

2 GPU local batch 1 的 steps 2-12：

| 指标 | BF16 | Primus-Turbo MXFP8 |
|---|---:|---:|
| mean step | `0.6991 s` | `0.7000 s` |
| throughput | `1.4357` | `1.4324 samples/GPU/s` |
| peak memory | `98.57 GB` | `103.63 GB` |

小 M 端到端没有收益且多约 `5.1 GB` peak memory，符合 double projection 小 shape 无收益的信号。

8 GPU、local batch 64 首轮在 first backward 的多 rank FlyDSL JIT/autotune 阶段超过 50 分钟仍未完成。随后把单 rank autotune 得到的 15 个 E4M3 F/D/W config 固定为 opt-in 表（Primus-Turbo commit `b0bcdc25`），由 `PRIMUS_TURBO_MXFP8_USE_FLUX_CONFIGS=1` 启用。首次 compile step 降至约 `366 s`，并完成 matched 5-step A/B：

| 指标，steps 2-5 | BF16 | Primus-Turbo MXFP8 | 结果 |
|---|---:|---:|---:|
| mean step | `1.5550 s` | `1.2325 s` | `20.7%` lower |
| throughput | `41.3865` | `51.9083 samples/GPU/s` | `25.4%` higher |
| peak memory | `206.57 GB` | `187.99 GB` | `18.58 GB` lower |

step 1 loss/grad norm 两边相同为 `1.8657/4.0168`；到 step 5 BF16/MXFP8 loss 为 `1.8655/1.8631`，短程 finite且接近。P2 的大 batch性能 gate 已通过，但 MXFP8 首次启动仍比 BF16 的约 `94 s` 慢约 4 倍；后续应生成可复用 AOT cache/image。下一步进入 100/500-step 数值对照和 DTCP save/resume，不直接开始完整 convergence。

### P3：数值与 convergence

顺序为：

1. 同 seed 100 step loss/grad norm；
2. 500 step validation trend；
3. 8 GPU 完整单 seed达到 `0.586`；
4. 多 seed samples-to-convergence 和 TTQ。

若误差类似 ALTO，先做 layer-selective fallback/per-depth gradient 对比，不更换 GEMM backend掩盖问题。

#### P3 100-step 与 DTCP 结果

8 GPU、local batch 64、AITER attention、block compile、AC ratio `0.25` 下运行到 step 50，保存完整 DTCP；新进程明确从 `checkpoint-50` 恢复 model、optimizer、scheduler 和 global step，并完成 step 51-100及 `checkpoint-100`，same-provider resume 通过。

matched BF16/MXFP8 每 10 step 对比显示 MXFP8 全程 finite，loss 趋势相关系数 `0.991`、MAE `0.0772`、平均相对差 `6.27%`；grad norm 相关系数 `0.974`、MAE `0.3648`、平均相对差 `27.6%`。step 100 BF16/MXFP8 loss 为 `0.8531/0.8111`，grad norm 为 `0.7166/0.5948`。MXFP8 当前比 BF16下降更快但已形成可见数值轨迹差，不能把较低 training loss直接解释为更好 convergence。

稳态 logged-window throughput 为 BF16 `41.52`、MXFP8约 `50.3 samples/GPU/s`，MXFP8提高约 `21%`；checkpoint/resume后的 first logged window包含重新 compile，不计入稳态。

#### P3 512-step validation 结果

matched BF16/MXFP8 从头运行 512 step，使用 GBS `512`、warmup `512`、AITER attention、block compile、AC ratio `0.25`，并在 samples `262,144` 做第一次完整 validation。precomputed DataLoader 使用 4 workers 时两边都会在约 step 500 耗尽文件描述符；将新暴露的 `DATALOADER_NUM_WORKERS=0` 后两边都完成，因此这是公共 dataloader问题，不是 MXFP8 kernel问题。

| 指标 | BF16 | MXFP8 | 差异 |
|---|---:|---:|---:|
| step 500 training loss | `0.6118` | `0.6166` | `+0.0048` |
| step 512 validation loss | `0.689553` | `0.690048` | `+0.000495` (`+0.072%`) |
| mean logged-window throughput | `41.2407` | `49.3235 samples/GPU/s` | `+19.6%` |
| peak memory | `206.57 GB` | `187.99 GB` | `-18.58 GB` |

第一次 validation 几乎重合，500-step validation gate 通过。

#### P3 正式 warmup 到 2048 step

随后使用正式 `warmup_steps=1600` 从头跑到 step 2048，得到四个 matched validation点：

| Step / samples | BF16 | MXFP8 | MXFP8 - BF16 |
|---:|---:|---:|---:|
| 512 / `262,144` | `0.713264` | `0.717329` | `+0.004065` |
| 1024 / `524,288` | `0.659750` | `0.658402` | `-0.001348` |
| 1536 / `786,432` | `0.635366` | `0.635705` | `+0.000339` |
| 2048 / `1,048,576` | `0.621658` | `0.621796` | `+0.000138` |

step 512 的早期差异随后收敛；step 1024-2048 的 validation差异绝对值不超过 `0.00135`，step 2048 只差 `0.000138`，数值趋势与 BF16 对齐。包含周期 validation和部分 post-validation慢窗口的 matched summary throughput 为 BF16 `40.1860`、MXFP8 `48.1165 samples/GPU/s`，MXFP8高 `19.7%`；peak memory仍少 `18.58 GB`。

正式 warmup多点 validation gate 通过，已具备启动单 seed完整 MLPerf convergence的依据。

#### P3 persistent compile cache

将 FlyDSL、Triton 和 TorchInductor cache 挂到持久目录后，同一 8 GPU fullgraph step 的首次启动从冷 cache `444.70 s` 降到热 cache `100.83 s`，减少 `77.3%`，已接近 BF16 的约 `94 s`。launcher 现在默认使用 worktree 下 `.cache/flux_mlperf/{flydsl,triton,torchinductor}`，可通过 `PRIMUS_FLUX_CACHE_ROOT` 覆盖。cache 必须与固定 image、PyTorch、FlyDSL、Primus-Turbo 和 Primus commit 一起管理；依赖变化后允许自然产生新 key，不能跨不兼容软件栈复制未知 cache。

热 cache 已显著降低 TTQ 固定成本。下一步可启动单 seed完整 MLPerf convergence，同时保留 cold/warm compile 时间和训练起止时间，分别报告 kernel cache收益与稳态 TTQ。

### P4：QKV 与 MXFP4

整体 MXFP8 通过后才测试 QKV MXFP8 wgrad。要求 500 step finite、validation 不回退、端到端至少提升 `1%`。

MXFP4 最后验证：先单阶段短程，再研究 `MXFP8 -> MXFP4` switch step。每次只改变一个变量，不同时引入 Hadamard、SR、de-oscillation 或低精度通信。

## 6. 当前改动边界

当前 worktree 只增加分析文档和 P0 benchmark，不修改生产 provider。原因是 Primus-Turbo 高层接口目前明确会 graph break；在真实性能 gate 前接入 228-layer 只会增加无法验收的分支。
