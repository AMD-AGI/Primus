# TorchTitan/ALTO MXFP8/MXFP4 实现分析与 Primus Flux 迁移计划

> 状态：已完成静态分析、迁移设计和 Gate 0；已在独立 `dev/zirui/flux-mxfp8` worktree（实现 commit `fdbe4d00b91854960e93ae8c1ed9a21e1b62bd8b`）增加最小 TorchAO MXFP8 Primus 适配，并在独立 TorchAO worktree 修复 gfx950 dim1 路由和 scale layout。单算子、真实 Flux GEMM shape、两 rank完整 Flux FSDP2、block compile、AC ratio=0.25、DTCP save/resume 和单元测试已通过。短程同口径性能显示当前 MXFP8 无加速且 compile 启动开销显著，暂不具备性能 recipe 价值；8 GPU、长程数值与完整收敛尚未验收。当前 Primus 原工作树已有独立完成的 native TorchAO dynamic tensor-wise FP8 实现和运行证据，本文将其作为 MXFP8 的正式回归基线。
>
> 结论先行：**支持并验证单阶段 MXFP8 是首要目标。`MXFP8 → MXFP4` 两阶段只是潜在提升训练效率的备选实验方案，不是当前交付前提。** 现有 ALTO MXFP4/两阶段实验相对 BF16 有较大精度损失，因此只有在 MXFP8 性能和收敛稳定后，且新的短程数值证据显示收益值得继续投入时，才进入 MXFP4。第一版只降低 Flux DiT transformer block 内大 `Linear` GEMM 的计算精度，参数、优化器、FSDP 通信、checkpoint、非 Linear 算子继续保持 FP32/BF16。

## 1. 执行摘要

### 1.1 关键判断

1. 必须区分两套 TorchTitan 参考：旧 `/shared_nfs/zirui/code/torchtitan` 的 ROCm 实验脚本通过外部 **ALTO** 提供 MXFP8/MXFP4；当前 `/shared_nfs/zirui/code/torchtitan-main` 已有 TorchAO `MXFP8Converter` 和 Flux MXFP8 preset，但没有 Flux MXFP4 训练实现。
2. 旧 `flux-mxfp8-docker.sh` 走 ALTO“全程 MXFP8”配方；当前 `torchtitan-main/run_flux_test.sh` 默认使用 `flux_schnell_mlperf_preprocessed`，仍是 BF16。只有显式选择 `flux_schnell_mxfp8` 或 `flux_dev_mxfp8` 才启用 MXFP8，而且这两个 preset 不是 MLPerf preprocessed 配方。
3. MXFP8/MXFP4 都不是端到端低比特训练：在 gfx950 native 路径上，低精度可覆盖被选中 `Linear` 的 forward、dgrad 和 wgrad GEMM；master weight、optimizer state、FSDP all-gather/reduce、bias、norm、activation、residual、loss 和 checkpoint 仍为高精度。非 gfx950 fallback 的 GEMM 本身会回到高精度，不能称为原生低精度训练。
4. 用户要求的 FP4 目标是两阶段 **`MXFP8 → MXFP4`**。但当前检出的 ALTO 配方不是这个顺序：
   - 当前工作树：`MXFP4 → BF16`；
   - 历史文件曾配置 `MXFP4 → MXFP8`；裸 `mxfp8` 会被 validator 规范化为 `mxfp8_e4m3`，静态 dispatch 可达，但未找到跨越切换点的训练日志，不能视为实跑成功证据；
   - 当前没有已落地的 `MXFP8 → MXFP4` recipe，也没有两个独立脚本自动传递完整 checkpoint 的闭环。
5. ALTO 的调度 wrapper 已具备在同一组高精度参数上运行 MXFP8 和 MXFP4 的能力，因此目标顺序可以通过单进程 schedule 实现，**不需要在切换点替换 module、Parameter、optimizer 或 FSDP handle**。
6. Primus 原生 diffusion Flux backend 已有 TorchAO dynamic tensor-wise FP8：转换 228 个 block Linear，190 个使用 FP8 wgrad、38 个 image/text QKV 使用高精度 wgrad，并已有 compile、FSDP2、DTCP resume 和一个完整 seed 的 BF16 收敛对齐证据；但当前没有 MXFP8/MXFP4 生命周期。
7. 当前 `torchtitan-main` 的统一 `MXFP8Converter` 已改用 TorchAO `MXFP8TrainingOpConfig`，不再使用旧 `MXLinearConfig`；但其运行时门禁仅允许 CUDA SM100+。历史提交曾允许 ROCm gfx950，统一 converter 重构时该门禁被移除，不能据此宣称当前 TorchTitan MXFP8 支持 MI355X。
8. 当前 TorchAO main 同时存在直接 `MXFP8Linear` 和 `MXFP8TrainingOpConfig` weight-wrapper 两条路径。wrapper 是当前 TorchTitan 的上游方向，但 module-level 转换会包装匹配 Linear 的所有本地 Parameter，而 MXFP8 Linear dispatch 没有可靠保留 bias；Flux 大量 Linear 使用非零 bias，因此第一版已选择直接 `MXFP8Linear`。上游 dense backward 默认 dim1 cast 仍硬编码 CUDA，且 `_scaled_mm` 无条件使用 NVIDIA blocked scale layout；独立 TorchAO worktree 已用“HIP 选择 FlyDSL + row-major scales”的最小 patch 修复并通过单算子/compile/Flux shape 测试，仍需提交上游或固定为不可变依赖。
9. TorchAO 与 ALTO 仍可作为两个互斥 provider，但不应在 P1 同时建设。先并行做两个小型 Gate 0 技术探针，再按结果顺序建设 provider：TorchAO gfx950 原生路径成立时以 TorchAO 为正式主线；若其 native kernel 短期阻塞而 MI355X MXFP8 有交付时限，才以 ALTO 作为版本固定的战术 fallback。只有 MXFP4 目标继续成立时，才把 ALTO 扩展为第二个正式 provider 和两阶段 schedule。单次运行固定 provider；full resume 只支持相同 provider/fingerprint，跨 provider 仅支持 model-only warm start。
10. 双 provider 的配置与分派代码复杂度低，但生产验证成本并不低；不应把“适配层很薄”误判为“可以低成本同时交付”。工程接入复杂度为：
   - 单阶段 MXFP8：**中等**；
   - 两个单阶段 MXFP8 provider 并存：**中高**；
   - 两阶段 MXFP8→MXFP4：**代码接入中等，完整数值/收敛交付高**；
   - 真正困难的部分不是配置或模块替换，而是 ROCm/PyTorch/torchao/ALTO 版本匹配、FSDP tensor subclass 兼容和收敛验证。

### 1.2 推荐路线

推荐采用 Gate 0 + P0-P3，P4 为后续可选优化：

- **Gate 0：并行做 TorchAO 与 ALTO 两个最小技术探针，不并行建设完整 provider。** TorchAO probe 验证 gfx950 dim0/dim1 cast、FWD/dgrad/wgrad、原生 scaled-mm、非零 bias 和 `fullgraph=True`；ALTO probe 验证最小无副作用 import、gfx950 native FWD/dgrad/wgrad、非零 bias 和 `fullgraph=True`。探针使用相同真实 Flux GEMM shape，并明确区分 native kernel、emulation 和高精度 fallback。
- **P0：冻结 BF16、native tensor-wise FP8 和 NeMo delayed FP8 三条基线并固定依赖身份。** 明确 Primus/PyTorch/TorchAO commit、wheel/hash 和容器；升级 TorchAO 后必须先回归现有 tensor-wise FP8。ALTO 身份和安装约束推迟到确实选用 ALTO 时。
- **P1：默认接当前 TorchAO main 的单阶段 MXFP8。** 先验证 weight-only wrapper 与直接 `MXFP8Linear`，默认采用能正确保留 bias/state dict/FSDP 语义的最小路径；补齐/消费 ROCm Triton/FlyDSL dim1 路径，并以 kernel trace 证明没有落入 CUDA-only 或软件 emulation。若 Gate 0 证明 TorchAO gfx950 native 路径短期不可用，而 MI355X MXFP8 必须尽快交付，则将 ALTO 单阶段 MXFP8 提前为战术 P1，但必须固定完整依赖身份。bring-up 可关闭 compile，交付验收必须恢复 block compile。
- **P2：按需接单阶段 ALTO MXFP8。** 仅当 MXFP4 目标继续成立时执行；与 TorchAO 使用完全相同的 Flux block Linear 集合、BF16 FSDP 通信和 checkpoint fingerprint。
- **P3：低优先级接 ALTO `MXFP8→MXFP4`。** 复用 ALTO schedule wrapper，在训练 step 开始前选择 effective precision；先关闭 de-oscillation，确认有限值、resume 和短程 loss，再独立研究收敛。

不建议第一版同时引入低精度 FSDP all-gather、扩大 compile 范围、全量 Linear 量化、永久 packed weight checkpoint 或通用 QuantizationManager。bring-up 可临时关闭 compile，但正式性能验收必须恢复当前逐 block `fullgraph=True` 基线。

## 2. 分析基线与可复现性

### 2.1 工作区基线

本次分析使用以下当前工作树：

| 仓库 | 分支/commit | 状态 |
|---|---|---|
| Primus Flux | `feat/zirui/flux-mlperf`，`56a918b8da2323084754f5cdf91864c34465c03f` | Flux 启动、trainer、config、model/tests 等有未提交改动 |
| TorchTitan ROCm/ALTO 实验树 | `dev/zirui/flux-mxfp4-mlperf`，`c779211952cd602332d5563d093d3768d80aa0ce` | Flux MLPerf 脚本和模型/数据/验证相关文件有未提交或 untracked 改动 |
| TorchTitan current | `dev/mlperf-rocm-compile`，`fc49c5e40e8849f6e8038301b024e626f64e683f` | Flux MLPerf 本地改动存在；MXFP8 converter/preset 来自已提交上游历史，当前门禁为 CUDA SM100+ |
| TorchAO | `main`，`e42806ffe52530029d9dcfd8b53f0a69b9e3d4e6` | 工作树干净；当前 dense API 与旧 ROCm/ALTO TorchTitan checkout 不同，`torchtitan-main` 已使用新 wrapper API |
| ALTO | `dev/zirui/flux-fp4`，`161b9152f522f9e1faaf94b5f3058e8cd49f9e3c` | `config_registry.py` 与 precision schedule recipe 有未提交改动 |

因此，旧 ROCm/ALTO Flux MX 结果不能只用 TorchTitan commit 复现。容器启动时执行 `pip install -e /zirui/code/ALTO`，而 ALTO 声明 `torchao="*"`；实际行为至少取决于 TorchTitan/ALTO 工作树、容器 image digest，以及 editable install 当时解析到的 TorchAO wheel/版本。当前 `torchtitan-main` 虽然已有独立的 TorchAO MXFP8 集成，但其 `pyproject.toml` 不固定 TorchAO，仍需记录实际 wheel/commit。

### 2.2 当前 Primus 的正式低精度基线

当前 Primus 未提交改动主要包括：内联 TorchAO tensorwise FP8、对应 model preset/test、Flux 启动/容器透传，以及 trainer 日志/格式相关变化。该路径已经转换精确的 228 个 block Linear，采用 190 个 FP8 wgrad + 38 个 image/text QKV 高精度 wgrad，并完成 MI355X compile、FSDP2、DTCP resume 和一个完整 seed 的 BF16 收敛对齐。它虽然尚未冻结为提交，技术上已是 MXFP8 必须保护的正式回归基线，而不是可忽略的实验干扰项。

依据：`primus/backends/diffusion/models/registrations/flux.py:194-278`、`flux-fp8.md:771-859` 和 `primus-nemo-perf-gap.md:116-134`。

低精度接入前必须冻结同一代码、数据、seed、batch、LR、AC、compile 和并行配置下的三条基线：BF16、native tensor-wise FP8、NeMo delayed FP8。当前记录中 native FP8 只有 seed 10007 的完整收敛证据，10008/10009 仍需完成或补录。升级 TorchAO 以获得 MXFP8 API 后，必须先重新验证现有 tensor-wise FP8 的 module 集合、190/38 wgrad 策略、compile、FSDP2、DTCP、短程数值和性能，不能默认新 TorchAO 对旧路径无回归。

## 3. TorchTitan Flux 的实际启动链

### 3.1 旧 ROCm/ALTO MXFP8 启动链

```text
/shared_nfs/zirui/code/torchtitan/flux-mxfp8-docker.sh
  └─ FLUX_SCRIPT=flux-mxfp8.sh
     └─ run_with_docker.sh
        ├─ 启动 ROCm 7.2 / PyTorch 2.12 dev 镜像
        ├─ 挂载 /shared_nfs 和 /zirui
        ├─ pip install -e /zirui/code/ALTO
        └─ flux-mxfp8.sh
           └─ FLUX_CONFIG=flux_schnell_full_mxfp8_mlperf
              └─ flux-mlperf.sh
                 └─ flux.sh
                    └─ MODULE=alto.models.flux
                       └─ run_train.sh
                          └─ torchrun -m torchtitan.train
```

关键文件：

- `/shared_nfs/zirui/code/torchtitan/flux-mxfp8-docker.sh:3`
- `/shared_nfs/zirui/code/torchtitan/run_with_docker.sh:5-46`
- `/shared_nfs/zirui/code/torchtitan/flux-mxfp8.sh:3-8`
- `/shared_nfs/zirui/code/torchtitan/flux-mlperf.sh:5-58`
- `/shared_nfs/zirui/code/torchtitan/run_train.sh:28-57`
- `/shared_nfs/zirui/code/ALTO/alto/models/flux/config_registry.py:199-201`

`flux_schnell_full_mxfp8_mlperf()` 最终把已有 converter 替换为 ALTO 的 MXFP8 recipe，所以这是单阶段全程 MXFP8，不是两阶段训练的 stage 1 自动入口。

### 3.2 当前 TorchTitan-main 原生 MXFP8 路径

`/shared_nfs/zirui/code/torchtitan-main` 已包含两组 Flux preset：

- `flux_schnell_mxfp8`；
- `flux_dev_mxfp8`。

二者开启 model compile，并通过 `MXFP8Converter` 选择 `double_blocks`、`single_blocks`、`img_in`、`txt_in`、`time_in`、`vector_in` 和 `final_layer`。converter 在 meta model 构建后、activation checkpoint/compile/FSDP2 前调用 TorchAO：

```text
MXFP8TrainingRecipe("mxfp8_rceil")
  → MXFP8TrainingOpConfig.from_recipe()
  → quantize_(model, filter_fn=...)
  → MXFP8TrainingWeightWrapperTensor
```

关键文件：

- `/shared_nfs/zirui/code/torchtitan-main/torchtitan/models/flux/config_registry.py:273-316`；
- `/shared_nfs/zirui/code/torchtitan-main/torchtitan/components/quantization/mx.py:25-128`；
- `/shared_nfs/zirui/code/torchtitan-main/torchtitan/trainer.py:245-279`；
- `/shared_nfs/zirui/code/torchtitan-main/torchtitan/models/flux/parallelize.py:43-64`。

这条路径提供了有价值的上游集成参考，但有四个边界：

1. `run_flux_test.sh` 默认选择 `flux_schnell_mlperf_preprocessed`，没有启用 converter；现成 MXFP8 preset 也不是 MLPerf preprocessed 配方。
2. 当前 hardware guard 仅检查 CUDA capability `>= 10.0`，文档目标是 NVIDIA SM100+（B100/B200），不是所谓“AM100+”，也不是当前 MI355X 证明。
3. FQN 使用子串匹配，范围比 Primus 已验证的 228 个 block Linear 更大，不能直接用于公平 A/B。
4. 当前 TorchTitan unit/integration tests 没有启用 Flux MXFP8 preset；已合入配置不等于完成数值、FSDP、checkpoint 或 convergence 验证。

`torchtitan-main` 没有 Flux MXFP4 converter、recipe 或训练测试。仓库中的 MXFP4 只用于 GPT-OSS Hugging Face checkpoint 读取并反量化，不能作为 Flux MXFP4 训练参考。

### 3.3 旧 TorchTitan 自带 MX converter 与 ALTO Flux 路径的区别

TorchTitan 自带通用实现位于：

- `/shared_nfs/zirui/code/torchtitan/torchtitan/components/quantization/mx.py:25-121`

它的行为是：

- 检查 `torchao`；
- 要求 CUDA SM100+ 或 ROCm gfx950+；
- 从 `torchao.prototype.mx_formats.config.MXLinearConfig` 创建 recipe；
- 用 `torchao.quantization.quantize_()` 把符合条件的 `nn.Linear` 转换为 MXLinear；
- 默认只支持动态 scaling；
- 当前不需要 post-optimizer hook。

但该旧实验树的 Flux 启动脚本使用 `alto.models.flux` 和 ALTO `ModelOptConverter`，并没有走这个 converter。迁移时若改用 TorchAO 通用 converter，工程会更短，但数值、kernel、过滤范围及与现有日志的可比性都会变化，不能称为“复现旧 ALTO Flux MXFP8”。

此外，旧 ROCm/ALTO 实验 checkout 的 `MXLinearConverter` 只实现 `convert()` 和 `post_optimizer_hook()`，而新的 `ModelConvertersContainer` 还会调用 `pre_step()`、`post_initialization()` 和 `finalize()`。因此它不仅有 TorchAO API 过期问题，还有 converter protocol 兼容缺口；不能理解为“固定旧 TorchAO wheel 就可直接使用”。

### 3.4 TorchAO API 代际与 ROCm 支持边界

旧 ROCm/ALTO 实验树中的 TorchTitan `MXLinearConverter` 引用的是旧 TorchAO API：

```text
MXLinearConfig.from_recipe_name("mxfp8_cublas")
  → torchao.quantization.quantize_()
  → MXLinear / MXTensor
  → torch._scaled_mm
```

该旧 converter 自己只负责 capability guard、Linear filter 和 converter 生命周期；MXFP8 block quantization、E8M0 scale、cast kernel、FWD/dgrad/wgrad 和 GEMM 均在 TorchAO/PyTorch。当前 `torchtitan-main` 已改用上一节所述的统一 `MXFP8TrainingOpConfig` wrapper，不应再把旧 API 描述成 TorchTitan current 状态。

TorchTitan commit `a25dd8f8`（`[ROCm] Support mxfp8 on gfx950`）把 dense Linear guard 从仅 SM100 改为：

```text
CUDA capability >= 10.0
或 ROCm capability >= 9.5
```

该提交明确依赖当时的 TorchAO ROCm enablement，但当前统一 converter 已移除该 ROCm guard。需要同时注意：

- 旧 TorchTitan dense `MXLinearConverter` 允许 gfx950，当前 `MXFP8Converter` 只允许 CUDA SM100+；
- 代码没有写死所谓 “M100” 产品名；NVIDIA 侧按 compute capability `>= 10.0` 判断，B100/B200/GB200 是否可用最终还取决于 TorchAO/PyTorch kernel；
- 旧 TorchTitan `MXGroupedMMConverter` 只有 SM100 guard，不能把历史 dense ROCm 支持外推到 MoE grouped-MM；
- `mxfp8_cublas` 是历史 recipe 名。旧实现的 GEMM preference 为 `AUTO`，最终调用 `torch._scaled_mm`；具体落到 NVIDIA 或 ROCm backend 由 PyTorch build 决定，并非 TorchTitan 直接调用 cuBLAS；
- `mxfp8_dim1_cast_kernel_choice=triton|cuda|torch` 只选择列向/反向所需的量化 cast，不选择 GEMM。其中 `cuda` 是明确 NVIDIA-only 的 TorchAO CUDA extension，`triton` 可覆盖 gfx950，`torch` 是较慢的通用表达式。

当前本地 TorchAO main（`e42806ffe`）已经删除 `MXLinearConfig` 和 `mxfp8_cublas` 训练入口，当前 dense 能力至少有两条：

1. 直接继承 `nn.Linear` 的 `MXFP8Linear`，forward 显式保留 bias；
2. `MXFP8TrainingOpConfig` + `MXFP8TrainingWeightWrapperTensor` 的通用 parameter-wrapper/dispatch 路径。

通用 wrapper 与 ALTO 的设计更接近，但当前 module-level `quantize_(..., filter_fn=...)` 会在未指定 parameter FQN 时包装匹配 Linear 的所有本地 Parameter，而 MXFP8 wrapper 的 Linear 分支没有加回 bias。对 Flux 中的 biased Linear，这会造成语义和梯度错误。wrapper 路径还依赖最新 PyTorch nightly 的内部 DTensor 符号，并不是当前 P1 的安全默认选择。

因此，Primus P1 应先做一个短实现 gate：验证 wrapper 是否已经做到“只转换 weight + 正确 bias 转发”；未满足时采用直接 `MXFP8Linear`。两条路径都必须通过非零 bias、bias gradient、state dict、FSDP2 和 compile 测试。无论选择哪一条，当前底层 `_to_mxfp8_then_scaled_mm()` 仍固定：

```text
dim0 cast = TRITON
dim1 cast = CUDA
```

因此在 gfx950 上，forward 所需 dim0 cast 可能可用，但 backward 的 dgrad/wgrad 相关 dim1 cast 会进入 NVIDIA-only 路径。TorchAO 已有 gfx950 Triton/FlyDSL cast 实现和部分 kernel 测试，但默认 dense training 路径尚未完整接通它们。

结论：正式方案不长期 pin 已删除的旧 API。P1 先验证当前上游 weight-wrapper 的 weight-only、bias、state-dict 和 FSDP 语义；不满足时采用直接 `MXFP8Linear`。无论哪条路径，都必须先在 TorchAO 上游或固定 patch 中完成 ROCm dim1 分派，并实测 `torch._scaled_mm` 的 E4M3 + E8M0 block-scale native 路径。该工作不应在 Primus 内通过 monkey patch 隐藏。

支持状态必须分级表述：

| 能力 | 状态 | 结论 |
|---|---|---|
| 旧 TorchTitan dense converter 接受 gfx950 | 历史代码明确允许 | 只代表旧 API/版本组合，不代表 current converter |
| 当前 TorchTitan-main converter | 仅 CUDA SM100+ guard | 不能在 MI355X 直接启用 |
| 旧 TorchAO `MXLinearConfig` ROCm recipe | 特定历史版本代码允许 | 可作为历史参考，不作为长期 Primus 基线 |
| 当前 TorchAO gfx950 Triton/FlyDSL cast | 有实现和部分 kernel 测试 | 基础量化能力存在 |
| 当前 TorchAO dense FWD/BWD on gfx950 | 尚未完整接通 | dim1 默认 CUDA，不能直接使用 |
| gfx950 `torch._scaled_mm` native MXFP8 | 待微基准证明 | 不能从 capability guard 推断 |
| 当前本仓 Flux ROCm MXFP8 长跑 | 有 | 属于 ALTO，不是 TorchAO provider |
| TorchAO MXFP8 + Primus Flux + gfx950 | 单算子、compile、真实 Flux GEMM shape 和 228-layer 构建已通过 | 完整训练、FSDP2、DTCP、性能和收敛仍待验证 |

### 3.5 TorchAO 与 ALTO wrapper 对比

| 维度 | TorchTitan 所接旧 TorchAO MXFP8 | 当前 TorchAO dense 路径 | ALTO MXFP8/MXFP4 wrapper |
|---|---|---|---|
| API | `MXLinearConfig` / `MXLinear` | 直接 `MXFP8Linear`；另有当前 TorchTitan 使用的通用 weight wrapper，P1 先做语义 gate | ALTO `TrainingWeightWrapper` + recipe |
| 转换对象 | `nn.Linear` 转为 MXLinear | direct 方案替换 Linear；wrapper 方案替换 Parameter，P1 gate 后二选一 | 默认替换 Linear 的 weight Parameter |
| 高精度主权重 | 保留 | `MXFP8Linear.weight`；wrapper 为 `wrapper._data` | `wrapper._data` |
| MXFP8 data / scale | E4M3 / E8M0 | E4M3 / E8M0 | E4M3 或 E5M2 / E8M0 |
| block scaling | 主要 1×32 | 1×32 | 1×32 或 32×32；Flux recipe 对 weight 使用 2D block |
| FWD/dgrad/wgrad | MXFP8 | MXFP8；wgrad 可选高精度；当前 dim1 仍 CUDA-only | MXFP8 或 MXFP4；可选 SR/DGE/Hadamard 等 |
| FSDP 通信 | 高精度 mixed-precision 参数 | 直接 Linear 走标准 FSDP BF16 all-gather；wrapper 候选有高精度 hook | wrapper hook，默认高精度 all-gather |
| step schedule | 无 | 无 | 支持 BF16/MXFP8/MXFP4/NVFP4 |
| MXFP4 | 当前 TorchTitan converter 未使用 | 当前通用 training config 不提供对等 schedule | 原生 E2M1 packed + E8M0，支持 scale selection/de-oscillation |
| ROCm | 旧特定版本允许 gfx950 | cast 基础能力存在，但 dense dim1 默认仍是 CUDA | gfx950/CDNA4 是主要原生目标 |
| compile | 上游证据较强，TorchTitan 建议开启 | 有 compile 基础测试，ROCm dense 路径仍需验证 | custom op/fake/schedule 覆盖不足，bring-up 关闭，交付前验证开启 |
| 本仓 Flux 实跑证据 | 未发现 | 未发现 | 有 ROCm 长跑日志，但不覆盖新的两阶段和 Primus FSDP2 |

两者共同点是动态量化低精度 GEMM、高精度 master weight 和高精度 FSDP 通信。主要差异不是“是否属于 MXFP8”，而是 API 代际、scale granularity、ROCm kernel、schedule/MXFP4 扩展和验证成熟度。

## 4. ALTO 模型转换和训练生命周期

### 4.1 配方注入

ALTO Flux 配置通过 `_with_recipe()` 注入一个 `ModelOptConverter`：

- `/shared_nfs/zirui/code/ALTO/alto/models/flux/config_registry.py:54-81`
- `/shared_nfs/zirui/code/ALTO/alto/components/converter.py:19-63`

`ModelOptConverter` 负责：

- 读取 recipe；
- 模型构建后执行 modifier conversion；
- 每个训练 step 前调用 modifier `pre_step`；
- optimizer step 后调用 modifier hook；
- 初始化和结束阶段调用对应 hook。

### 4.2 转换顺序

TorchTitan trainer 中的关键顺序是：

```text
meta device 构建模型
  → model converter 替换/包装 Linear weight
  → activation checkpoint / context parallel / FSDP2
  → materialize + 初始化权重
  → converter post-initialization
  → 创建 optimizer
  → 创建 checkpoint manager
```

对应 `/shared_nfs/zirui/code/torchtitan/torchtitan/trainer.py:268-457`（conversion、parallelize/materialize、post-initialization、optimizer 和 checkpoint manager）。

这保证：

- wrapper 在 FSDP2 前存在；
- optimizer 从一开始就看到稳定的 Parameter；
- 阶段切换不需要改变 Parameter identity；
- FSDP 可以使用 tensor subclass 的 extension hook。

### 4.3 Linear 选择与参数包装

ALTO `LowPrecisionTrainingModifier`：

- 按 target/ignore 选择 module；
- 对 `nn.Linear.weight` 调用 `swap_params()`；
- bias 保持原样；
- weight 任一维为 32 时直接跳过；
- 用 tensor subclass 包装底层高精度 `_data`，而不是永久把参数存成 4/8 bit。

关键位置：

- `/shared_nfs/zirui/code/ALTO/alto/modifiers/lpt/base.py:228-259`
- `/shared_nfs/zirui/code/ALTO/alto/kernels/dispatch/conversion.py:20-122`
- `/shared_nfs/zirui/code/ALTO/alto/kernels/dispatch/tensor.py:78-178`

## 5. ROCm MXFP8 实现

### 5.1 Recipe

当前 MXFP8 recipe：

- `scheme: mxfp8_e4m3`
- target：`Linear`
- 使用 2D block weight scaling，activation 不使用 2D block；
- 关闭 Hadamard、DGE 和 gradient stochastic rounding；
- 排除 time/vector embedding、modulation 和 final layer；
- 当前 `img_in`、`txt_in` 排除项被注释，因此会尝试转换，仍受 shape filter 约束。

见 `/shared_nfs/zirui/code/ALTO/alto/models/flux/configs/mxfp8_recipe.yaml:1-22`。

### 5.2 Forward / backward 数据边界

MXFP8 Linear 的语义是动态量化 GEMM：

```text
高精度 activation + 高精度 master weight
  → 按 32 元素 block 动态量化为 FP8 data + E8M0 scale
  → MXFP8 GEMM，FP32 accumulate
  → 输出回到输入的 BF16/FP32 dtype
```

forward：

- activation 和 weight 每次执行时动态量化；
- data 使用 E4M3/E5M2；
- scale 使用 E8M0；
- block size 为 32；
- 不维护永久 FP8 weight copy。

backward：

- `grad_output` 也动态量化；
- gfx950 native 路径的 dgrad 和 wgrad 走 MXFP8 GEMM；非 gfx950 是 QDQ 后的高精度 dot，不代表原生 MXFP8 计算；
- 返回给 autograd/optimizer 的 gradient 仍是高精度 dtype。

关键实现：

- `/shared_nfs/zirui/code/ALTO/alto/kernels/mxfp8/mxfp8_linear.py:299-439`

### 5.3 ROCm 硬件路径

ALTO 的 native 判定是 HIP + `gfx950`：

- gfx950 使用 Triton `tl.dot_scaled(..., "e4m3"/"e5m2")`；
- 非 gfx950 可能回退为反量化到 FP32 后普通 dot，只能用于功能调试，不能代表原生 MXFP8 性能。

因此首个目标硬件应固定为 MI355X/gfx950。必须把“能运行”和“使用 native MX tensor core”分别验收。

### 5.4 已有证据边界

历史日志 `/shared_nfs/zirui/code/torchtitan/flux-mxfp8-007-resume-11000.log` 提供了单阶段 ALTO MXFP8 的实跑证据：`:576-579` 从 `checkpoint/step-11000` 加载并从 step 11001 继续，`:2456` 到达 step 34681，`:2489` 记录 `Training completed`。该日志使用 `torchao==0.17.0` 和 editable ALTO，但未记录可复现的 TorchTitan/ALTO commit；数据、batch、activation checkpoint 和总步数也不同于当前 `flux-mxfp8-docker.sh` 默认链。因此它只能证明那次历史工作树组合可恢复并长跑，不能证明当前配置或 Primus provider。

但这不能自动证明：

- Primus FSDP2 checkpoint 兼容；
- 当前新 MLPerf precomputed data 路径；
- full/partial activation checkpoint；
- Primus 容器中的 PyTorch/torchao/Triton 版本；
- MXFP8→MXFP4 阶段切换；
- 与 BF16 相同的 MLPerf 收敛。

## 6. MXFP4 与两阶段训练

### 6.1 为什么不是端到端 FP4

MXFP4 更准确的描述是“使用动态 MXFP4 GEMM 的混合精度训练”，原因如下：

| 部分 | 实际精度/行为 |
|---|---|
| master weight | BF16/FP32 `_data`，每次 GEMM 前临时量化 |
| activation / weight GEMM input | E2M1 packed data + E8M0 block scale |
| GEMM accumulate | gfx950 native `tl.dot_scaled` 路径使用 FP32 accumulator；fallback 不保证 |
| Linear output | 转回 BF16/FP32 |
| dgrad / wgrad | gfx950 native 路径可走 MXFP4 GEMM；fallback 为高精度普通 matmul，最终 gradient 回原始 dtype |
| bias | 高精度加法 |
| optimizer / moments | 高精度 AdamW |
| FSDP all-gather | 按 mixed precision policy 通常为 BF16 |
| FSDP reduce | 当前为 FP32 |
| norm/activation/residual/loss | BF16/FP32 |
| checkpoint | 高精度模型和 optimizer/train state，不是纯 packed FP4 权重 |

核心实现：

- `/shared_nfs/zirui/code/ALTO/alto/kernels/fp4/mxfp4/mxfp_linear.py:258-540`
- `/shared_nfs/zirui/code/ALTO/alto/kernels/dispatch/tensor.py:180-233`

### 6.2 当前配方与目标顺序不一致

当前未提交工作树 `/shared_nfs/zirui/code/ALTO/alto/models/flux/configs/lpt_precision_schedule_recipe.yaml:29-36` 是：

```text
[1, 10753)  MXFP4
[10753, ∞)  BF16
```

ALTO HEAD `161b915` 中对应边界仍是 52224，说明 10753 是当前未提交实验值，必须作为 recipe fingerprint 的一部分记录。历史文件还配置过 MXFP4→MXFP8；其中裸 `mxfp8` 会在 schedule validator 中规范化为 `mxfp8_e4m3`，静态上可以进入 MXFP8 dispatch，但未发现实际跨越切换点的训练日志。用户本次要求的是：

```text
[1, switch_step)  MXFP8 E4M3
[switch_step, ∞)  MXFP4
```

当前未提交 recipe 还把 `deosc_step` 设为 2059；该值同样是实验配置而不是目标两阶段方案的默认值。P3 首版继续关闭 de-oscillation。

因此迁移计划必须把 `mxfp8_e4m3 → mxfp4` 当成**新目标 recipe**，不能声称照搬当前或历史配置即可完成。

### 6.3 最轻量的阶段切换机制

ALTO 的 `MXFP4TrainingWeightWrapperTensor` 可以按 effective precision 分派到：

- BF16；
- MXFP4；
- MXFP8 E4M3/E5M2；
- NVFP4。

关键位置：

- schedule step：`/shared_nfs/zirui/code/ALTO/alto/kernels/dispatch/tensor.py:48-62`
- Linear 分派：`/shared_nfs/zirui/code/ALTO/alto/kernels/dispatch/tensor.py:305-367`
- 每 step 更新：`/shared_nfs/zirui/code/ALTO/alto/modifiers/lpt/base.py:302-325`

所以目标 recipe 可保持基础 `scheme: mxfp4`，只把 schedule 配成先 MXFP8 后 MXFP4。模型构建时只包装一次；切换时只改变本 step 的 dispatch 结果。

这比“先跑一个 MXFP8 进程，再启动一个 MXFP4 进程”更简单可靠，因为它天然共享：

- 同一份 master weight；
- 同一 optimizer 和 moments；
- 同一 LR scheduler；
- 同一 FSDP 分片；
- 同一 global step 和 dataloader state；
- 同一 checkpoint 目录。

### 6.4 Step 语义

为了与 TorchTitan 行为对齐，Primus 应在每个 optimizer update 的 forward 之前设置：

```text
schedule_step = global_step + 1
```

schedule 区间采用 `[start_step, end_step)`：

- stage 1：step `1 ... switch_step-1` 使用 MXFP8；
- stage 2：从 step `switch_step` 的 forward 开始使用 MXFP4。

有 gradient accumulation 时，一个 optimizer update 的所有 micro-batch 必须看到同一个 `schedule_step`。不能在一组 micro-batch 中途切换精度。

### 6.5 Checkpoint 恢复

第一版无需额外保存 `precision_phase`。Primus 已保存 `global_step`，schedule 又是静态 config，因此 resume 后从：

```text
restored_global_step + 1
```

即可推导下一次 forward 的阶段。额外保存 phase 会形成两份可能不一致的状态，不符合轻量原则。

需要在 DTCP metadata 中保存并在 resume 时校验 precision policy/switch step 的规范化配置或指纹；日志只能帮助观察，不能替代恢复校验。如果恢复时 schedule 配置改变，应 fail fast，而不是静默进入不同阶段。

### 6.6 De-oscillation 风险

ALTO 的 FP4 weight de-oscillation 会维护额外 optimizer state，并可能把 master weight 吸附到 FP4 bin center。当前 hook 的 enable 条件主要取决于 `deosc_step`，没有充分按 effective precision 隔离。

对目标 `MXFP8 → MXFP4`：

- 第一版应关闭 de-oscillation；
- 第二轮实验只能在 `switch_step` 之后启用；
- 最好增加 `effective_precision == mxfp4` 的显式 gate；
- 必须单独比较启用/关闭时的收敛和 optimizer checkpoint。

这部分是数值算法实验，不应和基础接入绑在同一个首版 diff 中。

## 7. Primus Flux 当前接入边界

### 7.1 启动与配置链

```text
local_runs/run_flux_mlperf.sh
  → torchrun -m primus.cli.main train pretrain
  → YAML/preset/env 解析
  → DiffusionArgBuilder
  → DiffusionPretrainTrainer
  → build_flux_model()
  → FluxForTraining
  → FSDP2Trainer
  → AdamW + DTCP checkpoint
```

启动入口：`local_runs/run_flux_mlperf.sh:1-269`。

当前 diffusion backend 没有 MXFP8/MXFP4 config 和 model-converter 生命周期。仓库里已有低精度实现主要属于 Megatron、Megatron-Bridge 或 TorchTitan backend，不能直接认为原生 Flux 已支持。

### 7.2 模型构建

`primus/backends/diffusion/models/registrations/flux.py:149-237` 当前顺序为：

```text
创建 Flux DiT
  → init_weights()
  → 加载 pretrained DiT 权重
  → 构造 pipeline / FluxForTraining
  → 返回 trainer
```

最合适的转换点是：

```text
加载 pretrained DiT 权重之后
  → 应用 Flux MX precision converter
  → 构造 FluxForTraining
  → FSDP2 / optimizer
```

这样可以：

- 继续用原始 `nn.Linear` key 加载 pretrained checkpoint；
- 保证转换发生在 FSDP2 和 optimizer 之前；
- 不侵入 flow-match pipeline 和 loss；
- BF16 默认路径完全不变。

### 7.3 FSDP2

Primus 当前 FSDP2：

- 模型先移到 GPU；
- 冻结非训练参数；
- 要求 FSDP 包装前参数报告为 FP32；
- forward param dtype 为 BF16；
- gradient reduce dtype 为 FP32；
- 先显式 shard `img_in,time_in,vector_in,txt_in,final_layer`（其中 `final_layer` no-reshard），再逐 transformer block，最后 root fully shard。

关键位置：

- `primus/backends/diffusion/trainers/fsdp2.py:89-180`

ALTO tensor wrapper 的外层 dtype 与底层 master weight 一致，并实现 composable FSDP pre/post all-gather hook，理论上能保持 BF16 all-gather 后重新构造 wrapper。但这属于版本敏感点，必须用 Primus 目标 PyTorch build 做实测，不能仅凭 TorchTitan 环境结论放行。

### 7.4 Optimizer 与训练步

Primus 当前在 FSDP2 后收集稳定参数并创建 optimizer：MLPerf 路径使用 AdamW；非 MLPerf 且设置 `FP32_MASTER_WEIGHTS=1` 时可能切换为 `AdamWFP32State`。低精度 provider 不应改变既定 optimizer 选择；optimizer 类型和 master-weight policy 应进入 resume fingerprint。

训练 update 的关键位置：

- `primus/backends/diffusion/trainers/base.py:975-1017`

两阶段只需要在 forward 前设置 schedule step。不要在 `optimizer.step()` 后替换 module/Parameter，也不要让 precision 状态进入 optimizer 类。

### 7.5 Checkpoint

当前 `dtcp_full` 会保存 model、optimizer、scheduler、global step 和 RNG 等训练状态。建议低精度首版：

- 只支持 `dtcp_full` 作为完整 resume 验收路径；
- 不保存临时量化 activation、packed cache、Triton/compile cache；
- 不新建一套低精度 checkpoint 格式；
- `dit_only`/safetensors 导出需要单独验证 tensor subclass 是否会被正确 unwrap，不能默认宣称兼容。

## 8. 推荐的双 provider 架构

目标架构允许当前 TorchAO main 与 ALTO 在同一 Primus 版本中二选一，但按 P1→P2 顺序实现：P1 只接 TorchAO；仅当 MXFP4 目标继续成立时才增加 ALTO。单次 run 固定一个 provider，不保留旧 TorchAO `MXLinearConfig` 兼容分支。

### 8.1 Provider 定位

- `torchao`：上游兼容基线。P1 先验证当前 TorchTitan 所用 weight-wrapper 的 weight-only、bias/parameter、state-dict 和 FSDP 语义；不满足时采用直接 `MXFP8Linear`。两者都必须先补齐或消费 gfx950 dim1 Triton/FlyDSL 路径。
- `alto`：gfx950 原生性能和 MXFP4 schedule 路径。只使用无副作用的 LPT core，不导入 `alto.models.flux`、完整 `ModelOptConverter` 或 TorchTitan trainer。
- provider 只负责“如何转换已选中的 Linear”；Flux 层选择、转换时机、checkpoint policy 和日志由 Primus 统一控制。
- 同一 run 中 provider 不可切换。`MXFP8→MXFP4` 是 ALTO provider 内部的 precision schedule，不是从 TorchAO 切换到 ALTO。

### 8.2 单一、轻量的转换入口

建议把当前未提交、内联在 `build_flux_model()` 中的 TorchAO tensorwise FP8 逻辑收进一个 Flux 专用文件：

```text
primus/backends/diffusion/models/flux/precision.py
```

只提供普通函数，不建立 `QuantizationManager`、provider 基类、registry 或插件体系：

```text
validate_precision_config()
select_flux_block_linears()
apply_flux_precision()
  ├─ _apply_torchao()      # P1
  └─ _apply_alto()         # 按需在 P2 增加
set_precision_step()      # 仅 ALTO schedule 使用
build_precision_fingerprint()
```

两个 provider 必须共享一份精确 FQN 集合。转换顺序固定为：

```text
构建并初始化 DiT
  → 加载 pretrained dense 权重
  → 应用一个 provider
  → 构造 FluxForTraining
  → gradient checkpoint / FSDP2
  → optimizer
```

该顺序保证 optimizer 和 FSDP 从一开始就看到最终 Parameter/wrapper，也避免在 resume 或 forward 中重复扫描、重复包装。

### 8.3 ALTO import 隔离

当前正常 `import alto...` 会先执行 `alto/__init__.py`，eager import components/modifiers/models；models import 会执行全局 patch。converter/LPT/dispatch 还直接依赖 TorchTitan protocol、config、checkpoint、attention、MoE 和 logger，而 ALTO 包本身未完整声明 TorchTitan 依赖。

因此 ALTO provider 的前置条件是：

- package `__init__` 无副作用/lazy；
- LPT core 可独立 import；
- 未选择 ALTO 时绝不 import ALTO；
- Primus 不复制 ALTO kernel/wrapper 源码，也不通过 monkey patch 绕过依赖。

否则“双 provider A/B”会被 ALTO 的全局副作用污染，结果不可解释。

### 8.4 Checkpoint 与 provider 切换边界

未来 DTCP 设计保存 canonical precision fingerprint，至少包含：

- provider 与构建时注入的 immutable implementation ID（不在运行时猜测 git commit）；
- recipe、data/scale format、block/scaling mode；
- wgrad precision；
- 统一 layer profile 和实际 converted FQN；
- schedule；
- compile 与低精度 all-gather 状态。

规则和当前缺口：

- full resume 只允许相同 provider、相同 recipe 和相同 fingerprint；metadata 必须在 DTCP 写入 model/optimizer 之前预读和校验，当前“load 后再返回 meta”的流程不能实现真正 fail-fast；
- ALTO schedule 的 phase 不重复保存，由恢复后的 `global_step + 1` 推导；
- schedule/fingerprint 不一致直接 fail fast；
- TorchAO ↔ ALTO 未来只允许 model-only warm start：先得到规范 dense state/safetensors，再构建未包装模型、加载权重、应用目标 provider并新建 optimizer；
- 当前 `dit_only`/DTCP 没有 provider-independent canonical dense 导出/加载保证，因此跨 provider warm start 在该能力完成前视为“不支持/待实现”；
- 永不支持跨 provider 完整 optimizer resume。

## 9. 轻量配置设计

当前未提交配置只有 `float8_recipe=tensorwise`，它表示 TorchAO tensorwise FP8，不是 MXFP8；QKV 使用高精度 wgrad 是 `build_flux_model()` 内固定策略，不是公开的 `float8_wgrad_group`。为避免把四种不同能力塞进一个含义模糊的字符串，建议收敛为两个高层字段：

```yaml
model:
  config:
    low_precision_provider: ${FLUX_LOW_PRECISION_PROVIDER:}
    low_precision_recipe: ${FLUX_LOW_PRECISION_RECIPE:}
```

语义和互斥校验：

- 空 provider + 空 recipe：BF16，零行为变化；
- `provider=torchao, recipe=tensorwise_fp8`：现有 TorchAO Float8Linear 路径；
- `provider=torchao, recipe=mxfp8`：当前 TorchAO main MXFP8；具体使用 weight-only wrapper 或直接 `MXFP8Linear` 由 P1 实现 gate 决定；
- `provider=alto, recipe=mxfp8`：ALTO 单阶段 MXFP8；
- `provider=alto, recipe=mxfp8_to_mxfp4`：ALTO 两阶段；
- 其他组合 fail fast；未选 provider 的 options 不得静默生效。

现有 `FLUX_FLOAT8_RECIPE=tensorwise` 已用于性能和收敛基线，P1 必须继续显式映射为 `provider=torchao, recipe=tensorwise_fp8`，不能静默退回 BF16，也不应在 MXFP8 bring-up 同时强制迁移现有 launcher。是否弃用该兼容入口留到 P2 配置面稳定后单独决定。

首版不公开 scale dtype/granularity、block size、E4M3/E5M2、Hadamard、DGE、SR、filter regex、de-oscillation 和低精度 all-gather。这些固定在受版本控制的 provider recipe 中，避免公共配置与实验实现细节耦合。

## 10. Linear 覆盖范围

### 10.1 第一版转换范围

只转换 DiT transformer block 中的大 GEMM：

- `double_blocks.*.{img_attn,txt_attn}.{qkv,proj}`；
- `double_blocks.*.{img_mlp,txt_mlp}.{0,2}`；
- `single_blocks.*.{linear1,linear2}`。

标准 Flux.1（19 个 double block、38 个 single block）应得到 `19×8 + 38×2 = 228` 个候选 Linear。MXFP8 第一版还应继承现有数值策略：190 个非 QKV Linear 使用 MXFP8 wgrad，38 个 image/text QKV 保持高精度 wgrad；全 MXFP8 wgrad 只能作为后续独立消融。统一 profile 还必须定义 shape eligibility：TorchAO MX block 至少要求 32 对齐，当前 Triton dim1 路径对部分维度有更严格的 128 对齐；ALTO 另有“任一 weight 维等于 32 时跳过”的规则。两个 provider 应对相同候选执行共同 eligibility 检查；不满足时默认 fail fast，并把 expected/converted/skipped FQN 与原因写入 fingerprint，不能各自静默跳过后仍宣称公平 A/B。

### 10.2 第一版明确排除

- `img_in`；
- `txt_in`；
- `time_in`；
- `vector_in`；
- modulation/AdaLN Linear；
- `final_layer`；
- encoder/VAE；
- 非 `nn.Linear` 算子。

理由：

- 采用 MXFP8/MXFP4 recipe 的安全交集，减少两阶段前后覆盖范围变化；
- 避免小 shape、conditioning 和输出头的 kernel/收敛敏感性；
- 当前 MLPerf precomputed data 路径本身不需要训练 encoder/VAE；
- 大 block GEMM 才是主要性能收益来源。

扩大到 `txt_in` 或其他层应作为独立消融，不和基础接入合并。

第一阶段范围限定为 MLPerf Flux.1-schnell。虽然 registry 同时公开 Flux.1-dev，但在 Schnell 的依赖、FSDP、resume 和收敛矩阵完成前，不宣称 Dev preset 已支持双 provider。

## 11. 分阶段迁移计划

### Gate 0：TorchAO/ALTO 最小技术探针

目标是在不修改 Primus 训练生命周期、不启动完整收敛训练的前提下，用 MI355X/gfx950 和真实 Flux Linear shape 尽快消除两个最大不确定性：TorchAO 当前 ROCm native 路径是否成立，以及 ALTO 是否能在目标软件栈中进入 `torch.compile(fullgraph=True)`。两个 probe 可以并行执行，但 Gate 0 不建设双 provider、FSDP2、DTCP 或公共抽象。

共同约束：

1. 固定并打印 GPU arch、PyTorch、ROCm、Triton、TorchAO；ALTO probe 额外打印 ALTO immutable identity。禁止依赖未记录的 editable 环境。
2. 使用标准 Flux 的代表性 M/N/K 和非零 bias，至少覆盖 MLP、attention projection、QKV 及 32/128 对齐边界；先做单 Linear eager FWD/BWD，再做 `torch.compile(fullgraph=True)`。
3. 对 output、input grad、weight grad、bias grad 做 finite 和 BF16 reference 误差检查；分别计时 cast、FWD、dgrad、wgrad，warmup 后报告稳定区间，不用单次时延下结论。
4. 必须用 profiler/kernel trace 证明 gfx950 native MXFP8 GEMM；emulation、QDQ 后高精度 dot 或静默 BF16 fallback 均判为未通过。
5. probe 脚本和结果保存在新的 `dev/zirui/flux-mxfp8` worktree；不修改仍在做 NeMo FP8 性能对齐的原工作树。可复制原工作树 `local_runs/` 中有复用价值的未跟踪脚本，但只复制明确需要的文件，不复制历史输出、trace 或 rank log。

TorchAO probe 额外验证：

- 当前 `MXFP8Linear` 与 `MXFP8TrainingOpConfig` API 是否可导入；
- dim0/dim1 cast 是否都能在 gfx950 选择 Triton/FlyDSL，不能只删除 TorchTitan 的 SM100 guard；
- E4M3 + E8M0 block scale 的 `torch._scaled_mm` 是否为 native；
- wrapper 是否只包装 weight并正确保留 bias/bias gradient；失败时再验证直接 `MXFP8Linear`。

ALTO probe 额外验证：

- 不通过 `alto.models.flux` 或完整 `ModelOptConverter`，最小 LPT/kernel import 是否仍被 `alto/__init__.py`、TorchTitan 类型或全局 patch 阻塞；
- `convert_to_mxfp8`、`blockwise_mxfp8_gemm` 的 FWD/dgrad/wgrad 是否均为 gfx950 native；
- `allow_in_graph`、custom-op fake/meta 和 tensor wrapper 是否足以支持目标 block 的 `fullgraph=True`，不能把存在装饰器等同于集成通过。

Gate 0 决策：

- TorchAO native eager + compile 均通过：P1 采用 TorchAO，ALTO 暂不进入 Primus。
- TorchAO 仅缺小范围 ROCm dim1 路由且可用固定上游 patch 修复：仍采用 TorchAO P1，并先提交/固定该 patch。
- TorchAO 缺少 gfx950 native GEMM或修复范围不可控，而 ALTO native eager + compile 通过：ALTO 可前移为版本固定的单阶段 MXFP8 战术 P1；TorchAO 保持长期主线候选。
- 两者都未通过：停止框架接入，先修外部 kernel/compile blocker；不得以 fallback 性能宣称 MXFP8 已支持。

#### Gate 0 首轮结果（单算子，未完成）

首轮在 `crsuse2-m2m-234` 的单张 MI355X 上执行。最终共同验证栈使用镜像 `unifiedtrainingdockers.azurecr.io/utd/ci@sha256:872976f64a1265f95fbceeb46de4ce31fd2d4d29da2c48532348463f4af32848`，环境为 PyTorch `2.12.0+rocm7.14.0`、ROCm `7.14.60850`、Triton `3.7.1`。另用历史 TorchTitan 镜像确认了旧 PyTorch 与 TorchAO main 的 import API 不匹配，不把该结果混入 kernel 判断。探针与原始日志位于：

- worktree 脚本：`local_runs/probe_mxfp8.py`；
- TorchAO 默认、FlyDSL、FLOOR 和 emulation 结果/日志：`/shared_nfs/zirui/runs/mxfp8_gate0/torchao*.{json,log}`；
- ALTO 新栈结果/日志：`/shared_nfs/zirui/runs/mxfp8_gate0/alto-new-stack.{json,log}`。

当前结果：

1. **TorchAO 默认 dense 路径被 CUDA dim1 cast 阻断；显式 FlyDSL 能打通 gfx950 eager/compile，但当前 native 结果数值不正确。** 新 PyTorch 已可导入 TorchAO `e42806ffe` 的 MXFP8 路径；默认 `M,K,N=(64,3072,3072)` 在 eager 和 compile 中都报 ``mxfp8_quantize_cuda` needs ... CUDA capability 10.0+`，证明仅删除 TorchTitan SM100 guard 无效。探针随后绕过默认值，直接给同一个 `mx_mm` 选择 `MXFP8Dim1CastKernelChoice.FLYDSL`：`M=64` 的 wgrad 因 native block-scaled GEMM 要求 reduction K 为 128 倍数而 fail fast；改为 `(128,3072,3072)` 后，forward、dgrad、wgrad、bias grad 在 eager 与 `fullgraph=True` 均 finite，profiler同时记录 FlyDSL dim1 cast、dim0 cast 和 `aten::_scaled_mm`。但是 RCEIL native 相对 BF16 的 relative-L2 约为 output `0.322`、dgrad `0.313`、wgrad `0.289`，而同一 TorchAO 语义的 emulated 路径约为 `0.027/0.027/0.026`；FLOOR 更差，将 dim0 从 Triton 改为通用 Torch 表达式也不改变 native 误差。这把问题进一步定位到 native block-scale/`torch._scaled_mm` 语义或布局，而不是单纯的 dim0 cast。当前 TorchAO native 路径不能进入 Primus 集成。
2. **ALTO eager native 单算子可运行。** 使用 ALTO `161b9152` 工作树；其 `torchao="*"` 在新镜像中实际解析为镜像已有的 `0.15.0+gite9c7bead9`，再次证明必须固定依赖。相同 shape 的 E4M3 forward、dgrad、wgrad 和非零 bias gradient 均为 finite。`is_cdna4()` 通过，profiler 记录 `_convert_to_mxfp8_kernel.kd`、`alto::blockwise_mxfp8_gemm` 和 `blockwise_mxfp8_gemm_kernel.kd`，因此该次不是静默 BF16/QDQ fallback。该测试直接调用 ALTO MXFP8 autograd function，尚未验证完整 weight wrapper、FSDP 或 Flux block。
3. **ALTO `fullgraph=True` 在新旧两个 PyTorch 栈中均未通过。** 同一算子在 compile tracing 阶段报 `PendingUnbackedSymbolNotFound`，指向 custom-op fake/meta 返回中的未绑定动态符号。存在 `allow_in_graph` 装饰器不足以证明 compile 兼容；在修复 fake/meta 契约前，ALTO 不能作为 Primus 当前 block compile 的正式 provider。完整 editable install 还会因顶层 eager import 拉入本 probe 不需要的依赖，支持先解耦无副作用 LPT/kernel core 的判断。
4. **首轮不做性能结论。** 只测了一个缩小 M 的 shape，且短样本时延波动明显；必须在固定软件栈、compile gate 通过后覆盖真实 Flux shape并分别测 cast/FWD/dgrad/wgrad，才能比较 BF16/TorchAO/ALTO。

首轮因此把选择收敛为两个 blocker：TorchAO 的 ROCm native scaled-mm scale layout，ALTO 的 custom-op fake/meta 与 `fullgraph=True`。后续优先修复了 TorchAO，ALTO 暂不进入 Primus。

#### Gate 0 第二轮结果与最小 P1 接入

1. **TorchAO native 数值问题已定位并修复。** 独立 layout probe 对同一组 MXFP8 qdata/scales 直接调用 gfx950 `torch._scaled_mm`：TorchAO 当前无条件使用 NVIDIA blocked/swizzled scales 时 relative-L2 为 `0.4483`；ROCm 使用 row-major unswizzled scales 时为 `0.0041`。因此问题不是 MI355X GEMM 本身，而是 TorchAO `_addmm_mx_dispatch()` 将 NVIDIA scale layout 用到了 ROCm。
2. **外部 patch 保持最小。** 在独立 `/shared_nfs/zirui/code/ao-mxfp8`、分支 `dev/zirui/mxfp8-rocm`、commit `f9f48318d021c7efda1902951e86bb3810993556` 中只做两项生产修改：HIP 下 `_scaled_mm` 使用 row-major E8M0 scales；MXFP8 dense backward 在 HIP 下默认选择 FlyDSL dim1 cast，CUDA 继续选择原 CUDA cast。没有在 Primus 中 monkey patch TorchAO。
3. **修复后的 native 数值与 emulation 对齐。** `(128,3072,3072)` 的 eager/compile relative-L2 约为 output `0.0272`、dgrad `0.0267`、wgrad `0.0267`；TorchAO emulated 对照约为 `0.0271/0.0266/0.0264`。三个代表性 Flux shape `(256,3072,{3072,9216,12288})` 的 eager 和 `fullgraph=True` 全部通过，output/dgrad/wgrad relative-L2 均约 `0.0257-0.0271`，profiler 记录 FlyDSL、Triton dim0 cast 和 native `aten::_scaled_mm`。
4. **TorchAO 回归测试通过。** 扩展 `test_mxfp8_linear.py` 使 gfx950 运行 AUTO FWD/BWD SQNR、非连续 grad output 和 compile 测试；结果为 `12 passed`。layout probe 与结果保存在 `local_runs/probe_scaled_mm_layout.py` 和 `/shared_nfs/zirui/runs/mxfp8_gate0/`。
5. **Primus 最小适配已落地到独立 worktree。** 配置只接受空配置或 `provider=torchao, recipe=mxfp8`；pretrained load 后把精确的 228 个 Flux block Linear 替换为直接 `MXFP8Linear`，其中 190 个使用 MXFP8 wgrad、38 个 image/text QKV 使用高精度 wgrad；权重、bias 和 state-dict key 保持不变，input/conditioning/final layer 不转换。未引入 manager、registry、tensor wrapper 或 ALTO 生命周期。
6. **Primus 单元测试通过。** 完整 `tests/unit_tests/backends/diffusion/test_flux_backend.py` 为 `24 passed`，覆盖非法配置、10-layer tiny profile、QKV wgrad policy、state-dict/value preservation 和 BF16 默认路径。
7. **两 rank完整 Flux FSDP2、compile 和 AC 已通过。** `local_runs/probe_mxfp8_fsdp.py` 先验证两个 `MXFP8Linear`、BF16 param all-gather、FP32 reduce、fused AdamW 的 2 GPU step；随后 `local_runs/probe_primus_cli.py` 保留每 rank traceback并运行完整 228-layer Flux。compile-off 和 57 个 transformer block `fullgraph=True` compile-on 均完成 forward/backward/optimizer step，loss/grad norm finite；交付配置的 gradient checkpoint ratio `0.25` + compile-on 也通过。代表性一步记录为 loss `1.6148`、grad norm `3.3907`、peak memory `95.29 GB/GPU`；该单步时延包含首次编译和 warmup，不能作为性能结果。
8. **DTCP full save/resume 已通过。** 2 GPU 在 step 1 保存 `checkpoint-1`，新进程从该 checkpoint 恢复 model、optimizer、scheduler 和 global step，明确日志 `Resumed from step 1`，随后完成 step 2（loss `1.6042`）并保存 final checkpoint。当前 checkpoint 尚未加入 provider/fingerprint metadata，因此只证明同一代码/依赖/recipe 的恢复链可用，不允许据此开放跨 provider 或依赖漂移恢复。
9. **短程同口径性能未显示收益。** 2 GPU、local batch 1/GPU、SDPA、FSDP2、block compile、AC ratio `0.25` 下各运行 12 step，排除 step 1 后统计 steps 2-12：BF16 mean/median step time 为 `1.0927/1.0700 s`，MXFP8 为 `1.0991/1.0900 s`，即 MXFP8 mean 慢 `0.58%`、mean throughput 低 `0.14%`；peak allocated memory 都是 `97.97 GB`，MXFP8 steady reserved memory 从 `104.68 GB` 增至 `111.62 GB`；首次 compile/warmup step 从 BF16 `28.33 s` 增至 MXFP8 `112.69 s`。进一步对 `M=256,K=3072,N={3072,9216,12288}` 做 20 次 Linear FWD/BWD 探针，compiled BF16 为 `0.332-0.346 ms`，compiled MXFP8 为 `1.194-1.246 ms`，MXFP8 慢 `3.46-3.76×`；将 `(M,3072,3072)` 的 M 扩到 `512/1024/2048` 后仍慢 `3.52×/3.49×/2.98×`，测试范围内没有 crossover。因此瓶颈明确在动态 cast/GEMM 路径，不只是模型其他算子掩盖收益。结果见 `/shared_nfs/zirui/runs/mxfp8_perf/summary.md`。当前实现只达到功能实验能力，不晋升为性能 recipe。
10. **短程 loss 已开始分叉但尚不能作收敛判断。** 两者 step 1/2 loss 相同到 4 位，step 3 后出现小差异，至 step 12 BF16/MXFP8 分别为 `2.1421/1.9621`，grad norm 也不同；12 step 太短且 batch 太小，不能判断优劣。剩余 P1 工作是固定 TorchAO patch commit/wheel和镜像，profile cast/`_scaled_mm`/compile overhead，验证更大 local batch或 8 GPU 是否有收益，并做至少 100-step loss/grad norm 对照；若仍不优于 BF16和现有 tensor-wise FP8，则只保留实验入口，不进入默认训练配置。

### P0：固定基线和环境能力

目标：冻结可信的 BF16、native tensor-wise FP8、NeMo delayed FP8 三条基线和目标镜像能力矩阵。

计划动作：

1. 固定 Primus、PyTorch、TorchAO commit 和容器 digest；清楚记录仍需保留的本地 patch。ALTO commit/wheel/hash 在进入 P2 时再固定。
2. 用 `local_runs/run_flux_mlperf.sh` 固定：数据、seed、GBS、LR、warmup、AC ratio、compile、reshard、attention backend。
3. 同口径记录 BF16、native tensor-wise FP8 和 NeMo delayed FP8：step time、throughput、peak memory、loss、eval loss、checkpoint/resume；补齐或明确记录 native FP8 尚缺的完整 seed。
4. 在 Primus 目标容器做 import/API capability probe：
   - PyTorch/ROCm/Triton/torchao 版本；
   - gfx950；
   - 当前 TorchAO `MXFP8Linear` 与 `MXFP8TrainingOpConfig` API；
   - TorchAO gfx950 dim0/dim1 cast；
   - gfx950 `torch._scaled_mm` E4M3 + E8M0 native path；
   - composable FSDP extension API。
5. 固定 TorchAO 的 immutable build identity 和安装约束。进入 P2 时再要求 ALTO wrapper 无副作用 import、`tl.dot_scaled` native path、显式可选安装入口和 immutable identity；不能依赖运行时 `pip install -e` 解析任意 `torchao=*`。
6. 安装 MXFP8 所需的新 TorchAO build 后，先回归现有 tensor-wise FP8：228 个 module、190/38 wgrad、block compile、FSDP2、DTCP、短程 loss/grad norm 和性能均不得无解释回退。
7. 功能 bring-up 先关闭 activation checkpoint 和 block compile；进入集成验收后恢复并固定当前 MLPerf ratio=0.25 与逐 block `fullgraph=True`。P4 只研究其他 AC 模式/ratio 或 compile 策略，不重复验证这一固定基线。

公共交付门槛：三条基线可复现、目标容器能力明确、已有未提交基线改动不再漂移、新 TorchAO 未破坏 native tensor-wise FP8。TorchAO ROCm cast/GEMM 证据只阻塞 P1；ALTO LPT core 无副作用 import 只阻塞按需执行的 P2/P3。

### P1：当前 TorchAO main 单阶段 MXFP8

目标：在 Primus 原生 diffusion Flux + FSDP2 上实现可关闭、可 resume 的单阶段 MXFP8 Linear GEMM。

最小实现步骤：

1. 不增加旧 `MXLinearConfig` 兼容分支。先验证当前 wrapper 是否只包装 weight、正确保留非零 bias/bias gradient，并保持 state dict/FSDP 语义；未满足时采用直接 `MXFP8Linear`。
2. 在 TorchAO 上游或固定 patch 中让 dense dim1 cast 在 ROCm 选择 Triton/FlyDSL；不在 Primus monkey-patch。
3. 增加 `low_precision_provider=torchao`、`low_precision_recipe=mxfp8`。
4. 新增一个 Flux 专用 precision 适配文件，职责仅包含：
   - lazy dependency/hardware validation；
   - 选择 transformer block Linear；
   - 在 pretrained weight load 后用 P1 gate 选定的单一路径转换目标 Linear；
   - 记录实际 converted/skipped module 清单。
5. 在 `build_flux_model()` 中、pretrained load 后调用转换。
6. FSDP2 通信、optimizer 和非 Linear 算子保持现有 FP32/BF16 语义。
7. 复用现有 228 个候选和 190 个 MXFP8 wgrad + 38 个 QKV 高精度 wgrad 策略，不复制 TorchTitan 对 input/conditioning/final layer 的宽泛 FQN。
8. BF16/TorchAO tensorwise FP8 默认行为不得被 MXFP8 provider 改变。

验证顺序：

1. 配置解析和 module filter 单测；
2. gfx950 单 Linear FWD/BWD，覆盖非零 bias、bias gradient、32/128 对齐边界，检查 output/grad finite；
3. tiny Flux 单 GPU 2-10 step，先 compile off，再验证 block compile on；
4. 2 GPU FSDP2 2-10 step；
5. 8 GPU 100+ step smoke，使用交付配置的 block compile；
6. 当前 MLPerf `ContiguousDistributedSampler` 路径的 `dtcp_full` save/resume；非 MLPerf 普通 sampler 在补齐 batch offset/state 恢复前不承诺下一步严格连续；
7. 相同配置 BF16、native tensor-wise FP8、MXFP8 的短程 loss/grad norm；
8. 同口径比较 BF16、native tensor-wise FP8、MXFP8 和 NeMo delayed FP8 的 step time、throughput、显存及 profiler；确认 dim1 没有进入 CUDA-only 或软件 emulation，并确认 native MXFP8 GEMM backend；
9. 最后才进行完整 MLPerf 收敛。

P1 验收标准：

- BF16 和 native tensor-wise FP8 默认行为不变；
- converted module 清单稳定且严格为既定 228 个候选，wgrad 策略为 190/38；
- 单/多 GPU forward、backward、optimizer step 有限；
- 当前 MLPerf contiguous-sampler 路径中，FSDP2 save/resume 后下一步 loss 与连续运行在允许误差内；
- 未启用低精度通信；
- native gfx950 kernel 有可验证证据；
- block compile 开启后没有 graph break，性能和收敛口径可与现有基线直接比较；
- MXFP8 至少优于 BF16；若不优于 native tensor-wise FP8且不能提供明确的收敛、显存或可维护性价值，则只保留为实验 recipe，不进入默认训练配置；
- 记录相对 NeMo delayed FP8 的 gap 是否缩小，不能只报告相对 BF16 的收益。

### P2：ALTO 单阶段 MXFP8

前置条件：MXFP4 目标在 P1 结果后仍有明确需求；ALTO LPT core 可无副作用独立 import，且公共 FQN、checkpoint fingerprint 和 FSDP 验证框架已建立。P2 顺序执行，不与 P1 并行扩大初始交付范围。

最小实现步骤：

1. 增加 `low_precision_provider=alto`、`low_precision_recipe=mxfp8`。
2. 复用与 TorchAO 完全相同的 Flux block Linear FQN 集合，不使用 ALTO 自己的另一份 ignore list。
3. 只接 Linear weight wrapper；不启用 attention、LoRA 或 grouped-MM conversion。
4. FSDP all-gather 继续使用 BF16 mixed-precision 参数；不同时引入低精度通信。
5. bring-up 强制 `compile_transformer_blocks=false`；功能稳定后增加 compile-on gate，生产性能比较必须使用与 P1 相同的 block compile 配置。
6. 使用同一套 DTCP fingerprint 和 same-provider full resume 规则。

验收重点：ALTO 未选择时没有任何 import/patch 副作用；两个 provider 转换 FQN 和 190/38 wgrad policy 完全一致；单 GPU、2 GPU FSDP2、compile-on 和 same-provider resume 均通过。

### P3：ALTO 两阶段 MXFP8→MXFP4（备选、低优先级）

定位：这是潜在提升训练效率的实验方案，不是 MXFP8 支持的组成部分或交付前提。现有 ALTO MXFP4/两阶段实验相对 BF16 有较大精度损失，当前没有证据支持将其设为默认训练配方。

前置条件：P2 ALTO 单阶段路径和公共 checkpoint/fingerprint 基础已稳定，MXFP8 主目标已经完成性能与收敛验收，且 MXFP4 单算子和短程训练显示相对 BF16 的精度 gap 有可接受或可改善的趋势；不硬依赖 TorchAO P1 完整收敛。

最小实现步骤：

1. 使用一次性安装的 MXFP4 schedule wrapper；不要在切换点重新包装模型。
2. 新增 `low_precision_recipe=mxfp8_to_mxfp4` 和固定 recipe 中的 `mxfp4_start_step`。
3. 在每个 optimizer update 的 first micro-batch forward 前设置 `global_step + 1`。
   - Primus 薄适配层直接调用 ALTO `set_training_precision_schedule_step(step)`，不复用完整 `ModelOptConverter.pre_step()`；这样避免错误关键字被 `**kwargs` 静默吞掉，也不引入 de-oscillation 所需 optimizer hook。
   - 若 dataloader 长度不能整除 gradient accumulation steps，必须先正确处理 epoch 尾部 micro-batch（执行 update 或显式丢弃并清梯度）；否则 precision step、optimizer step 和 resume offset 会失配。
4. resume 后由恢复的 `global_step` 推导阶段；不新增 phase checkpoint 字段。
5. 第一版关闭 de-oscillation、compile 和低精度 all-gather。
6. 在 DTCP metadata 中保存规范化的实现 ID、precision policy、具体格式 token、schedule 区间和 switch step/fingerprint；恢复时必须先预读并校验 metadata，再加载 model/optimizer。
7. 在切换点记录一次明确日志：step、旧精度、新精度、recipe fingerprint。
8. 分别验证：
   - stage 1 内 save/resume；
   - switch 前一 step save，resume 后跨 switch；
   - stage 2 内 save/resume；
   - gradient accumulation 下不发生 micro-batch 中途切换。

数值验证必须把“能运行”与“能收敛”分开：

- Gate A：单算子 finite；
- Gate B：tiny Flux finite；
- Gate C：100/500/1000 step loss 与 BF16/MXFP8 对照；
- Gate D：固定 eval sample 的趋势；
- Gate E：完整 MLPerf 收敛。

如果与 BF16 有显著 gap，应优先调 stage boundary、层过滤和 FP4 scale selection；不要先把更多系统优化叠加进来。

### P4：后续可选优化

仅在 P1-P3 数值稳定后考虑：

- 两阶段 schedule 的 `torch.compile` 图缓存/重编译优化；
- 其他 activation checkpoint 模式/ratio 组合；
- 扩大 Linear 覆盖范围；
- de-oscillation；
- FP8/MXFP8 FSDP all-gather；
- AITER/Primus-Turbo 统一实现；
- tuned GEMM config；
- 低精度导出格式。

每项应独立实验和独立 diff，避免无法归因。

## 12. 预计文件改动

以下是实施时的目标范围，不是本次已执行改动。

### P1-P3 预计生产代码

| 文件 | 预计改动 |
|---|---|
| `local_runs/run_flux_mlperf.sh` | 增加 provider/recipe 环境变量、日志和 summary 字段 |
| `run_with_docker.sh` | 条件透传低精度环境变量和必要 kernel 配置 |
| `primus/configs/models/diffusion/flux.1_schnell_t2i.yaml` | 将现有未提交 FP8 字段收敛为 provider + recipe |
| `primus/backends/diffusion/models/flux/precision.py`（建议新增） | 共享 FQN、lazy provider import、一次性 convert、fingerprint、schedule step |
| `primus/backends/diffusion/models/registrations/flux.py` | 将内联 tensorwise FP8 移入单一入口；pretrained load 后调用 |
| `primus/backends/diffusion/trainers/base.py` | P3 增加 ALTO schedule step hook；P1/P2 无需动态 hook |
| `primus/backends/diffusion/trainers/fsdp2.py` | 从 P1 起保存/恢复 provider fingerprint；定义 canonical dense 导出边界 |
| `primus/backends/diffusion/distributed/checkpoint.py` | 在写入 model/optimizer 前预读 metadata 并 fail-fast 校验 fingerprint |
| 依赖约束/镜像构建清单 | 固定 PyTorch/TorchAO wheel 与 hash，提供可选 ALTO 安装和 immutable implementation ID |

外部前置改动也属于总范围：TorchAO 至少涉及 direct `MXFP8Linear` 的 ROCm dim1 路由和真实 kernel 测试；ALTO 至少涉及 package lazy import、移除 models 全局 patch 副作用，以及 LPT modifier/dispatch 对 TorchTitan 类型的隔离。这些不是 Primus 内一个 adapter 文件可以替代的工作。

### 测试

| 文件 | 覆盖 |
|---|---|
| `tests/unit_tests/backends/diffusion/test_flux_backend.py` | config、filter、BF16 no-op、非法组合 |
| 建议新增一个小型 precision test | Parameter identity、converted module、schedule 边界、resume step 推导 |
| GPU integration test | gfx950 kernel、FWD/BWD、FSDP2、DTCP resume |

不建议为了单一调用点新增多层 manager/factory/dataclass。一个 Flux precision 模块和两处明确调用足够。

## 13. 复杂度与工作量估计

以下仅是低置信度 order-of-magnitude 估计，不包含排队等待大规模 GPU 的时间，也不能作为排期承诺。TorchAO ROCm GEMM、ALTO 解耦和 metadata 预校验三个 gate 明确后必须重新估算：

| 工作项 | 复杂度 | 估计 | 主要不确定性 |
|---|---:|---:|---|
| P0 环境/基线固定 | 中 | 1-3 人日 | 四个仓库状态与镜像版本 |
| TorchAO main ROCm dim1 接通 | 中高 | 2-5 人日 | Triton/FlyDSL、上游 API、kernel 测试 |
| TorchAO gfx950 原生 GEMM 证明 | 高风险 gate | 1-3 人日探测；若缺 kernel 则显著扩大 | PyTorch `torch._scaled_mm` backend |
| P1 Primus TorchAO provider + 单 GPU | 中 | 2-4 人日 | direct Linear/API、统一 FQN、现有 tensorwise FP8 整理 |
| P1 FSDP2 + DTCP fingerprint/resume | 中高 | 2-5 人日 | module replacement、FSDP/state dict、metadata 预校验 |
| ALTO LPT core 解耦 | 中高 | 2-5 人日 | 顶层 import、全局 patch、TorchTitan 类型依赖 |
| P2 ALTO 单阶段 provider | 中高 | 2-5 人日 | import 隔离、wrapper/FSDP/checkpoint |
| 双 provider 短程 A/B | 中高 | 2-5 人日 | 同 FQN、不同 scaling/kernel 的公平比较 |
| P1/P2 完整收敛 | 高 | 取决于训练周期 | MLPerf 数值和随机性 |
| P3 schedule 接入 | 中 | 1-3 人日 | step/gradient accumulation/resume 语义 |
| P3 MXFP4 数值稳定 | 高 | 1-3 周实验量级 | scale selection、stage boundary、收敛 gap |

总体判断：

- **双 provider 的配置和分派代码本身复杂度低。** 一个入口、两个私有函数和一份 FQN 选择已经足够。
- **生产复杂度为中高。** 主要来自 TorchAO main 的 ROCm kernel gate、ALTO import 隔离、两种 tensor subclass 的 FSDP/DTCP 矩阵，而不是 provider 抽象。
- **P3 两阶段的代码增量仍小，但数值/收敛验证成本高。** 这是整项工作中最大的算法风险。
- 跨 provider full resume 不支持，可避免把复杂度推到很高。

## 14. 风险清单与缓解

| 风险 | 严重度 | 缓解 |
|---|---:|---|
| 当前配置并非 MXFP8→MXFP4 | 高 | 新建明确 recipe，单测 schedule 边界，不复用错误命名 |
| FP4 收敛与 BF16 gap 大 | 高 | P3 后置；分级数值 gate；先固定 layer filter 和 stage boundary |
| PyTorch/torchao/Triton/ALTO API 漂移 | 高 | pin commit/image digest；启动时精确 capability probe |
| 新 TorchAO 使现有 tensor-wise FP8 回退 | 高 | P0 先回归 228 层、190/38 wgrad、compile、FSDP2、DTCP、数值和性能 |
| TorchAO main dim1 固定 CUDA、ROCm scale layout 错误 | 外部 patch 已验证，尚未上游/固定交付 | HIP 选择 FlyDSL 和 row-major scales；已通过 kernel trace、SQNR、compile，下一步固定 commit/wheel |
| TorchAO 通用 wrapper 丢失 bias 语义 | 阻断（wrapper 方案） | P1 先做 wrapper/direct gate；wrapper 必须只包 weight、加回 bias并验证 bias gradient，否则采用直接 `MXFP8Linear` |
| `torch._scaled_mm` gfx950 原生路径不成立 | 阻断/高 | 先做 E4M3+E8M0 微基准；失败则重新评估 backend/kernel 范围 |
| TorchTitan 旧 converter protocol 不完整 | 高 | 只作历史分析，不直接复用该 converter/container |
| 当前 TorchTitan-main 被误当作 ROCm/MXFP4 证据 | 高 | 明确其仅有 SM100+ MXFP8 preset，默认 launcher 为 BF16，且没有 Flux MXFP8 integration test 或 MXFP4 训练实现 |
| provider-specific shape 对齐不同 | 高 | 公共 eligibility + fail-fast；记录 expected/converted/skipped FQN |
| gfx950 外回退导致“假性能” | 高 | profile/native kernel 证据作为验收项 |
| ALTO/候选 TorchAO wrapper 与 FSDP2 不兼容 | 高 | 单 Linear→tiny Flux→2 GPU DTCP 的渐进测试；直接 MXFP8Linear 单独验证 |
| 切换时替换 Parameter 破坏 FSDP/optimizer | 高 | 一次包装、按 step dispatch；断言 Parameter identity 不变 |
| compile 路径未验收导致性能比较失真 | 高 | P1/P2 bring-up 可关闭 compile，但交付必须恢复当前 block fullgraph；仅两阶段图重编译留到 P4 |
| gradient accumulation 中途切换 | 高 | 所有 micro-batch 使用同一 `global_step+1` |
| de-oscillation 在 MXFP8 阶段误触发 | 高 | 首版禁用；后续增加 effective precision gate |
| checkpoint 配置不一致 | 中高 | resume 校验 precision policy/switch step fingerprint |
| fingerprint 校验发生在状态写入之后 | 高 | 增加 metadata 预读；校验通过后才 restore model/optimizer |
| 全量 Linear 量化导致 shape/收敛问题 | 中高 | 首版只转换 transformer block 大 GEMM |
| 当前 Primus 未提交改动干扰归因 | 高 | 先冻结 BF16、native tensor-wise FP8 和 NeMo FP8 三条 baseline，低精度 diff 单独管理 |
| ALTO 顶层导入拖入 TorchTitan 和全局 patch | 高 | 把 core 解耦列为 P2/P3 前置 gate；不阻塞纯 TorchAO P1 |
| 两 provider 转换层集合不同 | 高 | Primus 维护唯一 FQN profile，并断言 converted FQN 完全一致 |
| 跨 provider full resume 状态不兼容 | 高 | 明确不支持；仅允许 dense model-only warm start |
| 低精度通信扩大调试面 | 中 | 首版保持 BF16 all-gather/FP32 reduce |

## 15. 明确不做项

第一版不做：

- 端到端 FP4 声明；
- 参数或 optimizer state 的永久 FP4/FP8 存储；
- FP8/MXFP8 FSDP all-gather；
- Tensor Parallel / Pipeline Parallel；
- encoder/VAE 量化；
- attention kernel 量化；
- 运行中替换 module/Parameter；
- 通用全仓库 QuantizationManager；
- compile graph 在阶段切换时自动重建；
- de-oscillation；
- 自动调优和 tuned config 生成；
- 两个独立进程的 stage handoff 编排。

## 16. 最终建议

1. 先把当前 Primus Flux BF16、native tensor-wise FP8 和 NeMo delayed FP8 固定成同口径可复现基线；补齐或明确记录 native FP8 尚缺的完整 seed。
2. 升级 TorchAO 后先证明现有 tensor-wise FP8 没有 module、190/38 wgrad、compile、FSDP2、DTCP、数值或性能回退。
3. 正式 TorchAO 路径不为已删除的 `MXLinearConfig` 增加长期兼容层；先验证当前 weight-wrapper 的 bias/parameter/state-dict 语义，未满足时采用直接 `MXFP8Linear`。
4. 在进入 Primus 前，先接通并证明 TorchAO gfx950 dim1 Triton/FlyDSL 和原生 `torch._scaled_mm`；当前 TorchTitan-main 的 SM100+ preset 不是 ROCm 证据。
5. 将当前内联 tensorwise FP8 和未来 MX provider 收敛到一个 `precision.py`，共用 228 个 Flux block Linear profile，并继承 190 个低精度 wgrad + 38 个 QKV 高精度 wgrad 起点；不引入 manager/registry/类体系。
6. P1 bring-up 可关闭 compile，但正式性能和收敛验收必须恢复当前逐 block `fullgraph=True`，并同时比较 BF16、native tensor-wise FP8、MXFP8 和 NeMo delayed FP8。
7. 只有 MXFP4 目标在 P1 后仍有明确价值时，才顺序接入 ALTO 单阶段 MXFP8；ALTO 必须先完成 LPT core 无副作用导入。单次运行固定 provider。
8. 保持 master weight、现有 optimizer policy 和 FSDP 通信为 FP32/BF16；full resume 在状态加载前严格校验 provider fingerprint。跨 provider model-only warm start 只有在 canonical dense 导出/加载完成后才支持。
9. ALTO 单阶段完成 FSDP、compile、resume 和短程数值验证后，再增加 `mxfp8_e4m3 → mxfp4`。两阶段由 `global_step + 1` 控制 dispatch，不替换 Parameter，不保存冗余 phase；DTCP 必须保存并校验 schedule fingerprint。
10. 对 MXFP4 采用“功能、短程数值、阶段 resume、完整收敛”四层 gate。根据用户提供的既有实验观察，特定 FP4 schedule 与 BF16 存在较大 gap；当前目标 `MXFP8→MXFP4` 尚无同口径收敛证据，因此默认不进入主训练配方。

该方案把 Primus 核心改动限制在配置校验、一份 FQN 选择、一次模型转换、checkpoint fingerprint 和一个可选 step hook。双 provider 的框架复杂度可控；真正需要投入的是 ROCm kernel、FSDP/checkpoint 和数值验证。
