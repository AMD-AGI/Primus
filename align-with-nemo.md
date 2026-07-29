# Primus / TorchTitan / NeMo FLUX MLPerf 对齐检查

## 检查范围

- Primus 启动脚本：`local_runs/run_flux_mlperf.sh`
- MLPerf NeMo 启动脚本：
  - `/shared_nfs/zirui/code/mlperf-training-6-0/flux1/nemo/run_flux_mlperf_docker.sh`
  - `/shared_nfs/zirui/code/mlperf-training-6-0/flux1/nemo/run_with_docker.sh`
- Primus 默认训练配置：`examples/diffusion/configs/MI355X/flux.1_schnell_t2i-pretrain.yaml`
- NeMo 实际训练配置来源：
  - `config_${DGXSYSTEM}.sh`
  - `config_common.sh`
  - `conf/flux1_schnell.yaml`
  - `conf/model/schnell.yaml`
  - `conf/data/cc12m.yaml`

本文同时包含静态配置对比、2026-07-24 的历史 Primus BF16 审计，以及 2026-07-28 至
2026-07-29 在三个单节点 8×MI355X 环境完成的 Primus BF16、TorchTitan BF16 和 NeMo FP8
`delayed_short`、seed 1234 的 MLPerf convergence 对齐实测。

## 总结结论

2026-07-28 至 2026-07-29 的单 seed 对齐实验已完成。三个框架都达到 validation loss
`<= 0.586` 并记录 `run_stop=success`：

- Primus BF16 和 TorchTitan BF16 都在 `7,340,032` samples 收敛。
- NeMo FP8 `delayed_short` 在 `7,602,176` samples 收敛，比前两者晚一个 validation interval
  （`262,144` samples）。
- 三个结果均落在 canonical `flux_ref_512` 的 `7,077,888–7,602,176` samples reference
  区间内。

这证明当前 Primus BF16 单节点 GBS512 路径已取得可信的单 seed convergence parity；但它
仍不是完整的 RCP 统计验收：尚未运行 10 个独立 seed，也未用 RCP checker 对结果分布进行
最终判定。

2026-07-24 的旧 Primus run **不能作为有效的 MLPerf convergence 结果**：审计发现当时
Primus 丢弃了 COCO 数据中的固定 `timestep`，validation 实际复用了训练路径的随机连续
timestep，计算的不是 MLPerf 定义的八个等间隔 timestep validation loss。该旧 run 在
step 4096 报告的 `2,097,152` samples-to-converge 已作废，仅保留为历史控制流记录。

如果按 NeMo 的 `MI355X_01x08x16` 单机 8 卡配置看，Primus 的 batch、LR、warmup、optimizer、FLUX Schnell 模型语义、VAE latent 归一化、随机 timestep 和 prompt dropout 大体对齐。

NeMo 本次使用 `FP8_RECIPE=delayed_short`，与 Primus/TorchTitan BF16 的数值和性能路径
不同；因此可以比较 RCP convergence samples，但不能把耗时或吞吐直接解释为同 precision
性能对比。

## 2026-07-28 至 2026-07-29 三框架 RCP 对齐结果

### 公共实验设置

| 项目 | 值 |
| --- | --- |
| 硬件 | 每个框架 1 节点，8× AMD Instinct MI355X |
| seed | `1234` |
| global / local batch | `512 / 64` |
| LR / warmup | `2e-4 / 1600 steps` |
| optimizer | AdamW，beta `0.9/0.95`，eps `1e-8`，weight decay `0.1` |
| validation interval | `262,144` samples = 512 training steps |
| eval samples | `29,696` |
| target | validation loss `<= 0.586` |
| train samples | `1,099,776`，按 epoch/repeat 继续采样直至收敛 |

### 最终结果

| 框架 | 节点 | Precision / recipe | 收敛 step | Samples to converge | 最终 validation loss | MLPerf stop |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Primus | `crsuse2-m2m-005` | BF16 | 14,336 | **7,340,032** | **0.585705280** | `success` |
| TorchTitan | `crsuse2-m2m-100` | BF16 | 14,336 | **7,340,032** | **0.585746765** | `success` |
| NeMo | `crsuse2-m2m-008` | FP8 `delayed_short` | 14,848 | **7,602,176** | **0.585948944** | `success` |

结论：Primus BF16 与 TorchTitan BF16 的 samples-to-converge 完全一致，最终 validation loss
也只相差约 `4.15e-5`。NeMo FP8 `delayed_short` 晚一个 eval interval 达标。该结果支持
Primus/TorchTitan 的单 seed RCP512 convergence parity，并表明 NeMo FP8 路径也在 reference
区间上沿内正常收敛。

### 运行可靠性与合规边界

- Primus 首次运行因 Docker 默认 soft `nofile=1024` 在 step 510 的 validation 阶段触发
  `Too many open files`。launcher 增加 `--ulimit nofile=1048576:1048576` 后从 checkpoint
  恢复，后续 validation 不再复现该错误。
- Primus 在 step 12,900 保存 checkpoint 时曾出现训练 rank 静默退出；不完整 checkpoint
  被隔离，任务从完整的 step 12,800 checkpoint 恢复。新任务成功重写完整 step 12,900
  checkpoint，并最终在 step 14,336 达标，生成 `checkpoint-final`。
- 因 Primus 中途恢复，本结果适合 convergence/RCP 对齐，不是一次不间断的正式 MLPerf
  submission run。
- NeMo 节点没有 passwordless sudo，本次跳过 host runtime tunables 和 cache clear；因此
  可用于数值收敛比较，不用于正式 performance/compliance 申报。
- 目前只有 seed 1234。要声明完整 RCP 统计通过，仍需运行独立 seeds 并执行 RCP checker。

### 数据与日志产物

- Primus train/eval：`/shared_nfs/zirui/data/cc12m-preprocessed`、
  `/shared_nfs/zirui/data/coco_preprocessed`
- NeMo Energon：`/shared_nfs/zirui/data/energon_mlperf`；tar 样本包含 `t5.bytes`、
  `clip.bytes`、`mean.bytes`、`logvar.bytes`
- Primus 日志：
  - `/shared_nfs/zirui/runs/flux_rcp_compare/launchers/primus-flux-seed1234-20260728T0311.log`
  - `/shared_nfs/zirui/runs/flux_rcp_compare/launchers/primus-flux-seed1234-resume-nofile-20260728T122152.log`
  - `/shared_nfs/zirui/runs/flux_rcp_compare/launchers/primus-flux-seed1234-resume12800-20260729T085740.log`
- TorchTitan 日志：
  `/shared_nfs/zirui/runs/flux_rcp_compare/launchers/torchtitan-flux-seed1234-20260728T0311.log`
- NeMo 日志：
  `/shared_nfs/zirui/runs/flux_rcp_compare/launchers/nemo-flux-seed1234-energon-20260728T121102.log`
- W&B project：`mlperf-flux-rcp`

## MLPerf v6.0 `flux_ref_512` 历史审计与修复状态

GBS512 BF16 RCP 的 20 个 reference run 位于 `7,077,888–7,602,176` samples，均值
`7,235,174`；旧 Primus 结果只有均值的约 29%。AMD/NVIDIA v6.0 的公开 Flux 提交分别使用
GBS1024/2048/2304，因此 GBS512 的直接验收基准是 `flux_ref_512`，不能把不同 GBS 的提交
样本数直接当作同一分布。

下表记录的是 2026-07-24 旧 run 暴露的问题，不代表当前代码仍处于相同状态。P0/P1
训练语义修复已经进入 2026-07-29 的有效单 seed run；统计和正式 submission 相关验收仍待完成。

| 优先级 | 历史不对齐项 | 影响 | 修复或验收动作 | 当前状态 |
| --- | --- | --- | --- | --- |
| P0 | eval dataset 的固定 `timestep` 被丢弃 | target `0.586` 不可比较，旧 convergence 无效 | eval 保留整数 timestep，严格使用 `timestep / 8.0`，全量归约 29,696 samples | 已修复；单 seed convergence 已验证 |
| P0 | Primus 使用 PyTorch 默认初始化 | 初始模型和收敛曲线不等价 | 移植 TorchTitan Xavier/Normal/zero AdaLN/final projection 初始化 | 已修复；单 seed convergence 已验证 |
| P0 | MLPerf train/eval 使用随机/交错 `DistributedSampler` | 样本顺序和每 rank shard 不同 | 使用 TorchTitan 风格的连续、无 padding、顺序 rank shard | 已修复；单 seed convergence 已验证 |
| P0 | QK RMSNorm 固定 `eps=1e-6` | 与 TorchTitan `nn.RMSNorm(dim)` 的 BF16 dtype epsilon 不同 | 直接采用 PyTorch RMSNorm 默认 epsilon 语义 | 已修复；单 seed convergence 已验证 |
| P1 | 所有 DP rank 使用相同训练 RNG | noise、VAE sampling、timestep、prompt dropout 缺少 rank 独立性 | common seed 初始化权重，随后按 DP rank 派生训练 seed | 已修复；单 seed convergence 已验证 |
| P1 | CFG dropout 在 GPU 上消耗 PyTorch RNG | 改变 latent noise/timestep 的 reference RNG 轨迹 | 与 TorchTitan 一样逐样本使用 rank-distinct Python RNG | 已修复；单 seed convergence 已验证 |
| P1 | BF16 parameter storage/reduce/default AdamW state | 不等价于 TorchTitan FP32 master 参数和 FP32 reduction | FP32 参数与 AdamW moments、BF16 forward、FP32 gradient reduce | 已修复；smoke 和单 seed convergence 已验证 |
| P1 | MLLOG 把 warmup 初始 LR `0.0` 记为 base LR | 无法匹配 `flux_ref_512` RCP | 记录配置 base LR | 已修复；短跑 checker 未报告日志结构错误 |
| P1 | INIT/RUN/EVAL 区间未包围实际工作，缺少 block events | compliance/performance timing 无效 | 按 TorchTitan lifecycle 记录并运行 compliance checker | 日志已修复；待不间断正式 run 验收 |
| P1 | launcher 只记录 `cache_clear=true` | 不满足正式提交的 cache-clear 行为 | 每节点实际清理 OS cache，失败则阻止正式 run | runner 已支持；正式 run 仍需逐节点确认 |
| P2 | 固定 seed `10007`，没有 10-run 编排 | 无法做统计 RCP 验收 | 支持环境 seed、独立 `result_0..9.txt` 和 RCP/package checker | runner 已实现；10 个 seed 尚未执行 |

截至 2026-07-25，上述代码修复和 10-seed runner 已实现；相关 unit tests 为
`25 passed`。8×MI355X、GBS512 的 2-step FP32-master/BF16-compute smoke 已通过。
固定-timestep 的 512-step validation smoke 也已通过，得到 loss `0.703748`；
29,696 个 eval samples 全量消费，训练稳态约 `1.62 s/step`。compliance checker
仅因该短 run 尚未达到 `0.586` 而失败，没有报告日志结构错误。2026-07-29 的单 seed
完整 convergence 已在 `7,340,032` samples 达标，并与 TorchTitan BF16 完全一致；因此
单 run 验收阶段已经完成。10-seed 统计 RCP 和正式 compliance/performance 验收仍未完成。

随后使用 `LR=2e-4, warmup=1600` 的完整 30,000-step BF16 run 未收敛：最终
`15,360,000` samples、`run_stop=aborted`。曾因把 Dell/NVIDIA 的某个 GBS512
提交参数误当作 reference RCP 参数，将 Primus 默认 LR 改为 `1e-4`；该判断错误。
MLPerf Logging v6.0 的 canonical `rcps_flux1.json` 明确规定 `flux_ref_512` 为
`opt_base_learning_rate=2e-4`、warmup 1600，20 次 reference run 在
`7,077,888–7,602,176` samples 收敛。因此 `1e-4` 的 512-step smoke（validation
loss `0.703868`）和截至约 5.5M samples 的未完成 run 均不作为 RCP512 parity
证据；默认值已恢复为 `2e-4`。先前 `2e-4` run 未收敛应继续作为实现或数值语义
在修复前仍未完全对齐的历史证据，而不能用更换 LR 来解释；2026-07-29 的修复后 run
已取代它成为当前单 seed 基线。

验收分两阶段，当前状态如下：

1. **已完成**：一个修复后的单节点 8×MI355X、GBS512、BF16 run，metric 正确且
   samples-to-converge 为 `7,340,032`。
2. **未完成**：运行 10 个独立 seed；只有统计 RCP checker 通过后，才声明完整的
   `flux_ref_512` convergence parity 完成。

## 2026-07-24 单节点 MI355X BF16 历史实测（结果已作废）

### 环境修复

当前节点首次启动暴露了两个环境问题，修复后短跑和正式训练均成功：

- 完整训练集实际路径为 `/data/cc12m-preprocessed`；旧默认
  `/data/cc12m_preprocessed` 缺少 26/4762 个 Arrow shards。
- 当前 Python 环境缺少 `mlperf_common`，补装与 NeMo requirements 一致的 commit
  `b86d175a05849d650a8ff69c1e2c37b9f4e61d51`。
- `run_flux_mlperf.sh` 增加 `torchrun --tee=3 --log-dir=...`，保留逐 rank stdout/stderr。

### 运行配置

| 项目 | 值 |
| --- | --- |
| 节点/GPU | 1 节点，8× AMD Instinct MI355X |
| precision | BF16 mixed precision |
| train/eval data | `/data/cc12m-preprocessed` / `/data/coco_preprocessed` |
| local/global batch | 64 / 512 |
| LR / warmup | `2e-4` / 1600 steps（canonical `flux_ref_512`） |
| optimizer | AdamW，beta `0.9/0.95`，eps `1e-8`，weight decay `0.1` |
| validation interval | 262144 samples = 512 training steps |
| target | validation loss `<= 0.586` |
| max steps | 30000，达到 target 后 early stop |

### 历史结果（仅保留用于问题追踪）

512-step E2E 验证首先成功触发一次真实 COCO validation，loss 为 `0.691528`，进程
exit code 0。该短跑会因为 `MAX_STEPS=512 < WARMUP_STEPS=1600` 将实际 warmup 截断到
512，只用于验证 E2E 流程，不能作为正式收敛曲线对照。

当时的 convergence run 保留完整 1600-step warmup，validation 曲线如下；这些数值使用了
错误的 validation timestep 语义，不能作为 RCP 对齐结果：

| Step | Samples | Validation loss | Target reached |
| ---: | ---: | ---: | :---: |
| 512 | 262144 | 0.735804 | No |
| 1024 | 524288 | 0.693313 | No |
| 1536 | 786432 | 0.657642 | No |
| 2048 | 1048576 | 0.617320 | No |
| 2560 | 1310720 | 0.595207 | No |
| 3072 | 1572864 | 0.629866 | No |
| 3584 | 1835008 | 0.610989 | No |
| 4096 | 2097152 | **0.583601** | **Yes** |

历史产出：

- early stop step：4096
- time-to-converge：`6119.05s`（约 101.98 分钟）
- MLLOG `run_stop`：`success`
- 进程 exit code：0
- mean step time：`1.4824s`
- mean throughput：`44.0544 samples/GPU/s`
- max GPU peak memory：`284.41 GiB`

产物：

- summary：`local_runs/flux_mlperf_mi355x_n1_convergence_30000_20260724_1152_summary.txt`
- MLLOG：`local_runs/flux_mlperf_mi355x_n1_convergence_30000_20260724_1152_mllog.txt`
- 主日志：`local_runs/flux_mlperf_mi355x_n1_convergence_30000_20260724_1152.log`
- 逐 rank 日志：`local_runs/flux_mlperf_mi355x_n1_convergence_30000_20260724_1152_ranklogs/`

## TorchTitan Flux 与 NeMo MLPerf 对齐情况

TorchTitan 启动脚本 `/zirui/code/torchtitan-main/run_flux_test.sh` 默认使用：

- `MODULE=flux`
- `CONFIG=flux_schnell_mlperf_preprocessed`
- `STEPS=30000`
- `LOCAL_BATCH_SIZE=64`
- recipe registry 继承的默认值为 `LR=1e-4`，但 canonical RCP512 是 `2e-4`；
  本地 wrapper 必须显式覆盖为 `2e-4`
- `WARMUP_STEPS=1600`

按默认单机 8 卡运行时，global batch 为 `64 * 8 = 512`，与 NeMo `MI355X_01x08x16` 的 `BATCHSIZE=512` 对齐。

TorchTitan 的 MLPerf 收敛逻辑已经比较完整：

- `FluxMLPerfConfig.target_eval_loss = 0.586`
- `FluxMLPerfConfig.eval_samples = 262144`
- 训练时转换为 `ceil(eval_samples / global_batch_size)` 个 step 触发 validation
- validation loss 达标后记录 `time_metrics/time_to_converge(s)` 并 early stop
- MLPerf logger 会记录 global batch、optimizer hparams、evaluation frequency、train/eval samples 等关键事件

TorchTitan 与 NeMo 的主要未对齐点是 FP8：

- NeMo 使用 TE/NVTE 风格 `FP8_RECIPE=delayed` 或 `delayed_short`，配置为 `fp8=hybrid`。
- TorchTitan Flux 没有看到等价的 TE delayed-scaling FP8 recipe。
- TorchTitan Flux 支持 `MXFP8Converter`，但这是 NVIDIA SM100+（B200/B100）+ torchao nightly 的 MXFP8 路径，不是 MI355X/ROCm 路径，也不是 NeMo 当前 MLPerf FP8 recipe 的等价实现。

Primus 的旧 run 使用了错误 validation metric，但修复后的 2026-07-28 至 2026-07-29
单 seed 对齐实验已经确认 Primus BF16 与 TorchTitan BF16 都在 `7,340,032` samples
收敛。TorchTitan 和 Primus BF16 路径仍不等价于 NeMo 的 TE delayed-scaling FP8 路径，
因此该结果证明 convergence parity，不证明 precision 或性能路径等价。

## Primus 对齐 TorchTitan MLPerf 训练的本次更新

本次将 TorchTitan 风格的 MLPerf 收敛控制接入 Primus diffusion/Flux FSDP2 路径：

- 支持 `data.eval_dataset_path`，用于构建独立 validation dataset。
- 支持 `mlperf.enable`、`mlperf.target_eval_loss`、`mlperf.eval_samples`。
- MLPerf 模式下按 `ceil(eval_samples / global_batch_size)` 触发 validation。
- validation loss `<= target_eval_loss` 时记录 time-to-converge 并 early stop。
- 增加 `mlperf_logging` 和 `mlperf_common` 依赖。
- `local_runs/run_flux_mlperf.sh` 默认启用 MLPerf 模式，并设置 `MAX_STEPS=30000`、`TARGET_ACCURACY=0.586`、`VAL_CHECK_INTERVAL=262144`。

这一步最初只跑通 TorchTitan 风格的 convergence 控制流；后续固定 timestep、初始化、
RNG、采样和日志语义修复后，2026-07-29 已验证有效的单 seed convergence。NeMo TE FP8
recipe、NeMo Energon loader/split byte-level parity、10-seed RCP、正式 MLPerf compliance
审核和多机 profile 自动切换仍需完成。

## 已对齐项

| 项目 | Primus 当前值 | NeMo 对应值 | 结论 |
| --- | --- | --- | --- |
| 模型 | `flux.1_schnell_t2i.yaml` / Schnell | `MODEL=schnell` | 对齐 |
| guidance embed | Schnell 为 `guidance_embed=False` | `guidance_embed: False` | 对齐 |
| 单机 8 卡 local batch | `LOCAL_BATCH_SIZE=64` | `MBS=64` | 对齐 `MI355X_01x08x16` |
| 单机 8 卡 global batch | `64 * 8 = 512` | `BATCHSIZE=512` | 对齐 `MI355X_01x08x16` |
| LR | `LR=2e-4` | canonical `flux_ref_512` 的 `opt_base_learning_rate=0.0002` | 对齐 reference RCP；单个提交的 `1e-4` 不能替代 RCP 参数 |
| warmup | `WARMUP_STEPS=1600` | `WARMUP_STEPS=1600` | 对齐 GBS512 |
| optimizer | AdamW | AdamW | 基本对齐 |
| Adam beta1/beta2/eps | `0.9 / 0.95 / 1e-8` | `0.9 / 0.95 / 1e-8` | 对齐 |
| weight decay | `0.1` | `0.1` | 对齐 |
| grad clip | `1.0` | `1.0` | 对齐 |
| prompt dropout | `PROMPT_DROPOUT_PROB=0.1` | `classifier_free_guidance_prob: 0.1` | 语义对齐 |
| VAE scale/shift | `0.3611 / 0.1159` | `0.3611 / 0.1159` | 对齐 |
| flow matching loss | random noise/timestep, target `noise - latent` | random noise/timestep, target `noise - latent` | 基本对齐 |
| image/text latent shape | `img_size=256`, T5 seq 256 | `seq_length: 256` | 基本对齐 |
| validation frequency | 262144 samples / 512 steps | 262144 samples | 对齐 |
| target / early stop | loss `<=0.586` 后停止 | loss `<=0.586` 后停止 | 行为对齐 |
| MLLOG convergence events | eval events、TTC、`run_stop=success` | 对应 MLPerf events | 关键事件已接入 |
| train/eval sample count | 1099776 / 29696 | 1099776 / 29696 | 数量对齐 |

## 未对齐项

### 1. FP8 没有对齐

NeMo 通过 `FP8_RECIPE` 和 Hydra plugin 启用 FP8：

- `FP8_RECIPE=delayed`：`fp8=hybrid`、`amax_history_len=1024`、`amax_compute_algo=max`
- `FP8_RECIPE=delayed_short`：`fp8=hybrid`、`amax_history_len=4`、`amax_compute_algo=most_recent`

Primus 当前 diffusion FSDP2 路径看起来是 bf16 mixed precision，没有看到等价的 FLUX FP8 recipe/TE plugin 接入。因此性能和数值路径都不算对齐。

### 2. 单 seed convergence parity 已验证，统计 parity 尚未完成

Primus 和 TorchTitan 已分别在单节点 8×MI355X 上用 seed 1234 完成 BF16 baseline，二者
都在 step 14,336、`7,340,032` samples 达标，最终 validation loss 分别为
`0.585705280` 和 `0.585746765`。NeMo FP8 `delayed_short` 在 step 14,848、
`7,602,176` samples 达标。因此单 seed convergence parity 已有实测证据。

尚未完成的是 10 个独立 seed 的结果分布和 RCP checker 验收；此外 NeMo 使用 FP8，不能
把它与两个 BF16 run 的耗时和吞吐直接比较。

### 3. 默认 stop condition 不对齐

NeMo 默认：

- `MAX_STEPS=-1`
- 主要靠 validation/target accuracy 决定收敛

Primus 原始默认：

- `MAX_STEPS=100`
- 更像 benchmark smoke/perf run，不是 MLPerf convergence run

本次更新后，`local_runs/run_flux_mlperf.sh` 默认 `MAX_STEPS=30000`，并允许 target loss
达标后提前停止。30000 是安全上限而非主要 stop condition；有效对齐 run 实际在
step 14,336 停止。

### 4. 数据 loader 和 split 不对齐

NeMo：

- 容器内数据路径默认为 `/dataset/energon`
- `conf/data/cc12m.yaml` 使用 Energon dataset
- data module 同时支持 train/val split

Primus 原始状态：

- 默认 `DATASET_PATH=/data/cc12m-preprocessed`
- 使用 HF `load_from_disk` 风格的 precomputed dataset
- `EMPTY_ENCODINGS_PATH=/data/empty_encodings`
- 使用独立 `/data/coco_preprocessed` eval dataset

即使底层样本内容来自同一 MLPerf 数据，loader、shuffle、split、validation 行为仍未完全对齐。

Primus 已实际消费该 eval dataset，并完成从 step 512 到 step 14,336 的 validation 序列；
最终 samples-to-converge 与 TorchTitan 相同。但 loader、样本顺序与 NeMo Energon 的
byte-level parity 仍需单独验证。

### 5. 分布式策略不对齐

NeMo：

- Megatron/NeMo strategy
- `data_parallel_sharding_strategy=no_shard`
- `use_distributed_optimizer=True`
- `overlap_param_gather=True`
- `overlap_grad_reduce=True`
- `num_distributed_optimizer_instances=ceil(DGXNNODES / SEGMENT)`

Primus：

- torch FSDP2 `fully_shard`
- `dp_replicate=1`
- `fsdp2_reshard_after_forward=True`

两者训练数学可能相近，但通信、参数/梯度 sharding、optimizer state 行为不同，不能认为分布式配置完全对齐。

### 6. 多机默认超参不会自动随 NeMo 配置切换

NeMo MI355X 典型默认展开：

| NeMo config | Nodes x GPUs | MBS | GBS | LR | Warmup |
| --- | --- | ---: | ---: | ---: | ---: |
| `MI355X_01x08x16` | 1 x 8 | 64 | 512 | 0.0002 | 1600 |
| `MI355X_02x08x16` | 2 x 8 | 32 | 512 | 0.0002 | 1600 |
| `MI355X_04x08x64` | 4 x 8 | 64 | 2048 | 0.00025 | 0 |
| `MI355X_08x08x16` | 8 x 8 | 32 | 2048 | 0.00025 | 0 |
| `MI355X_08x08x32` | 8 x 8 | 32 | 2048 | 0.00025 | 0 |

Primus 当前脚本只改 `NNODES` 时，`LOCAL_BATCH_SIZE`、`LR`、`WARMUP_STEPS` 不会按 NeMo config 自动切换。

例如 `NNODES=8 GPUS_PER_NODE=8` 时，Primus 默认会变成：

- global batch = `64 * 8 * 8 = 4096`
- LR = `0.0002`
- warmup = `1600`

这与 NeMo 8 节点默认 `GBS=2048`、`LR=0.00025`、`WARMUP_STEPS=0` 不对齐。

## 建议对齐动作

如果目标是单机 GBS512 BF16 convergence baseline，Primus 当前 run 已可作为可复现基线；
但它尚不是经过 compliance checker 审核的 MLPerf submission，也不是 NeMo 默认 FP8 run。

如果目标是对齐 NeMo MLPerf 训练配置，建议补齐：

1. 为 Primus 增加 NeMo-style `DGXSYSTEM` 或等价 profile，自动设置 `NNODES`、`GPUS_PER_NODE`、`LOCAL_BATCH_SIZE`、`LR`、`WARMUP_STEPS`。
2. 明确 BF16 baseline 与 NeMo 默认 FP8 run 的比较边界；若追求 NeMo 默认配置 parity，需要支持 `delayed` / `delayed_short` 的 FP8 行为。
3. 使用 10 个独立 seed 分别运行 Primus/TorchTitan 所需基线，并用 RCP checker 验证结果分布。
4. 如需 precision parity，再补 NeMo BF16 对照；当前 NeMo FP8 `delayed_short` 结果只用于
   convergence 区间比较。
5. 使用 MLPerf compliance checker 审核 logging；尤其确认 cache clear 不只是记录事件，还实际执行。
6. 确认 Primus precomputed train/eval 内容与 NeMo Energon、TorchTitan 数据路径逐样本等价。
7. 文档和脚本中继续区分两种模式：
   - performance smoke/benchmark：固定 `MAX_STEPS`
   - MLPerf convergence parity：`MAX_STEPS=-1` 或足够大，靠 validation target 停止

## 当前判断

Primus FLUX 当前已经完成 **单机 8 卡、GBS512、BF16、seed 1234 的有效 MLPerf
convergence**。旧 step 4096 的 `run_stop=success` 基于错误 metric，已经作废；新的有效
结果为 step 14,336、`7,340,032` samples、validation loss `0.585705280`。

单 seed 结果与 TorchTitan BF16 的 samples-to-converge 完全一致，并落在
`flux_ref_512` reference 区间内。因此当前状态是：**单 seed convergence parity 已完成，
10-seed RCP 统计与正式 MLPerf submission/compliance parity 尚未完成**。

最重要的 blocker 是：

- 尚未运行 10 个独立 seed 并通过 RCP checker
- NeMo 默认 FP8 recipe 与 Primus BF16 路径不同
- Primus run 曾从 checkpoint 恢复，NeMo run 跳过 host tunables/cache clear；二者都不能
  直接作为正式 performance/compliance submission
- 多机 profile 超参未自动对齐
- 数据 loader/split 与 MLPerf NeMo Energon 路径未确认等价
