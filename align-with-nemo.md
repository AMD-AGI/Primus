# Primus FLUX 与 MLPerf NeMo 配置对齐检查

## 检查范围

- Primus 启动脚本：`local_runs/run_flux_mlperf.sh`
- MLPerf NeMo 启动脚本：`/zirui/code/mlperf-training-6-0/flux1/nemo/run_with_docker_debug.sh`
- Primus 默认训练配置：`examples/diffusion/configs/MI355X/flux.1_schnell_t2i-pretrain.yaml`
- NeMo 实际训练配置来源：
  - `config_${DGXSYSTEM}.sh`
  - `config_common.sh`
  - `conf/flux1_schnell.yaml`
  - `conf/model/schnell.yaml`
  - `conf/data/cc12m.yaml`

本文同时包含静态配置对比和 2026-07-24 在单节点 8×MI355X 上完成的 Primus BF16
MLPerf convergence 实测。TorchTitan 与 NeMo 部分仍以代码/配置对比为主，本次没有在当前节点
独立运行 TorchTitan 或 NeMo baseline。

## 总结结论

Primus FLUX BF16 路径已经完成单节点 E2E 收敛验证，并在 TorchTitan 风格的
validation、target、MLLOG、time-to-converge 和 early-stop 行为上实现对齐。

如果按 NeMo 的 `MI355X_01x08x16` 单机 8 卡配置看，Primus 的 batch、LR、warmup、optimizer、FLUX Schnell 模型语义、VAE latent 归一化、随机 timestep 和 prompt dropout 大体对齐。

正式 Primus run 在 step 4096 达到 validation loss `0.583601 <= 0.586`，MLLOG
`run_stop` 为 `success`。这证明 Primus BF16 E2E 可以收敛，但不能单凭该结果宣称与
TorchTitan/NeMo 数值曲线或 MLPerf submission 完全等价：当前仍缺少同机 TorchTitan/NeMo
对照 run、MLPerf compliance checker 审核、数据 loader/split 等价证明和分布式策略对齐。
另外 NeMo `MI355X_01x08x16` 默认启用 `FP8_RECIPE=delayed`，与本次 Primus BF16 数值路径不同。

## 单节点 MI355X BF16 E2E 实测

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
| LR / warmup | `2e-4` / 1600 steps |
| optimizer | AdamW，beta `0.9/0.95`，eps `1e-8`，weight decay `0.1` |
| validation interval | 262144 samples = 512 training steps |
| target | validation loss `<= 0.586` |
| max steps | 30000，达到 target 后 early stop |

### 结果

512-step E2E 验证首先成功触发一次真实 COCO validation，loss 为 `0.691528`，进程
exit code 0。该短跑会因为 `MAX_STEPS=512 < WARMUP_STEPS=1600` 将实际 warmup 截断到
512，只用于验证 E2E 流程，不能作为正式收敛曲线对照。

正式 convergence run 保留完整 1600-step warmup，validation 曲线如下：

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

最终结果：

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
- `LR=2e-4`
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

Primus 当前已接入同类控制逻辑并完成实际收敛 run，因此两者在
**MLPerf validation/target/time-to-converge/early-stop 行为**上已经对齐。由于本次没有运行
TorchTitan baseline，目前只能确认控制流和主要超参对齐，不能确认两者逐 step loss 或
time-to-converge 数值一致。TorchTitan 和 Primus BF16 路径也都不等价于 NeMo 默认的
TE delayed-scaling FP8 路径。

## Primus 对齐 TorchTitan MLPerf 训练的本次更新

本次将 TorchTitan 风格的 MLPerf 收敛控制接入 Primus diffusion/Flux FSDP2 路径：

- 支持 `data.eval_dataset_path`，用于构建独立 validation dataset。
- 支持 `mlperf.enable`、`mlperf.target_eval_loss`、`mlperf.eval_samples`。
- MLPerf 模式下按 `ceil(eval_samples / global_batch_size)` 触发 validation。
- validation loss `<= target_eval_loss` 时记录 time-to-converge 并 early stop。
- 增加 `mlperf_logging` 和 `mlperf_common` 依赖。
- `local_runs/run_flux_mlperf.sh` 默认启用 MLPerf 模式，并设置 `MAX_STEPS=30000`、`TARGET_ACCURACY=0.586`、`VAL_CHECK_INTERVAL=262144`。

这一步让 Primus 对齐 TorchTitan 的 MLPerf convergence training 行为，并已通过真实
train/eval 数据验证收敛。NeMo TE FP8 recipe、NeMo Energon loader/split byte-level parity、
MLPerf compliance 审核和多机 profile 自动切换仍是后续工作。

## 已对齐项

| 项目 | Primus 当前值 | NeMo 对应值 | 结论 |
| --- | --- | --- | --- |
| 模型 | `flux.1_schnell_t2i.yaml` / Schnell | `MODEL=schnell` | 对齐 |
| guidance embed | Schnell 为 `guidance_embed=False` | `guidance_embed: False` | 对齐 |
| 单机 8 卡 local batch | `LOCAL_BATCH_SIZE=64` | `MBS=64` | 对齐 `MI355X_01x08x16` |
| 单机 8 卡 global batch | `64 * 8 = 512` | `BATCHSIZE=512` | 对齐 `MI355X_01x08x16` |
| LR | `LR=2e-4` | `LEARNING_RATE=0.0002` | 对齐 GBS512 |
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

### 2. 数值曲线 parity 尚未验证

Primus 已用真实 CC12M/COCO precomputed 数据跑通 validation、target、time-to-converge 和
early stop，并在 step 4096 收敛。但当前节点没有独立运行 TorchTitan/NeMo，因此还没有
同一硬件、seed、数据顺序和 precision 下的 validation 曲线对照。现阶段结论是
“Primus BF16 能收敛且控制流对齐”，不是“数值结果已与 TorchTitan/NeMo 完全一致”。

### 3. 默认 stop condition 不对齐

NeMo 默认：

- `MAX_STEPS=-1`
- 主要靠 validation/target accuracy 决定收敛

Primus 原始默认：

- `MAX_STEPS=100`
- 更像 benchmark smoke/perf run，不是 MLPerf convergence run

本次更新后，`local_runs/run_flux_mlperf.sh` 默认 `MAX_STEPS=30000`，并允许 target loss
达标后提前停止。30000 是安全上限而非主要 stop condition；本次实际在 step 4096 停止。

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

Primus 已实际消费该 eval dataset 并完成 8 次 validation，但仍需确认其样本内容、顺序和
timestep 生成与 NeMo Energon/TorchTitan validation 路径完全等价。

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
3. 在同一节点运行 TorchTitan BF16 baseline，比较 seed、数据顺序、逐 validation loss 和 TTC。
4. 在具备 Docker 的节点运行 NeMo BF16/FP8 对照，并分别报告数值与性能差异。
5. 使用 MLPerf compliance checker 审核 logging；尤其确认 cache clear 不只是记录事件，还实际执行。
6. 确认 Primus precomputed train/eval 内容与 NeMo Energon、TorchTitan 数据路径逐样本等价。
7. 文档和脚本中继续区分两种模式：
   - performance smoke/benchmark：固定 `MAX_STEPS`
   - MLPerf convergence parity：`MAX_STEPS=-1` 或足够大，靠 validation target 停止

## 当前判断

Primus FLUX 当前已经完成 **单机 8 卡、GBS512、BF16 的真实 MLPerf E2E convergence**：
关键训练超参、FLUX loss 语义、validation frequency、target、TTC、early stop 和主要 MLLOG
事件均已接入，step 4096 达标并 `run_stop=success`。

这标志着 Primus BF16 路径从“静态配置对齐/待验证”推进到“E2E 已收敛”。但
**TorchTitan/NeMo 数值 parity 和 MLPerf submission parity 仍未完成**。

最重要的 blocker 是：

- 未运行同机 TorchTitan/NeMo baseline，数值曲线 parity 未验证
- NeMo 默认 FP8 recipe 与 Primus BF16 路径不同
- MLPerf compliance checker 尚未审核
- 多机 profile 超参未自动对齐
- 数据 loader/split 与 MLPerf NeMo Energon 路径未确认等价
