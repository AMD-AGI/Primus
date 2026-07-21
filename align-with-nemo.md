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

本次检查为静态配置对比，未实际启动训练。

## 总结结论

当前 Primus FLUX 配置 **只部分对齐** MLPerf NeMo。

如果按 NeMo 的 `MI355X_01x08x16` 单机 8 卡配置看，Primus 的 batch、LR、warmup、optimizer、FLUX Schnell 模型语义、VAE latent 归一化、随机 timestep 和 prompt dropout 大体对齐。

但 Primus 当前还没有达到 MLPerf NeMo submission/convergence parity。主要差距在 FP8、validation/target accuracy、MLLOG、数据 split/loader、分布式策略，以及多机默认超参自动切换。

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

因此 TorchTitan 当前在 **MLPerf validation/target/time-to-converge 逻辑** 上比 Primus 原始 Flux 路径更接近 NeMo，但在 **NeMo FP8 recipe parity** 上仍未对齐。

## Primus 对齐 TorchTitan MLPerf 训练的本次更新

本次将 TorchTitan 风格的 MLPerf 收敛控制接入 Primus diffusion/Flux FSDP2 路径：

- 支持 `data.eval_dataset_path`，用于构建独立 validation dataset。
- 支持 `mlperf.enable`、`mlperf.target_eval_loss`、`mlperf.eval_samples`。
- MLPerf 模式下按 `ceil(eval_samples / global_batch_size)` 触发 validation。
- validation loss `<= target_eval_loss` 时记录 time-to-converge 并 early stop。
- 增加 `mlperf_logging` 和 `mlperf_common` 依赖。
- `local_runs/run_flux_mlperf.sh` 默认启用 MLPerf 模式，并设置 `MAX_STEPS=30000`、`TARGET_ACCURACY=0.586`、`VAL_CHECK_INTERVAL=262144`。

这一步让 Primus 先对齐 TorchTitan 的 MLPerf convergence training 行为。NeMo TE FP8 recipe、NeMo Energon loader/split byte-level parity、多机 profile 自动切换仍是后续工作。

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

## 未对齐项

### 1. FP8 没有对齐

NeMo 通过 `FP8_RECIPE` 和 Hydra plugin 启用 FP8：

- `FP8_RECIPE=delayed`：`fp8=hybrid`、`amax_history_len=1024`、`amax_compute_algo=max`
- `FP8_RECIPE=delayed_short`：`fp8=hybrid`、`amax_history_len=4`、`amax_compute_algo=most_recent`

Primus 当前 diffusion FSDP2 路径看起来是 bf16 mixed precision，没有看到等价的 FLUX FP8 recipe/TE plugin 接入。因此性能和数值路径都不算对齐。

### 2. MLPerf validation / target accuracy 没有对齐

NeMo 配置：

- `TARGET_ACCURACY=0.586`
- `VAL_CHECK_INTERVAL=262144` samples
- 实际 validation step 间隔为 `ceil(262144 / global_batch_size)`
- 满足目标 loss 后记录 time-to-converge

Primus 原始 `run_flux_mlperf.sh` 只跑固定 `MAX_STEPS`，并总结 step time/throughput，没有等价的 validation、target accuracy、time-to-converge 或 early stop 逻辑。

本次更新后，Primus Flux FSDP2 路径已经接入 TorchTitan 风格的 MLPerf validation/target accuracy/early stop；但该逻辑仍需用实际 COCO eval precomputed 数据跑通验证。

### 3. 默认 stop condition 不对齐

NeMo 默认：

- `MAX_STEPS=-1`
- 主要靠 validation/target accuracy 决定收敛

Primus 原始默认：

- `MAX_STEPS=100`
- 更像 benchmark smoke/perf run，不是 MLPerf convergence run

本次更新后，`local_runs/run_flux_mlperf.sh` 默认 `MAX_STEPS=30000`，并允许 target loss 达标后提前停止。

### 4. 数据 loader 和 split 不对齐

NeMo：

- 容器内数据路径默认为 `/dataset/energon`
- `conf/data/cc12m.yaml` 使用 Energon dataset
- data module 同时支持 train/val split

Primus 原始状态：

- 默认 `DATASET_PATH=/data/cc12m_preprocessed`
- 使用 HF `load_from_disk` 风格的 precomputed dataset
- `EMPTY_ENCODINGS_PATH=/data/empty_encodings`
- 当前 FSDP2 训练循环中没有看到对应 val split 消费逻辑

即使底层样本内容来自同一 MLPerf 数据，loader、shuffle、split、validation 行为仍未完全对齐。

本次更新后 Primus 支持通过 `EVAL_DATASET_PATH` / `data.eval_dataset_path` 指定独立 eval dataset，但仍需确认该目录与 NeMo/TorchTitan 的 COCO validation 样本完全等价。

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

如果目标是单机 GBS512 性能对比，Primus 当前配置可以作为初步对齐 baseline，但需要明确它不是 MLPerf convergence parity。

如果目标是对齐 NeMo MLPerf 训练配置，建议补齐：

1. 为 Primus 增加 NeMo-style `DGXSYSTEM` 或等价 profile，自动设置 `NNODES`、`GPUS_PER_NODE`、`LOCAL_BATCH_SIZE`、`LR`、`WARMUP_STEPS`。
2. 接入或明确禁用并记录 FLUX FP8 recipe；若追求 NeMo parity，需要支持 `delayed` / `delayed_short` 的 FP8 行为。
3. 验证本次新增 validation loop：按 `VAL_CHECK_INTERVAL` samples 转换 step 间隔，并使用 `TARGET_ACCURACY=0.586` 判断收敛。
4. 补齐/审核 MLPerf logging 字段，确保 global batch、train/eval samples、optimizer hparams、validation frequency、target accuracy、time-to-converge 与目标提交要求一致。
5. 明确 Primus 数据目录是否与 NeMo `/dataset/energon` 的 train/val split 样本完全等价；如果不等价，需要补数据转换或 split 映射说明。
6. 文档中区分两种模式：
   - performance smoke/benchmark：固定 `MAX_STEPS`
   - MLPerf convergence parity：`MAX_STEPS=-1` 或足够大，靠 validation target 停止

## 当前判断

Primus FLUX 当前 **对齐了单机 8 卡 GBS512 的一部分训练超参、FLUX loss 语义，并新增了 TorchTitan 风格的 MLPerf validation/early-stop 控制**，但 **没有完全对齐 MLPerf NeMo 的完整训练配置**。

最重要的 blocker 是：

- FP8 recipe 未对齐
- 新增 validation/target accuracy 逻辑尚未用实际 eval 数据验证
- 多机 profile 超参未自动对齐
- 数据 loader/split 与 MLPerf NeMo Energon 路径未确认等价
