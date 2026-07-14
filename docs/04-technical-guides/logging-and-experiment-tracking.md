# Logging and experiment tracking

This guide covers how Primus emits training metrics and logs, and how to wire up the supported experiment trackers—**TensorBoard**, **Weights & Biases (WandB)**, and **MLflow** (including Databricks-hosted MLflow)—across the Megatron and TorchTitan backends. Parameters are grounded in `primus/configs/modules/megatron/trainer_base.yaml`, `primus_megatron_module.yaml`, and `primus/configs/modules/torchtitan/pre_trainer.yaml`.

For an operations-oriented overview, see [Monitoring and logging](../05-operations/monitoring-logging.md). For required credentials/keys, see [Environment variables](../03-configuration-reference/environment-variables.md).

---

## 1. Tracker toggles at a glance (Megatron)

All three trackers are **opt-in** and disabled by default (`primus/configs/modules/megatron/primus_megatron_module.yaml`):

```yaml
disable_tensorboard: true
disable_wandb: true
disable_mlflow: true
```

Set the relevant `disable_*` to `false` to enable a tracker. Primus performs sanity checks at startup—e.g. it warns if WandB is enabled but `WANDB_API_KEY` is unset (`primus/backends/megatron/patches/args/wandb_config_patches.py`). MLflow logging is initialized in `primus/backends/megatron/training/global_vars.py`; Databricks-hosted MLflow additionally requires `DATABRICKS_HOST` (read by the `mlflow` client).

---

## 2. Console / step logging (Megatron)

Core logging cadence and content (`trainer_base.yaml`, overridden by `pre_trainer.yaml`):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `log_interval` | `100` (`pre_trainer.yaml` sets `1`) | Steps between log lines. |
| `log_throughput` | `false` (`pre_trainer.yaml` sets `true`) | Log tokens/s and TFLOP/s throughput. |
| `log_progress` | `false` | Log progress/ETA. |
| `log_params_norm` | `false` | Log parameter L2 norm. |
| `log_num_zeros_in_grad` | `false` | Log gradient sparsity. |
| `log_avg_skip_iterations` | `2` | Warmup iterations excluded from averages. |
| `log_avg_reset_interval` | `10` | Reset window for running averages. |
| `timing_log_level` | `0` | Verbosity of timer breakdowns. |
| `timing_log_option` | `minmax` | Timer aggregation across ranks. |
| `logging_level` | `null` | Python logging level override. |

---

## 3. TensorBoard (Megatron)

Enable with `disable_tensorboard: false` and set an output directory:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `tensorboard_dir` | `null` | Output directory for event files (**required** when enabled). |
| `tensorboard_log_interval` | `1` | Steps between TensorBoard writes. |
| `tensorboard_queue_size` | `1000` | Event queue size before flush. |
| `log_learning_rate_to_tensorboard` | `true` | Log LR. |
| `log_loss_scale_to_tensorboard` | `true` | Log loss scale (mixed precision). |
| `log_timers_to_tensorboard` | `false` | Log per-stage timers. |
| `log_batch_size_to_tensorboard` | `false` | Log batch size. |
| `log_memory_to_tensorboard` | `false` | Log GPU memory. |
| `log_world_size_to_tensorboard` | `false` | Log world size. |
| `log_validation_ppl_to_tensorboard` | `false` | Log validation perplexity. |

---

## 4. Weights and biases (Megatron)

Enable with `disable_wandb: false`. Configuration:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `wandb_project` | `null` | WandB project name. |
| `wandb_exp_name` | `null` | Run/experiment name. |
| `wandb_save_dir` | `null` | Local directory for WandB files. |
| `wandb_entity` | `null` | Team/entity. |

**Credentials** (see [Environment variables](../03-configuration-reference/environment-variables.md)):

```bash
export WANDB_API_KEY=...     # required when WandB is enabled
export WANDB_PROJECT=...     # optional
export WANDB_RUN_NAME=...    # optional
export WANDB_TEAM=...        # optional (entity)
```

`WANDB_API_KEY` is on the container passthrough whitelist (`runner/.primus.yaml`), so it propagates into the training container.

---

## 5. MLflow (Megatron)

Enable with `disable_mlflow: false`. Run identification and upload behavior:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `mlflow_run_name` | `null` | MLflow run name. |
| `mlflow_experiment_name` | `null` | MLflow experiment name. |
| `mlflow_upload_traces` | `false` | Upload profiler trace files. |
| `mlflow_upload_logs` | `false` | Upload training log files. |
| `mlflow_upload_performance_metrics` | `false` | Upload the comprehensive perf/memory/utilization metric set (implicitly enables throughput calc). |
| `mlflow_upload_tracelens_report` | `false` | Generate + upload TraceLens reports (see [Profiling & observability](./profiling-and-observability.md)). |

**Credentials / endpoints:**

```bash
export MLFLOW_TRACKING_URI=...     # tracking server URI
export MLFLOW_REGISTRY_URI=...     # optional model registry
# Databricks-hosted MLflow:
export DATABRICKS_HOST=...         # checked at startup when MLflow is enabled
export DATABRICKS_TOKEN=...
```

> The `mlflow_upload_*` flags are designed so MLflow stays opt-in: they only take effect when `disable_mlflow: false`.

---

## 6. One-logger (Megatron)

NVIDIA One-Logger telemetry is enabled by default in `trainer_base.yaml`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `enable_one_logger` | `true` | Enable One-Logger collection. |
| `one_logger_project` | `megatron-lm` | Project tag. |
| `one_logger_run_name` | `null` | Run name. |
| `one_logger_async` | `false` | Async upload. |
| `app_tag_run_name` / `app_tag_run_version` | `null` / `0.0.0` | Application tags. |

---

## 7. Metrics and logging (TorchTitan)

Configured under `metrics:` in `primus/configs/modules/torchtitan/pre_trainer.yaml`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `enable_tensorboard` | `false` | Enable TensorBoard logging. |
| `enable_wandb` | `false` | Enable WandB logging. |
| `log_freq` | `10` | Steps between metric logs. |
| `save_tb_folder` | `tb` | TensorBoard output subfolder. |
| `save_for_all_ranks` | `false` | Write metrics from every rank (default: rank 0 only). |
| `disable_color_printing` | `false` | Disable ANSI colors in console output. |

TorchTitan reads WandB settings from the environment (`WANDB_PROJECT`, `WANDB_RUN_NAME`, `WANDB_TEAM`) via `primus/backends/torchtitan/patches/wandb_patches.py` and `third_party/torchtitan/torchtitan/components/metrics.py`.

---

## 8. Logging (MaxText)

MaxText logging cadence is controlled by `log_period` (`primus/configs/modules/maxtext/pre_trainer.yaml`, default `100`). See [MaxText parameters](../03-configuration-reference/maxtext-parameters.md).

---

## 9. Recommended setup

1. **Local-only:** enable TensorBoard (`disable_tensorboard: false`, set `tensorboard_dir`)—no credentials required.
2. **Team tracking:** enable WandB (`disable_wandb: false`) + export `WANDB_API_KEY` and `wandb_project`/`wandb_entity`.
3. **Enterprise / scaling studies:** enable MLflow (`disable_mlflow: false`) + `MLFLOW_TRACKING_URI` (or Databricks host/token), and turn on `mlflow_upload_performance_metrics` for throughput/memory/utilization dashboards.
4. **Keep `WANDB_API_KEY` and tokens out of YAML**—pass them as environment variables (whitelisted for container passthrough). See [Security](../05-operations/security.md).

---

## Related documentation

- [Monitoring and logging](../05-operations/monitoring-logging.md)—operational view of trackers.
- [Profiling & observability](./profiling-and-observability.md)—traces, TraceLens, perf metrics.
- [Environment variables](../03-configuration-reference/environment-variables.md)—credentials and passthrough.
- [Megatron parameters](../03-configuration-reference/megatron-parameters.md) and [TorchTitan parameters](../03-configuration-reference/torchtitan-parameters.md).
