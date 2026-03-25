# Monitoring and Logging

This page summarizes how Primus configures application logging, experiment tracking (Weights and Biases, TensorBoard, MLflow), training metrics, profilers, ROCm memory probes, and how to capture a reproducible configuration snapshot.

---

## 1. Primus Logging System

Primus uses **loguru** for structured logging. Initialization wires **file sinks** (per log level) and a **stderr** sink, binds experiment and distributed context (`team`, `user`, `exp`, `module_name`, `node_ip`, `rank`, `world_size`), and installs an **intercept handler** so legacy `logging` output from frameworks such as Megatron is forwarded to loguru with consistent formatting.

**Rank-aware behavior**

- Worker processes write under `{exp_root}/logs/{module_name}/rank-{rank}/` with separate rotated files for `debug`, `info`, `warning`, and `error` (subject to `file_sink_level`).
- The launcher **master** process can use `logs/master/` when the master logger is configured with `is_head=True`.

**Levels from module configuration** (`primus/configs/modules/module_base.yaml`)

| Parameter | Default | Role |
|-----------|---------|------|
| `sink_level` | `null` | If set, overrides both file and stderr sink levels. |
| `file_sink_level` | `DEBUG` | Minimum level for file sinks when `sink_level` is unset. |
| `stderr_sink_level` | `INFO` | Minimum level for stderr when `sink_level` is unset. |

`init_worker_logger` in `primus/core/runtime/logging.py` reads `sink_level`, `file_sink_level`, and `stderr_sink_level` from the merged module config. The Megatron trainer maps `stderr_sink_level` to Megatron’s numeric `logging_level` (deprecated `logging_level` in `trainer_base.yaml` is replaced by this mapping).

**Shell / runner environment** (see `production_docs/03-configuration-reference/environment-variables.md`)

| Variable | Purpose |
|----------|---------|
| `PRIMUS_LOG_LEVEL` | Runner verbosity: `DEBUG`, `INFO`, `WARN`, `ERROR` (default `INFO`). |
| `PRIMUS_LOG_TIMESTAMP` | `1` enables timestamps on runner logs; `0` disables. |
| `PRIMUS_LOG_COLOR` | `1` enables ANSI colors when appropriate; often `0` in non-TTY contexts. |

**CLI**

- `primus-cli --debug` sets `PRIMUS_LOG_LEVEL=DEBUG` so launcher and shell logging are verbose (see `production_docs/02-user-guide/cli-reference.md`).

---

## 2. Weights and Biases

### Megatron (`primus/configs/modules/megatron/trainer_base.yaml`, `primus_megatron_module.yaml`)

Defaults in `primus_megatron_module.yaml` disable WandB; trainer fields in `trainer_base.yaml` supply names and paths when enabled.

| Parameter | Default (module / trainer) | Description |
|-----------|----------------------------|-------------|
| `disable_wandb` | `true` (`primus_megatron_module.yaml`) | Master switch; when `false`, Primus sets paths and default project/run names from experiment metadata. |
| `wandb_project` | `null` | If unset when WandB is enabled, defaults to `{work_group}_{user_name}`. |
| `wandb_exp_name` | `null` | If unset, defaults to `exp_name`. |
| `wandb_entity` | `null` | Optional WandB entity/team. |
| `wandb_save_dir` | `null` | Deprecated in favor of `{exp_root}`; artifacts use `{exp_root}/wandb`. |

**Environment**

- `WANDB_API_KEY` is **required** when WandB is enabled; the Megatron trainer emits a warning if it is missing (`primus/modules/trainer/megatron/trainer.py`).

### TorchTitan (`primus/configs/modules/torchtitan/pre_trainer.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metrics.enable_wandb` | `false` | Enables WandB in the TorchTitan metrics stack. |

When enabled, `primus/backends/torchtitan/patches/wandb_patches.py` can set `WANDB_PROJECT` and `WANDB_RUN_NAME` from Primus experiment metadata if unset. Use `WANDB_API_KEY` for authentication.

---

## 3. TensorBoard

### Megatron

**Module toggles** (`primus_megatron_module.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `disable_tensorboard` | `true` | When `false`, TensorBoard output is placed under `{exp_root}/tensorboard` (Primus overrides deprecated `tensorboard_dir` with this path). |

**Trainer** (`trainer_base.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tensorboard_log_interval` | `1` | Steps between TensorBoard writes. |
| `tensorboard_queue_size` | `1000` | Event file queue size. |
| `log_timers_to_tensorboard` | `false` | Log timer stats. |
| `log_batch_size_to_tensorboard` | `false` | Log batch size. |
| `log_learning_rate_to_tensorboard` | `true` | Log learning rate. |
| `log_validation_ppl_to_tensorboard` | `false` | Log validation perplexity. |
| `log_memory_to_tensorboard` | `false` | Log memory stats. |
| `log_world_size_to_tensorboard` | `false` | Log world size. |
| `log_loss_scale_to_tensorboard` | `true` | Log loss scale. |
| `tensorboard_dir` | `null` | Deprecated; Primus sets the directory under `exp_root`. |

**Note:** Enabling Megatron **profiling** (`profile: true`) forces `disable_tensorboard` off in `update_primus_config` so TensorBoard is available for profile-related views.

### TorchTitan (`pre_trainer.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metrics.enable_tensorboard` | `false` | Enables TensorBoard logging. |
| `metrics.save_tb_folder` | `tb` | Subfolder name (typically under the job dump directory in TorchTitan layouts). |

**Launch TensorBoard locally**

```bash
tensorboard --logdir <path-to-tensorboard-or-tb-folder>
```

Point `<path>` at the Megatron `tensorboard` directory under the experiment root, or at the TorchTitan metrics folder that contains the `save_tb_folder` subtree.

---

## 4. MLflow

MLflow integration is **Megatron-only** in the paths described here.

**Module** (`primus_megatron_module.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `disable_mlflow` | `true` | When `false`, MLflow run setup runs on the **last** global rank (`world_size - 1`). |
| `mlflow_run_name` | `null` | If unset when enabled, defaults to `{work_group}_{user_name}`. |
| `mlflow_experiment_name` | `null` | Passed to `mlflow.set_experiment` when set. |

**Startup behavior** (`primus/backends/megatron/training/global_vars.py`)

- Logs training `args` as parameters.
- Logs filtered environment variables with an `env__` prefix.
- Collects git metadata, sets MLflow source tags, and writes `system/git_metadata.json` as a run artifact.

**Environment** (typical Databricks / hosted tracking)

| Variable | Role |
|----------|------|
| `DATABRICKS_HOST` | Checked by the Megatron trainer when MLflow is enabled; a warning is printed if unset. |
| `DATABRICKS_TOKEN` | Authentication for Databricks-hosted tracking (see environment reference). |
| `MLFLOW_TRACKING_URI` | Tracking server URI; optional depending on deployment. |

---

## 5. Training Metrics

### Megatron (`trainer_base.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_interval` | `100` | Steps between standard training log lines. |
| `log_throughput` | `false` | Log throughput metrics. |
| `log_avg_skip_iterations` | `2` | Skip initial iterations when computing averages. |
| `log_avg_reset_interval` | `10` | Interval for resetting running averages. |
| `log_params_norm` | `false` | Log parameter norm. |
| `log_num_zeros_in_grad` | `false` | Log count of zero gradients. |
| `log_progress` | `false` | Progress-style logging. |
| `timing_log_level` | `0` | Timing log verbosity. |
| `timing_log_option` | `minmax` | Timing aggregation option. |

### TorchTitan (`pre_trainer.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metrics.log_freq` | `10` | Metric logging frequency (steps). |
| `metrics.disable_color_printing` | `false` | Disable colored console metrics. |
| `metrics.save_for_all_ranks` | `false` | Save metrics from every rank vs. reduced ranks. |

---

## 6. Profiling

### Megatron (`trainer_base.yaml`, `primus_megatron_module.yaml`)

| Parameter | Source | Default | Description |
|-----------|--------|---------|-------------|
| `profile` | `trainer_base.yaml` | `false` | Enables Megatron profiling path; also forces TensorBoard on when `true`. |
| `use_pytorch_profiler` | `trainer_base.yaml` | `false` | Use PyTorch profiler integration. |
| `profile_ranks` | `trainer_base.yaml` | `[0]` | Ranks to profile. |
| `profile_step_start` | `trainer_base.yaml` | `10` | First step to profile. |
| `profile_step_end` | `trainer_base.yaml` | `12` | Last step to profile. |
| `record_memory_history` | `trainer_base.yaml` | `false` | Record memory history. |
| `memory_snapshot_path` | `trainer_base.yaml` | `snapshot.pickle` | Memory snapshot file name. |
| `disable_profiler_activity_cpu` | `primus_megatron_module.yaml` | `false` | Disable CPU activities in the profiler. |
| `torch_profiler_record_shapes` | `primus_megatron_module.yaml` | `true` | Record tensor shapes. |
| `torch_profiler_with_stack` | `primus_megatron_module.yaml` | `true` | Capture Python stacks. |
| `torch_profiler_use_gzip` | `primus_megatron_module.yaml` | `false` | Gzip profiler traces. |

### TorchTitan (`pre_trainer.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `profiling.enable_profiling` | `false` | Master profiling toggle. |
| `profiling.profile_freq` | `10` | How often to capture traces. |
| `profiling.enable_memory_snapshot` | `false` | Enable memory snapshots. |
| `profiling.save_memory_snapshot_folder` | `memory_snapshot` | Output folder for snapshots. |
| `profiling.save_traces_folder` | `profile_traces` | Folder for profiler traces. |

---

## 7. ROCm Memory Monitoring

Configured in `primus/configs/modules/megatron/primus_megatron_module.yaml` and applied in the Megatron trainer when logging throughput.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_rocm_mem_info` | `false` | When `true`, collect ROCm memory information via `rocm-smi` on **every** iteration that hits the throughput logging branch. |
| `use_rocm_mem_info_iters` | `[1, 2]` | When `use_rocm_mem_info` is `false`, `rocm-smi` runs only on these iteration numbers (same branch). |

Collection is evaluated where `log_throughput` drives the extended iteration log (see `primus/modules/trainer/megatron/trainer.py`): enable `log_throughput` in `trainer_base.yaml` (or overrides) when you need ROCm memory lines in the training log.

---

## 8. Experiment Snapshots

**On disk (every run)**

- **Experiment root**: `{workspace}/{work_group}/{user_name}/{exp_name}` is created at config load time (`PrimusConfig`).
- **Per-rank logs**: `{exp_root}/logs/{module_name}/rank-{rank}/` with rotated level-specific files.
- **Checkpoints**: Megatron uses `{exp_root}/checkpoints` (trainer sets `save` to this path).
- **TensorBoard / WandB**: Under `exp_root` as described above when those features are enabled.

**MLflow** (Megatron, when enabled): Parameters, environment snapshot, and git metadata artifact provide a structured record of the run configuration and repository state.

**Merged configuration export**

Use the CLI to write the **fully merged** experiment YAML (after presets, `extends`, and overrides):

```bash
primus-cli train --config path/to/experiment.yaml --export_config /path/to/merged.yaml
```

`PrimusParser.export` writes the resolved `exp` object to YAML (`primus/core/launcher/parser.py`). This is the authoritative way to archive the exact configuration Primus will use, alongside log files and checkpoints under `exp_root`.
