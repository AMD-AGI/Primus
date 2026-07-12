# Profiling & Observability

This guide covers how to capture and analyze performance data in Primus: the PyTorch/Kineto profiler, GPU memory snapshots, AMD's TraceLens trace analysis, ROCm memory sampling, memory/performance projection, and pipeline-schedule visualization. Parameters are grounded in `primus/configs/modules/megatron/`, `primus/configs/modules/torchtitan/pre_trainer.yaml`, and `primus/configs/modules/maxtext/pre_trainer.yaml`.

---

## 1. Torch profiler (Megatron)

Megatron training integrates the PyTorch profiler. Defaults live in `primus/configs/modules/megatron/trainer_base.yaml` and `primus_megatron_module.yaml`.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `profile` | `false` | Master switch for profiling. |
| `use_pytorch_profiler` | `false` | Use the PyTorch (Kineto) profiler path. |
| `profile_ranks` | `[0]` | Which global ranks to profile. |
| `profile_step_start` | `10` | First step to capture. |
| `profile_step_end` | `12` | Last step to capture. |
| `disable_profiler_activity_cpu` | `false` | Drop CPU-side activity to shrink traces (GPU-only trace). |
| `torch_profiler_record_shapes` | `true` | Record tensor shapes per op. |
| `torch_profiler_with_stack` | `true` | Record Python/C++ stacks (larger traces). |
| `torch_profiler_use_gzip` | `false` | Gzip the exported trace. |

Enable via CLI overrides on `train pretrain`:

```bash
./runner/primus-cli direct -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml \
  --profile True \
  --use_pytorch_profiler True \
  --profile_step_start 5 \
  --profile_step_end 6 \
  --disable_profiler_activity_cpu False
```

Keep the capture window **short** (a few steps after warmup) — traces grow quickly, especially with `with_stack` and CPU activity enabled. Load the resulting trace in [Perfetto](https://ui.perfetto.dev/) to inspect CPU/GPU overlap, kernel launch delays, and idle gaps.

---

## 2. GPU memory profiling (Megatron)

Two complementary mechanisms:

**Memory history snapshot** (`trainer_base.yaml`):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `record_memory_history` | `false` | Record the CUDA/HIP allocator history for snapshot analysis. |
| `memory_snapshot_path` | `snapshot.pickle` | Output path for the allocator snapshot. |

Load the pickle with PyTorch's memory visualizer to find fragmentation and peak allocations.

**ROCm memory sampling** (`primus_megatron_module.yaml`):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `use_rocm_mem_info` | `false` | When `true`, collect ROCm memory info via `rocm-smi` **every** iteration. |
| `use_rocm_mem_info_iters` | `[1, 2]` | When `use_rocm_mem_info=false`, only sample at these iterations. |

Also relevant: `log_memory_to_tensorboard` (`trainer_base.yaml`) writes memory metrics to TensorBoard.

---

## 3. TraceLens automated trace analysis (Megatron)

[TraceLens](https://github.com/AMD-AGI/TraceLens) turns raw profiler traces into hierarchical breakdowns (roofline/efficiency, compute-vs-memory bound kernels, communication-vs-sync separation, trace diffing). Primus can generate and optionally upload these reports. Configured in `primus/configs/modules/megatron/primus_megatron_module.yaml`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `generate_tracelens_report` | `false` | Generate TraceLens reports locally (auto-enabled when upload is on). |
| `mlflow_upload_tracelens_report` | `false` | Upload reports to MLflow (auto-enables generation, profiling, tensorboard). |
| `mlflow_tracelens_ranks` | `null` | Ranks to analyze (`null` = all; e.g. `[0, 8]` for one rank/node). |
| `mlflow_tracelens_output_format` | `xlsx` | `xlsx` (fastest), `csv`, or `all`. |
| `mlflow_tracelens_cleanup_after_upload` | `false` | Delete local reports after upload to save disk. |
| `mlflow_tracelens_auto_install` | `true` | Auto-install TraceLens if missing (set `false` to disable). |

Related profiler/log uploads: `mlflow_upload_traces` (upload raw trace files) and `mlflow_upload_logs` (upload training logs). See [Logging & experiment tracking](./logging-and-experiment-tracking.md) for MLflow setup.

---

## 4. Performance metrics to MLflow (Megatron)

`mlflow_upload_performance_metrics: false` (`primus_megatron_module.yaml`) enables a comprehensive scaling-test metric set when turned on (implicitly enabling throughput calculation):

- `perf/throughput_tflops_per_gpu`, `perf/tps_tokens_per_sec_per_gpu`, `perf/iteration_time_ms`
- `perf/{rocm,hip}_current_mem_gb`, `perf/{rocm,hip}_mem_utilization_pct`
- `perf/gpu_utilization_pct_rank{N}`, `perf/gpu_utilization_pct_avg`

> GPU utilization collection uses an `all_gather` every `log_interval`, which synchronizes ranks — keep this in mind for throughput-sensitive runs.

---

## 5. Profiling (TorchTitan)

Configured under `profiling:` in `primus/configs/modules/torchtitan/pre_trainer.yaml`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `enable_profiling` | `false` | Enable the Torch profiler. |
| `profile_freq` | `10` | Capture every N steps. |
| `save_traces_folder` | `profile_traces` | Output folder for traces. |
| `enable_memory_snapshot` | `false` | Capture GPU memory snapshots. |
| `save_memory_snapshot_folder` | `memory_snapshot` | Output folder for snapshots. |

Communication tracing is configured under `comm:` (`trace_buf_size`, `save_traces_folder: comm_traces`).

---

## 6. Profiling (MaxText)

Configured in `primus/configs/modules/maxtext/pre_trainer.yaml`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `profiler` | `xplane` | Profiler backend (XPlane traces). |
| `skip_first_n_steps_for_profiler` | `3` | Warmup steps to skip before capture. |
| `profiler_steps` | `1` | Number of steps to capture. |

---

## 7. Memory & performance projection

Project resource usage **before** launching, without consuming a full cluster. Exposed through the Primus CLI `projection` subcommand (`primus/cli/subcommands/projection.py`).

```bash
# Memory projection from an experiment config
./primus-cli direct -- projection memory --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml

# Performance projection (single-node benchmarking)
./primus-cli direct -- projection performance --config <exp>.yaml

# Performance projection scaled to N nodes (simulation)
./primus-cli direct -- projection performance --config <exp>.yaml --target-nodes 4
```

Memory projection breaks VRAM down across parameters, gradients, activations, optimizer states, and mixed-precision overhead — invaluable for MoE/ultra-large models where activations dominate. See [Projection](../02-user-guide/projection.md) for the full reference.

---

## 8. Pipeline schedule visualization

Diagnose pipeline bubbles and stage imbalance with the built-in tool `tools/visualization/pp_vis/`.

1. Dump per-rank schedule data during training with `--dump_pp_data true` (Megatron flag `dump_pp_data`, `primus_megatron_module.yaml`). Output lands under `output/pp_data/` (`config.json`, `pp_rank_*.json`).
2. Install the tool deps and run the viewer:

```bash
pip install -r tools/visualization/pp_vis/requirements.txt
python tools/visualization/pp_vis/vis.py   # open http://127.0.0.1:8988
```

Configure `task_list` in `vis.py` to point at your dumped `log_path` and the iterations to render. The tool can also visualize the PP simulator output (see `tools/visualization/pp_vis/README.md`).

---

## 9. Recommended workflow

1. **Project first** — run `projection memory` to confirm the config fits before booking GPUs.
2. **Capture a short trace** — a few steps after warmup with `profile` + `use_pytorch_profiler`.
3. **Inspect** — Perfetto for the timeline; TraceLens for automated hierarchical analysis and trace diffs.
4. **Check memory** — `record_memory_history` / ROCm sampling for fragmentation and peaks.
5. **Check pipelines** — `dump_pp_data` + `pp_vis` for bubbles and stage imbalance.

---

## See also

- [Logging & experiment tracking](./logging-and-experiment-tracking.md) — WandB / TensorBoard / MLflow setup.
- [MoE training deep-dive](./moe-training.md) — applying this workflow to sparse models.
- [Performance tuning](./performance-tuning.md) — what to change once you've found the bottleneck.
- [Projection](../02-user-guide/projection.md) and [Monitoring and logging](../05-operations/monitoring-logging.md).
