# Primus Tools

Auxiliary tools for analysis, benchmarking, visualization, and diagnostics.

## Available Tools

| Tool | Directory | Description |
|------|-----------|-------------|
| **IRLens** | `tools/IRLens/` | Parses XLA HLO text dumps; prints execution skeleton with control flow, communication vs compute ops |
| **model_stats** | `tools/model_stats/` | Generates charts from the model config registry under `primus/configs/models` |
| **Pipeline Visualization** | `tools/visualization/pp_vis/` | Visualizes pipeline parallelism schedules from dumped data or PP simulator JSON via a local web UI |
| **Auto Benchmark** | `tools/auto_benchmark/` | Interactive benchmark menu for Megatron/TorchTitan on MI300X/MI355X with metrics collection |
| **Daily Report** | `tools/daily/` | Benchmark summary CSV generation used by CI workflows |
| **Docker Helpers** | `tools/docker/` | Container startup and proxy scripts |
| **Profile Trace** | `tools/profile_trace/` | Trace file merging utility |

## Per-Tool Documentation

Each tool has its own README with usage instructions:

- [IRLens README](./IRLens/README.md)
- [model_stats README](./model_stats/README.md)
- [Pipeline Visualization README](./visualization/pp_vis/README.md)
- [Auto Benchmark README](./auto_benchmark/Primus_Auto_Benchmark_README.md)
