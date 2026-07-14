# Tooling

Primus ships a set of auxiliary tools for analysis, benchmarking, visualization, installation, and diagnostics. They live under [`tools/`](../../tools/README.md) in the repository; each tool keeps its own README with detailed usage instructions. This page is a lightweight index so the tooling is discoverable from the documentation.

## Available tools

| Tool | Directory | What it does | Docs |
|------|-----------|--------------|------|
| **IRLens** | `tools/IRLens/` | Parses XLA HLO text dumps and prints an execution skeleton with control flow, separating communication vs compute ops. | [README](../../tools/IRLens/README.md) |
| **model_stats** | `tools/model_stats/` | Generates charts from the model config registry under `primus/configs/models`. | [README](../../tools/model_stats/README.md) |
| **Pipeline Visualization** | `tools/visualization/pp_vis/` | Visualizes pipeline-parallelism schedules from dumped data or PP-simulator JSON via a local web UI. | [README](../../tools/visualization/pp_vis/README.md) |
| **Auto Benchmark** | `tools/auto_benchmark/` | Interactive benchmark menu for Megatron/TorchTitan on MI300X/MI355X with metrics collection. | [README](../../tools/auto_benchmark/Primus_Auto_Benchmark_README.md) |
| **Backend Gap Report / Engineering Dashboard** | `tools/backend_gap_report/` | Generation and publishing toolchain for the shared Primus engineering dashboard and backend-gap reports. | [README](../../tools/backend_gap_report/README.md) |
| **Installation (venv)** | `tools/installation/` | Reproduces the Primus training Docker environment in a Python virtual environment (no Docker, no sudo). | [README](../../tools/installation/README.md) |
| **Daily Report** | `tools/daily/` | Benchmark summary CSV generation used by CI workflows. | — |
| **Docker Helpers** | `tools/docker/` | Container startup and proxy scripts. | — |
| **Profile Trace** | `tools/profile_trace/` | Trace-file merging utility. | — |

## Related documentation

- [Primus tools](../02-user-guide/primus-tools.md)—the full catalog of Primus tools (CLI, tuning agent, ecosystem) with how-to starting points.
- [Tools overview](../../tools/README.md)—the top-level index maintained alongside the code.
- [Profiling and observability](../04-technical-guides/profiling-and-observability.md)—how these tools fit into performance analysis.
- [Benchmarking](../02-user-guide/benchmarking.md)—running the benchmark suites the tools summarize.
