# Primus tools

A quick catalog of the tools that ship with Primus and the sibling projects
around it—command-line tools, the tuning agent, ecosystem projects, and
auxiliary utilities. Each row gives a short description and a how-to starting
point; follow a tool's link for the full reference.

| Tool | Type | What it does | How to use |
|------|------|--------------|------------|
| [`train`](./pretraining.md) | CLI | Launch pretraining or post-training on any backend from a YAML configuration. | `primus-cli <mode> -- train pretrain --config <yaml>` |
| [`benchmark`](./benchmarking.md) | CLI | GEMM, RCCL, and attention microbenchmarks for hardware and stack validation. | `primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096` |
| [`preflight`](./preflight.md) | CLI | Host, GPU, and network health checks before long jobs. | `primus-cli slurm srun -N 4 -- preflight --host --gpu --network` |
| [`projection`](./projection.md) | CLI | Estimate per-GPU memory and throughput without occupying a full cluster. | `primus-cli direct -- projection both --config <yaml>` |
| [Tuning agent](./tuning-agent.md) | Agent | LLM-driven search for a near-optimal training configuration, scored by projection. | `python -m primus.agents.tuning_agent --workload <yaml> --target-cluster <yaml>` |
| [Primus-LM](../01-getting-started/quickstart.md) | Ecosystem | The training framework in this repository (multi-backend, unified CLI). | See the [Quickstart](../01-getting-started/quickstart.md) |
| [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) | Ecosystem | High-performance ROCm operators (attention, GEMM, grouped GEMM, DeepEP, FP8/FP4). | Bundled in the `rocm/primus` image; enabled via configuration flags |
| [Primus-SaFE](https://github.com/AMD-AGI/Primus-SaFE) | Ecosystem | Kubernetes-native stability, scheduling, and fault-tolerance platform. | Deployed separately on Kubernetes (Helm) |
| [IRLens](../../tools/IRLens/README.md) | Auxiliary | Parse XLA HLO dumps into a communication-vs-compute execution skeleton. | See the [README](../../tools/IRLens/README.md) |
| [model_stats](../../tools/model_stats/README.md) | Auxiliary | Chart model dimensions from the config registry. | See the [README](../../tools/model_stats/README.md) |
| [Pipeline visualization](../../tools/visualization/pp_vis/README.md) | Auxiliary | Render pipeline-parallel schedules in a local web UI. | See the [README](../../tools/visualization/pp_vis/README.md) |
| [Auto benchmark](../../tools/auto_benchmark/Primus_Auto_Benchmark_README.md) | Auxiliary | Interactive Megatron/TorchTitan benchmark menu with metrics collection. | See the [README](../../tools/auto_benchmark/Primus_Auto_Benchmark_README.md) |

---

## Related documentation

- [CLI reference](./cli-reference.md)—full launcher grammar and subcommand options.
- [Tooling](../06-developer-guide/tooling.md)—developer-guide index of the `tools/` utilities.
- [Project overview](../01-getting-started/overview.md#primus-ecosystem)—how the Primus ecosystem layers fit together.
