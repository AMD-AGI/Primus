# User guide

Core workflows and day-to-day usage.

- [Primus tools](primus-tools.md): start here—an at-a-glance catalog of all Primus tools and ecosystem projects with how-to starting points
- [CLI reference](cli-reference.md): `primus-cli` modes, flags, and subcommands
- [Configuration system](configuration-system.md): YAML configuration model, presets, overrides, inheritance
- [Pretraining](pretraining.md): pretraining **concepts**: backends, YAML structure, parallelism, configuration inventory
- [Backend training recipes](training-recipes.md): pretraining **commands**: copy-paste, GPU-arch-specific run commands
- [Post-training](posttraining.md): SFT and LoRA fine-tuning via Megatron Bridge
- [Node-smoke test instruction](node-smoke-test-instruction.md): screen a cluster fast and exclude bad nodes before launching a real training job  
- [Preflight](preflight.md): cluster diagnostics and environment validation
- [Run preflight without a container](preflight-without-container.md): run cluster-diagnostic tool directly on the host
- [Benchmarking](benchmarking.md): GEMM, RCCL, and dense-GEMM benchmark suites
- [Projection](projection.md): memory and performance projection tools
- [Tuning agent](tuning-agent.md): LLM-driven search for an optimal training configuration (uses projection as an oracle)

---

[← Documentation home](../README.md)
