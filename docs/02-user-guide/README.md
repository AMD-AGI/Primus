# User guide

Core workflows and day-to-day usage.

- [Primus tools](primus-tools.md): start here—an at-a-glance catalog of all Primus tools and ecosystem projects with how-to starting points
- [CLI reference](cli-reference.md): `primus-cli` modes, flags, and subcommands
- [Configuration system](configuration-system.md): YAML configuration model, presets, overrides, inheritance
- [Pretraining](pretraining.md): pretraining **concepts**: backends, YAML structure, parallelism, configuration inventory
- [End-to-end training recipes](end-to-end-training-recipes.md): pretraining **commands**: copy-paste, GPU-arch-specific run commands
- [Megatron-LM training performance validation](megatron-lm-training.md): reproduce the published Megatron backend benchmarks on the `rocm/primus` image
- [TorchTitan training performance validation](torchtitan-training.md): reproduce the published TorchTitan backend benchmarks on the `rocm/primus` image
- [JAX MaxText training performance validation](jax-maxtext-training.md): reproduce the AMD-published MaxText benchmarks via Primus, MAD, or the standalone scripts
- [Post-training](posttraining.md): SFT and LoRA fine-tuning via Megatron Bridge
- [Node-smoke test instruction](node-smoke-test-instruction.md): screen a cluster fast and exclude bad nodes before launching a real training job
- [Preflight](preflight.md): cluster diagnostics and environment validation
- [Run preflight without a container](preflight-without-container.md): run cluster-diagnostic tool directly on the host
- [Micro-benchmarking](micro-benchmarking.md): GEMM, RCCL, and dense-GEMM benchmark suites
- [Projection](projection.md): memory and performance projection tools
- [Tuning agent](tuning-agent.md): LLM-driven search for an optimal training configuration (uses projection as an oracle)

---

[← Documentation home](../README.md)
