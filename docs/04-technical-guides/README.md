# Technical guides

Deep technical topics for advanced users.

- [Parallelism strategies](parallelism-strategies.md): DP, TP, PP, SP, CP, EP, FSDP explained
- [Parallelism configuration](parallelism-configuration.md): per-backend parallelism setup and batch size relationships
- [Collective operations](collective-operations.md): NCCL/RCCL operations and their role in each parallelism strategy
- [Performance tuning](performance-tuning.md): HipBLASLt, Primus-Turbo, FP8, MoE optimization
- [MoE training deep-dive](moe-training.md): bottlenecks and Primus-Turbo optimizations for Mixture-of-Experts models
- [Data preparation](data-preparation.md): tokenization, data formats, mock data
- [Checkpoint management](checkpoint-management.md): formats, save/load, distributed checkpointing
- [Multi-node networking](multi-node-networking.md): InfiniBand, RoCE, AINIC configuration
- [Profiling and observability](profiling-and-observability.md): Torch profiler, TraceLens, memory snapshots, projection, pp_vis
- [Logging and experiment tracking](logging-and-experiment-tracking.md): TensorBoard, WandB, MLflow setup per backend
- [Fault tolerance and elastic training](fault-tolerance-and-elastic-training.md): graceful exit, auto-resume, in-process restart, torchft
- [Determinism and reproducibility](determinism-and-reproducibility.md): deterministic mode, seeds, trade-offs
- [Diffusion models](diffusion-models/README.md): Flux diffusion architecture, data pipeline, and FP8 / MXFP4 training
- [Native SFT and LoRA](native-sft-lora.md): Megatron-native SFT/LoRA runbook (BF16 / FP8 / FP4), no Megatron-Bridge dependency

---

[← Documentation home](../README.md)
