# Changelog

All notable changes to **Primus** will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- Support for reinforcement learning workflows (Primus-RL module)
- FlashAttention v2 integration (planned)
- TorchTitan backend for FP8 + graph mode support
- Benchmark suite for GEMM, RCCL, and end-to-end performance
- Preflight sanity checker to validate cluster and env settings
- HipBLASLt autotuning integration for optimized GEMM kernels
- Extended model support: LLaMA2, LLaMA3, DeepSeek-V2/V3
- Megatron trainer module with flexible 3D parallelism support

---

## [2025.06.18] - TorchTitan backend

### Added
- TorchTitan backend integration
- Native FP8 support
- GraphMode support for PyTorch 2.x

---

## [2025.05.16] - Benchmark Suite

### Added
- GEMM benchmark: TFLOPS, bandwidth
- RCCL benchmark: AllReduce/AllGather latency
- End-to-end profiling: tokens/sec, memory usage

---

## [2025.04.18] - Preflight Tool

### Added
- Cluster preflight checker (`primus-cli preflight`)
- ROCm, NCCL, filesystem sanity checks

---

## [2025.04.14] - GEMM Autotuning

### Added
- HipBLASLt autotuning for Megatron dense matmuls
- ROCm 6.3+ compatibility improvements

---

## [2025.04.09] - Model Configs Update

### Added
- Support for LLaMA2, LLaMA3, DeepSeek-V2/V3
- Pretrain-ready YAML configs under `primus/configs/models/`

---

## [2025.03.04] - Megatron Trainer Release

### Added
- Initial Megatron trainer integration
- Supports TP, PP, EP
- Compatible with Slurm and container launch

---

_Last updated: 2025-09-17_
