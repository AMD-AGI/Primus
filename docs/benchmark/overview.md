# Benchmark Suite Overview

Primus provides an integrated benchmarking suite to evaluate performance at both the kernel and end-to-end levels. It supports operator-level profiling (e.g., GEMM, RCCL) as well as full training throughput metrics.

---

## 🎯 What It Covers

- **GEMM Benchmark**
  Measure matrix multiply (matmul) throughput in TFLOPS and bandwidth

- **RCCL Benchmark**
  Test communication performance for AllReduce, AllGather, etc.

- **End-to-End Benchmark**
  Evaluate training loop tokens/sec, memory usage, and GPU efficiency

- **Profiling Integration**
  Hook into ROCm SMI, tracing tools, or export performance summaries

---

## 🚀 Quick Start

### GEMM Benchmark

```bash
primus-cli direct -- benchmark gemm --m 4096 --n 4096 --k 4096
```

### RCCL Benchmark

```bash
primus-cli direct -- benchmark rccl --size 16777216
```

---

## 📄 Output Format

Primus benchmark results are printed to stdout and can optionally be saved as:

- Markdown (`--output result.md`)
- CSV (`--output result.csv`)
- JSON (coming soon)

Each run will report:

- Average latency (ms)
- Effective TFLOPS
- Bandwidth (GB/s)
- GPU and node metadata

---

## ⚙️ Advanced Options

- `--dtype` (fp16 / bf16 / fp32)
- `--repeat` number of runs
- `--output` save results to file
- `--topo` simulate interconnect topology (optional)

---

## 📈 Use Cases

- Validate ROCm stack on new hardware
- Detect performance regressions across driver/kernel updates
- Baseline multi-node scaling performance

---

_Last updated: 2025-09-17_
