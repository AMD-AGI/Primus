# NCCL/RCCL Collective Operations Guide

Distributed training spends a large fraction of wall time in **collective communication**: many GPUs must exchange gradients, parameters, or activations in coordinated patterns. On AMD GPUs, **RCCL** (ROCm Collective Communications Library) provides these operations with an API aligned to **NCCL** (NVIDIA Collective Communications Library), so most concepts and environment variables carry over between vendors.

This guide explains core collectives, how they map to parallelism strategies in Primus (Megatron-LM, TorchTitan), and how to benchmark and troubleshoot communication.

For Megatron knobs like `overlap_grad_reduce` and TorchTitan parallelism flags, see [Megatron parameters](../03-configuration-reference/megatron-parameters.md). For `NCCL_*` / `RCCL_*` environment variables, see [Environment variables](../03-configuration-reference/environment-variables.md).

---

## 1. Introduction

### What are collective operations?

A **collective** is a multi-party communication pattern where **every participant** (or a defined **process group**) follows the same operation: combine tensors, broadcast, scatter pieces, or exchange shards. Unlike a single **Send/Recv** pair, collectives are **synchronized** by construction and are implemented with optimized algorithms (ring, tree, etc.).

### NCCL vs RCCL

| | NCCL | RCCL |
|---|------|------|
| Vendor | NVIDIA CUDA | AMD ROCm |
| Role | GPU collective communication | GPU collective communication |
| Typical API surface | C/C++ and bindings used by PyTorch distributed | ROCm stack; PyTorch uses similar backends |

Application code written for **PyTorch distributed** (e.g. `torch.distributed`) generally selects the backend provided by the stack (**nccl** on NVIDIA, **rccl** on AMD). **Operation names and semantics** (AllReduce, AllGather, …) align so that **framework-level** code and tuning guides are largely **portable**.

### Process groups

Not every rank talks to every other rank in every step. **Process groups** define **subsets of ranks** that participate in a collective (e.g. only **tensor-parallel** ranks, only **data-parallel** ranks). Correctness and performance depend on **matching ranks** to the same group for each layer or phase of training.

---

## 2. Core Collective Operations

Below, \(n\) is the number of ranks in the process group, and \(S\) is the size of the logical tensor being reduced or moved (per-rank message size in ring formulations). **Complexity** expressions are **standard ring-style** approximations for **amount of data moved per rank** relative to \(S\); real implementations pick algorithms based on message size, topology, and environment.

---

### AllReduce

**What it does:** Each rank contributes a tensor; the **element-wise reduction** (typically **sum**) is applied across ranks, and the **full result** is **replicated** on every rank.

```
Rank 0: [a0]     Rank 1: [a1]     Rank 2: [a2]
        \           |           /
         \          |          /
          --> REDUCE(sum) <--
                    |
        All ranks: [a0+a1+a2]
```

**Complexity (ring, per-rank data moved):** about \(\frac{2(n-1)}{n} S\).

**Where used:** **Data-parallel** gradient synchronization; **tensor-parallel** partial sums; any step that needs **identical** tensors on all ranks after a reduction.

---

### AllGather

**What it does:** Each rank holds **one shard**; every rank receives the **concatenation** (or stacked layout) of **all shards**.

```
Rank 0: [x0]  Rank 1: [x1]  Rank 2: [x2]
          \       |       /
           \      |      /
            --> ALL GATHER -->
Each rank: [x0 | x1 | x2]
```

**Complexity (ring):** about \(\frac{n-1}{n} S\) **if** each rank contributes \(S/n\); more generally scales with gathering \(n-1\) other shards of comparable size.

**Where used:** **FSDP / ZeRO-3** parameter gather before forward; **TP** weight or activation assembly depending on layout.

---

### ReduceScatter

**What it does:** Conceptually **AllReduce** then **split**: each rank ends with **one shard** of the reduced result (each rank’s shard is the reduction over corresponding positions from all ranks’ inputs).

```
Inputs per rank: full-sized chunks (partial sums local)
        |
   Reduce + partition
        |
Rank i gets shard i of the fully reduced tensor
```

**Complexity (ring):** about \(\frac{n-1}{n} S\) for the common balanced case.

**Where used:** **FSDP** gradient **sharding** after backward; **sequence parallelism** with TP (activation distribution); distributed optimizer flows that **scatter** reduced pieces.

---

### AllToAll

**What it does:** Each rank sends a **distinct slice** to every other rank; every rank receives from every rank (matrix transpose of data ownership).

```
        From rank 0..n-1
              |
    +---------+---------+
    | scatter per dest |
    v         v         v
Each rank receives its column/row of the logical matrix
```

**Where used:** **Expert parallelism** token **dispatch** and **combine** in MoE; some **sparsity** and **parallel embedding** layouts.

---

### Broadcast

**What it does:** One **root** rank’s tensor is copied to all other ranks.

```
Root: [w] ----copy----> all other ranks: [w]
```

**Where used:** **Weight initialization**, **loading checkpoints** to a group, distributing hyperparameters or small metadata.

---

### Reduce

**What it does:** Like AllReduce, but the **full result** appears on **one root** rank only.

**Where used:** **Logging** or **metrics** where only rank 0 needs the scalar (e.g. reduced loss on one process).

---

### Send / Recv (point-to-point)

**What it does:** **One** rank sends a buffer to **one** other rank (possibly bidirectional with two ops).

```
Stage i ----Send/Recv----> Stage i+1
```

**Where used:** **Pipeline parallelism** activations in forward, gradients in backward; **ring attention** steps in **context parallelism** (often implemented as a ring of Send/Recv with careful ordering).

---

## 3. Which Collectives Are Used in Each Parallelism Strategy

| Parallelism | Forward Pass | Backward Pass | Optimizer Step |
|---------------|--------------|----------------|----------------|
| Data Parallel | — | AllReduce (gradients) | — |
| FSDP/ZeRO-3 | AllGather (params) | ReduceScatter (grads) + AllGather (params) | — |
| Tensor Parallel | AllReduce or AllGather+ReduceScatter | AllReduce or AllGather+ReduceScatter | — |
| Sequence Parallel | AllGather (activations) | ReduceScatter (activations) | — |
| Pipeline Parallel | Send/Recv (activations) | Send/Recv (gradients) | — |
| Expert Parallel | AllToAll (token dispatch) | AllToAll (gradient dispatch) | — |
| Context Parallel | Ring Send/Recv (KV chunks) | Ring Send/Recv | — |

Exact fusion and overlap depend on the backend (Megatron vs TorchTitan) and flags such as async TP or overlapped gradient reduction.

---

## 4. Communication Patterns in Megatron-LM

Primus trains with **Megatron-LM** patches and configurations. Typical patterns:

| Mode | Pattern |
|------|---------|
| **TP** | Column-parallel and row-parallel **linear** layers use **AllReduce** or **ReduceScatter/AllGather** sequences; with **sequence_parallel**, activations follow the Megatron **scatter/gather** pattern around TP regions. |
| **PP** | **Point-to-point** Send/Recv (or backend equivalents) between **pipeline stages** for activations and backward tensors. |
| **DP** | **AllReduce** for gradients when not using distributed optimizer; with **distributed optimizer**, **ReduceScatter**-style paths for shard-sized gradients. |
| **EP** | **AllToAll** for MoE **routing** (dispatch/combine) when experts are parallelized. |

### Overlap knobs

Megatron integrates **communication/compute overlap** options such as:

- `overlap_grad_reduce` — overlap gradient reduction with computation where supported.
- `overlap_param_gather` — overlap parameter gathering (e.g. with distributed optimizer / FSDP-style paths) with computation.

See [Megatron parameters](../03-configuration-reference/megatron-parameters.md) for defaults and compatibility with `use_distributed_optimizer`, `use_torch_fsdp2`, and checkpoint formats.

---

## 5. Communication Patterns in TorchTitan

TorchTitan (used as a backend in Primus) relies on **PyTorch** distributed primitives and **DTensor**-style layouts:

| Area | Pattern |
|------|---------|
| **FSDP / sharding** | **AllGather** / **ReduceScatter** orchestrated by **FSDP2** (`fully_shard` and related APIs) when `data_parallel_shard_degree` and sharding are enabled. |
| **TP** | Tensor parallelism is integrated with **DTensor** and model parallel helpers; schedules may use **collectives** inside module forward/backward. |
| **PP** | **Pipeline schedules** (`parallelism.pipeline_parallel_schedule`, `parallelism.pipeline_parallel_degree`) determine stage boundaries and buffering; communication is managed by the pipeline implementation. |

### Async tensor parallelism

Set `parallelism.enable_async_tensor_parallel: true` (where supported) to **overlap** TP communication with computation in eligible layers.

---

## 6. RCCL-Specific Features and Tuning

The following appear in ROCm / AMD deployments and partner integrations; availability depends on your **driver**, **RCCL build**, and **network** stack.

| Feature | Notes |
|---------|--------|
| **MSCCL** | Microsoft Collective Communication Library: **custom algorithms** and patterns; may be used when the stack is built and configured for them. |
| **MSCCL++** | User-space collective paths aimed at **lower latency** for specific patterns and hardware. |
| **ANP (AMD Network Plugin)** | Network backend integration (e.g. **AINIC**-oriented paths). Example: `NCCL_NET_PLUGIN` may point to `librccl-anp.so` or similar when installed (see Primus `examples/run_pretrain.sh` patterns). |

### Environment variables

Many deployments tune behavior with **NCCL-prefixed** variables (honored by RCCL for compatibility), for example:

- `NCCL_PROTO` — protocol selection hints.
- `NCCL_P2P_NET_CHUNKSIZE` — chunking for P2P/network paths.
- `NCCL_IB_*` — InfiniBand / RDMA-related settings when applicable.
- `NCCL_SOCKET_IFNAME` — **socket** interface selection for TCP fallback or hybrid setups.

Document your cluster’s recommended values in [Environment variables](../03-configuration-reference/environment-variables.md).

---

## 7. Benchmarking Collectives with Primus

Primus includes an **RCCL microbenchmark** suite to measure **latency and bandwidth** for common collectives across message sizes.

### Command

Invoke through **`primus-cli`** (after `runner/` / container setup per your installation):

```bash
./primus-cli direct -- benchmark rccl --op all_reduce --min-bytes 1M --max-bytes 128M
```

Useful flags (see `primus/tools/benchmark/rccl_bench_args.py`):

| Flag | Purpose |
|------|---------|
| `--op` | One or more of: `all_reduce`, `broadcast`, `reduce_scatter`, `all_gather`, `alltoall` |
| `--min-bytes`, `--max-bytes` | Sweep range (e.g. `1K`, `1M`, `128M`) |
| `--num-sizes`, `--scale` | Generated sweep (`log2` or `linear`) |
| `--dtype` | `bf16`, `fp16`, `fp32` |
| `--output-file` | Write Markdown/CSV/JSONL report (default `./rccl_report.md`) |
| `--check` | Lightweight correctness checks |

Example with multiple ops:

```bash
./primus-cli direct -- benchmark rccl --op all_reduce all_gather reduce_scatter --min-bytes 1M --max-bytes 128M
```

### Reading results

- **Bandwidth** (GB/s or similar): higher is better for large messages; compare against **peak NIC** or **GPU-GPU** limits for your topology.
- **Latency** (µs): dominates for **small** messages; important for **frequent small** collectives (e.g. some TP patterns).

Use results to spot **unexpected drops** (wrong NIC, congestion, fallback to TCP) before scaling full training.

---

## 8. Troubleshooting Communication Issues

| Symptom | Checks |
|---------|--------|
| Hangs / timeouts | Enable **`NCCL_DEBUG=INFO`** (or `TRACE` for deep dives) and inspect which collective stalls. |
| Wrong interface | Set **`NCCL_SOCKET_IFNAME`** to the intended **cluster** interface; verify with `ip link` / admin docs. |
| IB / RDMA not used | Confirm **`NCCL_IB_*`**, HCA names, and permissions; run **preflight** (below). |
| Slow AllReduce | Compare **`benchmark rccl`** to baseline; check **topology** (NVLink vs network), **contention**, **message sizes**. |

### Preflight: network validation

Primus **preflight** can aggregate host/GPU/network info:

```bash
./primus-cli preflight --network
```

Combine with GPU checks as needed:

```bash
./primus-cli preflight --gpu --network
```

Use this to confirm **RCCL/NCCL-related environment** snapshots and **connectivity expectations** before long jobs.

---

## See also

- [Parallelism strategies](./parallelism-strategies.md) — how TP, PP, DP, FSDP, EP, and CP fit together.
- [Megatron parameters](../03-configuration-reference/megatron-parameters.md)
- [Environment variables](../03-configuration-reference/environment-variables.md)
