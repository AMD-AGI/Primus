# Multi-Node Networking Guide

Multi-node training depends on **high-bandwidth, low-latency** communication between GPUs. On AMD systems, **RCCL** (ROCm Collective Communications Library) provides GPU collectives with an API aligned to **NCCL**, so most **NCCL-prefixed** environment variables apply to RCCL as well.

This guide summarizes how Primus configures networking, how **InfiniBand**, **RoCE**, and **AINIC (AMD Infinity NIC)** fit in, and how to validate and troubleshoot cluster connectivity.

**Primary sources in this repository**

| Topic | File |
|-------|------|
| Default NCCL/RCCL and socket setup | `runner/helpers/envs/base_env.sh` |
| IB HCA detection | `runner/helpers/envs/get_nccl_ib_hca.sh` |
| Socket / interface detection | `runner/helpers/envs/get_ip_interface.sh` |
| AINIC environment (legacy script) | `runner/helpers/envs/enable_ainic.sh` |
| AINIC hook (container/CLI integration) | `runner/helpers/hooks/03_enable_ainic.sh` |
| AINIC CLI defaults | `runner/use_ainic.yaml` |
| ANP / `NCCL_NET_PLUGIN` example | `examples/run_pretrain.sh` |

---

## 1. Overview

- **Goal:** Keep gradient and parameter exchanges from becoming the bottleneck when scaling across nodes.
- **Stack:** PyTorch distributed uses the ROCm **NCCL** backend name in many configs; the implementation is **RCCL** on AMD GPUs.
- **Transports:** Common fabrics include **InfiniBand (IB)**, **RoCE** (RDMA over Converged Ethernet), and **AINIC** on supported AMD platforms. Primus scripts set or detect **HCAs**, **socket interfaces**, and optional **AINIC** tuning.

---

## 2. InfiniBand configuration

These variables are standard in NCCL/RCCL deployments. Primus seeds several from `runner/helpers/envs/base_env.sh` when that script is sourced.

| Variable | Role |
|----------|------|
| `NCCL_IB_HCA` | Selects **InfiniBand Host Channel Adapters** (device:port list). |
| `NCCL_IB_GID_INDEX` | **GID index** for the active port (RoCE and IB differ; see vendor docs). |
| `NCCL_IB_TC` | **Traffic class** for InfiniBand. |
| `NCCL_IB_FIFO_TC` | Traffic class for FIFO traffic. |
| `NCCL_IB_RETRY_CNT` | Retry count for IB operations (tune with vendor guidance). |
| `NCCL_IB_TIMEOUT` | Timeout for IB operations. |
| `NCCL_IB_QPS_PER_CONNECTION` | Queue pairs per connection. |
| `NCCL_NET_GDR_LEVEL` | **GPUDirect RDMA** level for NIC/GPU transfers. |
| `NCCL_DMABUF_ENABLE` | Use **DMA-BUF** path where supported. |

### Auto-detection in Primus

If `NCCL_IB_HCA` is **unset**, `base_env.sh` runs `runner/helpers/envs/get_nccl_ib_hca.sh`, which enumerates `/sys/class/infiniband/`, skips bonded/storage-style devices, and builds a comma-separated `device:port` list for `NCCL_IB_HCA`.

Default in `base_env.sh`:

```bash
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
```

AINIC-oriented configs often override `NCCL_IB_GID_INDEX` to `1` (see `runner/use_ainic.yaml` and `03_enable_ainic.sh`).

---

## 3. RoCE (RDMA over Converged Ethernet)

RoCE reuses much of the **IB verb** stack; the same **`NCCL_IB_*`** knobs apply.

| Variable | Typical use |
|----------|-------------|
| `NCCL_IB_ROCE_VERSION_NUM` | RoCE version (commonly **2** for RoCE v2). |

GID selection (`NCCL_IB_GID_INDEX`) and traffic classes (`NCCL_IB_TC`, `NCCL_IB_FIFO_TC`) remain important on RoCE fabrics. Follow your network team’s mapping (often **GID index 1** for RoCE v2 vs **3** for some IB fabrics—your site may differ).

---

## 4. AINIC (AMD Infinity NIC)

**AINIC** refers to AMD’s high-performance NIC path used in some clusters. Enabling it is a combination of **environment**, **container image**, and **device pass-through**.

### Enable AINIC

- Set **`USING_AINIC=1`**. The hook `runner/helpers/hooks/03_enable_ainic.sh` runs when this is set and exports AINIC-related variables back to the caller (`env.VAR=VALUE` lines).
- Use container images built for AINIC when required by your site. Examples in this repository use tags such as `docker.io/tasimage/primus:<version>-ainic` (see `examples/customer_package/` and `.github/workflows/ci.yaml`). Match the image to your ROCm and ANP bundle.

### `runner/use_ainic.yaml`

Primus CLI system defaults for AINIC-oriented runs include:

- Container **`device`** mounts: `/dev/kfd`, `/dev/dri`, `/dev/infiniband` (required for GPU and IB access in the container).
- Environment entries such as `USING_AINIC=1`, `NCCL_PXN_DISABLE=0`, and `NCCL_IB_GID_INDEX=1`.

Adjust **`NCCL_IB_GID_INDEX`** and **`container.options.image`** to match your cluster; comments in `runner/use_ainic.yaml` call this out explicitly.

### Hook vs legacy script

- **`runner/helpers/hooks/03_enable_ainic.sh`** is the supported hook path: it sets ANP/RCCL/MPI home directories, IB QoS, RoCE version, P2P channel counts, GDR flush behavior, `LD_LIBRARY_PATH` (including `libibverbs` and RCCL/ANP/MPI build paths), and related flags. Default `NCCL_IB_FIFO_TC` in the hook is **192** (the legacy `enable_ainic.sh` uses **184** with a comment noting **192** as an alternative—align with your fabric).
- **`runner/helpers/envs/enable_ainic.sh`** is the older env file; the hook comment notes it as superseded for hook-based workflows.

### RCCL network plugin (ANP)

For ANP-based networking, clusters often set **`NCCL_NET_PLUGIN`** to **`librccl-anp.so`** when that library is present under `ANP_HOME_DIR`, falling back to `librccl-net.so` otherwise—see the logic in `examples/run_pretrain.sh`. This complements the library paths from `03_enable_ainic.sh`.

### Variables commonly set for AINIC

From `03_enable_ainic.sh` / `enable_ainic.sh` (non-exhaustive):

| Variable | Purpose |
|----------|---------|
| `ANP_HOME_DIR`, `RCCL_HOME_DIR`, `MPI_HOME_DIR` | Install roots for ANP, RCCL, and Open MPI. |
| `NCCL_IB_TC`, `NCCL_IB_FIFO_TC` | Traffic classes for IB/RoCE. |
| `NCCL_IB_GID_INDEX` | Often **1** for AINIC-oriented configs in Primus examples. |
| `NCCL_IB_ROCE_VERSION_NUM` | RoCE v2. |
| `RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING` | Stricter GDR flush ordering (set to `0` in these scripts). |
| `LD_LIBRARY_PATH` | Prepends `libibverbs`, RCCL, ANP, and MPI library paths. |

---

## 5. Socket configuration

CPU-side and fallback socket traffic uses interface selection:

| Variable | Role |
|----------|------|
| `NCCL_SOCKET_IFNAME` | Interface name or pattern for NCCL socket transport (e.g. `eth0`, or `^docker0,lo` to **exclude** virtual interfaces). |
| `GLOO_SOCKET_IFNAME` | Interface for **Gloo** process groups (CPU barriers and related). |

**Primus behavior:** `base_env.sh` sets `IP_INTERFACE` via `runner/helpers/envs/get_ip_interface.sh` (fallback: first address from `hostname -I`). Both `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` default to **`IP_INTERFACE`** when unset.

**Requirement:** All nodes must agree on a **reachable** address family and interface choice; mismatched bindings are a frequent source of hangs.

---

## 6. PCIe cross-NIC (PXN)

| Variable | Default in `base_env.sh` | Meaning |
|----------|--------------------------|---------|
| `NCCL_PXN_DISABLE` | `1` | **PXN disabled** by default (saves GPU memory per comment in `base_env.sh`). |

When **`NCCL_PXN_DISABLE=0`**, **PCIe cross-NIC** is enabled: GPUs may use NICs attached to **other** PCIe switches, which can improve **multi-rail** bandwidth at the cost of **higher GPU memory** use. `runner/use_ainic.yaml` sets `NCCL_PXN_DISABLE=0` for AINIC-oriented runs.

---

## 7. Network diagnostics

### Preflight

```bash
primus-cli direct -- preflight --network
```

For multi-node (Slurm example):

```bash
primus-cli slurm srun -N 4 -- preflight --host --gpu --network
```

See `production_docs/02-user-guide/preflight.md` for flags, output locations (`output/preflight` by default), and interpretation.

Set **`PRIMUS_EXPECT_IB=1`** when InfiniBand is **required** for validation; preflight uses this in `primus/tools/preflight/network/network_standard.py`.

### RCCL benchmark

```bash
primus-cli slurm srun -N 4 -- benchmark rccl --op all_reduce --min-bytes 1M --max-bytes 128M
```

This exercises collective bandwidth and latency across a message-size sweep. See `production_docs/02-user-guide/benchmarking.md` and `primus/tools/benchmark/rccl_bench_args.py` for options (dtypes, operations, output files).

### Verbose RCCL logs

```bash
export NCCL_DEBUG=INFO
```

Use for short, controlled runs; **TRACE** can be extremely verbose.

---

## 8. Multi-node setup checklist

- **ROCm version** matches across all nodes (driver and container image).
- **`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`** (or auto-detected `IP_INTERFACE`) identify the **same logical network** on every node.
- **InfiniBand or RoCE** is up (`ibstat`, `/dev/infiniband`, kernel modules such as `ib_core` / `mlx5_core` as appropriate).
- **Firewall** allows ports required by your launcher and collective tests (`MASTER_ADDR` / `MASTER_PORT` reachable).
- **`MASTER_ADDR`** resolves and is reachable from **all** nodes.
- **`GPUS_PER_NODE`** matches physical GPUs per node.
- **Containers** mount `/dev/kfd`, `/dev/dri`, and `/dev/infiniband` when using IB/RoCE/AINIC (see `runner/use_ainic.yaml`).

---

## 9. Troubleshooting network issues

| Symptom | What to check |
|---------|----------------|
| **Timeout or hang** at init | `MASTER_ADDR` / `MASTER_PORT`, firewall, VPN, wrong `NCCL_SOCKET_IFNAME`, or inconsistent interface across nodes. |
| **Slow** collectives | IB vs Ethernet path, `NCCL_NET_GDR_LEVEL`, fabric errors, or contention; compare **`benchmark rccl`** to baseline. |
| **IB not detected** | `/dev/infiniband` missing, modules not loaded, or wrong container devices. |
| **Wrong interface** | Restrict with `NCCL_SOCKET_IFNAME=^docker0,lo` (exclude loopback and Docker bridges). |
| **GID / RoCE issues** | `NCCL_IB_GID_INDEX` vs site documentation; RoCE v2 settings (`NCCL_IB_ROCE_VERSION_NUM`). |

For a consolidated list of `NCCL_*` / `RCCL_*` variables, see `production_docs/03-configuration-reference/environment-variables.md` and the upstream [RCCL environment variables](https://rocm.docs.amd.com/projects/rccl/en/develop/api-reference/env-variables.html) documentation.

---

## Related documentation

- [Preflight diagnostics](../02-user-guide/preflight.md)
- [Benchmark suite](../02-user-guide/benchmarking.md)
- [NCCL/RCCL collective operations](./collective-operations.md)
- [Environment variables](../03-configuration-reference/environment-variables.md)
