# rocshmem_runtime — ODC runtime helpers (rocSHMEM ops now in Primus-Turbo)

The ODC rocSHMEM host/GDA ops were migrated to **Primus-Turbo**
(`primus_turbo.pytorch._C.odc_rocshmem_host` / `odc_rocshmem_gda`) and are
consumed by `odc/primitives/_rocshmem_backend.py`. The former in-tree ctypes
bindings (`host_bindings/rs_host.cpp`, `gda_backend/rs_host_gda.cpp`) and the
`build_rocshmem_backend.sh` build script have been **removed**.

The rocSHMEM **static library** (`librocshmem.a` + headers) that Primus-Turbo
links against is an **external dependency**: point the Turbo build's
`ROCSHMEM_HOME` at a rocSHMEM install. Any local rocSHMEM build tree
(`rocshmem_src/`, `rocshmem_{single,gda}/`, `*.a`, `*.so`) is `.gitignore`d and
must never be committed.

> TODO(primus-turbo): pin the exact Primus-Turbo merge commit
> (`PRIMUS_TURBO_COMMIT`) once the ODC-ops PR merges.

This directory now only holds runtime helpers:

- `scripts/run_odc.sh` — single-node ODC launcher (plain torchrun). Default P2P
  backend is `mori`; pass `rocshmem` to use the Turbo-provided ops. Set
  `PRIMUS_TURBO_PATH` to a Primus-Turbo build tree if it is not already importable.
- `scripts/cleanup_rs.sh` — leftover-process cleanup.

## Selecting the rocSHMEM backend

These are Primus config items (read by the ODC library via its runtime config):

- `odc_p2p_backend: rocshmem` — use the rocSHMEM ops (default is `mori`).
- `odc_rocshmem_gda: true` — multi-node GPU-direct (GDA) path; otherwise
  single-node XGMI IPC host path.

The ops resolve from `primus_turbo.pytorch._C`. As an escape hatch,
`odc_rocshmem_lib: <path>/librs_host_gda.so` loads an external monolithic ctypes
binding instead of the Turbo submodule (see `_rocshmem_backend.py`).

## Multi-node (GDA) correctness & deployment env

**Correctness (automatic, no env needed).** Multi-node (`n_pes > GPUs-per-node`)
defaults to `ODC_GDA_DEFER_REDUCE=1`: each micro-batch's unsharded grad is
accumulated locally, then ONE barriered reduce-scatter runs per minibatch. This
keeps the collective-barrier count equal across ranks, so variable-length
(`nopad`) packing does not deadlock. Single-node defaults to `0`.

**Cluster deployment env** (set per your cluster; not hardcoded in the repo):

| env | example | notes |
|-----|---------|-------|
| `ROCSHMEM_HCA_LIST` | `mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9` | this cluster's compute NICs (else all traffic funnels through `mlx5_0`) |
| `ROCSHMEM_GDA_PROVIDER` | `mlx5` | RDMA provider |
| `ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME` | `eth0` | uid-over-socket bootstrap NIC |
| `ROCSHMEM_HEAP_SIZE` | `8589934592` | symmetric heap RAW bytes (decimal only, no K/M/G) |
| `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | `eth0` | torch.distributed control plane |
| `NCCL_IB_GID_INDEX` | `3` | RoCE v2 GID (fabric-dependent) |
| `ODC_GDA_WARMUP_MODE` / `ODC_GDA_STRIDE_BYTES` | `strided` / `65536` | GDA connection warmup |

The GDA backend bootstraps rocSHMEM with a unique-id over a TCP socket, so the
job launches with **plain torchrun** — no MPI / mpirun.
