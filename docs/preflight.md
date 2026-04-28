# Preflight

`preflight` is Primus’ cluster diagnostic tool. It can generate a **fast info report** (host/GPU/network) and can also run **performance tests** (GEMM + intra/inter-node comm) to help spot misconfiguration or outliers before large distributed runs.

- **User-facing entry**: `primus-cli … -- preflight [args]`
- **Implementation entrypoint**: `primus/cli/subcommands/preflight.py`

## Quick start

### Info report only (fast)

```bash
primus-cli direct -- preflight --host --gpu --network
```

### Full preflight (info + perf tests)

```bash
primus-cli direct -- preflight
```

### Perf tests only

```bash
primus-cli direct -- preflight --perf-test
```

## Common usage (Slurm)

Info report only (fast):

```bash
primus-cli slurm srun -N 4 -- preflight --host --gpu --network
```

Full preflight (info + perf tests):

```bash
primus-cli slurm srun -N 4 -- preflight
```

Perf tests only:

```bash
primus-cli slurm srun -N 4 -- preflight --perf-test
```

## CLI flags

Selection:
- `--host`: host info (CPU, memory, PCIe)
- `--gpu`: GPU info
- `--network`: network info
- `--perf-test`: run perf tests only (GEMM + comm). This is slower.

Cluster Sphere (built into Primus under `primus/tools/preflight/cluster_sphere/`):
- `--cluster-sphere`: enable **both** Cluster Sphere behaviors below when dependencies are present.
- `--cluster-sphere-env`: add **RDMA env recommendations** (NCCL/GLOO/rocSHMEM export hints from local verbs devices) to the **info** report (`preflight_report*.md`).
- `--cluster-sphere-rdma-bw`: append **Verbs `ib_write_bw`** results to the **perf** report (`*_perf.md`). Intended for **`WORLD_SIZE=2`** only (two processes, typically one per node). Requires [linux-rdma/perftest](https://github.com/linux-rdma/perftest) (`ib_write_bw` on `PATH`). Optional: set `PRIMUS_IB_WRITE_BW_PORT` (default `2000`).

Optional override: set **`PRIMUS_CLUSTER_SPHERE_ROOT`** to a directory only if you need to point at a custom/fork copy of the integration (defaults to the in-tree package path).

Preflight’s existing perf tests measure **PyTorch / NCCL-style** GPU communication and use sysfs link rate as a roofline; they are **not** a substitute for raw Verbs bandwidth. Cluster Sphere **`ib_write_bw`** validates the **host RDMA path** separately.

Reporting:
- `--dump-path`: output directory (default: `output/preflight`)
- `--report-file-name`: base report name (default: `preflight_report`)
- `--disable-pdf`: disable PDF generation

Perf-test extras:
- `--plot`: generate plots (only used with `--perf-test`)

Backward compatibility:
- `--check-host/--check-gpu/--check-network` are supported as aliases for `--host/--gpu/--network`.

## Outputs

By default, outputs are written under `output/preflight`.

Typical report files:
- `preflight_report.md` / `preflight_report.pdf`: **info report** (host/GPU/network, plus Cluster Sphere env section when enabled)
- `preflight_report_perf.md` / `preflight_report_perf.pdf`: **perf report** (GEMM + comm tests, plus optional `ib_write_bw` section)

## Slurm: Cluster Sphere without preflight / torchrun

Use this when you only need **host RDMA validation** and **NCCL-style export hints**, not full GPU preflight (no `torchrun`, no `primus-cli preflight` required). Run on **compute nodes** so `/sys/class/infiniband` and `ibv_devinfo` match real jobs.

**Checklist**

- Primus checkout on nodes (or bind-mount); set `PYTHONPATH` to the repo root that contains the `primus/` package.
- `ibv_devinfo` (often `rdma-core`), `lspci` for PCI vendor detection.
- For Pipeline B: [perftest](https://github.com/linux-rdma/perftest) (`ib_write_bw` on `PATH`).
- Your site may require Slurm `-t`, `-p`, `-A`, etc.

### Pipeline A — RDMA env recommender (one node)

Produces a NIC-oriented Markdown report (GID, per-device firmware, a grouped **firmware report** when multiple NICs exist, suggested `NCCL_*` / `GLOO_*` exports).

```bash
export PRIMUS_ROOT=/path/to/Primus
export PYTHONPATH="${PRIMUS_ROOT}"

srun -N 1 -n 1 -t 00:30:00 bash -lc '
  cd "${PRIMUS_ROOT}" && python3 -m primus.tools.preflight.cluster_sphere env --markdown \
    > cluster_sphere_env_${SLURM_JOB_ID:-local}.md
'
```

Or use the helper script: [`scripts/slurm/cluster_sphere_env_single_node.sh`](../scripts/slurm/cluster_sphere_env_single_node.sh).

### Pipeline B — Verbs `ib_write_bw` (two nodes)

Server listens on **node A**; client on **node B** connects to **`SERVER_RDMA_IP`** — the address of **A on the RDMA/RoCE network**, not necessarily the management DNS name. Obtain it from Pipeline A output / `ip` on the HCA netdev.

Manual (two terminals or SSH into allocated nodes):

```bash
# Node A
export PYTHONPATH=/path/to/Primus
python3 -m primus.tools.preflight.cluster_sphere verbs-server --device rdma0

# Node B (after server is up)
python3 -m primus.tools.preflight.cluster_sphere verbs-client --server-ip <A_RDMA_IP> --device rdma0
```

Port: `2000` or `PRIMUS_IB_WRITE_BW_PORT`.

**Single `srun` (server + client)** — set `SERVER_RDMA_IP` to the **server host’s** RDMA address (task 0 / first node); task 1 runs the client after a short delay:

```bash
export SERVER_RDMA_IP=<node_A_RDMA_IP>
export PYTHONPATH=/path/to/Primus
srun -N 2 -n 2 -t 00:30:00 env PYTHONPATH="${PYTHONPATH}" SERVER_RDMA_IP="${SERVER_RDMA_IP}" \
  python3 -m primus.tools.preflight.cluster_sphere verbs-pair --device rdma0
```

Automated helper (same allocation, two nodes): [`scripts/slurm/cluster_sphere_ib_write_bw_two_node.sh`](../scripts/slurm/cluster_sphere_ib_write_bw_two_node.sh) — set `SERVER_RDMA_IP` before running.

See [`scripts/slurm/README.md`](../scripts/slurm/README.md) for examples.

---

## Notes

- For multi-node runs, use `primus-cli slurm …` (or your preferred launcher) so distributed environment variables are set correctly.
- If you only want a quick environment snapshot, prefer `--host --gpu --network`.
- `--perf-test` skips the info report; use `--cluster-sphere-env` without `--perf-test` if you only need RDMA export hints.
