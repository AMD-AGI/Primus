
# 🧪 Preflight Overview

**Preflight** is a diagnostic tool designed for large-scale cluster environments. Before starting distributed training, it benchmarks the **compute performance of all GPUs**, as well as **intra-node and inter-node communication bandwidth and latency**. Its primary goal is to help users identify **underperforming nodes** or **network bottlenecks** in the cluster, ensuring reliable and efficient training runs.

## Run preflight

Torch / single node:

```bash
primus-cli preflight \
  --dump-path output/preflight \
  --report-file-name preflight_report
```

### Cluster Sphere (optional)

RDMA environment hints (NCCL/GLOO/rocSHMEM) from local InfiniBand/RoCE devices, plus optional Verbs `ib_write_bw` on two ranks:

```bash
# Info report: export hints only
primus-cli preflight --cluster-sphere-env --dump-path output/preflight

# Full preflight + Cluster Sphere env + ib_write_bw on the perf report (use WORLD_SIZE=2 for the Verbs test)
primus-cli slurm srun -N 2 --ntasks-per-node=1 -- primus-cli preflight --cluster-sphere
```

Cluster Sphere logic ships inside Primus; set **`PRIMUS_CLUSTER_SPHERE_ROOT`** only if overriding that path. Install **perftest** for `ib_write_bw`.

### Cluster Sphere without torchrun (Slurm or local)

Use the **module CLI** when you want RDMA checks **without** `torchrun`, `torch.distributed`, or `primus-cli preflight` (no GPU preflight / NCCL process group). Run on a **compute node** (via `srun` / `salloc`) when possible so verbs devices match real jobs—not only on the login node.

**Setup (every invocation)**

```bash
export PRIMUS_ROOT=/path/to/Primus          # repo root containing the primus/ package
export PYTHONPATH="${PRIMUS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
```

Add your site’s Slurm flags (`-t`, `-p`, `-A`, …) where shown. Optional: **`PRIMUS_CLUSTER_SPHERE_ROOT`** only if you override the in-tree integration path.

**Pipeline A — RDMA env recommender (one host, NIC report)**

Uses `ibv_devinfo` / sysfs (no GPU, no Torch). Writes Markdown with device table, a **firmware report** (grouped by NIC firmware version), and suggested NCCL/GLOO/rocSHMEM exports.

```bash
# From a shell with PYTHONPATH set (local or already on a compute node):
python3 -m primus.tools.preflight.cluster_sphere env --markdown > cluster_sphere_env.md

# Quick text summary to stderr instead of Markdown:
python3 -m primus.tools.preflight.cluster_sphere env

# Slurm — one allocate step, write report in cwd (add -p/-A/-t as needed):
srun -N 1 -n 1 -t 00:30:00 bash -lc \
  'cd "${HOME}" && python3 -m primus.tools.preflight.cluster_sphere env --markdown > cluster_sphere_env_${SLURM_JOB_ID:-local}.md'
```

Helper script (same idea; sets `PYTHONPATH` from `PRIMUS_ROOT`):  
[`scripts/slurm/cluster_sphere_env_single_node.sh`](../../../scripts/slurm/cluster_sphere_env_single_node.sh)

**Pipeline B — Verbs `ib_write_bw` (two hosts)**

Requires [perftest](https://github.com/linux-rdma/perftest) (`ib_write_bw` on `PATH`). Default TCP port **`2000`** or set **`PRIMUS_IB_WRITE_BW_PORT`**. The client must use the server’s **`SERVER_RDMA_IP`** on the **RDMA/RoCE network** (from Pipeline A output or `ip` on the HCA netdev)—not necessarily the hostname’s management IP.

1. Allocate two nodes (`salloc -N 2 -n 2 -t 01:00:00 …`).

2. **Server (node A)** — blocks until the run finishes:

```bash
python3 -m primus.tools.preflight.cluster_sphere verbs-server --device mlx5_0
# Omit --device to pick the first device under /sys/class/infiniband
```

3. **Client (node B)** — after the server is listening:

```bash
python3 -m primus.tools.preflight.cluster_sphere verbs-client \
  --server-ip "${SERVER_RDMA_IP}" --device mlx5_0
```

**Single Slurm step (recommended)** — one `srun` launches server + client (`SERVER_RDMA_IP` = server node’s address on the RDMA network; task 0 is the first node in the allocation, task 1 the second):

```bash
export SERVER_RDMA_IP=<first_node_RDMA_IP>
srun -N 2 -n 2 -t 00:30:00 env PYTHONPATH="${PYTHONPATH}" SERVER_RDMA_IP="${SERVER_RDMA_IP}" \
  python3 -m primus.tools.preflight.cluster_sphere verbs-pair --device mlx5_0
```

Optional: `--client-delay 15` (default) gives the server time to listen before the client connects.

Example coordinating two separate `srun` steps (replace `node-a` / `node-b` with hosts from `scontrol show hostnames "$SLURM_JOB_NODELIST"`; add `--overlap` on the first line if your Slurm allows overlapping steps):

```bash
srun -N 1 -n 1 -w node-a python3 -m primus.tools.preflight.cluster_sphere verbs-server --device mlx5_0 &
sleep 15
srun -N 1 -n 1 -w node-b python3 -m primus.tools.preflight.cluster_sphere verbs-client \
  --server-ip "${SERVER_RDMA_IP}" --device mlx5_0
```

Automated helper (requires **`export SERVER_RDMA_IP=…`**):  
[`scripts/slurm/cluster_sphere_ib_write_bw_two_node.sh`](../../../scripts/slurm/cluster_sphere_ib_write_bw_two_node.sh) · index: [`scripts/slurm/README.md`](../../../scripts/slurm/README.md).

More detail: **[`docs/preflight.md`](../../../docs/preflight.md)** (*Slurm: Cluster Sphere without preflight / torchrun*).

Slurm (multi-node preflight example):

```bash
NUM_NODES=8 srun -N ${NUM_NODES} --ntasks-per-node=1 --cpus-per-task=256 \
  primus-cli preflight --dump-path output/preflight --report-file-name preflight_report
```


## 📂 Output Directory

After running **Preflight**, all test results and reports are generated under the `output/preflight` directory.

The final reports are:

- `preflight_report.md` – a Markdown version of the test report
- `preflight_report.pdf` – a PDF version of the same report

These reports summarize GPU performance, intra-node and inter-node communication results, and help identify potential issues within the cluster.

---

## 📁 Directory Structure

```bash
output/preflight
├── inter_node_comm
├── intra_node_comm
├── preflight_report.md
├── preflight_report.pdf
├── square_gemm_tflops
└── ...
```

> *Note: The exact contents may vary depending on the tests enabled during runtime.*
