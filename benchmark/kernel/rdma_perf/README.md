# Cluster RDMA Perf Tests

Two-node `ib_write_bw` benchmark, launched via Slurm + Docker

Rank 0 runs `ib_write_bw` as the server; rank 1 connects as the client.
A small `socket_barrier.py` coordination step ensures both containers are
up before the perftest handshake fires.

## Contents

| File | Purpose |
|---|---|
| `run_slurm.sh` | sbatch entrypoint (resolves per-node IPs, pre-pulls image, runs container). |
| `run_rdma_tests.sh` | Inner container script: cross-node barrier, then `ib_write_bw` server or client. |
| `socket_barrier.py` | TCP barrier — opens a local port, polls all peers until they accept. |
| `socket_wait.py` | Optional helper to wait while a remote port stays open. Not on the critical path. |

## Prerequisites

- Slurm with `sbatch` / `srun`.
- Docker on every allocated node, with access to `/dev/infiniband`, `/dev/kfd`,
  `/dev/dri`.
- A Docker image that ships [`linux-rdma/perftest`](https://github.com/linux-rdma/perftest)
  with the `ib_write_bw` binary in `PATH`. The default
  (`lmsysorg/sglang:v0.5.7-rocm700-mi35x`) ships it.
- Healthy RoCE/IB fabric between the two nodes. See **Troubleshooting** below
  if `ib_write_bw` hangs or fails to transition QPs.

## Environment variables

All are optional; only `sbatch -N 2 -p <partition>` is strictly required.

| Variable | Default | Description |
|---|---|---|
| `DOCKER_IMAGE` | `lmsysorg/sglang:v0.5.7-rocm700-mi35x` | Image with `ib_write_bw` available. |
| `NNODES` | `${SLURM_NNODES:-2}` | Number of allocated nodes (must be 2 for client/server). |
| `PARTITION` | _(unset)_ | Slurm partition; prefer `sbatch -p` instead. |
| `IBDEVICES` | `rdma0` | HCA passed to `ib_write_bw -d`. |
| `LOG_PATH` | `${SLURM_SUBMIT_DIR}/logs` | Host directory for `ib_write_bw_node{0,1}.log` and sbatch out/err. |
| `CONTAINER_NAME` | `primus-rdma-tests` | Docker container name. |
| `MASTER_PORT` | `39566` | Reserved rendezvous port (parity with other Primus launchers; not used by perftest). |
| `BARRIER_PORT` | `5000` | TCP port used by `socket_barrier.py`. |
| `IB_WRITE_BW_PORT` | `2000` | Data port passed as `ib_write_bw -p`. |
| `EXTRA_DOCKER_ARGS` | _(empty)_ | Appended to `docker run`. Use for site-local mounts (see below). |

## Usage

### Default 2-node single-NIC run

```bash
LOG_PATH=/shared/logs sbatch -N 2 -p <partition> \
    Primus/benchmark/kernel/rdma_perf/run_slurm.sh
```

### Pick a different NIC + image

```bash
IBDEVICES=rdma3 \
DOCKER_IMAGE=my.registry/perftest:latest \
LOG_PATH=/shared/logs \
    sbatch -N 2 -p <partition> Primus/benchmark/kernel/rdma_perf/run_slurm.sh
```

### Pin specific nodes

```bash
sbatch -N 2 -p Compute-DCPT -w smci355-ccs-aus-n04-[25,29] \
    Primus/benchmark/kernel/rdma_perf/run_slurm.sh
```

## Output

Inside the container the test writes to `/run_logs/ib_write_bw_node${NODE_RANK}.log`
which is the same as `${LOG_PATH}/ib_write_bw_node{0,1}.log` on the host.
Slurm stdout/stderr land in the submit directory as `primus_rdma_perf_<jobid>.{out,err}`.

A successful run prints a `ib_write_bw` results table on the client
(node 1) similar to:

```
 #bytes  #iterations  BW peak[Gb/sec]  BW average[Gb/sec]  MsgRate[Mpps]
   ...
```

## Site-specific overrides

The default `docker run` mounts are intentionally minimal. If you may want to forward the vendor RDMA libraries,
Pass them via `EXTRA_DOCKER_ARGS`:

```bash
EXTRA_DOCKER_ARGS="\
  -v /it-share:/it-share \
  -v /it-share/models:/data \
  -v \$HOME/.ssh:/root/.ssh \
  -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
  -v /etc/libibverbs.d:/etc/libibverbs.d:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so:ro \
  -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro \
  " \
sbatch -N 2 -p <partition> Primus/benchmark/kernel/rdma_perf/run_slurm.sh
```

## Troubleshooting

### `Failed to modify QP to RTR` / "Unable to Connect the HCA's through the link"

Indicates a fabric-level problem before perftest even sends data. Common causes:

- **Wrong GID index.** Check `ibv_devinfo -v -d $IBDEVICES` and inspect
  `GID[*]` entries; only RoCEv2 GIDs that share the same VLAN/L3 routing on
  both nodes will work. Pass `-x <gid_index>` to `ib_write_bw` if the default
  GID is unreachable.
- **Rail mismatch.** The HCA you picked on node 0 must be in the same rail
  as the one on node 1. On multi-rail systems pick HCAs that share a
  switch / L3 segment.
- **Ethernet OOB unreachable.** Add `-R` to `ib_write_bw` to use RDMA-CM for
  connection establishment and bypass the OOB TCP handshake.
- **PFC / lossless config.** Verify the switch is configured for the
  expected priority class and ECN is on.

### Hang at "Waiting for nodes. . ."

The `socket_barrier.py` is unable to reach `${BARRIER_PORT}` on one of the
peer hosts. Verify:

- The container actually started on the peer (`docker ps` on that host).
- `BARRIER_PORT` is not blocked by a firewall on the management network.
- All peers can resolve each other's `hostname -I` first interface (which is
  what `run_slurm.sh` records into `IPADDRS`).

### `ib_write_bw: command not found`

Your Docker image doesn't ship perftest. Either switch to a perftest-bearing
image (e.g. `lmsysorg/sglang:v0.5.7-rocm700-mi35x`) or add an install step
via `EXTRA_DOCKER_ARGS` + an entrypoint override.

## Manual reference (without Slurm)

For ad-hoc debugging without `sbatch`, on two interactively-allocated nodes:

```bash
# Server (node A)
ib_write_bw -d rdma0 -q 4 -a --report_gbits -F -p 2000

# Client (node B), once server is listening
ib_write_bw -d rdma0 -q 4 -a --report_gbits -F <server-IP> -p 2000
```
