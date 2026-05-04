# Slurm helper scripts

## Cluster Sphere (no torchrun)

| Script | Nodes | Purpose |
|--------|-------|---------|
| [`cluster_sphere_env_single_node.sh`](cluster_sphere_env_single_node.sh) | 1 | Writes RDMA / NCCL export recommendations to Markdown (`CLUSTER_SPHERE_OUT` or cwd). |
| [`cluster_sphere_ib_write_bw_two_node.sh`](cluster_sphere_ib_write_bw_two_node.sh) | 2 | One `srun -N2 -n2`: `verbs-pair` (server on task 0, client on task 1). Requires `SERVER_RDMA_IP` (first node’s RDMA IP). |

See [Preflight docs](../../docs/preflight.md) for prerequisites (`ibv_devinfo`, `ib_write_bw`, `PYTHONPATH`) and manual two-terminal usage.

Examples:

```bash
export PRIMUS_ROOT=/path/to/Primus

# Pipeline A — one compute node
srun -N 1 -n 1 -t 00:30:00 -- bash Primus/scripts/slurm/cluster_sphere_env_single_node.sh

# Pipeline B — two nodes (after setting server IP from Pipeline A / `ip`)
export SERVER_RDMA_IP=10.224.0.73
salloc -N 2 -n 2 -t 01:00:00
./Primus/scripts/slurm/cluster_sphere_ib_write_bw_two_node.sh
```

Site-specific Slurm flags (`-p`, `-A`, `-t`) must be added by each cluster.
