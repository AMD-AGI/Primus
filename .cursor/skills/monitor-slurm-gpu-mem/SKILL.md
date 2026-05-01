---
name: monitor-slurm-gpu-mem
description: Monitor AMD GPU VRAM usage and/or GPU utilization across multiple Slurm nodes via rocm-smi, including nodes where rocm-smi only exists inside a podman/docker container (e.g. primus-training). Use when the user asks to "monitor / watch / check rocm-smi GPU memory or util" across several nodes (e.g. "mi355-gpu-7,mi355-gpu-8,..."), wants a periodic VRAM/utilization dashboard for a Slurm reservation, looks for stragglers in a multi-node training job, or asks about per-GPU memory or busy-percent during training.
---

# Monitor Slurm GPU memory + util (rocm-smi)

Show a refreshing per-GPU table across a set of Slurm nodes — VRAM usage, GPU util, or both. Works on AMD MI300/MI355-class hosts where `rocm-smi` is typically inside the training container, not on the host.

## When to use

Trigger phrases (any language):
- "监控 rocm-smi", "监控 GPU memory / util", "watch GPU memory/util on nodes ..."
- "check VRAM on mi355-gpu-7,8,12,26"
- "find the slowest node / straggler" — `METRIC=util` or `all` highlights util gaps.
- A user explicitly lists 2+ slurm node hostnames and wants a periodic memory or util view.

If the user just wants a one-off snapshot of *one* node, prefer a direct `ssh node 'podman exec ... rocm-smi'` instead — don't reach for this skill.

## How it works

The bundled script `scripts/monitor_rocm_mem.sh`:
1. Resolves a node list from (priority order) `$NODES`, `$JOB`, or `squeue --me -t R`.
2. SSHes to each node in parallel.
3. Runs `rocm-smi --showuse --showmeminfo vram` — by default inside the `primus-training` podman container, falling back to docker, with an option to call rocm-smi directly on the host.
4. Parses `GPU use (%)`, `VRAM Total Memory (B)`, `VRAM Total Used Memory (B)` per GPU and prints a refreshing table.
5. Output depends on `METRIC`:
   - `mem`: `used/total GiB pct%`
   - `util`: `XX%`
   - `all` (default): `util% / mem%`
6. Optionally appends every sample to a CSV for later plotting.

## Quick start

Default invocation (auto-detect nodes from the user's running Slurm job, container `primus-training`, `METRIC=all`, refresh every 5s):

```bash
bash ~/.cursor/skills/monitor-slurm-gpu-mem/scripts/monitor_rocm_mem.sh
```

Explicit node list (most common when the user names them):

```bash
NODES="mi355-gpu-7 mi355-gpu-8 mi355-gpu-12 mi355-gpu-26" \
  bash ~/.cursor/skills/monitor-slurm-gpu-mem/scripts/monitor_rocm_mem.sh
```

Memory only or util only:

```bash
METRIC=mem  NODES="mi355-gpu-7 mi355-gpu-8" bash ~/.cursor/skills/monitor-slurm-gpu-mem/scripts/monitor_rocm_mem.sh
METRIC=util NODES="mi355-gpu-7 mi355-gpu-8" bash ~/.cursor/skills/monitor-slurm-gpu-mem/scripts/monitor_rocm_mem.sh
```

## Configuration knobs

All via env vars; no flags.

| Var | Default | Meaning |
|---|---|---|
| `METRIC` | `all` | One of `mem`, `util`, `all`. `all` = `util% / mem%` per cell. |
| `NODES` | (auto) | Space-separated host list. If empty, derived from `$JOB` or `squeue --me`. |
| `JOB` | — | Take nodes from `squeue -j $JOB`. |
| `CONTAINER` | `primus-training` | Container to `podman/docker exec` into. Set to empty (`CONTAINER=""`) to call host `rocm-smi`. |
| `INTERVAL` | `5` | Seconds between refreshes. |
| `ONCE` | `0` | If `1`, take one snapshot and exit (no `clear`). |
| `CSV` | — | If set, append `timestamp,node,gpu,util_pct,used_bytes,total_bytes` rows to this file. |

## Operating workflow

When the user asks to monitor:

1. Confirm or derive the node list.
   - If they named nodes inline (e.g. "mi355-gpu-7_gpu-8_gpu-12_gpu-26"), expand to a space-separated list.
   - Otherwise let the script auto-pick from `squeue --me`.
2. Sanity-check one node first to confirm `rocm-smi` is reachable:
   ```bash
   ONCE=1 NODES=<one node> bash ~/.cursor/skills/monitor-slurm-gpu-mem/scripts/monitor_rocm_mem.sh
   ```
   - If it errors with `ssh-or-rocm-smi-failed`, fall back: try `CONTAINER=""` (host rocm-smi), or check the container name with `ssh <node> podman ps`.
3. Start the continuous monitor as a long-running background command so the user can watch it without blocking the chat.
4. Report the PID and the path of the terminal/log file the user can `kill` / `tail`.

## Failure modes

- `permission denied while trying to connect to the Docker daemon socket` on host: the user lacks docker group membership — use `podman` (already the default) or run inside Slurm.
- `rocm-smi: command not found` on host with no container: the host doesn't have ROCm installed; you must exec into the training container.
- `Permission denied (publickey,password)` on every node simultaneously: the Slurm job ended/restarted between samples — Slurm typically only grants ssh while a job is allocated. Re-check `squeue --me` and resume once the job is running again.
- Empty `squeue --me`: there is no running job; require explicit `NODES=`.
- `WARNING: AMD GPU device(s) is/are in a low-power state` is informational; parse continues normally.

## Sample output

`METRIC=all` (default):
```
[2026-05-01 01:31:13]  rocm-smi util% / mem%  container=primus-training  every 5s
NODE            | GPU0          | GPU1          | GPU2          | GPU3          | GPU4          | GPU5          | GPU6          | GPU7
mi355-gpu-7     |   85% / 27.1% |   87% / 27.3% |   86% / 27.4% |   82% / 27.3% |   86% / 27.3% |   86% / 27.2% |   79% / 27.3% |   87% / 27.2%
mi355-gpu-12    |   32% / 26.7% |   32% / 27.0% |   33% / 27.0% |   31% / 27.0% |   32% / 27.0% |   32% / 26.9% |   32% / 26.9% |   31% / 26.9%   <-- straggler
```

`METRIC=mem`:
```
mi355-gpu-7     |  78.3/288.0 27.2% |  79.1/288.0 27.5% | ...
```

## Stopping the monitor

```bash
pkill -f monitor_rocm_mem.sh
# or kill the specific PID reported when starting it
```
