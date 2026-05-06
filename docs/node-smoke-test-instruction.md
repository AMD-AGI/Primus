# Node-Smoke Test — Quick-Start Instructions

A short get-started guide for the per-node preflight smoke test. For the full design / aggregator section reference / implementation history, see [`node-smoke.md`](./node-smoke.md).

---

## 1. What it does

A lightweight, distributed-rendezvous-free preflight check that runs on every node in parallel under SLURM. It produces a **single PASS/FAIL verdict per node** plus SLURM-ready `passing_nodes.txt` / `failing_nodes.txt` you can pipe straight into `srun --nodelist=` / `--exclude=`.

Use it to **screen a cluster fast and exclude bad nodes before launching a real training job**. A bad GPU, NIC, wedged driver, or leaked process on any node will surface as a node FAIL — without a single global rendezvous, so a stuck node can't wedge its peers.

---

## 2. Prerequisites

| Prerequisite | How |
|---|---|
| Python venv on a shared filesystem | Same venv used by `run_preflight_direct.sh` (see [`preflight-direct.md`](./preflight-direct.md) §2). |
| `VENV_ACTIVATE` exported | `export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate` |
| Inside an existing SLURM allocation | `srun ... --ntasks-per-node=1 ...` (one task per node — wrapper handles per-GPU subprocesses internally). |

No `MASTER_ADDR`, no `MASTER_PORT`, no global rendezvous required.

---

## 3. Quick start

```bash
export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate

# Basic Tier 1 check (~5 s/GPU, ~30 s total)
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    bash runner/run_node_smoke_direct.sh

# Tier 1 + Tier 2 perf sanity (GEMM TFLOPS, HBM GB/s, local 8-GPU RCCL)
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    bash runner/run_node_smoke_direct.sh --tier2-perf

# Then re-run training, excluding any node the smoke test failed:
srun --exclude=$(paste -sd, output/preflight/failing_nodes.txt) ... your-real-job
```

Single-node sanity check (no SLURM):

```bash
bash runner/run_node_smoke_direct.sh
```

---

## 4. More examples (by configuration knob)

### 4.1 Hard-fail on partial NIC enumeration

Catches "7 of 8 RDMA NICs visible" — common cause of crashes after RoCE init.

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    bash runner/run_node_smoke_direct.sh --tier2-perf --expected-rdma-nics 8
```

### 4.2 Tighten Tier 2 perf thresholds

Reject GPUs that come in below your acceptance bar. Defaults: GEMM 600 TFLOPS, HBM 2000 GB/s, local RCCL 100 GB/s.

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    bash runner/run_node_smoke_direct.sh --tier2-perf \
        --gemm-tflops-min 700 --hbm-gbs-min 4500 --rccl-gbs-min 180
```

### 4.3 Tighten host limits

Fail nodes whose `RLIMIT_MEMLOCK` or `/dev/shm` is too small for production training.

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    bash runner/run_node_smoke_direct.sh \
        --ulimit-l-min-gb 64 --shm-min-gb 16
```

### 4.4 Custom dump path

Keep one report per smoke run instead of overwriting the default location.

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    bash runner/run_node_smoke_direct.sh --tier2-perf \
        --dump-path /shared/smoke-archive/$(date +%Y%m%d-%H%M%S)
```

### 4.5 Allow / extend the foreign-process whitelist

By default, leaked / foreign processes holding a GPU FAIL the node (most common cause of "training fails to launch on a healthy-looking node"). Allowed by default: `gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter`.

```bash
# Add a site-specific monitoring agent to the whitelist
srun ... bash runner/run_node_smoke_direct.sh \
    --allowed-procs gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter,my-monitor

# Don't fail at all on foreign processes (still reported in the markdown)
srun ... bash runner/run_node_smoke_direct.sh --allow-foreign-procs
```

### 4.6 Require specific tools

Make missing CLI tools a hard FAIL (default: warn-only).

```bash
srun ... bash runner/run_node_smoke_direct.sh --require-tools amd-smi,rocm-smi,lsof
```

### 4.7 Skip dmesg scan (containers with no privileges)

```bash
srun ... bash runner/run_node_smoke_direct.sh --skip-dmesg
```

### 4.8 Re-aggregate from existing per-node JSONs (no re-run)

Useful when you only want to refresh the markdown report, or when you've collected JSONs separately.

```bash
# From any node, no allocation needed if you're just reading local files:
NNODES=6 bash runner/run_node_smoke_direct.sh --aggregate-only --wait-timeout-sec 5
```

### 4.9 Silent mode (for CI)

Suppresses wrapper stdout, but the **final report path is still printed** and stderr / exit code are preserved.

```bash
srun ... bash runner/run_node_smoke_direct.sh --silent --tier2-perf
```

### 4.10 Combined "production-ready screen"

A representative one-shot for a production cluster screen:

```bash
srun -N "$SLURM_NNODES" --ntasks-per-node=1 \
    bash runner/run_node_smoke_direct.sh --silent --tier2-perf \
        --expected-rdma-nics 8 \
        --gemm-tflops-min 700 --hbm-gbs-min 4500 --rccl-gbs-min 180 \
        --ulimit-l-min-gb 64 --shm-min-gb 16 \
        --require-tools amd-smi,rocm-smi,lsof \
        --dump-path /shared/smoke-archive/$(date +%Y%m%d-%H%M%S)
```

---

## 5. Outputs

All written under `--dump-path` (default `output/preflight/`).

| File | Purpose |
|---|---|
| `smoke/<short-host>.json` | Per-node verdict + every collected metric. One file per node. |
| `smoke_report.md` | Human-readable cluster report (status table, drift sections, perf summary, failing-node detail). |
| `passing_nodes.txt` | Newline-separated short hostnames. Pipe into `srun --nodelist=`. |
| `failing_nodes.txt` | Newline-separated short hostnames. Pipe into `srun --exclude=`. |
| `expected_nodes.txt` | Auto-populated from `scontrol show hostnames "$SLURM_JOB_NODELIST"`. Lets the report name nodes that never reported. |

Read the cluster verdict at a glance:

```bash
head -10 output/preflight/smoke_report.md
```

Feed bad nodes into a re-run:

```bash
srun --exclude=$(paste -sd, output/preflight/failing_nodes.txt) ... your-real-job
```

---

## 6. Common knobs (cheat sheet)

| Flag | Default | When you'd change it |
|---|---|---|
| `--tier2-perf` | off | Always on for production screens — adds GEMM TFLOPS, HBM GB/s, local RCCL all-reduce. |
| `--gemm-tflops-min N` | 600 | Site-specific acceptance bar. |
| `--hbm-gbs-min N` | 2000 | Site-specific acceptance bar (MI300X healthy ≈ 4500–5000). |
| `--rccl-gbs-min N` | 100 | Site-specific acceptance bar. |
| `--expected-rdma-nics N` | unset | Hard-fail on partial NIC enumeration. |
| `--ulimit-l-min-gb GB` | 32 | Raise for production training profiles. |
| `--shm-min-gb GB` | 8 | Raise for large-batch / many-rank profiles. |
| `--allow-foreign-procs` | off | Co-tenant clusters or shared GPUs. |
| `--allowed-procs LIST` | `gpuagent,rocm-smi-daemon,amd-smi,dcgm-exporter` | Add site-specific monitoring agents. |
| `--require-tools LIST` | `""` | Fail-fast if a CLI tool is missing in PATH. |
| `--skip-dmesg` | off | Inside unprivileged containers. |
| `--dump-path DIR` | `output/preflight` | Archive each run separately. |
| `--silent` (wrapper) | off | CI / scripted runs. |
| `--aggregate-only` (wrapper) | off | Re-render report without re-running per-node checks. |

For the full flag list and the aggregator subcommand, see `python -m primus.tools.preflight.node_smoke run --help` and `... aggregate --help`, or [`node-smoke.md`](./node-smoke.md) §"Configuration knobs".

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `[run_node_smoke_direct][ERROR] VENV_ACTIVATE is not set` | `export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate` |
| Every node FAILs with `gpu_processes: ... name='N/A'` | Should no longer happen after the `/proc/<pid>/comm` fallback fix. If it does, check that `/proc/<pid>/comm` is readable on the node (`hidepid` mount?). Workaround: `--allow-foreign-procs`. |
| Some nodes never produce a JSON | Aggregator names them in `failing_nodes.txt` via `expected_nodes.txt`. If `scontrol` was unavailable, they'll appear as `<missing-N>`. |
| Tier 2 perf numbers below threshold on a known-good node | Almost always insufficient CPU cores on `srun` — pass `-c <cores-per-node>` so RCCL proxy threads have CPU. |
| Re-run on a smaller nodelist still shows the previously removed nodes as PASS | Default behavior cleans stale JSONs on rank 0. If you passed `--no-clean-dump-path`, either remove it or `rm -rf output/preflight` between runs. |

---

## 8. See also

- [`node-smoke.md`](./node-smoke.md) — full design, aggregator sections, configuration reference, implementation history.
- [`preflight-direct.md`](./preflight-direct.md) — the heavier `preflight` tool with global rendezvous and inter-node bandwidth tests.
- [`runner/run_node_smoke_direct.sh`](../runner/run_node_smoke_direct.sh) — the wrapper itself.
