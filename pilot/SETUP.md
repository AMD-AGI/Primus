# SETUP.md — User prerequisites before running Pilot

**Audience**: humans / SREs / DevOps preparing the runtime environment for Pilot.
**Counterpart**: `AGENTS.md` (covers what AI agents do *after* this setup is done).

Pilot is **not** a cluster manager. It does not pull docker images, start containers, run `salloc`, configure pyxis, or ssh between nodes. All of that is **your** job. Pilot only runs *inside* an environment that you have already prepared, and it learns about that environment from a single declarative file: **`cluster.yaml`** (schema: `schemas/cluster_config.schema.json`).

This document tells you exactly what to prepare for the two supported modes, and how to write the matching `cluster.yaml`.

---

## TL;DR

```
You prepare:                                      Then you run:
┌────────────────────────────────────┐            ┌─────────────────────────────────────┐
│ Mode A: a single-node container    │   ──→      │ pilot ... --cluster-config           │
│ Mode B: a SLURM allocation         │            │   /path/to/cluster.yaml              │
│         (+ container plumbing)     │            │                                      │
└────────────────────────────────────┘            └─────────────────────────────────────┘
```

If your environment is anything other than these two modes (raw ssh fan-out, k8s job, MPI rankfile, …), Pilot does not support it directly — wrap your launcher in a way that produces one of these two shapes first.

---

## Mode A — Single-node container

### When to use it

- You have one machine with N GPUs.
- All Pilot work fits within this machine (PREFLIGHT, PROJECTION, single-node OPTIMIZE_LOOP runs, etc.).
- Multi-node tuning is out of scope for this session.

### Step 1 — Bring up the container yourself

You decide everything here: image, mounts, network, GPU passthrough, env vars. Pilot does not influence it.

#### Minimal AMD ROCm example

```bash
docker run --rm -it \
    --name pilot-session \
    --network host \
    --device=/dev/kfd --device=/dev/dri \
    --group-add=video --ipc=host --shm-size=16G \
    --cap-add=SYS_PTRACE --security-opt=seccomp=unconfined \
    -v /path/to/Primus-dev:/workspace \
    -e PYTHONPATH=/workspace \
    rocm/mlperf-training:llama2_70b_training_6.0_2026-04-29-23-20-40 \
    bash
```

#### Minimal NVIDIA CUDA example

```bash
docker run --rm -it \
    --name pilot-session \
    --network host \
    --gpus all \
    --ipc=host --shm-size=16G \
    -v /path/to/Primus-dev:/workspace \
    -e PYTHONPATH=/workspace \
    nvcr.io/nvidia/pytorch:24.04-py3 \
    bash
```

### Step 2 — Verify the container is healthy

Inside the container:

```bash
# AMD: should print N cards
rocm-smi --showid 2>&1 | grep -c "Device Name"

# NVIDIA: should print N cards
nvidia-smi -L | wc -l

# Pilot's torch is reachable
python -c "import torch; print(torch.cuda.device_count())"
```

If any of these returns 0, **stop and fix the container** — Pilot will refuse to start with `failure.kind=CLUSTER`.

### Step 3 — Write `cluster.yaml`

Save this in your working directory (typically the workspace mount):

```yaml
# cluster.yaml
schema_version: "1.0"
cluster_id: mi355x-localhost
mode: single

runtime:
  image_label: "rocm7.2-torch2.10"   # optional; appears in ClusterProfile filenames
```

That is the **entire** file. No mounts, no image, no `--device` flags — those are already realised inside the container you just started.

### Step 4 — Hand off to Pilot

```bash
export PILOT_CLUSTER_CONFIG=/workspace/cluster.yaml
cd /workspace
pilot preflight run            # or any other Pilot tool; they all read the same file
```

---

## Mode B — Existing SLURM allocation

### When to use it

- You need cross-node measurements (RCCL curves > one node, multi-node calibration, scaled OPTIMIZE_LOOP runs).
- The cluster runs SLURM (with or without pyxis/enroot for container support).

### Step 1 — Allocate the nodes yourself

Pilot does **not** call `salloc` / `sbatch`. Allocate the cluster slice you want for this Pilot session:

```bash
salloc -N 4 \
    --gpus-per-node=8 \
    --partition=gpu \
    --time=02:00:00 \
    --job-name=pilot-session
# After it returns:
echo $SLURM_JOB_ID         # → 12345
echo $SLURM_JOB_NODELIST   # → smc[01-04]
```

You can also use `sbatch` and run Pilot from inside the batch script — same contract, just write `cluster.yaml` to record the job id.

### Step 2 — Make sure the container plumbing is in place

This is the trickiest part of multi-node and entirely your responsibility. The contract Pilot needs is: *when I run `srun --jobid=<id> -N <k> python -m pilot.tools._preflight_node_entry …`, that python is the same interpreter (with the same versions of torch / rccl / nccl / etc.) on every node.* Three common ways to achieve this:

#### Option B.1 — Cluster has pyxis configured (recommended)

The cluster admin has set sensible defaults so that `srun` automatically launches into a container image you can specify:

```bash
# From inside the salloc shell (or anywhere with $SLURM_JOB_ID exported):
srun --jobid=$SLURM_JOB_ID \
     --container-image=rocm/mlperf-training:llama2_70b_training_6.0_2026-04-29-23-20-40 \
     --container-mounts=/path/to/Primus-dev:/workspace \
     --pty bash
# You are now inside a container on the head node, with srun configured cluster-wide
# to launch into the same container image on the other nodes too.
```

When Pilot internally calls `srun --jobid=$SLURM_JOB_ID -N $k …`, pyxis will reuse the same `--container-image` defaults across the allocation.

#### Option B.2 — Pre-launched containers on every node

If you do not have pyxis, start a long-running container on each node *before* invoking Pilot, then attach to the head one:

```bash
# Use SLURM to broadcast container start to every node:
srun --jobid=$SLURM_JOB_ID -N $SLURM_NNODES --ntasks-per-node=1 \
     bash -c "docker run -d --rm --name pilot-session ... rocm/...:tag tail -f /dev/null"

# Then attach on head:
docker exec -it pilot-session bash
```

In this case, Pilot's internal `srun --jobid=...` must do `docker exec pilot-session python -m pilot.tools._preflight_node_entry ...` — wrap that in a per-cluster shim shell script and add it to PATH inside the head container, so Pilot's call resolves to it.

#### Option B.3 — Pure host / bare-metal SLURM (no containers)

If your cluster runs torch directly on the host (no containers at all), you do not need `--container-image`. Just ensure Python + Pilot dependencies are installed identically on every allocated node, and `python -m pilot.tools.preflight` runs from any one of them. `cluster.yaml` is identical to B.1 / B.2 (mode `slurm`).

### Step 3 — Verify the allocation is healthy

From the head node (inside your container if you have one):

```bash
# Allocation is RUNNING
scontrol show job $SLURM_JOB_ID | grep "JobState=RUNNING"

# All nodes are reachable via srun
srun --jobid=$SLURM_JOB_ID -N $SLURM_NNODES --ntasks-per-node=1 hostname

# Each node sees its GPUs
srun --jobid=$SLURM_JOB_ID -N $SLURM_NNODES --ntasks-per-node=1 \
     bash -c 'rocm-smi --showid 2>&1 | grep -c "Device Name"'
```

Each line should print N (or N copies of N for the last command).

### Step 4 — Write `cluster.yaml`

```yaml
# cluster.yaml
schema_version: "1.0"
cluster_id: mi355x-prod
mode: slurm

slurm:
  job_id: 12345                # ← from `echo $SLURM_JOB_ID` in step 1
  rdzv_port: 29400             # optional, default 29400; must be free on the head node

runtime:
  image_label: "rocm7.2-torch2.10"
```

Optional fields you may include for explicitness (Pilot will otherwise read them via `scontrol show job`):

```yaml
slurm:
  job_id: 12345
  nnodes: 4
  nodelist: "smc[01-04]"
  partition: gpu
```

If you specify them and they disagree with `scontrol`, Pilot fails with `failure.kind=CLUSTER`.

### Step 5 — Hand off to Pilot

```bash
export PILOT_CLUSTER_CONFIG=/workspace/cluster.yaml
pilot preflight run
```

Pilot will internally fan out via `srun --jobid=12345 -N 4 --ntasks-per-node=1 python -m pilot.tools._preflight_node_entry …`. You do not need to wrap your call in `srun` yourself.

---

## What Pilot validates the moment you invoke a tool

Every Pilot tool runs three fast-fail checks before doing any real work. They take well under a second and **do not consume tuning rounds**.

| Check | If it fails | What you should do |
|-------|-------------|--------------------|
| `cluster.yaml` resolves and validates against `schemas/cluster_config.schema.json` | `failure.kind=CLUSTER`, message *"missing or invalid cluster.yaml"* | Re-read `schemas/cluster_config.schema.json`; check schema version, mode value, `slurm.job_id` presence |
| `mode=slurm`: `scontrol show job <slurm.job_id>` returns `JobState=RUNNING` and matches your declared `nnodes`/`nodelist` | `failure.kind=CLUSTER`, message *"slurm allocation is no longer running"* (or *"nnodes mismatch"*) | Re-`salloc` (or wait for queued job to start) and update `slurm.job_id` in cluster.yaml |
| At least one GPU is visible inside the current process (`rocm-smi` or `nvidia-smi` returns ≥ 1) | `failure.kind=CLUSTER`, message *"no GPU visible; check container GPU passthrough"* | Re-launch the container with `--device=/dev/kfd /dev/dri` (ROCm) or `--gpus all` (NVIDIA) |

---

## Common pitfalls

### 1. Stale `slurm.job_id`

After your `salloc` exits (timeout, `exit`, or `scancel`), the `cluster.yaml` is dead. Pilot will detect this on the next invocation and refuse — **always re-`salloc` and update `slurm.job_id` for a new session**.

### 2. `cluster_id` collisions

Two `cluster.yaml` files with the same `cluster_id` are assumed to point at the same physical cluster. `state/cluster_profiles/<cluster_id>_*.yaml` is keyed by it. Use a unique `cluster_id` per physical cluster (`mi355x-prod`, `mi355x-staging`, `mi300x-aus-east`, …); do not reuse one across machines.

### 3. `image_label` lying

`runtime.image_label` is a *human label*. Pilot does not verify it against the actual image. Inside the ClusterProfile, Pilot records an automatically-collected `runtime_fingerprint` (rocm/torch/rccl versions). If your label says `rocm7.2-torch2.10` but the image actually has rocm 7.3, Pilot will warn but not fail. **Keep the label honest** — it is the only thing humans see in filenames.

### 4. Mounting the workspace inconsistently across nodes

In Mode B, every node must see the workspace at the **same** path (e.g. `/workspace` everywhere). Inconsistent mounts → `python -m pilot.tools._preflight_node_entry` crashes with `ModuleNotFoundError` on some nodes. Test with `srun -N $SLURM_NNODES --ntasks-per-node=1 ls /workspace/pilot/tools` before invoking Pilot.

### 5. `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` confusion

Pilot honors these inside Step 1 topology discovery (it counts only visible devices). If you set them on the head node before invoking Pilot, expect Pilot to use only that subset. To use **all** GPUs, unset them. To restrict to a subset, set them consistently on every node.

---

## Frequently asked questions

**Q: Can I have Pilot start the container / run salloc for me?**
No, by design. Coupling Pilot to your particular container runtime, scheduler, and registry would make it non-portable. The `cluster.yaml` boundary is what keeps Pilot small and predictable.

**Q: Can I run Pilot from outside a SLURM allocation, just by pointing `cluster.yaml` at a job id?**
Yes — Pilot uses `srun --jobid=<id>`, which works from any host that has the SLURM client and credentials, even outside the allocation shell. Useful for separating an Orchestrator session from the worker allocation.

**Q: What if my cluster uses k8s / Kubernetes Jobs / Volcano?**
Wrap your launcher to produce a *single-node container* environment for Pilot's session, and run a separate Pilot session per node count you want to profile. Or contribute a `mode: k8s` extension via `integrations/k8s/` (out of scope for v1).

**Q: Where do I keep `cluster.yaml`?**
Anywhere git-tracked. Common patterns: at the workspace root next to `.git`, or under `state/cluster_specs/<cluster_id>.yaml` with a stable filename. Note: `state/` is gitignored by default, so prefer the workspace root for shareable specs.

---

## Cross-references

| What you might also need | Where |
|--------------------------|-------|
| Field-by-field schema for `cluster.yaml` | `schemas/cluster_config.schema.json` |
| Universal tool input contract (agent-facing) | `AGENTS.md` §4 |
| What PREFLIGHT does after this setup is done | `skills/workflow/preflight.md` |
| Per-stage Worker prompts | `prompts/worker/<stage>.md` |
| Tool CLI reference | `tools/` (each module exposes `python -m pilot.tools.<module> --help`) |
