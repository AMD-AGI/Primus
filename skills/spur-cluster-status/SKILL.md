---
name: spur-cluster-status
description: Inspect the current Spur (AMD SLURM-compatible) cluster node-allocation state and produce a Markdown report covering the caller's account/QoS permissions, partitions and per-state node counts, per-QoS node caps vs live usage, per-QoS and per-account node usage, reservations, queue pressure, GPU capacity, and the caller's own jobs. Use when the user asks about Spur/SLURM cluster status, node allocation, which QoS/account/partition holds how many nodes, what a QoS's node limit is or how close it is to that limit, why jobs are stuck pending, how many idle nodes are available, or wants a cluster snapshot report.
---

# Spur Cluster Status Report

Generate a read-only snapshot of the Spur cluster's node-allocation state plus the
caller's account/QoS permissions, and emit a single Markdown report (with an
appendix of common commands). Spur is AMD's SLURM-compatible scheduler exposed via
`sinfo` / `squeue` / `scontrol` / `spur accounts`.

## Workflow

### Step 1: Generate the report

Run the collector from the repo root (it is read-only and takes a few seconds):

```bash
python3 .claude/skills/spur-cluster-status/scripts/spur_status.py
```

- Reports on the current user by default. To target another user: `--user <name>`.
- The script prints a complete Markdown report to stdout and never mutates cluster state.

### Step 2: Save and present

1. Save the output under the repo root (create the dir if missing):

```bash
mkdir -p output/skills
python3 .claude/skills/spur-cluster-status/scripts/spur_status.py \
  > "output/skills/spur-cluster-status-$(date +%Y%m%d.%H%M).md"
```

2. Show the report to the user. Present the tables inline; the idle-node list can be
   long, so summarize it (count + a short sample) unless the user wants the full list.
3. Print the saved file path.

### Step 3: Add insights (optional)

After the tables, add a short analysis when relevant, e.g.:
- Whether enough idle nodes exist for the user's target job size.
- Whether the user's jobs are on shared (`mix`) nodes (GPU-contention risk) vs `--exclusive`.
- Whether a low-priority QoS (e.g. `amd-burst-qos`, Prio=100, PreemptMode=cancel) is being
  used for a job that should use the account's normal QoS — burst jobs can be cancelled to
  make room for higher-priority QoS.
- **Why a job is stuck**: separate pending jobs held by `QOSGrpNodeLimit` (the QoS is at its
  cap — more hardware will not help, the cap has to be raised) from `Dependency` (waiting on
  another job), `Resources` (genuinely waiting on free nodes), and `JobHoldMaxRequeue` (the
  job exhausted its requeue budget and is held until `scontrol release <jobid>`).
- **Cap vs capacity**: caps are heavily oversubscribed (currently ~2.2x usable nodes), and
  `amd-burst-qos` is capped above the whole cluster, so it is effectively unconstrained
  while small-cap QoS starve. If a team keeps hitting `QOSGrpNodeLimit` on its own QoS but
  burst is well below its cap, that team is being pushed onto preemptible burst capacity —
  worth calling out.

## Report contents

The collector emits these sections (see `scripts/spur_status.py`). The report is
written in **English**:

1. **Account / QoS (you)** — the caller's account(s), default account, default QoS, the
   permission-model note, and the global QoS list with priority/preempt/node cap/per-user
   job caps.
2. **Partitions & Node States** — per-partition total + per-state node counts, overall
   state breakdown, utilization %, GPU capacity, idle nodelist, and drain/down nodes.
3. **QoS Node Limits vs Usage** — sorted by QoS name: running jobs, distinct nodes in
   use, the `GrpTRES=node=` cap, used %, pending jobs/nodes, how many pending jobs are
   held by `QOSGrpNodeLimit`, and `MaxSubmitPU`. Followed by the sum of all caps vs
   usable nodes (oversubscription factor), which QoS sit at their cap, which are capped
   at 0, and total demand held by the caps.
4. **Node Usage by QoS** — for running jobs, per QoS: #jobs, distinct nodes, node-job
   slots, GPUs, plus pending jobs and pending node demand.
5. **Node Usage by Account** — same breakdown grouped by account.
6. **Reservations** — name, node count, users, end time.
7. **Queue Pressure** — running vs pending job counts and pending node demand.
8. **My Jobs** — the caller's running/pending jobs.
9. **Jobs by User (all users)** — one row per user: total jobs (running/pending),
   distinct nodes held, number of accounts + which, which QoS, and which partitions.
10. **Appendix: Common Commands** — the command reference used to build the report.

## Spur quirks (important)

These are baked into the collector; keep them in mind if you extend it:

- `-o` format strings render **space-separated** regardless of literal delimiters — parse by whitespace column, not by a custom separator. `squeue` also **drops empty fields**, which shifts later columns (see `parse_jobs`).
- `spur accounts show user|account <name>` ignores the positional filter and lists everything — filter with `grep`/`awk`.
- The `sacctmgr` **binary is not on PATH**, but `spur accounts` *is* the same tool (its `--help` even prints `Usage: sacctmgr`), so policy queries do work through that entry point.
- There is **no** `scontrol show hostnames`; expand compressed nodelists locally (the script does this).
- **QoS membership is not enforced per account** — only the **account** is enforced at submit time, so any QoS from the global list is accepted. But per-QoS **capacity is enforced**: each QoS has a `GrpTRES=node=` cap, and jobs over it sit in `PENDING` with reason `QOSGrpNodeLimit`.
- `scontrol show assoc_mgr` output has **two blocks**: `QOS Records` then `Association Records`. Both contain `GrpTRES=` lines, so any parser must stop at `Association Records` or it will attribute per-account rows to the last QoS seen.
- In a TRES string, `node=N` means **unlimited** while `node=0` is a real cap of zero — do not collapse them (`amd-mlperf-training-qos` is currently capped at 0).
- The `(IN_USE)` figure in `assoc_mgr` is a **distinct** node count. Summing each job's `NumNodes` gives a larger number when two jobs of the same QoS share a node.

## Answering ad-hoc questions without the full report

For one-off questions, these are the underlying commands (all read-only):

```bash
# Per-QoS node cap + nodes currently in use, as `node=LIMIT(IN_USE)`
scontrol show assoc_mgr | awk '
  /^Association Records/ { exit }               # stop before the per-account block
  /^QOS=/ { q=$1; sub("QOS=","",q); next }
  /GrpTRES=/ { match($0, /node=[0-9N]+\([0-9]+\)/)
               if (RSTART) print q, substr($0, RSTART, RLENGTH) }' | column -t

# Caps only, in a clean parsable table (also has Priority / MaxJobsPU / MaxSubmitPU)
spur accounts show qos -P

# Who is running under a given QoS
squeue -t RUNNING -o "%i %u %a %q %T %D %N" | grep amd-burst-qos

# Which pending jobs are blocked by a QoS cap rather than by free hardware
squeue -h -t PENDING -o "%i %u %q %D %R" | grep QOSGrpNodeLimit

# True distinct node count for a QoS (deduplicates shared nodes)
squeue -h -t RUNNING -o "%q|%N" | awk -F'|' '$1=="amd-burst-qos"{print $2}' \
  | tr ',' '\n' | tr -d '[]' | sort -u | wc -l
```

Note that historical QoS data is **not** available: this cluster's `sacct` returns an
empty QoS column, so anything beyond the currently-resident queue has to come from the
cluster admins. Per-job GPU/CPU detail also needs `scontrol show job <id>` (`ReqTRES`,
`ReqGPUs`, `Exclusive`) — `scontrol` here takes **one** job ID at a time, so loop with
`xargs -P` for many jobs. When totalling resources, an `Exclusive=1` job holds the whole
node (236 CPUs / 8 GPUs) regardless of what it requested, while `Exclusive=0` jobs only
hold their requested share.

## Extending

To add a metric, add a parser + a section in `build_report()` in
`scripts/spur_status.py`, and add the underlying command to `COMMANDS_APPENDIX`
so the report's command list stays in sync. Candidate additions: top users by node
count, largest contiguous idle block for N-node jobs, `spur report cluster` historical
utilization, and per-node `CPUAlloc` from `scontrol show node`.
