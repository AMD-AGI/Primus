---
name: spur-cluster-status
description: Inspect the current Spur (AMD SLURM-compatible) cluster node-allocation state and produce a Markdown report covering the caller's account/QoS permissions, partitions and per-state node counts, per-QoS and per-account node usage, reservations, queue pressure, GPU capacity, and the caller's own jobs. Use when the user asks about Spur/SLURM cluster status, node allocation, which QoS/account/partition holds how many nodes, how many idle nodes are available, or wants a cluster snapshot report.
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
- Whether a low-priority QoS (e.g. `amd-burst-qos`, Prio=1) is being used for a job that
  should use the account's normal QoS.

## Report contents

The collector emits these sections (see `scripts/spur_status.py`). The report is
written in **English**:

1. **Account / QoS (you)** — the caller's account(s), default account, default QoS, the
   permission-model note, and the global QoS list with priority/preempt.
2. **Partitions & Node States** — per-partition total + per-state node counts, overall
   state breakdown, utilization %, GPU capacity, idle nodelist, and drain/down nodes.
3. **Node Usage by QoS** — for running jobs, per QoS: #jobs, distinct nodes, node-job
   slots, GPUs, plus pending jobs and pending node demand.
4. **Node Usage by Account** — same breakdown grouped by account.
5. **Reservations** — name, node count, users, end time.
6. **Queue Pressure** — running vs pending job counts and pending node demand.
7. **My Jobs** — the caller's running/pending jobs.
8. **Jobs by User (all users)** — one row per user: total jobs (running/pending),
   distinct nodes held, number of accounts + which, which QoS, and which partitions.
9. **Appendix: Common Commands** — the command reference used to build the report.

## Spur quirks (important)

These are baked into the collector; keep them in mind if you extend it:

- `-o` format strings render **space-separated** regardless of literal delimiters — parse by whitespace column, not by a custom separator.
- `spur accounts show user|account <name>` ignores the positional filter and lists everything — filter with `grep`/`awk`.
- `sacctmgr` policy queries are blocked ("Please ask your administrator"); use `spur accounts show qos` for the QoS list.
- There is **no** `scontrol show hostnames`; expand compressed nodelists locally (the script does this).
- The association table is empty and **QoS is not enforced per account** — only the **account** is enforced at submit time. QoS selects scheduling priority, not permission.

## Extending

To add a metric, add a parser + a section in `build_report()` in
`scripts/spur_status.py`, and add the underlying command to `COMMANDS_APPENDIX`
so the report's command list stays in sync. Candidate additions: top users by node
count, largest contiguous idle block for N-node jobs, `spur report cluster` historical
utilization, and per-node `CPUAlloc` from `scontrol show node`.
