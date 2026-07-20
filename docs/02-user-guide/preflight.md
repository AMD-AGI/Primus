# Preflight

`preflight` is Primus' cluster diagnostic tool. It produces:

- A **fast info report** (host / GPU / network configuration), and
- A configurable suite of **performance tests** (GEMM TFLOPS, intra-node and inter-node communication bandwidth, P2P, ring P2P).

Use it to spot misconfiguration, hardware degradation, or perf outliers **before** committing a large distributed training run to a global rendezvous.

- **User-facing entry**: `primus-cli ... -- preflight [args]`
- **No-container launcher**: `runner/primus-cli direct -- preflight ...` — see [`preflight-without-container.md`](./preflight-without-container.md).
- **Implementation entrypoint**: `primus/cli/subcommands/preflight.py` → `primus/tools/preflight/preflight_perf_test.py`.

> Looking for a faster, distributed-rendezvous-free per-node screen? See [`node-smoke-test-instruction.md`](./node-smoke-test-instruction.md). The recommended workflow is **smoke first, preflight second** — see [§10 Comparison with node-smoke](#10-comparison-with-node-smoke).

---

## 1. Two run modes (and how preflight picks one)

Preflight has two report types, controlled by a single precedence rule:

| Mode | Triggered by | What it does |
|---|---|---|
| **Info-only** | `--host`, `--gpu`, `--network` (in any combination) | Lightweight host / GPU / network introspection. Emits a per-node report **without requiring a rendezvous**; multi-node aggregation then uses a **timeout-bounded** rendezvous (`--dist-timeout-sec`), so it never hangs indefinitely on network misconfig. |
| **Perf-only** | `--perf-test`, `--tests ...`, or `--quick` | Runs the configured perf tests under a global rendezvous. **Implied** by `--tests` and `--quick`. |
| **Default (info + perf)** | No flags at all | Runs the info report first, then every perf test. |

### Mode precedence

1. **Any of `--perf-test` / `--tests` / `--quick` is set → perf-only mode.**
   If info selectors (`--host`/`--gpu`/`--network`) are also present, they are dropped and a `WARN` is emitted (also written as a `> Note:` at the top of the perf report). To get both reports, run two invocations.
2. **Otherwise, any of `--host`/`--gpu`/`--network` is set → info-only mode.**
   Perf-only tuning knobs (e.g. `--comm-sizes-mb`) are inert in this mode and trigger a single `WARN` listing them.
3. **Otherwise (no flags) → default**: info report **first** (no rendezvous), then perf tests.

The default order ensures you always get a report even if `torch.distributed` initialization later hangs.

---

## 2. Quick start

### Info report only (fast)

```bash
primus-cli direct -- preflight --host --gpu --network
```

### Full preflight (info + every perf test)

```bash
primus-cli direct -- preflight
```

### Perf tests only

```bash
primus-cli direct -- preflight --perf-test
```

### Fast pre-launch sanity check

```bash
primus-cli direct -- preflight --quick
```

Equivalent on SLURM via `primus-cli slurm`:

```bash
primus-cli slurm srun -N 4 -- preflight --quick
```

Without a container, see [`preflight-without-container.md`](./preflight-without-container.md) for the equivalent `runner/primus-cli direct -- preflight ...` invocations.

---

## 3. Test selection (`--tests`)

`--tests` takes a comma-separated list of canonical tokens (or `all`). Implies `--perf-test`.

| Token | What it runs |
|---|---|
| `gemm` | Single-GPU square GEMM TFLOPS sweep. |
| `intra-allreduce` | Intra-node `all_reduce` bandwidth at every selected `--intra-group-sizes` x `--intra-comm-sizes-mb`. |
| `intra-alltoall` | Intra-node `all_to_all` bandwidth, same configuration matrix. |
| `inter-allreduce` | Inter-node `all_reduce` bandwidth at every selected `--inter-group-sizes` x `--inter-comm-sizes-mb`. |
| `inter-alltoall` | Inter-node `all_to_all` bandwidth, same configuration matrix. |
| `inter-p2p` | Inter-node point-to-point send/recv between fixed **adjacent 2-node pairs** (does not use `--inter-group-sizes`). Sized by `--inter-comm-sizes-mb`, falling back to `--comm-sizes-mb`. |
| `inter-ring-p2p` | Inter-node ring-pattern P2P, sized by `--ring-p2p-sizes-mb`. |
| `all` | Every token above. Default when `--tests` is omitted. |

Examples:

```bash
# GEMM only
primus-cli direct -- preflight --tests gemm

# Just the inter-node bandwidth tests
primus-cli direct -- preflight --tests inter-allreduce,inter-alltoall

# Combine with size overrides
primus-cli direct -- preflight \
    --tests gemm,inter-allreduce \
    --comm-sizes-mb 64,1024 \
    --inter-group-sizes all
```

Unknown tokens fail fast (before any rendezvous):

```text
[Primus:Preflight] ERROR: invalid perf config: --tests: unknown token 'gem'.
Valid tokens: gemm, intra-allreduce, intra-alltoall, inter-allreduce,
inter-alltoall, inter-p2p, inter-ring-p2p, all
```

---

## 4. Quick preset (`--quick`)

`--quick` is the recommended **pre-launch sanity** preset. Implies `--perf-test`. It substitutes:

| Knob | `--quick` value |
|---|---|
| `--tests` | `gemm,intra-allreduce,inter-allreduce` |
| `--comm-sizes-mb` | `64,1024` |
| `--intra-group-sizes` | `LOCAL_WORLD_SIZE` (full intra-node group only) |
| `--inter-group-sizes` | `all` (full N-node group only) |
| `warmup` | `5` |
| `iteration` | `20` |

**User-supplied flags override the preset.** For example:

```bash
# Quick preset, but with a custom size set
primus-cli direct -- preflight --quick --comm-sizes-mb 32,256
```

A full perf run with default knobs takes minutes; `--quick` typically finishes in <60s on healthy hardware.

---

## 5. Tuning the perf tests

All perf tuning knobs default to `None` so preflight can tell whether you set them. When unset, the documented defaults below apply.

### 5.1 Message sizes (collective + P2P)

| Flag | Default | Applies to |
|---|---|---|
| `--comm-sizes-mb CSV` | `2,4,8,16,32,64,128,256,512,1024` | Default for both intra- and inter-node `allreduce` / `alltoall` and `inter-p2p` when no specific override is given. |
| `--intra-comm-sizes-mb CSV` | falls back to `--comm-sizes-mb` | Override for **intra-node** `allreduce` / `alltoall`. |
| `--inter-comm-sizes-mb CSV` | falls back to `--comm-sizes-mb` | Override for **inter-node** `allreduce` / `alltoall` / `inter-p2p`. |

```bash
# Smaller, focused sweep
primus-cli direct -- preflight --comm-sizes-mb 8,128

# Different sizes for intra vs inter
primus-cli direct -- preflight \
    --tests intra-allreduce,inter-allreduce \
    --comm-sizes-mb 8,128 \
    --intra-comm-sizes-mb 4,32
```

### 5.2 Group sizes

| Flag | Default | Notes |
|---|---|---|
| `--intra-group-sizes CSV` | `2,4,8` | Each value must divide `LOCAL_WORLD_SIZE`. |
| `--inter-group-sizes CSV` | `2,4,all` | `all` means the full N-node group. Other values are subgroup sizes. **Only `inter-allreduce` and `inter-alltoall` consult this flag** — for `inter-alltoall`, every requested per-group node count is internally capped at **16** before deduping (see "Inter-node alltoall is capped at 16 nodes" below), while `inter-allreduce` uses the requested sizes unchanged. `inter-p2p` and `inter-ring-p2p` ignore this flag (fixed adjacent 2-node pairs and a full-cluster ring, respectively). |

```bash
# All-GPU intra + full N-node inter only
primus-cli direct -- preflight \
    --tests intra-allreduce,inter-allreduce \
    --intra-group-sizes 8 \
    --inter-group-sizes all
```

Validation is gated by which tests are actually selected. For example, `--tests gemm --intra-group-sizes 3` does **not** abort on a host with `LOCAL_WORLD_SIZE=8`; the intra-group constraint is only checked when an intra test is enabled.

#### Inter-node alltoall is capped at 16 nodes

Regardless of the cluster size or what `--inter-group-sizes` requests, the `inter-alltoall` test always runs on per-group node counts of at most **16**. Concretely, every requested value `G` is replaced with `min(G, 16)`, and the resulting list is deduped. Examples:

| Cluster | `--inter-group-sizes` | Requested (resolved) | `inter-alltoall` actually runs |
|---|---|---|---|
| 8 N | `all` | `[8]` | `[8]` (no change) |
| 64 N | `2,4,all` | `[2, 4, 64]` | `[2, 4, 16]` |
| 128 N | `2,4,16,32,all` | `[2, 4, 16, 32, 128]` | `[2, 4, 16]` |
| 128 N | `64` | `[64]` | `[16]` |

When the cap actually changes the list, preflight emits a single one-line WARN to stdout so the row labels in the report (e.g. `alltoall-16nodes` instead of `alltoall-128nodes`) are not surprising.

Why the cap, and why 16:

- **It matches real-world usage.** Production MoE training rarely dispatches tokens across more than ~8 nodes (for example, DeepSeek-V3's largest published configuration uses `EP=64` over 8 nodes with per-token dispatch capped at 4 nodes). A 16-node ceiling covers every published configuration with comfortable headroom.
- **It keeps the test from exhausting per-node network resources.** A large `inter-alltoall` sub-group opens a near-full mesh of connections per rank during communicator setup. Capping it at 16 keeps that well within a node's ephemeral-port budget and avoids spurious `Address already in use` failures at scale (see [§7](#7-running-on-very-large-clusters--64-nodes)).
- **Other inter-node tests are unaffected.** The cap applies only to `inter-alltoall`: `inter-allreduce` uses `--inter-group-sizes` unchanged, while `inter-p2p` and `inter-ring-p2p` don't consult it at all. All three also open far fewer simultaneous connections than alltoall.
- **It is intentionally not configurable.** This is a known-safe ceiling for the communication shapes preflight characterizes, not a tuning knob.

### 5.3 Ring P2P sizes

| Flag | Default | Applies to |
|---|---|---|
| `--ring-p2p-sizes-mb CSV` | `10,20,40,80,160` | `inter-ring-p2p` only. |

```bash
primus-cli direct -- preflight \
    --tests inter-ring-p2p \
    --ring-p2p-sizes-mb 5,20,80
```

### 5.4 Plotting

| Flag | Effect |
|---|---|
| `--plot` | After each perf test, write per-size bandwidth bar charts under `<dump-path>/<test>/` and reference them in the markdown report. |

---

## 6. Reliability knobs

Two knobs that are inert under happy-path conditions but matter at scale or on flaky networks.

### 6.1 `--comm-cleanup-delay-sec FLOAT` (default `2.0`)

Delay (seconds) inserted between destroying NCCL/RCCL process groups and creating new ones. It provides cross-rank synchronization across the destroy → setup transition, so a rank doesn't try to connect to a peer whose listener hasn't finished closing.

- Default `2.0` is essentially free and worth keeping at every cluster size.
- Set to `0` to disable the sleep entirely (barrier only).
- Bump to e.g. `5` only on very flaky networks.

```bash
# Small/medium clusters: the default is fine. Override only if you
# see port-reuse races on a very flaky network.
primus-cli slurm srun -N 8 -- preflight --quick --comm-cleanup-delay-sec 5
```

See [§7](#7-running-on-very-large-clusters--64-nodes) for guidance on running at very large scale.

### 6.2 `--dist-timeout-sec INT` (default `120`)

Timeout (seconds) for `torch.distributed.init_process_group`. If init does not complete within this many seconds, preflight writes the info report (when applicable) plus a `Distributed Init` failure section to the markdown report, prints a clear error, and exits `2` — instead of hanging indefinitely.

```bash
# Fail fast if rendezvous does not work
primus-cli direct -- preflight --perf-test --dist-timeout-sec 30
```

---

## 7. Running on very large clusters (≥ 64 nodes)

At very large scale there are a few practical considerations beyond what smaller runs encounter. A default `preflight` invocation still runs correctly at every scale we test (up to 128 nodes) without special flags — the points below are limitations to be aware of, plus recommendations that make large-cluster runs faster and easier to interpret.

### 7.1 Limitations

- **`inter-alltoall` is measured on at most 16 nodes per sub-group.** Regardless of cluster size or `--inter-group-sizes`, the alltoall test is capped at 16-node sub-groups (see [§5.2](#52-group-sizes)). This is intentional — it matches real-world MoE dispatch patterns and keeps the test from exhausting per-node network resources during communicator setup — but it does mean preflight will not report alltoall bandwidth for a larger topology. The cap applies only to `inter-alltoall`: `inter-allreduce` honors `--inter-group-sizes` unchanged, while `inter-p2p` and `inter-ring-p2p` don't use it at all.
- **preflight briefly builds many communicators.** Unlike a real training job — which creates its communicators once at startup and reuses them — preflight repeatedly builds and tears down large communicators in a short window. On a cluster with an unusually narrow ephemeral-port range this can occasionally surface as `Address already in use` during setup. It is a preflight-specific artifact rather than a training failure mode; the one-line OS fix is in [§7.3](#73-optional-os-tuning).

### 7.2 Recommended: split large runs by test family

For clusters at or beyond ~128 nodes, run one test family per invocation instead of one large run. Each invocation stays short, and it becomes easy to see which specific communication shape is degraded if a number looks off.

```bash
# 1) GPU + intra-node fabric first (cheap, no inter-node OOB churn).
primus-cli slurm srun -N 128 -- preflight \
    --tests gemm,intra-allreduce,intra-alltoall

# 2) Inter-node DP-style collectives, all-nodes group only.
primus-cli slurm srun -N 128 -- preflight \
    --tests inter-allreduce \
    --inter-group-sizes all
# Note: --inter-group-sizes all is honored here for inter-allreduce.
# For inter-alltoall it would be capped at 16 (see §5.2).
primus-cli slurm srun -N 128 -- preflight \
    --tests inter-alltoall \
    --inter-group-sizes all

# 3) Inter-node PP-style ring P2P (the test that benefits most from
#    isolation — it's the closest match to what real pipeline-parallel
#    training actually exercises).
primus-cli slurm srun -N 128 -- preflight \
    --tests inter-ring-p2p

# 4) Optional: pairwise inter-node P2P scan (useful for finding a
#    single bad link, slower because it walks many pairs).
primus-cli slurm srun -N 128 -- preflight \
    --tests inter-p2p
```

Each invocation tears down its own `WORLD` on exit and touches only one `--tests` value, so you get a per-test wall clock, can re-run a single phase in isolation, and get a separate report per run via `--report-file-name`.

### 7.3 Optional OS tuning

`preflight` runs fine with default OS settings at every scale we test. If you do hit `Address already in use` on a cluster with a narrow ephemeral-port range, widen the range — this is good general practice for any RDMA host regardless of preflight:

```bash
# Widen the per-node ephemeral port range (default ~28k → ~64k ports).
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"

# Persist across reboots:
echo 'net.ipv4.ip_local_port_range = 1024 65535' | sudo tee /etc/sysctl.d/99-large-cluster.conf
sudo sysctl --system
```

---

## 8. Reporting

| Flag | Default | Effect |
|---|---|---|
| `--dump-path DIR` | `output/preflight` | Output directory for reports + plots. |
| `--report-file-name NAME` | auto-generated `preflight-${NNODES}N-YYYYMMDD-HHMMSS` | Base name for report files. Omit to let preflight auto-generate a unique timestamped name (prevents stale leftovers from prior runs being mistaken for fresh output). Pass an explicit value when you want a stable / well-known filename. |
| `--disable-pdf` | enabled | Skip PDF generation (Markdown only). Useful when `weasyprint`/`markdown2` aren't installed. |

Output files:

| File | Produced when | Notes |
|---|---|---|
| `<name>.md` / `<name>.pdf` | Info-only mode, or default mode | Info report. |
| `<name>_perf.md` / `<name>_perf.pdf` | Perf-only mode, or default mode | Perf report (GEMM + comm). |

Only **rank 0** writes the report.

### Perf report layout

A `<name>_perf.md` produced by a default run contains, in order:

1. (Optional) `> Note:` line listing dropped info selectors.
2. `# Nodes` legend — `Node N → Hostname` table, used by every subsequent table to keep host columns compact.
3. `=======IB Bandwidth roofline (GB/s)=======` — bandwidth of the first IB device on Node 0.
4. Per enabled test, in this order: `gemm`, `intra-comm`, `inter-comm`, `inter-p2p`, `inter-ring-p2p`. Each section has a configuration line, a results table (Node / Rank / hostname / per-size GB/s), optional plots, and a per-rank wall-clock summary.
5. `[Primus:Preflight] <test> done in <T>s` lines on stdout for at-a-glance progress on the launching shell.

---

## 9. Backward-compat aliases

| Flag | Equivalent | Notes |
|---|---|---|
| `--check-host`, `--check-gpu`, `--check-network` | `--host`, `--gpu`, `--network` | Same behavior. Keep working for older scripts. |
| `--no-split-nodes-subgroup` | `--inter-group-sizes all` **and** drops `inter-p2p` | Pre-`--tests`/`--inter-group-sizes` alias. Use the new flags in new scripts. |

---

## 10. Comparison with node-smoke

| Aspect | `node-smoke` | `preflight` |
|---|---|---|
| Rendezvous | None — every node independent | Global `torch.distributed` |
| Wall clock | ~30–60 s for 6 nodes (Tier 1+2) | Minutes; scales with N for inter-node tests |
| Granularity | Per-node PASS/FAIL | Per-rank measurements (no auto-fail by default) |
| Inter-node bandwidth matrix | Not tested (intentionally) | Yes (allreduce/alltoall/p2p/ring-p2p) |
| Drift detection | Yes (versions, NIC firmware, port count) | No |
| Host limits / RDMA roll-call | Yes (hard fail) | Reported via `collect_*_info` only |
| Output format | Per-node JSON + cluster md + SLURM-ready txt | Markdown + PDF |

**Recommended workflow**: run `node-smoke` first to exclude broken nodes, then run `preflight` on the surviving set to get cross-node bandwidth measurements. See [`node-smoke-test-instruction.md`](./node-smoke-test-instruction.md) §4 ("Quick start") for the integration commands.

---

## 11. Validation & error handling

Preflight resolves the perf config **before** any distributed rendezvous. This means typos and bad sizes/group-sizes fail in seconds, not after a 120s NCCL init:

```text
[Primus:Preflight] ERROR: invalid perf config: --tests: unknown token 'gem'.
[Primus:Preflight] ERROR: invalid perf config:
    --intra-group-sizes: [3] do not divide LOCAL_WORLD_SIZE=8
[Primus:Preflight] ERROR: invalid perf config: --comm-sizes-mb: values must be positive (got 0)
```

In info-only mode, perf-only tuning knobs trigger a single warning so you notice them but they don't abort:

```text
[Primus:Preflight] WARN: --comm-sizes-mb,--intra-group-sizes have no effect
in info-only mode (no --perf-test/--tests/--quick).
```

In default mode where info selectors are dropped because perf intent was set, the preserved warning is also written into the perf report header:

```text
> Note: info selectors --host were dropped because perf mode
> (--perf-test/--tests/--quick) takes precedence. Run them in a separate
> invocation if you want both reports.
```

---

## 12. Operational tips

- **For multi-node runs, always use `primus-cli slurm` or `primus-cli direct` under `srun`** so distributed environment variables (`NNODES` / `NODE_RANK` / `MASTER_ADDR`) are set correctly.
- **Make sure slurm requests GPU resources**. For some clusters, you may need to explicitly request GPU resources with `srun -N <nodes> --gpus-per-node=<gpus-per-node>`.
- **Insufficient CPU cores cause >30x perf slowdowns** — pass `srun -c <cores-per-node>` so RCCL's network proxy threads have CPU to spawn on. Verify with `srun -N 1 --gpus-per-node=8 bash -c 'nproc'`.
- **For a quick environment snapshot**, prefer `--host --gpu --network` — you always get a local per-node report even on a broken network, and any multi-node aggregation is timeout-bounded (`--dist-timeout-sec`), so the command never hangs.
- **Between each communication test phase**, preflight performs a global barrier + `--comm-cleanup-delay-sec` sleep (default 2 s) for cross-rank sync across the destroy → setup transition. The default works at every cluster size we test up to 128 nodes. See [§7](#7-running-on-very-large-clusters--64-nodes) for large-cluster guidance.
- **For pre-launch screening of a large cluster**, the recommended sequence is:
  1. `node-smoke` to prune broken nodes (`failing_nodes.txt`).
  2. `preflight --quick` on the surviving nodes for the perf sanity numbers.
  3. `preflight` (full) on the same set if the `--quick` numbers raise a flag.

---

## 13. Running preflight without a container

If you cannot (or prefer not to) use a container, see [`preflight-without-container.md`](./preflight-without-container.md) for the step-by-step `runner/primus-cli direct -- preflight ...` walkthrough — Python virtual-environment setup, SLURM invocation patterns, NCCL configuration for Broadcom and Pensando (AINIC) clusters, and many configurable-knob examples.

---

## 14. See also

- [`preflight-without-container.md`](./preflight-without-container.md) — quick-start guide for `primus-cli direct -- preflight` (no container).
- [`node-smoke-test-instruction.md`](./node-smoke-test-instruction.md) — full guide for the per-node smoke test (screen + exclude bad nodes).
- [`runner/primus-cli-direct.sh`](../../runner/primus-cli-direct.sh) — non-container launcher (`primus-cli direct` dispatches here).
- [`primus/tools/preflight/`](../../primus/tools/preflight/) — implementation.
- [`primus/tools/preflight/preflight_args.py`](../../primus/tools/preflight/preflight_args.py) — canonical CLI definition (single source of truth for flags + defaults).
