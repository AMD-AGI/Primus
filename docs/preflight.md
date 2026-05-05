# Preflight

`preflight` is Primus' cluster diagnostic tool. It produces:

- A **fast info report** (host / GPU / network configuration), and
- A configurable suite of **performance tests** (GEMM TFLOPS, intra-node and inter-node communication bandwidth, P2P, ring P2P).

Use it to spot misconfiguration, hardware degradation, or perf outliers **before** committing a large distributed training run to a global rendezvous.

- **User-facing entry**: `primus-cli ... -- preflight [args]`
- **Wrapper script (no container)**: [`runner/run_preflight_direct.sh`](../runner/run_preflight_direct.sh) — see [`preflight-direct.md`](./preflight-direct.md).
- **Implementation entrypoint**: `primus/cli/subcommands/preflight.py` → `primus/tools/preflight/preflight_perf_test.py`.

> Looking for a faster, distributed-rendezvous-free per-node screen? See [`node-smoke.md`](./node-smoke.md) (and the [quick-start guide](./node-smoke-test-instruction.md)). The recommended workflow is **smoke first, preflight second** — see [§9 Comparison with node-smoke](#9-comparison-with-node-smoke).

---

## 1. Two run modes (and how preflight picks one)

Preflight has two report types, controlled by a single precedence rule:

| Mode | Triggered by | What it does |
|---|---|---|
| **Info-only** | `--host`, `--gpu`, `--network` (in any combination) | Lightweight host / GPU / network introspection. **No `torch.distributed` rendezvous.** Cannot hang on network misconfig. |
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

### Info report only (fast, no rendezvous)

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

Without a container, see [`preflight-direct.md`](./preflight-direct.md) for the equivalent `runner/run_preflight_direct.sh` invocations.

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
| `inter-p2p` | Inter-node 2-rank P2P send/recv. Requires `--inter-group-sizes` to actually contain pair-able sizes. |
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
| `--inter-group-sizes CSV` | `2,4,all` | `all` means the full N-node group. Other values are subgroup sizes. |

```bash
# All-GPU intra + full N-node inter only
primus-cli direct -- preflight \
    --tests intra-allreduce,inter-allreduce \
    --intra-group-sizes 8 \
    --inter-group-sizes all
```

Validation is gated by which tests are actually selected. For example, `--tests gemm --intra-group-sizes 3` does **not** abort on a host with `LOCAL_WORLD_SIZE=8`; the intra-group constraint is only checked when an intra test is enabled.

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

Delay (seconds) after destroying NCCL/RCCL process groups before creating new ones. Prevents `Address already in use` from socket port-reuse races as preflight tears down and recreates communicators between phases.

- Default `2.0` is sufficient up to ~64 nodes.
- Increase (e.g. `5.0`) at very large scale (128+ nodes / 1000+ ranks).
- Set to `0` to disable the sleep (barrier only).

```bash
primus-cli slurm srun -N 128 -- preflight --quick --comm-cleanup-delay-sec 5
```

### 6.2 `--dist-timeout-sec INT` (default `120`)

Timeout (seconds) for `torch.distributed.init_process_group`. If init does not complete within this many seconds, preflight writes the info report (when applicable) plus a `Distributed Init` failure section to the markdown report, prints a clear error, and exits `2` — instead of hanging indefinitely.

```bash
# Fail fast if rendezvous does not work
primus-cli direct -- preflight --perf-test --dist-timeout-sec 30
```

---

## 7. Reporting

| Flag | Default | Effect |
|---|---|---|
| `--dump-path DIR` | `output/preflight` | Output directory for reports + plots. |
| `--report-file-name NAME` | `preflight_report` | Base name for report files. The wrapper `run_preflight_direct.sh` auto-generates a unique `preflight-${NNODES}N-YYYYMMDD-HHMMSS` when this is omitted. |
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

## 8. Backward-compat aliases

| Flag | Equivalent | Notes |
|---|---|---|
| `--check-host`, `--check-gpu`, `--check-network` | `--host`, `--gpu`, `--network` | Same behavior. Keep working for older scripts. |
| `--no-split-nodes-subgroup` | `--inter-group-sizes all` **and** drops `inter-p2p` | Pre-`--tests`/`--inter-group-sizes` alias. Use the new flags in new scripts. |

---

## 9. Comparison with node-smoke

| Aspect | `node-smoke` | `preflight` |
|---|---|---|
| Rendezvous | None — every node independent | Global `torch.distributed` |
| Wall clock | ~30–60 s for 6 nodes (Tier 1+2) | Minutes; scales with N for inter-node tests |
| Granularity | Per-node PASS/FAIL | Per-rank measurements (no auto-fail by default) |
| Inter-node bandwidth matrix | Not tested (intentionally) | Yes (allreduce/alltoall/p2p/ring-p2p) |
| Drift detection | Yes (versions, NIC firmware, port count) | No |
| Host limits / RDMA roll-call | Yes (hard fail) | Reported via `collect_*_info` only |
| Output format | Per-node JSON + cluster md + SLURM-ready txt | Markdown + PDF |

**Recommended workflow**: run `node-smoke` first to exclude broken nodes, then run `preflight` on the surviving set to get cross-node bandwidth measurements. See [`node-smoke-test-instruction.md`](./node-smoke-test-instruction.md) §3 ("Quick start") for the integration commands.

---

## 10. Validation & error handling

Preflight resolves the perf config **before** any distributed rendezvous. This means typos and bad sizes/group-sizes fail in seconds, not after a 120s NCCL init:

```text
[Primus:Preflight] ERROR: invalid perf config: --tests: unknown token 'gem'.
[Primus:Preflight] ERROR: invalid perf config:
    --intra-group-sizes: [3] do not divide LOCAL_WORLD_SIZE=8
[Primus:Preflight] ERROR: invalid perf config: --comm-sizes-mb: '0' must be positive
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

## 11. Operational tips

- **For multi-node runs, always use `primus-cli slurm`** (or `runner/run_preflight_direct.sh`) so distributed environment variables are set correctly.
- **Insufficient CPU cores cause >30x perf slowdowns** — pass `srun -c <cores-per-node>` so RCCL's network proxy threads have CPU to spawn on. Verify with `srun -N 1 --gpus-per-node=8 bash -c 'nproc'`.
- **For a quick environment snapshot**, prefer `--host --gpu --network` (no rendezvous, finishes in seconds even on broken networks).
- **Between each communication test phase**, preflight performs a global barrier + sleep to prevent `Address already in use`. At very large scale, increase `--comm-cleanup-delay-sec`.
- **For pre-launch screening of a large cluster**, the recommended sequence is:
  1. `node-smoke` to prune broken nodes (`failing_nodes.txt`).
  2. `preflight --quick` on the surviving nodes for the perf sanity numbers.
  3. `preflight` (full) on the same set if the `--quick` numbers raise a flag.

---

## 12. Running preflight without a container

If you cannot (or prefer not to) use a container, see [`preflight-direct.md`](./preflight-direct.md) for the step-by-step `runner/run_preflight_direct.sh` walkthrough — Python virtual-environment setup, SLURM invocation patterns, NCCL configuration for Broadcom and Pensando (AINIC) clusters, and many configurable-knob examples.

---

## 13. See also

- [`preflight-direct.md`](./preflight-direct.md) — quick-start guide for `runner/run_preflight_direct.sh` (no container).
- [`node-smoke.md`](./node-smoke.md) — full reference for the per-node smoke test.
- [`node-smoke-test-instruction.md`](./node-smoke-test-instruction.md) — short quick-start for the smoke test.
- [`runner/run_preflight_direct.sh`](../runner/run_preflight_direct.sh) — non-container wrapper.
- [`primus/tools/preflight/`](../primus/tools/preflight/) — implementation.
- [`primus/tools/preflight/preflight_args.py`](../primus/tools/preflight/preflight_args.py) — canonical CLI definition (single source of truth for flags + defaults).
