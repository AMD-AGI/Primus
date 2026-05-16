# Preflight

`preflight` is Primus' cluster diagnostic tool. It produces:

- A **fast info report** (host / GPU / network configuration), and
- A configurable suite of **performance tests** (GEMM TFLOPS, intra-node and inter-node communication bandwidth, P2P, ring P2P).

Use it to spot misconfiguration, hardware degradation, or perf outliers **before** committing a large distributed training run to a global rendezvous.

- **User-facing entry**: `primus-cli ... -- preflight [args]`
- **Wrapper script (no container)**: [`runner/run_preflight_direct.sh`](../runner/run_preflight_direct.sh) — see [`preflight-direct.md`](./preflight-direct.md).
- **Implementation entrypoint**: `primus/cli/subcommands/preflight.py` → `primus/tools/preflight/preflight_perf_test.py`.

> Looking for a faster, distributed-rendezvous-free per-node screen? See [`node-smoke.md`](./node-smoke.md) (and the [quick-start guide](./node-smoke-test-instruction.md)). The recommended workflow is **smoke first, preflight second** — see [§10 Comparison with node-smoke](#10-comparison-with-node-smoke).

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
| `--inter-group-sizes CSV` | `2,4,all` | `all` means the full N-node group. Other values are subgroup sizes. **For `inter-alltoall` only**, every requested per-group node count is internally clamped to **16** before deduping (see "Inter-node alltoall is capped at 16 nodes" below). The other inter-node tests (`inter-allreduce`, `inter-p2p`, `inter-ring-p2p`) use the requested sizes unchanged. |

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

Why a hard cap and why 16:

- **Real-world MoE training rarely dispatches across more than ~8 nodes.** DeepSeek-V3's largest published configuration uses `EP=64` over 8 nodes with per-token dispatch capped at 4 nodes; NVIDIA Megatron-Core's published EP recipes follow the same shape. 16 covers every published configuration with comfortable headroom while staying well clear of the per-node ephemeral-port pressure described in §7.
- **The cap eliminates the dominant source of `Address already in use` at large scale.** During each `ncclCommInit`, an inter-node alltoall sub-group of `K` nodes opens a near-full mesh of IB OOB sockets per local rank — empirically, peak simultaneous ESTAB sockets per node grow ~linearly with `K` at **~145 sockets per added node** (linear fit `peak_ESTAB ≈ 145·K + 683` over measurements at 24/32/48/56 N; see §7.1.2). Once peak ESTAB approaches the size of the kernel's ephemeral-port pool (default `28 232` ports), the next outgoing `bind()` walks the entire range without finding an allocatable port and fails. Capping the sub-group at 16 holds peak ESTAB at **~3.7 k** — measured directly on a 56 N cluster with the cap active — comfortably under any sensible pool, so the failure cannot occur regardless of cluster size or OS tuning.
- **Other inter-node tests are unaffected.** `inter-allreduce` (ring/tree, ~`log K` peers per rank) and `inter-ring-p2p` (ring, 2 peers per rank) and `inter-p2p` (pairwise) all open far fewer simultaneous OOB sockets than alltoall and continue to honor `--inter-group-sizes` exactly as written. As a concrete reference point: at 56 nodes, a default-configured `inter-allreduce` peaks at ~1.8 k ESTAB; an `inter-alltoall` over the same 56 nodes peaks at ~8.8 k.
- **Intentionally not configurable.** This is a known-safe ceiling for the comm shapes preflight is supposed to characterize, not a tuning knob; raising it would re-introduce the very failure mode preflight is meant to *detect* in the cluster, not *cause*.

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

Delay (seconds) inserted between destroying NCCL/RCCL process groups and creating new ones. Provides cross-rank synchronization across the destroy → setup transition (so a rank doesn't try to connect to a peer whose listener hasn't finished closing) and gives the kernel a moment to unlink closed-socket bookkeeping. See §7.1 for why this knob is *not* primarily defending against `TIME_WAIT` pressure (which doesn't apply to NCCL's connection pattern) — the actual `Address already in use` defense is the §5.2 inter-alltoall cap.

- Default `2.0` is essentially free and worth keeping as cheap insurance at every cluster size.
- Set to `0` to disable the sleep entirely (barrier only).
- Bump to e.g. `5` only on very flaky networks or pathological kernel scheduling; widening `ip_local_port_range` (§7.2) is a more direct fix when the binding constraint is genuinely hit.

```bash
# Small/medium clusters: defaults are fine. Override only if you see
# port-reuse races on very flaky networks or unusual kernel TIME_WAIT
# settings.
primus-cli slurm srun -N 8 -- preflight --quick --comm-cleanup-delay-sec 5
```

See §7 ("Running on very large clusters") for the rationale behind why the default works at scale and for the OS-level best-practices that apply to any RDMA workload.

### 6.2 `--dist-timeout-sec INT` (default `120`)

Timeout (seconds) for `torch.distributed.init_process_group`. If init does not complete within this many seconds, preflight writes the info report (when applicable) plus a `Distributed Init` failure section to the markdown report, prints a clear error, and exits `2` — instead of hanging indefinitely.

> Note: §6 used to also document `--comm-cleanup-large-threshold-nodes`, which forced a 60 s drain whenever a destroyed subgroup met or exceeded a size threshold (default 64 nodes). That flag was removed because the underlying failure mode it tried to paper over — peak simultaneous ESTAB exhausting the per-node ephemeral-port pool during a large inter-node alltoall `ncclCommInit` — is now prevented at the source by the §5.2 alltoall cap. The `--comm-cleanup-delay-sec` knob remains, but its primary role is now cross-rank synchronization across the destroy → setup transition rather than draining `TIME_WAIT`; the default 2 s is essentially free and worth keeping as cheap insurance.

```bash
# Fail fast if rendezvous does not work
primus-cli direct -- preflight --perf-test --dist-timeout-sec 30
```

---

## 7. Running on very large clusters (≥ 64 nodes)

Beyond ~64 nodes, two practical concerns dominate that smaller runs never see. Read this section once if you operate clusters in this range; it explains the failure mode, the OS knobs that fix it at the system level, and the recommended preflight invocation patterns.

### 7.1 Why "Address already in use" used to surface at scale

The failure is a **per-node ephemeral port exhaustion** during a single inter-node alltoall `ncclCommInit`, not a chronic accumulation of `TIME_WAIT` sockets across phases. Understanding this distinction is what motivates both the §5.2 cap and the §7.2 OS tuning recommendations.

#### 7.1.1 What actually consumes the per-node ephemeral pool

Linux's outgoing-connection allocator (`__inet_hash_connect()`) does **not** reject ports just because some other socket is in any state on them. It rejects a port only when the new connection's full 4-tuple `(saddr, sport, daddr, dport)` collides with an existing socket's 4-tuple:

- **Live ESTAB sockets** sit on a specific 4-tuple and prevent the kernel from reusing that exact 4-tuple. Each new outgoing connection that lands on a port already holding an ESTAB socket has to walk to the next port. As ESTAB density approaches one-socket-per-port across the entire ephemeral range, the walk takes longer and longer until eventually no port is allocatable — that's the EADDRINUSE.
- **TIME_WAIT sockets** also live on specific 4-tuples but are governed by `tcp_tw_reuse`. Critically, they only block a new connection when the new connection's *desired* 4-tuple collides with the historical one — i.e. when the new connection is to the *same* `(daddr, dport)` from the *same* `(saddr, sport)`.

For NCCL inter-node OOB traffic, the second case essentially never happens: each `ncclCommInit` connects to **fresh peer OOB listening ports** (the peer chooses an ephemeral listener per setup), so successive comm setups always have different `dport`. TIME_WAIT entries from a previous destroy sit on `(local, P_old, peer, dport_OLD)`; the next setup wants `(local, ?, peer, dport_NEW)`. Even when the new connection happens to land on `sport == P_old`, the dports differ → no 4-tuple collision → the TIME_WAIT entry is invisible to the allocator regardless of `tcp_tw_reuse`.

The practical consequence: under NCCL's connection pattern, **the binding constraint reduces to peak simultaneous ESTAB ≤ size of the ephemeral pool**.

#### 7.1.2 Why inter-node alltoall is the test that exhausts it

Per-node peak ESTAB scales very differently across the inter-node test families:

| Test (56 N, default knobs) | Peer topology per rank | Peak ESTAB per node |
|---|---|---|
| `inter-allreduce` | ring / tree, ~`log K` peers | ~1.8 k |
| `inter-alltoall` | full mesh, `K-1` peers | ~8.8 k |

The empirical scaling of `inter-alltoall` peak ESTAB in the default perf-test sweep. Four measurements at 24/32/48/56 N fit a near-perfect line:

```
peak_ESTAB(per node) ≈ 145.09 · K + 683
```

| K (nodes) | Measured peak ESTAB | Fit (145·K + 683) | Source |
|---|---|---|---|
| 16 (capped run) | **3 687** | 3 003 | direct measurement — fit slightly under-predicts at the low-N extrapolation |
| 24 | 4 165 | 4 165 | calibration point |
| 32 | 5 326 | 5 326 | calibration point |
| 48 | 7 647 | 7 647 | calibration point |
| 56 | 8 808 | 8 808 | calibration point |
| 128 (uncapped, extrapolated) | — | **~19 240** | linear extrapolation |

Two practical reads from this:

- **At 56 N the workload already sits at ~31 % of the default 28 232-port pool**, with ~19 k ports of headroom. That's why every default-pool 56 N run in our experiment succeeded.
- **An uncapped 128-N inter-alltoall would peak around ~19 k ESTAB** — still inside the default pool but with ~9 k ports of headroom, and *over the cliff* on any cluster that has narrowed `ip_local_port_range`, that runs additional outgoing TCP work concurrently, or that has the source-port allocator's random-walk hit a bad starting offset.

#### 7.1.3 Empirical confirmation: the binding constraint really is `peak_ESTAB ≤ pool_size`

Holding the workload constant (56 N, only inter-alltoall) and varying *only* the per-node ephemeral pool size:

| Pool | Peak ESTAB | Headroom | Result |
|---|---|---|---|
| 28 231 (default) | 8 805 | +19 426 | OK |
| 9 000 | 8 806 | +194 | OK (barely) |
| 8 000 | (~8 800 expected) | −800 | **FAIL** |

The transition is sharp and right at the predicted boundary. Note that peak `TIME_WAIT` count in the same runs ranged from ~16 k to ~24 k — **far above** the pool size in the 9 k and 8 k rows. If TIME_WAIT count drove the failure, both narrow-pool rows should fail. They don't; only the row where `peak_ESTAB > pool_size` does.

The mechanism is further corroborated by a direct cap experiment on the same 56 N cluster, default pool: re-running the inter-node alltoall test with the per-group node count capped at 16 (matching the §5.2 cap) yields peak ESTAB **3 687** — well under both the default 28 k pool and the 9 k / 8 k stressed pools — and the run completes cleanly. The cap acts directly on the binding constraint by holding peak ESTAB low enough that the pool cannot be exhausted, regardless of cluster size.

#### 7.1.4 The cap closes the failure mode at the source

The §5.2 inter-node alltoall cap (16 nodes max) holds peak ESTAB at **~3.7 k** regardless of cluster size — measured directly in §7.1.3, comfortably under the default 28 k pool (~13 % utilization) and still safe at half-default pool widths. The OS-level tunings in §7.2 remain useful general hygiene for any RDMA / multi-NIC workload, but a default preflight invocation no longer needs them to avoid `Address already in use`.

This is **not** a real training failure mode in any case — production training jobs create their TP / DP / PP / EP communicators *once* at startup and reuse them, and real-world MoE training rarely dispatches across more than ~8 nodes. The preflight tool is the only thing that builds many large communicators in a short window, which is why the issue was preflight-specific to begin with.

### 7.2 OS-level tuning (best-practice for any large-cluster node)

Given §7.1's mechanism (binding constraint = peak ESTAB ≤ pool size), the OS knobs split cleanly into "directly relevant" and "general hygiene":

```bash
# 1) DIRECTLY RELEVANT: widen the per-node ephemeral pool.
#    The default range is 32768-60999 (~28 k ports). Widening it
#    to 1024-65535 (~64 k ports) more than doubles the headroom
#    for peak simultaneous ESTAB — and that is the only thing that
#    can produce EADDRINUSE under the NCCL connection pattern.
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"

# 2) GENERAL HYGIENE: allow TIME_WAIT reuse for outgoing connections.
#    For the NCCL inter-node OOB pattern this is largely a no-op
#    (each ncclCommInit picks fresh peer destination ports, so
#    historical TIME_WAIT 4-tuples never collide with what the
#    next setup wants). It is still recommended for any host that
#    runs additional outgoing TCP workloads where the SAME
#    (daddr, dport) is hit repeatedly from the same source IP --
#    the textbook scenario the kernel doc is written around.
sudo sysctl -w net.ipv4.tcp_tw_reuse=1

# Persist across reboots:
cat <<'EOF' | sudo tee /etc/sysctl.d/99-large-cluster.conf
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_tw_reuse = 1
EOF
sudo sysctl --system
```

If you only have time to set one of these, pick `ip_local_port_range`. With the §5.2 inter-alltoall cap holding peak ESTAB at ~3.7 k even at 1024 N, neither knob is *required* for preflight — but the wider port range is the right insurance for any host that might also run other outgoing TCP traffic concurrently.

Note on `tcp_tw_reuse=2`: as of Linux 4.12, value `2` enables TIME_WAIT reuse **only for loopback** (`127.0.0.0/8`, `::1`). For inter-node IB OOB connections this is equivalent to `tcp_tw_reuse=0`. The fact that several of our 56 N runs succeeded under `tcp_tw_reuse=2` with `peak_WAIT > 16 k` is direct evidence that `TIME_WAIT` *count* doesn't gate inter-node bind() — only `peak_ESTAB > pool_size` does (see §7.1.3).

### 7.3 In-tool defenses

Preflight has two complementary defenses:

1. **The §5.2 inter-node alltoall cap (16 nodes max).** *This is the actual fix.* Regardless of cluster size or `--inter-group-sizes`, the alltoall test never builds a sub-group large enough to push peak ESTAB anywhere near the per-node ephemeral-port pool. This eliminates the historical EADDRINUSE failure mode by construction (see §7.1).
2. **The §6.1 per-phase cleanup delay (`--comm-cleanup-delay-sec`, default `2.0`).** A global barrier + sleep inserted after every comm destroy. Its primary job is **cross-rank synchronization** across the destroy → setup transition (so a rank doesn't try to connect to a peer whose listener hasn't finished closing) and giving the kernel a moment to unlink closed-socket bookkeeping. It is *not* protecting against `TIME_WAIT` 4-tuple collisions — those don't occur in NCCL's connection pattern (see §7.1.1). The default 2 s is essentially free and worth keeping as cheap insurance.

Together, a default preflight invocation is safe at every cluster size we test up to 1024 nodes. The only situation where you would consider raising `--comm-cleanup-delay-sec` is on a network with unusually narrow ephemeral-port ranges or pathological kernel scheduling — and even there, widening `ip_local_port_range` (§7.2) is the cleaner fix because it directly addresses the binding constraint.

### 7.4 Recommended invocation patterns at very large scale

For clusters at or beyond ~128 nodes, the most reliable and most informative way to use preflight is still to **split the run into one test family per invocation** rather than one big run — not for `Address already in use` reasons (the §7.3 defenses handle that), but because it keeps wall-clock per invocation small and makes it trivial to identify which specific comm shape is degraded if a metric looks off.

```bash
# 1) GPU + intra-node fabric first (cheap, no inter-node OOB churn).
primus-cli slurm srun -N 128 -- preflight \
    --tests gemm,intra-allreduce,intra-alltoall

# 2) Inter-node DP-style collectives, all-nodes group only.
primus-cli slurm srun -N 128 -- preflight \
    --tests inter-allreduce \
    --inter-group-sizes all
# Note: --inter-group-sizes all is honored here for inter-allreduce.
# For inter-alltoall it would be clamped to 16 (see §5.2).
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

Each invocation:

- Tears down its own `WORLD` at exit, so the next invocation starts with a fresh per-node port pool.
- Touches only one `--tests` value, so you get a per-test wall clock and can re-run a single phase if its numbers look off without paying for the others.
- Carries its own report under `--report-file-name` (or the wrapper-generated default), which makes archiving and comparison across runs straightforward.

### 7.5 Decision flow

| Cluster size | Recommended approach |
|---|---|
| ≤ 32 nodes | Single command, default knobs. Nothing special. |
| 33-127 nodes | Single command, default knobs. |
| ≥ 128 nodes | Single command works with default knobs (the §5.2 alltoall cap holds peak ESTAB at ~3.7 k, ≈ 7.6× under the default 28 k ephemeral pool). Splitting per `--tests` token as in §7.4 is recommended for diagnostic clarity rather than for safety. Widening `ip_local_port_range` (§7.2) is best-practice for any RDMA workload but no longer required for preflight specifically. |
| ≥ 256 nodes | Always split as in §7.4 — keeps every invocation snappy and makes regressions much easier to localize. |

---

## 8. Reporting

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

**Recommended workflow**: run `node-smoke` first to exclude broken nodes, then run `preflight` on the surviving set to get cross-node bandwidth measurements. See [`node-smoke-test-instruction.md`](./node-smoke-test-instruction.md) §3 ("Quick start") for the integration commands.

---

## 11. Validation & error handling

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

## 12. Operational tips

- **For multi-node runs, always use `primus-cli slurm`** (or `runner/run_preflight_direct.sh`) so distributed environment variables are set correctly.
- **Insufficient CPU cores cause >30x perf slowdowns** — pass `srun -c <cores-per-node>` so RCCL's network proxy threads have CPU to spawn on. Verify with `srun -N 1 --gpus-per-node=8 bash -c 'nproc'`.
- **For a quick environment snapshot**, prefer `--host --gpu --network` (no rendezvous, finishes in seconds even on broken networks).
- **Between each communication test phase**, preflight performs a global barrier + `--comm-cleanup-delay-sec` sleep (default 2 s) for cross-rank sync across the destroy → setup transition. The default works at every cluster size we test up to 1024 nodes because the inter-node alltoall sub-group is internally capped at 16 (see §5.2), which holds peak simultaneous ESTAB sockets per node well under the kernel's ephemeral-port pool — the only constraint that actually produces `Address already in use` under NCCL's connection pattern (see §7.1). Widening `ip_local_port_range` (§7.2) is best-practice for any RDMA workload.
- **For pre-launch screening of a large cluster**, the recommended sequence is:
  1. `node-smoke` to prune broken nodes (`failing_nodes.txt`).
  2. `preflight --quick` on the surviving nodes for the perf sanity numbers.
  3. `preflight` (full) on the same set if the `--quick` numbers raise a flag.

---

## 13. Running preflight without a container

If you cannot (or prefer not to) use a container, see [`preflight-direct.md`](./preflight-direct.md) for the step-by-step `runner/run_preflight_direct.sh` walkthrough — Python virtual-environment setup, SLURM invocation patterns, NCCL configuration for Broadcom and Pensando (AINIC) clusters, and many configurable-knob examples.

---

## 14. See also

- [`preflight-direct.md`](./preflight-direct.md) — quick-start guide for `runner/run_preflight_direct.sh` (no container).
- [`node-smoke.md`](./node-smoke.md) — full reference for the per-node smoke test.
- [`node-smoke-test-instruction.md`](./node-smoke-test-instruction.md) — short quick-start for the smoke test.
- [`runner/run_preflight_direct.sh`](../runner/run_preflight_direct.sh) — non-container wrapper.
- [`primus/tools/preflight/`](../primus/tools/preflight/) — implementation.
- [`primus/tools/preflight/preflight_args.py`](../primus/tools/preflight/preflight_args.py) — canonical CLI definition (single source of truth for flags + defaults).
