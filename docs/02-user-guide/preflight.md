# Preflight Diagnostics

`preflight` is Primus’s cluster diagnostic command. It can produce a **fast environment report** (host, GPU, and network facts) and optionally run **performance tests** (GEMM plus intra- and inter-node communication) to catch misconfiguration or outliers before large distributed training jobs.

**Implementation:** `primus/cli/subcommands/preflight.py` (delegates to `primus.tools.preflight`).

Related: [Benchmark suite](./benchmarking.md) (focused microbenchmarks), [Memory and performance projection](./projection.md), [Installation](../01-getting-started/installation.md).

---

## Overview: what preflight checks

| Category | What you get |
|----------|----------------|
| **Host** | CPU, memory, PCIe, and related system context |
| **GPU** | ROCm-visible GPU inventory and key attributes |
| **Network** | Network configuration relevant to distributed training |
| **Performance tests** | Heavier GEMM and communication tests (slower than info-only) |

Running `preflight` with **no selection flags** runs the full workflow (informational report plus performance tests). Narrow flags restrict work to the pieces you need.

---

## Quick start

### Information only (fast)

```bash
primus-cli direct -- preflight --host --gpu --network
```

### Full preflight (info and performance tests)

```bash
primus-cli direct -- preflight
```

### Performance tests only

Skips the host/GPU/network info report and runs GEMM + communication tests.

```bash
primus-cli direct -- preflight --perf-test
```

---

## CLI flags reference

| Flag | Purpose |
|------|---------|
| `--host` | Include host information (CPU, memory, PCIe). Alias: `--check-host`. |
| `--gpu` | Include GPU information. Alias: `--check-gpu`. |
| `--network` | Include network information. Alias: `--check-network`. |
| `--perf-test` | Run **only** performance tests (GEMM and intra/inter-node communication); skips the info report. |
| `--plot` | Generate plots when used with `--perf-test`. |
| `--dist-timeout-sec` | Timeout in seconds for `torch.distributed` process-group init (default: 120). On failure, preflight still attempts to write the info report and exits non-zero. |
| `--dump-path` | Output directory for reports (default: `output/preflight`). |
| `--report-file-name` | Base filename for reports (default: `preflight_report`). |
| `--disable-pdf` | Disable PDF generation (PDF is enabled by default when the toolchain allows). |

**Behavior notes**

- With **no** `--host`, `--gpu`, or `--network` flags and **no** `--perf-test`, preflight runs the **full** default (info plus perf tests).
- Combine `--host`, `--gpu`, and `--network` to limit the informational report to those sections.

---

## Usage modes

### Single-node

```bash
primus-cli direct -- preflight --host --gpu --network
```

### Multi-node (Slurm example)

Info report:

```bash
primus-cli slurm srun -N 4 -- preflight --host --gpu --network
```

Full preflight:

```bash
primus-cli slurm srun -N 4 -- preflight
```

Performance tests only:

```bash
primus-cli slurm srun -N 4 -- preflight --perf-test
```

Use the same launcher pattern you rely on for training so distributed environment variables (`WORLD_SIZE`, `RANK`, `MASTER_ADDR`, etc.) are consistent.

---

## Output files and contents

Default output directory: `output/preflight` (override with `--dump-path`).

| File(s) | Contents |
|---------|----------|
| `<name>.md` / `<name>.pdf` | **Informational** report: host, GPU, and network sections when those checks are enabled. |
| `<name>_perf.md` / `<name>_perf.pdf` | **Performance** report: GEMM and communication results from the perf test path. |

The base `<name>` comes from `--report-file-name` (default: `preflight_report`).

---

## Interpreting results

1. **Info report:** Confirm GPU count, model match expectations, and PCIe topology is sensible for your workload. Network sections should reflect the interfaces you intend for distributed training.
2. **Perf report:** Compare GEMM and collective results across nodes. Large outliers on one node often indicate driver, fabric, or process placement issues.
3. **Timeouts:** If `--dist-timeout-sec` is exceeded, inspect firewall rules, interface bindings, and `MASTER_ADDR` / `MASTER_PORT` before scaling up training.

---

## Common issues preflight helps detect

| Symptom | What to verify in reports |
|---------|---------------------------|
| Missing or wrong GPU count | GPU section; ROCm health on the node |
| Wrong network device or address | Network section; NCCL/RCCL environment |
| Slow or asymmetric inter-node comm | Perf report; compare ranks or nodes |
| Hangs at distributed init | Use `--dist-timeout-sec`; check rendezvous and Slurm network setup |

For deeper, single-purpose measurements, see the [Benchmark suite](./benchmarking.md).

---

## See also

- [Benchmark suite](./benchmarking.md)
- [Memory and performance projection](./projection.md)
- [Post-training workflows](./posttraining.md)
- [Installation](../01-getting-started/installation.md)
