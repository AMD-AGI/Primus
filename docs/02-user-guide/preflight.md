# Preflight diagnostics

`preflight` is Primus’s cluster diagnostic command. It can produce a **fast environment report** (host, GPU, and network facts) and optionally run **performance tests** (GEMM plus intra- and inter-node communication) to catch misconfiguration or outliers before large distributed training jobs.

`preflight` is implemented by `primus/cli/subcommands/preflight.py`, which in turn delegates to `primus.tools.preflight`.

---

## Overview: What preflight checks

| Category | What are checked |
|----------|----------------|
| **Host** | CPU, memory, PCIe, and related system context |
| **GPU** | ROCm-visible GPU inventory and key attributes |
| **Network** | Network configuration relevant to distributed training |
| **Performance tests** | Heavier GEMM and communication tests (slower than information-only) |

---

## Quick start

### Information only (fast)

```bash
primus-cli direct -- preflight --host --gpu --network
```

### Full preflight (information and performance tests)

```bash
primus-cli direct -- preflight
```

### Performance tests only

Skips the host, GPU, and network information report and runs GEMM + communication tests.

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
| `--perf-test` | Run **only** performance tests (GEMM plus intra- and inter-node communication); skip the information report. |
| `--plot` | Generate plots when used with `--perf-test`. |
| `--dist-timeout-sec` | Timeout in seconds for `torch.distributed` process-group initialization (default: 120). On failure, `preflight` still attempts to write the information report and exits with a non-zero status. |
| `--dump-path` | Output directory for reports (default: `output/preflight`). |
| `--report-file-name` | Base filename for reports (default: `preflight_report`). |
| `--disable-pdf` | Disable PDF generation (PDF is enabled by default when the toolchain allows). |

**Behavior notes**

- With **no** `--host`, `--gpu`, or `--network` flags and **no** `--perf-test`, `preflight` runs in the **full** workflow (information plus performance tests).
- Combine `--host`, `--gpu`, and `--network` to limit the information report to include only those sections.

---

## Usage modes

### Single-node

```bash
primus-cli direct -- preflight --host --gpu --network
```

### Multi-node (Slurm mode)

Info report:

```bash
primus-cli slurm srun -N 4 -- preflight --host --gpu --network
```

Full `preflight`:

```bash
primus-cli slurm srun -N 4 -- preflight
```

Performance tests only:

```bash
primus-cli slurm srun -N 4 -- preflight --perf-test
```

Use the same launcher pattern you rely on for training to ensure that distributed environment variables (`WORLD_SIZE`, `RANK`, `MASTER_ADDR`, etc.) are consistent.

---

## Output files and contents

Default output directory: `output/preflight` (override with `--dump-path`).

| File(s) | Contents |
|---------|----------|
| `<name>.md` / `<name>.pdf` | **Information** report: host, GPU, and network sections when those checks are enabled. |
| `<name>_perf.md` / `<name>_perf.pdf` | **Performance** report: GEMM and communication results from the performance test path. |

The base `<name>` comes from `--report-file-name` (default: `preflight_report`).

---

## Interpreting results

1. **Information report:** Confirm GPU count, model match expectations, and PCIe topology is sensible for your workload. Network sections should reflect the interfaces you intend for distributed training.
2. **Performance report:** Compare GEMM and collective results across nodes. Large outliers on one node often indicate driver, fabric, or process placement issues.
3. **Timeouts:** If `--dist-timeout-sec` is exceeded, inspect firewall rules, interface bindings, `MASTER_ADDR`, and `MASTER_PORT` before scaling up training.

---

## Common issues preflight helps detect

| Symptom | What to verify in reports |
|---------|---------------------------|
| Missing or wrong GPU count | GPU section: ROCm health on the node |
| Wrong network device or address | Network section: NCCL/RCCL environment |
| Slow or asymmetric inter-node comm | Performance report: compare ranks or nodes |
| Hangs at distributed process group initialization | Use `--dist-timeout-sec` to avoid, then check rendezvous and Slurm network setup |

For deeper, single-purpose measurements, see the [Benchmark suite](./benchmarking.md).

---

## Related documentation

- [Benchmark suite](./benchmarking.md)
- [Memory and performance projection](./projection.md)
- [Post-training workflows](./posttraining.md)
- [Installation and setup](../01-getting-started/installation.md)
