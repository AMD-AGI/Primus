# Preflight

`preflight` is Primus’ cluster diagnostic tool. It can generate a **fast info report** (host/GPU/network) and can also run **performance tests** (GEMM + intra/inter-node comm) to help spot misconfiguration or outliers before large distributed runs.

- **User-facing entry**: `primus-cli … -- preflight [args]`
- **Implementation entrypoint**: `primus/cli/subcommands/preflight.py`

## Quick start

### Info report only (fast)

```bash
primus-cli direct -- preflight --host --gpu --network
```

### Full preflight (info + perf tests)

```bash
primus-cli direct -- preflight
```

### Perf tests only

```bash
primus-cli direct -- preflight --perf-test
```

## Common usage (Slurm)

Info report only (fast):

```bash
primus-cli slurm srun -N 4 -- preflight --host --gpu --network
```

Full preflight (info + perf tests):

```bash
primus-cli slurm srun -N 4 -- preflight
```

Perf tests only:

```bash
primus-cli slurm srun -N 4 -- preflight --perf-test
```

## CLI flags

Selection:
- `--host`: host info (CPU, memory, PCIe)
- `--gpu`: GPU info
- `--network`: network info
- `--perf-test`: run perf tests only (GEMM + comm). This is slower.

Cluster Sphere (built into Primus under `primus/tools/preflight/cluster_sphere/`):
- `--cluster-sphere`: enable **both** Cluster Sphere behaviors below when dependencies are present.
- `--cluster-sphere-env`: add **RDMA env recommendations** (NCCL/GLOO/rocSHMEM export hints from local verbs devices) to the **info** report (`preflight_report*.md`).
- `--cluster-sphere-rdma-bw`: append **Verbs `ib_write_bw`** results to the **perf** report (`*_perf.md`). Intended for **`WORLD_SIZE=2`** only (two processes, typically one per node). Requires [linux-rdma/perftest](https://github.com/linux-rdma/perftest) (`ib_write_bw` on `PATH`). Optional: set `PRIMUS_IB_WRITE_BW_PORT` (default `2000`).

Optional override: set **`PRIMUS_CLUSTER_SPHERE_ROOT`** to a directory only if you need to point at a custom/fork copy of the integration (defaults to the in-tree package path).

Preflight’s existing perf tests measure **PyTorch / NCCL-style** GPU communication and use sysfs link rate as a roofline; they are **not** a substitute for raw Verbs bandwidth. Cluster Sphere **`ib_write_bw`** validates the **host RDMA path** separately.

Reporting:
- `--dump-path`: output directory (default: `output/preflight`)
- `--report-file-name`: base report name (default: `preflight_report`)
- `--disable-pdf`: disable PDF generation

Perf-test extras:
- `--plot`: generate plots (only used with `--perf-test`)

Backward compatibility:
- `--check-host/--check-gpu/--check-network` are supported as aliases for `--host/--gpu/--network`.

## Outputs

By default, outputs are written under `output/preflight`.

Typical report files:
- `preflight_report.md` / `preflight_report.pdf`: **info report** (host/GPU/network, plus Cluster Sphere env section when enabled)
- `preflight_report_perf.md` / `preflight_report_perf.pdf`: **perf report** (GEMM + comm tests, plus optional `ib_write_bw` section)

## Notes

- For multi-node runs, use `primus-cli slurm …` (or your preferred launcher) so distributed environment variables are set correctly.
- If you only want a quick environment snapshot, prefer `--host --gpu --network`.
- `--perf-test` skips the info report; use `--cluster-sphere-env` without `--perf-test` if you only need RDMA export hints.
