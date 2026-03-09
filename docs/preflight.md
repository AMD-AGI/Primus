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
- `preflight_report.md` / `preflight_report.pdf`: **info report** (host/GPU/network)
- `preflight_report_perf.md` / `preflight_report_perf.pdf`: **perf report** (GEMM + comm tests)

## Notes

- For multi-node runs, use `primus-cli slurm …` (or your preferred launcher) so distributed environment variables are set correctly.
- If you only want a quick environment snapshot, prefer `--host --gpu --network`.
