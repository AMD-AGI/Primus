# CLI Reference

This document describes the unified Primus launcher (`runner/primus-cli`) and how it invokes the Python CLI (`primus/cli/main.py`). For deeper background, see the in-repo guide at `docs/cli/PRIMUS-CLI-GUIDE.md`.

**Related documentation**

| Topic | Location |
| --- | --- |
| Installation and first steps | [Getting started: Quickstart](../01-getting-started/quickstart.md) |
| YAML experiment configuration | [Configuration System](configuration-system.md) |
| Pretraining workflows | [Pretraining Workflows](pretraining.md) |

---

## Command structure

```text
primus-cli [global-options] <mode> [mode-args] -- [command]
```

- **Global options** apply before the mode name and affect config loading and logging for the whole run.
- **Mode** is one of `direct`, `container`, or `slurm`.
- **`--` (required)** separates launcher options from the Primus Python CLI. Everything after the first `--` is passed to `primus/cli/main.py` (or another script if you override it in direct mode).

From the repository root, invoke the launcher as `./runner/primus-cli` (or install/link it as `primus-cli` on your `PATH`).

---

## Global options

These flags are parsed in `runner/primus-cli` before the mode is dispatched to `runner/primus-cli-<mode>.sh`.

| Option | Description |
| --- | --- |
| `--config FILE` | Load a YAML file for launcher defaults (see [Configuration precedence](#configuration-precedence)). |
| `--debug` | Verbose logging; sets `PRIMUS_LOG_LEVEL=DEBUG`. |
| `--dry-run` | Print the command that would run and exit without executing the mode script. |
| `--version` | Print the CLI version and exit. |
| `-h`, `--help` | Show top-level usage and exit. |

Mode-specific help:

```bash
./runner/primus-cli direct --help
./runner/primus-cli container --help
./runner/primus-cli slurm --help
```

Primus Python CLI help (after `--`):

```bash
./runner/primus-cli direct -- --help
./runner/primus-cli direct -- train --help
./runner/primus-cli direct -- benchmark --help
```

---

## Direct mode

Run training, benchmarks, or diagnostics on the current host (or inside an environment you already prepared). GPU-specific tuning is applied via `runner/helpers/envs/<GPU_MODEL>.sh` when present.

### Syntax

```bash
primus-cli direct [options] -- <command>
```

### Options

| Option | Description |
| --- | --- |
| `--config FILE` | Launcher YAML (same resolution as [global `--config`](#configuration-precedence)). |
| `--debug` | Debug logging for the direct launcher. |
| `--dry-run` | Show the resolved launch command without running training. |
| `--single` | Run with `python3` instead of `torchrun` (single process). |
| `--script PATH` | Python entry script (default: `primus/cli/main.py`). |
| `--env KEY=VALUE` | Set an environment variable before launch (repeatable). A path without `=` is treated as an env file (`--env_file`), loaded later in the launch sequence. |
| `--patch script.sh` | Run a shell snippet before the main script (repeatable). |
| `--log_file PATH` | Redirect logs to a file. |
| `--numa` | Force NUMA binding on. |
| `--no-numa` | Force NUMA binding off. |

### Distributed environment variables

For multi-node or multi-process runs, set these (via `export` or `--env`):

| Variable | Role | Typical default |
| --- | --- | --- |
| `NNODES` | Number of nodes | `1` |
| `NODE_RANK` | Rank of this node | `0` |
| `GPUS_PER_NODE` | GPUs per node | `8` (see `runner/.primus.yaml` `direct.gpus_per_node`) |
| `MASTER_ADDR` | Hostname or IP of rank 0 | `localhost` |
| `MASTER_PORT` | TCP port for the process group | `1234` |

---

## Container mode

Run the same Python CLI inside Docker or Podman with ROCm-oriented defaults from `runner/.primus.yaml`.

### Syntax

```bash
primus-cli container [options] -- <command>
```

### Common options

| Option | Description |
| --- | --- |
| `--image NAME` | Image tag (default from config: `rocm/primus:v26.1`). |
| `--volume HOST[:CONTAINER]` | Bind mount (repeatable). |
| `--env KEY=VALUE` | Pass into the **inner** `primus-cli direct` as `--env` (repeatable). |
| `--device PATH` | Extra device nodes (repeatable; defaults include GPU/RDMA devices). |
| `--name`, `--user`, `--network`, `--ipc` | Standard container runtime options. |
| `--clean` | Remove all containers before launch. |
| `--cpus N` | CPU limit. |
| `--memory SIZE` | Memory limit (e.g. `128G`). |
| `--shm-size SIZE` | Shared memory size. |
| `--gpus N` | GPU limit (when using a runtime that supports this flag). |

### Auto-mounted devices

When using `runner/.primus.yaml`, the default container section includes:

- `/dev/kfd` — ROCm kernel fusion driver  
- `/dev/dri` — GPU render nodes  
- `/dev/infiniband` — InfiniBand character devices (when present)

### Environment forwarding

`container.options.env` in `runner/.primus.yaml` lists **names** that are forwarded into the container as inner `--env` arguments when the variable is set in the host environment (for example `MASTER_ADDR`, `HF_TOKEN`, `NCCL_SOCKET_IFNAME`). The container script also auto-forwards host variables whose names start with `PRIMUS_`, `NCCL_`, `RCCL_`, `GLOO_`, `IONIC_`, or `HIPBLASLT_` when not already listed.

---

## Slurm mode

Launch distributed jobs with `srun` or `sbatch`. The Slurm launcher builds `srun`/`sbatch` flags, merges them with `slurm.*` entries from the loaded YAML, then runs `runner/primus-cli-slurm-entry.sh` on allocated nodes.

### Syntax

```text
primus-cli slurm [--config FILE] [--debug] [--dry-run] [srun|sbatch] [SLURM_FLAGS...] -- <entry> [ENTRY_ARGS...] [-- [PRIMUS_ARGS...]]
```

| Part | Meaning |
| --- | --- |
| First `--` | Separates Slurm launcher flags from the **entry** segment (for example `container -- train pretrain ...`). |
| Optional second `--` | If you need to separate arguments intended for an outer entry wrapper from the Primus Python CLI, you can insert a second `--` in the entry segment; the exact split depends on how the entry script parses its arguments. |
| Default launcher | If you omit `srun` and `sbatch`, **`srun` is used** (`LAUNCH_CMD` in `runner/primus-cli-slurm.sh`). |

### Examples

```bash
# Interactive multi-node training (container entry on each node)
./runner/primus-cli slurm srun -N 4 -p gpu -- container -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml

# Batch job
./runner/primus-cli slurm sbatch -N 8 -t 8:00:00 -o train.log -- container -- train pretrain --config exp.yaml
```

On each node, `primus-cli-slurm-entry.sh` sets `NNODES`, `NODE_RANK`, `GPUS_PER_NODE`, `MASTER_ADDR`, and `MASTER_PORT` from Slurm and **invokes `primus-cli-container.sh`** with matching `--env` injections (see `runner/primus-cli-slurm-entry.sh`). Your Slurm command line should still use the `container -- …` entry pattern so arguments line up with that flow.

---

## Python subcommands (after `--`)

These run under `primus/cli/main.py` unless you change `--script` in direct mode.

| Subcommand | Purpose |
| --- | --- |
| `train pretrain --config <yaml>` | Pretraining (Megatron-LM, TorchTitan, MaxText, Megatron Bridge, etc., per experiment YAML). |
| `train posttrain --config <yaml>` | Post-training (SFT / LoRA-style workflows; same top-level flags as pretrain in the parser). |
| `benchmark <suite> [args]` | Performance microbenchmarks (see table below). |
| `preflight [--host] [--gpu] [--network] [--perf-test]` | Cluster and node diagnostics. |
| `projection memory --config <yaml>` | Memory estimation from a merged config. |
| `projection performance --config <yaml>` | Performance projection from a merged config. |

### Benchmark suites

Implemented in `primus/cli/subcommands/benchmark.py`:

| Suite | Notes |
| --- | --- |
| `gemm` | General GEMM microbenchmark. |
| `gemm-dense` | Dense GEMM variant. |
| `gemm-deepseek` | DeepSeek-style dense GEMM. |
| `strided-allgather` | Communication microbenchmark. |
| `rccl` | RCCL collective microbenchmark. |

The same file also registers an `attention` suite for attention microbenchmarks.

---

## Configuration precedence (launcher YAML)

Resolution is implemented in `runner/lib/config.sh` (`resolve_config_file` / `load_config_auto`):

1. **`--config FILE`** on the command line (if given).  
2. **`~/.primus.yaml`** if it exists.  
3. **`runner/.primus.yaml`** (system default).

Within a chosen file, nested keys follow normal YAML structure; Slurm and container scripts merge CLI flags with their sections so that **explicit CLI arguments override file values** where applicable.

**Note:** This precedence applies to the **shell launcher** YAML. Training YAML merge order for experiments is documented in [Configuration System](configuration-system.md).

---

## Common examples

| Goal | Example |
| --- | --- |
| Direct pretrain | `./runner/primus-cli direct -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml` |
| Direct GEMM | `./runner/primus-cli direct -- benchmark gemm --M 4096 --N 4096 --K 4096` |
| Container pretrain | `./runner/primus-cli container --volume /data:/data -- train pretrain --config /data/exp.yaml` |
| Slurm + container | `./runner/primus-cli slurm srun -N 4 -- container -- train pretrain --config exp.yaml` |
| Preflight (fast) | `./runner/primus-cli slurm srun -N 4 -- preflight --host --gpu --network` |
| Export merged train config | `./runner/primus-cli direct -- train pretrain --config exp.yaml --export_config /tmp/merged.yaml` |
| Dry-run Slurm | `./runner/primus-cli --dry-run slurm srun -N 2 -- container -- train pretrain --config exp.yaml` |

---

## Exit codes

From `runner/primus-cli`:

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Library or dependency failure |
| 2 | Invalid arguments or configuration |
| 3 | Runtime execution failure |
