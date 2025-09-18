# CLI Usage

Primus provides a unified command-line interface (CLI) for launching training, benchmarking, and system validation tasks.

---

## 🔧 General Syntax

```bash
primus-cli <mode> [mode-args] -- <command> [command-args]
```

- `<mode>`: Launch context — one of:
  - `direct`: run directly on host
  - `slurm`: run using Slurm (srun/sbatch)
  - `container`: run inside Docker/Podman
- `<command>`: Operation to run — `train`, `benchmark`, `preflight`, etc.

---

## 📚 Available Subcommands

| Command      | Description                                 |
|--------------|---------------------------------------------|
| `train`      | Launch LLM training with Megatron/Titan     |
| `benchmark`  | Run GEMM / RCCL / attention perf tests      |
| `preflight`  | Run environment & cluster sanity check      |

---

## 🚀 Examples

### Direct mode

```bash
primus-cli direct -- train --config ./exp.yaml
```

### Slurm mode (interactive)

```bash
primus-cli slurm srun -N 4 -p AIG_Model -- train --config ./exp.yaml
```

### Slurm mode (batch)

```bash
primus-cli slurm sbatch -N 8 -p AIG_Model -- train --config ./exp.yaml
```

### Container mode

```bash
primus-cli container --mount /mnt/data:/data -- train --config /data/exp.yaml
```

---

## 🔍 Debugging and Logs

- Use `--debug` to enable verbose output
- Logs will be printed with rank, timestamp, and tag
- Environment variables can be passed with `--env` (e.g. `--env MASTER_PORT=12345`)

---

## 🛠️ Helpful Flags

| Flag           | Description                          |
|----------------|--------------------------------------|
| `--config`     | Path to experiment YAML/TOML config  |
| `--mount`      | Bind mount volumes in container mode |
| `--env`        | Set environment variables            |
| `--help`       | Print CLI help message               |

---

_Last updated: 2025-09-17_
