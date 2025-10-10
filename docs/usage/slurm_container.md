# Slurm & Container Usage

Primus supports **distributed training on Slurm clusters** and **containerized workflows** (Docker/Podman).
This page provides practical examples and tips.

---

## 🖥️ Slurm Cluster Training

Primus integrates with Slurm to automatically launch jobs inside containers.
By default, the ROCm image `rocm/megatron-lm:v25.8_py310` is used unless you specify another one.

### Interactive Run (`srun`)

```bash
primus-cli slurm srun -N 4 -p AIG_Model --     train pretrain --config ./examples/megatron/configs/llama3_8B-pretrain.yaml
```

- `-N 4`: number of nodes
- `-p AIG_Model`: Slurm partition

### Batch Run (`sbatch`)

```bash
primus-cli slurm sbatch -N 8 -p AIG_Model --     train pretrain --config ./examples/megatron/configs/llama3_8B-pretrain.yaml
```

Logs will be written to the Slurm output file (`slurm-%j.out`).

---

## 🐳 Container Mode (Docker / Podman)

Primus can also be run inside user-managed containers, useful for debugging, custom mounts, or environment variables.

### Run with Mounts

```bash
primus-cli container --mount /mnt/data:/data --     train pretrain --config /data/exp.yaml
```

- `--mount host_dir:container_dir` → bind host directories (datasets, configs, outputs).
- Multiple mounts can be specified by repeating `--mount`.

### Pass Environment Variables

```bash
primus-cli container --env MASTER_ADDR=10.1.1.1 --     train pretrain --config /data/exp.yaml
```

---

## ✅ Best Practices

- Ensure mounted paths exist (`mkdir -p /mnt/data`).
- For multi-node training, set `MASTER_ADDR` consistently across nodes.
- Use `--debug` with `primus-cli` to print expanded commands and environment.

---

## 🔍 Related Docs

- [Quickstart](../quickstart.md)
- [CLI Usage](../cli.md)
- [Benchmark Overview](../benchmark/overview.md)
- [FAQ](../faq.md)
