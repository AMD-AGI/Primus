# Slurm & Container Usage

Primus supports running distributed training jobs via Slurm as well as containerized workflows using Docker or Podman.

---

## 🖥️ Slurm Cluster Launch

### Interactive Mode (srun)

Use `srun` for interactive multi-node training:

```bash
primus-cli slurm srun -N 4 -p AIG_Model -- train --config ./exp.yaml
```

- `-N 4`: number of nodes
- `-p AIG_Model`: specify Slurm partition

### Batch Mode (sbatch)

Create a batch script:

```bash
#!/bin/bash
#SBATCH --job-name=primus-llama
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --partition=AIG_Model
#SBATCH --output=primus-%j.out

module load rocm/6.4

primus-cli direct -- train --config /mnt/data/exp.yaml
```

Submit it with:

```bash
sbatch run_slurm_job.sh
```

---

## 🐳 Container Mode (Docker / Podman)

Primus supports launching training jobs inside containers for reproducibility and portability.

```bash
primus-cli container --mount /mnt/data:/data -- train --config /data/exp.yaml
```

- `--mount`: bind host directory into the container
- You may mount multiple volumes using repeated `--mount`

### Passing Environment Variables

```bash
primus-cli container --env MASTER_ADDR=10.1.1.1 -- train --config /data/exp.yaml
```

---

## ✅ Best Practices

- Always ensure mounted paths exist (`mkdir -p`)
- For distributed jobs, ensure consistent NCCL/MASTER_ADDR settings
- Use `--debug` to print expanded CLI commands and envs

---

## 🔍 Related Docs

- [Quickstart](../quickstart.md)
- [CLI Usage](../cli.md)
- [Experiment Configuration](../config/overview.md)
