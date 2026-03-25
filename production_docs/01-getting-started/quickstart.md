# Quickstart (about five minutes)

This guide runs a **small Megatron-LM pretraining example** with **mock data** so you can validate the stack without preparing a full dataset. The same experiment YAML works across **direct**, **container**, and **Slurm** modes.

**Prerequisites:** ROCm, Docker (for container mode), and a cloned repository with submodules. See [installation](./installation.md).

---

## Step 1: Pull the container image

```bash
docker pull docker.io/rocm/primus:v26.1
```

---

## Step 2: Clone the repository

```bash
git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
cd Primus
```

---

## Step 3: Run training (container)

From the repository root:

```bash
./primus-cli container --image rocm/primus:v26.1 -- \
  train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
```

If your config accesses Hugging Face Hub for weights or tokenizers, pass credentials into the container:

```bash
./primus-cli container --image rocm/primus:v26.1 \
  --env HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" -- \
  train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
```

---

## Expected output

You should see the backend initialize distributed processes, load the experiment configuration, and emit **iteration-level logs** (loss, throughput, step index). Exact fields depend on the backend and logging configuration; a typical pattern resembles:

```
... [INFO] starting training ...
... iteration      1 | loss: 10.xxx | ...
... iteration      2 | loss:  9.xxx | ...
```

Let the job run briefly to confirm stability; stop with `Ctrl+C` when satisfied.

---

## Same config, three execution modes

Use one experiment file: `examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml`.

| Mode | Example command |
|------|------------------|
| **Container** | `./primus-cli container --image rocm/primus:v26.1 -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml` |
| **Direct** | `./primus-cli direct -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml` |
| **Slurm** | `./primus-cli slurm srun -N <nodes> ... -- container --image rocm/primus:v26.1 -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml` |

Replace `<nodes>` and Slurm resource flags with values appropriate for your cluster.

---

## Command structure

`primus-cli` parses **global options**, a **mode** (`direct`, `container`, `slurm`, …), optional **mode-specific arguments**, then a **`--` separator** followed by the **subcommand and its arguments** (for example `train`, `benchmark`).

```
primus-cli [global-options] <mode> [mode-args] -- <command> [command-args...]
```

Example:

```text
primus-cli  container  --image rocm/primus:v26.1  --  train pretrain --config path/to/experiment.yaml
            ^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            mode        mode args                     command + args
```

---

## Next steps

| Topic | Document |
|-------|----------|
| Full CLI flags and subcommands | [CLI reference](../02-user-guide/cli-reference.md) |
| YAML presets, overrides, and composition | [Configuration system](../02-user-guide/configuration-system.md) |
| Pretraining workflows and backend notes | [Pretraining workflows](../02-user-guide/pretraining-workflows.md) |
| Terminology | [Glossary](./glossary.md) |

Upstream project docs under `docs/` (for example `docs/cli/PRIMUS-CLI-GUIDE.md`) provide additional detail that may overlap with the production doc set.
