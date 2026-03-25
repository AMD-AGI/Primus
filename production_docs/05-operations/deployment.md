# Deployment Guide

This guide describes how to deploy Primus training across **container**, **direct (bare metal)**, and **Slurm** environments using the unified `primus-cli` launcher. For environment variable semantics, see [Environment variables](../03-configuration-reference/environment-variables.md). For YAML hierarchy and precedence, see [Configuration system](../02-user-guide/configuration-system.md).

---

## 1. Deployment overview

Primus supports three deployment modes:

| Mode | Description | Typical use |
|------|-------------|-------------|
| **Container** | Docker/Podman with ROCm-capable GPU devices and capabilities | Recommended default; reproducible images |
| **Direct** | Runs on the current host (or inside an existing container) | Local debugging, single-node, clusters with ROCm on nodes |
| **Slurm** | Wraps `srun`/`sbatch` and launches per-node entry scripts | Multi-node clusters with Slurm |

**Container image:** `docker.io/rocm/primus:v26.1` (default in `runner/.primus.yaml`). For clusters using **AINIC**, use `runner/use_ainic.yaml` and tune the image and NCCL-related variables (for example `USING_AINIC`, `NCCL_IB_GID_INDEX`) to match your fabric.

**Prerequisites (baseline):**

- **AMD ROCm** >= 7.0 on the host (or in the image when using containers)
- **Docker** or **Podman** >= 24.0 when using container mode
- **AMD Instinct** GPUs and working ROCm stack (`rocm-smi` should report devices)

---

## 2. Container deployment

### 2.1 Pull the image

```bash
docker pull docker.io/rocm/primus:v26.1
```

The default `container.options.image` in `runner/.primus.yaml` is `rocm/primus:v26.1` (equivalent to `docker.io/rocm/primus:v26.1` when the registry is omitted).

### 2.2 Required device mounts

System defaults (`runner/.primus.yaml`, `container.options.device`) pass each path as `--device` to the runtime:

| Device | Purpose |
|--------|---------|
| `/dev/kfd` | Kernel Fusion Driver (ROCm core) |
| `/dev/dri` | Direct Rendering Infrastructure (GPU access) |
| `/dev/infiniband` | InfiniBand character devices (multi-node / RDMA) |

### 2.3 Required capabilities

Defaults (`container.options.cap-add`):

| Capability | Purpose |
|------------|---------|
| `SYS_PTRACE` | Debugging and profiling tools |
| `CAP_SYS_ADMIN` | Administrative operations required by some ROCm/GPU workflows |

### 2.4 Container runtime options

Defaults in `runner/.primus.yaml` include:

| Option | Value |
|--------|--------|
| `ipc` | `host` |
| `network` | `host` |
| `privileged` | `true` |
| `security-opt` | `seccomp=unconfined` |
| `group-add` | `video` |

`primus-cli-container.sh` always mounts the **Primus repository root** into the container at the same path (`-v $PRIMUS_PATH:$PRIMUS_PATH`). Mount additional paths for **datasets**, **model weights**, and **outputs** with `--volume` (or `container.options.volume` in YAML).

### 2.5 Environment passthrough

`container.options.env` lists names that are forwarded into the **inner** `primus-cli` invocation as `--env` when set on the host (see `runner/.primus.yaml`). Examples include:

`MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `NODE_RANK`, `GPUS_PER_NODE`, `DOCKER_IMAGE`, `HF_TOKEN`, `WANDB_API_KEY`, `ENABLE_NUMA_BINDING`, `USING_AINIC`, and NCCL/GLOO socket and IB-related variables (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `NCCL_IB_GID_INDEX`, and others).

Additionally, `primus-cli-container.sh` auto-forwards any environment variable whose name starts with `PRIMUS_`, `NCCL_`, `RCCL_`, `GLOO_`, `IONIC_`, or `HIPBLASLT_` when present on the host.

### 2.6 Single-node example

```bash
./primus-cli container --volume /data:/data -- train pretrain --config /data/exp.yaml
```

### 2.7 Multi-node container deployment

Set `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `NODE_RANK`, and `GPUS_PER_NODE` on each node (Slurm or your orchestrator sets these; see `runner/primus-cli-slurm-entry.sh`). Example pattern when launching manually:

```bash
export MASTER_ADDR=<head-node-hostname>
export MASTER_PORT=1234
export NNODES=4
export NODE_RANK=<0-based index for this node>
export GPUS_PER_NODE=8

./primus-cli container -- train pretrain --config /path/to/config.yaml
```

Use `--clean` before launch to remove existing containers (`primus-cli-container.sh`).

---

## 3. Slurm deployment

### 3.1 `srun` (interactive or blocking)

```bash
./primus-cli slurm srun -N <nodes> -p <partition> -- container --image rocm/primus:v26.1 -- train pretrain --config <yaml>
```

If you omit `--image`, the value from `runner/.primus.yaml` (`rocm/primus:v26.1`) applies.

### 3.2 `sbatch` (batch jobs)

```bash
./primus-cli slurm sbatch -N <nodes> -p <partition> --time <HH:MM:SS> --job-name <name> -o <logfile> -- \
  container -- train pretrain --config <yaml>
```

Add `-e <errfile>` if you want separate stderr.

### 3.3 Slurm-to-Primus environment mapping

`runner/primus-cli-slurm-entry.sh` sets:

| Variable | Source (typical) |
|----------|-------------------|
| `NNODES` | `SLURM_NNODES`, or `SLURM_JOB_NUM_NODES`, or existing `NNODES` |
| `NODE_RANK` | `SLURM_NODEID`, or `SLURM_PROCID`, or existing `NODE_RANK` |
| `GPUS_PER_NODE` | Default `8` if unset |
| `MASTER_ADDR` | First host in `SLURM_NODELIST` if unset |
| `MASTER_PORT` | Default `1234` if unset |

The entry script then exports `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, `NODE_RANK`, and `GPUS_PER_NODE` into the container launcher.

### 3.4 Slurm YAML defaults (`runner/.primus.yaml`)

| Key | Default |
|-----|---------|
| `slurm.nodes` | `1` |
| `slurm.gpus_per_node` | `8` |
| `slurm.time` | `"4:00:00"` |
| `slurm.partition` | (commented; set per site) |

CLI Slurm flags override YAML when both are specified (see `runner/primus-cli-slurm.sh`).

### 3.5 Entry after the first `--`

Production examples use **`container`** as the Primus entry after the Slurm `--` separator, for example:

```bash
./primus-cli slurm srun -N 4 -p gpu -- container -- train pretrain --config exp.yaml
```

The shipped `primus-cli-slurm-entry.sh` invokes **`primus-cli-container.sh`** with distributed variables set from Slurm. For **bare-metal** nodes without Docker, run `primus-cli direct` under your allocation and ensure the same distributed variables and ROCm layout as in [Multi-node configuration](#5-multi-node-configuration).

---

## 4. Kubernetes deployment

Kubernetes integration is **not** shipped as a Helm chart or operator in this repository. The repo includes **`examples/run_k8s_pretrain.sh`**, a client script that talks to a Kubernetes **API** to create and manage training workloads (image default `docker.io/rocm/primus:v26.1`).

Use that script as a reference for your platform; adapt networking, storage, and scheduling to your cluster policies.

---

## 5. Multi-node configuration

Required variables for distributed training:

| Variable | Role |
|----------|------|
| `MASTER_ADDR` | Hostname or IP of rank-0 process |
| `MASTER_PORT` | TCP port for the process group rendezvous |
| `NNODES` | Number of nodes |
| `NODE_RANK` | Zero-based index of this node |
| `GPUS_PER_NODE` | GPUs per node used by `torchrun` |

**Flow:** User or Slurm sets the environment → `primus-cli` and `primus-cli-direct.sh` load GPU and comm settings → **`torchrun`** launches `primus/cli/main.py` with the distributed topology.

**Defaults from `runner/.primus.yaml` (`direct` section):**

| Key | Default |
|-----|---------|
| `direct.master_port` | `1234` |
| `direct.gpus_per_node` | `8` |
| `direct.nnodes` | `1` |
| `direct.master_addr` | `"localhost"` |

---

## 6. Startup and shutdown

**Lifecycle (high level):**

1. Parse CLI and load YAML (`--config` chain: see [Configuration system](../02-user-guide/configuration-system.md)).
2. Load environment (GPU detection, hooks, patches in `primus-cli-direct.sh`).
3. Launch training via **`torchrun`** into the Python CLI.

**Verification:**

- `--dry-run` prints the command that would run without executing (supported in container and Slurm scripts).
- `--debug` sets `PRIMUS_LOG_LEVEL=DEBUG` for verbose launcher and shell logging.

**Shutdown:**

- Normal completion or **Ctrl+C** terminates the training process.
- In container mode, **`--clean`** removes existing containers before launch (`primus-cli-container.sh`).

**Timeouts (config):**

| Backend | Parameter | Location |
|---------|-----------|----------|
| Megatron | `distributed_timeout_minutes` | `primus/configs/modules/megatron/trainer_base.yaml` (default `10`) |
| TorchTitan | `comm.init_timeout_seconds` | `primus/configs/modules/torchtitan/pre_trainer.yaml` (default `300`) |

---

## 7. Production checklist

| Item | Action |
|------|--------|
| ROCm drivers | Install and verify with `rocm-smi` |
| Container image | Pulled and aligned with host ROCm expectations |
| Network | Run `preflight --network` (see [Preflight](../02-user-guide/preflight.md)) |
| Shared data | Paths visible and consistent on all nodes |
| Hugging Face | Set `HF_TOKEN` if using gated models |
| Checkpoints | Save directory on shared or replicated storage with sufficient space |
| Monitoring | Configure WandB or TensorBoard (see [Monitoring and Logging](./monitoring-logging.md)) |
| Resources | Slurm time limits, partitions, and GPU counts match your YAML and hardware |

---

## Related documentation

- [CLI reference](../02-user-guide/cli-reference.md)
- [Primus CLI user guide](../../docs/cli/PRIMUS-CLI-GUIDE.md) (upstream examples and troubleshooting)
- [Multi-node networking](../04-technical-guides/multi-node-networking.md)
