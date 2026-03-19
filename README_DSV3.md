# DeepSeek-V3 Training on AMD MI355X (8-Node)

This guide walks through running DeepSeek-V3 pretraining across 8 nodes (64 GPUs) on AMD MI355X using the `start_training_dsv3.sh` script.

## 1. Repository Setup

```bash
git clone --recurse-submodules git@github.com:AMD-AGI/Primus.git
cd Primus
git checkout dev/benchmark_20260318
git submodule update --init --recursive
```

## 2. Launch Training
There are two launcher modes available for running DeepSeek-V3 training with Primus, each tailored to different environments:

- **Slurm Launcher**
  Use the `slurm` launcher mode when submitting jobs on a Slurm-managed cluster. In this mode, the script will set up Slurm resource requests, orchestrate multiple nodes, and automatically manage pulling the required Docker image on all nodes before launching the distributed training job. This is ideal for large-scale multi-node GPU clusters commonly found in enterprise or research data centers.

- **Direct Launcher**
  Use the `direct` launcher mode when you are *already inside* a training container on each node (for example, inside Kubernetes pods or standardized environments where the Docker image is pre-pulled), or when running locally for debugging/small-scale testing. In this case, distributed training must be manually configured by setting the master node address, ports, node rank, and node count. The `direct` mode avoids Slurm integration and assumes you have direct control over the runtime environment.

Choose the appropriate launcher based on your cluster environment:

- For **multi-node production jobs on managed clusters**, use `LAUNCHER=slurm`.
- For **container-native platforms (Kubernetes, Docker Compose), cloud VM clusters, or testing inside a single container**, use `LAUNCHER=direct`.


Training uses the Docker image `docker.io/tasimage/primus:pr-609-ainic`. This image requires authentication with an AMD-provided token.

### Docker Authentication

- **Slurm**: the launcher handles image pull automatically when these two variables are exported. Set the following environment variables (obtain `token_from_amd` from the AMD TAS team):

```bash
export DOCKER_LOGIN_USER=tasimage
export DOCKER_LOGIN_TOKEN=token_from_amd
```

- **Direct**: you must pull the image manually before launching:

```bash
docker login -u tasimage -p ${DOCKER_LOGIN_TOKEN}
docker pull docker.io/tasimage/primus:pr-609-ainic
```

### 2.1 Slurm Launch

Use this mode when running on a Slurm-managed cluster. The launcher wraps `srun` and handles container orchestration.

```bash
# Optionally configure Slurm-specific parameters:
# export SLURM_TIME=1:00:00
# export SLURM_PARTITION=amd-aig
# export SLURM_NODELIST="node-[001-008]"

LAUNCHER=slurm ./start_training_dsv3.sh
```

### 2.2 Direct Launch

Use this mode when code is already running inside a container (e.g., Kubernetes pods). You need to set the distributed environment on each node:

```bash
# Set on each node accordingly
export MASTER_ADDR=<master-node-ip>
export MASTER_PORT=1234
export NNODES=8
export NODE_RANK=<node-rank>
export GPUS_PER_NODE=8

LAUNCHER=direct ./start_training_dsv3.sh
```

### 2.3 Training Logs

Logs are written to `output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME/`. The default values (defined in `start_training_dsv3.sh`) are:

| Variable | Default Value | Description |
|---|---|---|
| `PRIMUS_TEAM` | `amd` | Team/organization identifier |
| `PRIMUS_USER` | `tas-$(date +%Y%m%d)` | User tag (auto-stamped with today's date) |
| `PRIMUS_EXP_NAME` | `dsv3-pretrain-type_$PRETRAIN_TYPE` | Experiment name (includes precision type) |

For an FP8 run launched on 2026-03-19, the log path would be:

```
output/amd/tas-20260319/dsv3-pretrain-type_FP8/log_node_0.txt
```

## 3. Training Configuration

### 3.1 Precision Mode

Set `PRETRAIN_TYPE` to select the precision mode before launching. This controls which YAML config is loaded:

| `PRETRAIN_TYPE` | Config File |
|---|---|
| `BF16` | `examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml` |
| `FP8` | `examples/megatron/configs/MI355X/deepseek_v3-FP8-pretrain.yaml` |

```bash
# BF16 training
PRETRAIN_TYPE=BF16 LAUNCHER=slurm ./start_training_dsv3.sh

# FP8 training (default)
PRETRAIN_TYPE=FP8 LAUNCHER=slurm ./start_training_dsv3.sh
```

### 3.2 Distributed Strategy

The training uses **PP8 EP8 VPP2** (Pipeline Parallelism 8, Expert Parallelism 8, Virtual Pipeline Parallelism 2).

The VPP stage layout distributes 61 transformer layers across 16 virtual stages (8 PP ranks x 2 VPP stages):

```
pipeline_model_parallel_layout: "Et*3|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3|t*3,L"
```

- `E` — embedding layer (placed on the first stage)
- `L` — final language-model head (placed on the last stage)
- `t*N` — N transformer layers per virtual stage
- `|` — stage boundary

### 3.3 NUMA Binding

**Parameter**: `ENABLE_NUMA_BINDING=1` (env var in `start_training_dsv3.sh`) + `--numa` flag on `primus-cli`

Binds each GPU process to its closest NUMA node, ensuring memory allocations are local to the CPU socket connected to the GPU. This reduces cross-socket memory traffic and improves memory bandwidth utilization, which is critical for stability and throughput on multi-socket systems with large models.

### 3.4 Manual GC

**Parameters** (in the YAML config):

```yaml
manual_gc: true
manual_gc_interval: 1
```

Disables Python's automatic garbage collection and triggers it manually at controlled intervals (every N iterations). Automatic GC can fire at unpredictable times during training, causing stalls that disrupt pipeline parallelism schedules. Manual GC ensures collection happens at safe synchronization points, reducing jitter and improving step-time consistency.

### 3.5 Kernel Fusion

The following kernel fusion flags are enabled in the training configs to reduce kernel launch overhead and intermediate memory allocations:

- **`gradient_accumulation_fusion`** — Fuses the gradient accumulation step directly into the GEMM backward pass, avoiding a separate `add` kernel and reducing memory round-trips.

- **`cross_entropy_loss_fusion`** — Fuses the softmax and cross-entropy loss computation into a single kernel (via TransformerEngine), reducing peak memory from the full logits tensor.

- **`apply_rope_fusion`** — Fuses Rotary Position Embedding (RoPE) application into a single kernel instead of decomposing it into multiple trigonometric and element-wise ops.

- **`moe_permute_fusion`** — Fuses the token-to-expert permutation and un-permutation operations in MoE layers, cutting down on intermediate tensor allocations and memory copies.

- **`moe_use_fused_router_with_aux_score`** — Fuses the MoE router gating computation with the auxiliary load-balancing loss calculation into a single kernel, avoiding redundant top-k and score computations.

### 3.6 Turbo Optimization

These optimizations are provided by the [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) performance layer:

- **`use_turbo_attention`** — Integrates high-performance FlashAttention assembly kernels from [AITER](https://github.com/ROCm/aiter), delivering optimized fused attention for AMD GPUs.

- **`use_turbo_deepep`** — Enables the DeepEP optimization for MoE all-to-all communication. Reduces redundant token transfers across nodes by pruning zero-routed tokens before the inter-node exchange, improving communication bandwidth efficiency.

- **`turbo_sync_free_moe_stage`** — Enables CPU sync-free MoE execution. Eliminates device-to-host (D2H) synchronization points during MoE dispatch/combine, improving kernel launch efficiency by allowing the CPU to pipeline kernel submissions without blocking on GPU completion signals. Stage 1 is currently enabled; stage 2 provides further overlap.

## 4. Preflight Sanity Check

Primus includes a built-in preflight diagnostic tool for validating cluster health before launching large training runs. It checks host/GPU/network configuration and can run GEMM and communication performance tests to identify outlier nodes.

Refer to [docs/preflight.md](./docs/preflight.md) for full details. Quick examples:

```bash
# Info report only (fast) — single node
primus-cli direct -- preflight --host --gpu --network

# Full preflight with perf tests — multi-node via Slurm
primus-cli slurm srun -N 8 -- preflight

# Perf tests only
primus-cli slurm srun -N 8 -- preflight --perf-test
```

Reports are written to `output/preflight/` by default, including markdown and PDF summaries.
