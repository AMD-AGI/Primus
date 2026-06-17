# Backend Training Recipes

Task-oriented, copy-paste commands for launching pretraining runs with each Primus backend on AMD Instinct GPUs.

> **Pretraining has two docs — pick the one that matches your question:**
> - **Use this doc** when you already know *what* you want to run and need the exact command for a given model, precision, and GPU.
> - **Use [Pretraining](pretraining.md)** when you want to understand the *why*: how backends work, YAML structure and inheritance, parallelism vocabulary, and the full per-backend config inventory.

> **Authoritative full matrices.** AMD publishes per-model reproduction pages with verified images, commits, and tuned batch sizes for every supported model. Treat those as the source of truth for exact performance reproduction; this doc gives the canonical *pattern* plus a representative example per backend and links out.
>
> - [Training with Primus + Megatron-LM](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/primus-megatron.html)
> - [Training with Primus + PyTorch (TorchTitan)](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/primus-pytorch.html)
> - [Training with Primus + JAX MaxText](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/jax-maxtext.html)

---

## How recipes are structured

Every recipe follows the same four-step pattern:

1. **Pull and launch the AMD Docker image** (reproducible environment).
2. **Set the GPU-architecture environment** (perf env vars differ by GPU).
3. **Pick the experiment YAML** for your GPU arch under `examples/<backend>/configs/<ARCH>/`.
4. **Launch** with `runner/primus-cli` in `direct`, `container`, or `slurm` mode.

### GPU-architecture config folders

Experiment YAMLs are organized by GPU architecture. Always pick the folder that matches your hardware:


| Backend                             | `MI300X` | `MI325X` | `MI355X` / `MI350X` |
| ----------------------------------- | -------- | -------- | ------------------- |
| `examples/megatron/configs/`        | yes      | yes      | yes                 |
| `examples/torchtitan/configs/`      | yes      | yes      | yes                 |
| `examples/maxtext/configs/`         | yes      | —        | yes                 |
| `examples/megatron_bridge/configs/` | yes      | —        | yes                 |


> MI350X uses the same configs as MI355X (both gfx950). If your exact arch folder is missing a model, the closest same-generation folder is the right starting point.

### GPU-architecture environment variables

MI300X and MI325X benefit from the following performance settings; MI355X/MI350X do **not** need them:

```bash
# MI300X / MI325X only -- improves performance
export HSA_NO_SCRATCH_RECLAIM=1
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1
export NVTE_CK_IS_V3_ATOMIC_FP32=1
```

### Choosing the Docker image

Applies to **container** and **Slurm** modes (direct mode runs in whatever environment you launched it from). The repo default is `rocm/primus:v26.2` (`runner/.primus.yaml`); for reproducing published benchmarks use the AMD-published tag for your release (the AMD pages above list the current tag, e.g. `rocm/primus:v26.3`). JAX MaxText ships a separate image family, `rocm/jax-training:maxtext-...`.

Image precedence is `DOCKER_IMAGE` env var > `--image` CLI arg > config file — see [Selecting the container image](../01-getting-started/quickstart.md#selecting-the-container-image) for the full explanation, and [Configuration System](configuration-system.md) for config loading.

---

## Shared setup (read first)

These apply across all backends — set them up before running the recipes below.

### Hugging Face token (gated models / real data)

```bash
export HF_TOKEN=<your_hf_token>
```

`runner/.primus.yaml` forwards `HF_TOKEN` into container mode automatically. MaxText configs may also read `${HF_TOKEN:""}` directly.

### Mock vs. real data

- **Mock/synthetic data** (default for most examples): validates the stack without datasets. Megatron/TorchTitan set `mock_data: true`; MaxText sets `dataset_type: "synthetic"`.
- **Real data:** set `mock_data: false` and point `train_data_path` (Megatron) or the backend's dataset fields at paths visible *inside* your container mounts.

### Multi-node networking checklist

The `primus-cli` launcher sets sensible `NCCL_`* defaults, but auto-detection can pick the wrong device on multi-NIC nodes. Before multi-node runs, confirm and (if needed) export:

```bash
export NCCL_IB_HCA=<your_rdma_interfaces>      # from `ibv_devices`
export NCCL_SOCKET_IFNAME=<your_net_interface> # from `ip a`
export GLOO_SOCKET_IFNAME=<same_as_NCCL_SOCKET_IFNAME>
export NCCL_IB_GID_INDEX=3                      # 3 for RoCE (1 for AMD AINIC)
```

For AMD AINIC clusters also set `USING_AINIC=1`, `NCCL_PXN_DISABLE=0`, `NCCL_IB_GID_INDEX=1`. See [Multi-Node Networking](../04-technical-guides/multi-node-networking.md) for the full reference.

---

## Megatron-LM

**Image:** `rocm/primus`  |  **Configs:** `examples/megatron/configs/<ARCH>/`  |  **Precisions:** BF16, FP8

### 1. Launch the container

```bash
docker pull rocm/primus:v26.3
docker run -it \
    --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --privileged \
    -v $HOME:$HOME --shm-size 128G \
    --name primus_training_env \
    rocm/primus:v26.3
```

Re-enter later with `docker start primus_training_env && docker exec -it primus_training_env bash`.

### 2. Run pretraining (direct mode, inside the container)

Llama 3.1 8B BF16 on **MI355X / MI350X**:

```bash
./runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
```

Same model on **MI300X / MI325X** (add the perf env vars):

```bash
export HSA_NO_SCRATCH_RECLAIM=1
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1
export NVTE_CK_IS_V3_ATOMIC_FP32=1

./runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

Switch model/precision by changing the config filename (e.g. `llama3.1_70B-FP8-pretrain.yaml`, `mixtral_8x7B_v0.1-BF16-pretrain.yaml`). The full inventory is the `examples/megatron/configs/<ARCH>/` directory; see the parallelism table in [Pretraining](pretraining.md#example-configs-under-examplesmegatronconfigsmi300x).

**Model-specific notes:**

- **Zebra-Llama** configs require the legacy runtime: prefix with `PRIMUS_TRAIN_RUNTIME=legacy`.
- **MoE models** (DeepSeek-V2-Lite, Mixtral) may need extra grouped-GEMM/router flags; the AMD Megatron page lists the exact flags per model.

### 3. Multi-node (Slurm)

```bash
./runner/primus-cli slurm srun -N 8 -p <partition> -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml \
  --micro_batch_size 4 --global_batch_size 1024
```

Scale batch size with node count and align `tensor_model_parallel_size` / `pipeline_model_parallel_size` / `expert_model_parallel_size` to your topology. See the [multi-node networking checklist](#multi-node-networking-checklist) above.

---

## TorchTitan (PyTorch)

**Image:** `rocm/primus`  |  **Configs:** `examples/torchtitan/configs/<ARCH>/`  |  **Precisions:** BF16, FP8

Use the same `rocm/primus` container as Megatron (step 1 above). TorchTitan parameters use a dotted namespace (e.g. `--training.local_batch_size`).

### Run pretraining (direct mode)

Llama 3.1 8B BF16 on **MI355X / MI350X**:

```bash
./runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
```

On **MI300X / MI325X**, export the perf env vars first (see above) and use the `MI300X` config path.

### Multi-node (Slurm)

```bash
./runner/primus-cli slurm srun -N 4 -- train pretrain \
  --config examples/torchtitan/configs/MI355X/llama3.1_70B-FP8-pretrain.yaml \
  --training.local_batch_size 6 \
  --training.global_batch_size 192 \
  --training.mock_data True
```

Available models include Llama 3.1 (8B/70B/405B), Llama 4, DeepSeek V3, and Qwen 3 — see `examples/torchtitan/configs/<ARCH>/`.

---

## JAX MaxText

**Image:** `rocm/jax-training:maxtext-...` (separate family)  |  **Configs:** `examples/maxtext/configs/<ARCH>/`

MaxText uses a different Docker image than Megatron/TorchTitan, and it is **not** the default in `runner/.primus.yaml`. In container/Slurm mode you must point Primus at it explicitly.

### 1. Launch the container

```bash
docker pull rocm/jax-training:maxtext-v26.4-jax0.9.1-te2.12.0
docker run -it \
    --device /dev/dri --device /dev/kfd \
    --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --privileged \
    -v $HOME:$HOME -v $HOME/.ssh:/root/.ssh \
    --shm-size 64G \
    --name training_env \
    rocm/jax-training:maxtext-v26.4-jax0.9.1-te2.12.0
```

If you run Primus directly on the host instead of the prebuilt image, install JAX deps first: `pip install -r requirements-jax.txt`.

### 2. Run pretraining

Direct mode (inside the container) — Llama 3 8B on **MI355X**:

```bash
./runner/primus-cli direct \
  -- train pretrain \
  --config examples/maxtext/configs/MI355X/llama3_8B-pretrain.yaml
```

Container mode — pass the MaxText image with `--image`:

```bash
./runner/primus-cli container --image rocm/jax-training:maxtext-v26.4-jax0.9.1-te2.12.0 \
  -- train pretrain \
  --config examples/maxtext/configs/MI355X/llama3_8B-pretrain.yaml
```

Slurm mode — supply the image (and any env) via a config file:

```bash
./runner/primus-cli --config my_maxtext_config.yaml slurm srun -N 8 \
  -- train pretrain \
  --config examples/maxtext/configs/MI300X/llama3_8B-pretrain.yaml
```

MaxText parallelism is expressed with `ici_*` (intra-node) and `dcn_*` (inter-node) fields — see the [MaxText config table](pretraining.md#maxtext-jax-pretraining) and [MaxText Parameters](../03-configuration-reference/maxtext-parameters.md).

---

## Megatron Bridge (post-training)

Megatron Bridge configs live under `examples/megatron_bridge/configs/<ARCH>/` and are primarily **SFT / LoRA post-training** recipes (e.g. `qwen3_32b_sft_posttrain.yaml`, `llama31_70b_lora_posttrain.yaml`). Launch with `train posttrain`:

```bash
./runner/primus-cli direct \
  --log_file /tmp/primus_qwen3_32b_sft.log \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain.yaml
```

See [Post-Training](posttraining.md) for the full SFT/LoRA workflow.

---

## See also

- [Pretraining](pretraining.md) — backend concepts, config walkthroughs, parallelism vocabulary, full config inventories.
- [Post-Training](posttraining.md) — SFT and LoRA via Megatron Bridge.
- [CLI Reference](cli-reference.md) — `direct` / `container` / `slurm` modes and flags.
- [Configuration System](configuration-system.md) — YAML inheritance, overrides, image/env precedence.
- [Performance Tuning](../04-technical-guides/performance-tuning.md) — HipBLASLt autotuning, Primus-Turbo, FP8, MoE.

