# Backend training recipes

Task-oriented, copy-paste commands for launching pretraining runs with each Primus backend on AMD Instinct™ GPUs.

This section is for users who already know what they want to run and need the specific command for a given model, precision, and GPU. To understand concepts related to the Primus workflow (how backends work, YAML structure and inheritance, parallelism vocabulary, the full per-backend configuration inventory, etc.), see **[Pretraining](pretraining.md)**.

> **Authoritative full matrices.** AMD publishes per-model reproduction pages with verified images, commits, and tuned batch sizes for every supported model at the following locations—treat them as the source of truth for achieving the expected performance; this page only gives the canonical *pattern* plus a representative example per backend and links to other reference materials.
>
> - [Training with Primus + Megatron-LM](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/primus-megatron.html)
> - [Training with Primus + PyTorch (TorchTitan)](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/primus-pytorch.html)
> - [Training with Primus + JAX MaxText](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/jax-maxtext.html)

---

## How recipes are structured

Every recipe follows the same four-step pattern:

1. **Pull and launch the AMD Docker image** (for a reproducible environment).
2. **Set the GPU-architecture environment** (performance environment variable settings differ by GPU).
3. **Pick the configuration YAML** for your GPU architecture under `examples/<backend>/configs/<ARCH>/` in the [Primus repository](https://github.com/AMD-AGI/Primus).
4. **Launch** with `runner/primus-cli` in `direct`, `container`, or `slurm` mode.

### GPU-architecture config folders

Configuration YAMLs are organized by GPU architecture. Always pick the folder that matches your hardware:


| Backend                             | `MI300X` | `MI325X` | `MI355X` / `MI350X` |
| ----------------------------------- | -------- | -------- | ------------------- |
| `examples/megatron/configs/`        | yes      | yes      | yes                 |
| `examples/torchtitan/configs/`      | yes      | yes      | yes                 |
| `examples/maxtext/configs/`         | yes      | —        | yes                 |
| `examples/megatron_bridge/configs/` | yes      | —        | yes                 |


> MI350X uses the same configurations as MI355X because both are based on the gfx950 architecture. If a configuration for your model is not available in the architecture-specific folder, use the closest match from the same generation as a starting point.

### GPU-architecture environment variables

MI300X and MI325X benefit from the following performance settings; MI355X/MI350X do **not** need them:

```bash
# MI300X / MI325X only -- improves performance
export HSA_NO_SCRATCH_RECLAIM=1
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1
export NVTE_CK_IS_V3_ATOMIC_FP32=1
```

### Choosing the Docker image

For **container** and **Slurm** modes (direct mode runs in whatever environment you launched it from), the default image is `rocm/primus:v26.3` (`runner/.primus.yaml`). For reproducing published benchmarks, use the AMD-published tag for your release (the AMD pages under **Authoritative full matrices** above list the most current tag). JAX MaxText has its own separate image family of `rocm/jax-training:maxtext-...`.

Image is picked in the priority order of `DOCKER_IMAGE` environment variable > `--image` CLI argument > config file. See [Selecting the container image](../01-getting-started/quickstart.md#selecting-the-container-image) for a full explanation, and [Configuration system](configuration-system.md) for configuration loading.

---

## Shared setup for all backends

These apply across all backends. Set them up before running the recipes below.

### Hugging Face token (for gated models or real data)

```bash
export HF_TOKEN=<your_hf_token>
```

`runner/.primus.yaml` forwards `HF_TOKEN` into the container automatically. MaxText configurations might also read `${HF_TOKEN:""}` directly.

### Mock vs. real data

- **Mock/synthetic data** (default for most examples): validates the stack without datasets. Megatron and TorchTitan set `mock_data: true`; MaxText sets `dataset_type: "synthetic"`.
- **Real data:** set `mock_data: false` and point `train_data_path` (for Megatron) or the backend's dataset fields at paths visible *inside* your container mounts.

### Multi-node networking checklist

The `primus-cli` launcher sets sensible `NCCL_`* defaults, but auto-detection can pick the wrong device on multi-NIC nodes. Before multi-node, confirm and export if needed:

```bash
export NCCL_IB_HCA=<your_rdma_interfaces>      # from `ibv_devices`
export NCCL_SOCKET_IFNAME=<your_net_interface> # from `ip a`
export GLOO_SOCKET_IFNAME=<same_as_NCCL_SOCKET_IFNAME>
export NCCL_IB_GID_INDEX=3                      # 3 for RoCE (1 for AMD AINIC)
```

For AMD AINIC clusters also set `USING_AINIC=1`, `NCCL_PXN_DISABLE=0`, `NCCL_IB_GID_INDEX=1`. See [Multi-Node Networking](../04-technical-guides/multi-node-networking.md) for the full reference.

---

## Megatron-LM

**Image:** `rocm/primus`  |  **Configurations:** `examples/megatron/configs/<ARCH>/`  |  **Precisions:** BF16, FP8

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

Access the container later with `docker start primus_training_env && docker exec -it primus_training_env bash`.

### 2. Run pretraining (direct mode, inside the container)

Pretrain Llama 3.1 8B BF16 on **MI355X / MI350X**:

```bash
./runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
```

Pretrain the same model on **MI300X / MI325X** (add the performance environment variables):

```bash
export HSA_NO_SCRATCH_RECLAIM=1
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1
export NVTE_CK_IS_V3_ATOMIC_FP32=1

./runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

Switch model or precision by changing the config filename (e.g. `llama3.1_70B-FP8-pretrain.yaml`, `mixtral_8x7B_v0.1-BF16-pretrain.yaml`). The full configuration inventory is the repository's `examples/megatron/configs/<ARCH>/` directory. See the parallelism table in [Pretraining](pretraining.md#example-configurations-under-examplesmegatronconfigsmi300x).

**Model-specific notes:**

- **Zebra-Llama** (hybrid Mamba+MLA) pretrain presets ship at `examples/megatron/configs/<ARCH>/zebra_llama_{1B,3B,8B}-pretrain.yaml` and run via the standard core runtime; Megatron Bridge SFT variants live under `examples/megatron_bridge/configs/<ARCH>/`.
- **MoE models** (DeepSeek-V2-Lite, Mixtral) might need extra grouped-GEMM or router flags; [Training with Primus + Megatron-LM](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/primus-megatron.html) lists the exact flags per model.

### 3. Multi-node (Slurm mode)

```bash
./runner/primus-cli slurm srun -N 8 -p <partition> -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml \
  --micro_batch_size 4 --global_batch_size 1024
```

Scale batch size with node count and align `tensor_model_parallel_size`, `pipeline_model_parallel_size`, and `expert_model_parallel_size` to your topology. See the [multi-node networking checklist](#multi-node-networking-checklist) above.

---

## TorchTitan (PyTorch)

**Image:** `rocm/primus`  |  **Configurations:** `examples/torchtitan/configs/<ARCH>/`  |  **Precisions:** BF16, FP8

Use the same `rocm/primus` container as Megatron (step 1 above). TorchTitan parameters use a dotted namespace (e.g. `--training.local_batch_size`).

### Run pretraining (direct mode)

Pretrain Llama 3.1 8B BF16 on **MI355X / MI350X**:

```bash
./runner/primus-cli direct \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
```

On **MI300X / MI325X**, export the performance environment variables first (see above) and use the `MI300X` config path.

### Multi-node (Slurm mode)

```bash
./runner/primus-cli slurm srun -N 4 -- train pretrain \
  --config examples/torchtitan/configs/MI355X/llama3.1_70B-FP8-pretrain.yaml \
  --training.local_batch_size 6 \
  --training.global_batch_size 192 \
  --training.mock_data True
```

Available models include Llama 3.1 (8B/70B/405B), Llama 4, DeepSeek V3, and Qwen 3. See the `examples/torchtitan/configs/<ARCH>/` directory in the repository.

---

## JAX MaxText

**Image:** `rocm/jax-training:maxtext-...` (separate family from the other backends)  |  **Configurations:** `examples/maxtext/configs/<ARCH>/`

MaxText uses a different Docker image than Megatron and TorchTitan, and it is **not** the default image pointed to in `runner/.primus.yaml`. In container or Slurm mode, you must point Primus at your MaxText image explicitly.

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

If you run Primus directly on the host instead of inside the prebuilt Docker image, install the JAX dependencies first by `pip install -r requirements-jax.txt`.

### 2. Run pretraining

Direct mode (inside the container)—pretraining Llama 3 8B on **MI355X**:

```bash
./runner/primus-cli direct \
  -- train pretrain \
  --config examples/maxtext/configs/MI355X/llama3_8B-pretrain.yaml
```

Container mode—passing the MaxText image with `--image`:

```bash
./runner/primus-cli container --image rocm/jax-training:maxtext-v26.4-jax0.9.1-te2.12.0 \
  -- train pretrain \
  --config examples/maxtext/configs/MI355X/llama3_8B-pretrain.yaml
```

Slurm mode—supplying the image (and any environment variables) via a config file:

```bash
./runner/primus-cli --config my_maxtext_config.yaml slurm srun -N 8 \
  -- train pretrain \
  --config examples/maxtext/configs/MI300X/llama3_8B-pretrain.yaml
```

MaxText parallelism is set with `ici_*` (intra-node) and `dcn_*` (inter-node) fields—see the [MaxText config table](pretraining.md#maxtext-jax-pretraining) and [MaxText parameters](../03-configuration-reference/maxtext-parameters.md).

---

## Megatron Bridge (post-training)

Megatron Bridge configurations are under `examples/megatron_bridge/configs/<ARCH>/` in the repository and are primarily **SFT and LoRA post-training** recipes (e.g. `qwen3_32b_sft_posttrain.yaml`, `llama31_70b_lora_posttrain.yaml`). Launch with `train posttrain`:

```bash
./runner/primus-cli direct \
  --log_file /tmp/primus_qwen3_32b_sft.log \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain.yaml
```

See [Post-training](posttraining.md) for the full SFT/LoRA workflow.

---

## Related documentation

- [Pretraining](pretraining.md): backend concepts, configuration walkthroughs, parallelism vocabulary, full configuration inventories.
- [Post-training](posttraining.md): SFT and LoRA via Megatron Bridge.
- [CLI reference](cli-reference.md): `direct` / `container` / `slurm` modes and flags.
- [Configuration system](configuration-system.md): YAML inheritance, overrides, image/env precedence.
- [Performance tuning](../04-technical-guides/performance-tuning.md): HipBLASLt autotuning, Primus-Turbo, FP8, MoE.
