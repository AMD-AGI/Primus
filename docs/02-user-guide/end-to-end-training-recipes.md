# End-to-end training recipes

Task-oriented, copy-paste commands for launching pretraining runs with each Primus backend on AMD Instinct™ GPUs.

This page covers what is common to every backend — image, architecture folders, environment, shared setup — and gives one worked example each. For the **complete per-model command set**, follow the backend recipe page in the table below. For the concepts behind the workflow (how backends work, YAML structure and inheritance, parallelism vocabulary, the full configuration inventory), see [Pretraining](pretraining.md).

## Choose your recipe

| Backend | Image family | Configurations | Full recipe |
| ------- | ------------ | -------------- | ----------- |
| Megatron-LM | `rocm/primus` | `examples/megatron/configs/<ARCH>/` | [Megatron-LM training](megatron-lm-training.md) |
| TorchTitan (PyTorch) | `rocm/primus` | `examples/torchtitan/configs/<ARCH>/` | [TorchTitan training](torchtitan-training.md) |
| JAX MaxText | `rocm/jax-training:maxtext-…` | `examples/maxtext/configs/<ARCH>/` | [JAX MaxText training](jax-maxtext-training.md) |
| Megatron Bridge (post-training) | `rocm/primus` | `examples/megatron_bridge/configs/<ARCH>/` | [Post-training](posttraining.md) |

Each backend recipe page opens with an **Important notes** section listing the settings that release requires, the architecture-specific tuning, and any known issues. Read it before your first run on a new image tag.

> **Image contents.** The exact ROCm, PyTorch/JAX, Transformer Engine, and RCCL versions in every published tag are in [Release notes](../01-getting-started/release-notes.md), which is the single source of truth for image contents.

---

## How recipes are structured

Every recipe follows the same four-step pattern:

1. **Clone the Primus branch** matching your image, on the host.
2. **Set the GPU-architecture environment** (performance environment variable settings differ by GPU).
3. **Pick the configuration YAML** for your GPU architecture under `examples/<backend>/configs/<ARCH>/` in the [Primus repository](https://github.com/AMD-AGI/Primus).
4. **Launch** with `runner/primus-cli` — in `container` mode from the host, which starts the container for you.

### Choosing a launch mode

| Mode | Run it from | What it does |
| ---- | ----------- | ------------ |
| **`container`** (recommended) | The host, in your Primus checkout | Starts the container, mounts your checkout into it at the same path, and runs the training inside. Nothing to set up by hand. |
| `direct` | **Inside** a container, or a bare-metal install | Runs training in the current environment. Use it when you already have a shell inside the container, or when Primus is installed directly on the host. |
| `slurm` | The host, in your Primus checkout | Allocates nodes and runs `container` mode on each of them. |

Container mode mounts your checkout at the same absolute path inside the container and runs from there, so **the branch you cloned is the code that executes** — the `/workspace/Primus` copy baked into the image is not used. It also forwards a list of environment variables from the host (including `HF_TOKEN`, the gfx942 tuning variables, and the `NCCL_*` networking variables), so exports you make on the host take effect inside. The forwarded list is `container.options.env` in `runner/.primus.yaml`.

> Only the Primus tree is mounted automatically. Mount datasets, checkpoints, and output directories with `--volume /host/path` (or `--volume /host/path:/container/path`).

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

**Megatron-LM and TorchTitan** on MI300X/MI325X (gfx942) benefit from the following settings. They are not needed on MI355X/MI350X (gfx950):

```bash
# MI300X / MI325X only -- improves performance
export HSA_NO_SCRATCH_RECLAIM=1
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1
export NVTE_CK_IS_V3_ATOMIC_FP32=1
```

**JAX MaxText does not use these.** Its backend adapter applies the correct architecture environment automatically, and the one variable you may need to export yourself is `RCCL_WARP_SPEED_AUTO=0` on MI355X. See [JAX MaxText → Architecture-specific settings](jax-maxtext-training.md#architecture-specific-settings).

### Choosing the Docker image

For **container** and **Slurm** modes (direct mode runs in whatever environment you launched it from), the default image is `rocm/primus:v26.5`, set in `runner/.primus.yaml`. JAX MaxText has its own separate image family, `rocm/jax-training:maxtext-…`, which is **not** the default — pass it explicitly in container and Slurm modes.

The image is picked in priority order: `DOCKER_IMAGE` environment variable > `--image` CLI argument > config file. See [Selecting the container image](../01-getting-started/quickstart.md#selecting-the-container-image) for a full explanation, and [Configuration system](configuration-system.md) for configuration loading.

---

## Shared setup for all backends

These apply across all backends. Set them up before running the recipes below.

### Get the Primus source

Clone the branch matching your image, on the host. Every command on this page runs from this directory:

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout release/v26.5
git submodule update --init --recursive
```

Container mode mounts this checkout into the container, so this is the code that runs — you do not need the `/workspace/Primus` copy baked into the image, which lags the release branch. See [Release notes → Primus source for v26.5](../01-getting-started/release-notes.md#primus-source-for-v265).

### Pull the image (optional)

`primus-cli container` pulls the image on first use, so this is only needed if you want to warm the cache:

```bash
docker pull rocm/primus:v26.5                  # Megatron-LM, TorchTitan, Megatron Bridge
docker pull rocm/jax-training:maxtext-v26.5    # JAX MaxText
```

<details>
<summary>Starting a container by hand instead</summary>

If you want an interactive shell in the container — for debugging, or to run `primus-cli direct` yourself — start one manually and bind your Primus checkout:

```bash
docker run -it \
    --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --privileged \
    -v $PWD:$PWD -w $PWD --shm-size 128G \
    --name primus_training_env \
    rocm/primus:v26.5
```

Re-enter it later with `docker start primus_training_env && docker exec -it primus_training_env bash`. Inside, use `primus-cli direct`. Remember to re-export `HF_TOKEN` and any architecture or `NCCL_*` variables, since a manual `docker run` does not forward them.

Bind only the directories you need rather than your whole home directory.

</details>

### Hugging Face token (for gated models or real data)

```bash
export HF_TOKEN=<your_hf_token>
```

`runner/.primus.yaml` forwards `HF_TOKEN` into the container automatically. MaxText configurations might also read `${HF_TOKEN:""}` directly.

### Mock vs. real data

- **Mock/synthetic data** (default for most examples): validates the stack without datasets. Megatron and TorchTitan set `mock_data: true`; MaxText sets `dataset_type: "synthetic"`.
- **Real data:** set `mock_data: false` and point `train_data_path` (for Megatron) or the backend's dataset fields at paths visible *inside* your container mounts.

### Multi-node networking checklist

The `primus-cli` launcher sets sensible `NCCL_*` defaults, but auto-detection can pick the wrong device on multi-NIC nodes. Before multi-node, confirm and export if needed:

```bash
export NCCL_IB_HCA=<your_rdma_interfaces>      # from `ibv_devices`
export NCCL_SOCKET_IFNAME=<your_net_interface> # from `ip a`
export GLOO_SOCKET_IFNAME=<same_as_NCCL_SOCKET_IFNAME>
export NCCL_IB_GID_INDEX=3                     # 3 for RoCE (1 for AMD AINIC)
```

For AMD AINIC clusters also set `USING_AINIC=1`, `NCCL_PXN_DISABLE=0`, `NCCL_IB_GID_INDEX=1`. See [Multi-node networking](../04-technical-guides/multi-node-networking.md) for the full reference.

---

## Megatron-LM

**Configurations:** `examples/megatron/configs/<ARCH>/`  |  **Precisions:** BF16, FP8 (all architectures); MXFP8 and MXFP4 for Llama 3.1 8B on MI355X

**➜ Full recipe with every model and precision: [Megatron-LM training](megatron-lm-training.md)**

Pretrain Llama 3.1 8B BF16 on **MI355X / MI350X**, from your Primus checkout on the host:

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
```

The same model on **MI300X / MI325X** — export the gfx942 performance variables first, and container mode forwards them into the container:

```bash
export HSA_NO_SCRATCH_RECLAIM=1
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1
export NVTE_CK_IS_V3_ATOMIC_FP32=1

./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

If you already have a shell inside the container, swap `container` for `direct` in any of these commands.

Switch model or precision by changing the config filename (for example `llama3.1_70B-FP8-pretrain.yaml`, `mixtral_8x7B_v0.1-BF16-pretrain.yaml`). See the parallelism table in [Pretraining](pretraining.md#example-configurations-under-examplesmegatronconfigsmi300x).

Multi-node with Slurm:

```bash
./runner/primus-cli slurm srun -N 8 -p <partition> -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml \
  --micro_batch_size 4 --global_batch_size 1024
```

Scale batch size with node count and align `tensor_model_parallel_size`, `pipeline_model_parallel_size`, and `expert_model_parallel_size` to your topology. The [Megatron-LM recipe](megatron-lm-training.md#32-multi-node-training) lists per-model, per-node-count batch sizes.

**Model-specific notes:**

- **Zebra hybrid** (hybrid Mamba+MLA) pretrain presets ship at `examples/megatron/configs/<ARCH>/zebra_mamba_1B_hybrid-pretrain.yaml` (and `_3B`, `_8B`), and run via the legacy runtime — prefix the command with `PRIMUS_TRAIN_RUNTIME=legacy`. Megatron Bridge SFT variants live under `examples/megatron_bridge/configs/<ARCH>/`.
- **MoE models** (DeepSeek-V2-Lite, Mixtral, Qwen3-A3B, GPT-OSS) may need extra grouped-GEMM or router flags; the [Megatron-LM recipe](megatron-lm-training.md#31-single-node-training) gives the exact command per model.

---

## TorchTitan (PyTorch)

**Configurations:** `examples/torchtitan/configs/<ARCH>/`  |  **Precisions:** BF16, FP8

**➜ Full recipe with every model and precision: [TorchTitan training](torchtitan-training.md)**

Uses the same `rocm/primus` container as Megatron-LM. TorchTitan parameters use a dotted namespace (for example `--training.local_batch_size`).

Pretrain Llama 3.1 8B BF16 on **MI355X / MI350X**, from your Primus checkout on the host:

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
```

On **MI300X / MI325X**, export the gfx942 performance variables first and use the `MI300X` config path.

Multi-node with Slurm:

```bash
./runner/primus-cli slurm srun -N 4 -- train pretrain \
  --config examples/torchtitan/configs/MI355X/llama3.1_70B-FP8-pretrain.yaml \
  --training.local_batch_size 6 \
  --training.global_batch_size 192 \
  --training.mock_data True
```

Available models include Llama 3.1 (8B/70B/405B), Llama 4 (17Bx16E/17Bx128E), DeepSeek V3 (16B/236B/671B), Qwen 3 (0.6B–32B), and GPT-OSS (20B/120B). See `examples/torchtitan/configs/<ARCH>/`.

---

## JAX MaxText

**Configurations:** `examples/maxtext/configs/<ARCH>/`  |  **Precisions:** BF16

**➜ Full recipe with every model and precision: [JAX MaxText training](jax-maxtext-training.md)**

MaxText uses a different Docker image than the PyTorch backends and it is **not** the default in `runner/.primus.yaml`, so pass it explicitly with `--image` in container and Slurm modes.

> On MI355X, export `RCCL_WARP_SPEED_AUTO=0` before launching or training can produce NaN losses. It is a no-op on MI300X. See [Important notes](jax-maxtext-training.md#important-notes-for-v265).

Pretrain Llama 3 8B on **MI355X**, from your Primus checkout on the host:

```bash
export RCCL_WARP_SPEED_AUTO=0
./runner/primus-cli container --image rocm/jax-training:maxtext-v26.5 \
  -- train pretrain \
  --config examples/maxtext/configs/MI355X/llama3_8B-pretrain.yaml
```

If you already have a shell inside the MaxText container, use `direct` instead — no `--image` needed:

```bash
export RCCL_WARP_SPEED_AUTO=0
./runner/primus-cli direct \
  -- train pretrain \
  --config examples/maxtext/configs/MI355X/llama3_8B-pretrain.yaml
```

Slurm mode — supply the image (and any environment variables) via a config file:

```bash
./runner/primus-cli --config my_maxtext_config.yaml slurm srun -N 8 \
  -- train pretrain \
  --config examples/maxtext/configs/MI300X/llama3_8B-pretrain.yaml
```

MaxText parallelism is set with `ici_*` (intra-node) and `dcn_*` (inter-node) fields — see the [MaxText config table](pretraining.md#maxtext-jax-pretraining) and [MaxText parameters](../03-configuration-reference/maxtext-parameters.md).

> **Quantized MaxText runs.** The `examples/maxtext/configs/` YAMLs are BF16 only, so there is no FP8 config to select by path. The image does support FP8 (gfx950) and NANOO FP8 (gfx942) — reach them through the `-q fp8` / `-q nanoo_fp8` flags of the standalone benchmark scripts, described in [JAX MaxText → Standalone benchmarking](jax-maxtext-training.md#standalone-benchmarking).

---

## Megatron Bridge (post-training)

Megatron Bridge configurations are under `examples/megatron_bridge/configs/<ARCH>/` and are primarily **SFT and LoRA post-training** recipes (for example `qwen3_32b_sft_posttrain.yaml`, `llama31_70b_lora_posttrain.yaml`). Launch with `train posttrain`:

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_qwen3_32b_sft.log \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI355X/qwen3_32b_sft_posttrain.yaml
```

See [Post-training](posttraining.md) for the full SFT/LoRA workflow.

---

## Related documentation

- [Megatron-LM training](megatron-lm-training.md), [TorchTitan training](torchtitan-training.md), [JAX MaxText training](jax-maxtext-training.md): full per-model recipes.
- [Release notes](../01-getting-started/release-notes.md): what is inside each published image tag.
- [Pretraining](pretraining.md): backend concepts, configuration walkthroughs, parallelism vocabulary, full configuration inventories.
- [Post-training](posttraining.md): SFT and LoRA via Megatron Bridge.
- [CLI reference](cli-reference.md): `direct` / `container` / `slurm` modes and flags.
- [Configuration system](configuration-system.md): YAML inheritance, overrides, image/env precedence.
- [Performance tuning](../04-technical-guides/performance-tuning.md): HipBLASLt autotuning, Primus-Turbo, FP8, MoE.
