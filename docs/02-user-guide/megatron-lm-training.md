# Training a model with Primus and Megatron-LM

Training performance validation of the Primus Docker image with the Megatron backend on AMD Instinct accelerators.

## Overview

The Primus framework with the Megatron backend is designed to enable efficient training of large-scale language models on AMD GPUs. By leveraging AMD Instinct™ MI300X/MI350X accelerators, the Primus Megatron framework delivers enhanced scalability, performance, and resource utilization for AI workloads. It is purpose-built to support models like Llama 2, Llama 3/3.1, DeepSeek V2/V3, and Mixtral MoE, enabling developers to train next-generation AI models with greater efficiency. See the GitHub repository at [AMD-AGI/Primus](https://github.com/AMD-AGI/Primus).

The ROCm PyTorch training Docker image `rocm/primus:v26.9`, available through [Docker hub](https://hub.docker.com/r/rocm/primus/tags), provides a prebuilt, optimized environment for pre-training a model on the AMD Instinct™ MI300X, MI325X, MI350X, and MI355X accelerators.

For the full software stack of this image (ROCm, PyTorch, Transformer Engine, Flash Attention, hipBLASLt, Triton, RCCL, and the rest), see [Release notes → `rocm/primus:v26.9`](../01-getting-started/release-notes.md#rocmprimusv269). The release notes are the single source of truth for image contents, and also cover the previous [`rocm/primus:v26.8`](../01-getting-started/release-notes.md#rocmprimusv268).

Training is launched with `primus-cli`, the unified Primus CLI that covers direct, container, and Slurm execution from the same YAML configuration. See the [CLI reference](./cli-reference.md).

---

## Important notes for v26.9

Read this section before starting a training run. It collects the settings this release requires, the architecture-specific tuning, and the known issues. The contents change from release to release, so re-read it when you move to a new image tag.

### Required settings

**Use the `release/v26.9` branch.** It is the Primus branch matching the `rocm/primus:v26.9` image. The image was built from `db1a4160` on that branch, and the `/workspace/Primus` copy baked into it is that commit. The branch has moved since: it is now at `93dd9201`, which adds MI325X GPT-OSS-20B and ZAYA1-8B recipe tuning ([#1355](https://github.com/AMD-AGI/Primus/pull/1355)), so use the checkout rather than the in-image copy. See [Release notes → Primus source for v26.9](../01-getting-started/release-notes.md#primus-source-for-v269). [Environment setup](#1-environment-setup) has the clone command.

### Changes you may need to act on

**Mock-data runs use the model's real vocabulary.** With `mock_data: true`, Primus used to size the vocabulary at 131072 for every model. It now takes `vocab_size` from the model preset (every preset that names a tokenizer sets it) or from the tokenizer's Hugging Face config, and falls back to 131072 with a warning when neither is available; an explicit `vocab_size` in your config is still kept ([#1249](https://github.com/AMD-AGI/Primus/pull/1249)). Embedding, output-layer and logits memory now match the real model, so mock-data throughput and memory are not comparable with earlier releases: models with a smaller vocabulary (Llama 2, Mixtral) use less memory, and models with a larger one (Qwen, GPT-OSS) use more.

**Turbo fusions are switched on by config keys, not environment variables.** `PRIMUS_FUSED_QK_RMSNORM_ROPE` is replaced by `use_turbo_fused_qk_rmsnorm_rope` ([#1247](https://github.com/AMD-AGI/Primus/pull/1247)), and `PRIMUS_TURBO_CROSS_ENTROPY` / `PRIMUS_TURBO_CE_OVERWRITE_INPUT` by `use_turbo_cross_entropy` / `turbo_ce_overwrite_input` ([#1245](https://github.com/AMD-AGI/Primus/pull/1245)). The MXFP4 MoE permute-quantization fusions are enabled with `moe_permute_quant_fusion`, `moe_backward_permute_quant_fusion` and `moe_skip_identity_sort`, and no longer read environment variables ([#1241](https://github.com/AMD-AGI/Primus/pull/1241)). All of these default to `false` in `primus_turbo.yaml`; the old environment variables have no effect, so move them into your experiment YAML.

**Two MI355X recipes changed model or batch settings.** `deepseek_v2_lite-BF16-pretrain.yaml` now enables multi-latent attention (`multi_latent_attention: true`), the model's real attention, and `gpt_oss_20B-FP8-pretrain.yaml` raises `global_batch_size` from 80 to 96 together with its micro-batch size ([#1252](https://github.com/AMD-AGI/Primus/pull/1252)). Results from these recipes are not directly comparable with v26.8.

**A preset `HIP_VISIBLE_DEVICES` is kept.** The runner used to overwrite it with `0..GPUS_PER_NODE-1`, so a job restricted to particular GPUs could land on another user's GPU on a shared node. It now keeps the value you or your scheduler set, and warns when it lists a different number of GPUs than `GPUS_PER_NODE` ([#1237](https://github.com/AMD-AGI/Primus/pull/1237)).

**Megatron model studies and guides moved under `examples/megatron/`.** For example, `examples/deepseek-v4/` is now `examples/megatron/models/deepseek-v4/`, and `examples/moe_package/` is now `examples/megatron/guides/moe_package/`; `customer_package`, `odc`, `offline_tune` and the tuning agent moved to `examples/megatron/guides/` as well ([#1130](https://github.com/AMD-AGI/Primus/pull/1130)). Update scripts that call the old paths. Recipe YAMLs under `examples/megatron/configs/` did not move.

**The `examples/run_*` launchers are gone.** Launch with `./runner/primus-cli`, or use `runner/helpers/launch/slurm_pretrain.sh` if you rely on the `EXP` / `NNODES` / `DATA_PATH` contract ([#999](https://github.com/AMD-AGI/Primus/pull/999)).

### Architecture-specific settings

**MI300X and MI325X (gfx942) — enable the fp32 atomic paths.** Export these before launching for best performance on gfx942. They are not needed on MI350X/MI355X (gfx950):

```bash
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_IS_V3_ATOMIC_FP32=1
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1
```

**MI300X and MI325X (gfx942) — an expandable allocator for large models.** On gfx942, large-model runs can hit an out-of-memory error caused by allocator *fragmentation* rather than by genuinely running out of HBM. An expandable allocator lets PyTorch reuse memory that is reserved but unallocated:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The MI325X Llama 3.1 / 3.3 70B BF16 and ZAYA1-8B recipes set it in their `env:` block (`PYTORCH_HIP_ALLOC_CONF`) ([#1318](https://github.com/AMD-AGI/Primus/pull/1318)). The MI300X Llama 3.1 / 3.3 70B recipes leave it commented out and are sized to fit without it, because expandable segments cannot run on hosts with older amdgpu drivers ([#1332](https://github.com/AMD-AGI/Primus/pull/1332)) — see the next item. Only enable it on a host whose amdgpu driver supports it.

A launch hook now checks the setting whenever `expandable_segments:True` is requested, from the environment or from the recipe's `env:` block: the launch fails with an error, rather than silently falling back or hanging, when PyTorch does not apply it or when a first expandable allocation does not finish ([#1257](https://github.com/AMD-AGI/Primus/pull/1257), [#1302](https://github.com/AMD-AGI/Primus/pull/1302)). Primus also warns when `use_turbo_grouped_gemm` is enabled without expandable segments, because Turbo grouped GEMM allocates a few very large gradient buffers per layer that raise peak memory on a fragmented pool ([#1257](https://github.com/AMD-AGI/Primus/pull/1257)).

> **This one is not forwarded by a host `export`.** Container mode auto-forwards only `PRIMUS_*`, `NCCL_*`, `RCCL_*`, `GLOO_*`, `IONIC_*`, and `HIPBLASLT_*`, and `PYTORCH_CUDA_ALLOC_CONF` is not in the `container.options.env` allowlist in `runner/.primus.yaml`. Exporting it on the host has no effect — pass it explicitly instead.

Per run:

```bash
./runner/primus-cli container \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_70B-FP8-pretrain.yaml
```

For every run, add it to `container.options.env` in `runner/.primus.yaml`:

```yaml
    env:
      # Use an expandable allocator so large-model runs (for example 70B FP8 on
      # 8 GPUs) reuse "reserved but unallocated" memory instead of OOMing on
      # fragmentation. Explicit value so it applies without a host export.
      - "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
```

In `direct` mode inside a container, a plain `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` works, because there is no container boundary to cross. You can also set it in the recipe's top-level `env:` block, as the MI325X recipes do.

**Hosts with an amdgpu driver older than DRM 3.64 — keep HIP memory pools off virtual memory.** The ROCm 10.x userspace waits after every virtual-memory map on a signal that these drivers never send ([ROCm/rocm-systems#9820](https://github.com/ROCm/rocm-systems/issues/9820)). HIP memory pools map memory that way by default, and Primus Turbo's MoE permute allocates from them, so every MoE model hung on its first step. `primus-cli` now detects such a host and sets `HIP_MEM_POOL_USE_VM=0` for you, and keeps a value you set yourself ([#1302](https://github.com/AMD-AGI/Primus/pull/1302)). If you launch training outside `primus-cli` on such a host, export it yourself:

```bash
export HIP_MEM_POOL_USE_VM=0
```

Expandable segments hang on the same hosts, which is why the launch hook above stops the run instead.

### Known issues

The v26.9 QA runs found no Megatron-LM issues. They covered MI300X and MI355X; the MI325X recipes were not re-run on this image.

**Fixed in v26.9:**

- The DeepEP MoE recipes no longer set `PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE`. The v26.8 recipes forced a DeepEP fence that could let a rank read stale dispatch data; Primus-Turbo repaired that fence and made it the default, so the setting is no longer needed ([#1236](https://github.com/AMD-AGI/Primus/pull/1236)). If you copied it into your own recipe, remove it.
- The Turbo fused grouped MLP (`turbo_fused_grouped_gemm`) skipped the DDP parameter-gather hooks under `overlap_param_gather`, so whenever expert data parallelism exceeded one (multi-node, or expert parallelism smaller than the GPUs per node) it trained on partly stale expert weights, and the loss drifted without an error. It now waits for the gather ([#1252](https://github.com/AMD-AGI/Primus/pull/1252)).
- With DeepEP and `moe_router_force_load_balancing_type: even` (the default), most routed slots got zero weight, so the MoE output collapsed towards the shared expert. The router now applies the even remap itself, so the DeepEP and MegaMoE paths route identically ([#1243](https://github.com/AMD-AGI/Primus/pull/1243)).
- The Turbo FP8 weight cache is stored in the padded layout the op expects ([#1333](https://github.com/AMD-AGI/Primus/pull/1333)).
- DeepSeek-V4 hyper-connections (mHC) contracted the mixing matrix on the wrong axis, so the residual stream drifted from the inference kernels with depth. It now matches SGLang ([#1040](https://github.com/AMD-AGI/Primus/pull/1040)).

### Registry change

The `rocm/pytorch-training` Docker Hub registry is deprecated. Use `rocm/primus` for the latest ROCm PyTorch training images, which cover all the PyTorch training ecosystem frameworks (TorchTitan, TorchTune, Megatron-LM, and others).

---

## Supported features and models

The Primus Megatron backend provides the following key features to train large language models efficiently:

- Primus Turbo with optimized attention and grouped GEMM kernels
- Transformer Engine (TE)
- APEX
- GEMM tuning
- `torch.compile`
- Flash Attention (FA) 3
- AITER Attention
- Fused kernels
- Pre-training
- FP8 GEMM
- Multi-node support
- 3D parallelism: TP + SP + CP
- Distributed optimizer

The following models are pre-optimized for performance on the AMD Instinct MI300X accelerator:

- Llama 2 7B
- Llama 2 70B
- Llama 3/3.1 8B
- Llama 3/3.1/3.3 70B
- DeepSeek-V2-lite
- DeepSeek-V3
- DeepSeek-V4 (BF16 SFT, and packed-sequence THD SFT at 4k/128k — see [`examples/megatron/models/deepseek-v4`](https://github.com/AMD-AGI/Primus/tree/main/examples/megatron/models/deepseek-v4); gfx942 recipes added in v26.7)
- Mixtral 8x7B
- Mixtral 8x22B
- Qwen 2.5 7B/72B
- Zebra hybrid 1B/3B/8B
- Qwen3-30B-A3B
- Qwen3-235B-A22B
- Qwen3 32B (SFT / LoRA)
- GPT-OSS-20B
- GPT-OSS-120B
- ZAYA1-8B (BF16 and FP8 mock-data recipes on MI300X, MI325X and MI355X; FP8 and MI325X added in v26.9)
- GLM-5.3-Flash (new in v26.9; MI355X BF16 bring-up recipe — see the [model support matrix](../06-developer-guide/model-support-matrix.md#megatron-model-configs) for its limits)

---

## System validation steps

If you have already validated your system, skip this step. Otherwise, complete the [system validation and optimization steps](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/system-setup/prerequisite-system-validation.html#prerequisite-system-validation-before-running-ai-workloads) to set up your system before starting training.

### Disable NUMA auto-balancing

Generally, application performance can benefit from disabling NUMA auto-balancing. However, it might be detrimental to performance with certain types of workloads.

Run the command `cat /proc/sys/kernel/numa_balancing` to check your current NUMA (Non-Uniform Memory Access) settings. Output `0` indicates this setting is disabled. If there is no output or the output is `1`, run the following command to disable NUMA auto-balancing.

```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```

See [Disable NUMA auto-balancing](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#mi300x-disable-numa) for more information.

### Start training on AMD Instinct accelerators

The pre-built ROCm Primus Megatron backend environment allows you to quickly validate system performance, conduct training benchmarks, and achieve superior performance for models like Llama 2 and Llama 3.1. The Docker image is powered by Primus Turbo optimizations to achieve optimal performance.

This container should not be expected to provide generalized performance across all training workloads. Expect the container to perform in the model configurations described below; other configurations and run conditions are not validated by AMD.

Use the following instructions to set up the environment, configure the script to train models, and reproduce the benchmark results on the MI300X accelerators with the AMD Megatron-LM Docker image.

---

## 1. Environment setup

**Clone the Primus branch matching the image.** Do this on the host — every command in this guide runs from this directory:

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout release/v26.9
git submodule update --init --recursive
```

That is all the setup required. The training commands below use `primus-cli container`, which starts `rocm/primus:v26.9` for you, mounts this checkout into it at the same path, and runs the training inside. You do not need to `docker run` or `docker exec` by hand, and the `/workspace/Primus` copy baked into the image is not used — see [Release notes → Primus source for v26.9](../01-getting-started/release-notes.md#primus-source-for-v269).

Container mode also forwards environment variables you export on the host, including `HF_TOKEN`, the gfx942 tuning variables, and the `NCCL_*` networking variables. The forwarded list is `container.options.env` in `runner/.primus.yaml`.

> Only the Primus tree is mounted automatically. Mount datasets, checkpoints, and output directories with `--volume /host/path`.

<details>
<summary>Starting a container by hand instead</summary>

If you want an interactive shell — for debugging, or to run `primus-cli direct` yourself — start the container manually and bind your Primus checkout:

```bash
docker pull rocm/primus:v26.9
docker run -it --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --network host --ipc host --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --privileged \
    -v $PWD:$PWD -w $PWD --shm-size 128G \
    --name primus_training_env rocm/primus:v26.9
```

Re-enter it later with `docker start primus_training_env && docker exec -it primus_training_env bash`. Inside the container, replace `primus-cli container` with `primus-cli direct` in every command below. Remember to re-export `HF_TOKEN` and any architecture or `NCCL_*` variables, since a manual `docker run` does not forward them.

Bind only the directories you need rather than your whole home directory.

</details>

---

## 2. Configurations in YAML files (`examples/megatron/configs/`)

Primus defines a training YAML for each model inside [`examples/megatron/configs/`](https://github.com/AMD-AGI/Primus/tree/main/examples/megatron/configs). For example, use `examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml` to update Llama 3.1 8B training parameters. Other YAML files for the supported models follow the `examples/megatron/configs/<ARCH>/${MODEL_NAME}-<PRECISION>-pretrain.yaml` naming convention, where `<ARCH>` is `MI300X`, `MI325X`, or `MI355X`.

You can toggle various training parameters such as `micro_batch_size`, `global_batch_size`, `train_iters`, and others inside the pretraining YAML files.

> **Note:**
>
> - Supported model definitions can be found inside [`primus/configs/models/megatron/`](https://github.com/AMD-AGI/Primus/tree/main/primus/configs/models/megatron).
> - To migrate an existing workload from ROCm/Megatron-LM to Primus, or to add a new workload, follow the [Migration Guide](https://github.com/ROCm/MAD/blob/develop/benchmark/megatron_lm/Migration_Guide.md).

### 2.1 Dataset

You can use either mock data or real data for training.

- **Mock data:** the pretraining YAML scripts use `mock_data: true` by default.
- **Real data:** to use real data for training, set the variable `train_data_path` to your tokenized data path and set `mock_data: false`.

### 2.2 Tokenizer

In Primus, each model uses a tokenizer from Hugging Face. For example, the Llama 3.1 8B model uses `tokenizer_model: meta-llama/Llama-3.1-8B` and `tokenizer_type: Llama3Tokenizer`, defined in the [Llama 3.1 8B model configuration](https://github.com/AMD-AGI/Primus/blob/main/examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml). Use an `HF_TOKEN` with the right permissions to access the tokenizer for each model.

```bash
# Export your HF_TOKEN in the workspace
export HF_TOKEN=<your_hftoken>
```

---

## 3. How to run

### 3.1 Single-node training

To run model training on a single node, run the commands below from your `release/v26.9` Primus checkout on the host (recommended). When using `./runner/primus-cli container`, no additional `pip install` step is required.

#### MI300X performance configs

On MI300X/MI325X, export the gfx942 tuning variables from [Architecture-specific settings](#architecture-specific-settings) before running any of the commands below.

- **Llama3.1-8B FP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_8B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml
```

- **Llama3.1-8B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

- **Llama2-7B FP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama2_7B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-FP8-pretrain.yaml
```

- **Llama2-7B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama2_7B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
```

- **Llama3.1-70B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_70B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.1_70B-BF16-pretrain.yaml
```

- **Llama2-70B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama2_70B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_70B-BF16-pretrain.yaml
```

- **Llama3.3-70B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.3_70B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama3.3_70B-BF16-pretrain.yaml
```

Examples for MoE models with expert parallelism enabled (that is, `expert_model_parallel_size > 1`):

- **DeepSeekV2-Lite BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_deepseek_v2_lite.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml
```

- **Mixtral 8x7B:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_mixtral_8x7B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/mixtral_8x7B_v0.1-BF16-pretrain.yaml
```

- **Qwen2.5 7B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_qwen2.5_7B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/qwen2.5_7B-BF16-pretrain.yaml
```

- **Qwen2.5 7B FP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_qwen2.5_7B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/qwen2.5_7B-FP8-pretrain.yaml
```

- **Qwen2.5 72B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_qwen2.5_72B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/qwen2.5_72B-BF16-pretrain.yaml
```

- **Zebra hybrid-1B BF16:**

```bash
PRIMUS_TRAIN_RUNTIME=legacy ./runner/primus-cli container \
  --log_file /tmp/primus_zebra_mamba_1B_hybrid.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/zebra_llama_mamba_1B_BF16-pretrain.yaml
```

- **Qwen3-32B BF16 LoRA:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_qwen3_32b.log \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/qwen3_32b_lora_posttrain.yaml
```

- **Qwen3-32B BF16 SFT:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_qwen3_32b_sft.log \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI300X/qwen3_32b_sft_posttrain.yaml
```

- **Qwen3-30B (A3B) BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_qwen3_30B_A3B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/qwen3_30B_A3B-BF16-pretrain.yaml
```

- **Qwen3-30B (A3B) FP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_qwen3_30B_A3B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/qwen3_30B_A3B-FP8-pretrain.yaml
```

- **GPT-OSS-20B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_gpt_oss_20B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/gpt_oss_20B-BF16-pretrain.yaml
```

- **GPT-OSS-20B FP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_gpt_oss_20B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/gpt_oss_20B-FP8-pretrain.yaml
```

#### MI35X performance configs

- **Llama3.1-8B FP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_8B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.1_8B-FP8-pretrain.yaml
```

- **Llama3.1-8B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
```

- **Llama3.1-8B MXFP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_8B_mxfp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.1_8B-MXFP8-pretrain.yaml
```

- **Llama3.1-8B MXFP4:**

```bash
NVTE_USE_CAST_TRANSPOSE_TRITON=0 ./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_8B_mxfp4.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.1_8B-MXFP4-pretrain.yaml
```

- **Llama2-7B FP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama2_7B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama2_7B-FP8-pretrain.yaml
```

- **Llama2-7B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama2_7B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama2_7B-BF16-pretrain.yaml
```

- **Llama3.1-70B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_70B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.1_70B-BF16-pretrain.yaml
```

- **Llama3.1-70B FP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_70B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.1_70B-FP8-pretrain.yaml
```

- **Llama2-70B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama2_70B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama2_70B-BF16-pretrain.yaml
```

- **Llama3.3-70B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.3_70B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/llama3.3_70B-BF16-pretrain.yaml
```

- **DeepSeekV2-Lite BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_deepseek_v2_lite.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/deepseek_v2_lite-BF16-pretrain.yaml
```

- **Mixtral 8x7B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_mixtral_8x7B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/mixtral_8x7B_v0.1-BF16-pretrain.yaml
```

- **Qwen2.5 7B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_qwen2.5_7B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/qwen2.5_7B-BF16-pretrain.yaml
```

- **Qwen2.5 7B FP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_qwen2.5_7B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/qwen2.5_7B-FP8-pretrain.yaml
```

- **Qwen2.5 72B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_qwen2.5_72B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/qwen2.5_72B-BF16-pretrain.yaml
```

- **Zebra hybrid-1B BF16:**

```bash
PRIMUS_TRAIN_RUNTIME=legacy ./runner/primus-cli container \
  --log_file /tmp/primus_zebra_mamba_1B_hybrid.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/zebra_llama_mamba_1B_BF16-pretrain.yaml
```

- **Qwen3-32B BF16 LoRA:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_qwen3_32b_lora.log \
  -- train posttrain \
  --config examples/megatron_bridge/configs/MI355X/qwen3_32b_lora_posttrain.yaml
```

- **Qwen3-30B (A3B) BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_qwen3_30B_A3B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-pretrain.yaml
```

- **Qwen3-30B (A3B) FP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_qwen3_30B_A3B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/qwen3_30B_A3B-FP8-pretrain.yaml
```

- **GPT-OSS-20B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_gpt_oss_20B.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/gpt_oss_20B-BF16-pretrain.yaml
```

- **GPT-OSS-20B FP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_gpt_oss_20B_fp8.log \
  -- train pretrain \
  --config examples/megatron/configs/MI355X/gpt_oss_20B-FP8-pretrain.yaml
```

### 3.2 Multi-node training

To run training on multiple nodes, you can use `primus-cli` (recommended) or the shared [`runner/helpers/launch/slurm_pretrain.sh`](../../runner/helpers/launch/slurm_pretrain.sh) helper, which turns the `EXP` / `NNODES` environment contract into the same `primus-cli slurm srun ... -- container -- train pretrain` invocation. Below are the multi-node setup and examples to run multi-node tests.

**Multi-node setup**

> **Verify NCCL / network env first.** The `primus-cli` launcher script sets sensible `NCCL_*` defaults via `base_env.sh`, but auto-detection can pick the wrong device on multi-NIC nodes. Always confirm `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `NCCL_SOCKET_IFNAME`, and `GLOO_SOCKET_IFNAME` (set to the same value as `NCCL_SOCKET_IFNAME`) are correct for your fabric. If necessary, you can `export` these environment variables before running.

From your `release/v26.9` checkout (see [Environment setup](#1-environment-setup)), export the cluster settings:

```bash
export DOCKER_IMAGE=rocm/primus:v26.9
export HF_TOKEN=<your_HF_token>
export NCCL_IB_HCA=<your_NCCL_IB_HCA> # specify which RDMA interfaces to use for communication
export NCCL_SOCKET_IFNAME=<your_NCCL_SOCKET_IFNAME> # your network interface
export GLOO_SOCKET_IFNAME=<your_GLOO_SOCKET_IFNAME> # your network interface
export NCCL_IB_GID_INDEX=3 # Set InfiniBand GID index for NCCL communication. Default is 3 for RoCE

# On MI300X/MI325X also export the gfx942 tuning variables; see "Architecture-specific settings"
```

> **Note:** `release/v26.9` is the branch matching the `rocm/primus:v26.9` image. If you are reproducing published v26.4 numbers instead, use `git checkout 236cfa9` with `rocm/primus:v26.4` — that release is now summarised under [Release notes → Earlier releases](../01-getting-started/release-notes.md#earlier-releases).

For clusters using AMD AINIC, set the following environment variables:

```bash
export USING_AINIC=1
export NCCL_PXN_DISABLE=0
export NCCL_IB_GID_INDEX=1
```

Notes:

- Make sure the correct network drivers are installed on the nodes. If inside a Docker container, either install the drivers inside the container or pass the network drivers from the host while creating the container.
- If `NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME` are not set, Primus tries to auto-detect them. However, since NICs can vary across clusters, explicitly export your NCCL parameters for the cluster.
- To find your network interface, use `ip a`.
- To find RDMA interfaces, use `ibv_devices` to get the list of all RDMA/IB devices.

- **Llama3.1-8B FP8, 8 nodes:**

```bash
# Adjust the training parameters. For example, `global_batch_size: 8 * #single_node_bs` for 8 nodes in this case
NNODES=8 EXP=examples/megatron/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml bash ./runner/helpers/launch/slurm_pretrain.sh --global_batch_size 1024
```

- **Llama2-7B FP8, 8 nodes:**

```bash
# Adjust the training parameters. For example, `global_batch_size: 8 * #single_node_bs` for 8 nodes in this case
NNODES=8 EXP=examples/megatron/configs/MI300X/llama2_7B-FP8-pretrain.yaml bash ./runner/helpers/launch/slurm_pretrain.sh --global_batch_size 2048
```

- **Llama3.1-70B FP8, 8 nodes:**

```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/llama3.1_70B-FP8-pretrain.yaml bash ./runner/helpers/launch/slurm_pretrain.sh --micro_batch_size 4 --global_batch_size 256 --recompute_num_layers 80
```

- **Llama3.1-70B BF16, 8 nodes:**

```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/llama3.1_70B-BF16-pretrain.yaml bash ./runner/helpers/launch/slurm_pretrain.sh --micro_batch_size 1 --global_batch_size 256 --recompute_num_layers 12
```

- **Llama2-70B FP8, 8 nodes:**

```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/llama2_70B-FP8-pretrain.yaml bash ./runner/helpers/launch/slurm_pretrain.sh --micro_batch_size 10 --global_batch_size 640 --recompute_num_layers 80
```

- **Llama2-70B BF16, 8 nodes:**

```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/llama2_70B-BF16-pretrain.yaml bash ./runner/helpers/launch/slurm_pretrain.sh --micro_batch_size 2 --global_batch_size 1536 --recompute_num_layers 12
```

- **Llama3.3-70B FP8, 8 nodes:**

```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/llama3.3_70B-FP8-pretrain.yaml bash ./runner/helpers/launch/slurm_pretrain.sh --micro_batch_size 4 --global_batch_size 256 --recompute_num_layers 80
```

- **Llama3.3-70B BF16, 8 nodes:**

```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/llama3.3_70B-BF16-pretrain.yaml bash ./runner/helpers/launch/slurm_pretrain.sh --micro_batch_size 1 --global_batch_size 256 --recompute_num_layers 12
```

- **Mixtral 8x7B BF16, 8 nodes:**

```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/mixtral_8x7B_v0.1-BF16-pretrain.yaml bash ./runner/helpers/launch/slurm_pretrain.sh --micro_batch_size 2 --global_batch_size 256
```

- **Qwen2.5-72B FP8, 8 nodes:**

```bash
NNODES=8 EXP=examples/megatron/configs/MI300X/qwen2.5_72B-FP8-pretrain.yaml bash ./runner/helpers/launch/slurm_pretrain.sh --micro_batch_size 8 --global_batch_size 512 --recompute_num_layers 80
```

- **Mixtral-8x22B BF16, 4 nodes, MI355X**

Launch the training using `primus-cli` (recommended):

```bash
# In the Primus directory
./runner/primus-cli slurm srun -N 4 -- train pretrain --config examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml --micro_batch_size 1 --global_batch_size 512 --num_virtual_stages_per_pipeline_rank 2 --pipeline_model_parallel_size 4 --expert_model_parallel_size 8 --recompute_num_layers 1 --moe_use_legacy_grouped_gemm True --gradient_accumulation_fusion True
```

Or with the shared `EXP` / `NNODES` helper:

```bash
NNODES=4 EXP=examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml bash ./runner/helpers/launch/slurm_pretrain.sh --micro_batch_size 1 --global_batch_size 512 --num_virtual_stages_per_pipeline_rank 2 --pipeline_model_parallel_size 4 --expert_model_parallel_size 8 --recompute_num_layers 1 --moe_use_legacy_grouped_gemm True --gradient_accumulation_fusion True
```

- **Llama3.1-405B FP8, 8 nodes, MI325X**

Launch the training using `primus-cli` (recommended):

```bash
# In the Primus directory
./runner/primus-cli slurm srun -N 8 -- train pretrain --config examples/megatron/configs/MI325X/llama3.1_405B-FP8-pretrain.yaml --micro_batch_size 1 --global_batch_size 256 --decoder_first_pipeline_num_layers 15 --decoder_last_pipeline_num_layers 15
```

We use TP=8 for the Llama 3.1 405B model on 8 nodes. Because it has 126 layers, which is not divisible by 8, you need to set `decoder_first_pipeline_num_layers` and `decoder_last_pipeline_num_layers`.

Or with the shared `EXP` / `NNODES` helper:

```bash
NNODES=8 EXP=examples/megatron/configs/MI325X/llama3.1_405B-FP8-pretrain.yaml bash ./runner/helpers/launch/slurm_pretrain.sh --micro_batch_size 1 --global_batch_size 256 --decoder_first_pipeline_num_layers 15 --decoder_last_pipeline_num_layers 15
```

---

## 4. Key variables to pay attention to

- **fp8:** `--fp8 hybrid` enables FP8 GEMMs.

- **use_torch_fsdp2:** `use_torch_fsdp2: 1` enables Torch FSDP v2.

  Note that if FSDP is enabled, set these variables to false: `use_distributed_optimizer: false` and `overlap_param_gather: false`.

- **profile:** to enable PyTorch profiling, set all of these parameters:

```yaml
profile: true
use_pytorch_profiler: true
profile_step_end: 7
profile_step_start: 6
```

- **train_iters:** set the total number of iterations (default: 50).

- **mock_data:** set to `true` by default.

- **micro_batch_size:** micro batch size.

- **global_batch_size:** global batch size.

- **recompute_granularity:** activation checkpointing (`null`, `sel`, `full`). Default: `null`. When set to `full`, also set `recompute_num_layers` and `recompute_method` (`uniform` or `block`).

- **num_layers:** use a reduced number of layers as a proxy model.

---

## Related documentation

- [End-to-end training recipes](./end-to-end-training-recipes.md)
- [Pretraining workflows](./pretraining.md)
- [Post-training workflows](./posttraining.md)
- [Megatron parameters](../03-configuration-reference/megatron-parameters.md)
- [CLI reference](./cli-reference.md)
- [Multi-node networking](../04-technical-guides/multi-node-networking.md)
