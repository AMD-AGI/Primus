# Training a model with Primus and PyTorch TorchTitan

Training performance validation with the AMD PyTorch Docker image on AMD Instinct accelerators, using Primus with the TorchTitan backend.

## Overview

PyTorch is an open-source machine learning framework that is widely used for model training, with GPU-optimized components for transformer-based models.

The ROCm PyTorch training Docker image `rocm/primus:v26.5`, available through [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html), provides a prebuilt, optimized environment for fine-tuning and pre-training a model on the AMD Instinct™ MI300X and MI325X accelerators.

For the full software stack of this image (ROCm, PyTorch, Transformer Engine, Flash Attention, hipBLASLt, Triton, RCCL, and the rest), see [Release notes → `rocm/primus:v26.5`](../01-getting-started/release-notes.md#rocmprimusv265). The release notes are the single source of truth for image contents, and also cover the previous [`rocm/primus:v26.4`](../01-getting-started/release-notes.md#rocmprimusv264).

Training is launched with `primus-cli`, the unified Primus CLI that covers direct, container, and Slurm execution from the same YAML configuration. See the [CLI reference](./cli-reference.md).

---

## Important notes for v26.5

Read this section before starting a training run. It collects the settings this release requires, the architecture-specific tuning, and the known issues. The contents change from release to release, so re-read it when you move to a new image tag.

### Required settings

**Use the `release/v26.5` branch.** It is the Primus branch matching the `rocm/primus:v26.5` image. The `/workspace/Primus` checkout baked into the image is built from commit `b511d1b6` and the branch has moved on since — see [Release notes → Primus source for v26.5](../01-getting-started/release-notes.md#primus-source-for-v265). [Environment setup](#get-the-primus-source) has the clone command.

### Architecture-specific settings

**MI300X and MI325X (gfx942) — enable the fp32 atomic paths.** Export these before launching for best performance on gfx942. They are not needed on MI350X/MI355X (gfx950):

```bash
export PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1
export NVTE_CK_IS_V3_ATOMIC_FP32=1
```

### Known issues

No TorchTitan backend issues are currently tracked for v26.5.

### Registry change

The `rocm/pytorch-training` Docker Hub registry is deprecated. Use `rocm/primus` for the latest ROCm PyTorch training images, which cover all the PyTorch training ecosystem frameworks (TorchTitan, TorchTune, Megatron-LM, and others).

---

## Models

Examples of the following models are pre-optimized for performance on the AMD Instinct MI300X and MI325X accelerators.

### Pre-training

| Model | Variants |
| ------------- | ------------- |
| **Llama 3.1** | 8B, 70B, 405B |
| **DeepSeek V3** | 16B |

> **Note:** Some models, such as Llama 3, require an external license agreement through a third party (for example, Meta).

---

## System validation steps

If you have already validated your system, skip this step. Otherwise, complete the [system validation and optimization steps](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/prerequisite-system-validation.html) to set up your system before starting training.

### Disable NUMA auto-balancing

Generally, application performance can benefit from disabling NUMA auto-balancing. However, it might be detrimental to performance with certain types of workloads.

Run the command `cat /proc/sys/kernel/numa_balancing` to check your current NUMA (Non-Uniform Memory Access) settings. Output `0` indicates this setting is disabled. If there is no output or the output is `1`, run the following command to disable NUMA auto-balancing.

```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```

See [Disable NUMA auto-balancing](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#mi300x-disable-numa) for more information.

---

## Start training on AMD Instinct accelerators

> **Note:** The only models supported in this workflow are those listed in the section above.

This container should not be expected to provide generalized performance across all training workloads. Expect the container to perform in the model configurations described below; other configurations and run conditions are not validated by AMD.

Use the following instructions to set up the environment, configure the script to train models, and reproduce the benchmark results on the MI300X, MI325X, MI350X, and MI355X accelerators with the Docker image.

The instructions reproduce the benchmark results on an MI300X accelerator with a prebuilt PyTorch Docker image. For best performance on MI325X, MI350X, and MI355X, adjust configurations (for example, batch sizes) accordingly.

There are two ways to run training, listed in the order we recommend:

| Method | Use it when |
| ------ | ----------- |
| [**`primus-cli`**](#running-training-with-primus-cli-recommended) **(recommended)** | Any new work. One CLI covers direct, container, and Slurm launches, and the same YAML configurations work across every Primus backend. |
| [MAD-integrated benchmarking](#mad-integrated-benchmarking) *(legacy)* | You are reproducing published AMD numbers through the ROCm MAD dashboarding pipeline. |

---

## Environment setup

### Get the Primus source

Clone the branch matching the image. Do this on the host — every command in this guide runs from this directory:

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout release/v26.5
git submodule update --init --recursive
```

That is all the setup required. The training commands below use `primus-cli container`, which starts `rocm/primus:v26.5` for you, mounts this checkout into it at the same path, and runs the training inside. You do not need to `docker run` or `docker exec` by hand, and the `/workspace/Primus` copy baked into the image is not used — see [Release notes → Primus source for v26.5](../01-getting-started/release-notes.md#primus-source-for-v265).

Container mode also forwards environment variables you export on the host, including `HF_TOKEN`, the gfx942 tuning variables, and the `NCCL_*` networking variables. The forwarded list is `container.options.env` in `runner/.primus.yaml`.

> Only the Primus tree is mounted automatically. Mount datasets, checkpoints, and output directories with `--volume /host/path`.

<details>
<summary>Starting a container by hand instead</summary>

If you want an interactive shell — for debugging, or to run `primus-cli direct` yourself — start the container manually and bind your Primus checkout:

```bash
docker pull rocm/primus:v26.5
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    -v $PWD:$PWD -w $PWD -v $HOME/.ssh:/root/.ssh \
    --shm-size 64G --name training_env rocm/primus:v26.5
```

Re-enter it later with `docker start training_env && docker exec -it training_env bash`. Inside the container, replace `primus-cli container` with `primus-cli direct` in every command below. Remember to re-export `HF_TOKEN` and any architecture or `NCCL_*` variables, since a manual `docker run` does not forward them.

Bind only the directories you need rather than your whole home directory.

</details>

### Prepare training datasets and dependencies

The following benchmarking examples may require downloading models and datasets from Hugging Face. To ensure successful access to gated repos, set your `HF_TOKEN`:

```bash
# pass your HF_TOKEN
export HF_TOKEN=$your_personal_hf_token
```

---

## Running training with primus-cli (recommended)

For detailed usage of `primus-cli`, see the [CLI reference](./cli-reference.md).

Run these from your `release/v26.5` checkout **on the host**. Container mode starts the image and runs the training inside it for you. If you already have a shell inside the container, swap `container` for `direct`.

### Benchmarking examples

On MI300X/MI325X, export the gfx942 tuning variables from [Architecture-specific settings](#architecture-specific-settings) before running any of the commands below.

#### MI300X performance configs

- **Llama3.1-70B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_70B.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI300X/llama3.1_70B-BF16-pretrain.yaml
```

- **Llama3.1-8B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml
```

- **DeepSeek-V3-16b BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_deepseek_v3_16b.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI300X/deepseek_v3_16b-BF16-pretrain.yaml
```

- **Llama3.1-70B FP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_70B_fp8.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI300X/llama3.1_70B-FP8-pretrain.yaml
```

- **Llama3.1-8B FP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_8B_fp8.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI300X/llama3.1_8B-FP8-pretrain.yaml
```

#### MI35X performance configs

- **Llama3.1-70B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_70B.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI355X/llama3.1_70B-BF16-pretrain.yaml
```

- **Llama3.1-8B BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_8B.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml
```

- **DeepSeek-V3-16b BF16:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_deepseek_v3_16b.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI355X/deepseek_v3_16b-BF16-pretrain.yaml
```

- **Llama3.1-70B FP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_70B_fp8.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI355X/llama3.1_70B-FP8-pretrain.yaml
```

- **Llama3.1-8B FP8:**

```bash
./runner/primus-cli container \
  --log_file /tmp/primus_llama3.1_8B_fp8.log \
  -- train pretrain \
  --config examples/torchtitan/configs/MI355X/llama3.1_8B-FP8-pretrain.yaml
```

### Multi-node training

Multi-node training with TorchTitan is similar to Megatron-LM. See [Megatron-LM multi-node training](./megatron-lm-training.md#32-multi-node-training) for how to set the environment variables.

Here are two examples for multi-node training on MI355X.

- **Llama3.1-70B FP8, 4 nodes, MI355X**

Launch the training using `primus-cli` (recommended):

```bash
# In the Primus directory
./runner/primus-cli slurm srun -N 4 -- train pretrain --config examples/torchtitan/configs/MI355X/llama3.1_70B-FP8-pretrain.yaml --training.local_batch_size 6 --training.global_batch_size 192 --training.mock_data True
```

Launch the training using the legacy script:

```bash
NNODES=4 EXP=examples/torchtitan/configs/MI355X/llama3.1_70B-FP8-pretrain.yaml bash examples/run_slurm_pretrain.sh --training.local_batch_size 6 --training.global_batch_size 192 --training.mock_data True
```

- **Llama3.1-405B FP8, 8 nodes, MI355X**

Launch the training using `primus-cli` (recommended):

```bash
# In the Primus directory
./runner/primus-cli slurm srun -N 8 -- train pretrain --config examples/torchtitan/configs/MI355X/llama3.1_405B-FP8-pretrain.yaml --training.local_batch_size 3 --training.global_batch_size 192 --training.mock_data True
```

Launch the training using the legacy script:

```bash
NNODES=8 EXP=examples/torchtitan/configs/MI355X/llama3.1_405B-FP8-pretrain.yaml bash examples/run_slurm_pretrain.sh --training.local_batch_size 3 --training.global_batch_size 192 --training.mock_data True
```

---

## MAD-integrated benchmarking

> **Legacy path.** MAD-integrated benchmarking is retained for reproducing published AMD numbers through the ROCm MAD dashboarding pipeline. For new work use [`primus-cli`](#running-training-with-primus-cli-recommended) instead.

Clone the ROCm Model Automation and Dashboarding (MAD) repository to a local directory and install the required packages on the host machine.

```sh
git clone https://github.com/ROCm/MAD
cd MAD
pip install -r requirements.txt
```

Use this command to run a performance benchmark test of the Llama 3.1 8B model through Primus on one GPU with the `float16` data type on the host machine.

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
python3 tools/run_models.py --tags primus_pyt_train_llama-3.1-8b --keep-model-dir --live-output --timeout 28800
```

ROCm MAD launches a Docker container with the name `container_ci-primus_pyt_train_llama-3.1-8b`. The latency and throughput reports of the model are collected in the following path:

```sh
~/MAD/perf.csv
```

### Available models

| model_name |
| --------------------------------- |
| `primus_pyt_train_llama-3.1-8b` |
| `primus_pyt_train_llama-3.1-70b` |
| `primus_pyt_train_deepseek-v3-16b` |

To start the pretraining benchmark, use the following command:

```bash
./pytorch_benchmark_report.sh -t $training_mode -m $model_repo -p $datatype
```

---

## Related documentation

- [End-to-end training recipes](./end-to-end-training-recipes.md)
- [Pretraining workflows](./pretraining.md)
- [Megatron-LM training performance validation](./megatron-lm-training.md)
- [TorchTitan parameters](../03-configuration-reference/torchtitan-parameters.md)
- [CLI reference](./cli-reference.md)
- [Multi-node networking](../04-technical-guides/multi-node-networking.md)
