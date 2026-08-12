# Training a model with Primus and JAX MaxText

Training performance validation with the ROCm JAX MaxText training Docker image on AMD Instinct accelerators.

## Overview

MaxText for ROCm is a specialized fork of upstream MaxText, designed to enable training of large language models (LLMs) on AMD GPUs. By leveraging AMD Instinct™ MI300X and MI355X GPUs, MaxText delivers scalability, performance, and resource utilization for AI workloads. See the GitHub repository at [ROCm/maxtext](https://github.com/ROCm/maxtext/).

AMD provides a ready-to-use Docker image for AMD Instinct MI300X and MI355X GPUs containing essential components, including JAX, XLA, ROCm libraries, and MaxText utilities.

For the full software stack of this image (ROCm, JAX, Transformer Engine, hipBLASLt, RCCL, TensorFlow, and the rest), see [Release notes → `rocm/jax-training:maxtext-v26.5`](../01-getting-started/release-notes.md#rocmjax-trainingmaxtext-v265). The release notes are the single source of truth for image contents.

> **Primus source:** use the `release/v26.5` branch rather than the Primus checkout baked into the image — see [Release notes → Primus source for v26.5](../01-getting-started/release-notes.md#primus-source-for-v265) for why.

---

## Important notes for v26.5

Read this section before starting a training run. It collects the settings this release requires, the architecture-specific workarounds, and the known issues. The contents change from release to release, so re-read it when you move to a new image tag.

### Required settings

**Enable Shardy.** Shardy is the partitioning system in JAX. The v26.5 image ships JAX 0.10.0, which requires it, so set `shardy=True` during the training run. You may see partitioning-related errors if it is not configured correctly. See the [Shardy migration guide](https://docs.jax.dev/en/latest/shardy_jax_migration.html) for details.

### Architecture-specific settings

**MI355X (gfx950) — disable RCCL WarpSpeed.** RCCL's WarpSpeed feature (`RCCL_WARP_SPEED_AUTO`) is a gfx950-only optimization that is enabled by default in gfx950 builds, and it can cause **NaN losses** during training. Set the following before launching:

```bash
export RCCL_WARP_SPEED_AUTO=0
```

This variable is a no-op on MI300X (gfx942), so it is safe to export unconditionally. The MAD-integrated path applies it for you through the gfx950 environment scripts under `scripts/jax-maxtext/env_scripts/` (for example, `gfx950_llama3_8b_env.sh`); on every other launch path you must export it yourself.

### Known issues

**Mixtral-8x7B performance regression.** Mixtral-8x7B is slower in v26.5 than in the previous release. This is being tracked and will be addressed in a future release.

**Loss curve discrepancy with `packing=false`.** With `packing=false` the loss converges at a slightly higher value than in previous images. To reproduce the earlier convergence, set `NVTE_CK_USES_FWD_V3=0`, which uses Flash Attention v2 for the forward pass instead of v3. This is being tracked and will be addressed in a future release.

---

## Supported features and models

MaxText supports the following key features to train large language models efficiently:

- Transformer Engine (TE)
- Flash Attention (FA) 3, with or without input sequence packing
- GEMM tuning
- Multi-node support
- NANOO FP8 (for MI300X) or FP8 (for MI355X)

The following models are pre-optimized for performance on the AMD Instinct MI300X and MI355X accelerators:

- Llama 2 7B
- Llama 2 70B
- Llama 3/3.1 8B
- Llama 3/3.1 70B
- Llama 3.1 405B
- Llama 3.3 70B
- DeepSeek-V2-lite (16B)
- Mixtral-8x7B
- Qwen3 14B
- Qwen3 30B-A3B

> **Note:** Some models, such as Llama 3, require an external license agreement through a third party (for example, Meta). The only models supported in this workflow are those listed above.

---

## System validation

If you have already validated your system, skip this step. Otherwise, complete the [system validation and optimization steps](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/prerequisite-system-validation.html#train-a-model-system-validation) to set up your system before starting training.

---

## Environment setup

This Docker image is optimized for specific model configurations outlined below. Performance can vary for other training workloads, as AMD doesn't validate configurations and run conditions outside those described.

This container should not be expected to provide generalized performance across all training workloads. Expect the container to perform in the model configurations described below; other configurations and run conditions are not validated by AMD.

### Multi-node prerequisites (RDMA only)

For multi-node runs, make sure all packages are installed based on the network device you use. You only need the setup below if you are using multi-node with RDMA—otherwise skip this part.

Install the packages below for building and installing the RDMA driver:

```bash
apt install iproute2 -y
apt install -y linux-headers-"$(uname -r)" libelf-dev
apt install -y gcc make libtool autoconf librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool libibverbs-dev rdma-core strace libibmad5 libibnetdisc5 ibverbs-providers libibumad-dev libibumad3 libibverbs1 libnl-3-dev libnl-route-3-dev
```

Refer to your NIC manufacturer's webpage for further steps about compiling and installing the RoCE driver. For Broadcom, see the section **Compiling Broadcom NIC Software from Source** in the [Ethernet Networking Guide for AMD Instinct MI300X GPU Clusters](https://docs.broadcom.com/doc/957608-AN2XX).

### Multi-node environment variables

Set the following environment variables.

**Master address**—change `localhost` to the master node's hostname:

```bash
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
```

**Number of nodes**—set the number of nodes you want to train on (for example, 2, 4, 8):

```bash
export NNODES="${NNODES:-1}"
```

**Node rank**—set the rank of each node (0 for master, 1 for the first worker node, and so on):

```bash
export NODE_RANK="${NODE_RANK:-0}"
```

**Network interface**—update the network interface in the script to match your system's network interface. To find your network interface, run this outside the container:

```bash
ip a
```

Then update the following variable in the script:

```bash
export NCCL_SOCKET_IFNAME=ens50f0np0
```

**RDMA interface**—first make sure the packages above are installed on all the nodes. Then set the RDMA interfaces to use for communication:

```bash
# If using Broadcom NIC
export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7

# If using Mellanox NIC
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
```

For Primus-specific networking guidance, see [Multi-node networking](../04-technical-guides/multi-node-networking.md).

---

## Training and benchmarking options

Use the following instructions to set up the environment, configure the script to train models, and reproduce the benchmark results on the MI300X, MI325X, MI350X, and MI355X accelerators with the Docker image.

There are three ways to run training, listed in the order we recommend:

| Method | Use it when |
| ------ | ----------- |
| [**`primus-cli`**](#running-training-with-primus-cli-recommended) **(recommended)** | Any new work. One CLI covers direct, container, and Slurm launches, and the same YAML configurations work across every Primus backend. |
| [Standalone benchmarking](#standalone-benchmarking) *(legacy)* | You want to run the MAD benchmark scripts yourself, outside the MAD harness. |
| [MAD-integrated benchmarking](#mad-integrated-benchmarking) *(legacy)* | You are reproducing published AMD numbers through the ROCm MAD dashboarding pipeline. |

JAX MaxText is integrated into [Primus](https://github.com/AMD-AGI/Primus), which supports multiple backends including Megatron-LM, TorchTitan, and JAX MaxText alongside ROCm-optimized components. The unified `primus-cli` runs training jobs with the JAX MaxText backend and is the path we recommend.

---

---

---

## Running training with primus-cli (recommended)

**Clone the Primus branch matching the image.** Do this on the host — every command below runs from this directory:

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout release/v26.5
git submodule update --init third_party/maxtext/
```

That is all the setup required. `primus-cli container` starts the image for you, mounts this checkout into it at the same path, and runs the training inside — so this is the code that executes, and the `/workspace/Primus` copy baked into the image is not used. It also forwards environment variables you export on the host; the forwarded list is `container.options.env` in `runner/.primus.yaml`.

> **Pass `--image` for MaxText.** The default image in `runner/.primus.yaml` is `rocm/primus`, which is the PyTorch image. MaxText runs need `--image rocm/jax-training:maxtext-v26.5` in container mode, or the image set in your Slurm config file.

For detailed usage of `primus-cli`, refer to the [CLI reference](./cli-reference.md).

The examples below target MI355X, so they export `RCCL_WARP_SPEED_AUTO=0` first — see [Architecture-specific settings](#architecture-specific-settings). The variable is a no-op on MI300X, so you can leave it in place on either architecture.

**Container mode (recommended)**—run from the host; `primus-cli` starts the container:

```bash
export RCCL_WARP_SPEED_AUTO=0
./runner/primus-cli container --image rocm/jax-training:maxtext-v26.5 \
  -- train pretrain --config examples/maxtext/configs/MI355X/llama2_7B-pretrain.yaml
```

**Direct mode**—only when you already have a shell **inside** the MaxText container (or a bare-metal JAX install); no `--image` needed:

```bash
export RCCL_WARP_SPEED_AUTO=0
./runner/primus-cli direct \
  -- train pretrain --config examples/maxtext/configs/MI355X/llama2_7B-pretrain.yaml
```

**Slurm mode**—distributed training on a Slurm cluster:

```bash
export RCCL_WARP_SPEED_AUTO=0
# Use a custom config file, where you can specify the docker image and set environment variables.
./runner/primus-cli --config my_maxtext_config.yaml slurm srun -N 8 \
  -- train pretrain --config examples/maxtext/configs/MI355X/llama2_7B-pretrain.yaml
```

To run a different model or GPU architecture, swap the `--config` path. Configurations live under `examples/maxtext/configs/MI300X/` and `examples/maxtext/configs/MI355X/`. See [End-to-end training recipes](./end-to-end-training-recipes.md) for the full inventory and [MaxText parameters](../03-configuration-reference/maxtext-parameters.md) for the YAML fields.

---

## Standalone benchmarking

Use the following command to pull the Docker image from Docker Hub:

```bash
docker pull rocm/jax-training:maxtext-v26.5
```

### Single-node training

#### Setup

> **Note:** Adjust the following variables based on your environment.

Export variables:

- `MAD_SECRETS_HFTOKEN` is your Hugging Face token to access models, tokenizers, and data. See [User access tokens](https://huggingface.co/docs/hub/en/security-tokens) for more information.
- `HF_HOME` is where `huggingface_hub` will store local data. Refer to the [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#hf-download) on how to download the data. If you already have downloaded or cached Hugging Face artifacts, set this variable to that path. Downloaded files typically get cached to a place like `~/.cache/huggingface`.

```bash
export MAD_SECRETS_HFTOKEN=<Your HuggingFace token>
export HF_HOME=<Location of saved/cached HuggingFace models>
```

Launch the Docker container:

```bash
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    -v $HOME:$HOME -v $HOME/.ssh:/root/.ssh -v $HF_HOME:/hf_cache \
    -e HF_HOME=/hf_cache -e MAD_SECRETS_HFTOKEN=$MAD_SECRETS_HFTOKEN \
    --shm-size 64G --name training_env rocm/jax-training:maxtext-v26.5
```

Execute the `training_env` container (optional if you are already in the container):

```bash
docker start training_env
docker exec -it training_env bash
```

Clone the Model Automation and Dashboarding (MAD) repository:

```bash
git clone https://github.com/ROCm/MAD.git
cd MAD/scripts/jax-maxtext
```

Run the setup scripts to install libraries and datasets needed for benchmarking:

```bash
./jax-maxtext_benchmark_setup.sh -m <model>
```

Run the benchmark in quantized or unquantized mode:

```bash
# For unquantized training
./jax-maxtext_benchmark_report.sh -m <model>

# Or for quantized training
./jax-maxtext_benchmark_report.sh -m <model> -q nanoo_fp8
```

The performance results should be written to a file in the parent folder.

### Benchmarking examples

1. **Single-node training with the Llama 2 7B model**

```bash
./jax-maxtext_benchmark_setup.sh -m Llama-2-7B

# For unquantized training
./jax-maxtext_benchmark_report.sh -m Llama-2-7B

# Or for nanoo_fp8 quantized training on MI300X
./jax-maxtext_benchmark_report.sh -m Llama-2-7B -q nanoo_fp8

# Or for fp8 quantized training on MI355X
./jax-maxtext_benchmark_report.sh -m Llama-2-7B -q fp8
```

2. **Single-node training with the Llama 2 70B model**

```bash
./jax-maxtext_benchmark_setup.sh -m Llama-2-70B

# For unquantized training
./jax-maxtext_benchmark_report.sh -m Llama-2-70B

# Or for nanoo_fp8 quantized training on MI300X
./jax-maxtext_benchmark_report.sh -m Llama-2-70B -q nanoo_fp8

# Or for fp8 quantized training on MI355X
./jax-maxtext_benchmark_report.sh -m Llama-2-70B -q fp8
```

3. **Single-node training with the Llama 3.1 8B model**

```bash
./jax-maxtext_benchmark_setup.sh -m Llama-3.1-8B

# For unquantized training
./jax-maxtext_benchmark_report.sh -m Llama-3.1-8B

# Or for nanoo_fp8 quantized training on MI300X
./jax-maxtext_benchmark_report.sh -m Llama-3.1-8B -q nanoo_fp8

# Or for fp8 quantized training on MI355X
./jax-maxtext_benchmark_report.sh -m Llama-3.1-8B -q fp8
```

4. **Single-node training with the Llama 3.1 70B model**

```bash
./jax-maxtext_benchmark_setup.sh -m Llama-3.1-70B

# For unquantized training
./jax-maxtext_benchmark_report.sh -m Llama-3.1-70B

# Or for fp8 quantized training on MI355X
./jax-maxtext_benchmark_report.sh -m Llama-3.1-70B -q fp8
```

5. **Single-node training with the Llama 3.3 70B model**

```bash
./jax-maxtext_benchmark_setup.sh -m Llama-3.3-70B

# For unquantized training
./jax-maxtext_benchmark_report.sh -m Llama-3.3-70B

# Or for fp8 quantized training on MI355X
./jax-maxtext_benchmark_report.sh -m Llama-3.3-70B -q fp8
```

6. **Single-node training with the DeepSeek2 16B model**

```bash
./jax-maxtext_benchmark_setup.sh -m DeepSeek-V2-lite

# For unquantized training
./jax-maxtext_benchmark_report.sh -m DeepSeek-V2-lite

# Or for nanoo_fp8 quantized training on MI300X
./jax-maxtext_benchmark_report.sh -m DeepSeek-V2-lite -q nanoo_fp8

# Or for fp8 quantized training on MI355X
./jax-maxtext_benchmark_report.sh -m DeepSeek-V2-lite -q fp8
```

7. **Single-node training with the Mixtral-8x7B model**

```bash
./jax-maxtext_benchmark_setup.sh -m Mixtral-8x7B

# For unquantized training
./jax-maxtext_benchmark_report.sh -m Mixtral-8x7B

# Or for nanoo_fp8 quantized training on MI300X
./jax-maxtext_benchmark_report.sh -m Mixtral-8x7B -q nanoo_fp8

# Or for fp8 quantized training on MI355X
./jax-maxtext_benchmark_report.sh -m Mixtral-8x7B -q fp8
```

8. **Single-node training with the Qwen3 14B model**

```bash
./jax-maxtext_benchmark_setup.sh -m Qwen3-14B

# For unquantized training
./jax-maxtext_benchmark_report.sh -m Qwen3-14B

# Or for nanoo_fp8 quantized training on MI300X
./jax-maxtext_benchmark_report.sh -m Qwen3-14B -q nanoo_fp8

# Or for fp8 quantized training on MI355X
./jax-maxtext_benchmark_report.sh -m Qwen3-14B -q fp8
```

9. **Single-node training with the Qwen3 30B-A3B model (MoE)**

```bash
./jax-maxtext_benchmark_setup.sh -m Qwen3-30B-A3B

# For unquantized training
./jax-maxtext_benchmark_report.sh -m Qwen3-30B-A3B

# Or for nanoo_fp8 quantized training on MI300X
./jax-maxtext_benchmark_report.sh -m Qwen3-30B-A3B -q nanoo_fp8

# Or for fp8 quantized training on MI355X
./jax-maxtext_benchmark_report.sh -m Qwen3-30B-A3B -q fp8
```

### Multi-node training

> **Note:** These scripts will launch the Docker container and execute the benchmark, so **run them outside of any Docker container**.

The examples below use Slurm for running on multiple nodes. The unified multi-node benchmark script accepts a configuration file that specifies the model and training parameters.

#### Running multi-node training

To run multi-node training, use the following command:

```bash
sbatch -N <NUM_NODES> jax_maxtext_multinode_benchmark.sh <config_file.yml> [docker_image]
```

**Parameters:**

- `<NUM_NODES>`: number of nodes to use for training (for example, 2, 4, 8)
- `<config_file.yml>`: path to the YAML configuration file containing model and training parameters
- `[docker_image]`: (optional) Docker image to use. If not specified, defaults to `rocm/jax-training:maxtext-v26.5`

**Configuration files** are available in the `scripts/jax-maxtext/env_scripts/` directory for different models and GPU architectures.

For MI300X (gfx942):

- `llama2_7b.yml`—Llama 2 7B
- `llama2_70b.yml`—Llama 2 70B
- `llama3_8b.yml`—Llama 3 8B
- `llama3_70b.yml`—Llama 3 70B
- `qwen3_14b.yml`—Qwen3 14B
- `qwen3_30b_a3b.yml`—Qwen3 30B-A3B

For MI355X (gfx950):

- `gfx950_llama2_7b.yml`—Llama 2 7B
- `gfx950_llama2_70b.yml`—Llama 2 70B
- `gfx950_llama3_8b.yml`—Llama 3 8B
- `gfx950_llama3_70b.yml`—Llama 3 70B
- `gfx950_llama3.1_405b.yml`—Llama 3.1 405B
- `gfx950_qwen3_14b.yml`—Qwen3 14B
- `gfx950_qwen3_30b_a3b.yml`—Qwen3 30B-A3B

#### Example commands

1. **Multi-node training with the Llama 2 7B model on 2 nodes:**

```bash
sbatch -N 2 jax_maxtext_multinode_benchmark.sh env_scripts/llama2_7b.yml
```

2. **Multi-node training with the Llama 2 70B model on 4 nodes with a custom image:**

```bash
sbatch -N 4 jax_maxtext_multinode_benchmark.sh env_scripts/llama2_70b.yml rocm/jax-training:maxtext-v26.5
```

3. **Multi-node training with the Llama 3 8B model on 2 nodes:**

```bash
sbatch -N 2 jax_maxtext_multinode_benchmark.sh env_scripts/llama3_8b.yml
```

4. **Multi-node training with the Llama 3 70B model on 8 nodes:**

```bash
sbatch -N 8 jax_maxtext_multinode_benchmark.sh env_scripts/llama3_70b.yml
```

5. **Multi-node training with the Llama 3.1 405B model on MI355X (gfx950) with 8 nodes:**

```bash
sbatch -N 8 jax_maxtext_multinode_benchmark.sh env_scripts/gfx950_llama3.1_405b.yml
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

Run models through MAD-integrated benchmarking with the following command:

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
python3 tools/run_models.py --tags <mad_model> --keep-model-dir --live-output --timeout 28800
```

For example, use this command to run a performance benchmark test of the Llama 2 7B model on one GPU with the `bf16` data type on the host machine:

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
python3 tools/run_models.py --tags jax_maxtext_train_llama-2-7b --keep-model-dir --live-output --timeout 28800
```

> **Note:** The `madengine` package is now available, allowing for the replacement of `run_models.py`.

```sh
export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
python3 madengine run --tags jax_maxtext_train_llama-2-7b --keep-model-dir --live-output --timeout 28800
```

ROCm MAD launches a Docker container with the name `container_ci-jax_maxtext_train_llama-2-7b`. The latency and throughput reports of the model are collected in the following path:

```sh
~/MAD/perf.csv
```

### Available models

| model_name |
| --------------------------------------- |
| `jax_maxtext_train_llama-2-7b` |
| `jax_maxtext_train_llama-2-70b` |
| `jax_maxtext_train_llama-3.1-8b` |
| `jax_maxtext_train_llama-3.1-70b` |
| `jax_maxtext_train_llama-3.1-405b` |
| `jax_maxtext_train_llama-3.3-70b` |
| `jax_maxtext_train_deepseek-v2-lite-16b` |
| `jax_maxtext_train_mixtral-8x7b` |
| `jax_maxtext_train_qwen3-14b` |
| `jax_maxtext_train_qwen3-30b-a3b` |

---

## Profiling with JAX XPlane Profiler

MaxText has built-in XPlane profiling support via JAX's profiler. Traces capture GPU kernel timelines, RCCL collectives, HLO graphs, and more. The output can be viewed in TensorBoard's Trace Viewer or analyzed with TraceLens.

### Key MaxText profiler flags

The following MaxText config keys control profiling:

```text
profiler=xplane                    # Use xplane format (produces .xplane.pb files)
skip_first_n_steps_for_profiler=2  # Skip compilation/warmup steps
profiler_steps=5                   # Number of steps to profile
upload_all_profiler_results=True   # Save all GPU profiles (not just GPU0)
```

**Choosing step counts:**

- `steps` should be greater than `skip_first_n_steps_for_profiler` + `profiler_steps` (for example, `steps=12` with `skip=2` and `profile=5` gives 5 warmup + 5 profiled + 2 cooldown)
- `skip_first_n_steps_for_profiler=2` skips step 0 (compilation) and step 1 (warmup)
- `profiler_steps=5` is typically enough; more steps mean larger `.xplane.pb` files

### Profiling with MAD/madengine

The model YAML configs under `scripts/jax-maxtext/env_scripts/` already include a `profiler` key (set to `""` by default). To enable profiling when running through MAD or madengine, edit the YAML config for your model and set the profiler fields:

```yaml
profiler: "xplane"
skip_first_n_steps_for_profiler: 2
profiler_steps: 5
upload_all_profiler_results: True
steps: 12
```

Then run the benchmark as usual:

```bash
# Via madengine
python3 madengine run --tags jax_maxtext_train_llama-3.1-8b --keep-model-dir --live-output --timeout 28800

# Or via run_models.py
python3 tools/run_models.py --tags jax_maxtext_train_llama-3.1-8b --keep-model-dir --live-output --timeout 28800
```

Profile output will be written under the `base_output_directory` specified in the YAML (see [Output structure](#output-structure) below). Use `--keep-model-dir` so the container's output directory is preserved after the run.

### Example: profile a model standalone in Docker

```bash
#!/bin/bash
set -e

IMAGE="$1"       # Docker image, e.g. rocm/jax-training:maxtext-v26.5
TAG="$2"         # Short tag for output folder, e.g. v26.5_llama2_7b
PROFILE_DIR="/path/to/profiles/${TAG}"

mkdir -p "${PROFILE_DIR}"

docker run --rm --privileged --network=host \
  --device=/dev/dri --device=/dev/kfd --ipc=host \
  -v "${PROFILE_DIR}:/mnt/profile" \
  "${IMAGE}" bash -c '
export XLA_PYTHON_CLIENT_MEM_FRACTION=.97
export LD_LIBRARY_PATH=/usr/local/lib/:/opt/rocm/lib:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=True --xla_gpu_enable_command_buffer= <your other XLA flags>"
export GPU_MAX_HW_QUEUES=2
# On MI355X (gfx950), disable RCCL WarpSpeed to avoid NaN losses (no-op on MI300X)
export RCCL_WARP_SPEED_AUTO=0

cd /workspace/maxtext

python3 -m MaxText.train src/MaxText/configs/base.yml \
  run_name=profile \
  base_output_directory=/mnt/profile \
  hardware=gpu \
  steps=12 \
  model_name=<your-model> \
  dataset_type=synthetic \
  enable_checkpointing=False \
  enable_goodput_recording=False \
  monitor_goodput=False \
  <your model-specific flags> \
  profiler=xplane \
  skip_first_n_steps_for_profiler=2 \
  profiler_steps=5 \
  upload_all_profiler_results=True
' 2>&1 | tee "${PROFILE_DIR}/run.log"

echo "Profile files:"
find "${PROFILE_DIR}" -name "*.xplane.pb" -o -name "*.trace.json.gz" 2>/dev/null
```

### Output structure

MaxText writes profiles in TensorBoard format:

```text
<base_output_directory>/
└── profile/
    └── tensorboard/
        └── plugins/
            └── profile/
                └── <YYYY_MM_DD_HH_MM_SS>/
                    ├── <hostname>.xplane.pb      # Raw XPlane proto (GPU timelines)
                    ├── <hostname>.trace.json.gz  # Trace viewer data
                    └── *.hlo_proto.pb            # HLO graphs for each compiled module
```

### Viewing traces in TensorBoard

```bash
pip install tensorboard tensorboard-plugin-profile

# Point --logdir at the directory containing the tensorboard/ folder
tensorboard --logdir /path/to/profiles/<TAG>/profile --port 6006
```

Navigate to **Profile > Trace Viewer** in the TensorBoard UI.

**Tips:**

- Zoom into a single training step (skip the first profiled step as it may have residual warmup)
- Look at individual GPU streams to see compute/RCCL overlap

### Keeping profile files small

- Use `profiler_steps=5` (not more) to keep `.xplane.pb` under approximately 100 MB
- Too many steps can produce files over 500 MB that TensorBoard struggles to load
- `enable_checkpointing=False` avoids checkpoint I/O noise in the trace
- `dataset_type=synthetic` eliminates data loading variability

---

## Profiling with rocprofv3

If you need to collect a trace and the JAX profiler isn't working, you can use `rocprofv3` as a temporary workaround:

```bash
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace --rccl-trace --output-format pftrace -d ./v3_traces -- python3 app.py
```

- Replace `python3 app.py` with any command line command that you want to run, such as `./jax-maxtext_benchmark_report.sh -m Llama-2-7B`.
- You can set the directory where you want the `.json` traces to be saved using `-d <TRACE_DIRECTORY>`.
- The resulting traces can be opened in [Perfetto](https://ui.perfetto.dev/).

---

## Related documentation

- [End-to-end training recipes](./end-to-end-training-recipes.md)
- [Pretraining workflows](./pretraining.md)
- [MaxText parameters](../03-configuration-reference/maxtext-parameters.md)
- [CLI reference](./cli-reference.md)
- [Multi-node networking](../04-technical-guides/multi-node-networking.md)
