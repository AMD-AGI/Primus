# NVIDIA GPU support

Primus is primarily designed for AMD/ROCm, but the Megatron-LM and TorchTitan backends work on NVIDIA GPUs too. This document covers setup and the `target_gpu` field that controls which GPU-specific code runs.

> **Note:** MaxText/JAX on NVIDIA requires a separate JAX+CUDA container image, not covered by `docker/Dockerfile.nvidia`. See [MaxText on NVIDIA](#maxtext-on-nvidia).

---

## Prerequisites

- NVIDIA GPU with CUDA compute capability ≥ 8.0 (Ampere or newer)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA driver ≥ 525

```bash
# Verify prerequisites
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## Quick start (Megatron-LM)

### 1. Build the Docker image

```bash
docker build -t primus-nvidia:latest -f docker/Dockerfile.nvidia .
```

### 2. Run training

```bash
GPUS_PER_NODE=2 ./primus-cli --config runner/nvidia.yaml container \
  -- train pretrain --config examples/megatron/configs/nvidia/llama3.2_1B-BF16-pretrain.yaml
```

---

## The `target_gpu` field

All Primus YAML experiment configs accept a top-level `target_gpu` field:

```yaml
target_gpu: auto   # auto | amd | nvidia  (default: auto)
```

| Value | Behavior |
|-------|----------|
| `auto` | Detects GPU at runtime: AMD if `torch.version.hip` is set, NVIDIA otherwise |
| `amd` | Forces AMD/ROCm — enables ROCm validation and Primus-Turbo patches |
| `nvidia` | Forces NVIDIA/CUDA — skips ROCm validation and Primus-Turbo patches |

| `target_gpu` | `torch.version.hip` | Effective target | ROCm validation | Primus-Turbo |
|---|---|---|---|---|
| `auto` | set (ROCm) | `amd` | called | applied if installed |
| `auto` | unset (CUDA) | `nvidia` | skipped | skipped |
| `amd` | any | `amd` | called | applied if installed |
| `nvidia` | any | `nvidia` | skipped | skipped |

---

## Provided example configs

| Backend | Config file |
|---------|-------------|
| Megatron-LM | `examples/megatron/configs/nvidia/llama3.2_1B-BF16-pretrain.yaml` |
| TorchTitan | `examples/torchtitan/configs/nvidia/llama3.1_8B-BF16-pretrain.yaml` |
| MaxText | `examples/maxtext/configs/nvidia/llama2_7B-pretrain.yaml` |

All NVIDIA configs set `target_gpu: nvidia` and disable Primus-Turbo:

```yaml
target_gpu: nvidia

# Turbo (AMD-only; disabled for NVIDIA)
enable_primus_turbo: false
use_turbo_attention: false
use_turbo_grouped_mlp: false
```

---

## Runner configuration (`runner/nvidia.yaml`)

`runner/nvidia.yaml` holds NVIDIA defaults for both `container` and `direct` modes:

```yaml
container:
  options:
    image: "primus-nvidia:latest"
    gpus: "all"          # NVIDIA GPU access (replaces ROCm --device flags)
    device: []           # no /dev/kfd or /dev/dri needed
    ...

direct:
  gpus_per_node: 2
  run_mode: "torchrun"
  numa: "false"
  env:
    - "GPUS_PER_NODE=2"
```

Pass it with `--config runner/nvidia.yaml` to override the default `runner/.primus.yaml`.

---

## Docker image (`docker/Dockerfile.nvidia`)

The NVIDIA Dockerfile starts from `nvcr.io/nvidia/pytorch:26.01-py3` and installs:

- Megatron-LM from `third_party/Megatron-LM/` (not on PyPI)
- TorchTitan from `third_party/torchtitan/`
- Primus Python dependencies, excluding AMD-only packages like `pyrsmi`

```bash
docker build -t primus-nvidia:latest -f docker/Dockerfile.nvidia .

# Use a different base image
docker build -t primus-nvidia:latest -f docker/Dockerfile.nvidia \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3 .
```

---

## Environment variables

When `nvidia-smi` is detected (and `rocm-smi` is absent), `runner/helpers/envs/NVIDIA.sh` is sourced automatically. It:

- Sets `CUDA_VISIBLE_DEVICES` from `GPUS_PER_NODE`
- Unsets AMD-specific variables (`HIP_VISIBLE_DEVICES`, `HSA_ENABLE_SDMA`, `NVTE_ROCM_ENABLE_MXFP8`, ROCm CK kernel flags, etc.)
- Sets `CUDA_DEVICE_MAX_CONNECTIONS=1`

---

## Per-backend notes

### Megatron-LM

- ROCm-specific argument validation (`validate_args_on_rocm`) is skipped when `target_gpu` resolves to `nvidia`.
- Primus-Turbo patches check `importlib.util.find_spec("primus_turbo")` before activating — they are no-ops when the package is absent.
- Set `gradient_accumulation_fusion: false` (default in the NVIDIA config) to avoid a CUDA kernel incompatibility.

### TorchTitan

- All Primus-Turbo patch conditions include a `find_spec("primus_turbo")` guard, so patches are silently skipped when `primus_turbo` is not installed.
- **Known limitation:** `primus/backends/torchtitan/models/moe/moe.py` and `components/quantization/mx.py` have unconditional top-level `import primus_turbo` statements. These files are only imported for MoE and MX-precision models — LLaMA BF16 is not affected. NVIDIA support for those model types requires wrapping those imports in `try/except ImportError`.

### MaxText on NVIDIA

MaxText uses JAX, not PyTorch. Running it on NVIDIA requires a JAX+CUDA container image (e.g. `nvcr.io/nvidia/jax:24.04-py3` or the Google JAX stable-stack wheel), which `docker/Dockerfile.nvidia` does not provide.

When `target_gpu: nvidia` is set in a MaxText config, the prepare hook (`runner/helpers/hooks/train/pretrain/maxtext/prepare.py`) emits only NVIDIA-compatible environment variables and skips all AMD/ROCm-specific ones (`HIP_FORCE_DEV_KERNARG`, `HSA_FORCE_FINE_GRAIN_PCIE`, `NVTE_CK_*`, etc.).

---

## Usage examples

```bash
# Megatron-LM: LLaMA 3.2 1B on 2× RTX 3090
docker build -t primus-nvidia:latest -f docker/Dockerfile.nvidia .
GPUS_PER_NODE=2 ./primus-cli --config runner/nvidia.yaml container \
  -- train pretrain --config examples/megatron/configs/nvidia/llama3.2_1B-BF16-pretrain.yaml

# TorchTitan: LLaMA 3.1 8B on 2× RTX 3090
GPUS_PER_NODE=2 ./primus-cli --config runner/nvidia.yaml container \
  -- train pretrain --config examples/torchtitan/configs/nvidia/llama3.1_8B-BF16-pretrain.yaml

# Direct mode (inside container or bare-metal CUDA environment)
GPUS_PER_NODE=2 ./primus-cli --config runner/nvidia.yaml direct \
  -- train pretrain --config examples/megatron/configs/nvidia/llama3.2_1B-BF16-pretrain.yaml

# AMD (unchanged)
./primus-cli container \
  -- train pretrain --config examples/megatron/configs/MI300X/llama3.2_1B-BF16-pretrain.yaml
```

---

## Troubleshooting

**Container exits with "No GPU devices configured" or `/dev/kfd` errors**

Make sure you are using `runner/nvidia.yaml`. It sets `gpus: all` and `device: []`, which skips the ROCm device path validation.

**`ImportError: No module named 'primus_turbo'`**

Set all Turbo flags to `false` in your config and add `target_gpu: nvidia`. Primus-Turbo is AMD-only and is not in the NVIDIA Docker image.

**NCCL errors on multi-GPU**

Set `NCCL_SOCKET_IFNAME` and `MASTER_ADDR` to the correct network interface/IP. Add them to the `env:` list in `runner/nvidia.yaml` if needed.

**`torchrun` only starts 1 process instead of 2**

Check that `GPUS_PER_NODE` is exported before calling `primus-cli`, or set it under `direct.env` in `runner/nvidia.yaml`.
