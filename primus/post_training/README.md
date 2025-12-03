# Post-Training Setup Guide

## Quick Start

### Step 1: Start Docker Container

Start your Docker container with the required environment.

### Step 2: Install Dependencies

Install the required packages:
- Megatron-LM (3rd party, Primus-supported version)
- Transformers (4.57 with patches)
- ONNX (1.20.0rc1)
- NVIDIA ModelOpt (0.39.0)
- nvidia_resiliency_ext (0.4.1)

Set the environment variable:
```bash
export HSA_NO_SCRATCH_RECLAIM=1
```

### Step 3: Setup Python Paths

Configure the PYTHONPATH for Megatron-LM and post_training:

```bash
export PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM
export PYTHONPATH=$PYTHONPATH:/workspace/Primus/primus/post_training
```

### Step 4: Convert Model from HuggingFace to Megatron Format

Convert the Qwen3 8B model:

```bash
python post_training/examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen3-8B \
  --megatron-path /workspace/models/qwen3-8b_megatron
```

### Step 5: Run Qwen3 8B Finetuning

Launch the finetuning script:

```bash
torchrun --nproc_per_node=8 \
  /workspace/Primus/primus/post_training/examples/recipes/qwen3/finetune_qwen3_8b.py \
  --pretrained-checkpoint /workspace/models/qwen3-8b_megatron/
```

---

## Detailed Installation Instructions

### 1. Clone and Install Megatron-LM

Install Megatron-LM at a specific commit:

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout 847781764fe468c90caec16309deded245c1022c
pip install -e .
```

### 2. Install Transformers 4.57 Release Branch

#### Clone Transformers

```bash
cd ..
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.57-release
```

#### Modify `modeling_utils.py`

Open the file:
```bash
src/transformers/modeling_utils.py
```

Add the following import block near the other imports:

```python
from .pytorch_utils import (  # noqa: F401
    Conv1D,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    id_tensor_storage,
    prune_conv1d_layer,
    prune_layer,
    prune_linear_layer,
)
```

#### Install Transformers from Source

```bash
pip install -e .
```

### 3. Install ONNX

Install the specific prerelease version:

```bash
pip install onnx==1.20.0.rc1
```

### 4. Install NVIDIA ModelOpt

Install the latest version:

```bash
pip install -U nvidia-modelopt
```

---

## Additional Notes

- Ensure all environment variables are set before running training scripts
- The Megatron-LM commit hash is critical for compatibility
- The Transformers patch is required for proper functionality
