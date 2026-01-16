# Primus – Megatron-Bridge + Transformers Setup (Qwen 3-8B SFT Example Example Workflow)

> **Note**: This is an initial example workflow. The process needs to be streamlined, automated, and paths/code need to be refined for production use.

## Setup Instructions

### 1. Install Megatron-LM

Install the third-party Megatron-LM package in editable mode:

```bash
cd /workspace/Primus/third_party/megatron-lm
pip install -e .
```

### 2. Apply Megatron Bridge Patches

Install and apply patches for megatron-bridge and transformers:

```bash
bash /workspace/Primus/primus/backends/megatron_bridge/setup_env_megatron_bridge.sh
```

**What this script does:**
- Clones and installs HuggingFace Transformers v4.57-release from source
- Patches `transformers/modeling_utils.py` with required imports
- Replaces megatron-bridge `pyproject.toml` with custom version
- Installs NVIDIA packages (onnx, modelopt, resiliency_ext)
- Sets `HSA_NO_SCRATCH_RECLAIM=1` environment variable
- Patches megatron-bridge files (`cuda_graph_impl` → `cuda_graph_scope`, enables `use_te_rng_tracker`)
- Installs megatron-bridge in editable mode

### 3. Convert Checkpoints

Convert the Qwen 3-8B model from HuggingFace format to Megatron format:

```bash
cd /workspace/Primus
python third_party/megatron-bridge/examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen3-8B \
  --megatron-path /workspace/post_training/qwen3-8b_megatron
```

### 4. Run SFT Training

Execute the supervised fine-tuning example:

```bash
cd /workspace/Primus
bash runner/primus-cli direct \
  -- train post_train \
  --config examples/megatron_bridge/qwen3_8b_posttrain.yaml
```

## Future Improvements

- [ ] Automate the setup process
- [ ] Streamline installation steps
- [ ] Refine and standardize paths
- [ ] Add error handling and validation
- [ ] Create configuration templates
- [ ] Add detailed documentation for each step
