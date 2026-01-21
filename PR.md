# Post-Training Framework Support

## Overview
Add support for Megatron-Bridge post-training with automatic framework detection and dependency management.

## Usage

```bash
./runner/primus-cli container \
--image rocm/primus:v25.10 \
-- --env HF_TOKEN=$HF_TOKEN \
train posttrain --config ./examples/configs/megatron_bridge/qwen3_32B_lora_posttrain.yaml
```

## Key Features
- Automatic framework detection from config (`post_trainer.framework`)
- Framework-specific hooks organized in subdirectories
- Multi-rank checkpoint conversion with synchronization
- Automatic dependency installation
