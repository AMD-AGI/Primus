# Qwen3 30B-A3B MoE Post-training 使用指南

## 模型概述

**Qwen3-30B-A3B** 是一个混合专家（MoE）模型：
- **总专家数**: 128 个专家
- **激活专家数**: 每个 token 激活 8 个专家
- **总参数量**: ~87B
- **激活参数**: ~30B（每个 token）
- **架构**: 48 层，2048 隐藏维度，32 个注意力头

## 配置文件

### 1. 标准 SFT 配置
**文件**: `qwen3_30B_A3B_posttrain.yaml`

适用场景：
- 全量微调（所有参数可训练）
- 需要最佳性能
- 有足够 GPU 资源（8+ GPUs）

```bash
torchrun --nproc-per-node=8 \
    -m primus.cli.train \
    --framework megatron_bridge \
    --config examples/configs/megatron_bridge/qwen3_30B_A3B_posttrain.yaml \
    --load /path/to/pretrained/checkpoint \
    --data_path /path/to/instruction/data
```

### 2. LoRA 微调配置（推荐）
**文件**: `qwen3_30B_A3B_lora_posttrain.yaml`

适用场景：
- 参数高效微调（只训练少量参数）
- GPU 资源有限（4 GPUs 即可）
- 快速实验和迭代
- 多任务适配（可训练多个 LoRA adapter）

```bash
torchrun --nproc-per-node=4 \
    -m primus.cli.train \
    --framework megatron_bridge \
    --config examples/configs/megatron_bridge/qwen3_30B_A3B_lora_posttrain.yaml \
    --data_path /path/to/instruction/data
```

## 关键配置说明

### MoE 并行策略

对于 128 专家的 MoE 模型，**Expert Parallelism (EP)** 至关重要：

```yaml
parallelism:
  expert_model_parallel_size: 8  # 128专家 / 8 = 每个GPU 16专家
```

**EP 值选择**:
- `expert_model_parallel_size=4`: 需要 4 GPUs（每 GPU 32 专家）
- `expert_model_parallel_size=8`: 需要 8 GPUs（每 GPU 16 专家）- **推荐**
- `expert_model_parallel_size=16`: 需要 16 GPUs（每 GPU 8 专家）

### LoRA 配置

```yaml
lora:
  use_lora: true
  lora_rank: 32        # 越大容量越强，内存越多
  lora_alpha: 64       # 通常 = 2 * rank
  lora_dropout: 0.1    # 防止过拟合

  lora_target_modules:
    - q_proj          # Query 投影
    - k_proj          # Key 投影
    - v_proj          # Value 投影
    - o_proj          # Output 投影
```

**LoRA 优势**:
- 只训练 ~1-2% 的参数
- 内存需求降低 50-60%
- 训练速度提升 2-3x
- Checkpoint 大小仅 500MB（vs 全量 ~170GB）

### 学习率设置

```yaml
# 全量微调
training:
  lr: 2.0e-6          # 非常低的学习率
  min_lr: 2.0e-7

# LoRA 微调
training:
  lr: 2.0e-4          # 可以用更高的学习率
  min_lr: 2.0e-5
  weight_decay: 0.0    # LoRA 不需要 weight decay
```

## 资源需求

### 全量微调

| 配置 | GPU | GPU 内存 | 预计时间 |
|------|-----|----------|----------|
| EP=8, BS=128 | 8x A100 80GB | ~70GB/GPU | 基准 |
| EP=4, BS=64  | 4x A100 80GB | ~75GB/GPU | ~2x 慢 |

### LoRA 微调

| 配置 | GPU | GPU 内存 | 预计时间 |
|------|-----|----------|----------|
| EP=4, BS=32, Rank=32 | 4x A100 80GB | ~35GB/GPU | 基准 |
| EP=4, BS=32, Rank=16 | 4x A100 40GB | ~30GB/GPU | 相似 |
| EP=8, BS=32, Rank=32 | 8x A100 40GB | ~25GB/GPU | 更快 |

## 数据准备

### Alpaca 格式

```json
{
  "instruction": "解释量子计算的基本原理",
  "input": "",
  "output": "量子计算利用量子力学的叠加和纠缠特性..."
}
```

### ShareGPT 格式（多轮对话）

```json
{
  "conversations": [
    {"from": "human", "value": "你好！"},
    {"from": "gpt", "value": "你好！我是Qwen，有什么可以帮你的吗？"},
    {"from": "human", "value": "介绍一下量子计算"}
  ]
}
```

### Qwen Chat 格式

使用 `prompt_template: "qwen"` 和 `chat_format: "chatml"`

## 使用示例

### 示例 1: 从 HuggingFace 加载并 LoRA 微调

```bash
#!/bin/bash

export NPROC_PER_NODE=4

torchrun --nproc-per-node=$NPROC_PER_NODE \
    -m primus.cli.train \
    --framework megatron_bridge \
    --convert_from_hf \
    --hf_model_name_or_path Qwen/Qwen3-30B-A3B \
    --use_lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --data_path /data/instruction_data \
    --dataset_format alpaca \
    --seq_length 2048 \
    --micro_batch_size 1 \
    --global_batch_size 32 \
    --lr 2e-4 \
    --train_iters 3000 \
    --save /checkpoints/qwen3_lora \
    --save_interval 500 \
    --save_lora_only
```

### 示例 2: 从 Megatron Checkpoint 全量微调

```bash
#!/bin/bash

export NPROC_PER_NODE=8

torchrun --nproc-per-node=$NPROC_PER_NODE \
    -m primus.cli.train \
    --framework megatron_bridge \
    --config examples/configs/megatron_bridge/qwen3_30B_A3B_posttrain.yaml \
    --load /checkpoints/qwen3_pretrained \
    --data_path /data/instruction_data \
    --micro_batch_size 2 \
    --global_batch_size 128 \
    --lr 2e-6 \
    --train_iters 5000 \
    --save /checkpoints/qwen3_finetuned \
    --convert_to_hf \
    --hf_save_path /output/qwen3_hf
```

### 示例 3: 多任务 LoRA 适配

训练多个任务特定的 LoRA adapters：

```bash
# 任务 1: 代码生成
torchrun --nproc-per-node=4 \
    -m primus.cli.train \
    --framework megatron_bridge \
    --config examples/configs/megatron_bridge/qwen3_30B_A3B_lora_posttrain.yaml \
    --data_path /data/code_instructions \
    --save /checkpoints/qwen3_lora_code

# 任务 2: 数学推理
torchrun --nproc-per-node=4 \
    -m primus.cli.train \
    --framework megatron_bridge \
    --config examples/configs/megatron_bridge/qwen3_30B_A3B_lora_posttrain.yaml \
    --data_path /data/math_instructions \
    --save /checkpoints/qwen3_lora_math

# 任务 3: 对话
torchrun --nproc-per-node=4 \
    -m primus.cli.train \
    --framework megatron_bridge \
    --config examples/configs/megatron_bridge/qwen3_30B_A3B_lora_posttrain.yaml \
    --data_path /data/chat_data \
    --dataset_format sharegpt \
    --save /checkpoints/qwen3_lora_chat
```

## 性能优化

### 1. 内存优化

```yaml
# 启用激活检查点
activation_checkpointing:
  recompute_activations: true
  recompute_granularity: "selective"

# 使用 LoRA
lora:
  use_lora: true
  lora_rank: 16  # 降低 rank

# 减小批次
training:
  micro_batch_size: 1
  global_batch_size: 16
```

### 2. 速度优化

```yaml
# 增加并行度
parallelism:
  expert_model_parallel_size: 8

# 梯度累积融合
gradient_accumulation_fusion: true

# 重叠通信
parallelism:
  overlap_grad_reduce: true
  overlap_param_gather: true

# 使用更短的序列（如果数据允许）
data:
  seq_length: 2048
```

### 3. MoE 特定优化

```yaml
# 强制负载均衡
training:
  moe_router_force_load_balancing: true

# 使用优化的 GEMM
moe:
  moe_use_legacy_grouped_gemm: false

# Optional: Flex dispatcher
# moe_flex_dispatcher_backend: "cuBLAS"
```

## 故障排除

### 问题 1: OOM (Out of Memory)

**解决方案**:
1. 使用 LoRA: `use_lora: true`, `lora_rank: 16`
2. 启用激活检查点: `recompute_activations: true`
3. 增加 EP: `expert_model_parallel_size: 16`
4. 减小批次: `micro_batch_size: 1`, `global_batch_size: 16`
5. 减小序列长度: `seq_length: 1024`

### 问题 2: 训练不稳定

**解决方案**:
1. 降低学习率: `lr: 1e-6` (全量) 或 `lr: 1e-4` (LoRA)
2. 增加 warmup: `lr_warmup_iters: 200`
3. 启用梯度裁剪: `clip_grad: 1.0`
4. 检查数据质量

### 问题 3: 专家负载不均衡

**解决方案**:
1. 启用强制负载均衡: `moe_router_force_load_balancing: true`
2. 监控专家使用情况
3. 调整 router 参数

### 问题 4: Checkpoint 太大

**解决方案**:
1. 使用 LoRA: checkpoint 只有 ~500MB
2. 不保存优化器状态: `no_save_optim: true`
3. 仅保存 LoRA: `save_lora_only: true`

## 评估和部署

### 1. 评估

```bash
# 在验证集上评估
torchrun --nproc-per-node=4 \
    -m primus.cli.evaluate \
    --framework megatron_bridge \
    --load /checkpoints/qwen3_finetuned \
    --data_path /data/validation
```

### 2. 导出到 HuggingFace

```bash
# 导出全量模型
python -m primus.cli.convert \
    --framework megatron_bridge \
    --load /checkpoints/qwen3_finetuned \
    --convert_to_hf \
    --hf_save_path /output/qwen3_hf

# 导出 LoRA（会自动合并到基础模型）
python -m primus.cli.convert \
    --framework megatron_bridge \
    --load /checkpoints/qwen3_lora \
    --convert_to_hf \
    --hf_save_path /output/qwen3_lora_merged_hf
```

### 3. 推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained("/output/qwen3_hf")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

# 推理
prompt = "解释一下量子计算"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

## 最佳实践

1. **从 LoRA 开始**: 先用 LoRA 快速验证数据和超参数
2. **监控专家负载**: MoE 模型需要关注专家使用的均衡性
3. **适当的序列长度**: Qwen3 支持长序列，但 2-4K 对大多数任务足够
4. **保存频繁**: MoE 训练可能不稳定，多保存 checkpoint
5. **使用 bf16**: 比 fp16 更稳定，特别是对 MoE
6. **合理的 EP 值**: 通常 8 或 16 是最优的

## 参考资源

- [Qwen3 模型卡](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- [Megatron-Bridge 文档](https://docs.nvidia.com/nemo/megatron-bridge/latest/)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [MoE 架构说明](https://arxiv.org/abs/2101.03961)
