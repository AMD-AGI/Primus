# Megatron-Bridge Backend - 快速开始指南

## 简介

本指南帮助您快速开始使用 Primus 的 Megatron-Bridge backend 进行**后训练任务**（SFT、指令调优、LoRA 微调）。

**注意**: Megatron-Bridge 在 Primus 中专门用于后训练（post-training）。如需预训练，请使用 `megatron` 或 `torchtitan` backend。

## 安装

### 1. 初始化 Submodule

```bash
# 从项目根目录执行
cd /path/to/Primus

# 初始化并更新 Megatron-Bridge submodule
git submodule update --init --recursive third_party/Megatron-Bridge
```

### 2. 安装依赖

```bash
# 安装 Megatron-Bridge 及其依赖
cd third_party/Megatron-Bridge
pip install -e .

# 或者使用 uv（推荐）
uv pip install -e .
```

## 快速开始

### 方式 1: 从 Megatron Checkpoint 微调

```bash
# 监督微调 (SFT)
torchrun --nproc-per-node=8 \
    -m primus.cli.train \
    --framework megatron_bridge \
    --config examples/configs/megatron_bridge/llama_sft_posttrain.yaml \
    --load /path/to/pretrained/checkpoint \
    --data_path /path/to/instruction/data
```

### 方式 2: 从 HuggingFace 模型微调

```bash
# 加载 HuggingFace 模型并微调
torchrun --nproc-per-node=8 \
    -m primus.cli.train \
    --framework megatron_bridge \
    --convert_from_hf \
    --hf_model_name_or_path meta-llama/Llama-3-8B \
    --data_path /path/to/instruction/data \
    --lr 5e-6 \
    --train_iters 5000
```

### 方式 3: LoRA 参数高效微调

```bash
# LoRA 微调（显著降低资源需求）
torchrun --nproc-per-node=8 \
    -m primus.cli.train \
    --framework megatron_bridge \
    --config examples/configs/megatron_bridge/llama_lora_posttrain.yaml \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32
```

## 配置示例

### 基础 SFT 配置

创建 `my_sft_config.yaml`:

```yaml
framework: megatron_bridge

# 加载预训练模型
model:
  load: /path/to/pretrained/checkpoint

# 后训练超参数
training:
  micro_batch_size: 1
  global_batch_size: 128
  train_iters: 5000
  lr: 5.0e-6
  min_lr: 5.0e-7
  lr_warmup_iters: 100

# 混合精度
precision:
  bf16: true

# 并行策略（通常比预训练简单）
parallelism:
  tensor_model_parallel_size: 2
  pipeline_model_parallel_size: 1

# 指令数据集
data:
  data_path: /path/to/instruction/data
  dataset_format: "alpaca"
  prompt_template: "alpaca"
  split: "95,5,0"
  seq_length: 2048

# Checkpointing
checkpointing:
  save: /path/to/finetuned/checkpoints
  save_interval: 500

# 转换为 HuggingFace 格式
convert_to_hf: true
hf_save_path: /path/to/output/hf_model
```

运行:

```bash
torchrun --nproc-per-node=8 \
    -m primus.cli.train \
    --config my_sft_config.yaml
```

### LoRA 配置

创建 `my_lora_config.yaml`:

```yaml
framework: megatron_bridge

# 从 HuggingFace 加载
convert_from_hf: true
hf_model_name_or_path: "meta-llama/Llama-3-8B"

# LoRA 参数
lora:
  use_lora: true
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj

# 训练参数（LoRA 可以用更高的学习率）
training:
  micro_batch_size: 2
  global_batch_size: 64
  train_iters: 3000
  lr: 2.0e-4
  weight_decay: 0.0

# 混合精度
precision:
  bf16: true

# 最小并行（LoRA 内存需求小）
parallelism:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1

# 数据
data:
  data_path: /path/to/instruction/data
  dataset_format: "alpaca"
  seq_length: 2048

# 仅保存 LoRA 权重（体积小）
checkpointing:
  save: /path/to/lora/checkpoints
  save_interval: 500
  save_lora_only: true
```

## 常见场景

### 1. 标准监督微调 (SFT)

```bash
torchrun --nproc-per-node=8 \
    -m primus.cli.train \
    --framework megatron_bridge \
    --config examples/configs/megatron_bridge/llama_sft_posttrain.yaml \
    --load /path/to/pretrained/checkpoint \
    --data_path /path/to/instruction/data \
    --lr 5e-6 \
    --train_iters 5000
```

### 2. 指令调优

```bash
torchrun --nproc-per-node=8 \
    -m primus.cli.train \
    --framework megatron_bridge \
    --data_path /path/to/instruction/data \
    --dataset_format alpaca \
    --prompt_template alpaca \
    --load /path/to/pretrained/checkpoint
```

### 3. 对话模型微调

```bash
torchrun --nproc-per-node=8 \
    -m primus.cli.train \
    --framework megatron_bridge \
    --data_path /path/to/chat/data \
    --dataset_format sharegpt \
    --chat_format chatml \
    --prompt_template chatml \
    --load /path/to/pretrained/checkpoint
```

### 4. LoRA 微调（低资源）

```bash
# 可能在单个 GPU 或少量 GPU 上运行
torchrun --nproc-per-node=2 \
    -m primus.cli.train \
    --framework megatron_bridge \
    --config examples/configs/megatron_bridge/llama_lora_posttrain.yaml \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 16
```

### 5. 导出到 HuggingFace

```bash
# 在训练配置中设置 convert_to_hf: true
# 或者单独运行转换
python -m primus.cli.convert \
    --framework megatron_bridge \
    --load /path/to/finetuned/checkpoint \
    --convert_to_hf \
    --hf_save_path /path/to/output/hf_model
```

## 数据集格式

### Alpaca 格式

```json
{
  "instruction": "给出三个保持健康的秘诀。",
  "input": "",
  "output": "1. 均衡饮食...\n2. 定期锻炼...\n3. 充足睡眠..."
}
```

### ShareGPT 格式（多轮对话）

```json
{
  "conversations": [
    {"from": "human", "value": "你好！"},
    {"from": "gpt", "value": "你好！我能帮你什么？"},
    {"from": "human", "value": "介绍一下 AI"}
  ]
}
```

## 超参数调优

### 全量微调 vs LoRA

| 参数              | 全量微调      | LoRA 微调      |
|-------------------|---------------|----------------|
| 学习率            | 5e-6 ~ 1e-5   | 1e-4 ~ 3e-4    |
| Batch Size        | 64-128        | 32-64          |
| 训练步数          | 3K-10K        | 2K-5K          |
| GPU 内存          | 高            | 低             |
| 训练速度          | 慢            | 快             |
| Checkpoint 大小   | 大            | 小（仅 adapter）|

### LoRA 参数选择

```yaml
lora:
  lora_rank: 8-16      # 越大容量越强，但内存越多
  lora_alpha: 16-32    # 通常是 rank 的 2 倍
  lora_dropout: 0.05   # 防止过拟合
```

**建议:**
- 小数据集: rank=8, alpha=16
- 中等数据集: rank=16, alpha=32
- 大数据集或复杂任务: rank=32, alpha=64

## 并行配置

### SFT 并行策略

由于微调通常不需要像预训练那样激进的并行:

```yaml
# 小模型（7B-13B）
parallelism:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  # 仅使用数据并行

# 中等模型（30B-40B）
parallelism:
  tensor_model_parallel_size: 2
  pipeline_model_parallel_size: 1

# 大模型（70B+）
parallelism:
  tensor_model_parallel_size: 4
  pipeline_model_parallel_size: 2
```

### LoRA 并行策略

LoRA 微调通常只需要数据并行:

```yaml
parallelism:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
```

## 性能优化

### 内存优化

```yaml
# 使用梯度检查点
activation_checkpointing:
  recompute_activations: true

# 混合精度
precision:
  bf16: true

# LoRA（最有效的内存优化）
lora:
  use_lora: true
  lora_rank: 8
```

### 速度优化

```yaml
training:
  # 增大批次大小（如果内存允许）
  micro_batch_size: 2
  global_batch_size: 128

# 序列长度（较短的序列训练更快）
data:
  seq_length: 1024  # 如果你的数据较短
```

## 故障排除

### 问题 1: OOM (Out of Memory)

**解决方案:**
- 使用 LoRA: `--use_lora --lora_rank 8`
- 减小批次: `--micro_batch_size 1`
- 启用梯度检查点: `recompute_activations: true`
- 减小序列长度: `seq_length: 1024`
- 增加张量并行: `tensor_model_parallel_size: 2`

### 问题 2: 训练不稳定

**解决方案:**
- 降低学习率: `--lr 1e-6`
- 增加 warmup: `--lr_warmup_iters 500`
- 使用梯度裁剪: `--clip_grad 1.0`
- 检查数据质量

### 问题 3: 过拟合

**解决方案:**
- 增加 LoRA dropout: `lora_dropout: 0.1`
- 减少训练步数
- 增加 weight decay: `weight_decay: 0.1`
- 使用更多训练数据

### 问题 4: 训练太慢

**解决方案:**
- 使用 LoRA（显著加速）
- 增大批次大小
- 减小序列长度
- 使用更多 GPU

## 使用示例脚本

```bash
# 交互式模式（选择示例）
bash examples/run_megatron_bridge.sh

# 命令行模式
bash examples/run_megatron_bridge.sh sft      # SFT from checkpoint
bash examples/run_megatron_bridge.sh hf       # SFT from HuggingFace
bash examples/run_megatron_bridge.sh lora     # LoRA fine-tuning
bash examples/run_megatron_bridge.sh chat     # Chat fine-tuning
bash examples/run_megatron_bridge.sh export   # Export to HuggingFace
```

## 最佳实践

### 1. 数据准备
- 确保数据格式正确（Alpaca, ShareGPT 等）
- 清理数据质量
- 合理的数据量（1K-100K 样本）

### 2. 模型选择
- 从 HuggingFace 加载流行模型
- 或使用已有的 Megatron checkpoint

### 3. 超参数
- 从推荐值开始
- 根据验证集表现调整
- LoRA: 优先调整 rank 和 alpha

### 4. 评估
- 定期在验证集上评估
- 监控训练损失曲线
- 测试生成质量

### 5. 部署
- 导出为 HuggingFace 格式
- 测试推理性能
- 考虑量化（INT8/INT4）

## 更多资源

- [Megatron-Bridge GitHub](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- [Megatron-Bridge 文档](https://docs.nvidia.com/nemo/megatron-bridge/latest/)
- [Backend README](primus/backends/megatron_bridge/README.md)
- [配置示例](examples/configs/megatron_bridge/)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)

## 获取帮助

如果遇到问题:
1. 检查本文档的故障排除部分
2. 查看配置示例
3. 查看 Megatron-Bridge 官方文档
4. 在 Primus 仓库提交 issue

## 重要提示

⚠️ **Megatron-Bridge 用途**:
- ✅ 监督微调 (SFT)
- ✅ 指令调优
- ✅ LoRA 微调
- ✅ 对话模型微调
- ❌ 预训练（使用 megatron 或 torchtitan backend）

Megatron-Bridge 的优势在于其优秀的 HuggingFace 转换能力和后训练工作流优化。
