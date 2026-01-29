# Primus 数据集处理概述

## 问题：它的数据集是怎么处理的？

本文档回答了 Primus 训练框架中数据集的处理方式。

## 快速回答

Primus 的数据集处理分为三个主要阶段：

### 1. 数据准备阶段
- 下载或获取原始数据（JSON/JSONL 格式）
- 将数据分割为训练集、验证集、测试集
- 清洗和格式化数据

### 2. 预处理阶段  
- 使用分词器将文本转换为 token
- 创建二进制索引数据集（Megatron）或保持原始格式（MaxText、TorchTitan）
- 生成 `.bin` 和 `.idx` 文件（Megatron 特有）

### 3. 训练加载阶段
- 训练时按需加载数据
- 支持分布式数据采样
- 内存映射文件，无需全部加载到内存

## 主要工具和脚本

| 工具 | 用途 |
|------|------|
| `prepare.py` | 自动化数据准备（下载、分割、预处理）|
| `preprocess_data.py` | 手动预处理数据（分词、二进制转换）|
| `GPTDataset` | Megatron 训练时的数据加载器 |
| HuggingFace datasets | MaxText 的数据源 |

## 不同后端的数据处理方式

### Megatron 后端
```bash
# 1. 预处理数据
python preprocess_data.py \
    --input data.jsonl \
    --output-prefix dataset \
    --tokenizer-type GPT2BPETokenizer

# 2. 配置训练
# 在 YAML 中指定预处理后的数据路径
train_data_path: /path/to/dataset_text_document

# 3. 训练
./runner/primus-cli train pretrain --config config.yaml
```

**特点：**
- 需要离线预处理
- 生成二进制索引文件（.bin + .idx）
- 支持数据集混合（多个数据集按权重组合）
- 内存映射，高效加载

### MaxText 后端
```yaml
# 直接在配置中指定数据集
dataset_name: "c4"  # HuggingFace 数据集
# 或
train_data_path: /path/to/data.jsonl  # 本地文件

# 在线分词，无需预处理
tokenizer_path: /path/to/tokenizer
```

**特点：**
- 在线分词，无需预处理
- 支持 HuggingFace 数据集
- 使用 Grain 进行高效数据加载
- 支持数据打包（packing）

### TorchTitan 后端
```yaml
# PyTorch 原生数据加载
dataset: "c4"
data_path: /path/to/data
```

**特点：**
- 使用 PyTorch DataLoader
- 配置驱动的数据加载
- 灵活的数据管道

## 数据格式

### 原始数据格式（JSON/JSONL）
```json
{"text": "这是训练文本。"}
{"text": "另一个训练样本。"}
```

### Megatron 二进制格式
```
dataset_text_document.bin  # 实际数据（大文件）
dataset_text_document.idx  # 索引文件（小文件）
```

### 监督微调（SFT）数据格式
```json
{
  "instruction": "将以下内容翻译成英文：",
  "input": "你好世界",
  "response": "Hello world"
}
```

## 配置示例

### 基本配置
```yaml
modules:
  pre_trainer:
    framework: megatron
    overrides:
      # 数据路径
      train_data_path: /path/to/train_text_document
      valid_data_path: /path/to/valid_text_document
      
      # 序列长度
      seq_length: 2048
      
      # 分词器
      tokenizer_type: GPT2BPETokenizer
      
      # 数据分割比例
      split: "98,2,0"  # 训练/验证/测试
```

### 混合数据集（Megatron）
```yaml
# 按权重混合多个数据集
train_data_path: "0.5 /data/wiki 0.3 /data/books 0.2 /data/code"
```

### 模拟数据（测试用）
```yaml
# 使用合成数据测试配置
mock_data: true
```

## 常见问题

### 1. 如何准备数据？

**方法 1：使用 prepare.py（推荐）**
```bash
cd examples/megatron
python prepare.py --dataset bookcorpus --tokenizer-type GPT2BPETokenizer
```

**方法 2：手动预处理**
```bash
python preprocess_data.py \
    --input my_data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type GPT2BPETokenizer \
    --workers 8
```

### 2. 数据文件在哪里？

预处理后的文件通常在：
```
data/
  └── bookcorpus/
      └── GPT2BPETokenizer/
          ├── bookcorpus_train_text_document.bin
          ├── bookcorpus_train_text_document.idx
          ├── bookcorpus_valid_text_document.bin
          └── bookcorpus_valid_text_document.idx
```

### 3. 如何验证数据处理成功？

```bash
# 检查文件是否存在
ls -lh /path/to/dataset_text_document.*

# .bin 文件应该远大于 .idx 文件
# 例如：
# dataset_text_document.bin  (2.3 GB)
# dataset_text_document.idx  (4.5 MB)
```

### 4. 预处理需要多长时间？

取决于：
- 数据集大小
- CPU 核心数（workers 参数）
- 是否使用分区（partitions 参数）

**示例时间：**
- 小数据集（<1GB）：几分钟
- 中等数据集（1-10GB）：10-30 分钟
- 大数据集（>10GB）：可能需要数小时

**加速技巧：**
```bash
# 使用多个工作进程和分区
python preprocess_data.py \
    --workers 16 \
    --partitions 8 \
    --input large_dataset.jsonl
```

### 5. 能否跳过预处理？

- **Megatron**: 不能，必须预处理
- **MaxText**: 可以，支持在线分词
- **TorchTitan**: 取决于数据集类型

### 6. 如何处理大数据集？

```bash
# 使用分区处理大文件
python preprocess_data.py \
    --input very_large_file.jsonl \
    --output-prefix dataset \
    --workers 8 \
    --partitions 16  # 分成 16 个部分处理
```

### 7. 训练时找不到数据文件？

**检查清单：**
1. 文件路径是否正确？
2. `.bin` 和 `.idx` 文件是否都存在？
3. 配置中的路径是否包含完整路径？
4. 路径中不要包含 `.bin` 或 `.idx` 后缀

```yaml
# 正确的配置
train_data_path: /path/to/dataset_text_document  # ✅

# 错误的配置
train_data_path: /path/to/dataset_text_document.bin  # ❌
```

## 完整工作流程示例

```bash
# 步骤 1：准备数据
cd examples/megatron
python prepare.py \
    --dataset bookcorpus \
    --tokenizer-type GPT2BPETokenizer \
    --output-dir ./data/bookcorpus/GPT2BPETokenizer

# 步骤 2：验证数据
ls -lh data/bookcorpus/GPT2BPETokenizer/
# 应该看到 .bin 和 .idx 文件

# 步骤 3：创建训练配置
cat > my_config.yaml << EOF
work_group: amd
user_name: user
exp_name: my_experiment
workspace: ./output

modules:
  pre_trainer:
    framework: megatron
    config: pre_trainer.yaml
    model: llama2_7B.yaml
    overrides:
      train_data_path: data/bookcorpus/GPT2BPETokenizer/bookcorpus_train_text_document
      valid_data_path: data/bookcorpus/GPT2BPETokenizer/bookcorpus_valid_text_document
      seq_length: 2048
      train_iters: 1000
EOF

# 步骤 4：运行训练
cd ../..
./runner/primus-cli train pretrain --config examples/megatron/my_config.yaml
```

## 性能优化建议

### 1. 预处理性能
```bash
# 使用所有 CPU 核心
python preprocess_data.py --workers $(nproc)

# 大文件使用分区
python preprocess_data.py --partitions 8
```

### 2. 训练性能
```yaml
# 使用内存映射（自动）
# Megatron 的 .bin 文件会自动内存映射，无需全部加载到内存

# 关闭不必要的数据增强
create_attention_mask_in_dataloader: false
```

### 3. 存储优化
```bash
# 预处理后删除原始 JSON 文件（如果不需要）
# 但建议保留备份

# 使用压缩格式存储原始数据
gzip data.jsonl  # 创建 data.jsonl.gz
```

## 进阶主题

### 数据集混合
```yaml
# 混合多个数据集，指定权重
train_data_path: "0.6 /data/wiki 0.2 /data/books 0.2 /data/code"

# 作用：
# - 60% 的批次来自 wiki
# - 20% 来自 books
# - 20% 来自 code
```

### 自定义数据集
```python
# 创建自定义数据加载器
from megatron.core.datasets import GPTDataset

class CustomDataset(GPTDataset):
    def __init__(self, ...):
        # 自定义初始化
        pass
    
    def __getitem__(self, idx):
        # 自定义数据加载逻辑
        pass
```

### SFT 数据处理
```bash
# SFT 需要特殊的数据格式和 loss_mask
# 参见：docs/sft_native.md
```

## 更多信息

详细文档请参阅：
- **[完整数据集处理指南](./dataset_processing.md)** - 英文+中文双语详细文档
- **[SFT 原生指南](./sft_native.md)** - 监督微调数据处理
- **[快速开始指南](./quickstart.md)** - Primus 入门

## 总结

Primus 的数据集处理流程：

1. **准备阶段**: 获取或下载原始 JSON/JSONL 数据
2. **预处理阶段**: 分词并转换为二进制格式（Megatron）或保持原始格式（其他）
3. **配置阶段**: 在 YAML 中指定数据路径
4. **训练阶段**: 自动加载和分布式采样

不同后端有不同的处理方式，但核心流程类似。选择合适的后端和工具可以大大简化数据处理工作。
