# Dataset Processing in Primus / Primus 数据集处理指南

*[中文版本在下方](#中文版本) / Chinese version below*

---

## English Version

This guide explains how datasets are processed in the Primus training framework, covering data preparation, preprocessing, and loading for different backends.

## Overview

Primus supports multiple training backends (Megatron-LM, TorchTitan, MaxText), each with specific data processing requirements. The general workflow is:

```
Raw Data (JSON/JSONL)
    ↓
Data Preparation (download, clean, split)
    ↓
Preprocessing (tokenization, binary conversion)
    ↓
Training Data Loading (distributed sampling)
    ↓
Training
```

## Data Formats

### Supported Input Formats

| Format | Backend Support | Description |
|--------|----------------|-------------|
| **JSON/JSONL** | All backends | Raw text in JSON Lines format |
| **Megatron Binary** (`.bin` + `.idx`) | Megatron | Memory-mapped indexed dataset |
| **HuggingFace Datasets** | MaxText | Direct loading from HF Dataset Hub |
| **Mock Data** | All backends | Synthetic data for testing |

### JSON/JSONL Format

Raw data should be in JSON Lines format with text fields:

```json
{"text": "This is a training example."}
{"text": "Another document for training."}
{"content": "Can use different key names."}
```

**Key fields:**
- Configurable via `--json-keys` argument (default: "text")
- Multiple keys supported for multi-field extraction
- Each line is one document

## Data Preparation Pipeline

### 1. Megatron Backend

#### Step 1: Prepare Raw Data

Use `examples/megatron/prepare.py` for automated preparation:

```bash
cd examples/megatron

# Prepare BookCorpus dataset (example)
python prepare.py \
    --dataset bookcorpus \
    --tokenizer-type GPT2BPETokenizer \
    --output-dir ./data/bookcorpus/GPT2BPETokenizer
```

This script:
1. Downloads dataset from HuggingFace
2. Splits into train/validation sets
3. Saves as JSON files
4. Tokenizes and creates binary indexed datasets

#### Step 2: Preprocess Data (Manual)

For custom datasets, use `preprocess_data.py`:

```bash
python preprocess_data.py \
    --input /path/to/data.jsonl \
    --output-prefix /path/to/output/dataset \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file gpt2-vocab.json \
    --merge-file gpt2-merges.txt \
    --append-eod \
    --workers 8
```

**Important arguments:**
- `--input`: Path to JSON/JSONL file
- `--output-prefix`: Output path prefix (creates `.bin` and `.idx` files)
- `--tokenizer-type`: Tokenizer to use (GPT2BPETokenizer, SentencePieceTokenizer, etc.)
- `--append-eod`: Add end-of-document token
- `--workers`: Number of parallel workers for tokenization
- `--partitions`: Split processing into N partitions

**Output files:**
- `dataset_text_document.bin`: Binary data file (memory-mapped)
- `dataset_text_document.idx`: Index file (document boundaries)

#### Step 3: Configure Training

In your experiment YAML:

```yaml
modules:
  pre_trainer:
    framework: megatron
    overrides:
      # Single dataset
      train_data_path: /path/to/train_text_document
      valid_data_path: /path/to/valid_text_document
      
      # Or blended datasets (space-separated, with optional weights)
      # Format: "weight1 path1 weight2 path2 ..."
      train_data_path: "0.7 /path/to/dataset1 0.3 /path/to/dataset2"
      
      # Dataset configuration
      split: "98,2,0"  # train/valid/test split percentages
      seq_length: 2048
      tokenizer_type: GPT2BPETokenizer
```

### 2. MaxText Backend (JAX)

MaxText uses HuggingFace datasets with on-the-fly tokenization:

```yaml
modules:
  pre_trainer:
    framework: maxtext
    overrides:
      # Dataset from HuggingFace
      dataset_name: "c4"
      dataset_path: "en"
      
      # Or local dataset
      dataset_type: "json"
      train_data_path: /path/to/train.jsonl
      
      # Tokenization happens online
      tokenizer_path: /path/to/tokenizer
      max_target_length: 2048
      
      # Data processing options
      use_data_packing: true
      shuffle_dataset: true
      grain_worker_count: 8
```

**MaxText features:**
- Online tokenization (no preprocessing required)
- Grain-based efficient data loading
- Data packing for variable-length sequences
- Support for multimodal/vision tasks

### 3. TorchTitan Backend

TorchTitan uses PyTorch native data loading:

```yaml
modules:
  pre_trainer:
    framework: torchtitan
    overrides:
      # Dataset configuration
      dataset: "c4"  # or custom dataset name
      data_path: /path/to/data
      
      # Tokenization
      tokenizer_path: /path/to/tokenizer
      seq_length: 2048
      
      # DataLoader options
      batch_size: 8
      num_workers: 4
```

## Dataset Loading During Training

### Megatron Dataset Loading

**Code location:** `primus/modules/trainer/megatron/trainer.py`

```python
# Dataset builder
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder
)
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig

# Configuration
config = GPTDatasetConfig(
    random_seed=args.seed,
    sequence_length=args.seq_length,
    blend=get_blend_from_list(args.data_path),
    split=args.split,
    path_to_cache=args.data_cache_path,
    # ... other options
)

# Build datasets
train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
    GPTDataset,
    train_val_test_num_samples,
    is_dataset_built_on_rank,
    config,
).build()
```

**Features:**
- Distributed data loading (only on specific ranks)
- Memory-mapped files (no RAM pressure)
- Automatic shuffling and epoch handling
- Support for cyclic vs. single-pass iteration

### Mock Data for Testing

All backends support mock data to test training without actual datasets:

```yaml
overrides:
  mock_data: true  # Generate synthetic data
  # No need for train_data_path when using mock data
```

## Advanced Topics

### Data Blending (Megatron)

Mix multiple datasets with different weights:

```yaml
# Format: "weight1 path1 weight2 path2 ..."
train_data_path: "0.5 /data/wikipedia 0.3 /data/books 0.2 /data/code"

# Or equal weights (automatically calculated)
train_data_path: "/data/wikipedia /data/books /data/code"
```

**Blending behavior:**
- Datasets sampled proportionally to weights
- Useful for domain mixing
- Each dataset can have different sizes

### Sentence Splitting

Enable sentence-level tokenization:

```bash
python preprocess_data.py \
    --split-sentences \
    --lang en \
    --keep-newlines
```

**When to use:**
- Preserves sentence boundaries
- Useful for tasks requiring sentence-level understanding
- NLTK required (`pip install nltk`)

### Multiprocessing

Accelerate preprocessing with parallel workers:

```bash
# 8 workers, 4 partitions
python preprocess_data.py \
    --workers 8 \
    --partitions 4 \
    --input large_dataset.jsonl \
    --output-prefix processed_data
```

**Recommendations:**
- Workers: Number of CPU cores
- Partitions: Split large files for parallel processing
- Monitor memory usage with large datasets

### SFT (Supervised Fine-Tuning) Data

For SFT tasks, datasets need additional fields:

```json
{
  "instruction": "Translate this to French:",
  "input": "Hello world",
  "response": "Bonjour le monde"
}
```

**SFT preprocessing:**
- Concatenate instruction + input + response
- Generate `loss_mask` (0 for instruction, 1 for response)
- See `docs/sft_native.md` for details

## Configuration Reference

### Common Parameters

```yaml
# Data paths
train_data_path: /path/to/train
valid_data_path: /path/to/valid
test_data_path: /path/to/test

# Tokenizer
tokenizer_type: GPT2BPETokenizer
vocab_file: vocab.json
merge_file: merges.txt

# Sequence length
seq_length: 2048
max_position_embeddings: 2048

# Dataset splits
split: "98,2,0"  # train/valid/test percentages

# Data processing
eod_mask_loss: true  # Mask loss on EOD tokens
create_attention_mask_in_dataloader: false

# Mock data (testing)
mock_data: false
```

## Troubleshooting

### Issue: "FileNotFoundError: .bin or .idx not found"

**Solution:** Ensure preprocessing completed successfully:
```bash
# Check output files exist
ls -lh /path/to/dataset_text_document.*

# Rerun preprocessing if needed
python preprocess_data.py --input data.jsonl --output-prefix dataset
```

### Issue: "RuntimeError: mmap length is greater than file size"

**Solution:** Corrupted or incomplete dataset files. Regenerate:
```bash
# Remove corrupted files
rm /path/to/dataset_text_document.*

# Rerun preprocessing
python preprocess_data.py ...
```

### Issue: "Out of memory during preprocessing"

**Solution:** Reduce workers or use partitions:
```bash
# Reduce memory usage
python preprocess_data.py \
    --workers 2 \
    --partitions 8 \
    --input large_file.jsonl
```

### Issue: "NLTK punkt tokenizer not found"

**Solution:** Download NLTK data:
```python
import nltk
nltk.download('punkt')
```

Or set NLTK_DATA path:
```bash
export NLTK_DATA=/path/to/nltk_data
```

## Best Practices

1. **Preprocessing First**: Always preprocess data before training for Megatron backend
2. **Check File Sizes**: `.bin` files should be much larger than `.idx` files
3. **Use Mock Data**: Test configurations with `mock_data: true` first
4. **Memory Efficiency**: Use memory-mapped files, don't load entire dataset to RAM
5. **Distributed Loading**: Only specific ranks load/build datasets (automatic)
6. **Data Validation**: Check token counts and sequence lengths after preprocessing
7. **Backup Raw Data**: Keep original JSON files, preprocessing is deterministic

## Example Workflow

Complete example for Megatron backend:

```bash
# 1. Prepare dataset
cd examples/megatron
python prepare.py --dataset bookcorpus --tokenizer-type GPT2BPETokenizer

# 2. Verify preprocessing
ls -lh data/bookcorpus/GPT2BPETokenizer/
# Should see: bookcorpus_train_text_document.bin/idx
#             bookcorpus_valid_text_document.bin/idx

# 3. Create experiment config
cat > my_pretrain.yaml << EOF
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
      global_batch_size: 256
EOF

# 4. Run training
cd ../..
./runner/primus-cli train pretrain --config examples/megatron/my_pretrain.yaml
```

## References

- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM)
- [MaxText Data Pipeline](https://github.com/google/maxtext)
- [TorchTitan Documentation](https://github.com/pytorch/torchtitan)
- [Primus SFT Native Guide](./sft_native.md)

---

## 中文版本

# Primus 数据集处理完整指南

本指南详细介绍 Primus 训练框架中的数据集处理方法，包括数据准备、预处理和加载。

## 概述

Primus 支持多个训练后端（Megatron-LM、TorchTitan、MaxText），每个后端都有特定的数据处理需求。总体工作流程为：

```
原始数据 (JSON/JSONL)
    ↓
数据准备（下载、清洗、分割）
    ↓
预处理（分词、二进制转换）
    ↓
训练数据加载（分布式采样）
    ↓
训练
```

## 支持的数据格式

| 格式 | 后端支持 | 描述 |
|------|----------|------|
| **JSON/JSONL** | 所有后端 | JSON Lines 格式的原始文本 |
| **Megatron 二进制** (`.bin` + `.idx`) | Megatron | 内存映射的索引数据集 |
| **HuggingFace 数据集** | MaxText | 从 HF Dataset Hub 直接加载 |
| **模拟数据** | 所有后端 | 用于测试的合成数据 |

### JSON/JSONL 格式示例

```json
{"text": "这是一个训练样本。"}
{"text": "另一个用于训练的文档。"}
{"content": "可以使用不同的键名。"}
```

## 数据处理流程

### 1. Megatron 后端数据处理

#### 步骤 1：准备原始数据

使用 `examples/megatron/prepare.py` 自动准备数据：

```bash
cd examples/megatron

# 准备 BookCorpus 数据集（示例）
python prepare.py \
    --dataset bookcorpus \
    --tokenizer-type GPT2BPETokenizer \
    --output-dir ./data/bookcorpus/GPT2BPETokenizer
```

此脚本会：
1. 从 HuggingFace 下载数据集
2. 分割为训练/验证集
3. 保存为 JSON 文件
4. 分词并创建二进制索引数据集

#### 步骤 2：预处理数据

对于自定义数据集，使用 `preprocess_data.py`：

```bash
python preprocess_data.py \
    --input /path/to/data.jsonl \
    --output-prefix /path/to/output/dataset \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file gpt2-vocab.json \
    --merge-file gpt2-merges.txt \
    --append-eod \
    --workers 8
```

**重要参数：**
- `--input`: JSON/JSONL 文件路径
- `--output-prefix`: 输出路径前缀（生成 `.bin` 和 `.idx` 文件）
- `--tokenizer-type`: 使用的分词器
- `--append-eod`: 添加文档结束标记
- `--workers`: 并行分词的工作进程数

**输出文件：**
- `dataset_text_document.bin`: 二进制数据文件（内存映射）
- `dataset_text_document.idx`: 索引文件（文档边界）

#### 步骤 3：配置训练

在实验配置 YAML 中：

```yaml
modules:
  pre_trainer:
    framework: megatron
    overrides:
      # 单个数据集
      train_data_path: /path/to/train_text_document
      valid_data_path: /path/to/valid_text_document
      
      # 或混合数据集（空格分隔，可选权重）
      train_data_path: "0.7 /path/to/dataset1 0.3 /path/to/dataset2"
      
      # 数据集配置
      split: "98,2,0"  # 训练/验证/测试分割百分比
      seq_length: 2048
      tokenizer_type: GPT2BPETokenizer
```

### 2. MaxText 后端（JAX）

MaxText 使用 HuggingFace 数据集进行在线分词：

```yaml
modules:
  pre_trainer:
    framework: maxtext
    overrides:
      # 从 HuggingFace 加载数据集
      dataset_name: "c4"
      dataset_path: "en"
      
      # 或本地数据集
      dataset_type: "json"
      train_data_path: /path/to/train.jsonl
      
      # 在线分词
      tokenizer_path: /path/to/tokenizer
      max_target_length: 2048
      
      # 数据处理选项
      use_data_packing: true
      shuffle_dataset: true
```

### 3. TorchTitan 后端

```yaml
modules:
  pre_trainer:
    framework: torchtitan
    overrides:
      dataset: "c4"
      data_path: /path/to/data
      tokenizer_path: /path/to/tokenizer
      seq_length: 2048
```

## 常见问题解决

### 问题："找不到 .bin 或 .idx 文件"

**解决方案：** 确保预处理成功完成：
```bash
# 检查输出文件是否存在
ls -lh /path/to/dataset_text_document.*

# 如需重新运行预处理
python preprocess_data.py --input data.jsonl --output-prefix dataset
```

### 问题："预处理时内存不足"

**解决方案：** 减少工作进程或使用分区：
```bash
python preprocess_data.py \
    --workers 2 \
    --partitions 8 \
    --input large_file.jsonl
```

### 问题："NLTK punkt 分词器未找到"

**解决方案：** 下载 NLTK 数据：
```python
import nltk
nltk.download('punkt')
```

## 最佳实践

1. **先预处理**: Megatron 后端训练前务必先预处理数据
2. **检查文件大小**: `.bin` 文件应该比 `.idx` 文件大得多
3. **使用模拟数据**: 先用 `mock_data: true` 测试配置
4. **内存效率**: 使用内存映射文件，不要将整个数据集加载到内存
5. **数据验证**: 预处理后检查 token 数量和序列长度
6. **备份原始数据**: 保留原始 JSON 文件，预处理是确定性的

## 完整示例工作流程

```bash
# 1. 准备数据集
cd examples/megatron
python prepare.py --dataset bookcorpus --tokenizer-type GPT2BPETokenizer

# 2. 验证预处理结果
ls -lh data/bookcorpus/GPT2BPETokenizer/

# 3. 创建实验配置
cat > my_pretrain.yaml << EOF
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

# 4. 运行训练
cd ../..
./runner/primus-cli train pretrain --config examples/megatron/my_pretrain.yaml
```

## 参考资料

- [Primus SFT 原生指南](./sft_native.md)
- [Megatron-LM 文档](https://github.com/NVIDIA/Megatron-LM)
- [MaxText 数据管道](https://github.com/google/maxtext)

---

*本文档涵盖了 Primus 中所有主要的数据集处理方法。如有问题，请参考相关后端的官方文档。*
