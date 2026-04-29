# Megatron-Bridge Loss Function 移植说明

## 改进内容

将 Megatron-Bridge 的 `masked_next_token_loss` 实现移植到 SFT Trainer 中。

## 关键变化

### 1. **模型调用方式**

**之前**：
```python
# 不传 labels，手动计算 cross_entropy
output_tensor = model(tokens, position_ids, attention_mask=None)
# 然后在 loss_func 中手动计算 logits 的 cross_entropy
```

**现在**（Megatron-Bridge 方式）：
```python
# 传入 labels，让模型内部计算 per-token losses
output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
# model 返回 per-token losses，我们只需要 mask 和 sum
```

### 2. **Loss 函数签名**

**之前**：
```python
def loss_func(loss_mask, output_tensor):
    # ... 计算 ...
    return loss  # 只返回 loss
```

**现在**（Megatron 标准）：
```python
def loss_func(loss_mask, output_tensor):
    # ... 计算 ...
    return (loss, num_tokens, {"lm loss": reporting_loss})  # 返回三元组
```

这是 Megatron 的标准 loss 函数签名，返回：
- `loss`: 用于反向传播的 loss tensor
- `num_tokens`: 有效 token 数量（用于正确平均）
- `metrics_dict`: 用于日志记录的指标字典

### 3. **Loss 计算方式**

**之前**（手动计算）：
```python
# 获取 logits
logits = output_tensor

# Shift 和 flatten
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()
shift_mask = loss_mask[:, 1:].contiguous()

# 计算 cross_entropy
losses = F.cross_entropy(shift_logits, shift_labels, reduction='none')

# 应用 mask
masked_losses = losses * shift_mask
loss = masked_losses.sum() / shift_mask.sum()
```

**现在**（Megatron-Bridge 方式）：
```python
# Model 已经返回 per-token losses
losses = output_tensor.view(-1).float()
loss_mask = loss_mask.view(-1).float()

# 直接应用 mask 和 sum
loss = torch.sum(losses * loss_mask)

# 计算 token 数量
num_tokens = loss_mask.sum()
```

### 4. **返回格式**

**Megatron 训练循环期望的格式**：

```python
(loss, num_tokens, metrics_dict)
```

其中：
- `loss`: 用于 `loss.backward()` 的 tensor
- `num_tokens`: 用于正确计算平均 loss（在 DP 间平均时需要）
- `metrics_dict`: 包含 `{"lm loss": [loss_value, num_tokens]}` 的字典

Megatron 训练循环会：
1. 用 `loss` 做反向传播
2. 用 `num_tokens` 来正确平均（考虑不同 micro-batch 的 token 数量不同）
3. 用 `metrics_dict` 记录日志

## 优势

### 1. **简化代码**
不需要手动：
- Shift logits 和 labels
- Flatten tensors
- 计算 cross_entropy
- 处理维度问题

### 2. **更高效**
Model 内部可以使用更优化的 loss 计算实现。

### 3. **标准化**
遵循 Megatron 的标准 loss 函数接口，与其他训练脚本一致。

### 4. **正确的平均**
通过返回 `num_tokens`，Megatron 可以正确地在不同 micro-batch 和 DP rank 之间平均 loss。

## 对比

| 方面 | 之前的实现 | Megatron-Bridge 实现 |
|------|-----------|---------------------|
| **Model 调用** | 不传 labels | 传 labels |
| **Loss 计算** | 手动 cross_entropy | Model 内部计算 |
| **代码行数** | ~60 行 | ~20 行 |
| **维度处理** | 需要手动 shift/flatten | 只需 flatten |
| **返回值** | 单个 loss | (loss, num_tokens, metrics) |
| **DP 平均** | 手动 all_reduce | Megatron 自动处理 |
| **可维护性** | 中等 | 高（标准接口） |

## 代码示例

### 完整的 forward_step

```python
def forward_step(data_iterator, model):
    import torch
    from megatron.training import get_args
    
    args = get_args()
    
    # Get batch
    try:
        batch = next(data_iterator)
    except StopIteration:
        return None, lambda output: (torch.tensor(0.0), torch.tensor(0), {})
    
    # Extract tensors
    tokens = batch['input_ids'].long().cuda()
    labels = batch['labels'].long().cuda()
    loss_mask = batch['loss_mask'].float().cuda()
    
    # Ensure 2D [batch, seq]
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    if loss_mask.dim() == 1:
        loss_mask = loss_mask.unsqueeze(0)
    
    # Generate position_ids
    batch_size, seq_len = tokens.size()
    position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    
    # ✅ Forward with labels - model returns per-token losses
    output_tensor = model(tokens, position_ids, attention_mask=None, labels=labels)
    
    # ✅ Simple loss function (Megatron-Bridge style)
    def loss_func(loss_mask, output_tensor):
        # Model returns per-token losses
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        
        # Apply mask and sum
        loss = torch.sum(losses * loss_mask)
        
        # Count tokens
        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        
        # Reporting metrics
        reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])
        
        # Return standard Megatron format
        return (loss, num_tokens, {"lm loss": reporting_loss})
    
    return output_tensor, lambda output: loss_func(loss_mask, output)
```

## 注意事项

### 1. **Model 必须支持返回 losses**

确保你的 GPTModel 配置了返回 per-token losses。大多数 Megatron models 当传入 `labels` 参数时会自动返回 losses。

### 2. **Loss Mask 的含义**

- `loss_mask = 1`: 在该位置计算 loss（response tokens）
- `loss_mask = 0`: 跳过该位置（instruction tokens）

### 3. **返回值处理**

Megatron 训练循环会：
```python
loss, num_tokens, metrics = loss_func(output_tensor)

# 用于反向传播
loss.backward()

# 用于日志（会自动在 DP ranks 间平均）
# metrics = {"lm loss": [loss_value, num_tokens]}
```

### 4. **Shifted Loss**

Model 内部会自动处理 next-token prediction 的 shifting：
- `logits[..., :-1, :]` 预测 `labels[..., 1:]`
- 我们不需要手动 shift

## 测试验证

修改后应该能看到：
1. ✅ 训练正常启动
2. ✅ Loss 逐渐下降
3. ✅ 日志显示正确的 `lm loss` 值
4. ✅ 没有维度错误
5. ✅ DP averaging 正确工作

## 总结

这次移植将我们的实现与 Megatron-Bridge 对齐，采用了：
- ✅ 标准的 Megatron loss 函数接口
- ✅ 简化的代码（少了 60% 的代码）
- ✅ 更好的可维护性
- ✅ 正确的 DP averaging 机制

这是一个更加**生产级**的实现。
